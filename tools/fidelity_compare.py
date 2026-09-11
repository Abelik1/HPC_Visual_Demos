"""Compare galaxy_collision_3d runs recorded by ``tools/fidelity_run.py``.

    python tools/fidelity_compare.py REFERENCE_RUN [OTHER_RUN ...]

Prints, for every run, energy / momentum / angular-momentum conservation, and
for every non-reference run the difference from the reference at shared
checkpoints: galaxy-centre separation, centre-of-mass errors, per-particle
displacement by component, and final Lagrangian radii (radii enclosing 10%,
50% and 90% of each galaxy's disc and bulge particles).

Per-particle displacements always grow in a chaotic N-body system, whatever
the time step; the macroscopic quantities are the meaningful accuracy tests.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

COMPONENTS = {0: "disc", 1: "bulge", 2: "halo"}


def load(run):
    d = Path(run) / "fidelity"
    summary = json.loads((d / "summary.json").read_text())
    checkpoints = {int(p.stem.split("_")[1]): p for p in sorted((d / "ckpt").glob("pos_*.npy"))}
    return summary, checkpoints, d


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("reference")
    ap.add_argument("runs", nargs="*")
    a = ap.parse_args()

    ref_summary, ref_ck, ref_dir = load(a.reference)
    origin = np.load(ref_dir / "origin.npy")
    component = np.load(ref_dir / "component.npy")
    mass = np.load(ref_dir / "mass.npy").astype(np.float64)
    align = ref_summary["frames"] - 1
    align //= ref_summary.get("checkpoint_ratio", 1)
    span = ref_summary["log"][-1]["t_gyr"]

    def centres(p):
        return [np.average(p[origin == g], axis=0, weights=mass[origin == g]) for g in (0, 1)]

    def lagrangian(p, g, comp, fractions=(.1, .5, .9)):
        c = centres(p)[g]
        r = np.sort(np.linalg.norm(p[(origin == g) & (component == comp)] - c, axis=1))
        return [float(r[int(f * (len(r) - 1))]) for f in fractions]

    print("=== Conservation (full state, every saved frame) ===")
    for run in [a.reference, *a.runs]:
        s, _, _ = load(run)
        log = s["log"]
        e0 = log[0]["e"]
        de = np.array([(r["e"] - e0) / abs(e0) for r in log])
        l0 = np.array(log[0]["l"])
        dl = max(np.linalg.norm(np.array(r["l"]) - l0) for r in log) / np.linalg.norm(l0)
        momentum = np.array([r["p"] for r in log])
        # Momentum drift as an equivalent centre-of-mass velocity change.
        dv = np.linalg.norm(momentum - momentum[0], axis=1).max() / mass.sum()
        print(f"{s['label']:14s} {s['method']:15s} {s.get('precision', 'fp32'):5s} "
              f"dt={s['dt_myr']:7.3f} Myr steps={s['total_steps']:5d} sim={s['sim_seconds']:6.1f}s "
              f"wall={s['wall_seconds']:6.1f}s | dE/E final={de[-1]:+.2e} max={np.abs(de).max():.2e} "
              f"| dL/L max={dl:.2e} | CoM dv max={dv:.2e} km/s "
              f"| nonfinite={max(r['nonfinite'] for r in log)}")

    for run in a.runs:
        s, ck, _ = load(run)
        shared = sorted(set(ck) & set(ref_ck))
        print(f"\n=== {s['label']} vs {ref_summary['label']} at shared times ===")
        print(" t_Gyr  sep_run  sep_ref  |dsep|  centre_err(MW,M31)  median|dx| disc/bulge/halo  p90|dx| disc")
        for idx in shared:
            p = np.load(ck[idx]).astype(np.float64)
            q = np.load(ref_ck[idx]).astype(np.float64)
            cp_, cq = centres(p), centres(q)
            sp, sq = np.linalg.norm(cp_[1] - cp_[0]), np.linalg.norm(cq[1] - cq[0])
            dx = np.linalg.norm(p - q, axis=1)
            med = [np.median(dx[component == k]) for k in (0, 1, 2)]
            if idx % 9 == 0 or idx == shared[-1]:
                print(f" {idx / align * span:5.2f}  {sp:7.1f}  {sq:7.1f}  {abs(sp - sq):6.2f}  "
                      f"{np.linalg.norm(cp_[0] - cq[0]):6.2f} {np.linalg.norm(cp_[1] - cq[1]):6.2f}"
                      f"       {med[0]:7.2f} {med[1]:7.2f} {med[2]:7.2f}"
                      f"      {np.percentile(dx[component == 0], 90):7.2f}")
        last = shared[-1]
        p = np.load(ck[last]).astype(np.float64)
        q = np.load(ref_ck[last]).astype(np.float64)
        print(" final Lagrangian radii (10/50/90%, kpc)  run / reference:")
        for g, name in ((0, "MW"), (1, "M31")):
            for k in (0, 1):
                pairs = zip(lagrangian(p, g, k), lagrangian(q, g, k))
                print(f"   {name:3s} {COMPONENTS[k]:5s}  " + "  ".join(f"{x:6.1f}/{y:6.1f}" for x, y in pairs))


if __name__ == "__main__":
    main()
