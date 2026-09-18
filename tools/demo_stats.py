"""Presentation stats for demo runs: setup, time, physics vs drawing, devices, result.

One table per run, in the form handed to the presenters ("Star in a Bottle,
job 7654, 13 Sep, gpu-12 ..."), built only from what each run recorded in its
meta.json: settings, phase timings, the node it ran on, and, for cluster runs,
the devices Slurm allocated and how busy they actually were.

    python tools/demo_stats.py runs/fusion_plasma_A_showcase7654 runs/fluid_A_showcase7655
    python tools/demo_stats.py --campaign benchmarks/campaign_discoverer.json --md out.md --json out.json

Physics = the "simulation" (and "initialization") stages; drawing = rendering,
visualisation, frame copies and JPEG writing. A pipelined run overlaps the two,
so they can add up to more than the total time; the total is wall clock.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DRAW_STAGES = ("render", "visualization", "frame_copy", "frame_write", "jpeg_encode")


def _n(v, d=0):
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    return f"{v:,.{d}f}"


def _dur(seconds):
    s = float(seconds or 0)
    if s < 60:
        return f"{s:.0f} s"
    m, s = divmod(int(round(s)), 60)
    if m < 60:
        return f"{m} min {s:02d} s"
    h, m = divmod(m, 60)
    return f"{h} h {m:02d} min"


def _per_frame(seconds, frames):
    if not frames:
        return ""
    per = float(seconds) / frames
    return f"{per * 1000:.0f} ms/frame" if per < 1 else f"{per:.2f} s/frame"


def setup_text(m: dict) -> str:
    d, s, p = m.get("demo"), m.get("settings") or {}, m.get("params") or {}
    method, frames = m.get("method"), m.get("frames")
    g = s.get
    if d == "galaxy_collision_3d":
        return (f"{_n(g('particles'))} bodies, direct all-pairs gravity, {_n(g('span_gyr'), 1)} Gyr, "
                f"{frames} frames, {m.get('precision', 'fp32')}")
    if d == "fluid":
        return (f"{_n(g('nx'))}×{_n(g('ny'))} lattice-Boltzmann grid, {_n(g('total_steps'))} steps, "
                f"{_n(g('tracers'))} tracers, {frames} frames")
    if d == "neuro_racers":
        return (f"{_n(g('population'))} cars per search × {_n(g('ensemble'))} independent searches, "
                f"{_n(g('generations'))} generations, {frames} frames")
    if d == "bat_vs_moth":
        return (f"{_n(g('population'))} bats and moths per cave × {_n(g('ensemble'))} caves, "
                f"{_n(g('generations'))} generations, {frames} frames")
    if d == "black_hole":
        return (f"{_n(g('width'))}×{_n(g('height'))} px, {_n(g('supersample'))}× supersampling, "
                f"exact Schwarzschild rays through the Gaia sky, {frames} frames")
    if d == "molecular_dynamics":
        if method == "shuttle":
            return f"rotaxane ring on an axle, {_n(g('shuttle_steps'))} Langevin steps, {frames} frames"
        return f"{_n(m.get('chain_length') or g('particles'))}-bead chain, {_n(g('total_steps'))} Langevin steps, {frames} frames"
    if d == "nbody_murb":
        return f"{_n(g('bodies'))} bodies, {_n(g('iterations'))} iterations, MUrB {method}, {frames} frames"
    if d == "fusion_plasma":
        if method == "guardian":
            return (f"{_n(g('ensemble'))} simulations run together, {_n(g('n'))}² grid, {_n(g('shots'))} shots, "
                    f"{_n(g('train_updates'))} neural-network updates, {frames} frames")
        return (f"{_n(g('n'))}² grid, {_n(g('total_steps'))} steps, {_n(g('tracers'))} tracers, "
                f"{_n(p.get('magnetic_field'), 1)} T field, {_n(p.get('heating'))} MW heating, {frames} frames")
    if d == "cosmic_web":
        return f"{_n(g('grid'))} grid, {_n(g('particles'))} particles, {_n(g('total_steps'))} steps, {frames} frames"
    return ", ".join(f"{k} {v}" for k, v in list(s.items())[:5]) + f", {frames} frames"


def result_text(m: dict) -> str:
    summary = m.get("summary") or {}
    d = m.get("demo")
    if d == "molecular_dynamics" and summary:
        if summary.get("mode") == "shuttle":
            return f"ring made {summary.get('trips')} trips for {summary.get('flips')} switch flips"
        return (f"folded to Rg {summary.get('radius_of_gyration')}, {round(100 * summary.get('buried_oil', 0))}% of "
                f"oily beads buried, {summary.get('salt_bridges')} salt bridges")
    overlay = m.get("overlay") or {}
    if summary:
        items = [f"{k.replace('_', ' ')} {v}" for k, v in summary.items()
                 if isinstance(v, (int, float, str)) and k not in ("mode",)][:4]
        if items:
            return "; ".join(items)
    return "; ".join(f"{k} {v}" for k, v in list(overlay.items())[:4])


def stats(run_dir: Path) -> dict:
    m = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    t = m.get("timings") or {}
    sec = lambda k: float((t.get(k) or {}).get("seconds", 0.0))
    physics = sec("simulation") + sec("initialization")
    drawing = sum(sec(k) for k in DRAW_STAGES)
    total = float(m.get("elapsed") or m.get("duration_seconds") or 0)
    frames = int(m.get("frames") or 0)
    res = m.get("resources") or {}
    usage = m.get("device_usage") or {}
    remote = m.get("remote") or {}
    from run_demo import load_specs
    name = load_specs().get(m.get("demo"), {}).get("name", m.get("demo"))
    method_label = ""
    try:
        from leonardo_demos.registry import DEMOS
        method_label = DEMOS[m["demo"]].method_labels.get(m.get("method"), "")
    except Exception:
        pass
    created = m.get("created")
    row = {
        "run": run_dir.name, "demo": m.get("demo"), "name": name, "method": m.get("method"),
        "method_label": method_label, "machine": remote.get("label") or res.get("host") or "",
        "node": usage.get("node") or res.get("host"), "job_id": remote.get("job_id") or usage.get("job_id"),
        "date": time.strftime("%d %b %Y", time.localtime(created)) if created else "",
        "backend": m.get("backend"), "gpu": res.get("gpu_name") or res.get("gpu") or "",
        "setup": setup_text(m), "total_seconds": round(total, 1), "total": _dur(total),
        "physics_seconds": round(physics, 1), "physics": f"{_dur(physics)} ({_per_frame(physics, frames)})",
        "drawing_seconds": round(drawing, 1), "drawing": f"{_dur(drawing)} ({_per_frame(drawing, frames)})",
        "physics_share": round(physics / total, 3) if total else None,
        "result": result_text(m), "status": m.get("status"), "frames": frames,
        "device_usage": usage,
    }
    if usage:
        alloc = []
        if usage.get("gpus_allocated"):
            gpu_busy = usage.get("gpu_util_mean")
            alloc.append(f"{usage['gpus_allocated']} GPU" + (f", busy {gpu_busy:.0f}% on average" if gpu_busy is not None else ""))
        if usage.get("cpus_allocated"):
            alloc.append(f"{usage['cpus_allocated']} cores, {usage.get('cpu_cores_busy_avg')} busy on average")
        row["devices"] = "; ".join(alloc)
    return row


def to_markdown(rows: list[dict]) -> str:
    out = []
    for r in rows:
        title = f"**{r['name']}**" + (f" ({r['method_label']})" if r["method_label"] else "")
        where = ", ".join(x for x in [r["machine"] and f"{r['machine']}", r["job_id"] and f"job {r['job_id']}",
                                      r["date"], r["node"] != r["machine"] and r["node"]] if x)
        out += [f"{title}, {where}", "", "| | |", "|---|---|",
                f"| Setup | {r['setup']} |", f"| Total time | **{r['total']}** |",
                f"| Physics | {r['physics']} |", f"| Drawing frames | {r['drawing']} |"]
        if r.get("devices"):
            out.append(f"| Devices | {r['devices']} |")
        out += [f"| Compute | {r['backend']} |", f"| Result | {r['result']} |", ""]
    return "\n".join(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", help="run directories")
    ap.add_argument("--campaign", help="campaign results JSON (list of run ids under runs/)")
    ap.add_argument("--md")
    ap.add_argument("--json")
    a = ap.parse_args()
    dirs = [Path(r) for r in a.runs]
    if a.campaign:
        data = json.loads(Path(a.campaign).read_text(encoding="utf-8"))
        dirs += [ROOT / "runs" / r["run"] for r in data.get("runs", []) if r.get("run")]
    rows = []
    for d in dirs:
        try:
            rows.append(stats(d))
        except (OSError, ValueError, KeyError) as error:
            print(f"skip {d}: {error}", file=sys.stderr)
    md = to_markdown(rows)
    if a.md:
        Path(a.md).write_text(md, encoding="utf-8")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not a.md and not a.json:
        print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
