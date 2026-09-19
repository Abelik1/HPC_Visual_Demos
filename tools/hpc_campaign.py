"""Run every demo of one machine's lineup at HPC scale, inside a time budget.

The campaign is config/hpc_plan.json -> campaign.<machine>. Each entry is the
production (demo-day) run. Nothing is submitted without --confirm.

    python tools/hpc_campaign.py plan discoverer
        What would run, on which resources, and what it bills per hour.

    python tools/hpc_campaign.py pilot discoverer --confirm
        Every demo at production scale but only a few frames / steps, all in
        parallel. Measures real speed on the real hardware and estimates each
        production run. Results: benchmarks/campaign_<machine>_pilot.json.

    python tools/hpc_campaign.py run discoverer --confirm [--only fluid,...]
        Submits the production runs whose pilot estimate is under
        target_minutes (25; the hard limit is 30), waits, fetches them into
        runs/, and writes the presentation stats:
        benchmarks/campaign_<machine>_stats.md / .json

Jobs go through the same code path as the dashboard's "Run on" button
(leonardo_demos/remote.py): code sync, right-sized sbatch, polling, fetch, and
per-run device-usage records.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from leonardo_demos import remote  # noqa: E402
from leonardo_demos.registry import DEMOS  # noqa: E402
from run_demo import load_profiles, load_specs  # noqa: E402

RUNS = ROOT / "runs"
BENCH = ROOT / "benchmarks"
# The quantity a demo's physics cost is proportional to (per run).
WORK_KEY = {"galaxy_collision_3d": "frames", "fluid": "total_steps", "neuro_racers": "generations",
            "bat_vs_moth": "generations", "black_hole": "frames", "molecular_dynamics": None,
            "nbody_murb": "iterations", "fusion_plasma": None, "cosmic_web": "total_steps"}


def load_plan() -> dict:
    return json.loads((ROOT / "config" / "hpc_plan.json").read_text(encoding="utf-8"))


def work_key(entry: dict) -> str:
    demo, method = entry["demo"], entry.get("method")
    if demo == "molecular_dynamics":
        return "shuttle_steps" if method == "shuttle" else "total_steps"
    if demo == "fusion_plasma":
        return "train_updates" if method == "guardian" else "total_steps"
    return WORK_KEY.get(demo) or "frames"


def label(entry: dict) -> str:
    return entry["demo"] + (f"/{entry['method']}" if entry.get("method") else "")


def resolved(entry: dict, pilot: bool) -> dict:
    """Frames, settings and params for the production run or its pilot."""
    frames = int(entry["frames"])
    settings = dict(entry.get("settings") or {})
    params = dict(entry.get("params") or {})
    if pilot:
        p = entry.get("pilot") or {}
        pframes = int(p.get("frames", 6))
        settings.update(p.get("settings") or {})
        if entry["demo"] == "galaxy_collision_3d":
            # Same time step as production: the simulated span shrinks with
            # the frame count, so each force evaluation costs the same.
            settings["span_gyr"] = float(entry["settings"]["span_gyr"]) * (pframes - 1) / (frames - 1)
        frames = pframes
    return {"frames": frames, "settings": settings, "params": params}


def work_amount(entry: dict, spec: dict) -> float:
    key = work_key(entry)
    if key == "frames":
        return float(spec["frames"])
    value = spec["settings"].get(key)
    if value is None:
        value = load_profiles()[entry.get("profile", "hpc")][entry["demo"]].get(key)
    return float(value or 1)


def validate(entry: dict) -> list[str]:
    problems = []
    demo = entry["demo"]
    if demo not in DEMOS:
        return [f"unknown demo {demo}"]
    cls = DEMOS[demo]
    method = entry.get("method") or cls.default_method
    if method not in getattr(cls, "remote_methods", cls.methods):
        problems.append(f"{demo}: unknown method {method}")
    spec_params = load_specs()[demo]["params"]
    for k, v in (entry.get("params") or {}).items():
        if k.startswith("_"):
            continue
        if k not in spec_params:
            problems.append(f"{demo}: unknown parameter {k}")
        elif not spec_params[k]["min"] <= float(v) <= spec_params[k]["max"]:
            problems.append(f"{demo}: {k}={v} outside {spec_params[k]['min']}..{spec_params[k]['max']}")
    preset = load_profiles()[entry.get("profile", "hpc")][demo]
    for k in (entry.get("settings") or {}):
        if k not in preset:
            problems.append(f"{demo}: setting {k} is not in the {entry.get('profile', 'hpc')} preset")
    if demo == "galaxy_collision_3d" and float((entry.get("settings") or {}).get("particles", 0)) > 1_000_000:
        problems.append("galaxy_collision_3d: more than 1,000,000 bodies")
    return problems


def job_kwargs(entry: dict, spec: dict, run_dir: Path) -> dict:
    demo = entry["demo"]
    cls = DEMOS[demo]
    method = entry.get("method") or cls.default_method
    need = remote.demo_needs(demo, method)
    return dict(demo=demo, profile=entry.get("profile", "hpc"), frames=spec["frames"],
                params={k: float(v) for k, v in spec["params"].items()},
                backend=entry.get("backend") or need.get("backend", "auto"), run_dir=run_dir, method=method,
                numerical_substeps=None, settings_override=spec["settings"],
                precision=entry.get("precision", "fp32"))


def submit(machine: str, entry: dict, kind: str, walltime: str) -> str:
    c = remote.cluster(machine)
    spec = resolved(entry, pilot=(kind == "pilot"))
    stamp = time.strftime("%Y%m%d_%H%M%S")
    rid = f"{entry['demo']}_{entry.get('method') or 'default'}_{machine}_{kind}_{stamp}".replace("+", "")
    kwargs = job_kwargs(entry, spec, RUNS / rid)
    remote.create_job(RUNS, rid, c, kwargs, walltime, {})
    return rid


def wait(run_ids: list[str], poll: int = 20) -> dict:
    status = {}
    while True:
        for rid in run_ids:
            try:
                meta = json.loads((RUNS / rid / "meta.json").read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            status[rid] = meta
        pending = [r for r in run_ids if status.get(r, {}).get("status") not in ("complete", "failed")]
        line = " | ".join(f"{r.split('_')[0]}:{(status.get(r, {}).get('remote') or {}).get('stage', '?')}"
                          for r in pending)
        print(time.strftime("%H:%M:%S"), f"{len(run_ids) - len(pending)}/{len(run_ids)} done", line, flush=True)
        if not pending:
            return status
        time.sleep(poll)


# Stages that run alongside the solver loop instead of after it, and on how
# many workers: the 3-D galaxy draws on one thread; the FramePipeline demos
# draw in worker processes (allocated cores - 1, at most 16).
OVERLAPPED = {"galaxy_collision_3d": 1, "fluid": "pipeline", "fusion_plasma/passive": "pipeline",
              "black_hole": "pipeline", "nbody_murb": "pipeline"}
# Demos whose parallel drawing comes after the physics rather than beside it.
DRAW_AFTER = {"nbody_murb"}
SIDE_STAGES = ("render", "jpeg_encode", "frame_write")


def estimate(entry: dict, meta: dict) -> dict:
    """Production minutes from a pilot.

    One-off costs (start-up, catalogue loading, GPU kernel compilation on the
    first call) are counted once. Every stage's steady per-call cost (its
    total minus its slowest call) is scaled to the production frame count,
    and the solver's by the work per frame as well. Stages that run beside
    the solver count as overlapped; everything else is serial.
    """
    t = meta.get("timings") or {}
    prod, pilot = resolved(entry, False), resolved(entry, True)
    frame_ratio = prod["frames"] / max(1, pilot["frames"])
    work_per_frame_ratio = ((work_amount(entry, prod) / prod["frames"]) /
                            max(1e-9, work_amount(entry, pilot) / pilot["frames"]))
    overlap = OVERLAPPED.get(label(entry), OVERLAPPED.get(entry["demo"]))
    if overlap == "pipeline":
        need = remote.demo_needs(entry["demo"], entry.get("method"))
        overlap = max(1, min(int(need["cpus"]) - 1, 16, prod["frames"] // 6))
    fixed = main = side = 0.0
    for stage, row in t.items():
        seconds, count, worst = float(row["seconds"]), int(row["count"]), float(row["max_seconds"])
        steady = (seconds - worst) / (count - 1) if count > 1 else seconds
        fixed += max(0.0, worst - steady) if count > 1 else (seconds if stage == "initialization" else 0.0)
        if stage == "initialization":
            continue
        # A stage called once per frame interval (frames - 1 times) scales
        # with intervals, not frames; otherwise short pilots undercount.
        pf, F = pilot["frames"], prod["frames"]
        calls = count * (F - 1) / (pf - 1) if count == pf - 1 and pf > 1 else max(1, count) * frame_ratio
        cost = steady * calls * (work_per_frame_ratio if stage == "simulation" else 1.0)
        if overlap and stage in SIDE_STAGES:
            side += cost / overlap
        else:
            main += cost
    elapsed = float(meta.get("elapsed") or 0)
    report = ((meta.get("murb") or {}).get("report") or {})
    if report.get("average_ms_per_iteration"):
        # MUrB is a separate program: its own per-iteration timing is the
        # physics, and the pilot's remaining time is start-up and drawing.
        murb = float(report["average_ms_per_iteration"]) / 1000
        pilot_iters = float(meta["murb"].get("iterations") or work_amount(entry, pilot))
        main += murb * work_amount(entry, prod)
        unaccounted = max(0.0, elapsed - sum(float(r["seconds"]) for r in t.values()) - murb * pilot_iters)
        fixed += unaccounted
    else:
        fixed += max(0.0, elapsed - sum(float(r["seconds"]) for r in t.values()))
    seconds = fixed + (main + side if entry["demo"] in DRAW_AFTER else max(main, side))
    return {"pilot_elapsed_s": round(elapsed, 1), "fixed_s": round(fixed, 1), "main_s": round(main, 1),
            "side_s": round(side, 1), "frame_ratio": round(frame_ratio, 1),
            "estimate_minutes": round(seconds / 60, 1)}


def cmd_plan(machine: str, entries: list[dict]) -> None:
    c = remote.cluster(machine)
    print(f"{c['label']}: {len(entries)} runs\n")
    for e in entries:
        method = e.get("method") or DEMOS[e["demo"]].default_method
        res = remote.resources(c, None, e["demo"], method)
        spec = resolved(e, False)
        problems = validate(e)
        print(f"- {label(e)}: {spec['frames']} frames, {spec['settings']}")
        print(f"    {res['note']}; {res['partition'] or 'default partition'}; account {res['account']}"
              + ("  !! " + "; ".join(problems) if problems else ""))
        print(f"    why: {res['why']}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["plan", "pilot", "estimate", "run"])
    ap.add_argument("machine", choices=["discoverer", "leonardo"])
    ap.add_argument("--only", default="", help="comma-separated demo ids (or demo/method)")
    ap.add_argument("--confirm", action="store_true", help="actually submit jobs")
    ap.add_argument("--force", action="store_true", help="run even if the estimate is over target")
    a = ap.parse_args()
    plan = load_plan()
    entries = plan["campaign"][a.machine]
    only = set(filter(None, a.only.split(",")))
    if only:
        entries = [e for e in entries if e["demo"] in only or label(e) in only]
    problems = [p for e in entries for p in validate(e)]
    if problems:
        print("Plan problems:\n  " + "\n  ".join(problems))
        return 2
    BENCH.mkdir(exist_ok=True)
    pilot_file = BENCH / f"campaign_{a.machine}_pilot.json"

    if a.command == "plan":
        cmd_plan(a.machine, entries)
        return 0

    if a.command in ("pilot", "run") and not a.confirm:
        print(f"Refusing to submit {len(entries)} {a.command} job(s) to {a.machine} without --confirm.")
        cmd_plan(a.machine, entries)
        return 1

    if a.command == "pilot":
        ids = {label(e): submit(a.machine, e, "pilot", "00:15:00") for e in entries}
        print("submitted:", json.dumps(ids, indent=1))
        metas = wait(list(ids.values()))
        results = json.loads(pilot_file.read_text(encoding="utf-8")) if pilot_file.exists() else {}
        for e in entries:
            meta = metas[ids[label(e)]]
            row = {"run": ids[label(e)], "status": meta.get("status"), "error": meta.get("error"),
                   "device_usage": meta.get("device_usage")}
            if meta.get("status") == "complete":
                row.update(estimate(e, meta))
            results[label(e)] = row
        pilot_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        a.command = "estimate"

    if a.command == "estimate":
        results = json.loads(pilot_file.read_text(encoding="utf-8")) if pilot_file.exists() else {}
        for e in entries:
            r = results.get(label(e)) or {}
            try:
                meta = json.loads((RUNS / r["run"] / "meta.json").read_text(encoding="utf-8"))
            except (KeyError, OSError, ValueError):
                continue
            if meta.get("status") == "complete":
                r.update(estimate(e, meta))
        pilot_file.write_text(json.dumps(results, indent=2), encoding="utf-8")
        target = float(plan.get("target_minutes", 25))
        for e in entries:
            r = results.get(label(e)) or {}
            est = r.get("estimate_minutes")
            flag = "no pilot yet" if r == {} else (r.get("error") or r.get("status")) if est is None else \
                ("OK" if est <= target else f"OVER {target:.0f} min: scale down")
            usage = r.get("device_usage") or {}
            print(f"{label(e):32} est {est!s:>6} min  {flag:28} gpu {usage.get('gpu_util_mean')}%  "
                  f"cores {usage.get('cpu_cores_busy_avg')}/{usage.get('cpus_allocated')}")
        return 0

    # production
    results = json.loads(pilot_file.read_text(encoding="utf-8")) if pilot_file.exists() else {}
    target = float(plan.get("target_minutes", 25))
    chosen = []
    for e in entries:
        est = (results.get(label(e)) or {}).get("estimate_minutes")
        if est is None and not a.force:
            print(f"skip {label(e)}: no pilot estimate (run the pilot first)")
        elif est is not None and est > target and not a.force:
            print(f"skip {label(e)}: estimated {est} min > {target} min")
        else:
            chosen.append(e)
    walltime = "00:40:00"
    ids = {label(e): submit(a.machine, e, "run", walltime) for e in chosen}
    print("submitted:", json.dumps(ids, indent=1))
    metas = wait(list(ids.values()), poll=30)
    out = {"machine": a.machine, "runs": [{"label": k, "run": v, "status": metas[v].get("status"),
                                           "error": metas[v].get("error")} for k, v in ids.items()]}
    (BENCH / f"campaign_{a.machine}_runs.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    import subprocess
    subprocess.run([sys.executable, str(ROOT / "tools" / "demo_stats.py"), "--campaign",
                    str(BENCH / f"campaign_{a.machine}_runs.json"),
                    "--md", str(BENCH / f"campaign_{a.machine}_stats.md"),
                    "--json", str(BENCH / f"campaign_{a.machine}_stats.json")], cwd=str(ROOT))
    print(f"stats: benchmarks/campaign_{a.machine}_stats.md")
    return 0 if all(m.get("status") == "complete" for m in metas.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
