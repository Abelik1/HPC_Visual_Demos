"""Is the stand ready? Checks what demo day actually needs, and nothing else.

For every demo in the active lineup (or all of them), it checks that the runs
the walk-up viewer will play are really there and complete: the frames the
meta.json claims, the interactive views the demo declares, the overlays, and
the videos for the video-only entries. It touches no cluster and starts no
computation, so it is safe to run five minutes before the doors open.

    python tools/check_stand.py                 # the active demo day
    python tools/check_stand.py --machine all   # both demo days
    python tools/check_stand.py --quiet          # only problems
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

RUNS = ROOT / "runs"
OK, WARN, BAD = "ok", "warn", "problem"


def load(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def check_run(rid: str) -> tuple[str, str]:
    """One showcase run: does the viewer have everything it will ask for?"""
    run_dir = RUNS / rid
    meta = load(run_dir / "meta.json")
    if meta is None:
        return BAD, "missing (not imported?)"
    if meta.get("status") != "complete":
        return BAD, f"status {meta.get('status')}"
    claimed = int(meta.get("frames") or 0)
    found = len(list((run_dir / "frames").glob("frame_*.jpg"))) if (run_dir / "frames").is_dir() else 0
    notes = []
    if found < claimed:
        return BAD, f"{found} of {claimed} frames on disk"
    # Interactive viewers: each declares its own folder of per-frame files.
    for key, pattern in (("galaxy3d_view", "*.json"), ("fusion_view", "*.json")):
        view = meta.get(key)
        if isinstance(view, dict) and view.get("folder"):
            folder = run_dir / view["folder"]
            n = len(list(folder.glob(pattern))) if folder.is_dir() else 0
            if not n:
                notes.append(f"no {key} files")
            elif n < int(view.get("frames") or 0):
                notes.append(f"{key} {n}/{view.get('frames')}")
    for overlay in meta.get("overlays") or []:
        folder = run_dir / "overlays" / overlay
        if not folder.is_dir() or not any(folder.iterdir()):
            notes.append(f"no {overlay} overlay")
    if meta.get("reveal") and not (run_dir / "reveal.jpg").exists():
        notes.append("no reveal image")
    size = sum(f.stat().st_size for f in run_dir.rglob("*") if f.is_file()) / 2**30
    summary = f"{found} frames, {size:.1f} GB" + (" - " + "; ".join(notes) if notes else "")
    return (WARN if notes else OK), summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--machine", default="active", help="discoverer, leonardo, all, or active (default)")
    ap.add_argument("--quiet", action="store_true", help="print only warnings and problems")
    a = ap.parse_args(argv)

    lineups = load(ROOT / "config" / "lineups.json") or {}
    library = load(RUNS / "_library.json") or {"showcase": {}, "favourites": []}
    machines = lineups.get("machines") or {}
    wanted = list(machines) if a.machine in ("all", "active") and a.machine == "all" else \
        [a.machine] if a.machine in machines else \
        ([lineups.get("active")] if lineups.get("active") in machines else list(machines))

    worst = OK
    for machine in wanted:
        block = machines[machine]
        print(f"\n{block.get('label', machine)} demo day")
        for demo in block.get("demos", []):
            extra = (lineups.get("extras") or {}).get(demo)
            if extra:
                folder = ROOT / "videos" / (extra.get("folder") or "")
                films = sorted(folder.glob("*.mp4")) if folder.is_dir() else []
                state = OK if films else BAD
                line = f"{len(films)} video(s)" if films else f"no videos in videos/{extra.get('folder')}"
                worst = BAD if state == BAD else worst
                if state != OK or not a.quiet:
                    print(f"  [{state:7}] {demo:22} {line}")
                continue
            picks = (library.get("showcase") or {}).get(demo) or []
            if not picks:
                worst = BAD
                print(f"  [{BAD:7}] {demo:22} no showcase run: the stand has nothing to play")
                continue
            for rid in picks:
                state, line = check_run(rid)
                if state == BAD:
                    worst = BAD
                elif state == WARN and worst == OK:
                    worst = WARN
                if state != OK or not a.quiet:
                    print(f"  [{state:7}] {demo:22} {rid}\n{'':32}{line}")

    gaia = (ROOT / "data" / "gaia_sky.npz").exists()
    if not gaia:
        print("\n  [warn   ] data/gaia_sky.npz is missing: saved black-hole runs still replay,"
              "\n            but a live run cannot start (python tools/fetch_gaia_sky.py)")
        worst = WARN if worst == OK else worst
    print("\n" + {OK: "Stand is ready.", WARN: "Stand will run, with the warnings above.",
                  BAD: "NOT ready: see the problems above."}[worst])
    return 0 if worst != BAD else 1


if __name__ == "__main__":
    raise SystemExit(main())
