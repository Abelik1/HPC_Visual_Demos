"""Redraw the ray-path view (modes/3d) of saved exact black-hole runs.

The ray paths depend only on the camera's distance and the frame's progress,
so a change to how they are drawn does not require re-tracing the Gaia sky:

    python tools/rerender_black_hole_rays.py runs/black_hole_exact_A_showcase
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leonardo_demos.base import RunContext  # noqa: E402
from leonardo_demos.demos.black_hole import BlackHoleDemo  # noqa: E402
from leonardo_demos.schwarzschild import EscapeTable  # noqa: E402


def rerender(run_dir: Path) -> None:
    meta = json.loads((run_dir / "meta.json").read_text(encoding="utf-8"))
    if meta.get("method") != "schwarzschild":
        raise SystemExit(f"{run_dir.name} is not an exact (schwarzschild) run")
    params, frames = meta["params"], int(meta["frames"])
    ctx = RunContext(run_dir, "black_hole", meta.get("profile", "local"), frames, params, "numpy",
                     method="schwarzschild")
    ctx.jpeg_subsampling = 0
    demo = BlackHoleDemo(ctx, {})
    start_rs, dive = float(params.get("camera_distance", 10.0)), float(params.get("dive", 0.0))
    bundles = {}
    for frame in range(frames):
        progress = frame / max(1, frames - 1)
        r_camera = 2.0 * max(1.6, start_rs * (1 - dive * progress))
        key = round(r_camera, 2)
        if key not in bundles:
            if len(bundles) > 4:
                bundles.clear()
            bundles[key] = demo.ray_bundle(r_camera, EscapeTable(r_camera, samples=4000))
        ctx.save_frame(demo.render_rays(r_camera, bundles[key], .6 + 1.2 * progress),
                       run_dir / "modes" / "3d" / f"frame_{frame:04d}.jpg")
        if frame % 100 == 0:
            print(f"{run_dir.name}: {frame}/{frames}", flush=True)
    print(f"{run_dir.name}: done", flush=True)


if __name__ == "__main__":
    for argument in sys.argv[1:]:
        rerender(Path(argument).resolve())
