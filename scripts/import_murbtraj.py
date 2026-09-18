"""Turn a .murbtraj recorded by NBody-EuroHPC (e.g. on Leonardo) into a saved run.

    python scripts/import_murbtraj.py path/to/run.murbtraj [--frames 140] [--profile desktop] [--zoom 1.0]

No MUrB executable is needed: the trajectory is only rendered, not recomputed.
The run appears under "Saved runs" in the dashboard as a MUrB N-body run.
"""
from pathlib import Path
import argparse, sys, time, uuid

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_demo import ROOT, canonical_profile, load_profiles
from leonardo_demos.base import RunContext
from leonardo_demos.demos.nbody_murb import NBodyMurbDemo, Trajectory

ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
ap.add_argument("trajectory", type=Path)
ap.add_argument("--frames", type=int, default=140, help="frames to render (default 140)")
ap.add_argument("--profile", default="desktop", help="sets the render size (default desktop)")
ap.add_argument("--zoom", type=float, default=1.0, help="1 = MUrB's own camera")
a = ap.parse_args()

traj = Trajectory(a.trajectory)  # validates the file before creating a run
profile = canonical_profile(a.profile)
settings = dict(load_profiles()[profile]["nbody_murb"])
rid = f"nbody_murb_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:5]}"
params = {"scheme": 0, "dt": traj.dt, "zoom": a.zoom,
          "_trajectory_path": str(a.trajectory.resolve())}
# The method records the backend that actually computed the trajectory.
ctx = RunContext(ROOT / "runs" / rid, "nbody_murb", profile, a.frames, params, "cpu",
                 "cpu", traj.backend or "imported")
ctx.write_meta({"settings": settings, "settings_override": {}})
try:
    NBodyMurbDemo(ctx, settings).run()
except Exception as exc:
    ctx.fail(exc)
    raise
print(f"{traj.n:,} bodies, {traj.frame_count} recorded frames from {traj.backend} -> runs/{rid}")
