import json, struct, tempfile, unittest
from pathlib import Path

import numpy as np

from leonardo_demos.base import RunContext
from leonardo_demos.demos.nbody_murb import NBodyMurbDemo, Trajectory, find_murb


def write_murbtraj(path, positions, velocities, dt=3600.0, stride=1, backend=b"cpu+omp"):
    """Minimal .murbtraj v1 writer following TRAJECTORY_FORMAT.md."""
    frames, n, _ = positions.shape
    commit = b"test"
    header = 72 + len(backend) + len(commit) + 4 * n
    with open(path, "wb") as f:
        f.write(b"MURBTRJ\0")
        f.write(struct.pack("<4I", 1, 0x01020304, 1, 1))
        f.write(struct.pack("<4Q", header, n, frames, stride))
        f.write(struct.pack("<d", dt))
        f.write(struct.pack("<2I", len(backend), len(commit)))
        f.write(backend + commit)
        f.write(np.full(n, 1.0e6, "<f4").tobytes())
        for k in range(frames):
            f.write(struct.pack("<Q", (k + 1) * stride))
            for a in (positions[k].T, velocities[k].T):
                f.write(np.ascontiguousarray(a, "<f4").tobytes())


class NBodyMurbTest(unittest.TestCase):
    def test_imported_trajectory_renders_every_view(self):
        rng = np.random.default_rng(1)
        pos = rng.normal(0, 1.5e8, (6, 200, 3))
        vel = rng.normal(0, 1e3, (6, 200, 3))
        with tempfile.TemporaryDirectory() as t:
            traj = Path(t) / "a.murbtraj"
            write_murbtraj(traj, pos, vel)
            self.assertEqual(Trajectory(traj).frame_count, 6)
            run = Path(t) / "run"
            ctx = RunContext(run, "nbody_murb", "local", 3,
                             {"zoom": 1.0, "_trajectory_path": str(traj)}, "cpu", "cpu", "cpu+omp")
            NBodyMurbDemo(ctx, {"width": 160, "height": 90}).run()
            meta = json.loads((run / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["frames"], 3)
            for folder in ("frames", "modes/side", "modes/top"):
                self.assertTrue((run / folder / "frame_0002.jpg").exists(), folder)

    def test_truncated_trajectory_is_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            traj = Path(t) / "a.murbtraj"
            write_murbtraj(traj, np.zeros((2, 10, 3)), np.zeros((2, 10, 3)))
            traj.write_bytes(traj.read_bytes()[:-8])
            with self.assertRaises(ValueError):
                Trajectory(traj)

    @unittest.skipUnless(find_murb(), "MUrB executable not built")
    def test_live_run_with_the_executable(self):
        with tempfile.TemporaryDirectory() as t:
            ctx = RunContext(Path(t), "nbody_murb", "local", 4, {"scheme": 0, "dt": 3600, "zoom": 1},
                             "cpu", "cpu", "cpu+omp")
            NBodyMurbDemo(ctx, {"bodies": 300, "iterations": 20, "warmup": 1,
                                "width": 160, "height": 90}).run()
            meta = json.loads((Path(t) / "meta.json").read_text(encoding="utf-8"))
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["murb"]["report"]["completed_iterations"], "20")
            self.assertFalse((Path(t) / "trajectory.murbtraj").exists())


if __name__ == "__main__":
    unittest.main()
