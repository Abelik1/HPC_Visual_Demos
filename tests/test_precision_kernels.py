import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from leonardo_demos import tuning
from leonardo_demos.base import RunContext
from leonardo_demos.demos.fluid import FluidDemo
from leonardo_demos.demos.galaxy_collision import G, GalaxyCollisionDemo
from leonardo_demos.demos.galaxy_collision_3d import GalaxyCollision3DDemo
import run_demo

try:
    import cupy as cp
    HAS_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:  # pragma: no cover - CPU-only machines
    cp = None
    HAS_GPU = False


def context(root, demo, backend="numpy", precision="fp32", method="default"):
    return RunContext(Path(tempfile.mkdtemp(dir=root)), demo, "local", 2, {}, backend,
                      method=method, precision=precision)


def host(value):
    return cp.asnumpy(value) if cp is not None and isinstance(value, cp.ndarray) else np.asarray(value)


def direct_3d(positions, masses, softening):
    p = np.asarray(positions, dtype=np.float64)
    m = np.asarray(masses, dtype=np.float64)
    delta = p[None, :, :] - p[:, None, :]
    r2 = np.sum(delta * delta, axis=2) + softening ** 2
    return G * np.sum(delta * (m[None, :] / (r2 * np.sqrt(r2)))[:, :, None], axis=1)


class PrecisionSelectionTests(unittest.TestCase):
    def test_state_dtype_follows_precision(self):
        with tempfile.TemporaryDirectory() as temp:
            for precision, dtype in (("fp32", np.float32), ("mixed", np.float64), ("fp64", np.float64)):
                ctx = context(temp, "galaxy_collision_3d", precision=precision, method="leapfrog")
                pos, vel, mass = GalaxyCollision3DDemo(ctx, {}).setup(120, .35, 1., 35., 1.5e12, 1.5e12)[:3]
                self.assertEqual((pos.dtype, vel.dtype, mass.dtype), (dtype,) * 3)
                ctx.finish()
                self.assertEqual(json.loads((ctx.run_dir / "meta.json").read_text())["precision"], precision)
            ctx = context(temp, "galaxy_collision", precision="fp64")
            self.assertEqual(GalaxyCollisionDemo(ctx, {}).setup(80, 1, .55, .75, 18)[0].dtype, np.float64)
            ctx = context(temp, "fluid", precision="fp64")
            f = FluidDemo(ctx, {}).init(40, 24, .06, 0)[0]
            self.assertEqual(f.dtype, np.float64)

    def test_unsupported_precision_is_rejected_not_silently_downgraded(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaisesRegex(ValueError, "does not implement mixed"):
                run_demo.run("fluid", "benchmark", 2, run_dir=Path(temp) / "r", backend="numpy",
                             precision="mixed")
            with self.assertRaisesRegex(ValueError, "unknown precision"):
                RunContext(Path(temp) / "x", "fluid", "local", 2, {}, "numpy", precision="fp16")

    def test_cpu_fp64_and_fp32_agree_for_every_collision_solver(self):
        with tempfile.TemporaryDirectory() as temp:
            for method in GalaxyCollisionDemo.methods:
                results = []
                for precision in ("fp32", "fp64"):
                    demo = GalaxyCollisionDemo(context(temp, "galaxy_collision", precision=precision,
                                                       method=method), {})
                    state = list(demo.setup(300, 1, .55, .75, 18))
                    for _ in range(3):
                        state[:6] = demo.step(*state[:8], .004, 4)
                    results.append(np.asarray(state[0], dtype=np.float64))
                with self.subTest(method=method):
                    np.testing.assert_allclose(results[0], results[1], rtol=0, atol=2e-3)

    def test_leapfrog_force_reuse_is_bit_identical(self):
        """Reusing the closing force across frames must not change the orbit."""
        with tempfile.TemporaryDirectory() as temp:
            finals = []
            for reuse in (True, False):
                ctx = context(temp, "galaxy_collision_3d", method="leapfrog")
                demo = GalaxyCollision3DDemo(ctx, {"force_tile": 64})
                pos, vel, mass = demo.setup(150, .35, 1., 35., 1.5e12, 1.5e12)[:3]
                for _ in range(3):
                    if not reuse:
                        demo._acc_cache = None
                    pos, vel = demo.step(pos, vel, mass, .01, 2, 4.0)
                finals.append((pos.copy(), vel.copy()))
            np.testing.assert_array_equal(finals[0][0], finals[1][0])
            np.testing.assert_array_equal(finals[0][1], finals[1][1])


@unittest.skipUnless(HAS_GPU, "CUDA device required")
class GpuKernelTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        patcher = patch.dict(os.environ, {"LEONARDO_DEMO_TUNING_CACHE":
                                          str(Path(self.temp.name) / "tuning.json")})
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(self.temp.cleanup)
        tuning._MEMORY.clear()

    def test_all_pairs_kernel_matches_direct_sum_in_every_precision_and_launch_shape(self):
        tolerance = {"fp32": 5e-5, "mixed": 5e-5, "fp64": 1e-12}
        for precision in GalaxyCollision3DDemo.precisions:
            ctx = context(self.temp.name, "galaxy_collision_3d", "gpu", precision, "leapfrog")
            demo = GalaxyCollision3DDemo(ctx, {})
            pos, _, mass = demo.setup(1203, .35, 1., 35., 1.5e12, 1.5e12)[:3]
            expected = direct_3d(host(pos), host(mass), 4.0)
            scale = np.abs(expected).max()
            got = host(demo.acceleration(pos, mass, 4.0))
            self.assertIn(ctx._kernels["galaxy3d_all_pairs"]["mode"], {"measured", "cached"})
            np.testing.assert_allclose(got, expected, rtol=0, atol=tolerance[precision] * scale)
            # Every candidate shape must compute the same field, including
            # ragged final blocks (1203 bodies is prime-ish on purpose).
            for block, per_thread in ((64, 8), (512, 1), (128, 4)):
                out = cp.empty((1203, 3), dtype=ctx.precision_spec["accumulate"])
                demo._gpu_launch(demo._body, out, 1203, 1203, 4.0,
                                 {"block": block, "per_thread": per_thread})
                np.testing.assert_allclose(host(out), expected, rtol=0,
                                           atol=tolerance[precision] * scale)
            ctx.finish()

    def test_mixed_precision_reduces_momentum_error_of_the_force_sum(self):
        """Mixed mode's FP64 tile accumulation should conserve momentum better."""
        residual = {}
        for precision in ("fp32", "mixed"):
            ctx = context(self.temp.name, "galaxy_collision_3d", "gpu", precision, "leapfrog")
            demo = GalaxyCollision3DDemo(ctx, {})
            pos, _, mass = demo.setup(20000, .35, 1., 35., 1.5e12, 1.5e12)[:3]
            force = host(demo.acceleration(pos, mass, 4.0)).astype(np.float64) * host(mass)[:, None]
            residual[precision] = np.linalg.norm(force.sum(axis=0)) / np.abs(force).sum()
            ctx.finish()
        self.assertLess(residual["mixed"], residual["fp32"])

    def test_fused_tracer_kernel_matches_cpu_solver(self):
        for method in GalaxyCollisionDemo.methods:
            for precision, atol in (("fp32", 5e-4), ("fp64", 1e-9)):
                states = []
                for backend in ("numpy", "gpu"):
                    ctx = context(self.temp.name, "galaxy_collision", backend, precision, method)
                    demo = GalaxyCollisionDemo(ctx, {})
                    state = list(demo.setup(2049, 1, .55, .75, 18))
                    for _ in range(3):
                        state[:6] = demo.step(*state[:8], .004, 5)
                    states.append([host(x).astype(np.float64) for x in state[:6]])
                with self.subTest(method=method, precision=precision):
                    for cpu, gpu in zip(*states):
                        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=atol)

    def test_fused_lattice_boltzmann_kernel_matches_cpu_solver(self):
        for precision, atol in (("fp32", 2e-5), ("fp64", 1e-12)):
            for preset in (0, 2):
                fields = []
                for backend in ("numpy", "gpu"):
                    ctx = context(self.temp.name, "fluid", backend, precision)
                    demo = FluidDemo(ctx, {})
                    f, c, w, mask = demo.init(90, 50, .06, preset)
                    for steps in (7, 30):
                        f, ux, uy, vort, rho = demo.step(f, c, w, mask, .06, steps)
                    fields.append([host(x).astype(np.float64) for x in (ux, uy, rho, vort)])
                with self.subTest(precision=precision, preset=preset):
                    for cpu, gpu in zip(*fields):
                        np.testing.assert_allclose(gpu, cpu, rtol=0, atol=atol)

    def test_hybrid_render_overlap_writes_the_same_ordered_run_as_sequential(self):
        outputs = {}
        for backend in ("gpu", "hybrid"):
            run_dir = Path(self.temp.name) / backend
            ctx = RunContext(run_dir, "galaxy_collision_3d", "local", 5,
                             {"impact": .35, "speed": 1.0, "disc_tilt": 35, "softening": 4},
                             backend, method="leapfrog", timings_enabled=True)
            GalaxyCollision3DDemo(ctx, {"particles": 400, "substeps": 1, "span_gyr": .2}).run()
            meta = json.loads((run_dir / "meta.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["frame"], 4)
            self.assertEqual(sorted(p.name for p in (run_dir / "frames").glob("*.jpg")),
                             [f"frame_{i:04d}.jpg" for i in range(5)])
            outputs[backend] = [json.loads((run_dir / f"interactive/frame_{i:04d}.json").read_text())
                                for i in range(5)]
            self.assertEqual(meta["render_pipeline"],
                             "overlapped with GPU integration" if backend == "hybrid" else "sequential")
        for sequential, overlapped in zip(outputs["gpu"], outputs["hybrid"]):
            self.assertEqual(sequential["positions"], overlapped["positions"])
            self.assertEqual(sequential["time_gyr"], overlapped["time_gyr"])

    def test_autotuner_caches_its_choice_per_device_and_problem(self):
        calls = []
        pick, report = tuning.select(cp, "unit_test_kernel", "n8", [{"block": 1}, {"block": 2}],
                                     lambda c: calls.append(c["block"]), default={"block": 1})
        self.assertEqual(report["mode"], "measured")
        tuning._MEMORY.clear()  # force the file cache to be used
        again, report = tuning.select(cp, "unit_test_kernel", "n8", [{"block": 1}, {"block": 2}],
                                      lambda c: calls.append(-1), default={"block": 1})
        self.assertEqual((again, report["mode"]), (pick, "cached"))
        self.assertNotIn(-1, calls)
        with patch.dict(os.environ, {"LEONARDO_DEMO_AUTOTUNE": "0"}):
            chosen, report = tuning.select(cp, "other", "n8", [{"block": 2}], lambda c: None,
                                           default={"block": 7})
        self.assertEqual(chosen, {"block": 7})


if __name__ == "__main__":
    unittest.main()
