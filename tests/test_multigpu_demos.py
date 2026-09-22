"""Split physics must give the same answer as one device.

Each demo's multi-GPU path is exercised with NumPy and repeated device ids
(three "GPUs" that are really the host), and its multi-core path with four
CPU workers; both are compared with the plain one-device result.  When CuPy
and a GPU are present (on the clusters) the same comparison runs split across
the GPU treated as several devices.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leonardo_demos.base import RunContext
from leonardo_demos.demos.black_hole import BlackHoleDemo
from leonardo_demos.demos.galaxy_collision_3d import GalaxyCollision3DDemo
from leonardo_demos.demos.neuro_racers import NeuroRacersDemo, RaceSim, track_geometry
from leonardo_demos.multigpu import Devices
from leonardo_demos.neuroevo import Population
from leonardo_demos.schwarzschild import EscapeTable

try:
    import cupy as cp
    HAVE_GPU = cp.cuda.runtime.getDeviceCount() > 0
except Exception:
    cp, HAVE_GPU = None, False


def context(root, demo, backend="numpy", **kw):
    return RunContext(Path(root), demo, "local", 2, {}, backend, **kw)


class GalaxySplitTests(unittest.TestCase):
    def bodies(self, count=700):
        rng = np.random.default_rng(3)
        return (rng.normal(0, 5, (count, 3)).astype(np.float32),
                rng.uniform(1e8, 1e9, count).astype(np.float32))

    def test_cores_split_by_target_gives_identical_forces(self):
        positions, masses = self.bodies()
        with tempfile.TemporaryDirectory() as t:
            ctx = context(t, "galaxy_collision_3d", method="leapfrog")
            demo = GalaxyCollision3DDemo(ctx, {"force_tile": 64})
            ctx.cpu_workers = 1
            one = demo.acceleration(positions, masses, 1.5)
            ctx.cpu_workers = 4
            many = demo.acceleration(positions, masses, 1.5)
            ctx.shutdown_compute()
        np.testing.assert_array_equal(one, many)

    @unittest.skipUnless(HAVE_GPU, "needs CuPy and a GPU")
    def test_gpus_split_by_target_match_one_gpu(self):
        positions, masses = self.bodies(5003)
        with tempfile.TemporaryDirectory() as t:
            ctx = context(t, "galaxy_collision_3d", backend="gpu", method="leapfrog")
            demo = GalaxyCollision3DDemo(ctx, {})
            p, m = cp.asarray(positions), cp.asarray(masses)
            one = cp.asnumpy(demo.acceleration(p, m, 1.5))
            ctx._devices = Devices(cp, [0] * 3 if cp.cuda.runtime.getDeviceCount() < 3 else [0, 1, 2])
            many = cp.asnumpy(demo.acceleration(p, m, 1.5))
            ctx.shutdown_compute()
        np.testing.assert_allclose(one, many, rtol=1e-6, atol=1e-7)


class BlackHoleSplitTests(unittest.TestCase):
    SETTINGS = {"width": 64, "height": 37, "supersample": 1, "fov_deg": 80}

    def frame(self, ctx, xp=np):
        rng = np.random.default_rng(1)
        fine = xp.asarray(rng.random((6, 16, 16, 3)).astype(np.float32))
        glow = xp.asarray(rng.random((32, 64, 3)).astype(np.float32))
        demo = BlackHoleDemo(ctx, dict(self.SETTINGS))
        axis, forward, right, up = np.array([0, 0, 1.]), np.array([0, 0, -1.]), np.array([1., 0, 0]), np.array([0, 1., 0])
        return demo.render_camera(fine, glow, EscapeTable(8.0), axis, forward, right, up, 8.0)

    def check(self, one, many, to_host=np.asarray):
        np.testing.assert_allclose(to_host(one[0]), to_host(many[0]), rtol=1e-12)
        np.testing.assert_allclose(to_host(one[1]), to_host(many[1]), rtol=1e-12)
        self.assertAlmostEqual(one[2], many[2], places=12)

    def test_row_bands_on_devices_and_cores_match_one_image(self):
        with tempfile.TemporaryDirectory() as t:
            ctx = context(t, "black_hole", method="schwarzschild")
            ctx.cpu_workers = 1
            one = self.frame(ctx)
            ctx._devices = Devices(np, [0, 0, 0])
            bands = self.frame(ctx)
            ctx._devices = None
            ctx.cpu_workers = 4
            cores = self.frame(ctx)
            ctx.shutdown_compute()
        self.check(one, bands)
        self.check(one, cores)

    @unittest.skipUnless(HAVE_GPU, "needs CuPy and a GPU")
    def test_row_bands_on_gpus_match_one_gpu(self):
        with tempfile.TemporaryDirectory() as t:
            ctx = context(t, "black_hole", backend="gpu", method="schwarzschild")
            one = self.frame(ctx, cp)
            ctx._devices = Devices(cp, [0, 0])
            many = self.frame(ctx, cp)
            ctx.shutdown_compute()
        self.check(one, many, cp.asnumpy)

    @unittest.skipUnless(HAVE_GPU, "needs CuPy and a GPU")
    def test_fused_lensing_kernel_matches_the_array_code(self):
        # The fused kernel must reproduce the reference array path, ray for ray,
        # including supersampling and rays that fall into the hole.
        self.SETTINGS = {**self.SETTINGS, "supersample": 2}
        with tempfile.TemporaryDirectory() as t:
            ctx = context(t, "black_hole", backend="gpu", method="schwarzschild")
            fused = self.frame(ctx, cp)
            self.SETTINGS = {**self.SETTINGS, "array_lensing": True}
            array = self.frame(ctx, cp)
            ctx.shutdown_compute()
        self.assertGreater(array[2], 0.01)                     # some rays were captured
        np.testing.assert_allclose(cp.asnumpy(fused[0]), cp.asnumpy(array[0]), rtol=1e-9, atol=1e-12)
        np.testing.assert_allclose(cp.asnumpy(fused[1]), cp.asnumpy(array[1]), rtol=1e-9, atol=1e-12)
        self.assertAlmostEqual(fused[2], array[2], places=12)


class FluidSplitTests(unittest.TestCase):
    @unittest.skipUnless(HAVE_GPU, "needs CuPy and a GPU")
    def test_strips_with_halo_exchange_match_the_whole_lattice(self):
        from leonardo_demos.demos.fluid import FluidDemo
        nx, ny = 300, 157                       # rows do not divide evenly into three strips
        fields = []
        for devices in (None, [0, 0, 0]):
            with tempfile.TemporaryDirectory() as t:
                ctx = context(t, "fluid", backend="gpu")
                if devices:
                    ctx._devices = Devices(cp, devices)
                demo = FluidDemo(ctx, {"nx": nx, "ny": ny})
                f, c, w, mask = demo.init(nx, ny, .06, 1, None)
                for _ in range(5):
                    f, ux, uy, vort, rho = demo.step(f, c, w, mask, .06, 5)
                fields.append([cp.asnumpy(a) for a in (ux, uy, vort, rho)])
                ctx.shutdown_compute()
        for one, split_ in zip(*fields):
            np.testing.assert_array_equal(one, split_)


class RacersSplitTests(unittest.TestCase):
    def race(self, ctx, xp=np):
        demo = NeuroRacersDemo(ctx, {})
        demo.catalogue_data = demo.catalogue()
        demo.brain = demo.brain_spec(demo.catalogue_data)
        demo.track = track_geometry(0)
        sim = RaceSim(xp, demo.track, demo.brain, demo.catalogue_data)
        population = Population(xp, 90, demo.brain["layer_sizes"], seed=5)
        return demo.simulate(sim, population, 240, 3)

    def check(self, one, many, to_host=np.asarray):
        for key in ("fitness", "progress", "lap_step", "crash_step", "history", "inputs", "outputs"):
            np.testing.assert_allclose(to_host(one[key]), to_host(many[key]), rtol=1e-5, atol=1e-5, err_msg=key)

    def test_population_split_on_devices_and_cores_matches_one_race(self):
        with tempfile.TemporaryDirectory() as t:
            ctx = context(t, "neuro_racers")
            ctx.cpu_workers = 1
            one = self.race(ctx)
            ctx._devices = Devices(np, [0, 0, 0])
            split = self.race(ctx)
            ctx._devices = None
            ctx.cpu_workers = 4
            cores = self.race(ctx)
            ctx.shutdown_compute()
        self.check(one, split)
        self.check(one, cores)

    @unittest.skipUnless(HAVE_GPU, "needs CuPy and a GPU")
    def test_population_split_on_gpus_matches_one_gpu(self):
        with tempfile.TemporaryDirectory() as t:
            ctx = context(t, "neuro_racers", backend="gpu")
            one = self.race(ctx, cp)
            ctx._devices = Devices(cp, [0, 0])
            many = self.race(ctx, cp)
            ctx.shutdown_compute()
        self.check(one, many, cp.asnumpy)


if __name__ == "__main__":
    unittest.main()
