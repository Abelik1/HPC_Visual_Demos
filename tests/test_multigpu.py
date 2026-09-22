import os
import tempfile
import unittest
from pathlib import Path

import numpy as np

from leonardo_demos.base import RunContext
from leonardo_demos.multigpu import Devices, requested_gpus, split


class SplitTests(unittest.TestCase):
    def test_slices_cover_the_range_in_balanced_contiguous_pieces(self):
        for length in (1, 7, 8, 1000, 1001):
            for parts in (1, 2, 3, 4):
                pieces = split(length, parts)
                self.assertEqual(pieces[0].start, 0)
                self.assertEqual(pieces[-1].stop, length)
                sizes = [s.stop - s.start for s in pieces]
                self.assertLessEqual(max(sizes) - min(sizes), 1)
                for a, b in zip(pieces, pieces[1:]):
                    self.assertEqual(a.stop, b.start)

    def test_never_more_parts_than_items(self):
        self.assertEqual(len(split(3, 4)), 3)


class DevicesTests(unittest.TestCase):
    # NumPy with repeated ids: the decomposition logic without any GPU.
    def setUp(self):
        self.dev = Devices(np, [0, 0, 0])

    def test_run_calls_every_part_and_keeps_order(self):
        self.assertEqual(self.dev.run(lambda i: i * 10), [0, 10, 20])

    def test_scatter_then_gather_round_trips(self):
        a = np.arange(40, dtype=np.float32).reshape(10, 4)
        parts = self.dev.scatter(a)
        self.assertEqual(sum(p.shape[0] for p in parts), 10)
        np.testing.assert_array_equal(self.dev.gather(parts), a)
        for whole in self.dev.allgather(parts):
            np.testing.assert_array_equal(whole, a)

    def test_periodic_halo_exchange_matches_a_whole_domain_roll(self):
        field = np.arange(12 * 5, dtype=np.float64).reshape(12, 5)
        pieces = split(12, 3)
        strips = [np.zeros((s.stop - s.start + 2, 5)) for s in pieces]
        for strip, s in zip(strips, pieces):
            strip[1:-1] = field[s]
        self.dev.halo_exchange(strips)
        down, up = np.roll(field, 1, 0), np.roll(field, -1, 0)
        for strip, s in zip(strips, pieces):
            np.testing.assert_array_equal(strip[0], down[s][0])
            np.testing.assert_array_equal(strip[-1], up[s][-1])

    def test_open_boundaries_leave_the_outer_ghosts_alone(self):
        strips = [np.full((4, 2), float(k)) for k in range(2)]
        for s in strips:
            s[0] = s[-1] = -1
        self.dev.halo_exchange(strips, periodic=False)
        self.assertTrue((strips[0][0] == -1).all())
        self.assertTrue((strips[1][-1] == -1).all())
        self.assertTrue((strips[0][-1] == 1).all())
        self.assertTrue((strips[1][0] == 0).all())


class RunContextGpuTests(unittest.TestCase):
    def test_gpu_count_comes_from_the_run_parameters(self):
        self.assertEqual(requested_gpus({}), 1)
        self.assertEqual(requested_gpus({"_gpus": 4}), 4)
        self.assertEqual(requested_gpus({"_gpus": "bad"}), 1)
        with tempfile.TemporaryDirectory() as t:
            ctx = RunContext(Path(t), "x", "local", 1, {"_gpus": 4}, "numpy")
            self.assertEqual(ctx.gpus, 4)
            # A CPU run has one host part, so demos keep their one-device path.
            self.assertEqual(ctx.devices.count, 1)

    def test_environment_override_for_benchmarks(self):
        old = os.environ.get("LEONARDO_DEMO_GPUS")
        os.environ["LEONARDO_DEMO_GPUS"] = "2"
        try:
            self.assertEqual(requested_gpus({}), 2)
        finally:
            if old is None:
                os.environ.pop("LEONARDO_DEMO_GPUS")
            else:
                os.environ["LEONARDO_DEMO_GPUS"] = old


if __name__ == "__main__":
    unittest.main()
