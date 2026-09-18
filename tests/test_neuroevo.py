import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from fastapi import HTTPException

from app import RunReq, start
from leonardo_demos.backend import to_numpy
from leonardo_demos.base import RunContext
from leonardo_demos.demos.neuro_racers import (HALF_WIDTH, TRACKS, WORLD_H, WORLD_W, NeuroRacersDemo,
                                               RaceSim, track_geometry)
from leonardo_demos.neuroevo import (BrainError, Population, brain_catalogue, input_labels, load_checkpoint, load_checkpoint,
                                     load_champion, parameter_count, save_champion, validate_brain)

SPECS = json.loads((Path(__file__).resolve().parents[1] / "config" / "demo_specs.json").read_text())
CATALOGUE = brain_catalogue(SPECS, "neuro_racers")
# 3 long rays (15) + compass (8) + two 16-neuron layers (18) = 41 > 40 points.
OVER_BUDGET = [{"block": "long_ray", "angle": a} for a in (-15, 0, 15)] + [{"block": "compass"}]


def preset(name):
    return {k: CATALOGUE["presets"][name][k] for k in ("sensors", "hidden", "actions")}


def evolve(brain, track, generations, size=48, steps=450, seed=7):
    sim = RaceSim(np, track_geometry(track), brain, CATALOGUE)
    pop = Population(np, size, brain["layer_sizes"], seed=seed)
    best, result = [], None
    for _ in range(generations):
        if result is not None:
            pop.evolve(result["fitness"])
        result = sim.run(pop, steps, 3)
        best.append(float(result["progress"].max()) / sim.length)
    return best


class BrainSpecTests(unittest.TestCase):
    def test_every_preset_is_valid_and_within_budget(self):
        for name in CATALOGUE["presets"]:
            brain = validate_brain(preset(name), CATALOGUE)
            self.assertLessEqual(brain["points"], CATALOGUE["budget"])
            self.assertEqual(brain["parameters"], parameter_count(brain["layer_sizes"]))

    def test_spec_is_canonical_and_derives_layers(self):
        brain = validate_brain({"sensors": [{"block": "speed"}, {"block": "ray", "angle": 45},
                                            {"block": "ray", "angle": -45}],
                                "hidden": [8], "actions": ["brake", "throttle", "steer"]}, CATALOGUE)
        self.assertEqual([s.get("angle") for s in brain["sensors"]], [-45, 45, None])
        self.assertEqual(brain["actions"], ["steer", "throttle", "brake"])
        self.assertEqual(brain["layer_sizes"], [3, 8, 3])
        self.assertEqual(input_labels(brain, CATALOGUE), ["ray L45°", "ray R45°", "speed"])

    def test_rejects_invalid_designs(self):
        bad = [
            {"sensors": OVER_BUDGET, "hidden": [16, 16], "actions": ["steer", "throttle"]},             # over budget
            {"sensors": [], "hidden": [], "actions": ["steer"]},                               # missing throttle
            {"sensors": [{"block": "ray", "angle": 7}], "hidden": [], "actions": ["steer", "throttle"]},
            {"sensors": [{"block": "ray", "angle": 0}, {"block": "ray", "angle": 0}], "hidden": [],
             "actions": ["steer", "throttle"]},                                                # duplicate
            {"sensors": [{"block": "laser"}], "hidden": [], "actions": ["steer", "throttle"]},  # unknown
            {"sensors": [], "hidden": [5], "actions": ["steer", "throttle"]},                  # bad width
            {"sensors": [], "hidden": [4, 4, 4, 4], "actions": ["steer", "throttle"]},         # too deep
        ]
        for spec in bad:
            with self.assertRaises(BrainError, msg=str(spec)):
                validate_brain(spec, CATALOGUE)

    def test_replay_endpoint_needs_saved_generations(self):
        from app import RUNS, replay
        with tempfile.TemporaryDirectory(dir=RUNS) as t:
            run = Path(t)
            (run / "meta.json").write_text(json.dumps({"demo": "neuro_racers"}))
            with self.assertRaises(HTTPException):
                replay(run.name, "1")                      # no checkpoints folder yet
            (run / "checkpoints").mkdir()
            with self.assertRaises(HTTPException):
                replay(run.name, "3")                      # generation not saved
            with self.assertRaises(HTTPException):
                replay(run.name, "1,2,3,4,5,6,7")          # too many at once
            (run / "checkpoints" / "gen_0001.npz").write_bytes(b"")
            with self.assertRaises(HTTPException):
                replay(run.name, "1", track=9)             # no such track
            with self.assertRaises(HTTPException):
                replay(run.name, "1", cave=5)              # not a Neuro-Racers setting
            with self.assertRaises(HTTPException):
                replay(run.name, "1", ghosts="../outside")

    def test_api_validates_brains_and_ghosts(self):
        over = {"sensors": OVER_BUDGET, "hidden": [16, 16], "actions": ["steer", "throttle"]}
        with self.assertRaises(HTTPException):
            start("neuro_racers", RunReq(brain=over))
        with self.assertRaises(HTTPException):
            start("fluid", RunReq(brain=preset("tiny")))
        with self.assertRaises(HTTPException):
            start("neuro_racers", RunReq(brain=preset("tiny"), ghosts=["../outside"]))
        with self.assertRaises(HTTPException):
            start("fluid", RunReq(name="Sam"))                  # name tags are for the AI games


class PopulationTests(unittest.TestCase):
    def test_forward_shape_and_range(self):
        pop = Population(np, 10, [4, 8, 3], seed=1)
        out = pop.forward(np.random.default_rng(0).normal(size=(10, 4)))
        self.assertEqual(out.shape, (10, 3))
        self.assertTrue(np.all(np.abs(out) <= 1))

    def test_elites_survive_and_islands_stay_independent(self):
        pop = Population(np, 8, [2, 3], seed=2, groups=2)
        before = pop.genome.copy()
        fitness = np.zeros(16); fitness[5] = 10; fitness[12] = 20
        pop.evolve(fitness)
        np.testing.assert_array_equal(pop.genome[0], before[5])   # best of island 0
        np.testing.assert_array_equal(pop.genome[8], before[12])  # best of island 1

    def test_champion_round_trip(self):
        brain = validate_brain(preset("balanced"), CATALOGUE)
        pop = Population(np, 4, brain["layer_sizes"], seed=3)
        with tempfile.TemporaryDirectory() as t:
            save_champion(Path(t) / "c.npz", pop.genome[2:3], brain, {"track": 1})
            genome, loaded, meta = load_champion(Path(t) / "c.npz")
        np.testing.assert_array_equal(genome, pop.genome[2:3])
        self.assertEqual(loaded["layer_sizes"], brain["layer_sizes"])
        self.assertEqual(meta["track"], 1)


class RacingBehaviourTests(unittest.TestCase):
    def test_tracks_never_touch_themselves_or_the_frame(self):
        for track_id in TRACKS:
            t = track_geometry(track_id)
            c, n = t["centre"], len(t["centre"])
            d = np.linalg.norm(c[:, None] - c[None], axis=-1)
            index = np.arange(n); sep = np.abs(index[:, None] - index[None]); sep = np.minimum(sep, n - sep) * t["length"] / n
            self.assertGreater(d[sep > 2.6].min(), 2 * HALF_WIDTH + .2, t["name"])
            walls = np.vstack([t["left"], t["right"]])
            self.assertTrue((walls.min(0) > 0).all() and walls[:, 0].max() < WORLD_W and walls[:, 1].max() < WORLD_H, t["name"])

    def test_evolution_improves_the_balanced_brain(self):
        best = evolve(validate_brain(preset("balanced"), CATALOGUE), 0, 6)
        self.assertGreater(best[-1], best[0] + .5)

    def test_a_blind_brain_cannot_learn_to_drive(self):
        blind = evolve(validate_brain(preset("blind"), CATALOGUE), 0, 6)
        seeing = evolve(validate_brain(preset("balanced"), CATALOGUE), 0, 6)
        self.assertLess(blind[-1], .6)
        self.assertGreater(seeing[-1], 2 * blind[-1])

    def test_fused_cuda_kernels_match_the_array_reference(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError
        except Exception:
            self.skipTest("CUDA/CuPy unavailable")
        brain = validate_brain(preset("big"), CATALOGUE)
        track = track_geometry(3)
        reference = Population(np, 96, brain["layer_sizes"], seed=5)
        cpu = RaceSim(np, track, brain, CATALOGUE).run(reference, 300, 3)
        gpu_pop = Population(cp, 96, brain["layer_sizes"], seed=5)
        gpu_pop.genome = cp.asarray(reference.genome)
        gpu = RaceSim(cp, track, brain, CATALOGUE).run(gpu_pop, 300, 3)
        np.testing.assert_array_equal(to_numpy(gpu["crash_step"]) >= 0, cpu["crash_step"] >= 0)
        self.assertLess(np.median(np.abs(to_numpy(gpu["progress"]) - cpu["progress"])), 1e-3)

    def test_run_writes_the_full_contract(self):
        with tempfile.TemporaryDirectory() as t:
            ctx = RunContext(Path(t), "neuro_racers", "local", 3, {"track": 0, "mutation": .5, "seed": 7,
                                                                   "_brain": validate_brain(preset("tiny"), CATALOGUE)}, "cpu")
            NeuroRacersDemo(ctx, {"population": 12, "generations": 2, "sim_steps": 60, "record_cars": 6,
                                  "ensemble": 4, "reveal_population": 6, "reveal_generations": 1}).run()
            root = Path(t)
            for path in ("frames/frame_0002.jpg", "overlays/network/frame_0002.jpg", "frame_data/frame_0002.json",
                         "interactive/arena.json", "interactive/gen_0002.json", "champion.npz", "reveal.jpg"):
                self.assertTrue((root / path).exists(), path)
            meta = json.loads((root / "meta.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["arena_view"]["kind"], "racers")
            self.assertEqual(len(meta["arena_view"]["frames"]), 3)
            self.assertEqual(meta["summary"]["track"], "Oval")
            self.assertEqual(len(meta["reveal_tiles"]), 4)
            # One-per-box is also animation data, one champion drive per search.
            grid = json.loads((root / meta["reveal_view"]["file"]).read_text())
            self.assertEqual(len(grid["boxes"]), 4)
            car = grid["boxes"][0]["gen"]["cars"][0]
            self.assertEqual(len(car["x"]), len(car["y"]))
            self.assertGreater(len(car["x"]), 1)
            gen = json.loads((root / "interactive/gen_0002.json").read_text())
            self.assertEqual(len(gen["cars"]), 6)
            # Live brain for the champion: one activation row per recorded sample.
            self.assertEqual(len(gen["brains"][0]["acts"][0]), len(gen["cars"][0]["x"]))
            self.assertTrue((root / "checkpoints/gen_0002.npz").exists())
            self.assertTrue(meta["lab"]["compare"])
            one = NeuroRacersDemo.replay(root, meta, [1, 2], 5)
            again = NeuroRacersDemo.replay(root, meta, [1, 2], 5)
            other = NeuroRacersDemo.replay(root, meta, [1, 2], 6)
            self.assertEqual([c["label"] for c in one["cars"]], ["gen 1", "gen 2"])
            self.assertEqual(one["cars"][0]["x"][0], one["cars"][1]["x"][0])     # same start for every generation
            self.assertEqual(one["cars"][0]["x"], again["cars"][0]["x"])          # a seed replays exactly
            self.assertNotEqual(one["cars"][0]["x"][0], other["cars"][0]["x"][0])  # a new seed is a new start
            self.assertEqual(len(one["brains"]), 2)
            # The test world: the same frozen networks on another track, and
            # racing another visitor's saved champion (with its own name tag).
            moved = NeuroRacersDemo.replay(root, meta, [2], 5, {"track": 1})
            self.assertEqual((moved["replay"]["track"], moved["replay"]["trained_track"]), (1, 0))
            self.assertEqual(moved["arena"]["track"], "Kidney")
            (root / "rival").mkdir()
            save_champion(root / "rival" / "champion.npz", load_checkpoint(root / "checkpoints/gen_0002.npz")["champion"],
                          meta["brain"], {"track": 0, "name": "Sam"})
            race = NeuroRacersDemo.replay(root, meta, [2], 5, {"ghosts": [str(root / "rival")]})
            self.assertEqual([(c["label"], bool(c.get("ghost"))) for c in race["cars"]], [("gen 2", False), ("Sam", True)])
            self.assertEqual(race["cars"][0]["x"], race["cars"][1]["x"])          # same network, same start, same drive
            self.assertEqual(len(race["brains"]), 1)                              # brains line up with own cars only


# ---- Bat vs Moth -------------------------------------------------------
import leonardo_demos.demos.bat_vs_moth as bm
from leonardo_demos.neuroevo import validate_brains

BAT_MOTH = brain_catalogue(SPECS, "bat_vs_moth")
EARS = {"sensors": [{"block": "ear_left"}, {"block": "ear_right"}, {"block": "whiskers"}],
        "hidden": [], "actions": ["steer", "speed", "chirp"]}


class HandBat:
    """Turn toward the louder ear: a fixed, known-good hunter for pinning the sensing model."""

    def __init__(self, size, ears=2):
        self.size, self.ears = size, ears

    def forward(self, x, genome=None, return_hidden=False):
        left, right = x[:, 0], x[:, 2]
        if self.ears == 1:
            right = np.zeros_like(right)            # one ear: no left/right comparison
        turn = np.clip(3 * (right - left) / (np.maximum(left, right) + .05), -1, 1)
        turn = turn + np.where(x[:, 5] < .6, np.where(x[:, 4] < x[:, 6], 1, -1), 0)
        return np.stack([np.clip(turn, -1, 1), np.full(len(x), .6), np.ones(len(x))], 1).astype(np.float32)


class HandMoth:
    def __init__(self, size, jam):
        self.size, self.jam, self.genome = size, jam, np.zeros((size, 1), np.float32)

    def forward(self, x, genome=None, return_hidden=False):
        heard = x[:, 0]
        jam = np.where(heard > .55, 1.0, -1.0) if self.jam else np.full(len(x), -1.0)
        return np.stack([np.zeros(len(x)), np.where(heard > .3, 1.0, -.3), np.full(len(x), -1.0), jam], 1).astype(np.float32)


def hunt_rate(bat, moth, size=60, hunts=4):
    brains = validate_brains({"bat": EARS, "moth": {"sensors": [{"block": "tympanum"}, {"block": "direction"}],
                                                    "hidden": [], "actions": ["steer", "speed", "dive", "jam"]}}, BAT_MOTH)
    sim = bm.CaveSim(np, bm.cave_geometry(11), brains["bat"], brains["moth"], 4)
    rates = []
    for hunt in range(hunts):
        index = bm.assign_moths(np.random.default_rng(hunt), 1, size, size, 4)
        rates.append((sim.run(bat, moth, index, 360, hunt)["catch_step"] >= 0).mean())
    return float(np.mean(rates))


class BatVsMothTests(unittest.TestCase):
    def test_roles_default_and_memory_width(self):
        brains = validate_brains({"bat": {**EARS, "sensors": EARS["sensors"] + [{"block": "memory"}]}}, BAT_MOTH)
        self.assertEqual(brains["bat"]["inputs"], 7 + 3)          # memory feeds back one value per action
        self.assertEqual(brains["moth"]["actions"], BAT_MOTH["roles"]["moth"]["presets"]["full_kit"]["actions"])
        with self.assertRaises(BrainError):
            validate_brains({"bird": EARS}, BAT_MOTH)
        for role, catalogue in BAT_MOTH["roles"].items():
            for name, preset_spec in catalogue["presets"].items():
                validate_brain({k: preset_spec[k] for k in ("sensors", "hidden", "actions")}, catalogue)

    def test_two_ears_hunt_better_than_one(self):
        size = 60
        two = hunt_rate(HandBat(size, ears=2), HandMoth(size, jam=False), size)
        one = hunt_rate(HandBat(size, ears=1), HandMoth(size, jam=False), size)
        self.assertGreater(two, one + .15)

    def test_jamming_protects_moths(self):
        size = 60
        quiet = hunt_rate(HandBat(size), HandMoth(size, jam=False), size)
        jamming = hunt_rate(HandBat(size), HandMoth(size, jam=True), size)
        self.assertLess(jamming, quiet - .15)

    def test_quiet_start_silences_jam_and_dive(self):
        brains = validate_brains({}, BAT_MOTH)
        moths = bm.quiet_start(Population(np, 20, brains["moth"]["layer_sizes"], seed=1), brains["moth"])
        out = moths.forward(np.zeros((20, brains["moth"]["layer_sizes"][0]), np.float32))
        jam = brains["moth"]["actions"].index("jam")
        self.assertTrue((out[:, jam] < bm.TRIGGER).all())

    def test_fused_cuda_kernels_match_the_array_reference(self):
        try:
            import cupy as cp
            if cp.cuda.runtime.getDeviceCount() < 1:
                raise RuntimeError
        except Exception:
            self.skipTest("CUDA/CuPy unavailable")
        # Every sensor block, hidden layers, dive and jam between the two pairs.
        for bat, moth in (("two_ears", "full_kit"), ("deluxe", "dodger")):
            roles = BAT_MOTH["roles"]
            brains = validate_brains({role: {k: roles[role]["presets"][name][k] for k in ("sensors", "hidden", "actions")}
                                      for role, name in (("bat", bat), ("moth", moth))}, BAT_MOTH)
            size, k, cave = 256, 4, bm.cave_geometry(11)
            bats = Population(np, size, brains["bat"]["layer_sizes"], seed=1, **bm.EVOLUTION)
            moths = Population(np, size, brains["moth"]["layer_sizes"], seed=2, **bm.EVOLUTION)
            moths.genome[:, -len(brains["moth"]["actions"]):] += .6          # jam and dive fire now and then
            index = bm.assign_moths(np.random.default_rng(0), 1, size, size, k)
            cpu = bm.CaveSim(np, cave, brains["bat"], brains["moth"], k).run(bats, moths, index, 300, 5)
            gpu_bats = Population(cp, size, brains["bat"]["layer_sizes"], seed=1); gpu_bats.genome = cp.asarray(bats.genome)
            gpu_moths = Population(cp, size, brains["moth"]["layer_sizes"], seed=2); gpu_moths.genome = cp.asarray(moths.genome)
            sim = bm.CaveSim(cp, cave, brains["bat"], brains["moth"], k)
            self.assertTrue(sim.fused)
            gpu = sim.run(gpu_bats, gpu_moths, index, 300, 5)
            # Float rounding differs (FMA-free kernels vs NumPy's libm), so a
            # few caves may drift past a threshold late in the hunt; the rest
            # must match step for step.
            caught_cpu, caught_gpu = cpu["catch_step"], to_numpy(gpu["catch_step"])
            self.assertGreater(caught_cpu.size and (caught_cpu >= 0).sum(), 20, bat)
            self.assertLessEqual(abs(int((caught_gpu >= 0).sum()) - int((caught_cpu >= 0).sum())), 3, bat)
            self.assertGreater(np.mean(caught_gpu == caught_cpu), .97, bat)
            for key in ("bat_fitness", "slot_fitness"):
                diff = np.abs(to_numpy(gpu[key]) - cpu[key])
                self.assertGreater(np.mean(diff < 1e-4), .95, f"{bat} {key}")
            self.assertEqual(to_numpy(gpu["record"]["bat"]).shape, cpu["record"]["bat"].shape)
            self.assertEqual(to_numpy(gpu["record"]["bat_in"]).dtype, cpu["record"]["bat_in"].dtype)
            np.testing.assert_allclose(to_numpy(gpu["record"]["bat"][:10]), cpu["record"]["bat"][:10], atol=1e-4)
            self.assertGreater(np.mean(to_numpy(gpu["chirp_log"]) == cpu["chirp_log"]), .99, bat)

    def test_run_writes_the_full_contract(self):
        with tempfile.TemporaryDirectory() as t:
            ctx = RunContext(Path(t), "bat_vs_moth", "local", 3, {"cave": 11, "moths": 3, "mutation": 1.0,
                                                                  "_brain": validate_brains({}, BAT_MOTH)}, "cpu")
            bm.BatVsMothDemo(ctx, {"population": 12, "generations": 3, "hunt_steps": 60, "moth_head_start": 1,
                                   "ensemble": 4, "reveal_population": 6, "reveal_generations": 1}).run()
            root = Path(t)
            for path in ("frames/frame_0002.jpg", "modes/lit/frame_0002.jpg", "overlays/network/frame_0002.jpg",
                         "overlays/arms_race/frame_0002.jpg", "frame_data/frame_0002.json",
                         "interactive/arena.json", "interactive/gen_0003.json", "reveal.jpg"):
                self.assertTrue((root / path).exists(), path)
            meta = json.loads((root / "meta.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["arena_view"]["kind"], "batmoth")
            self.assertEqual(meta["view_modes"][1]["folder"], "modes/lit")
            self.assertEqual(len(meta["reveal_tiles"]), 4)
            # Every box is its own cave, so it carries its own rocks.
            grid = json.loads((root / meta["reveal_view"]["file"]).read_text())
            self.assertEqual(len(grid["boxes"]), 4)
            self.assertNotEqual(grid["boxes"][0]["arena"]["rocks"], grid["boxes"][1]["arena"]["rocks"])
            self.assertEqual(len(grid["boxes"][0]["gen"]["moths"]), 3)
            gen = json.loads((root / "interactive/gen_0003.json").read_text())
            self.assertEqual(len(gen["moths"]), 3)
            self.assertEqual(len(gen["others"]), 11)
            # One per box: the best caves' full hunts, champion's cave first.
            self.assertEqual(meta["arena_view"]["lanes"], 9)
            self.assertEqual(len(gen["lanes"]), 8)
            self.assertEqual(len(gen["lanes"][0]["moths"]), 3)
            self.assertEqual(len(gen["lanes"][0]["cars"][0]["x"]), len(gen["cars"][0]["x"]))
            self.assertEqual(len(gen["brains"][0]["acts"][0]), len(gen["cars"][0]["x"]))
            saved = load_checkpoint(root / "checkpoints/gen_0003.npz")
            self.assertEqual(saved["moths"].shape[0], 3)
            replay = bm.BatVsMothDemo.replay(root, meta, [2], 9)
            self.assertEqual(replay["replay"], {"gens": [2], "seed": 9, "cave": 11, "moths": 3,
                                                "trained_cave": 11, "trained_moths": 3})
            # A test world: another cave layout and more moths, same networks.
            moved = bm.BatVsMothDemo.replay(root, meta, [2], 9, {"cave": 12, "moths": 5})
            self.assertEqual(len(moved["moths"]), 5)
            self.assertNotEqual(moved["arena"]["rocks"], replay["arena"]["rocks"])
            self.assertEqual([b["title"] for b in replay["brains"]],
                             ["Generation 2 champion bat", "Generation 2 leading moth"])
            self.assertEqual(len(replay["brains"][1]["acts"][0]), len(replay["cars"][0]["x"]))


if __name__ == "__main__":
    unittest.main()
