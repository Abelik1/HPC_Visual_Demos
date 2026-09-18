import tempfile
import unittest
from pathlib import Path

from fastapi import HTTPException

from app import RUNS, RemoteRunReq, RunReq, prepare_run, remote_plan
from leonardo_demos import lineups, remote
from leonardo_demos.registry import DEMOS


class LineupTests(unittest.TestCase):
    def test_shipped_lineups_are_valid(self):
        data = lineups.load()
        clean = lineups.validate(data, set(DEMOS))
        self.assertEqual(set(clean["machines"]), {"discoverer", "leonardo"})
        self.assertIn("molecular_dynamics", clean["machines"]["discoverer"]["demos"])
        self.assertIn("raytracer", clean["machines"]["leonardo"]["demos"])
        self.assertIn("neural_wall", clean["archived"])

    def test_unknown_demos_are_dropped_and_bad_values_rejected(self):
        clean = lineups.validate({"active": "a", "machines": {"a": {"demos": ["fluid", "nope"]}}}, set(DEMOS))
        self.assertEqual(clean["machines"]["a"]["demos"], ["fluid"])
        with self.assertRaises(ValueError):
            lineups.validate({"active": "missing", "machines": {"a": {"demos": []}}}, set(DEMOS))
        with self.assertRaises(ValueError):
            lineups.validate({"machines": {"a": {}}, "extras": {"x": {"folder": "../etc"}}}, set(DEMOS))
        with self.assertRaises(ValueError):
            lineups.validate({"machines": {"a": {}}, "extras": {"x": {"url": "javascript:alert(1)"}}}, set(DEMOS))


class RemoteRunTests(unittest.TestCase):
    def test_dry_validation_writes_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            rd = Path(tmp) / "run"
            png = ("data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk"
                   "YPhfDwAChwGA60e6kgAAAABJRU5ErkJggg==")
            kwargs = prepare_run("neural_wall", RunReq(target_image=png), rd, remote=True, dry=True)
            self.assertFalse(rd.exists())
            self.assertIn("_target_path", kwargs["params"])

    def test_chain_is_validated(self):
        kwargs = prepare_run("molecular_dynamics", RunReq(chain="hhpp+-pp"), RUNS / "unused", dry=True)
        self.assertEqual(kwargs["params"]["_chain"], "HHPP+-PP")
        for bad in ("HPX", "HP"):
            with self.assertRaises(HTTPException):
                prepare_run("molecular_dynamics", RunReq(chain=bad), RUNS / "unused", dry=True)
        with self.assertRaises(HTTPException):
            prepare_run("fluid", RunReq(chain="HHPPHH"), RUNS / "unused", dry=True)

    def test_ghost_races_are_local_only(self):
        with self.assertRaises(HTTPException) as caught:
            prepare_run("neuro_racers", RunReq(ghosts=["x"]), RUNS / "unused", remote=True, dry=True)
        self.assertEqual(caught.exception.status_code, 422)

    def test_plan_lists_changed_settings_and_warnings(self):
        body = RemoteRunReq(cluster="leonardo", request=RunReq(profile="local", params={"temperature": 400},
                                                               settings={"particles": 64}))
        plan = remote_plan("molecular_dynamics", body)
        temperature = next(p for p in plan["params"] if p["key"] == "temperature")
        self.assertEqual(temperature["value"], 400)
        particles = next(s for s in plan["settings"] if s["key"] == "particles")
        self.assertTrue(particles["changed"])
        self.assertTrue(any("preset" in w for w in plan["warnings"]))

    def test_job_script_uses_cluster_shape_and_python_override(self):
        c = remote.cluster("discoverer")
        script = remote.job_script(c, "/runs/r1", "fusion_plasma", "guardian", "01:00:00")
        self.assertIn("#SBATCH --account=ehpc-school-2026", script)
        self.assertIn("#SBATCH --gres=gpu:4", script)
        self.assertIn("--gres=gpu:1", script.splitlines()[-2])
        self.assertIn("visual-demos-torch", script)
        self.assertIn('tools/run_job.py "/runs/r1/job.json"', script)

    def test_walltime_and_injection_are_rejected(self):
        with self.assertRaises(ValueError):
            remote.save_overrides("leonardo", {"walltime": "forever"})
        with self.assertRaises(ValueError):
            remote.save_overrides("leonardo", {"account": "x; rm -rf ~"})


if __name__ == "__main__":
    unittest.main()
