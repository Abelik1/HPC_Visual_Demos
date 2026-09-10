import json, sys, tempfile, unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from leonardo_demos.base import RunContext
from leonardo_demos.backend import choose_backend

class TestContract(unittest.TestCase):
    def test_numpy_backend(self):
        xp,name=choose_backend('numpy'); self.assertEqual(name,'numpy'); self.assertTrue(hasattr(xp,'zeros'))
    def test_explicit_gpu_does_not_silently_accept_zero_devices(self):
        cupy=SimpleNamespace(cuda=SimpleNamespace(
            runtime=SimpleNamespace(getDeviceCount=lambda:0)))
        with patch.dict(sys.modules,{"cupy":cupy}):
            with self.assertRaisesRegex(RuntimeError,"no visible CUDA devices"):
                choose_backend('gpu')
    def test_context_meta(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'x','local',3,{},'numpy'); self.assertTrue((Path(t)/'meta.json').exists()); m=json.loads((Path(t)/'meta.json').read_text()); self.assertEqual(m['status'],'starting')
            self.assertIn('host',m['resources']); self.assertIn('architecture',m['resources'])
            self.assertIn('cpu_workers',m['resources']); self.assertIn('slurm',m['resources'])
            c.finish()
            m=json.loads((Path(t)/'meta.json').read_text())
            self.assertIn('duration_seconds',m); self.assertGreaterEqual(m['duration_seconds'],0)
    def test_frame_overlay_is_stored_separately_from_the_image(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'x','local',3,{},'numpy')
            c.write_status(1,'working',{'temperature':310,'regime':'stable'})
            data=json.loads((Path(t)/'frame_data/frame_0001.json').read_text())
            self.assertEqual(data['values']['temperature'],'310')
            self.assertEqual(data['values']['regime'],'stable')
            self.assertTrue(json.loads((Path(t)/'meta.json').read_text())['frame_data'])
if __name__=='__main__': unittest.main()
