import tempfile, unittest, json
from pathlib import Path
from leonardo_demos.base import RunContext
from leonardo_demos.demos.reaction_diffusion import ReactionDiffusionDemo
from leonardo_demos.demos.black_hole import BlackHoleDemo
from leonardo_demos.demos.crystal import CrystalDemo
from leonardo_demos.demos.fusion_plasma import FusionPlasmaDemo
from leonardo_demos.demos.plasma_guardian import PlasmaGuardianDemo
from leonardo_demos.demos.weather_ensemble import WeatherEnsembleDemo
from leonardo_demos.demos.molecular_dynamics import MolecularDynamicsDemo
from leonardo_demos.demos.galaxy_collision_3d import GalaxyCollision3DDemo

class SmallDemoTests(unittest.TestCase):
    def test_reaction(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'reaction_diffusion','local',2,{'feed':.0367,'kill':.0649},'numpy'); ReactionDiffusionDemo(c,{'n':48,'total_steps':40,'sweep_steps':20,'ensemble':1}).run(); self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists()); self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
    def test_blackhole(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'black_hole','local',2,{'mass':1.2,'spin':.2,'lens_x':.3,'lens_y':-.2,'lens_count':2,'lens_separation':.45,'lens_angle':35},'numpy'); BlackHoleDemo(c,{'width':120,'height':68,'ensemble':1}).run()
            self.assertTrue((Path(t)/'reveal.jpg').exists())
            self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists())
            self.assertTrue((Path(t)/'modes/3d/frame_0001.jpg').exists())
            meta=json.loads((Path(t)/'meta.json').read_text())
            self.assertEqual(meta['default_view_mode'],'frames')
            self.assertEqual(meta['view_modes'][1]['folder'],'modes/3d')
            self.assertEqual(meta['source_plane']['name'],'Hubble Deep Field (PIA12110)')
            self.assertEqual(len(meta['lens_wells']),2)
            self.assertNotEqual((Path(t)/'frames/frame_0000.jpg').read_bytes(),(Path(t)/'frames/frame_0001.jpg').read_bytes())
    def test_crystal(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'crystal','local',2,{'undercooling':.75,'anisotropy':.055},'numpy'); CrystalDemo(c,{'depth':3,'ensemble':1,'zoom_levels':1,'zoom_depth':4,'zoom_tile':64}).run(); self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists()); self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
    def test_fusion_plasma(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'fusion_plasma','local',2,{'magnetic_field':5.0,'heating':25,'density':1.0},'numpy')
            demo=FusionPlasmaDemo(c,{'n':24,'total_steps':4,'ensemble':4,'sweep_n':20,'sweep_steps':3,'tracers':12,'trail':4})
            real,imag=demo.initialise(24,43)
            trails=demo.initialise_tracers(12,4)
            moved,_=demo.advance_tracers(trails,real,imag,5.0,25.0,8)
            self.assertFalse((moved[:,-1] == moved[:,0]).all())
            demo.run()
            self.assertTrue((Path(t)/'reveal.jpg').exists())
            manifest=json.loads((Path(t)/'fusion_view.json').read_text())
            self.assertEqual(len(manifest['texture']),manifest['shape'][0]*manifest['shape'][1])
            meta=json.loads((Path(t)/'meta.json').read_text())
            self.assertEqual(meta['status'],'complete')
            self.assertEqual(meta['fusion_view']['folder'],'modes/fusion3d')
            self.assertTrue((Path(t)/'modes/fusion3d/frame_0001.json').exists())
    def test_weather_ensemble(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'weather_ensemble','local',2,{'warming':1.5,'jet_stream':1.0,'uncertainty':25},'numpy')
            WeatherEnsembleDemo(c,{'n':24,'total_steps':4,'ensemble':1,'sweep_n':24,'sweep_steps':3}).run()
            self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists())
            self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
    def test_plasma_guardian(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'plasma_guardian','local',2,{'instability':1.0},'cpu','torch')
            PlasmaGuardianDemo(c,{'batch':16,'horizon':8,'train_updates':4,'display_steps':20}).run()
            self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists())
            self.assertTrue((Path(t)/'overlays/network/frame_0001.jpg').exists())
            meta=json.loads((Path(t)/'meta.json').read_text())
            self.assertEqual(meta['status'],'complete')
    def test_molecular_dynamics(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'molecular_dynamics','local',2,{'temperature':310,'attraction':1.0,'solvent':.65,'sequence':0},'numpy')
            MolecularDynamicsDemo(c,{'particles':18,'total_steps':4,'ensemble':1,'sweep_particles':14,'sweep_steps':3}).run()
            self.assertTrue((Path(t)/'reveal.jpg').exists())
            self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
    def test_self_gravitating_galaxy_3d(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'galaxy_collision_3d','local',2,
                         {'impact':.35,'speed':1.0,'disc_tilt':35,'softening':4},
                         'numpy',method='leapfrog')
            GalaxyCollision3DDemo(c,{'particles':96,'substeps':1,'span_gyr':.05,
                                     'force_tile':32}).run()
            self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists())
            frame=json.loads((Path(t)/'interactive/frame_0001.json').read_text())
            self.assertEqual(frame['kind'],'nbody-galaxy-3d')
            self.assertEqual(len(frame['positions']),96)
            self.assertTrue(all(len(point)==3 for point in frame['positions']))
            initial=json.loads((Path(t)/'interactive/frame_0000.json').read_text())
            self.assertEqual(initial['time_gyr'],0.0)
            meta=json.loads((Path(t)/'meta.json').read_text())
            self.assertEqual(meta['status'],'complete')
            self.assertEqual(meta['physics']['complexity'],'O(N^2)')
            self.assertIn('not a fitted equilibrium',meta['physics']['model_status'])
if __name__=='__main__': unittest.main()
