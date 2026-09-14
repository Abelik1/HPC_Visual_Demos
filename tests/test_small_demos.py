import tempfile, unittest, json
from pathlib import Path
from leonardo_demos.base import RunContext
from leonardo_demos.demos.reaction_diffusion import ReactionDiffusionDemo
from leonardo_demos.demos.black_hole import BlackHoleDemo
from leonardo_demos.demos.crystal import CrystalDemo
from leonardo_demos.demos.fusion_plasma import FusionPlasmaDemo
from leonardo_demos.demos.weather_ensemble import WeatherEnsembleDemo
from leonardo_demos.demos.molecular_dynamics import MolecularDynamicsDemo
from leonardo_demos.demos.galaxy_collision_3d import GalaxyCollision3DDemo
from leonardo_demos.demos.fluid import FluidDemo
from leonardo_demos.demos.cosmic_web import CosmicWebDemo

class SmallDemoTests(unittest.TestCase):
    def test_reaction(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'reaction_diffusion','local',2,{'feed':.0367,'kill':.0649},'numpy'); ReactionDiffusionDemo(c,{'n':48,'total_steps':40,'sweep_steps':20,'ensemble':1}).run(); self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists()); self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
    def test_blackhole(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'black_hole','local',2,{'mass':1.2,'spin':.2,'lens_x':.3,'lens_y':-.2,'lens_count':2,'lens_separation':.45,'lens_angle':35},'numpy',method='weak_field'); BlackHoleDemo(c,{'width':120,'height':68,'ensemble':1}).run()
            self.assertTrue((Path(t)/'reveal.jpg').exists())
            self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists())
            self.assertTrue((Path(t)/'modes/3d/frame_0001.jpg').exists())
            meta=json.loads((Path(t)/'meta.json').read_text())
            self.assertEqual(meta['default_view_mode'],'frames')
            self.assertEqual(meta['view_modes'][1]['folder'],'modes/3d')
            self.assertEqual(meta['source_plane']['name'],'Hubble Deep Field (PIA12110)')
            self.assertEqual(len(meta['lens_wells']),2)
            self.assertNotEqual((Path(t)/'frames/frame_0000.jpg').read_bytes(),(Path(t)/'frames/frame_0001.jpg').read_bytes())
    def test_wind_tunnel_custom_shape_only_runs_to_completion(self):
        # Every obstacle choice must survive a whole run, readouts included:
        # the "custom shape only" option once built its mask and then crashed
        # on the first frame's readout.
        grid=[[0]*24 for _ in range(12)]
        for row in range(3,9): grid[row][10]=1
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'fluid','local',2,{'speed':.06,'obstacle':3,'_obstacle_grid':grid},'numpy')
            FluidDemo(c,{'nx':64,'ny':36,'total_steps':12,'tracers':40,'trail':4}).run()
            meta=json.loads((Path(t)/'meta.json').read_text())
            self.assertEqual(meta['status'],'complete')
            self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists())
    def test_cosmic_recipes_change_how_much_structure_grows(self):
        # Same seed, same steps: only the recipe differs. Our universe must
        # grow far more structure than a radiation-dominated one or one without
        # dark matter - that contrast is the whole lesson of the control.
        # Growth is measured between the first and last frame: in a box this
        # small the absolute contrast is dominated by particle shot noise that
        # is already there at the start.
        growth={}
        for recipe in (0,1,2):
            with tempfile.TemporaryDirectory() as t:
                c=RunContext(Path(t),'cosmic_web','local',3,{'recipe':recipe,'seed':42,'gravity':.8,'helium':.24,
                                                             'expanding_space':1,'dark_energy':1,'warm_dark_matter':0},'numpy')
                CosmicWebDemo(c,{'grid':48,'particles':4000,'total_steps':260,'sweep_steps':10,'ensemble':1}).run()
                meta=json.loads((Path(t)/'meta.json').read_text())
                self.assertEqual(meta['status'],'complete')
                self.assertEqual(meta['recipe']['id'],recipe)
                read=lambda i:float(json.loads((Path(t)/f'frame_data/frame_{i:04d}.json').read_text())['values']['clumping density contrast'])
                growth[recipe]=read(2)-read(0)
        self.assertGreater(growth[0],2*growth[1])
        self.assertGreater(growth[0],2*growth[2])
    def test_crystal(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'crystal','local',2,{'undercooling':.75,'anisotropy':.055},'numpy'); CrystalDemo(c,{'depth':3,'ensemble':1,'zoom_levels':1,'zoom_depth':4,'zoom_tile':64}).run(); self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists()); self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
    def test_fusion_plasma_passive(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'fusion_plasma','local',2,{'magnetic_field':5.0,'heating':25,'density':1.0},'numpy',method='passive')
            demo=FusionPlasmaDemo(c,{'n':24,'total_steps':4,'ensemble':4,'sweep_n':20,'sweep_steps':3,'tracers':12,'trail':4})
            real,imag=demo.initialise(24,43)
            trails=demo.initialise_tracers(12,4)
            moved,_=demo.advance_tracers(trails,real,imag,5.0,25.0,8)
            self.assertFalse((moved[:,-1] == moved[:,0]).all())
            demo.run()
            self.assertTrue((Path(t)/'reveal.jpg').exists())
            manifest=json.loads((Path(t)/'fusion_view.json').read_text())
            self.assertEqual(manifest['mode'],'passive')
            self.assertEqual(len(manifest['texture']),manifest['shape'][0]*manifest['shape'][1])
            meta=json.loads((Path(t)/'meta.json').read_text())
            self.assertEqual(meta['status'],'complete')
            self.assertEqual(meta['fusion_view']['folder'],'modes/fusion3d')
            self.assertTrue((Path(t)/'modes/fusion3d/frame_0001.json').exists())
    def test_fusion_plasma_guardian(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'fusion_plasma','local',2,
                         {'magnetic_field':5.0,'heating':25,'density':1.0,'instability':1.0},
                         'cpu',method='guardian')
            FusionPlasmaDemo(c,{'n':24,'total_steps':8,'ensemble':2,'sweep_n':20,'sweep_steps':3,
                                'tracers':12,'trail':6,'particles':40,'batch':16,'horizon':6,
                                'train_updates':4,'display_steps':6,'shots':2}).run()
            self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists())
            self.assertTrue((Path(t)/'overlays/network/frame_0001.jpg').exists())
            self.assertTrue((Path(t)/'overlays/poloidal/frame_0001.jpg').exists())
            self.assertTrue((Path(t)/'overlays/shots/frame_0001.jpg').exists())
            self.assertTrue((Path(t)/'reveal.jpg').exists())
            manifest=json.loads((Path(t)/'modes/fusion3d/frame_0001.json').read_text())
            self.assertEqual(manifest['mode'],'guardian')
            self.assertEqual(len(manifest['particles']),40)
            self.assertTrue(all(len(p)==3 for p in manifest['particles']))
            self.assertEqual(len(manifest['clearance']),40)
            self.assertEqual(len(manifest['commands']),3)
            meta=json.loads((Path(t)/'meta.json').read_text())
            self.assertEqual(meta['status'],'complete')
            self.assertEqual(meta['overlays'],['network','poloidal','shots'])
            self.assertEqual(meta['fusion_view']['folder'],'modes/fusion3d')
            # Training is a phase between shots, so every shot must be scored.
            self.assertEqual(len(meta['shot_history']),meta['shots'])
            self.assertTrue(all('lost' in row and 'baseline' in row for row in meta['shot_history']))
    def test_weather_ensemble(self):
        with tempfile.TemporaryDirectory() as t:
            c=RunContext(Path(t),'weather_ensemble','local',2,{'warming':1.5,'jet_stream':1.0,'uncertainty':25},'numpy')
            WeatherEnsembleDemo(c,{'n':24,'total_steps':4,'ensemble':1,'sweep_n':24,'sweep_steps':3}).run()
            self.assertTrue((Path(t)/'frames/frame_0001.jpg').exists())
            self.assertEqual(json.loads((Path(t)/'meta.json').read_text())['status'],'complete')
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
