import base64
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from fastapi import HTTPException

from app import RunReq, save_target_image, specs, start
from leonardo_demos.base import RunContext
from leonardo_demos.demos.neural_wall import NeuralWallDemo
from run_demo import run


class InputValidationTests(unittest.TestCase):
    def test_run_rejects_nonpositive_frame_count(self):
        with self.assertRaisesRegex(ValueError, 'frames must be at least 1'):
            run('reaction_diffusion', frames=0)

    def test_api_rejects_unknown_profile(self):
        with self.assertRaises(HTTPException):
            start('reaction_diffusion', RunReq(profile='invalid'))

    def test_api_rejects_gpu_for_cpu_only_demo(self):
        with self.assertRaises(HTTPException):
            start('crystal',RunReq(backend='gpu'))

    def test_api_rejects_unknown_and_out_of_range_parameters(self):
        with self.assertRaises(HTTPException):
            start('reaction_diffusion', RunReq(params={'unexpected': 1}))
        with self.assertRaises(HTTPException):
            start('reaction_diffusion', RunReq(params={'feed': 1.0}))

    def test_every_profile_setting_is_published_as_editable(self):
        payload=specs()
        schema=payload['profile_setting_schema']
        for profile,demos in payload['profiles'].items():
            for demo,settings in demos.items():
                self.assertEqual(set(settings),set(schema[demo]),f'{profile}/{demo}')
                for name,value in settings.items():
                    self.assertLessEqual(schema[demo][name]['min'],value)
                    self.assertGreaterEqual(schema[demo][name]['max'],value)

    def test_api_validates_and_accepts_profile_setting_overrides(self):
        with self.assertRaises(HTTPException):
            start('reaction_diffusion',RunReq(settings={'not_a_setting':12}))
        with self.assertRaises(HTTPException):
            start('reaction_diffusion',RunReq(settings={'n':999999}))
        with self.assertRaises(HTTPException):
            start('reaction_diffusion',RunReq(settings={'n':128.5}))
        with patch('app.threading.Thread') as thread:
            result=start('reaction_diffusion',RunReq(settings={'n':144,'ensemble':10}))
        self.assertTrue(result['id'].startswith('reaction_diffusion_'))
        thread.return_value.start.assert_called_once()

    def test_galaxy_mass_controls_are_real_scientific_parameters(self):
        for demo in ('galaxy_collision','galaxy_collision_3d'):
            params=specs()['demos'][demo]['params']
            self.assertIn('milky_way_mass',params)
            self.assertIn('andromeda_mass',params)

    def test_api_rejects_invalid_parallel_and_obstacle_grids(self):
        with self.assertRaises(HTTPException):
            start('cosmic_web', RunReq(parallel_count=10))
        with self.assertRaises(HTTPException):
            start('fluid', RunReq(obstacle_grid=[[0,1],[1,0]]))

    def test_api_accepts_one_reveal_simulation_for_a_fast_test_run(self):
        with patch('app.threading.Thread') as thread:
            result=start('reaction_diffusion', RunReq(parallel_count=1))
        self.assertTrue(result['id'].startswith('reaction_diffusion_'))
        thread.return_value.start.assert_called_once()

    def test_api_publishes_and_validates_galaxy_solvers(self):
        capability=specs()['capabilities']['galaxy_collision']
        self.assertEqual(capability['default_method'],'leapfrog')
        self.assertIn('murb_kinematic',capability['methods'])
        self.assertIn('rk4',capability['method_labels'])
        with self.assertRaisesRegex(HTTPException,'supports these solvers'):
            start('galaxy_collision',RunReq(method='not_a_solver'))

    def test_collision_numerical_step_request_is_bounded(self):
        self.assertEqual(RunReq(numerical_substeps=12).numerical_substeps,12)
        with self.assertRaises(ValueError):
            RunReq(numerical_substeps=0)

    def test_neural_difficulty_changes_the_training_target(self):
        demo = NeuralWallDemo(None, {})
        easy = demo.target(24, kind=1, difficulty=0.5)
        hard = demo.target(24, kind=1, difficulty=2.0)
        self.assertFalse((easy == hard).all())

    def test_canvas_png_becomes_a_neural_target(self):
        image=Image.new('L',(32,32),0)
        for y in range(12,21):
            for x in range(12,21):
                image.putpixel((x,y),255)
        buf=BytesIO(); image.save(buf,format='PNG')
        data='data:image/png;base64,'+base64.b64encode(buf.getvalue()).decode()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'target.png'
            save_target_image(data,path)
            target=NeuralWallDemo(None,{}).drawn_target(16,path)
        self.assertGreater(float(target.max()), .5)
        self.assertLess(float(target[0,0].max()), .01)

    def test_neural_targets_are_rgb(self):
        self.assertEqual(NeuralWallDemo(None,{}).target(16,kind=2).shape,(16,16,3))

    def test_neural_hero_renders_live_weight_view(self):
        demo=NeuralWallDemo(None,{})
        state={'width':6,'w1':np.ones((2,6),dtype=np.float32),
               'w2':np.eye(6,dtype=np.float32),'w3':np.ones((6,1),dtype=np.float32)}
        image=demo.hero_image(np.zeros((16,16)),np.ones((16,16)),np.array([.1]),0,
                              [.5,.1],1,'torch·cpu',1.0,20,100,True,state)
        self.assertEqual(image.size,(1280,720))

    def test_neural_demo_renders_network_view(self):
        with tempfile.TemporaryDirectory() as directory:
            context=RunContext(Path(directory),'neural_wall','local',2,{'target':0,'difficulty':1.0},'numpy')
            NeuralWallDemo(context,{'networks':4,'tile':10,'total_steps':4}).run()
            self.assertTrue((Path(directory)/'frames'/'frame_0000.jpg').exists())


if __name__ == '__main__':
    unittest.main()
