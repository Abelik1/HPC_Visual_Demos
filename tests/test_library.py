import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from fastapi import HTTPException

import app
from app import FavouriteReq, ShowcaseReq


class RunLibraryTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.runs=Path(self.tmp.name)
        for rid,demo in (('fluid_a','fluid'),('fluid_b','fluid'),('bh_a','black_hole')):
            (self.runs/rid/'frames').mkdir(parents=True)
            (self.runs/rid/'frames'/'frame_0000.jpg').write_bytes(b'x')
            (self.runs/rid/'meta.json').write_text(json.dumps({'demo':demo,'status':'complete'}))
        self.patches=[patch.object(app,'RUNS',self.runs),patch.object(app,'LIBRARY',self.runs/'_library.json')]
        for p in self.patches: p.start()

    def tearDown(self):
        for p in self.patches: p.stop()
        self.tmp.cleanup()

    def test_favourites_round_trip_into_the_run_listing(self):
        app.set_favourite('fluid_b',FavouriteReq(favourite=True))
        listed={r['id']:r['favourite'] for r in app.list_runs()}
        self.assertEqual(listed,{'fluid_a':False,'fluid_b':True,'bh_a':False})
        app.set_favourite('fluid_b',FavouriteReq(favourite=False))
        self.assertEqual(app.library()['favourites'],[])

    def test_library_file_is_not_listed_as_a_run(self):
        app.set_favourite('fluid_a',FavouriteReq(favourite=True))
        self.assertNotIn('_library',[r['id'] for r in app.list_runs()])

    def test_showcase_only_accepts_runs_of_that_demo(self):
        app.set_showcase('fluid',ShowcaseReq(runs=['fluid_b','fluid_a']))
        self.assertEqual(app.library()['showcase'],{'fluid':['fluid_b','fluid_a']})
        with self.assertRaises(HTTPException):
            app.set_showcase('fluid',ShowcaseReq(runs=['bh_a']))
        with self.assertRaises(HTTPException):
            app.set_favourite('../outside',FavouriteReq(favourite=True))
        with self.assertRaises(ValueError):
            ShowcaseReq(runs=['a','b','c','d'])

    def test_deleted_runs_drop_out_of_the_library(self):
        app.set_showcase('fluid',ShowcaseReq(runs=['fluid_a']))
        app.set_favourite('fluid_a',FavouriteReq(favourite=True))
        for f in (self.runs/'fluid_a').rglob('*'):
            if f.is_file(): f.unlink()
        (self.runs/'fluid_a'/'frames').rmdir(); (self.runs/'fluid_a').rmdir()
        self.assertEqual(app.library(),{'favourites':[],'showcase':{'fluid':[]}})


class CompressionImageTests(unittest.TestCase):
    def test_default_pictures_are_listed_and_readme_is_not(self):
        listed=app.compression_images()
        names=[item['url'] for item in listed]
        self.assertTrue(listed)
        self.assertFalse(any(url.endswith('.md') for url in names))
        self.assertIn('Hubble deep field',[item['name'] for item in listed])


if __name__=='__main__':
    unittest.main()
