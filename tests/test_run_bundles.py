import json
import tempfile
import unittest
import zipfile
from pathlib import Path

from leonardo_demos import run_bundles as rb


def make_run(runs, rid, demo, slurm_job=None):
    (runs/rid/'frames').mkdir(parents=True)
    (runs/rid/'frames'/'frame_0000.jpg').write_bytes(b'jpeg'*100)
    resources={'host':'desk','slurm':{'job_id':slurm_job} if slurm_job else {}}
    (runs/rid/'meta.json').write_text(json.dumps({'demo':demo,'status':'complete','resources':resources}))


class RunBundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        base=Path(self.tmp.name)
        self.src=base/'desktop'/'runs'; self.dst=base/'laptop'/'runs'
        self.src.mkdir(parents=True); self.dst.mkdir(parents=True)
        make_run(self.src,'fluid_fav','fluid')
        make_run(self.src,'fluid_plain','fluid')
        make_run(self.src,'galaxy_7653','galaxy_collision_3d',slurm_job='7653')
        make_run(self.src,'bh_show','black_hole')
        (self.src/'_library.json').write_text(json.dumps(
            {'favourites':['fluid_fav'],'showcase':{'black_hole':['bh_show']}}))
        self.zip=base/'leonardo_runs_desk.zip'

    def tearDown(self):
        self.tmp.cleanup()

    def update_dst_library(self, change):
        path=self.dst/'_library.json'
        path.write_text(json.dumps(change(rb.read_library(self.dst))))

    def test_selects_favourites_showcase_and_cluster_runs(self):
        self.assertEqual(sorted(rb.select_runs(self.src)),['bh_show','fluid_fav','galaxy_7653'])
        self.assertEqual(rb.select_runs(self.src,cluster=False,showcase=False,extra=['fluid_plain']),
                         ['fluid_fav','fluid_plain'])

    def test_round_trip_restores_runs_and_library(self):
        rb.export_bundle(self.src,rb.select_runs(self.src),self.zip,log=lambda m:None)
        make_run(self.dst,'local_bh','black_hole')
        self.dst.joinpath('_library.json').write_text(json.dumps(
            {'favourites':['local_bh'],'showcase':{'black_hole':['local_bh']}}))
        rb.import_pending(self.dst,[self.zip.parent],self.update_dst_library,log=lambda m:None)
        for rid in ('fluid_fav','galaxy_7653','bh_show'):
            self.assertEqual((self.dst/rid/'frames'/'frame_0000.jpg').read_bytes(),b'jpeg'*100)
        self.assertFalse((self.dst/'fluid_plain').exists())
        library=rb.read_library(self.dst)
        self.assertEqual(library['favourites'],['local_bh','fluid_fav'])
        # A showcase already chosen on this machine is kept.
        self.assertEqual(library['showcase'],{'black_hole':['local_bh']})
        self.assertFalse((self.dst/'_import'/'.partial').exists())

    def test_bundle_is_not_imported_twice(self):
        rb.export_bundle(self.src,['fluid_fav'],self.zip,log=lambda m:None)
        first=rb.import_pending(self.dst,[self.zip.parent],self.update_dst_library,log=lambda m:None)
        self.assertEqual(list(first.values())[0]['imported'],['fluid_fav'])
        for f in sorted((self.dst/'fluid_fav').rglob('*'),reverse=True):
            f.unlink() if f.is_file() else f.rmdir()
        (self.dst/'fluid_fav').rmdir()
        self.assertEqual(rb.import_pending(self.dst,[self.zip.parent],self.update_dst_library,log=lambda m:None),{})
        self.assertFalse((self.dst/'fluid_fav').exists())

    def test_existing_runs_are_not_overwritten(self):
        rb.export_bundle(self.src,['fluid_fav'],self.zip,log=lambda m:None)
        make_run(self.dst,'fluid_fav','fluid')
        (self.dst/'fluid_fav'/'meta.json').write_text('{"demo":"fluid","mine":true}')
        result=rb.import_bundle(self.zip,self.dst,self.update_dst_library,log=lambda m:None)
        self.assertEqual(result,{'imported':[],'skipped':['fluid_fav']})
        self.assertIn('mine',(self.dst/'fluid_fav'/'meta.json').read_text())

    def test_paths_escaping_the_run_folder_are_rejected(self):
        with zipfile.ZipFile(self.zip,'w') as zf:
            zf.writestr(rb.MANIFEST,json.dumps({'format':rb.FORMAT,'runs':[],'library':{}}))
            zf.writestr('runs/x/../../evil.txt','boom')
        with self.assertRaises(rb.BundleError):
            rb.import_bundle(self.zip,self.dst,self.update_dst_library,log=lambda m:None)
        self.assertFalse((self.dst.parent/'evil.txt').exists())

    def test_the_viewer_imports_bundles_however_it_was_started(self):
        """A stand started with uvicorn must unpack the zip on the desk too.

        The import used to hang off app.py's __main__ block, so `uvicorn
        app:app` came up with an empty gallery."""
        import app
        names = {getattr(handler, '__name__', '') for handler in app.app.router.on_startup}
        self.assertIn('on_start', names, 'the viewer imports bundles only from __main__')

    def test_only_named_bundles_are_picked_up_outside_the_import_folder(self):
        root=self.zip.parent; (root/'other.zip').write_bytes(b'')
        drop=self.dst/'_import'; drop.mkdir()
        (drop/'drive-download.zip').write_bytes(b'')
        rb.export_bundle(self.src,['fluid_fav'],self.zip,log=lambda m:None)
        found=[p.name for p in rb.find_bundles([drop,root])]
        self.assertEqual(found,['drive-download.zip','leonardo_runs_desk.zip'])


if __name__=='__main__':
    unittest.main()
