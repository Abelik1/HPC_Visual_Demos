"""Portable bundles of saved runs.

runs/ is not versioned and a finished run can be gigabytes of frames, so the
runs worth keeping travel between machines as one zip (copied through Google
Drive, a USB stick, scp from Discoverer...). tools/export_runs.py writes the
zip; the viewer unpacks any bundle it finds in runs/_import/ or the project
root when it starts.

Layout of a bundle:
    _bundle.json            manifest: runs, sizes, favourites and showcase picks
    runs/<run id>/...       each run folder exactly as it was saved
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

BUNDLE_PREFIX = 'leonardo_runs_'
MANIFEST = '_bundle.json'
FORMAT = 1
IMPORT_DIR_NAME = '_import'
LEDGER_NAME = 'imported.json'
# Frames are already JPEG/PNG: deflating thousands of them costs minutes and
# saves almost nothing, so they are stored as-is.
STORED_SUFFIXES = {'.jpg', '.jpeg', '.png', '.webp', '.gif', '.npz', '.npy',
                   '.mp4', '.webm', '.gz', '.zip'}


class BundleError(Exception):
    pass


def read_meta(run_dir: Path):
    try:
        return json.loads((run_dir/'meta.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return None


def read_library(runs_dir: Path):
    try:
        data = json.loads((runs_dir/'_library.json').read_text(encoding='utf-8'))
    except (OSError, ValueError):
        data = {}
    return {'favourites': [r for r in data.get('favourites', []) if isinstance(r, str)],
            'showcase': {k: [r for r in v if isinstance(r, str)]
                         for k, v in (data.get('showcase') or {}).items() if isinstance(v, list)}}


def is_cluster_run(meta: dict) -> bool:
    """True when the run executed inside a Slurm job (Discoverer, Leonardo)."""
    resources = meta.get('resources') or {}
    return bool((resources.get('slurm') or {}).get('job_id') or resources.get('slurm_job_id'))


def select_runs(runs_dir: Path, *, favourites=True, showcase=True, cluster=True, extra=()):
    """Run ids worth carrying: favourites, showcase picks and cluster runs."""
    library = read_library(runs_dir)
    chosen = list(library['favourites']) if favourites else []
    if showcase:
        for ids in library['showcase'].values():
            chosen += ids
    if cluster:
        for d in sorted(runs_dir.iterdir()):
            if d.is_dir() and not d.name.startswith('_') and is_cluster_run(read_meta(d) or {}):
                chosen.append(d.name)
    chosen += list(extra)
    return [r for r in dict.fromkeys(chosen)
            if not r.startswith('_') and read_meta(runs_dir/r) is not None]


def run_files(run_dir: Path):
    return sorted(p for p in run_dir.rglob('*') if p.is_file())


def export_bundle(runs_dir: Path, run_ids, out_path: Path, log: Callable[[str], None] = print):
    """Write the runs into one zip. Returns the manifest."""
    run_ids = list(dict.fromkeys(run_ids))
    library = read_library(runs_dir)
    wanted = set(run_ids)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Written under another name first so a half-finished export is never
    # mistaken for a bundle by an import scan.
    partial = out_path.with_name(out_path.name+'.partial')
    runs = []
    with zipfile.ZipFile(partial, 'w', allowZip64=True) as zf:
        for index, rid in enumerate(run_ids, 1):
            rd = runs_dir/rid
            files = run_files(rd)
            size = 0
            for f in files:
                size += f.stat().st_size
                method = zipfile.ZIP_STORED if f.suffix.lower() in STORED_SUFFIXES else zipfile.ZIP_DEFLATED
                zf.write(f, f'runs/{rid}/{f.relative_to(rd).as_posix()}', compress_type=method)
            runs.append({'id': rid, 'demo': (read_meta(rd) or {}).get('demo'),
                         'files': len(files), 'bytes': size})
            log(f'  [{index}/{len(run_ids)}] {rid}  {size/1e6:,.0f} MB')
        manifest = {
            'format': FORMAT,
            'created': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'host': socket.gethostname(),
            'runs': runs,
            'library': {
                'favourites': [r for r in library['favourites'] if r in wanted],
                'showcase': {demo: kept for demo, ids in library['showcase'].items()
                             if (kept := [r for r in ids if r in wanted])},
            },
        }
        zf.writestr(MANIFEST, json.dumps(manifest, indent=2))
    partial.replace(out_path)
    return manifest


def merge_library(local: dict, incoming: dict, available: set[str]) -> dict:
    """Add a bundle's favourites and showcase picks without undoing local choices."""
    favourites = list(local.get('favourites', []))
    for rid in incoming.get('favourites', []):
        if rid in available and rid not in favourites:
            favourites.append(rid)
    showcase = {k: list(v) for k, v in (local.get('showcase') or {}).items()}
    for demo, ids in (incoming.get('showcase') or {}).items():
        ids = [r for r in ids if r in available]
        if ids and not showcase.get(demo):
            showcase[demo] = ids
    return {'favourites': favourites, 'showcase': showcase}


def _bundle_members(zf: zipfile.ZipFile):
    """Group zip entries by run id, rejecting any path that escapes runs/<id>/."""
    members: dict[str, list[tuple[zipfile.ZipInfo, tuple[str, ...]]]] = {}
    for info in zf.infolist():
        if info.filename == MANIFEST:
            continue
        path = PurePosixPath(info.filename.replace('\\', '/'))
        parts = path.parts
        if (path.is_absolute() or len(parts) < 3 or parts[0] != 'runs'
                or any(p in ('', '.', '..') or ':' in p for p in parts)
                or parts[1].startswith('_')):
            raise BundleError(f'unexpected entry {info.filename!r}')
        members.setdefault(parts[1], []).append((info, parts[2:]))
    return members


def _rename_when_free(src: Path, dst: Path):
    """os.rename, retried while Windows holds a freshly written file open.

    Defender and the search indexer open new files to scan them, and renaming a
    folder with any file open fails with WinError 5. Unpacking a 14 GB bundle
    lost that race at a different run each time and stopped half way."""
    for attempt in range(100):
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            time.sleep(0.2)
    os.rename(src, dst)


def import_bundle(zip_path: Path, runs_dir: Path, update_library: Callable[[Callable[[dict], dict]], None],
                  log: Callable[[str], None] = print):
    """Unpack one bundle into runs_dir. Runs that already exist are left alone.

    update_library receives a function mapping the current library to the
    merged one, so the caller can apply it under its own lock.
    """
    staging_root = runs_dir/IMPORT_DIR_NAME/'.partial'
    with zipfile.ZipFile(zip_path) as zf:
        try:
            manifest = json.loads(zf.read(MANIFEST))
        except (KeyError, ValueError):
            raise BundleError('not a run bundle (no _bundle.json)')
        if manifest.get('format') != FORMAT:
            raise BundleError(f'bundle format {manifest.get("format")!r} is not supported')
        members = _bundle_members(zf)
        pending = [rid for rid in members if not (runs_dir/rid).exists()]
        present = {rid for rid in members if (runs_dir/rid).exists()}
        needed = sum(info.file_size for rid in pending for info, _ in members[rid])
        free = shutil.disk_usage(runs_dir).free
        if needed > free:
            raise BundleError(f'needs {needed/1e9:.1f} GB free, only {free/1e9:.1f} GB available')
        if present:
            log(f'  {len(present)} run(s) already here, skipped')
        for index, rid in enumerate(pending, 1):
            staging = staging_root/rid
            shutil.rmtree(staging, ignore_errors=True)
            newest = 0.0
            for info, rel in members[rid]:
                target = staging.joinpath(*rel)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                with zf.open(info) as src, open(target, 'wb') as dst:
                    shutil.copyfileobj(src, dst, 1 << 20)
                newest = max(newest, time.mktime(info.date_time+(0, 0, -1)))
            if not (staging/'meta.json').is_file():
                shutil.rmtree(staging, ignore_errors=True)
                log(f'  [{index}/{len(pending)}] {rid}: no meta.json, skipped')
                continue
            _rename_when_free(staging, runs_dir/rid)
            # The run list is ordered by folder time; keep the original age
            # instead of floating every imported run to the top.
            if newest:
                os.utime(runs_dir/rid, (newest, newest))
            present.add(rid)
            log(f'  [{index}/{len(pending)}] {rid}')
        update_library(lambda local: merge_library(local, manifest.get('library') or {}, present))
    shutil.rmtree(staging_root, ignore_errors=True)
    return {'imported': [r for r in pending if r in present], 'skipped': sorted(set(members)-set(pending))}


def find_bundles(folders):
    """Bundle zips waiting to be imported. Any zip counts inside runs/_import;
    elsewhere only files named like an export."""
    found = []
    for folder in folders:
        folder = Path(folder)
        if not folder.is_dir():
            continue
        pattern = '*.zip' if folder.name == IMPORT_DIR_NAME else BUNDLE_PREFIX+'*.zip'
        found += sorted(p for p in folder.glob(pattern) if p.is_file())
    return list(dict.fromkeys(found))


def import_pending(runs_dir: Path, folders, update_library, log: Callable[[str], None] = print):
    """Import every new bundle in folders. A ledger in runs/_import remembers
    finished bundles, so a zip left in place is not unpacked again and runs
    deleted afterwards stay deleted."""
    import_dir = runs_dir/IMPORT_DIR_NAME
    import_dir.mkdir(parents=True, exist_ok=True)
    ledger_path = import_dir/LEDGER_NAME
    try:
        ledger = json.loads(ledger_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        ledger = {}
    results = {}
    for bundle in find_bundles(folders):
        key = f'{bundle.name}:{bundle.stat().st_size}'
        if key in ledger:
            continue
        log(f'Importing saved runs from {bundle} ...')
        try:
            result = import_bundle(bundle, runs_dir, update_library, log)
        except (BundleError, zipfile.BadZipFile, OSError) as exc:
            log(f'  could not import {bundle.name}: {exc}')
            continue
        ledger[key] = {'imported_at': time.strftime('%Y-%m-%dT%H:%M:%S'), **result}
        tmp = ledger_path.with_suffix('.tmp')
        tmp.write_text(json.dumps(ledger, indent=2), encoding='utf-8')
        tmp.replace(ledger_path)
        results[str(bundle)] = result
        log(f'  done: {len(result["imported"])} run(s) added. '
            f'The zip is no longer needed and can be deleted: {bundle}')
    return results
