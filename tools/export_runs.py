"""Pack the saved runs worth keeping into one zip for another machine.

By default the bundle holds every favourite, every showcase pick and every run
that executed on a cluster (Discoverer, Leonardo). Copy the zip anywhere (e.g.
Google Drive), put it in runs/_import/ or the project root on the other
machine, and start the viewer: it unpacks the runs and restores the stars.

    python tools/export_runs.py              # write exports/leonardo_runs_<host>_<date>.zip
    python tools/export_runs.py --list       # show what would be packed, write nothing
    python tools/export_runs.py --no-cluster --run fluid_20260912_233626_40cc2
"""
from __future__ import annotations

import argparse
import socket
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from leonardo_demos.run_bundles import BUNDLE_PREFIX, export_bundle, read_meta, run_files, select_runs  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--runs-dir', type=Path, default=ROOT/'runs')
    ap.add_argument('--out', type=Path, help='zip to write (default: exports/leonardo_runs_<host>_<date>.zip)')
    ap.add_argument('--run', action='append', default=[], metavar='RUN_ID', help='also include this run (repeatable)')
    ap.add_argument('--no-favourites', action='store_true')
    ap.add_argument('--no-showcase', action='store_true')
    ap.add_argument('--no-cluster', action='store_true', help='leave out runs made on Discoverer/Leonardo')
    ap.add_argument('--list', action='store_true', help='only list the selected runs and their sizes')
    args = ap.parse_args(argv)

    missing = [r for r in args.run if read_meta(args.runs_dir/r) is None]
    if missing:
        ap.error(f'not a saved run: {", ".join(missing)}')
    ids = select_runs(args.runs_dir, favourites=not args.no_favourites, showcase=not args.no_showcase,
                      cluster=not args.no_cluster, extra=args.run)
    if not ids:
        print('No runs selected: star some runs in the viewer or pass --run RUN_ID.')
        return 1

    if args.list:
        total = 0
        for rid in ids:
            size = sum(f.stat().st_size for f in run_files(args.runs_dir/rid))
            total += size
            print(f'{size/1e6:>10,.0f} MB  {rid}')
        print(f'{total/1e9:>10.2f} GB  total, {len(ids)} runs')
        return 0

    host = socket.gethostname().split('.')[0]
    out = args.out or ROOT/'exports'/f'{BUNDLE_PREFIX}{host}_{time.strftime("%Y%m%d_%H%M")}.zip'
    print(f'Packing {len(ids)} runs into {out}')
    started = time.perf_counter()
    manifest = export_bundle(args.runs_dir, ids, out)
    total = sum(r['bytes'] for r in manifest['runs'])
    print(f'Wrote {out} ({out.stat().st_size/1e9:.2f} GB, {total/1e9:.2f} GB of runs) '
          f'in {time.perf_counter()-started:.0f} s')
    print('On the other machine: put the zip in runs/_import/ (or the project root) and start the viewer.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
