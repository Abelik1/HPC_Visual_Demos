#!/usr/bin/env python3
"""Download the Gaia DR3 star catalogue used by the black-hole ray tracer.

Two parts, merged and de-duplicated by source_id:

* the whole sky to G < 11 (about 1.25 million stars): the sky as seen from the
  Solar System, and the distant background as seen from anywhere nearby;
* around each of the three Gaia black holes, every star within 150 pc of it
  in 3-D with G < 16 and a parallax measured to better than 20 %: seen from
  the black hole these neighbours are close, so faint-from-Earth stars there
  can be among the brightest in its sky.

Black-hole parameters are the published orbit solutions, not DR3's single-star
astrometry (which ignores the orbital motion; for BH2 it puts the system at
1.49 kpc instead of 1.16 kpc). Output: data/gaia_sky.npz (~35 MB, not versioned).

    python tools/fetch_gaia_sky.py
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_galaxy_catalogs import query  # noqa: E402  (shared anonymous Data Lab client)

COLUMNS = "source_id,ra,dec,parallax,parallax_error,phot_g_mean_mag,bp_rp"
NEIGHBOURHOOD_PC = 150.0

# Companion stars in Gaia DR3 and the published black-hole solutions.
BLACK_HOLES = {
    "gaia_bh3": {"label": "Gaia BH3", "companion_source_id": 4318465066420528000,
                 "mass_msun": 32.70, "mass_err_msun": 0.82, "distance_pc": 590.0,
                 "reference": "Gaia Collaboration, Panuzzo et al. 2024, A&A 686, L2 (arXiv:2404.10486)"},
    "gaia_bh1": {"label": "Gaia BH1", "companion_source_id": 4373465352415301632,
                 "mass_msun": 9.62, "mass_err_msun": 0.18, "distance_pc": 480.0,
                 "reference": "El-Badry et al. 2023, MNRAS 518, 1057 (arXiv:2209.06833)"},
    "gaia_bh2": {"label": "Gaia BH2", "companion_source_id": 5870569352746779008,
                 "mass_msun": 8.9, "mass_err_msun": 0.3, "distance_pc": 1160.0,
                 "reference": "El-Badry et al. 2023, MNRAS 521, 4323 (arXiv:2302.07880)"},
}


def retry(sql: str, attempts: int = 4) -> list[dict[str, str]]:
    for attempt in range(attempts):
        try:
            return query(sql, timeout=600)
        except Exception as error:  # network hiccups and Data Lab timeouts
            if attempt == attempts - 1:
                raise
            print(f"  retrying after: {error}", flush=True)
            time.sleep(10 * (attempt + 1))
    return []


def fetch_all_sky(g_limit: float, workers: int) -> list[dict[str, str]]:
    bands = list(range(-90, 90, 10))

    def band(lower: int) -> list[dict[str, str]]:
        upper = lower + 10
        top = "<=" if upper == 90 else "<"
        rows = retry(f"select {COLUMNS} from gaia_dr3.gaia_source where phot_g_mean_mag < {g_limit} "
                     f"and dec >= {lower} and dec {top} {upper}")
        print(f"  all-sky dec {lower:+d}..{upper:+d}: {len(rows):,}", flush=True)
        return rows

    with ThreadPoolExecutor(workers) as pool:
        return [row for rows in pool.map(band, bands) for row in rows]


def fetch_neighbourhood(ra: float, dec: float, distance_pc: float) -> list[dict[str, str]]:
    radius = math.degrees(math.asin(min(1.0, NEIGHBOURHOOD_PC / distance_pc)))
    near = 1000.0 / (distance_pc + NEIGHBOURHOOD_PC)
    far = 1000.0 / max(1.0, distance_pc - NEIGHBOURHOOD_PC)
    return retry(f"select {COLUMNS} from gaia_dr3.gaia_source "
                 f"where q3c_radial_query(ra, dec, {ra:.6f}, {dec:.6f}, {radius:.4f}) "
                 f"and parallax between {near:.5f} and {far:.5f} "
                 "and parallax_over_error > 5 and phot_g_mean_mag < 16")


def number(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    try:
        return float(value)
    except ValueError:
        return math.nan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parents[1] / "data" / "gaia_sky.npz")
    parser.add_argument("--g-limit", type=float, default=11.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    ids = ",".join(str(bh["companion_source_id"]) for bh in BLACK_HOLES.values())
    companions = {int(r["source_id"]): r for r in retry(f"select {COLUMNS} from gaia_dr3.gaia_source where source_id in ({ids})")}
    targets = {}
    for key, bh in BLACK_HOLES.items():
        c = companions[bh["companion_source_id"]]
        targets[key] = {**bh, "ra_deg": float(c["ra"]), "dec_deg": float(c["dec"]),
                        "dr3_parallax_mas": number(c, "parallax"), "companion_g_mag": number(c, "phot_g_mean_mag")}
        print(f"{bh['label']}: RA {targets[key]['ra_deg']:.5f} Dec {targets[key]['dec_deg']:+.5f}", flush=True)

    print(f"All-sky stars with G < {args.g_limit}", flush=True)
    rows = fetch_all_sky(args.g_limit, args.workers)
    for key, target in targets.items():
        print(f"Neighbourhood of {target['label']} (within {NEIGHBOURHOOD_PC:.0f} pc)", flush=True)
        extra = fetch_neighbourhood(target["ra_deg"], target["dec_deg"], target["distance_pc"])
        print(f"  {len(extra):,} stars", flush=True)
        target["neighbourhood_rows"] = len(extra)
        rows.extend(extra)

    seen: set[int] = set()
    unique = []
    for row in rows:
        source = int(row["source_id"])
        if source not in seen:
            seen.add(source)
            unique.append(row)

    columns = {key: np.array([number(r, key) for r in unique], dtype=np.float32)
               for key in ("ra", "dec", "parallax", "parallax_error", "phot_g_mean_mag", "bp_rp")}
    good = np.isfinite(columns["ra"]) & np.isfinite(columns["dec"]) & np.isfinite(columns["phot_g_mean_mag"])
    metadata = {
        "source": "Gaia DR3 gaia_source via NOIRLab Astro Data Lab (https://datalab.noirlab.edu)",
        "credit": "ESA/Gaia/DPAC",
        "selection": {"all_sky_g_limit": args.g_limit, "neighbourhood_pc": NEIGHBOURHOOD_PC,
                      "neighbourhood_g_limit": 16, "neighbourhood_parallax_over_error": 5},
        "stars": int(good.sum()),
        "retrieved": time.strftime("%Y-%m-%d"),
        "black_holes": targets,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **{k: v[good] for k, v in columns.items()},
                        metadata=np.array(json.dumps(metadata)))
    size = args.output.stat().st_size / 1e6
    print(f"Wrote {good.sum():,} stars to {args.output} ({size:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
