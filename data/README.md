# Reduced observational seed catalogue

`galaxy_observations.npz` is a compact, reproducible input asset for the
`galaxy_collision_3d` exhibition demo. It contains:

- 4,192 Gaia DR3 Milky Way positions transformed from Galactic longitude,
  latitude, and `distance_gspphot` into a Galactocentric Cartesian frame;
- 4,946 PHAT v3 M31 sky positions and F475W-F814W colours, projected onto the
  tangent plane and deprojected with position angle 38 degrees and inclination
  77 degrees.

Regenerate it from the public services with:

```bash
python tools/fetch_galaxy_catalogs.py
```

Sources: [ESA Gaia Archive](https://gea.esac.esa.int/archive/) and
[NOIRLab Astro Data Lab PHAT](https://datalab.noirlab.edu/data/phat).

This is morphology conditioning, not a one-observed-star/one-simulation-body
mapping. Gaia's selected distance sample is local and incomplete through the
obscured Milky Way disc, so its measured radial and vertical distribution is
repeated around an analytic four-arm pattern. PHAT provides projected M31
astrometry, not useful individual line-of-sight depths; the code deprojects
the disc and then models its vertical thickness. These distinctions are also
embedded in every run's `meta.json`.

# Gaia sky for the black-hole demo

`gaia_sky.npz` (35 MB, not committed) holds 1,780,460 Gaia DR3 stars with
position, parallax, parallax error, G magnitude and BP-RP colour:

- every star brighter than G = 11 over the whole sky;
- every star brighter than G = 16 with parallax/error > 5 within 150 pc of
  Gaia BH1, BH2 and BH3, so the sky seen from beside each hole is complete
  close by;
- the three black holes' companion stars, plus each system's published mass,
  distance and reference in the embedded metadata.

Build it (anonymous access, no account needed) with:

```bash
python tools/fetch_gaia_sky.py
```

Source: Gaia DR3 `gaia_source` through the
[NOIRLab Astro Data Lab](https://datalab.noirlab.edu). Credit: ESA/Gaia/DPAC.
