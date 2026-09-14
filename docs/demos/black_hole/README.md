# Black-hole lensing

## Purpose

A virtual camera hovers beside one of the dormant black holes that Gaia found
through the orbit of a companion star, and looks at the real sky through the
hole's gravity. Every pixel is a light ray followed backwards along its exact
path to the Gaia star it came from.

There are two solvers, chosen with `--method`:

| Method | What it is |
| --- | --- |
| `schwarzschild` (default) | Exact null geodesics of a non-rotating black hole, lighting a sky built from 1.8 million Gaia DR3 stars. |
| `weak_field` (legacy) | The original exhibition model: a weak-field deflection applied to a NASA Hubble Deep Field image. Kept for comparison and for older saved runs. |

Saved runs from before the choice existed record `method: "default"`; the
viewers treat those as `weak_field`.

## Implementation map

- Geodesics: `leonardo_demos/schwarzschild.py`
- Gaia sky (catalogue, parallax shift, cube map, glow map):
  `leonardo_demos/gaia_sky.py`
- Demo, tone mapping and the ray-path renderer:
  `leonardo_demos/demos/black_hole.py`
- Catalogue fetcher: `tools/fetch_gaia_sky.py` → `data/gaia_sky.npz`
  (35 MB, not committed; see `data/README.md`)
- Parameters, `schwarzschild`: `target`, `camera_distance`, `orbit`, `dive`
- Parameters, `weak_field`: `mass`, `spin`, `lens_x`, `lens_y`, `lens_count`,
  `lens_separation`, `lens_angle`, `disk`
  (each parameter's `methods` field in `config/demo_specs.json` says which
  solver reads it; both viewers hide the others)
- Profile settings: `width`, `height`, `sky_face` (cube-map texels per face),
  `supersample` (rays per pixel = supersample²), `fov_deg`, `exposure`, `glow`,
  `star_colour` (colour boost; 1 = plain blackbody colour, default 2.5)
- Outputs: `frames/` (camera view) and `modes/3d/` (exact ray paths)
- Tests: `tests/test_black_hole_exact.py`

## The targets

| Target | Mass | Distance | Reference |
| --- | --- | --- | --- |
| Gaia BH3 | 32.70 ± 0.82 M☉ | 590 pc | Gaia Collaboration, Panuzzo et al. 2024, A&A 686, L2 |
| Gaia BH1 | 9.62 ± 0.18 M☉ | 480 pc | El-Badry et al. 2023, MNRAS 518, 1057 |
| Gaia BH2 | 8.9 ± 0.3 M☉ | 1.16 kpc | El-Badry et al. 2023, MNRAS 521, 4323 |
| Solar System | 10 M☉ (hypothetical) | 0 | — |

Positions are the companions' Gaia DR3 astrometry. Distances are the
published orbital-solution values rather than the DR3 single-star parallax,
which is biased by the orbit (for BH2 it would give 1.49 kpc).

## How the picture is made

1. **The sky from the black hole.** Each catalogue star with
   `parallax > 5σ` is placed at 1/parallax along its Gaia direction; the rest
   go on a 3 kpc background shell. The camera position is subtracted, so
   nearby stars shift and brighten or fade by `5 log10(d_new / d_earth)`.
   The companion star itself is removed (it would sit on top of the camera).
   Colour comes from BP−RP → effective temperature → blackbody RGB, balanced
   so the median catalogue star (about 4900 K) is white, as in an
   astrophotograph. Hotter stars then read blue and cooler ones orange.
2. **Sky maps.** Stars G < 8 (as seen from the camera) are splatted into a
   cube map as radiance (flux ÷ texel solid angle). Fainter stars also feed a
   blurred equirectangular map, the unresolved Milky Way glow.
3. **Exact escape angles.** In geometric units (M = 1, horizon r = 2, photon
   sphere r = 3), a ray leaving a static camera at radius r with angle ψ from
   the inward direction has impact parameter `b = r sin ψ / √(1 − 2/r)`.
   Rays with b below 3√3 on the inward side are captured. For the rest, the
   total swept angle comes from the orbit equation
   `(du/dφ)² = 1/b² − u² + 2u³` by composite Gauss–Legendre quadrature, with a
   substitution that removes the endpoint singularity and a factorisation
   that avoids cancellation near the photon sphere. This is tabulated once per
   camera radius against ψ, finely spaced towards the shadow edge where the
   winding diverges logarithmically.
4. **Render.** Every pixel's direction is rotated into its orbital plane,
   its escape angle looked up, and the sky maps sampled in the direction the
   ray leaves to. Surface brightness is conserved by lensing; a static
   observer also sees starlight blueshifted, so intensity is scaled by g⁴
   with `g = 1/√(1 − 2/r)`.
5. **Tone map.** Stars use a fixed linear gain and a colour boost (`star_colour`); diffuse glow uses a per-sky
   contrast curve anchored to the sky's own median and 99.9th percentile, so
   every target is exposed alike. Bright stars get a two-scale bloom. The
   curve compresses by the brightest channel, so bright stars keep their hue
   instead of bleaching to white, and frames are saved as 4:4:4 JPEG so
   single-pixel stars keep their colour.
6. **Ray paths.** `modes/3d/` draws fans of geodesics in three planes through
   the camera, integrated with RK4 on `u'' = 3u² − u` for display. Escaping
   rays are coloured by their exact bend (escape angle from the table minus
   the straight-line angle): blue near 0°, green 90°, yellow 180°, magenta
   360° and beyond. Captured rays are red.

## Validation

`tests/test_black_hole_exact.py` checks:

- the weak-field series `α = 4/b + 15π/(4b²)` far from the hole;
- capture switching exactly at b = 3√3, and the analytic shadow size;
- the strong-deflection slope `dφ / d ln(b − b_c) = −1`;
- the table against direct quadrature and against RK4 (better than 2×10⁻⁵ rad
  from r = 3.4 to 80);
- a flat-space identity, cube-map flux conservation, parallax and
  inverse-square brightness, companion removal, and a full small run.

## Scientific boundary

Not modelled:

- spin (Kerr);
- accretion light (these black holes are dormant, so this is realistic);
- the companion star;
- the camera's own orbital motion (each frame is a camera held still at that
  point, so "orbit" and "dive" are a sequence of static views);
- dust extinction re-derived along the new lines of sight;
- stars brighter than about G = 3, which Gaia does not measure (this mainly
  affects the Solar-System view);
- stars outside the catalogue cuts: G < 11 all-sky, plus G < 16 within 150 pc
  of each black hole.

## Extension points

A Kerr solver would replace `EscapeTable` and `lens_directions` (the ray
direction no longer stays in one plane). Keep both output folders and publish
method details through `meta.json`, not in the frames.
