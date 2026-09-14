# Agent notes: black-hole lensing

- Read `../../../leonardo_demos/demos/AGENTS.md` and this guide first.
- Two solvers: `schwarzschild` (default, exact geodesics on the Gaia sky) and
  `weak_field` (legacy Hubble image). Keep both working; old saved runs record
  `method: "default"` and are weak-field.
- Preserve `lens(bg, mass, spin, t=...)` compatibility used by tests and tools.
- Never replace the exact geodesics with a fitted or decorative deflection.
  Any change to `leonardo_demos/schwarzschild.py` must keep
  `tests/test_black_hole_exact.py` passing at its current tolerances.
- Keep ray paths numerical; do not replace them with decorative curves.
- Keep the camera view in `frames/` and ray paths in `modes/3d/`. The exact
  solver defaults the viewer to the camera view.
- The Gaia catalogue is `data/gaia_sky.npz`, built by `tools/fetch_gaia_sky.py`;
  missing it raises `CatalogueMissing` with that instruction.
- Describe the exact method's limits honestly (no spin, no disc, no companion,
  no re-derived extinction, Gaia bright-star incompleteness).
- Verify one frame from each mode and run `tests/test_black_hole_exact.py` and
  `tests/test_small_demos.py`.
