# Galaxy collision — full 3-D gravity

## Purpose

Runs a softened direct-force encounter in which every visible disc, bulge, and
dark-halo super-particle contributes to the gravitational acceleration.

## Implementation map

- Solver and JPEG renderer: `leonardo_demos/demos/galaxy_collision_3d.py`
- Interactive renderer: `web/galaxy3d_view.js`
- Scientific parameters: impact, speed, disc tilt, softening, Milky Way mass,
  and Andromeda mass
- Editable profile settings: massive super-particle count, solver substeps,
  simulated Gyr span, force tile size, and maximum gravity step
- Main frames: `frames/`
- Rotatable state: one JSON file per frame under `interactive/`

## Data and scope

Gaia and PHAT-derived samples condition the visible starting morphology, while
the reduced particle system remains an illustrative super-particle experiment,
not a fitted equilibrium prediction of the Local Group.

## Performance

Force work scales quadratically with particle count. CPU calculation is tiled
and may use assigned workers. CUDA uses a register-blocked, shared-memory
tiled all-pairs kernel whose block size and targets-per-thread are autotuned
on the running GPU. The leapfrog reuses each frame's closing force as the next
frame's opening force. Precision is `fp32`, `mixed` (FP64 state and force
accumulation around FP32 pair maths) or `fp64`; see `docs/PERFORMANCE.md`.
`tools/fidelity_run.py` and `tools/fidelity_compare.py` measure energy,
momentum and trajectory accuracy between runs.
