# Demo implementation guidance

This file applies to every solver in this directory. Read the matching
`docs/demos/<demo-id>/README.md` and `AGENTS.md` before changing a demo.

## Shared contract

- Keep `frames/frame_NNNN.jpg` free of titles, readouts, legends, and control
  panels. Presentation belongs in `web/app.js` as HTML.
- Publish changing values through `RunContext.write_status(..., overlay={...})`.
- Put alternate scientific views in `modes/<mode>/`; put optional diagnostic
  imagery such as neural graphs in `overlays/<name>/`.
- Keep total numerical work controlled by the active profile, not multiplied
  accidentally by the requested frame count.
- Make CPU, GPU, and hybrid results scientifically equivalent. Never label a
  fallback as GPU or training when it is not.
- List only precisions a solver genuinely implements in `precisions`; derive
  array dtypes from `ctx.state_dtype`. Hand-written CUDA kernels need a test
  against the NumPy path in FP64 and must pick launch shapes through
  `leonardo_demos/tuning.py`, never a single card's hard-coded optimum.
- Treat reveals as real independent simulations or parameter sweeps. Honour
  `_parallel_count` when the demo supports a reveal ensemble.
- Update `config/demo_specs.json`, every relevant profile, viewer explanations,
  scientific notes, and tests when adding a parameter or output mode.
- State model limitations honestly. These are exhibition models, not automatic
  substitutes for validated research solvers.

## Per-file cautions

- `black_hole.py`: preserve separate 2-D observer and numerical 3-D ray modes;
  do not describe the weak-field spin term as a Kerr integrator.
- `pbh.py`: preserve the collapse/dispersion threshold story and reduced-model
  disclaimer; do not claim full numerical relativity.
- `fluid.py`: the rendered obstacle must be the exact bounce-back mask; custom
  grids must remain bounded and keep inlet/outlet edges open.
- `cosmic_web.py`: expansion, dark-energy acceleration, and warm-DM cutoff are
  independently selectable; label the latter two as qualitative comparisons.
- `galaxy_collision.py`: retain Milky Way/M31 physical units and the genuine
  transverse-velocity reveal sweep.
- `galaxy_collision_3d.py`: every displayed super-particle participates in the
  softened direct force; keep interactive JSON synchronized with JPEG frames.
- `reaction_diffusion.py`: preserve Gray-Scott stability and mature the reveal
  sweep enough for neighbouring feed/kill values to differ visibly.
- `crystal.py`: deep zoom must regenerate geometry, not enlarge a bitmap; keep
  recursive work bounded and process-parallel where appropriate.
- `neural_wall.py`: the main image is the RGB coordinate-network output; target,
  weights, loss, and network graph stay outside the main frame.
- `fusion_plasma.py`: two modes selected through `method`. Write one rotatable
  3-D state per frame in both; magnetic geometry is explanatory and must not be
  called a solved equilibrium. `passive` must keep its pre-merge behaviour. In
  `guardian`, main frames contain vessel state only - sensor panels, policy loss
  and topology are overlays - and every rendered spark is a counted wall
  contact. The reduced control environment and the confined-marker model live in
  `leonardo_demos/plasma_control.py`. See `docs/demos/fusion_plasma/AGENTS.md`.
- `weather_ensemble.py`: the reveal must remain an initial-condition ensemble,
  with uncertainty affecting members rather than only their labels.
- `neuro_racers.py`: the NumPy `RaceSim` path is the reference and the fused
  CuPy kernels must match it; no cuBLAS in `neuroevo.Population.forward`; the
  brain catalogue lives only in `demo_specs.json`. See
  `docs/demos/neuro_racers/AGENTS.md`.
- `bat_vs_moth.py`: main frames are the bat's senses only; "jamming evolved"
  means selective jamming; keep jamming a private benefit and the moth head
  start/quiet start. The NumPy `CaveSim.run` path is the reference and the
  fused CuPy kernels must match it; random draws stay on the seeded host RNG.
  See `docs/demos/bat_vs_moth/AGENTS.md`.
- `molecular_dynamics.py`: preserve stable integration, bounded all-pairs work,
  and the coarse-grained—not atomistic—description. The fused `FOLD_KERNEL`
  must match `fold_forces()`. See `docs/demos/molecular_dynamics/AGENTS.md`.

## Verification

Run `node --check web/app.js`, then `.venv/Scripts/python.exe -m unittest
discover -s tests` on Windows. Render at least one short run for any changed
solver and inspect both the main frame and every alternative view it writes.
