# Star in a Bottle

## Purpose

One magnetic bottle, two selectable modes.

- **`passive` — Mode 1 · Passive confinement.** Evolves a reduced nonlinear
  plasma-wave field on a periodic lattice, maps that state onto a torus, and
  advects passive tracers through its derived drift.
- **`guardian` — Mode 2 · AI plasma guardian.** Keeps that field as the
  turbulence source and hands the coils to a small neural policy. A population
  of guiding-centre-like markers is held inside the torus by the field those
  commands shape, and every marker the policy fails to hold reaches the wall,
  sparks and is recycled into the core.

  The run is a sequence of **virtual shots**, not one continuous episode.
  Simulation and training are separate phases:

  1. A shot runs with the policy **frozen**. Nothing the network does can
     perturb the physics being displayed.
  2. The shot is scored on the markers it lost to the wall.
  3. Between shots the policy is optimized — half of each training batch is
     replayed from the states that shot actually visited, half is sampled at
     random — and then the next shot starts.

  Every shot is the *identical* experiment: same field seed, same marker seed,
  same start state, same disturbance sequence. Only the policy differs, so the
  scoreboard is a controlled comparison. A typical local run goes 616 → 92 →
  55 → 73 → 15 → 12 markers lost against a constant no-control reference of
  571; the small regression is real, not smoothed away.

Both modes share the same solver, the same parameters and the same rotatable
3-D viewer. The mode is the run's `method`, chosen in the run panel.

## Implementation map

- Solver, control loop and renderers: `leonardo_demos/demos/fusion_plasma.py`
- Control environment, policies and confined markers: `leonardo_demos/plasma_control.py`
- Interactive 3-D renderer: `web/fusion_view.js`
- Parameters: magnetic field, heating, density, instability drive (guardian only)
- Main frames: clean pre-rendered torus views
- Live 3-D state: `modes/fusion3d/frame_NNNN.json`
- Legacy final-state fallback: `fusion_view.json`
- Guardian overlays: `overlays/network/frame_NNNN.jpg` (policy graph),
  `overlays/poloidal/frame_NNNN.jpg` (vessel cross-section) and
  `overlays/shots/frame_NNNN.jpg` (the training scoreboard)
- Per-shot results are published in `meta.json` under `shot_history`

## Viewer behaviour

The rotatable 3-D canvas is the default in both modes and consumes one state
per playback frame. Plasma and magnetic geometry are independent layers, and in
guardian mode a third layer toggles the wall losses. Visitors can isolate one
subset of tracers or markers without changing the simulated state. The run's
`meta.json` publishes the overlay folders it wrote, and the viewer offers only
those.

## Visual language (guardian mode)

In the torus view:

- Cyan markers sit deep inside the bottle; amber ones are close to the wall
- Every orange burst on the wall is a marker that escaped confinement
- Eight rings encircle the machine just outside the wall: the same three coil
  banks as the cross-section, extruded toroidally. Brightness and thickness are
  the command magnitude, so an idle bank stays dark. The far half of each ring
  is drawn behind the vessel and the near half in front of everything inside it.
- Helical lines are the confinement geometry the three coil commands are shaping

The cross-section overlay is one slice through the torus seen end-on, with every
marker in the volume drawn at its own poloidal angle, so all toroidal positions
collapse onto that plane. It carries its own legend and:

- The eight ringed coils are the policy's **three** output banks drawn as the
  coil cross-sections this plane would cut through: two radial (cyan, left and
  right), two vertical (orange, top and bottom), four shaping (violet, on the
  diagonals). Size and brightness are the command magnitude, printed beside one
  member of each bank. Do not lay the banks out one-per-side: three banks on a
  four-sided frame reads as a missing fourth bank.
- The white dot is the magnetic axis and the pale outline around it is the
  controlled column
- The smooth amber blob inside the column is the tearing-risk proxy
- The faint red outline is the uncontrolled reference. Once it has diverged it
  is held against the wall rather than drawn outside the vessel — it disrupted
  there, it did not leave the machine.

## Reveals

- `passive`: an operating map, magnetic field against heating power. Every
  torus is a separate field integration.
- `guardian`: the trained policy re-tested against instability drives it never
  saw. Every tile integrates its own plasma field and runs its own closed-loop
  confinement evaluation; the wall load in each tile is counted, and the tile
  labels are published in `meta.json` under `reveal_sweep`.

## Scientific boundary

The wave model is an exhibition-scale amplitude equation, not a predictive
tokamak code. Magnetic lines are illustrative helical confinement geometry, not
a solved tokamak equilibrium, q-profile, MHD equilibrium, or disruption
calculation. The guardian mode's control environment is a reduced,
differentiable state space, and its markers are transport-flavoured tracers
with no gyromotion, collision operator or self-consistent field response — not
a gyrokinetic or full-orbit particle code.
