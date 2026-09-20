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

  **The budget is spread geometrically, not evenly** (`training_plan`). The
  policy needs tens of updates, not hundreds: measured from scratch on the
  exhibition PC, one update takes marker losses from 81,871 to 80,017, twenty
  reach 41,882, sixty reach 2,477 and six hundred only 1,502. So an even split
  of a cluster-sized budget spends enough in the *first* gap alone to solve the
  task, and every later shot then scores the same — which is what the 19 Sep
  Leonardo run did (1,267 lost at shot 2 and 1,273 at shot 10, with 1,500
  updates split evenly). Each gap now trains about as much as all the training
  before it: 1,500 updates over nine gaps become 2, 5, 11, 26, 58, 131, 295,
  666, 1,500, which keeps the steep part of the learning curve spread over the
  early shots and still spends the whole budget by the last one. Flying those
  nine controllers through the same scored shot gives 81,871 → 78,537 →
  74,227 → 72,851 → 15,628 → 2,796 → 1,750 → 1,566 → 1,505 → 1,452: every shot
  better than the one before, where the even split gave one cliff and nine
  identical bars.

  **A run can continue an earlier one.** Every shot's controller is saved, and
  a new run can start from the last one instead of from an untrained network
  (`↻` on a saved run in the dashboard, `--resume` on the command line, or
  `resume_from` in the API). The new run trains on top of what it inherited and
  records where it came from in `meta.json` under `resumed_from`. Because the
  policy converges well before a cluster-sized budget is spent, continuing a
  *finished* run mostly buys the last few per cent; the feature is there to
  carry training across machines and sessions, not to escape the plateau.

  Every shot is the *identical* experiment: same field seed, same marker seed,
  same start state, same disturbance sequence. Only the policy differs, so the
  scoreboard is a controlled comparison. A typical local run goes 616 → 92 →
  55 → 73 → 15 → 12 markers lost against a constant no-control reference of
  571; the small regression is real, not smoothed away.

  The floor is the model's own edge transport: turbulent kicks are
  edge-weighted and the well-controlled population sits at about 0.7 of the
  wall radius, so losses become rare but never stop. No amount of training
  removes them.

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
