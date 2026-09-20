# Molecular Machine

## Purpose

Two things molecules do, as four solver methods of one demo:

* **Fold your own protein** (`fold`, default): a chain of beads, each one
  water-avoiding (H), water-loving (P), positive (+) or negative (-). Visitors
  write the sequence bead by bead (`web/chain_builder.js`) or start from a
  preset: Oily core, Hairpin, Charge zipper, Soluble. The oily beads hide in a
  core, opposite charges zip together, and a soluble chain stays a floppy coil.
* **Molecular shuttle** (`shuttle`): a rotaxane. A ring is threaded on an axle
  that has two binding stations and fat stoppers at the ends. A switch flips
  which station is sticky every *N* frames. The ring is never pushed: it
  diffuses on thermal kicks alone and is caught by the sticky station. This is
  the kind of machine the 2016 Chemistry Nobel was awarded for.

* **Walking motor** (`walker`): a flashing Brownian ratchet, the textbook
  mechanism behind motor proteins such as kinesin. Two feet joined by a
  springy leg sit on a lopsided sawtooth track (wells every 4 units, the
  steep barrier 0.8 units ahead of each well, 8 kT deep). Fuel flashes the
  track off and on; while it is off the feet diffuse freely, and being caught
  again is lopsided, so noise turns into steps forward. A load pulls back:
  0.3 already stalls it and 1.0 drags it backwards. No fuel, no motion.

* **Rotary motor** (`rotor`): the F1 part of ATP synthase, the motor every
  cell runs. One overdamped angle in a potential well 40 kT deep; each fuel
  event moves the well 120° on and the rotor follows it, so nothing ever
  pushes it round. Fuel only binds once the rotor has settled within 0.5 rad
  of its site (tight mechanochemical coupling), which is what makes it stall
  rather than burn fuel for nothing. Cargo tilts the landscape: up to about
  0.4 the speed hardly changes, 0.6 loses most steps to futile cycles, 0.8
  stalls, and 1.0 flattens the barrier so the cargo unwinds the motor
  backwards. The rotor is three-fold symmetric, so a marker bead on its own
  arm shows the rotation — as in the single-molecule experiments.

## Implementation map

- Solver and renderer: `leonardo_demos/demos/molecular_dynamics.py`
- Parameters (`config/demo_specs.json`): temperature (both); sequence,
  attraction, water strength and salt (fold); switch strength and switch
  interval (shuttle); fuel and load (walker); proton flow and cargo (rotor). A visitor-written sequence arrives as `RunReq.chain` and
  becomes `_chain`.
- Presets (`config/profiles.json`): chain length `particles`, `total_steps`
  (fold), `shuttle_steps`, `walker_steps`, `rotor_steps`, bending stiffness
  `bend`.
- Frames: depth-sorted, shaded ball and stick. Each frame's 3-D state is saved
  in `interactive/` as `molecule-3d` JSON, drawn by `web/galaxy3d_view.js`
  ("Rotate in 3D" in both front ends).
- No reveal: the old grid of independent trajectories was dropped (see
  docs/DEMO_MODE.md on the scale reveal).

## Model

Reduced units: bead diameter 1, kT(310 K) = 0.5, Langevin (BAOAB) dynamics.

* Fold: harmonic bonds (k = 180), bending E = k(cos θ - 0.35)², WCA excluded
  volume between every non-bonded pair, an LJ attraction between H beads
  scaled by "water strength", a faint attraction otherwise, and screened
  Debye-Hückel charges whose screening length shrinks with salt.
* Shuttle: rigid axle; a 12-bead ring with bonds, bending and a planarity
  restraint (a real macrocycle is stiff); LJ wells at the stations, normalised
  so a sticky station holds the ring by about 4 kT × drive and an idle one
  barely at all.
* Walker and rotor: one and two overdamped coordinates, Euler-Maruyama. The
  rotor's torque is -depth·sin(θ - site) - load; a step that ends with the
  rotor more than π behind its site is a futile cycle, and the site behind
  takes it back.

## Compute

On a GPU the fold runs as one fused CUDA kernel (`FOLD_KERNEL`). One thread
block holds the chain, and each bead sums the force from every other bead and
advances up to 2000 Langevin steps per launch. On an RTX 3060 Ti that is about
15 µs per step at 40 beads and 40 µs per step at 240 beads (the HPC preset,
1.5 M steps in about a minute). Chains of more than 1024 beads, and the CPU
backend, use the vectorised NumPy path. The shuttle (12 mobile beads) and the
walker (2 feet) and the rotor (one angle) always integrate with NumPy and
record that in `compute_note`; the device plan gives them CPU cores only.

## Scientific boundary

Beads stand for groups of atoms. This is an illustrative HP-style model, a
cartoon rotaxane and textbook ratchet models of the two motors, not a force
field. The motors reproduce the shape of a real torque-speed curve, not any
particular protein's numbers. The time steps are reduced units, not
femtoseconds, and nothing here predicts a real protein's structure.
