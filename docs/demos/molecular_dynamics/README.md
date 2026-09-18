# Molecular Machine

## Purpose

Two things molecules do, as two solver methods of one demo:

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

## Implementation map

- Solver and renderer: `leonardo_demos/demos/molecular_dynamics.py`
- Parameters (`config/demo_specs.json`): temperature (both); sequence,
  attraction, water strength and salt (fold); switch strength and switch
  interval (shuttle). A visitor-written sequence arrives as `RunReq.chain` and
  becomes `_chain`.
- Presets (`config/profiles.json`): chain length `particles`, `total_steps`
  (fold), `shuttle_steps`, bending stiffness `bend`.
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

## Compute

On a GPU the fold runs as one fused CUDA kernel (`FOLD_KERNEL`). One thread
block holds the chain, and each bead sums the force from every other bead and
advances up to 2000 Langevin steps per launch. On an RTX 3060 Ti that is about
15 µs per step at 40 beads and 40 µs per step at 240 beads (the HPC preset,
1.5 M steps in about a minute). Chains of more than 1024 beads, and the CPU
backend, use the vectorised NumPy path. The shuttle (12 mobile beads) always
integrates with NumPy and records that in `compute_note`.

## Scientific boundary

Beads stand for groups of atoms. This is an illustrative HP-style model and a
cartoon rotaxane, not a force field. The time steps are reduced units, not
femtoseconds, and nothing here predicts a real protein's structure.
