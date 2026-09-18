# Molecular Machine: what to improve

Back in the Discoverer lineup as of 18 September 2026.

**Status (18 September 2026):** B, D and a first version of A are built:
* the sequence builder and HP-style fold with charges (B);
* the reveal dropped, a rotatable 3-D view, and a fused CUDA kernel (D);
* the molecular shuttle as a second method (A).

See docs/demos/molecular_dynamics/README.md. Still open: a walking motor or
rotary motor, and C (GROMACS). The rest of this note is the original analysis.

## What it is today

`leonardo_demos/demos/molecular_dynamics.py`: a chain of 56–72 beads (four
bead types) joined by harmonic bonds, with a softened Lennard-Jones attraction
between all pairs. A velocity-rescaling thermostat holds the temperature. Each
frame is a Pillow ball-and-stick render, and the run ends on a 3×3–4×4 mosaic of
smaller chains at different temperatures and stickiness.

What holds it back on the stand:

1. **It is not a machine.** A chain collapsing into a blob is polymer physics.
   Nothing turns, walks, pumps or switches, so the name promises more than the
   picture delivers.
2. **Nothing to make.** Every other Discoverer demo has a visitor-creative hook
   (draw the obstacle, build the brain, pick the black hole). Here there are four
   sliders, and "sequence" is a number from 0 to 3.
3. **It ends in the scale reveal.** The "virtual laboratory" mosaic is the
   population-tiles story we dropped for physics demos
   (see [DEMO_MODE.md](DEMO_MODE.md)). `/demo` hides it, but it still costs
   compute on every run.
4. **Too small for the machine it runs on.** It builds dense N×N arrays, so it
   cannot scale past a few thousand beads. On a GB200 that is milliseconds of
   work. The 3-D galaxy's tiled all-pairs kernel solved exactly this.
5. **Flat picture.** No rotatable 3-D view, unlike the galaxy and the fusion torus.
6. **Loose physics.** Forces are clipped at 75, the thermostat is ad hoc, and there
   is no solvent. That is fine for an illustration, but there is no honest number
   to quote (no energy, no simulated nanoseconds).

## Directions

### A. Make it a real molecular *machine* (recommended flagship)

The 2016 Chemistry Nobel (Sauvage, Stoddart, Feringa) was for exactly this, and
it is a story visitors can follow in one sentence.

* **Molecular shuttle (rotaxane).** A ring threaded on an axle with two sticky
  "stations". The visitor flips a switch (acid/base or light) to make the other
  station stickier, and the ring hops across, pushed by nothing but thermal
  jiggling. Coarse-grained, cheap, and very readable on screen.
* **Walking motor (kinesin-like).** Two "feet" on a track, and an ATP-fuel
  control. With fuel it walks one way; with a load slider it stalls or gets
  dragged backwards. This is the Brownian ratchet idea: noise plus asymmetry plus
  energy gives directed motion.
* **Rotary motor (ATP synthase or Feringa's light-driven rotor).** A visitor
  presses "flash light" or "open proton flow" and watches a rotor turn in one
  direction only.

Headline control: fuel or switch. Readout: net steps or turns versus random
back-steps. The creative hook: visitors design the track or the station
pattern.

### B. "Fold your own protein" (best visitor hook, smallest change)

Keep the chain, but make it the classic **HP model**: water-hating (H) and
water-loving (P) beads. Visitors **build the sequence bead by bead** in a strip
under the picture, the same way the brain builder works. The chain then buries
its H beads in a core. Presets could mimic real fast-folding mini-proteins
(Trp-cage, villin headpiece) as recognisable targets. Readouts: size of the
hydrophobic core, radius of gyration, "folded or not".

This fixes points 1, 2 and 6 cheaply and reuses almost all of the current solver.

### C. The real-code version (the MUrB pattern)

Just as NBody runs the actual MUrB code, run **GROMACS** (installed on both
EuroHPC systems as a module) on a small protein in water. Chignolin or Trp-cage
fold in microseconds and fit a demo-day job. Import the trajectory and render it
here. The "Run on Discoverer" button now makes this a one-click flow: submit,
wait, fetch, play. The honest story: *this is the software that runs on the
machine for real.*

### D. Whichever you pick: quick wins

* Drop the mosaic reveal. If temperature matters, run **replica exchange**,
  where copies at different temperatures swap. That is parallelism the algorithm
  itself needs, so it is honest. Show it only as a selectable view.
* Tiled all-pairs kernel (from `galaxy_collision_3d`), to go from ~70 to ~50,000
  beads on one GB200, with a thin "solvent" haze.
* A 3-D rotatable view (reuse `galaxy3d_view.js`), with colour by bead type and
  the contact map in a strip **below** the picture, never over it.
* Real units on the readouts: simulated time, energy, pair evaluations per second.

## Suggested order

1. **D (drop the reveal, 3-D view) + B (sequence builder)** for demo day: about
   a day of work, and all local.
2. **A (shuttle or walker)** as a second *method* of the same demo, so the
   presenter can switch from "fold" to "machine".
3. **C (GROMACS)** if the Discoverer module works: the strongest "real HPC"
   story, and the cluster-run pipeline already exists.
