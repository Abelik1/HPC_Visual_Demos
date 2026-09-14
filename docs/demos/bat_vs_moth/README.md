# Bat vs Moth

## Purpose

Two visitors each build a brain from blocks: one for a sonar-hunting bat, one
for a moth. Hundreds of caves run at once, each with one bat and a few moths.
Bats are selected for catches and for flying at moths they can hear; moths for
surviving and keeping their distance. Moths start evolving only after a head
start, so the bats become worth countering and an arms race can appear: moths
may evolve *selective jamming*, clicking only when a bat is near, as real
tiger moths do.

## Implementation map

- Solver and renderer: `leonardo_demos/demos/bat_vs_moth.py`
- Shared engine: `leonardo_demos/neuroevo.py` (`Population`, `validate_brains`)
- Block catalogues (one per role), budgets and presets:
  `config/demo_specs.json` → `bat_vs_moth.brain.roles.{bat,moth}`
- Builders: two `BrainBuilder`s from `web/brain_builder.js`, wired in `web/games.js`
- Parameters: cave layout, moths per cave, mutation strength
- Main frames: **Bat's senses**, meaning only what the champion bat's calls
  revealed: call rings, rock edges they reached, moth echoes, jamming phantoms
- `modes/lit/`: the whole cave lit, with moths, jamming rings and catches
- `overlays/network/`: both champion brains; `overlays/arms_race/`: catch rate
  and near/far jamming per generation, with the head start shaded
- `interactive/`: champion-cave hunt per generation for the animated arena,
  which has its own senses/lit toggle
- Reveal: the final bats and moths released into new procedural caves,
  continuing to co-evolve independently in each

## Tuning notes (why the numbers are what they are)

- Loudness is linear in distance and echo range is 6 units, so hearing clearly
  beats wandering: a hand-coded two-eared bat catches ~68% of moths, a deaf
  one ~6–18%.
- Bats are rewarded for *aiming* at the nearest audible moth, not for being
  near it, so a lucky start position is not selected for.
- Jamming replaces only the jammer's own echo (private benefit). When it
  scrambled every moth's echo, the benefit was shared and jamming never
  evolved.
- Moths start with jam/dive outputs biased off (`quiet_start`); random
  jamming otherwise stops bats from ever learning.
- Tournament size 5 and a 2× initial weight scale (`EVOLUTION`) let hidden
  layers learn in tens of generations; reflex brains learn fastest, which is
  why the default bat preset has no hidden layer.
- Moths hear calls further (9 units) than bats hear echoes (6), and remember a
  call for a few steps, so they can react while the bat keeps calling.

## Compute

The NumPy `CaveSim.run` step is the reference implementation. On CUDA the same
step runs as four fused CuPy kernels (`_cuda_kernels()`): senses, the bat and
moth forward passes (hand-written, no cuBLAS), and one thread per cave for
flight, echoes with jamming phantoms, hearing, dives and catches. Phantom and
dive noise still come from the seeded host RNG in the same order, and
`tests/test_neuroevo.py` checks both paths give the same catches and fitness.
The array-by-array CuPy version was launch-bound at ~20 ms/step at any size.

Measured on an RTX 3060 Ti (Windows, default presets, 4 moths per cave):

| caves  | NumPy ms/step | fused CUDA ms/step |
|-------:|--------------:|-------------------:|
|    160 |           1.3 |               0.20 |
|  1,024 |           3.3 |               0.27 |
|  4,096 |             — |               0.45 |
|  8,192 |             — |               0.68 |
| 32,768 |             — |               2.25 |

CUDA is faster at every size, so Auto uses it whenever it is available. The
`desktop` profile runs 4,096 caves and the `hpc` profile 32,768.

## Scientific boundary

A reduced exhibition sonar model, not acoustics. See
`docs/SCIENTIFIC_NOTES.md`.
