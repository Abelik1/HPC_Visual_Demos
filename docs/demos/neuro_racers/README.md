# Neuro-Racers

## Purpose

A visitor builds a car's neural network from blocks (sensors, hidden-layer
bricks, controls) within a 40-point budget. A population of cars with exactly
that architecture, but different weights, is evolved on a race track by
selection, crossover and mutation. The "show every search" view re-runs the
same brain as independent evolutions from different random starts.

## Implementation map

- Solver and renderer: `leonardo_demos/demos/neuro_racers.py`
- Brain validation and batched evolution: `leonardo_demos/neuroevo.py`
- Brain diagram overlay: `leonardo_demos/neuro_render.py`
- Block catalogue, presets and budget: `config/demo_specs.json` → `neuro_racers.brain`
- Builder UI: `web/brain_builder.js`; animated replay: `web/arena_view.js`;
  viewer wiring: `web/games.js`; styles: `web/games.css`
- Parameters: track, mutation strength, random start
- Main frames: track plus the latest generation's trails; champion in gold
- `overlays/network/`: the champion's real weights and activations
- `interactive/arena.json` + `interactive/gen_NNNN.json`: recorded trajectories
  for the animated replay (`meta.arena_view.frames` maps frame → generation)
- `champion.npz`: best genome and brain spec, used by later ghost races
- `meta.summary`: track, best lap and brain cost for the leaderboard

## Running a saved design headlessly

```bash
python run_demo.py neuro_racers --profile hpc --frames 60 --brain my_brain.json --param track=3
```

`my_brain.json` is the builder's spec, e.g.
`{"sensors":[{"block":"ray","angle":0},{"block":"speed"}],"hidden":[8],"actions":["steer","throttle"]}`.

## Visual language

- Gold trail and car: champion of the generation shown
- Blue trails: next-best cars; red crosses: crashes
- Violet: ghosts, i.e. earlier visitors' champions re-simulated on this track

## Scientific boundary

Reduced top-down driving model, not vehicle dynamics. See
`docs/SCIENTIFIC_NOTES.md`.
