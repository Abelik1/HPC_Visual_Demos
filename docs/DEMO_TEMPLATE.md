# Building a demo

The house style for this repository. Every rule below exists because breaking
it silently produced a demo that looked finished and was not. Copy
`leonardo_demos/demos/_template.py` and work through the checklist at the end.

## The contract

A demo is a class with `id`, `title`, and `run()`. It receives a `RunContext`
and a settings dict from the active profile, and it writes:

```text
runs/<run-id>/
  meta.json            status, params, backend, timing
  frames/frame_NNNN.jpg
  frame_data/frame_NNNN.json  optional live readout values
  modes/<name>/frame_NNNN.jpg optional alternative simulation views
  reveal.jpg
  zoom/                optional tile pyramid
```

Nothing else in the system needs to know what the demo computes. Keep it that
way: the viewer, the SLURM job, the sync script and the run library all work
off this contract alone.

## Four layers, kept separate

1. **Simulation state** — arrays, particles, fields. No PIL, no text, no I/O.
2. **Headless renderer** — state to a frame. No solver calls.
3. **Frame data** — small JSON values, with no HTML or baked-in labels.
4. **Viewer** — real HTML controls/overlays, transitions, playback and zoom.

The numbered frame is the clean simulation viewport. Do not draw titles,
legends, explanatory cards or control panels into it. Alternative scientific
views belong in `modes/`; optional explanatory material belongs in the viewer.

Never change a scientific state because a transition looks better. Change how
it is rendered.

## Rules

### 1. Budget total work, not work per frame

Derive the run from a `total_steps` budget in the profile and distribute it
across frames. Physical maturity and frame count are independent: Gray-Scott
needs O(10⁴) updates and a coordinate network O(10³) optimiser steps whatever
the frame count.

```python
target = int(round(total * (i + 1) / frames))
state = self.step(state, max(1, target - done)); done = target
```

Deriving work from `frames * steps_per_frame` meant a 70-frame reaction-diffusion
run stopped at ~1000 steps with 4% of the domain covered, and a 40-frame neural
wall trained for 80 steps and showed flat blurs. Both looked like rendering
bugs. Keep the `steps_per_frame` fallback so old profiles still load.

### 2. Simulate at the display aspect ratio

The frame is 16:9. Simulate 16:9, or letterbox. Do not resize a square field to
fill the frame — it stretches every feature by 1.78 and turns circular Turing
spots into ellipses and six-fold symmetry into five-and-a-bit.

### 3. Truncation must be symmetric

Any budget-limited generation must degrade evenly. Expanding a recursive
structure depth-first spends the whole budget on the first branch and produces
a visibly lopsided result. Use breadth-first, or a priority queue ordered by
visual significance, and prefer to stop *before* a partial generation.

### 4. Normalisation is not free

`colors.palette` autoscales by default, which is right for a field with a
meaningful dynamic range and wrong for one already in [0, 1]. Renormalising a
legitimately uniform field maps it to zero. Pass `normalize_input=False` when
your data is already scaled.

Watch for the same trap anywhere you normalise against a subset: the crystal's
`birth` times are scaled to the radii present in whatever set was generated, so
a windowed generation put every segment at birth≈1.

### 5. The reveal must be real computation

The reveal is the point of the exhibition: *"that was not the whole
calculation."* It has to be true. Run genuine extra simulations — a parameter
sweep, an ensemble, an observational-uncertainty sweep — and label what varies
across the tiles. A grid of decorative copies of the last frame makes the
claim a lie.

Prefer sweeps that mean something: the galaxy demo sweeps the *measured
uncertainty* on Andromeda's transverse velocity, because that is the number
that actually decides the outcome.

### 6. Say what you are actually doing

Every approximation gets an entry in `docs/SCIENTIFIC_NOTES.md`, and the frame
itself reports the truth. If PyTorch is missing and the neural wall falls back
to a Fourier surrogate, the headline, subtitle and badge all say it is not
training. If a model is procedural geometry rather than a solved PDE, say so —
the crystal is honestly labelled as a recursive geometric model, not a
phase-field solver.

State the physics in real units where you can. Working in kpc, km/s and solar
masses is what lets the galaxy demo put a meaningful clock in Gyr on screen.

### 7. Respect the compute request

Use `self.ctx.xp` for array work so one source runs on NumPy or CuPy, and
`to_numpy()` before anything touches PIL. If your demo uses another framework,
route its device through `backend.torch_device(self.ctx.backend_requested)`.

A run asked to stay on the CPU must stay on the CPU. Silently taking CUDA
because it happened to be present makes the reported backend a lie.

### 8. Write status as you go

Call `ctx.write_status(i, message)` every frame. That is what makes a partially
finished run usable, which is what lets the viewer stream frames rsync'd from
Leonardo while the job is still running.

### 9. Make it seekable, not just watchable

Zoom that magnifies a finished JPEG shows nothing that was not already there.
If your model can be evaluated over an arbitrary window, expose a
`render(cx, cy, span, size)` and you get both the pre-baked pyramid
(`deepzoom.build_pyramid`, works with no solver present) and live re-rendering
(`/api/zoom/{run}`, no depth limit) for free.

For truly unbounded zoom the model needs detail at *every* scale, not just a
few generations — see `crystal_growth.generate` and its carrier trick.

### 10. Costs are a design parameter

Give all three profiles honest numbers. `local` should finish in tens of
seconds on a laptop CPU; `desktop` assumes CUDA; `hpc` is a *starting
target* to benchmark, not a promise. Measure before you raise a resolution —
several demos here spend more time rendering than solving.

## Checklist

- [ ] `id` matches the keys in `demo_specs.json` and `profiles.json`
- [ ] registered in `leonardo_demos/registry.py`
- [ ] settings block in **all three** profiles
- [ ] params in `demo_specs.json` with `min`/`max`/`step`/`value` (the API
      validates against these; anything outside is rejected)
- [ ] work driven by a `total_steps` budget
- [ ] simulation aspect ratio matches the frame
- [ ] all array work through `ctx.xp`, `to_numpy()` before PIL
- [ ] honours `ctx.backend_requested`
- [ ] `ctx.write_status()` every frame, `ctx.finish(reveal)` at the end
- [ ] reveal is real extra computation, and its tiles are labelled
- [ ] story line added to `stories` in `web/app.js`
- [ ] smoke test in `tests/` using a tiny profile
- [ ] entry in `docs/SCIENTIFIC_NOTES.md` stating the approximations
- [ ] timed on `local`; profile numbers reflect reality

## Testing

Smoke tests use deliberately tiny settings and check the output contract, not
the physics:

```python
c = RunContext(Path(t), 'my_demo', 'local', 2, {...}, 'numpy')
MyDemo(c, {'n': 48, 'total_steps': 40, 'ensemble': 4}).run()
self.assertTrue((Path(t) / 'frames/frame_0001.jpg').exists())
self.assertEqual(json.loads((Path(t) / 'meta.json').read_text())['status'], 'complete')
```

Where a bug was subtle, pin the *behaviour* rather than the output — see
`tests/test_deep_zoom.py`, which asserts that deeper windows produce thinner
branches and that a deep render is not a flat field.

```bash
.venv/Scripts/python.exe -m unittest discover -s tests
```
