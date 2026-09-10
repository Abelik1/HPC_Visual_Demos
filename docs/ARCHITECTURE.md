# Architecture

## Design goal

The event PC should be a **presentation appliance**, not the scientific compute engine. The simulator can run locally during development, or headlessly on Leonardo. Both produce the same artifact format:

```text
runs/<run-id>/
  meta.json                 # configuration, timing, and execution resources
  reveal.jpg
  frames/
    frame_0000.jpg
    frame_0001.jpg
    ...
  zoom/                 (optional, demos that support deep zoom)
    zoom.json
    L0/0_0.jpg
    L1/0_0.jpg L1/1_0.jpg ...
```

`meta.json` is portable provenance as well as playback state. Every run records
the actual host and architecture, allocated CPU workers, Slurm job/account/
partition/node details when present, visible CUDA device identity and memory
when used, and the final `duration_seconds`. This lets saved Leonardo and
Discoverer runs be replayed and compared honestly on the local dashboard.

The browser viewer only needs HTTP access to those files. It can therefore:

- show a run being generated on the same machine;
- show frames copied incrementally from Leonardo;
- replay a previously computed run.

## Why numbered image frames?

For a public exhibition they are deliberately boring and robust:

- no GPU graphics context is required on the HPC node;
- no video encoder is required;
- a partially completed job already has usable output;
- `rsync` can copy completed frames while later frames are still being produced;
- the local viewer can use smooth CSS transitions independent of the simulation timestep;
- the exact same output can be archived and replayed.

Later, high-bandwidth demos can replace JPEG with Zarr/HDF5/NPZ state streaming while preserving the same controller API.

## Deep zoom

Scaling a finished JPEG only ever magnifies pixels, so the viewer's zoom
control could never show detail that was not already in the frame. There are
two sources of imagery, used together.

### Live re-render (unbounded)

`GET /api/zoom/{run}?cx&cy&span&w&h` regenerates the model's geometry for
exactly the window on screen. There is no level count and therefore no depth
limit: magnification is bounded only by double precision. Crystal growth
supports this because its geometry is continuous; the generator is lazy and
pruned to the window, so cost stays roughly flat as the zoom deepens.

Two properties of the generator make this work (see `crystal_growth.generate`):

- **Scale-free branching.** Every segment spawns two half-length *carriers*
  covering its own halves. Carriers draw nothing; they re-apply the branching
  rule at half scale, and again, so side branches exist at every scale along
  every segment. Without them a segment was a bare line between two branch
  points and magnifying there showed a straight line forever.
- **Window pruning and best-first expansion.** Nodes are pruned against their
  own segment, and expanded in order of distance to the window, so the segment
  budget is spent on what is actually on screen.

### Tile pyramid (playback fallback)

Runs also bake a shallow pyramid: level `L` is a `2^L x 2^L` grid, every tile an
independent render, described by `zoom.json`. It paints instantly while a live
render is in flight, and it is the *only* source when the endpoint is
unavailable — a run synced from Leonardo onto a machine with no solver still
zooms, just to a finite depth. The exhibition machine therefore keeps working
as a pure playback device.

`leonardo_demos/deepzoom.py` takes any `render(cx, cy, span, size)` callback,
so adding the pyramid to another demo means supplying a windowed renderer;
grid-based demos would need their state stored rather than re-derived.

The browser side is `web/deepzoom.js`. Note that it must disable the legacy
CSS-transform zoom handlers on `.screenWrap` while it is active: the canvas is
a child of that element, so pointer events bubbled into the old handler, which
took pointer capture and left panning stuck on with no button held.

## Compute backend

Array-oriented demos use a small NumPy-compatible backend layer:

- NumPy on a normal machine.
- CuPy on CUDA when available.

The neural-network wall additionally detects PyTorch and performs genuine
network training when possible. The ensemble is held as stacked weight tensors
and trained with a single batched forward/backward plus a per-network learning
rate, so scaling the wall is a matmul-shape change rather than a longer Python
loop. It has a deterministic visual fallback, clearly labelled as *not*
training, so the exhibition UI remains testable before a CUDA environment is
installed.

## Step budgets

Several demos take a `total_steps` budget from the profile rather than deriving
work from `frames * steps_per_frame`. Physical maturity and frame count are
different things: a Gray-Scott pattern needs O(10^4) updates and a coordinate
network O(10^3) optimiser steps regardless of how many frames you want to
render. Tying them together meant asking for fewer frames silently produced an
unfinished simulation.

## Multi-GPU scaling

This first repository is intentionally **single-process first**. Get each demo stable and visually excellent before introducing MPI. The natural multi-GPU decompositions are:

- black hole: split output pixels / observer ensemble;
- PBH: split independent parameter batches;
- fluid: spatial domain decomposition with halo exchange;
- cosmic web: domain-decomposed particle mesh or multiple realisations;
- galaxy collision: independent encounters or domain-decomposed particles;
- reaction diffusion: independent parameter tiles, or spatial decomposition for a single huge field;
- crystal: independent environmental conditions;
- neural wall: distribute model batches across ranks/GPUs.
- fusion plasma: independent operating points, or spatial decomposition of the periodic lattice;
- weather ensemble: independent forecast members with different admissible initial conditions;
- molecular dynamics: independent trajectories/candidates, or distributed pair-force evaluation.

For an event, **ensemble parallelism is often preferable** because it makes the scaling story visible and dramatically reduces inter-GPU communication.

## Story layer

Every demo follows the same presentation grammar:

1. Wonder: show one phenomenon.
2. Control: visitor changes one intuitive parameter.
3. Compute: the run begins.
4. Result: the phenomenon becomes visible.
5. Reveal: "Actually… that was not the whole computation."
6. Scale: show many independent cases or the hidden grid/domain.
7. Leonardo: map that work to GPUs.
8. Return: zoom back into one result.

The `reveal.jpg` is generated by the scientific module, not by the web UI, so most demos reveal **real additional simulations**, not duplicated decorative tiles.
