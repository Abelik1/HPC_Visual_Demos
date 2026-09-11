# Galaxy collision

## Purpose

Shows a restricted N-body Milky Way–Andromeda encounter using physical masses,
distances, velocities, and a camera that follows the evolving system.

## Implementation map

- Solver and renderer: `leonardo_demos/demos/galaxy_collision.py`
- Optional reduced M31 catalogue: `data/m31_catalog_reduced.npz`
- Scientific parameters: preset, impact, speed, tilt, Milky Way mass, and
  Andromeda mass
- Editable profile settings: tracer-particle count, solver substeps, simulated
  Gyr span, and reveal ensemble size
- Reveal: a transverse-velocity uncertainty sweep

## Scientific boundary

Galaxy centres drive the restricted potential while tracer stars expose tidal
structure. This is not a self-consistent live dark-matter simulation; use the
separate `galaxy_collision_3d` demo for direct super-particle gravity.

## Performance

Because tracers never pull on the galaxy centres, the centres' orbit is
integrated first (FP64, host) and tabulated per force stage; one fused CUDA
kernel then advances every tracer through a whole frame's substeps in
registers. The CPU path advances disjoint tracer chunks on the assigned cores.
Both honour `fp32` or `fp64` precision; see `docs/PERFORMANCE.md`.

## Units

Working units are kpc, km/s, and solar masses. Simulation time converts using
1 kpc/(km/s) = 0.97779 Gyr; retain that conversion in displayed values.
