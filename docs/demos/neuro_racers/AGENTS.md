# Agent notes: Neuro-Racers

- The block catalogue in `config/demo_specs.json` is the single source of truth
  for blocks, costs and presets; `validate_brain` (API and solver) and the
  builder both read it. Never hard-code a block or price elsewhere.
- Main frames contain only the track and trails; generation, lap times and the
  network live in `frame_data/` and `overlays/network/`.
- The NumPy `RaceSim` code is the reference. The fused CuPy kernels in
  `_cuda_kernels()` must mirror it line for line;
  `tests/test_neuroevo.py::test_fused_cuda_kernels_match_the_array_reference`
  checks that.
- Do not use batched GEMM (`matmul`/`einsum`) on CuPy in `Population.forward`:
  cuBLAS crashes the viewer process (access violation) when PyTorch's CUDA
  libraries are also loaded, which `/api/specs` does.
- Ghosts are re-simulated with their own brain, never replayed from disk.
- The reveal must stay a set of independent island evolutions (`groups`), with
  every tile labelled by its own result.
- Re-run the track self-clearance test after editing any control points.
