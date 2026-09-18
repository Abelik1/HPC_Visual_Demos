# Agent notes: Molecular Machine

- Describe the model as coarse-grained, not atomistic.
- Keep `fold_forces()` (NumPy/CuPy) and `FOLD_KERNEL` (fused CUDA) the same
  force field; change them together and compare Rg / buried-oil on both backends.
- Keep `preset_text()` and `ChainBuilder.preset()` in `web/chain_builder.js`
  identical.
- The shuttle's binding is normalised per station-pair; check that the ring
  follows each switch flip (trips ≈ flips) after any change to it.
- No reveal grid: this is a physics demo (see docs/DEMO_MODE.md).
- Treat chain-length increases as O(N²) performance changes; the fused kernel
  needs N ≤ 1024.
