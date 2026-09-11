# Agent notes: Bat vs Moth

- Main frames are the bat's senses only; the lit cave lives in `modes/lit/`.
  Never draw moths directly in the senses view except as echoes or catches.
- "Jamming evolved" means *selective* jamming (near > 35% and near − far >
  15% for three generations). Random or blanket jamming must not count.
- Keep jamming a private benefit (only the jammer's echo becomes a phantom);
  the shared version was measured never to evolve.
- `quiet_start` and the head start are part of the story, not tuning noise:
  without them bats never learn and no arms race appears.
- Re-run `tests/test_neuroevo.py::BatVsMothTests` after any change to the sonar
  model; the hand-coded hunters pin that two ears beat one and that jamming
  protects moths.
- No cuBLAS in the forward pass (see `neuroevo.Population.forward`).
