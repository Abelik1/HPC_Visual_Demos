# Agent notes: Star in a Bottle

## Both modes

- Write one valid rotatable JSON state for every saved frame.
- Preserve camera orientation while frame state updates during playback.
- Keep 3-D as the default and retain the clean 2-D fallback.
- Keep canvas labels/readouts in HTML overlays, not `fusion_view.js` drawing code.
- Describe magnetic lines as illustrative geometry, never solved equilibrium.
- Verify animation in both plasma and magnetic layers while rotating.
- `passive` is the default method and must keep behaving exactly as it did
  before the two modes were merged. Changing the shared solver changes both.

## Guardian mode

- Main frames contain the vessel state only; sensor values, policy loss and
  policy topology belong to `overlays/`, not to the frame.
- Amber/red is proximity to the wall; sparks are counted wall contacts, never
  decoration. If a frame shows a spark, a marker really left confinement.
- The controlled and uncontrolled populations start from the same seed and the
  same state; the only difference is whether the policy is allowed to act.
- Training happens **between** shots and never during one. A shot must be a
  clean frozen-policy episode, and every shot must face identical conditions -
  same field seed, marker seed, start state and disturbance sequence - or the
  scoreboard stops being a controlled comparison.
- Scoreboard bars are counted losses. Keep the printed value the raw count if
  the bar height is rescaled, and say so on the panel.
- The policy is optimized against a differentiable model of the *marker loss
  mechanism* (`wall_load_terms`), not merely against column displacement. If
  the marker balance in `ConfinedParticles.advance` changes, change that too or
  the policy is optimizing a different machine than the one on screen.
- Never label the analytical fallback as neural training.
- Keep the policy graph and cross-section images synchronized with their frame.
- The coil rings in the torus view and the coil cross-sections in the overlay
  are the same hardware at the same poloidal angles. Change one and change both,
  in `COIL_BANKS` and in `coilSegments` in `web/fusion_view.js`.
- The reveal must stay a genuine re-evaluation across the drive range, and the
  quoted wall load must be the count the evaluation produced.
- Publish the overlay folders the run wrote in `meta.json` (`overlays`) so the
  viewer offers exactly the diagnostics that exist.
