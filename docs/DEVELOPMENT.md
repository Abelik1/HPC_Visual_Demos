# Development guide

## Adding a demo

`docs/DEMO_TEMPLATE.md` is the house style and the checklist;
`leonardo_demos/demos/_template.py` is a skeleton to copy.
`docs/IMPROVEMENTS.md` explains why each rule is there.

## Add a new visual effect without breaking the science

Keep these layers separate:

1. **simulation state** - arrays/particles/fields;
2. **headless renderer** - converts state to a frame;
3. **frame data** - small JSON readouts with no presentation markup;
4. **web presentation** - HTML overlays, mode controls, transitions and zoom.

The main JPEG stream must contain simulation imagery only. Put labels,
counters, legends and control panels in the viewer so visitors can switch them
independently and the same numerical frame can support different stories.

Never change a scientific state merely because a camera transition looks better. Change how it is rendered.

## Performance workflow

Use `local` until the story is visually convincing. Then measure:

```bash
python run_demo.py DEMO --profile desktop --frames 20
```

Only then test `hpc`.

Profiles are presets, not hard-coded modes. Every numeric key present for a
demo in `config/profiles.json` is published by `/api/specs`, rendered as an
editable number field, validated by `/api/run/<demo>`, and persisted in the
run's `meta.json`. Keep the same keys in all four profiles. Add explanatory
labels in `web/app.js` when the raw key would be unclear. From the CLI, use
`--setting key=value`; reserve `--param key=value` for visitor-facing
scientific inputs from `config/demo_specs.json`.

For array demos, forcing CPU/GPU is useful. The CLI flag, the environment
variable and the viewer's Compute selector all feed the same request:

```bash
python run_demo.py reaction_diffusion --profile desktop --frames 20 --backend cpu
python run_demo.py reaction_diffusion --profile desktop --frames 20 --backend gpu
LEONARDO_DEMO_BACKEND=cupy python run_demo.py reaction_diffusion --profile desktop --frames 20
```

Check what the machine actually offers:

```bash
python -c "from leonardo_demos.backend import probe; import json; print(json.dumps(probe(), indent=2))"
```

A demo that uses another framework must route its device through
`backend.torch_device()` so a CPU run really stays on the CPU.

## Frame count

For event playback, 60–120 scientific states is normally enough. The browser can transition between them smoothly. Increasing solver resolution is usually more meaningful than writing 1,000 nearly identical JPEGs.

## Testing

Run:

```bash
python -m unittest discover -s tests -v
```

The smoke tests intentionally use tiny profiles and only verify that output contracts remain valid.
