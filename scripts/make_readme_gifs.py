"""Encode the README GIFs from the curated demo-day showcase runs.

Unlike ``generate_showcase.py`` this renders nothing: it cuts a clip out of a
saved run that is already on disk (the large runs picked for each demo's
showcase in ``/demo``) and encodes it. ``runs/`` is not in git, so this only
works on a machine that has those runs - export them with
``tools/export_runs.py`` if needed.

    python scripts/make_readme_gifs.py            # every GIF
    python scripts/make_readme_gifs.py fluid      # just one
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

# GIF name -> (saved run, first frame, last frame, step). Clips skip frames
# rather than play every one, so every GIF is ~100 frames (a few MB).
CLIPS = {
    "fusion_plasma": ("fusion_plasma_A_showcase7654", 0, 599, 6),
    "plasma_guardian": ("fusion_plasma_B_showcase7654", 0, 599, 6),
    "neuro_racers": ("neuro_racers_A_showcase", 0, 449, 4),
    "bat_vs_moth": ("bat_vs_moth_A_showcase", 0, 449, 4),
    "galaxy_collision_3d": ("galaxy_collision_3d_A_showcase7653", 0, 599, 6),
    # The network is nearly converged after ~100 steps; show the learning.
    "neural_wall": ("neural_wall_A_showcase", 0, 99, 1),
    "black_hole": ("black_hole_exact_B_showcase", 0, 1199, 12),
    "fluid": ("fluid_B_showcase7655", 0, 599, 6),
    # The web forms in the first third; the rest is slow sharpening.
    "cosmic_web": ("cosmic_web_B_showcase", 0, 299, 3),
    "galaxy_collision": ("galaxy_collision_A_showcase", 0, 899, 9),
}


def make_clip(run_dir: Path, output: Path, first: int, last: int, step: int, width: int, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to create the README GIFs")
    count = (last - first) // step + 1
    graph = (
        f"select='not(mod(n\\,{step}))',setpts=N/({fps}*TB),"
        f"scale={width}:-2:flags=lanczos,split[frames][palette_in];"
        "[palette_in]palettegen=max_colors=128:stats_mode=diff[palette];"
        "[frames][palette]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    subprocess.run(
        [
            ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-start_number", str(first),
            "-i", str(run_dir / "frames" / "frame_%04d.jpg"),
            "-frames:v", str(count),
            "-filter_complex", graph,
            "-loop", "0",
            str(output),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("names", nargs="*", help=f"GIFs to encode (default: all of {', '.join(CLIPS)})")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs")
    parser.add_argument("--gif-dir", type=Path, default=ROOT / "docs" / "assets" / "demos")
    parser.add_argument("--width", type=int, default=560)
    parser.add_argument("--fps", type=int, default=12)
    args = parser.parse_args()

    unknown = set(args.names) - set(CLIPS)
    if unknown:
        parser.error(f"unknown GIF name(s): {', '.join(sorted(unknown))}")

    args.gif_dir.mkdir(parents=True, exist_ok=True)
    for name in args.names or CLIPS:
        run_id, first, last, step = CLIPS[name]
        run_dir = args.runs_dir / run_id
        if not (run_dir / "frames").is_dir():
            raise SystemExit(f"{name}: saved run {run_id} not found under {args.runs_dir}")
        output = args.gif_dir / f"{name}.gif"
        make_clip(run_dir, output, first, last, step, args.width, args.fps)
        print(f"{output.name:26s} {output.stat().st_size / 1e6:5.1f} MB  <- {run_id}", flush=True)


if __name__ == "__main__":
    main()
