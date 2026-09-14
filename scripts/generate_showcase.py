from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from run_demo import run  # noqa: E402


# Showcase entries are slugs, not demo ids: a demo with selectable modes
# contributes one GIF per mode so the README can show both.
ENTRIES = {
    "fusion_plasma": ("fusion_plasma", "passive"),
    "plasma_guardian": ("fusion_plasma", "guardian"),
}

DEMOS = (
    "black_hole",
    "pbh",
    "fluid",
    "cosmic_web",
    "galaxy_collision",
    "galaxy_collision_3d",
    "reaction_diffusion",
    "crystal",
    "neural_wall",
    "fusion_plasma",
    "weather_ensemble",
    "molecular_dynamics",
    "plasma_guardian",
    "neuro_racers",
    "bat_vs_moth",
)


def entry(slug):
    """Map a showcase slug to the (demo id, solver method) that renders it."""
    return ENTRIES.get(slug, (slug, "default"))


def completed_run(run_dir: Path, profile: str, frames: int, method: str) -> bool:
    """Return True only for a complete run with the requested frame contract."""
    meta_path = run_dir / "meta.json"
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    images = sorted((run_dir / "frames").glob("frame_*.jpg"))
    return (
        meta.get("status") == "complete"
        and meta.get("profile") == profile
        and meta.get("frames") == frames
        and (method == "default" or meta.get("method") == method)
        and len(images) == frames
    )


def make_gif(frame_dir: Path, output: Path, width: int, fps: int) -> None:
    """Encode all numbered JPEGs into a compact, looping, palette GIF."""
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to create the showcase GIFs")
    output.parent.mkdir(parents=True, exist_ok=True)
    graph = (
        f"fps={fps},scale={width}:-2:flags=lanczos,split[frames][palette_in];"
        "[palette_in]palettegen=max_colors=128:stats_mode=diff[palette];"
        "[frames][palette]paletteuse=dither=sierra2_4a:diff_mode=rectangle"
    )
    result = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frame_dir / "frame_%04d.jpg"),
            "-filter_complex",
            graph,
            "-loop",
            "0",
            str(output),
        ],
        text=True,
        capture_output=True,
    )
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or f"ffmpeg exited with {result.returncode}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Render and encode the README demo showcase")
    parser.add_argument("--profile", default="desktop")
    parser.add_argument("--frames", type=int, default=150)
    parser.add_argument("--backend", default="auto")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs" / "showcase_desktop_150")
    parser.add_argument("--gif-dir", type=Path, default=ROOT / "docs" / "assets" / "demos")
    parser.add_argument(
        "--demo",
        action="append",
        choices=DEMOS,
        help="render only this showcase entry (repeatable); defaults to the complete showcase",
    )
    parser.add_argument("--force", action="store_true", help="rerender completed runs")
    args = parser.parse_args()

    if args.frames < 1 or args.width < 2 or args.fps < 1:
        parser.error("frames, width and fps must be positive")

    selected = tuple(args.demo or DEMOS)
    if len(selected) > 1:
        # CuPy and PyTorch can load different CUDA runtime components. Keeping
        # each demo in a fresh interpreter avoids carrying allocator/runtime
        # state from an array solver into a later neural workload.
        for index, demo in enumerate(selected, start=1):
            print(f"\n=== showcase {index}/{len(selected)}: {demo} ===", flush=True)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--profile",
                args.profile,
                "--frames",
                str(args.frames),
                "--backend",
                args.backend,
                "--width",
                str(args.width),
                "--fps",
                str(args.fps),
                "--runs-dir",
                str(args.runs_dir),
                "--gif-dir",
                str(args.gif_dir),
                "--demo",
                demo,
            ]
            if args.force:
                command.append("--force")
            result = subprocess.run(command)
            if result.returncode:
                raise SystemExit(result.returncode)
        return

    for index, slug in enumerate(selected, start=1):
        demo, method = entry(slug)
        run_dir = args.runs_dir / slug
        print(f"\n[{index}/{len(selected)}] {slug}", flush=True)
        if args.force or not completed_run(run_dir, args.profile, args.frames, method):
            run(
                demo,
                profile=args.profile,
                frames=args.frames,
                backend=args.backend,
                run_dir=run_dir,
                method=method,
            )
        else:
            print("render already complete; reusing frames", flush=True)

        frames = sorted((run_dir / "frames").glob("frame_*.jpg"))
        if len(frames) != args.frames:
            raise RuntimeError(f"{slug}: expected {args.frames} frames, found {len(frames)}")
        gif_path = args.gif_dir / f"{slug}.gif"
        make_gif(run_dir / "frames", gif_path, args.width, args.fps)
        size_mib = gif_path.stat().st_size / (1024 * 1024)
        print(f"wrote {gif_path.relative_to(ROOT)} ({size_mib:.1f} MiB)", flush=True)


if __name__ == "__main__":
    main()
