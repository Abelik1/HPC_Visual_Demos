"""Render the demo-day showcase runs on this PC, then collect Discoverer's.

Runs one simulation at a time (they all share one GPU), in the order below,
and is safe to stop and start again: a run whose meta.json says "complete" is
skipped, anything else is deleted and redone. Every finished run is starred and
added to its demo's showcase in runs/_library.json, so demo mode plays it when
the stand is idle.

When the local queue is done it keeps checking Discoverer for the six big
showcase runs (fluid, fusion_plasma, galaxy_collision_3d), copies each one here
as soon as it is complete, and adds it to the showcase too.

    python scripts/render_showcase_desktop.py            # everything
    python scripts/render_showcase_desktop.py --only neuro_racers
    python scripts/render_showcase_desktop.py --skip-discoverer
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs"
LOG = RUNS / "_showcase_desktop.log"
LIBRARY = RUNS / "_library.json"
SCRATCH = RUNS / "_showcase_inputs"

REMOTE = "abelik@login.brainplusplus.bg"
REMOTE_PORT = "2226"
REMOTE_RUNS = "/weka/ehpc-school-2026/abelik/runs"
DISCOVERER_RUNS = ("galaxy_collision_3d_A", "galaxy_collision_3d_B", "fluid_A", "fluid_B",
                   "fusion_plasma_A", "fusion_plasma_B")

SPECS = json.loads((ROOT / "config" / "demo_specs.json").read_text(encoding="utf-8"))


def brain_file(name: str, spec: dict) -> str:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    path = SCRATCH / f"{name}.json"
    path.write_text(json.dumps(spec), encoding="utf-8")
    return str(path)


racer_presets = SPECS["neuro_racers"]["brain"]["presets"]
bat_roles = SPECS["bat_vs_moth"]["brain"]["roles"]

# (run name, demo, command-line arguments). Two variants per demo, differing in
# their advanced settings as well as their scientific parameters.
QUEUE = [
    # Exact Schwarzschild ray tracing on the Gaia sky. A circles Gaia BH3 once
    # at a comfortable distance; B dives from far out towards Gaia BH1.
    ("black_hole_exact_A", "black_hole", [
        "--profile", "hpc", "--frames", "1200", "--method", "schwarzschild",
        "--setting", "width=1920", "--setting", "height=1080", "--setting", "supersample=3",
        "--setting", "sky_face=3072", "--setting", "fov_deg=80",
        "--param", "target=0", "--param", "camera_distance=12", "--param", "orbit=360", "--param", "dive=0"]),
    ("black_hole_exact_B", "black_hole", [
        "--profile", "hpc", "--frames", "1200", "--method", "schwarzschild",
        "--setting", "width=2560", "--setting", "height=1440", "--setting", "supersample=2",
        "--setting", "sky_face=3072", "--setting", "fov_deg=95", "--setting", "glow=1.3",
        "--param", "target=1", "--param", "camera_distance=30", "--param", "orbit=90", "--param", "dive=0.8"]),

    ("cosmic_web_A", "cosmic_web", [
        "--profile", "hpc", "--frames", "800",
        "--setting", "grid=1536", "--setting", "particles=5898240", "--setting", "total_steps=12000",
        "--setting", "sweep_steps=10", "--setting", "ensemble=1",
        "--param", "recipe=0", "--param", "seed=42", "--param", "gravity=0.8", "--param", "helium=0.24",
        "--param", "expanding_space=1", "--param", "dark_energy=1", "--param", "warm_dark_matter=0",
        "--param", "_parallel_count=1"]),
    ("cosmic_web_B", "cosmic_web", [
        "--profile", "hpc", "--frames", "800",
        "--setting", "grid=2048", "--setting", "particles=10485760", "--setting", "total_steps=14000",
        "--setting", "ic_amplitude=1.05", "--setting", "sweep_steps=10", "--setting", "ensemble=1",
        "--param", "recipe=0", "--param", "seed=7", "--param", "gravity=1.0", "--param", "helium=0.3",
        "--param", "expanding_space=1", "--param", "dark_energy=1", "--param", "warm_dark_matter=1",
        "--param", "_parallel_count=1"]),

    ("galaxy_collision_A", "galaxy_collision", [
        "--profile", "hpc", "--frames", "900", "--method", "leapfrog",
        "--setting", "particles=1000000", "--setting", "substeps=16", "--setting", "span_gyr=8.0",
        "--setting", "ensemble=1",
        "--param", "impact=0.55", "--param", "speed=0.75", "--param", "tilt=18",
        "--param", "milky_way_mass=1.5", "--param", "andromeda_mass=1.5", "--param", "_parallel_count=1"]),
    ("galaxy_collision_B", "galaxy_collision", [
        "--profile", "hpc", "--frames", "900", "--method", "leapfrog",
        "--setting", "particles=1200000", "--setting", "substeps=20", "--setting", "span_gyr=9.0",
        "--setting", "ensemble=1",
        "--param", "impact=0.3", "--param", "speed=1.1", "--param", "tilt=55",
        "--param", "milky_way_mass=1.2", "--param", "andromeda_mass=2.2", "--param", "_parallel_count=1"]),

    ("neural_wall_A", "neural_wall", [
        "--profile", "hpc", "--frames", "400",
        "--setting", "tile=128", "--setting", "networks=64", "--setting", "total_steps=30000",
        "--param", "_target_path=" + str(ROOT / "data" / "compression_images" / "hubble_deep_field.jpg"),
        "--param", "difficulty=1.0", "--param", "fourier=1", "--param", "_parallel_count=64"]),
    ("neural_wall_B", "neural_wall", [
        "--profile", "hpc", "--frames", "400",
        "--setting", "tile=96", "--setting", "networks=128", "--setting", "total_steps=40000",
        "--param", "_target_path=" + str(ROOT / "data" / "compression_images" / "star_in_a_bottle.webp"),
        "--param", "difficulty=1.0", "--param", "fourier=1", "--param", "_parallel_count=128"]),

    ("neuro_racers_A", "neuro_racers", [
        "--profile", "hpc", "--frames", "450",
        "--setting", "population=4096", "--setting", "generations=150", "--setting", "sim_steps=1500",
        "--setting", "record_cars=64", "--setting", "ensemble=16",
        "--setting", "reveal_population=512", "--setting", "reveal_generations=40",
        "--param", "track=3", "--param", "mutation=0.5", "--param", "seed=7", "--param", "_parallel_count=16",
        "--brain", brain_file("racers_balanced", {k: racer_presets["balanced"][k] for k in ("sensors", "hidden", "actions")})]),
    ("neuro_racers_B", "neuro_racers", [
        "--profile", "hpc", "--frames", "450",
        "--setting", "population=8192", "--setting", "generations=120", "--setting", "sim_steps=1800",
        "--setting", "record_cars=64", "--setting", "ensemble=25",
        "--setting", "reveal_population=512", "--setting", "reveal_generations=40",
        "--param", "track=1", "--param", "mutation=0.35", "--param", "seed=21", "--param", "_parallel_count=25",
        "--brain", brain_file("racers_big", {k: racer_presets["big"][k] for k in ("sensors", "hidden", "actions")})]),

    ("bat_vs_moth_A", "bat_vs_moth", [
        "--profile", "hpc", "--frames", "450",
        "--setting", "population=8192", "--setting", "generations=300", "--setting", "hunt_steps=900",
        "--setting", "moth_head_start=100", "--setting", "ensemble=9",
        "--setting", "reveal_population=512", "--setting", "reveal_generations=30",
        "--param", "cave=11", "--param", "moths=4", "--param", "mutation=0.45", "--param", "_parallel_count=9"]),
    ("bat_vs_moth_B", "bat_vs_moth", [
        "--profile", "hpc", "--frames", "450",
        "--setting", "population=16384", "--setting", "generations=250", "--setting", "hunt_steps=1000",
        "--setting", "moth_head_start=60", "--setting", "ensemble=16",
        "--setting", "reveal_population=1024", "--setting", "reveal_generations=30",
        "--param", "cave=523", "--param", "moths=6", "--param", "mutation=0.6", "--param", "_parallel_count=16",
        "--brain", brain_file("batmoth_deluxe_jammer", {
            "bat": {k: bat_roles["bat"]["presets"]["deluxe"][k] for k in ("sensors", "hidden", "actions")},
            "moth": {k: bat_roles["moth"]["presets"]["jammer"][k] for k in ("sensors", "hidden", "actions")}})]),
]


def log(message: str) -> None:
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {message}"
    print(line, flush=True)
    with LOG.open("a", encoding="utf-8") as handle:
        handle.write(line + "\n")


def status(run_dir: Path) -> str | None:
    try:
        return json.loads((run_dir / "meta.json").read_text(encoding="utf-8")).get("status")
    except (OSError, ValueError):
        return None


# A new showcase run that supersedes an older one takes its place in the
# showcase; the old run stays on disk and in the history.
SUPERSEDES = {"black_hole_exact_A_showcase": "black_hole_A_showcase",
              "black_hole_exact_B_showcase": "black_hole_B_showcase"}


def add_to_showcase(run_id: str, demo: str) -> None:
    """Star the run and append it to its demo's showcase (three at most)."""
    try:
        data = json.loads(LIBRARY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = {}
    favourites = [r for r in data.get("favourites", []) if r != run_id]
    data["favourites"] = [run_id] + favourites
    showcase = data.setdefault("showcase", {})
    picks = [r for r in showcase.get(demo, [])
             if r not in (run_id, SUPERSEDES.get(run_id)) and (RUNS / r).is_dir()]
    showcase[demo] = (picks + [run_id])[-3:]
    tmp = LIBRARY.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    tmp.replace(LIBRARY)


def render_local(names: set[str] | None) -> None:
    python = sys.executable
    for name, demo, args in QUEUE:
        if names and demo not in names and name not in names:
            continue
        run_dir = RUNS / f"{name}_showcase"
        if status(run_dir) == "complete":
            log(f"skip {name}: already complete")
            add_to_showcase(run_dir.name, demo)
            continue
        if run_dir.exists():
            shutil.rmtree(run_dir)
        log(f"start {name}")
        started = time.time()
        with (RUNS / f"_showcase_{name}.log").open("w", encoding="utf-8") as out:
            code = subprocess.call([python, "run_demo.py", demo, "--backend", "gpu", "--timings",
                                    "--run-dir", str(run_dir), *args],
                                   cwd=ROOT, stdout=out, stderr=subprocess.STDOUT)
        minutes = (time.time() - started) / 60
        if code == 0 and status(run_dir) == "complete":
            add_to_showcase(run_dir.name, demo)
            log(f"done  {name} in {minutes:.1f} min, added to the {demo} showcase")
        else:
            log(f"FAILED {name} after {minutes:.1f} min (exit {code}); see runs/_showcase_{name}.log")


def remote(command: str) -> str:
    return subprocess.run(["ssh", "-p", REMOTE_PORT, "-o", "BatchMode=yes", "-o", "ConnectTimeout=30", REMOTE, command],
                          capture_output=True, text=True, timeout=300).stdout


def fetch_run(directory: str) -> int:
    """Stream one remote run here as a compressed tar.

    A run can hold thousands of per-frame JSON files (the fusion torus is ~3 GB),
    which scp copies one round trip at a time. The run is unpacked beside the
    others under a temporary name and only renamed into place once complete, so
    the viewer never lists a half-copied run.
    """
    import tarfile
    staging = RUNS / f"_incoming_{directory}"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    proc = subprocess.Popen(["ssh", "-p", REMOTE_PORT, "-o", "BatchMode=yes", REMOTE,
                             f"tar -czf - -C {REMOTE_RUNS} {directory}"], stdout=subprocess.PIPE)
    try:
        with tarfile.open(fileobj=proc.stdout, mode="r|gz") as archive:
            archive.extractall(staging, filter="data")
    except (tarfile.TarError, OSError) as error:
        log(f"transfer of {directory} broke off: {error}")
        proc.kill()
    code = proc.wait()
    if code != 0 or status(staging / directory) != "complete":
        shutil.rmtree(staging, ignore_errors=True)
        return code or 1
    local = RUNS / directory
    if local.exists():
        shutil.rmtree(local)
    (staging / directory).rename(local)
    shutil.rmtree(staging, ignore_errors=True)
    return 0


def collect_discoverer(max_hours: float) -> None:
    """Copy each finished Discoverer showcase run here, newest job per name."""
    wanted = set(DISCOVERER_RUNS)
    deadline = time.time() + max_hours * 3600
    while wanted and time.time() < deadline:
        try:
            listing = remote(
                f"cd {REMOTE_RUNS} && for d in *_showcase[0-9]*; do [ -f \"$d/meta.json\" ] && "
                "printf '%s %s\\n' \"$d\" \"$(python3 -c 'import json,sys;print(json.load(open(sys.argv[1])).get(\"status\"))' \"$d/meta.json\")\"; done")
        except (subprocess.SubprocessError, OSError) as error:
            log(f"Discoverer not reachable ({error}); retrying in 10 min")
            time.sleep(600)
            continue
        latest = {}
        for row in listing.strip().splitlines():
            parts = row.split()
            if len(parts) != 2:
                continue
            directory, state = parts
            base = directory.rsplit("_showcase", 1)[0]
            if base in wanted:
                job = int(directory.rsplit("_showcase", 1)[1])
                if base not in latest or job > latest[base][0]:
                    latest[base] = (job, directory, state)
        for base, (job, directory, state) in sorted(latest.items()):
            if state != "complete":
                continue
            local = RUNS / directory
            if status(local) != "complete":
                log(f"copying {directory} from Discoverer")
                code = fetch_run(directory)
                if code != 0 or status(local) != "complete":
                    log(f"copy of {directory} failed (exit {code}); will retry")
                    continue
            demo = json.loads((local / "meta.json").read_text(encoding="utf-8"))["demo"]
            add_to_showcase(directory, demo)
            log(f"collected {directory} into the {demo} showcase")
            wanted.discard(base)
        if wanted:
            log(f"waiting for Discoverer: {', '.join(sorted(wanted))}")
            time.sleep(600)
    if wanted:
        log(f"gave up waiting for Discoverer runs: {', '.join(sorted(wanted))}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--only", nargs="*", help="demo ids or run names to render")
    parser.add_argument("--skip-local", action="store_true")
    parser.add_argument("--skip-discoverer", action="store_true")
    parser.add_argument("--discoverer-hours", type=float, default=48)
    a = parser.parse_args()
    RUNS.mkdir(exist_ok=True)
    os.environ.setdefault("PYTHONUNBUFFERED", "1")
    if not a.skip_local:
        render_local(set(a.only) if a.only else None)
    if not a.skip_discoverer:
        collect_discoverer(a.discoverer_hours)
    log("showcase queue finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
