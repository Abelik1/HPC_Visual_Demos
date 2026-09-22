"""Why HPC: each demo's physics timed on 1 / 2 / 4 GPUs and 1 … N cores of one node.

Produces the tables the stand's "Why HPC" view shows.  Only the physics is
timed (no drawing, no JPEG): 3 untimed warm-up calls, then the median of 5,
as in NBody-EuroHPC's BENCHMARK_RESULTS.md.  A point whose single call would
take longer than the budget is left empty rather than run (Alberto capped
``cpu+naive`` the same way).

Run from this PC; everything executes on the cluster:

    python tools/scaling_sweep.py submit discoverer              # the three jobs
    python tools/scaling_sweep.py submit discoverer --pilot      # smallest size, 1 repetition
    python tools/scaling_sweep.py collect discoverer             # fetch -> benchmarks/scaling/discoverer/

On a node (what the jobs call):

    python tools/scaling_sweep.py run --machine discoverer --kind gpu|cpu|murb-gpu|murb-cpu

Kinds: ``gpu`` = the Python demos on the first 1, 2 and 4 GPUs of one job;
``cpu`` = the same demos on 1, 8, 32 and all cores; ``murb-gpu`` = MUrB
``gpu+tile+full`` on 1 GPU and its MPI backend on 2 and 4 (one GPU per rank);
``murb-cpu`` = MUrB ``cpu+omp`` with 1 … N OpenMP threads.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import platform
import re
import shlex
import socket
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

WARMUP, REPS = 3, 5
BUDGET_S = 60.0            # skip a point whose single call is predicted to exceed this
GPU_COUNTS = (1, 2, 4)
OUT = ROOT / "benchmarks" / "scaling"

# MUrB builds and MPI launch per machine (see docs/NBODY_MURB.md).
MURB = {
    "discoverer": {"cuda": "build-discoverer-cuda", "multi": "build-discoverer-multi",
                   "mpi_module": "openmpi4/gcc/4.1.8", "srun_mpi": ["--mpi=pmix"]},
    "leonardo": {"cuda": "build-leonardo", "multi": "build-leonardo-multi",
                 "mpi_module": "openmpi/4.1.6--gcc--12.2.0-cuda-12.2", "srun_mpi": []},
}


# ------------------------------------------------------------------ benches --
# Each bench builds one problem and returns (step, work, flops): ``step()`` runs
# one physics step, ``work`` scales the cost between sizes (to predict skips),
# ``flops`` is the analytical FLOP count of one step or None.
def _context(tmp, demo, backend, **kw):
    from leonardo_demos.base import RunContext
    return RunContext(Path(tmp), demo, "hpc", 2, {}, backend, **kw)


def bench_galaxy(ctx, bodies):
    from leonardo_demos.demos.galaxy_collision_3d import GalaxyCollision3DDemo
    xp = ctx.xp
    rng = np.random.default_rng(0)
    p = xp.asarray(rng.normal(0, 5, (bodies, 3)).astype(np.float32))
    m = xp.asarray(rng.uniform(1e8, 1e9, bodies).astype(np.float32))
    demo = GalaxyCollision3DDemo(ctx, {"force_tile": 160})
    return (lambda: demo.acceleration(p, m, 1.5)), float(bodies) ** 2, 20.0 * bodies * bodies


def bench_black_hole(ctx, size):
    from leonardo_demos.demos.black_hole import BlackHoleDemo
    from leonardo_demos.schwarzschild import EscapeTable
    width, height, supersample = size
    xp = ctx.xp
    rng = np.random.default_rng(1)
    fine = xp.asarray(rng.random((6, 3072, 3072, 3), dtype=np.float32))     # the hpc preset's sky
    glow = xp.asarray(rng.random((360, 720, 3), dtype=np.float32))
    demo = BlackHoleDemo(ctx, {"width": width, "height": height, "supersample": supersample, "fov_deg": 80})
    table = EscapeTable(8.0)
    axis, fwd, right, up = np.array([0, 0, 1.]), np.array([0, 0, -1.]), np.array([1., 0, 0]), np.array([0, 1., 0])
    return (lambda: demo.render_camera(fine, glow, table, axis, fwd, right, up, 8.0),
            float(width * height * supersample ** 2), None)


def bench_racers(ctx, cars):
    from leonardo_demos.demos.neuro_racers import NeuroRacersDemo, RaceSim, track_geometry
    from leonardo_demos.neuroevo import Population
    demo = NeuroRacersDemo(ctx, {})
    demo.full_drives = True
    demo.catalogue_data = demo.catalogue()
    demo.brain = demo.brain_spec(demo.catalogue_data)
    demo.track = track_geometry(0)
    sim = RaceSim(ctx.xp, demo.track, demo.brain, demo.catalogue_data)
    population = Population(ctx.xp, cars, demo.brain["layer_sizes"], seed=5)
    return (lambda: demo.simulate(sim, population, 1500, 3)), float(cars), None


def bench_fluid(ctx, size):
    from leonardo_demos.demos.fluid import FluidDemo
    nx, ny = size
    demo = FluidDemo(ctx, {"nx": nx, "ny": ny})
    state = list(demo.init(nx, ny, .04, 1, None))

    def step():
        state[0] = demo.step(state[0], state[1], state[2], state[3], .04, 20)[0]
    step.per_call = 20                      # timed as 20 lattice steps, reported per step
    return step, float(nx * ny), None


# demo -> (bench, sizes, row label, unit of one step)
BENCHES = {
    "fluid": (bench_fluid, [(3840, 2160), (7680, 4320), (15360, 8640)],
              lambda s: f"{s[0]:,}×{s[1]:,} cells", "one lattice-Boltzmann step"),
    "galaxy_collision_3d": (bench_galaxy, [100_000, 500_000, 1_000_000],
                            lambda n: f"{n:,} bodies", "one all-pairs force evaluation"),
    "black_hole": (bench_black_hole, [(3840, 2160, 3), (7680, 4320, 4), (15360, 8640, 4)],
                   lambda s: f"{s[0]}×{s[1]}, {s[2] ** 2} rays/pixel", "one ray-traced frame"),
    "neuro_racers": (bench_racers, [32_768, 262_144, 1_048_576],
                     lambda n: f"{n:,} cars", "one generation (1,500 steps)"),
}
MURB_SIZES = [100_000, 200_000, 500_000]


# ------------------------------------------------------------------ timing --
def measure(step, sync, pilot=False):
    """Median seconds of one ``step()`` after warm-up; one call if a call is slow."""
    t0 = time.perf_counter(); step(); sync(); first = time.perf_counter() - t0
    warmup, reps = (0, 1) if pilot or first > 20 else (WARMUP - 1, REPS)
    for _ in range(warmup):
        step(); sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter(); step(); sync(); times.append(time.perf_counter() - t0)
    return statistics.median(times), reps


def sweep(counts, sizes, run_point, work, pilot=False):
    """{count: [cell per size]}; stops a column once the next size is predicted over budget."""
    table = {}
    for count in counts:
        cells, last = [], None
        for size in sizes[:1] if pilot else sizes:
            if last and last[0] * work(size) / work(last[1]) > BUDGET_S:
                cells.append(None)
                continue
            cell = run_point(count, size)
            cells.append(cell)
            last = (cell["seconds"], size) if cell else last
            print(f"  {count:>4} → {size}: {cell}", flush=True)
        table[str(count)] = cells
    return table


def run_python(kind, demos, cores, pilot):
    from leonardo_demos.multigpu import Devices
    results = {}
    for demo in demos:
        bench, sizes, label, unit = BENCHES[demo]
        print(f"== {demo} ({kind})", flush=True)

        def point(count, size):
            with tempfile.TemporaryDirectory() as tmp:
                ctx = _context(tmp, demo, "gpu" if kind == "gpu" else "numpy",
                               **({"method": "leapfrog"} if demo == "galaxy_collision_3d" else
                                  {"method": "schwarzschild"} if demo == "black_hole" else {}))
                if kind == "gpu":
                    import cupy as cp
                    ctx._devices = Devices(cp, list(range(count)))
                    sync = ctx.devices.synchronize
                else:
                    ctx.cpu_workers = count
                    sync = lambda: None
                try:
                    step, _, flops = bench(ctx, size)
                    seconds, reps = measure(step, sync, pilot)
                    seconds /= getattr(step, "per_call", 1)
                except Exception as exc:                    # e.g. out of memory at the largest size
                    print(f"  {count} × {size}: {type(exc).__name__}: {exc}", flush=True)
                    return None
                finally:
                    ctx.shutdown_compute()
            cell = {"seconds": seconds, "ms": round(seconds * 1000, 3), "repetitions": reps}
            if flops:
                cell["gflops"] = round(flops / seconds / 1e9, 1)
            return cell

        work = lambda size: bench_work(demo, size)
        counts = GPU_COUNTS if kind == "gpu" else cores
        results[demo] = {"unit": unit, "sizes": [label(s) for s in (sizes[:1] if pilot else sizes)],
                         "columns": sweep(counts, sizes, point, work, pilot)}
    return results


def bench_work(demo, size):
    if demo == "galaxy_collision_3d":
        return float(size) ** 2
    if demo == "black_hole":
        return float(size[0] * size[1] * size[2] ** 2)
    if demo == "fluid":
        return float(size[0] * size[1])
    return float(size)


def run_murb(kind, machine, cores, pilot):
    from leonardo_demos.demos.nbody_murb import NBODY_REPO
    cfg = MURB[machine]

    def murb(args, env=None, launcher=()):
        out = subprocess.run([*launcher, *args], capture_output=True, text=True,
                             env={**os.environ, **(env or {})}, timeout=3600)
        found = re.search(r"average_ms_per_iteration=([0-9.eE+-]+)", out.stdout)
        if not found:
            print(out.stdout[-800:], out.stderr[-800:], flush=True)
            return None
        return float(found.group(1)) / 1000

    def point(count, n):
        iterations = 3 if kind == "murb-cpu" else 10
        common = ["-n", str(n), "-i", "1" if pilot else str(iterations), "--warmup", "1" if pilot else "3", "--nv"]
        if kind == "murb-cpu":
            seconds = murb([str(NBODY_REPO / cfg["cuda"] / "bin" / "murb"), *common, "--im", "cpu+omp"],
                           env={"OMP_NUM_THREADS": str(count), "OMP_PROC_BIND": "close", "OMP_PLACES": "cores"})
        elif count == 1:
            seconds = murb([str(NBODY_REPO / cfg["cuda"] / "bin" / "murb"), *common, "--im", "gpu+tile+full"],
                           launcher=["srun", "--exact", "--ntasks=1", "--gpus-per-task=1", "--cpus-per-task=8"])
        else:
            launch = ["srun", "--exact", *cfg["srun_mpi"], f"--ntasks={count}", "--gpus-per-task=1",
                      "--cpus-per-task=8"]
            seconds = murb([str(NBODY_REPO / cfg["multi"] / "bin" / "murb"), *common, "--im", "gpu+multinode"],
                           env={"OMPI_MCA_btl": "^openib"}, launcher=launch)
        if seconds is None:
            return None
        return {"seconds": seconds, "ms": round(seconds * 1000, 3), "repetitions": 1,
                "gflops": round(20.0 * n * n / seconds / 1e9, 1)}

    print(f"== nbody_murb ({kind})", flush=True)
    counts = GPU_COUNTS if kind == "murb-gpu" else cores
    sizes = MURB_SIZES[:1] if pilot else MURB_SIZES
    return {"nbody_murb": {"unit": "one simulation step (MUrB, median over the timed iterations)",
                           "sizes": [f"{n:,} bodies" for n in sizes],
                           "columns": sweep(counts, MURB_SIZES, point, lambda n: float(n) ** 2, pilot)}}


# -------------------------------------------------------------- provenance --
def provenance(machine, kind):
    info = {"machine": machine, "kind": kind, "node": socket.gethostname(), "architecture": platform.machine(),
            "date": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
            "slurm_job_id": os.getenv("SLURM_JOB_ID"), "git_revision": os.getenv("SWEEP_GIT_REV"),
            "cpus_allocated": os.getenv("SLURM_CPUS_PER_TASK") or os.getenv("SLURM_CPUS_ON_NODE")}
    marker = ROOT / ".dashboard_code"
    if marker.exists():
        info["code_digest"] = marker.read_text().strip()
    try:
        model = subprocess.run(["lscpu"], capture_output=True, text=True).stdout
        info["cpu_model"] = re.search(r"Model name:\s*(.+)", model).group(1).strip()
    except Exception:
        pass
    if kind in ("gpu", "murb-gpu"):
        try:
            names = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                   capture_output=True, text=True).stdout.split("\n")
            info["gpu_model"] = next(n.strip() for n in names if n.strip())
        except Exception:
            pass
    return info


def run(args):
    machine, kind = args.machine, args.kind
    cores = [int(c) for c in args.cores.split(",")] if args.cores else None
    if kind in ("cpu", "murb-cpu") and not cores:
        full = int(os.getenv("SLURM_CPUS_PER_TASK") or os.cpu_count())
        cores = sorted({c for c in (1, 8, 32, full) if c <= full})
    demos = args.demos.split(",") if args.demos else list(BENCHES)
    tables = (run_murb(kind, machine, cores, args.pilot) if kind.startswith("murb")
              else run_python(kind, demos, cores, args.pilot))
    folder = OUT / machine
    folder.mkdir(parents=True, exist_ok=True)
    stamp = provenance(machine, kind)
    for demo, table in tables.items():
        path = folder / f"{demo}.{kind}{'.pilot' if args.pilot else ''}.json"
        path.write_text(json.dumps({**table, "demo": demo, "provenance": stamp}, indent=2))
        print(f"wrote {path}", flush=True)


# ------------------------------------------------------------ from this PC --
def job_script(c, machine, kind, pilot):
    q = shlex.quote
    root = c["root"]
    full = int((c.get("cpu_node") or c.get("gpu_node") or {}).get("cpus", 32))
    shape = {"gpu": ["--ntasks=1", "--cpus-per-task=16", "--gres=gpu:4", "--mem=256G"],
             "cpu": ["--ntasks=1", f"--cpus-per-task={full}", "--mem=256G"],
             "murb-gpu": ["--ntasks=4", "--cpus-per-task=8", "--gpus-per-task=1", "--mem=256G"],
             "murb-cpu": ["--ntasks=1", f"--cpus-per-task={full}", "--mem=64G"]}[kind]
    account = c.get("account") if kind in ("gpu", "murb-gpu") else (c.get("cpu_account") or c.get("account"))
    partition = c.get("partition") if kind in ("gpu", "murb-gpu") else (c.get("cpu_partition") or c.get("partition"))
    qos = c.get("qos") if kind in ("gpu", "murb-gpu") else (c.get("cpu_qos") or c.get("qos"))
    log = f"{root}/benchmarks/scaling/{machine}/slurm-{kind}-%j.log"
    lines = ["#!/bin/bash", f"#SBATCH --job-name=why-hpc-{kind}", "#SBATCH --nodes=1",
             f"#SBATCH --time={'00:20:00' if pilot else '01:30:00'}", f"#SBATCH --output={log}",
             f"#SBATCH --account={account}"]
    lines += [f"#SBATCH --qos={qos}"] if qos else []
    lines += [f"#SBATCH --partition={partition}"] if partition else []
    lines += [f"#SBATCH {flag}" for flag in shape] + [f"#SBATCH {x}" for x in c.get("sbatch_extra") or []]
    lines += list(c.get("job_setup") or [])
    if kind == "murb-gpu":
        lines.append(f"module load {MURB[machine]['mpi_module']}")
    rev = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True, cwd=ROOT).stdout.strip()
    lines += ["export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1",
              f"export SWEEP_GIT_REV={q(rev + ('+local changes' if dirty else ''))}",
              f"cd {q(root)}",
              f"{q(c['python'])} tools/scaling_sweep.py run --machine {machine} --kind {kind}"
              f"{' --pilot' if pilot else ''}"]
    return "\n".join(lines) + "\n"


def submit(args):
    from leonardo_demos import remote
    c = remote.cluster(args.machine)
    remote.sync_code(c, say=print)
    remote.run_remote(c, f"mkdir -p {shlex.quote(c['root'])}/benchmarks/scaling/{args.machine}")
    for kind in args.kinds.split(","):
        script = job_script(c, args.machine, kind, args.pilot)
        out = remote.run_remote(c, "sbatch --parsable", stdin=script.encode())
        print(f"{kind}: job {out.strip()}")


def collect(args):
    from leonardo_demos import remote
    c = remote.cluster(args.machine)
    folder = OUT / args.machine
    folder.mkdir(parents=True, exist_ok=True)
    listing = remote.run_remote(c, f"cd {shlex.quote(c['root'])}/benchmarks/scaling/{args.machine} && ls *.json 2>/dev/null || true")
    for name in listing.split():
        text = remote.run_remote(c, f"cat {shlex.quote(c['root'])}/benchmarks/scaling/{args.machine}/{shlex.quote(name)}")
        (folder / name).write_text(text)
        print(f"fetched {name}")
    merge(folder)


def merge(folder):
    """Combine a demo's per-kind files into benchmarks/scaling/<machine>/<demo>.json."""
    parts = {}
    for path in sorted(folder.glob("*.*.json")):
        demo, kind = path.name.split(".")[:2]
        if path.name.endswith(".pilot.json"):
            continue
        parts.setdefault(demo, {})[kind] = json.loads(path.read_text())
    for demo, kinds in parts.items():
        gpu = kinds.get("gpu") or kinds.get("murb-gpu")
        cpu = kinds.get("cpu") or kinds.get("murb-cpu")
        first = gpu or cpu
        (folder / f"{demo}.json").write_text(json.dumps({
            "demo": demo, "machine": folder.name, "unit": first["unit"], "sizes": first["sizes"],
            "gpus": gpu["columns"] if gpu else {}, "cores": cpu["columns"] if cpu else {},
            "provenance": {"gpu": gpu and gpu["provenance"], "cpu": cpu and cpu["provenance"]}}, indent=2))
        print(f"merged {demo}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    r = sub.add_parser("run", help="on a node: time the physics and write JSON")
    r.add_argument("--machine", required=True)
    r.add_argument("--kind", required=True, choices=["gpu", "cpu", "murb-gpu", "murb-cpu"])
    r.add_argument("--demos", help="comma-separated (default: all Python demos)")
    r.add_argument("--cores", help="comma-separated core counts (default 1,8,32,all)")
    r.add_argument("--pilot", action="store_true", help="smallest size, one repetition")
    s = sub.add_parser("submit", help="from this PC: sync code and submit the jobs")
    s.add_argument("machine")
    # CPU columns only where CPU runs make sense: MUrB's compiled OpenMP backend.
    # The Python demos are GPU codes and get GPU columns only.
    s.add_argument("--kinds", default="gpu,murb-gpu,murb-cpu")
    s.add_argument("--pilot", action="store_true")
    c = sub.add_parser("collect", help="from this PC: fetch results and merge per demo")
    c.add_argument("machine")
    args = ap.parse_args()
    {"run": run, "submit": submit, "collect": collect}[args.command](args)


if __name__ == "__main__":
    main()
