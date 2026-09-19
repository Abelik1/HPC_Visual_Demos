"""MUrB N-body: the external NBody-EuroHPC C++/CUDA code, driven from the dashboard.

Unlike the other demos this one does not simulate in Python. It launches the
real ``murb`` executable from the NBody-EuroHPC repository with ``--record``,
reads the ``.murbtraj`` trajectory it writes (format: TRAJECTORY_FORMAT.md in
that repository) and renders the frames the way MUrB's own OpenGL viewer draws
them: wire-frame spheres on deep-space blue, coloured by speed.

Where the executable comes from, in order:
  1. the MURB_EXE environment variable;
  2. ../NBody-EuroHPC/build-*/bin/murb(.exe) next to this repository.

A trajectory computed on Leonardo can be imported without a local executable:
``python scripts/import_murbtraj.py file.murbtraj``.
"""
from __future__ import annotations

import json, os, re, struct, subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from ..base import Demo

ROOT = Path(__file__).resolve().parents[2]
NBODY_REPO = Path(os.getenv("MURB_REPO", ROOT.parent / "NBody-EuroHPC"))

CPU_METHODS = ("cpu+omp", "cpu+optim", "cpu+simd", "cpu+naive")
GPU_METHODS = ("gpu+tile+full", "gpu+tile")
METHOD_LABELS = {
    "cpu+omp": "CPU · OpenMP (all cores)",
    "cpu+optim": "CPU · optimised, 1 core",
    "cpu+simd": "CPU · SIMD (MIPP), 1 core",
    "cpu+naive": "CPU · naive reference, 1 core",
    "gpu+tile+full": "GPU · CUDA tiled, device-resident",
    "gpu+tile": "GPU · CUDA tiled",
}
METHOD_DESCRIPTIONS = {
    "cpu+omp": "MUrB's O(N²) direct sum parallelised over every CPU core with OpenMP.",
    "cpu+optim": "Single-core O(N²) with the reference loop restructured for the cache.",
    "cpu+simd": "Single-core O(N²) vectorised with the MIPP SIMD library.",
    "cpu+naive": "The single-core reference implementation every other backend is checked against.",
    "gpu+tile+full": "CUDA shared-memory tiles, bodies kept on the GPU for the whole run (the Leonardo A100 path).",
    "gpu+tile": "CUDA shared-memory tiles, with positions copied back to the host each step.",
}
SCHEMES = ("galaxy", "random")

# MUrB's viewer: camera at (0, 0, 5) looking down -z, world units of 1e8 m,
# 45° vertical field of view.  Each extra view is the same camera orbited.
WORLD_SCALE = 1.0e-8
FOV_Y = np.radians(45.0)
VIEWS = (
    ("frames", "MUrB camera", 0.0, 0.0),
    ("side", "Side view", np.radians(90.0), 0.0),
    ("top", "Top view", 0.0, np.radians(90.0)),
)
BACKGROUND = (1, 3, 12)


def find_murb():
    """Path of the murb executable, or None."""
    explicit = os.getenv("MURB_EXE")
    if explicit:
        return Path(explicit) if Path(explicit).is_file() else None
    name = "murb.exe" if os.name == "nt" else "murb"
    candidates = sorted(NBODY_REPO.glob(f"build-*/bin/{name}"),
                        key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def murb_build_info(exe):
    """Parse ``murb --version`` (``revision=... cuda=0 openmp=1 ...``)."""
    if exe is None:
        return {}
    try:
        out = subprocess.run([str(exe), "--version"], capture_output=True, text=True,
                             timeout=20).stdout
    except (OSError, subprocess.SubprocessError):
        return {}
    return dict(re.findall(r"(\w+)=(\S+)", out))


def available_methods():
    """Only offer GPU backends when the executable was built with CUDA."""
    info = murb_build_info(find_murb())
    return CPU_METHODS + (GPU_METHODS if info.get("cuda") == "1" else ())


# ---- .murbtraj version 1 ------------------------------------------------

class Trajectory:
    """Read-only view of a finalised .murbtraj file."""

    def __init__(self, path):
        self.path = Path(path)
        with open(self.path, "rb") as f:
            head = f.read(72)
            if len(head) < 72 or head[:8] != b"MURBTRJ\0":
                raise ValueError(f"{self.path.name} is not a .murbtraj file")
            version, endian, scalar, flags = struct.unpack_from("<4I", head, 8)
            if version != 1 or endian != 0x01020304 or scalar != 1 or flags not in (0, 1):
                raise ValueError(f"unsupported .murbtraj header in {self.path.name}")
            (self.header_bytes, self.n, self.frame_count, self.stride,
             ) = struct.unpack_from("<4Q", head, 24)
            (self.dt,) = struct.unpack_from("<d", head, 56)
            backend_len, commit_len = struct.unpack_from("<2I", head, 64)
            self.backend = f.read(backend_len).decode("utf-8", "replace")
            self.commit = f.read(commit_len).decode("utf-8", "replace")
            self.radii = (np.fromfile(f, dtype="<f4", count=self.n) if flags == 1
                          else np.full(self.n, 1.0e6, dtype=np.float32))
        self.frame_bytes = 8 + 24 * self.n
        expected = self.header_bytes + self.frame_count * self.frame_bytes
        if self.path.stat().st_size != expected:
            raise ValueError(f"{self.path.name} is incomplete or not finalised")

    def iteration(self, index):
        """The solver iteration a recorded frame was taken at (8 bytes read)."""
        with open(self.path, "rb") as f:
            f.seek(self.header_bytes + index * self.frame_bytes)
            return struct.unpack("<Q", f.read(8))[0]

    def frame(self, index):
        """(iteration, positions[N,3], velocities[N,3]) of one recorded frame."""
        offset = self.header_bytes + index * self.frame_bytes
        with open(self.path, "rb") as f:
            f.seek(offset)
            (iteration,) = struct.unpack("<Q", f.read(8))
            data = np.fromfile(f, dtype="<f4", count=6 * self.n).reshape(6, self.n)
        return iteration, data[:3].T.astype(np.float64), data[3:].T.astype(np.float64)


# ---- MUrB-style renderer --------------------------------------------------

def murb_colours(velocities):
    """MUrB's velocity palette (OGLSpheresVisuGS.cpp), without its music strobe.

    Speed² is normalised per frame; slow bodies stay deep-space blue, faster
    ones ramp to cyan and the fastest are white.
    """
    norm = np.einsum("ij,ij->i", velocities, velocities)
    t = (norm - norm.min()) / (norm.max() - norm.min() + 1e-6)
    rgb = np.empty((len(t), 3))
    rgb[:] = (0.0, 0.02, 0.1)
    fast = t > 0.1
    rgb[fast] += np.outer(t[fast], (0.1, 0.9, 1.5))
    rgb[t > 0.8] = (0.8, 1.0, 1.0)
    # Pure navy on navy would vanish in a compressed frame; keep a floor so
    # slow bodies read as dim blue rings, as they do in the OpenGL window.
    rgb = np.maximum(rgb, (0.08, 0.38, 0.80))
    return (np.clip(rgb, 0, 1) * 255).astype(np.uint8)


def render_view(positions, velocities, radii, width, height, yaw, pitch,
                camera_distance=5.0, supersample=2):
    """Draw every body as a small ring, back to front, through MUrB's camera."""
    s = supersample
    w, h = width * s, height * s
    p = positions * WORLD_SCALE
    # Orbit the camera: rotate the world about y (yaw), then about x (pitch).
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    x = p[:, 0] * cy - p[:, 2] * sy
    z = p[:, 0] * sy + p[:, 2] * cy
    y = p[:, 1] * cp - z * sp
    z = p[:, 1] * sp + z * cp
    depth = camera_distance - z
    focal = (h / 2) / np.tan(FOV_Y / 2)
    visible = (depth > 0.1) & (radii > 0)
    sx = w / 2 + focal * x / np.where(visible, depth, 1)
    sy_ = h / 2 - focal * y / np.where(visible, depth, 1)
    rad = focal * radii * WORLD_SCALE / np.where(visible, depth, 1)
    visible &= (sx > -20) & (sx < w + 20) & (sy_ > -20) & (sy_ < h + 20)
    order = np.argsort(-depth[visible])
    idx = np.flatnonzero(visible)[order]
    colours = murb_colours(velocities)

    image = Image.new("RGB", (w, h), BACKGROUND)
    draw = ImageDraw.Draw(image)
    line = max(1, s)
    for i in idx:
        r = max(float(rad[i]), 0.9 * s)
        c = tuple(int(v) for v in colours[i])
        box = (sx[i] - r, sy_[i] - r, sx[i] + r, sy_[i] + r)
        if r < 2.2 * s:
            draw.ellipse(box, fill=c)
        else:
            draw.ellipse(box, outline=c, width=line)
    if s > 1:
        image = image.resize((width, height), Image.LANCZOS)
    return image


def render_trajectory(ctx, traj, frames, width, height, zoom=1.0, extra=None):
    """Render ``frames`` evenly spaced trajectory frames into every view.

    Every view orbits the bodies' starting centre. The distance is that of
    MUrB's fixed camera at (0, 0, 5), so the default view at zoom 1 is exactly
    what its OpenGL window shows, for the offset "random" cloud too.
    """
    extra = dict(extra or {})
    count = int(traj.frame_count)
    if count == 0:
        raise RuntimeError("the trajectory contains no frames")
    _, first, _ = traj.frame(0)
    centre = np.median(first[traj.radii > 0] if (traj.radii > 0).any() else first, axis=0)
    distance = float(np.linalg.norm(np.array([0.0, 0.0, 5.0]) - centre * WORLD_SCALE)) / max(zoom, 1e-3)
    picks = np.unique(np.linspace(0, count - 1, max(1, min(frames, count))).round().astype(int))
    for view_id, _, _, _ in VIEWS[1:]:
        (ctx.run_dir / "modes" / view_id).mkdir(parents=True, exist_ok=True)
    ctx.frames = len(picks)
    ctx.write_meta({
        "frames": len(picks),
        "view_modes": [{"id": v, "label": label, "folder": "frames" if v == "frames" else f"modes/{v}"}
                       for v, label, _, _ in VIEWS],
        "default_view_mode": "frames",
    })
    # MUrB has finished by now, so every allocated core can draw: each worker
    # reads its own frame from the trajectory file (no 100k-body arrays are
    # copied between processes) and draws all three views of it.
    from ..pipeline import FramePipeline
    workers = max(1, min(ctx.cpu_workers - 1, 16, max(1, len(picks) // 4)))
    with FramePipeline(ctx, workers=workers) as pipeline:
      for out, index in enumerate(picks):
        iteration = traj.iteration(int(index))
        payload = {"trajectory": str(traj.path), "index": int(index), "centre": centre, "distance": distance,
                   "width": width, "height": height,
                   "extra_paths": {view_id: str(ctx.run_dir / "modes" / view_id / f"frame_{out:04d}.jpg")
                                   for view_id, _, _, _ in VIEWS[1:]}}
        days = iteration * traj.dt / 86400.0
        pipeline.submit(out, draw_murb_frame, payload, ctx.frame_path(out),
                        f"iteration {iteration:,} · t = {days:.1f} days", {
            "code": "MUrB (NBody-EuroHPC)",
            "backend": traj.backend,
            "bodies": f"{traj.n:,}",
            "iteration": f"{iteration:,}",
            "simulated time": f"{days:.1f} days",
            "timestep": f"{traj.dt:g} s",
            "interactions / step": f"{traj.n * (traj.n - 1):,}",
            **extra,
        })


def draw_murb_frame(p):
    """All three views of one trajectory frame (runs in a worker process).

    The main view is returned for the pipeline to save; the side and top
    views are written here."""
    traj = Trajectory(p["trajectory"])
    _, pos, vel = traj.frame(p["index"])
    main = None
    for view_id, _, yaw, pitch in VIEWS:
        image = render_view(pos - p["centre"], vel, traj.radii, p["width"], p["height"], yaw, pitch, p["distance"])
        if view_id == "frames":
            main = image
        else:
            target = Path(p["extra_paths"][view_id])
            tmp = target.with_name(f".{target.stem}.tmp.jpg")
            image.save(tmp, format="JPEG", quality=92)
            tmp.replace(target)
    return main


class NBodyMurbDemo(Demo):
    id = "nbody_murb"
    title = "MUrB N-body"
    backend_kind = "cpu"          # the executable owns the device, not CuPy
    supported_backends = ("cpu", "gpu")
    methods = available_methods()
    # A cluster's own murb decides what it can run (Leonardo's build has CUDA,
    # this PC's may not): cluster requests are validated against every method
    # and run_demo.py re-checks against the node's build.
    remote_methods = CPU_METHODS + GPU_METHODS
    default_method = "cpu+omp"
    method_labels = {m: METHOD_LABELS[m] for m in methods}
    method_descriptions = {m: METHOD_DESCRIPTIONS[m] for m in methods}

    def run(self):
        ctx, st, p = self.ctx, self.settings, self.ctx.params
        width, height = int(st.get("width", 1280)), int(st.get("height", 720))
        zoom = float(p.get("zoom", 1.0))
        imported = p.get("_trajectory_path")
        if imported:
            traj = Trajectory(imported)
            ctx.set_backend_name(f"MUrB {traj.backend} (imported trajectory)")
            ctx.write_meta({"murb": {"source": "imported", "file": Path(imported).name,
                                     "backend": traj.backend, "commit": traj.commit,
                                     "bodies": traj.n, "recorded_frames": traj.frame_count,
                                     "record_every": traj.stride, "dt": traj.dt}})
            render_trajectory(ctx, traj, ctx.frames, width, height, zoom)
            ctx.finish(None)
            return

        exe = find_murb()
        if exe is None:
            raise RuntimeError(
                f"MUrB executable not found. Build NBody-EuroHPC in {NBODY_REPO} "
                "(see docs/NBODY_MURB.md) or set MURB_EXE to murb(.exe).")
        method = ctx.method
        requested = ctx.backend_requested.lower()
        if requested in {"cpu", "numpy"} and method.startswith("gpu"):
            raise ValueError(f"{method} is a GPU backend; choose Auto or GPU compute")
        if requested in {"gpu", "cuda", "cupy"} and method.startswith("cpu"):
            raise ValueError(f"{method} is a CPU backend; pick a GPU implementation")

        bodies = int(st["bodies"])
        iterations = int(st["iterations"])
        warmup = int(st.get("warmup", 0))
        scheme = SCHEMES[int(round(p.get("scheme", 0))) % len(SCHEMES)]
        dt = float(p.get("dt", 3600))
        record_every = max(1, iterations // max(1, ctx.frames))
        traj_path = (ctx.run_dir / "trajectory.murbtraj").resolve()
        cmd = [str(exe), "-n", str(bodies), "-i", str(iterations), "--im", method,
               "--dt", f"{dt:g}", "--scheme", scheme, "--record", str(traj_path),
               "--record-every", str(record_every), "--gf", "-v"]
        if warmup > 0:
            cmd += ["--warmup", str(warmup)]
        env = dict(os.environ)
        if method == "cpu+omp":
            env.setdefault("OMP_NUM_THREADS", str(ctx.cpu_workers))
        ctx.set_backend_name(f"MUrB {method}")
        ctx.write_meta({"murb": {"source": "local", "executable": str(exe),
                                 "command": cmd[1:], "build": murb_build_info(exe),
                                 "bodies": bodies, "iterations": iterations,
                                 "warmup": warmup, "record_every": record_every,
                                 "scheme": scheme, "dt": dt}})

        report, log = {}, []
        creation = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, env=env, cwd=str(ctx.run_dir),
                              creationflags=creation) as proc:
            for raw in proc.stdout:
                line = raw.strip()
                log.append(line)
                done = re.match(r"completed=(\d+)/(\d+)", line)
                if done:
                    ctx.write_meta({"status": "running", "frame": -1,
                                    "message": f"MUrB {method}: iteration {done.group(1)} / {done.group(2)}",
                                    "elapsed": ctx.elapsed_seconds()})
                if line.startswith("completed_iterations="):
                    report = dict(re.findall(r"(\w+)=(\S+)", line))
            code = proc.wait()
        (ctx.run_dir / "murb.log").write_text("\n".join(log), encoding="utf-8")
        if code != 0:
            tail = " | ".join(l for l in log[-4:] if l)
            raise RuntimeError(f"murb exited with code {code}: {tail}")

        ctx.write_meta({"murb": {**self._meta_murb(), "report": report}})
        extra = {}
        if report.get("interactions_per_second"):
            extra["interactions / s"] = f"{float(report['interactions_per_second']):.3g}"
        if report.get("estimated_GFLOP_per_second"):
            extra["GFLOP/s"] = f"{float(report['estimated_GFLOP_per_second']):.1f}"
        if report.get("average_ms_per_iteration"):
            extra["ms / iteration"] = f"{float(report['average_ms_per_iteration']):.3g}"
        render_trajectory(ctx, Trajectory(traj_path), ctx.frames, width, height, zoom, extra)
        # The rendered frames are the saved run; the raw trajectory is only
        # an intermediate and would dominate the run's size.
        traj_path.unlink(missing_ok=True)
        ctx.finish(None)

    def _meta_murb(self):
        try:
            return json.loads((self.ctx.run_dir / "meta.json").read_text(encoding="utf-8")).get("murb", {})
        except (OSError, ValueError):
            return {}
