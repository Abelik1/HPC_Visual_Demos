"""Neuro-Racers: evolve a population of cars that all share a visitor-built brain.

The visitor chooses which sensors the car has, how many hidden neurons it gets
and which controls it may use.  Every car in the population has that exact
architecture but different weights.  Each generation all cars drive the track
at once (one batched forward pass per time step); the ones that get furthest
become the parents of the next generation.

This is a reduced top-down driving model: a kinematic bicycle with a lateral
grip limit, not a tyre or vehicle-dynamics simulation.  Distance sensors are
sphere-traced through a signed-distance field of the track walls.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..backend import to_numpy
from ..base import Demo
from ..neuroevo import (Population, action_labels, brain_catalogue, brain_payload, input_labels,
                        load_champion, load_checkpoint, save_champion, save_checkpoint, validate_brain)
from ..neuro_render import draw_brain
from ..render import font

WORLD_W, WORLD_H = 16.0, 9.0          # world units; the frame is 16:9 (rule 2)
PX = 80.0                             # pixels per world unit in a 1280x720 frame
HALF_WIDTH = 0.55                     # half track width
SDF_RES = 1 / 24                      # signed-distance grid cell size
DT = 1 / 30                           # seconds per simulation step
V_MAX, A_MAX, COAST, DRAG = 5.2, 4.0, 0.9, 0.12
BRAKE, K_MAX, GRIP, SKID_LOSS = 7.0, 2.6, 7.5, 2.2
CAR_RADIUS = 0.11
RAY_RANGE = {"ray": 3.0, "long_ray": 6.0}
RAY_ITERS = {"ray": 12, "long_ray": 18}
COMPASS_LOOKAHEAD = 1.6

# Closed centre-line control points, world units with y pointing down.
TRACKS = {
    0: ("Oval", [(8 + 5.7 * math.cos(t), 4.5 + 3.0 * math.sin(t))
                 for t in np.linspace(0, 2 * math.pi, 14, endpoint=False)]),
    1: ("Kidney", [(2.3, 4.5), (3.0, 2.0), (5.6, 1.3), (8.0, 2.9), (10.4, 1.3), (13.0, 2.0),
                   (13.7, 4.5), (13.0, 7.0), (10.2, 7.8), (5.8, 7.8), (3.0, 7.0)]),
    2: ("Hairpin", [(2.0, 1.9), (8.0, 1.4), (13.6, 1.6), (14.6, 3.0), (13.4, 4.2), (7.0, 4.3),
                    (6.2, 5.2), (7.2, 6.1), (13.4, 6.2), (14.4, 7.3), (13.0, 8.1), (3.2, 8.0),
                    (1.4, 6.6), (1.2, 3.4)]),
    3: ("Grand Prix", [(1.6, 4.6), (2.0, 1.8), (4.6, 1.1), (6.4, 2.2), (8.0, 3.2), (9.6, 1.6),
                       (12.6, 1.1), (14.5, 2.3), (14.2, 4.5), (12.3, 4.9), (10.9, 5.6), (11.4, 6.7),
                       (13.1, 6.75), (14.5, 7.25), (14.1, 8.15), (12.0, 8.2), (8.6, 7.95), (6.6, 6.5), (4.6, 7.9), (2.4, 7.6)]),
}


def _catmull_rom(points, samples=40):
    p = np.asarray(points, dtype=np.float64)
    n = len(p)
    out = []
    for i in range(n):
        p0, p1, p2, p3 = p[i - 1], p[i], p[(i + 1) % n], p[(i + 2) % n]
        t = np.linspace(0, 1, samples, endpoint=False)[:, None]
        out.append(.5 * (2 * p1 + (-p0 + p2) * t + (2 * p0 - 5 * p1 + 4 * p2 - p3) * t ** 2
                         + (-p0 + 3 * p1 - 3 * p2 + p3) * t ** 3))
    return np.concatenate(out)


@lru_cache(maxsize=8)
def track_geometry(track_id: int, points: int = 800):
    """Uniform arc-length centre line, walls, and a signed-distance grid.

    Cached because the reveal, the ghosts and the tests all reuse a track.
    """
    name, control = TRACKS[int(track_id)]

    def resample(curve):
        closed = np.vstack([curve, curve[:1]])
        seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
        s_curve = np.concatenate([[0], np.cumsum(seg)])
        total = float(s_curve[-1])
        s = np.linspace(0, total, points, endpoint=False)
        return np.stack([np.interp(s, s_curve, closed[:, 0]), np.interp(s, s_curve, closed[:, 1])], axis=1), total

    centre, length = resample(_catmull_rom(control))
    # Smooth along the arc so no corner is tighter than the track is wide;
    # a sharper bend would fold the inner wall back on itself.
    sigma = .38 / (length / points)
    offsets = np.arange(-int(3 * sigma), int(3 * sigma) + 1)
    kernel = np.exp(-.5 * (offsets / sigma) ** 2); kernel /= kernel.sum()
    centre = sum(w * np.roll(centre, -int(o), 0) for o, w in zip(offsets, kernel))
    centre, length = resample(centre)
    tangent = np.roll(centre, -1, 0) - np.roll(centre, 1, 0)
    tangent /= np.linalg.norm(tangent, axis=1, keepdims=True)
    # Start on the straightest stretch so every car begins pointing down a straight.
    heading = np.unwrap(np.arctan2(tangent[:, 1], tangent[:, 0]))
    curvature = np.abs(np.gradient(heading))
    window = max(3, points // 40)
    smooth = np.convolve(np.concatenate([curvature[-window:], curvature, curvature[:window]]),
                         np.ones(window * 2 + 1) / (window * 2 + 1), mode="same")[window:-window]
    start = int(np.argmin(smooth))
    centre, tangent = np.roll(centre, -start, 0), np.roll(tangent, -start, 0)
    normal = np.stack([-tangent[:, 1], tangent[:, 0]], axis=1)
    left, right = centre + normal * HALF_WIDTH, centre - normal * HALF_WIDTH

    ny, nx = int(round(WORLD_H / SDF_RES)), int(round(WORLD_W / SDF_RES))
    gx = (np.arange(nx) + .5) * SDF_RES
    gy = (np.arange(ny) + .5) * SDF_RES
    cells = np.stack(np.meshgrid(gx, gy), axis=-1).reshape(-1, 2)
    a, b = centre, np.roll(centre, -1, 0)
    ab = b - a
    ab_len2 = np.maximum((ab ** 2).sum(1), 1e-12)
    step = length / points
    # Coarse nearest centre-line sample first, then exact point-to-segment
    # distance over the neighbouring segments only.
    coarse = 8
    window = np.arange(-coarse - 2, coarse + 3)
    dist = np.empty(len(cells))
    nearest_s = np.empty(len(cells))
    for lo in range(0, len(cells), 8192):
        c = cells[lo:lo + 8192]
        guess = np.argmin(((c[:, None, :] - centre[None, ::coarse]) ** 2).sum(-1), axis=1) * coarse
        j = (guess[:, None] + window[None, :]) % points
        t = np.clip(((c[:, None, :] - a[j]) * ab[j]).sum(-1) / ab_len2[j], 0, 1)
        d2 = ((a[j] + t[..., None] * ab[j] - c[:, None, :]) ** 2).sum(-1)
        k = np.argmin(d2, axis=1)
        rows = np.arange(len(c))
        dist[lo:lo + 8192] = np.sqrt(d2[rows, k])
        nearest_s[lo:lo + 8192] = ((j[rows, k] + t[rows, k]) * step) % length
    sdf = (HALF_WIDTH - dist).reshape(ny, nx).astype(np.float32)
    return {
        "id": int(track_id), "name": name, "length": length, "centre": centre, "tangent": tangent,
        "left": left, "right": right, "sdf": sdf, "s_grid": nearest_s.reshape(ny, nx).astype(np.float32),
    }


_CUDA_KERNELS = {}


def _cuda_kernels():
    """Fused CuPy kernels mirroring RaceSim's array code line for line.

    The NumPy path is the reference.  On CUDA the same arithmetic launched as
    ~100 tiny elementwise kernels per step is dominated by launch latency, so
    the ray march and the car update each become one kernel.
    ``tests/test_neuroevo.py`` checks the two paths agree.
    """
    if _CUDA_KERNELS:
        return _CUDA_KERNELS
    import cupy as cp
    lookup = f"""
        bool inside = px >= 0.f && px < {WORLD_W}f && py >= 0.f && py < {WORLD_H}f;
        int ix = min(max((int)(px / {SDF_RES!r}f), 0), nx - 1);
        int iy = min(max((int)(py / {SDF_RES!r}f), 0), ny - 1);
    """
    _CUDA_KERNELS["cast"] = cp.ElementwiseKernel(
        "float32 x, float32 y, float32 angle, raw float32 sdf, int32 nx, int32 ny, float32 reach, int32 iters",
        "float32 out",
        f"""
        float dx = cosf(angle), dy = sinf(angle), t = 0.02f;
        for (int k = 0; k < iters; ++k) {{
            float px = x + t * dx, py = y + t * dy;
            {lookup}
            float d = inside ? sdf[iy * nx + ix] : -1.0f;
            if (d > 0.01f) t = t + fmaxf(d, 0.03f);
            t = fminf(t, reach);
        }}
        out = t / reach;
        """, "neuro_racers_cast")
    _CUDA_KERNELS["step"] = cp.ElementwiseKernel(
        "float32 steer, float32 throttle, float32 brake, raw float32 sdf, raw float32 sgrid, "
        "int32 nx, int32 ny, float32 length, int32 step",
        "float32 x, float32 y, float32 h, float32 v, float32 skid, float32 prog, float32 sprev, "
        "int32 lap, int32 crash, bool alive",
        f"""
        if (alive) {{
            float accel = throttle > 0.f ? throttle * {A_MAX}f : throttle * {COAST}f;
            float nv = fminf(fmaxf(v + (accel - {DRAG}f * v - brake * {BRAKE}f) * {DT!r}f, 0.f), {V_MAX}f);
            float curv = steer * {K_MAX}f;
            float lat = nv * nv * fabsf(curv);
            float nskid = fmaxf(lat - {GRIP}f, 0.f) / {GRIP}f;
            if (lat > {GRIP}f) curv = copysignf(1.f, curv) * {GRIP}f / fmaxf(nv * nv, 1e-4f);
            nv = nv * (1.f - {SKID_LOSS}f * fminf(nskid, 1.f) * {DT!r}f);
            float nh = h + nv * curv * {DT!r}f;
            x = x + nv * cosf(nh) * {DT!r}f;
            y = y + nv * sinf(nh) * {DT!r}f;
            h = nh; v = nv; skid = nskid;
        }} else {{ v = 0.f; skid = 0.f; }}
        float px = x, py = y;
        {lookup}
        float snow = sgrid[iy * nx + ix];
        float ds = snow - sprev;
        if (ds < -length / 2.f) ds += length; else if (ds > length / 2.f) ds -= length;
        if (alive) prog = prog + ds;
        sprev = snow;
        if (lap < 0 && prog >= length) lap = step + 1;
        float d = inside ? sdf[iy * nx + ix] : -1.0f;
        if (alive && d < {CAR_RADIUS}f) {{ crash = step + 1; alive = false; }}
        """, "neuro_racers_step")
    return _CUDA_KERNELS


class RaceSim:
    """All cars of one brain architecture driving one track in lockstep."""

    def __init__(self, xp, track, brain, catalogue, fused=None):
        self.xp, self.track, self.brain, self.catalogue = xp, track, brain, catalogue
        # Fused kernels are the CUDA implementation; ``fused=False`` forces the
        # reference array code on CuPy too (used by the equivalence test).
        self.fused = (xp is not np) if fused is None else bool(fused)
        self.sdf = xp.asarray(track["sdf"])
        self.s_grid = xp.asarray(track["s_grid"])
        self.centre = xp.asarray(track["centre"].astype(np.float32))
        self.ny, self.nx = track["sdf"].shape
        self.length = track["length"]
        self.actions = brain["actions"]
        self.rays = {}
        for sensor in brain["sensors"]:
            if sensor["block"] in RAY_RANGE:
                self.rays.setdefault(sensor["block"], []).append(math.radians(sensor["angle"]))
        self.rays = {k: xp.asarray(np.asarray(v, dtype=np.float32)) for k, v in self.rays.items()}

    def _cell(self, x, y):
        xp = self.xp
        ix = xp.clip((x / SDF_RES).astype(xp.int32), 0, self.nx - 1)
        iy = xp.clip((y / SDF_RES).astype(xp.int32), 0, self.ny - 1)
        inside = (x >= 0) & (x < WORLD_W) & (y >= 0) & (y < WORLD_H)
        return iy, ix, inside

    def distance(self, x, y):
        iy, ix, inside = self._cell(x, y)
        return self.xp.where(inside, self.sdf[iy, ix], -1.0)

    def cast(self, block, x, y, heading):
        """Sphere-trace every ray of one block type; returns distance / range."""
        xp = self.xp
        angles = heading[:, None] + self.rays[block][None, :]
        reach = RAY_RANGE[block]
        if self.fused:
            return _cuda_kernels()["cast"](x[:, None], y[:, None], angles.astype(xp.float32), self.sdf,
                                           np.int32(self.nx), np.int32(self.ny), np.float32(reach),
                                           np.int32(RAY_ITERS[block]))
        dx, dy = xp.cos(angles), xp.sin(angles)
        t = xp.full(angles.shape, .02, dtype=xp.float32)
        for _ in range(RAY_ITERS[block]):
            d = self.distance(x[:, None] + t * dx, y[:, None] + t * dy)
            t = xp.minimum(xp.where(d > .01, t + xp.maximum(d, .03), t), reach)
        return t / reach

    def sense(self, x, y, heading, speed, skid, s_now):
        xp = self.xp
        columns, done = [], set()
        for sensor in self.brain["sensors"]:
            block = sensor["block"]
            if block in RAY_RANGE:
                if block not in done:
                    columns.append(self.cast(block, x, y, heading))
                    done.add(block)
            elif block == "speed":
                columns.append((speed / V_MAX)[:, None])
            elif block == "grip":
                columns.append(xp.clip(skid, 0, 1)[:, None])
            elif block == "compass":
                index = ((s_now + COMPASS_LOOKAHEAD) / self.length * len(self.centre)).astype(xp.int32) % len(self.centre)
                target = self.centre[index]
                bearing = xp.arctan2(target[:, 1] - y, target[:, 0] - x) - heading
                columns.append(xp.stack([xp.sin(bearing), xp.cos(bearing)], axis=1))
        if not columns:
            return xp.ones((x.shape[0], 1), dtype=xp.float32)
        return xp.concatenate(columns, axis=1).astype(xp.float32)

    def _record(self, step, steps, every, history, inputs_history, outputs_history, x, y, heading, inputs, out):
        if step % every == 0 or step == steps - 1:
            history.append(self.xp.stack([x, y, heading], axis=1))
            inputs_history.append(inputs)
            outputs_history.append(out)

    def start_pose(self, rng=None):
        """(x, y, heading) at the start line, or a random pose when ``rng`` is given.

        A random pose is anywhere along the track, up to 0.2 units off the
        centre line and 0.3 rad off its direction: a fresh start the champion
        never trained from.
        """
        centre, tangent = self.track["centre"], self.track["tangent"]
        if rng is None:
            i, lateral, jitter = 0, 0.0, 0.0
        else:
            i = int(rng.integers(0, len(centre)))
            lateral, jitter = float(rng.uniform(-.2, .2)), float(rng.uniform(-.3, .3))
        normal = np.array([-tangent[i, 1], tangent[i, 0]])
        x, y = centre[i] + normal * lateral
        return float(x), float(y), math.atan2(tangent[i, 1], tangent[i, 0]) + jitter

    def run(self, population, steps, record_every=3, genome=None, pose=None):
        """Drive every car for ``steps`` steps and return fitness plus history.

        Crashed cars freeze where they hit the wall; the run stops early if no
        car is still driving.  ``pose`` overrides the start line.
        """
        xp = self.xp
        count = (population.genome if genome is None else genome).shape[0]
        x0, y0, h0 = pose or self.start_pose()
        x = xp.full(count, x0, dtype=xp.float32)
        y = xp.full(count, y0, dtype=xp.float32)
        heading = xp.full(count, h0, dtype=xp.float32)
        speed = xp.zeros(count, dtype=xp.float32)
        skid = xp.zeros(count, dtype=xp.float32)
        alive = xp.ones(count, dtype=bool)
        progress = xp.zeros(count, dtype=xp.float32)
        lap_step = xp.full(count, -1, dtype=xp.int32)
        crash_step = xp.full(count, -1, dtype=xp.int32)
        iy, ix, _ = self._cell(x, y)
        s_prev = self.s_grid[iy, ix]
        steer_i = self.actions.index("steer"); throttle_i = self.actions.index("throttle")
        brake_i = self.actions.index("brake") if "brake" in self.actions else None
        history, inputs_history, outputs_history = [], [], []
        length = self.length
        no_brake = xp.zeros(count, dtype=xp.float32)
        for step in range(steps):
            inputs = self.sense(x, y, heading, speed, skid, xp.mod(progress, length))
            out = population.forward(inputs, genome)
            steer, throttle = out[:, steer_i], out[:, throttle_i]
            brake = (out[:, brake_i] + 1) * .5 if brake_i is not None else no_brake
            if self.fused:
                steer, throttle = xp.ascontiguousarray(steer), xp.ascontiguousarray(throttle)
                _cuda_kernels()["step"](steer, throttle, xp.ascontiguousarray(brake), self.sdf, self.s_grid,
                                        np.int32(self.nx), np.int32(self.ny), np.float32(length), np.int32(step),
                                        x, y, heading, speed, skid, progress, s_prev, lap_step, crash_step, alive)
                self._record(step, steps, record_every, history, inputs_history, outputs_history, x, y, heading, inputs, out)
                if step % 30 == 29 and not bool(alive.any()):
                    self._record(step, steps, 1, history, inputs_history, outputs_history, x, y, heading, inputs, out)
                    break
                continue
            accel = xp.where(throttle > 0, throttle * A_MAX, throttle * COAST)
            new_speed = xp.clip(speed + (accel - DRAG * speed - brake * BRAKE) * DT, 0, V_MAX)
            curvature = steer * K_MAX
            lateral = new_speed ** 2 * xp.abs(curvature)
            new_skid = xp.maximum(lateral - GRIP, 0) / GRIP
            curvature = xp.where(lateral > GRIP, xp.sign(curvature) * GRIP / xp.maximum(new_speed ** 2, 1e-4), curvature)
            new_speed = new_speed * (1 - SKID_LOSS * xp.minimum(new_skid, 1) * DT)
            new_heading = heading + new_speed * curvature * DT
            new_x = x + new_speed * xp.cos(new_heading) * DT
            new_y = y + new_speed * xp.sin(new_heading) * DT
            x = xp.where(alive, new_x, x); y = xp.where(alive, new_y, y)
            heading = xp.where(alive, new_heading, heading)
            speed = xp.where(alive, new_speed, 0); skid = xp.where(alive, new_skid, 0)
            iy, ix, inside = self._cell(x, y)
            s_now = self.s_grid[iy, ix]
            ds = s_now - s_prev
            ds = xp.where(ds < -length / 2, ds + length, xp.where(ds > length / 2, ds - length, ds))
            progress = progress + xp.where(alive, ds, 0)
            s_prev = s_now
            lap_step = xp.where((lap_step < 0) & (progress >= length), step + 1, lap_step)
            hit = alive & ((xp.where(inside, self.sdf[iy, ix], -1.0) < CAR_RADIUS))
            crash_step = xp.where(hit, step + 1, crash_step)
            alive = alive & ~hit
            self._record(step, steps, record_every, history, inputs_history, outputs_history, x, y, heading, inputs, out)
            if step % 30 == 29 and not bool(alive.any()):
                self._record(step, steps, 1, history, inputs_history, outputs_history, x, y, heading, inputs, out)
                break
        crashed = crash_step >= 0
        fitness = progress - .35 * crashed
        return {
            "fitness": fitness, "progress": progress, "lap_step": lap_step, "crash_step": crash_step,
            "history": xp.stack(history), "inputs": xp.stack(inputs_history), "outputs": xp.stack(outputs_history),
        }


class NeuroRacersDemo(Demo):
    id = "neuro_racers"
    title = "Neuro-Racers"
    timing_methods = {"simulate": "simulation", "render_frame": "render"}
    cpu_only_methods = frozenset({"render_frame"})

    # ---- setup ---------------------------------------------------------
    def catalogue(self):
        specs = json.loads((Path(__file__).resolve().parents[2] / "config" / "demo_specs.json").read_text(encoding="utf-8"))
        return brain_catalogue(specs, self.id)

    def brain_spec(self, catalogue):
        spec = self.ctx.params.get("_brain")
        if not spec:
            preset = catalogue["presets"][catalogue.get("default_preset", "balanced")]
            spec = {k: preset[k] for k in ("sensors", "hidden", "actions")}
        return validate_brain(spec, catalogue)

    def simulate(self, sim, population, steps, record_every):
        return sim.run(population, steps, record_every)

    # ---- rendering -----------------------------------------------------
    def background(self, track):
        """Track surface rendered from the signed-distance field itself."""
        sdf = Image.fromarray(track["sdf"], mode="F").resize((1280, 720), Image.Resampling.BILINEAR)
        field = np.asarray(sdf)
        yy, xx = np.mgrid[0:720, 0:1280]
        base = np.zeros((720, 1280, 3), dtype=np.float32)
        base[:] = (3, 8, 19)
        grid = ((xx % 40) == 0) | ((yy % 40) == 0)
        base[grid] = (10, 22, 40)
        inside = field > 0
        base[inside] = (22, 27, 38)
        edge = np.exp(-(field / .035) ** 2)[..., None]
        base = base * (1 - edge) + np.array((96, 220, 255), dtype=np.float32) * edge
        image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
        glow = image.filter(ImageFilter.GaussianBlur(6))
        image = Image.blend(image, glow, .35)
        d = ImageDraw.Draw(image, "RGBA")
        centre = track["centre"] * PX
        d.line([tuple(p) for p in np.vstack([centre, centre[:1]])], fill=(255, 255, 255, 18), width=1)
        # Start/finish line, chequered across the track.
        left, right = track["left"][0] * PX, track["right"][0] * PX
        along = track["tangent"][0] * 5
        for row in (-1, 1):
            for k in range(8):
                a = left + (right - left) * k / 8 + along * row * .5
                b = left + (right - left) * (k + 1) / 8 + along * row * .5
                light = (k + (row > 0)) % 2 == 0
                d.line((*a, *b), fill=(245, 248, 255, 235) if light else (10, 12, 18, 235), width=5)
        return image

    @staticmethod
    def _car(d, x, y, heading, colour, scale=1.0):
        c, s = math.cos(heading), math.sin(heading)
        pts = [(.20, 0), (-.13, .09), (-.08, 0), (-.13, -.09)]
        d.polygon([((x + (px * c - py * s) * scale) * PX, (y + (px * s + py * c) * scale) * PX) for px, py in pts],
                  fill=colour, outline=(255, 255, 255, 235))

    def render_frame(self, background, record, fraction, ghosts=()):
        """Trails of the whole visible population, champion drawn last and brightest."""
        image = background.copy().convert("RGBA")
        trails = Image.new("RGBA", image.size, (0, 0, 0, 0))
        d = ImageDraw.Draw(trails, "RGBA")
        history = record["history"]                 # (S, N, 3), row 0 = champion
        samples = history.shape[0]
        upto = max(2, int(math.ceil(samples * fraction)))
        rank = record["rank"]
        for i in range(history.shape[1] - 1, -1, -1):
            path = history[:upto, i]
            if i == 0:
                continue
            warm = 1 - rank[i]
            colour = (int(60 + 170 * warm), int(150 + 60 * (1 - warm)), int(255 - 120 * warm), 70)
            d.line([(float(px) * PX, float(py) * PX) for px, py, _ in path], fill=colour, width=2)
            crash = record["crash_sample"][i]
            if 0 <= crash < upto:
                cx, cy = history[crash, i, 0] * PX, history[crash, i, 1] * PX
                d.line((cx - 5, cy - 5, cx + 5, cy + 5), fill=(255, 92, 92, 170), width=2)
                d.line((cx - 5, cy + 5, cx + 5, cy - 5), fill=(255, 92, 92, 170), width=2)
            else:
                self._car(d, *history[upto - 1, i], (120, 200, 255, 150), .85)
        for ghost in ghosts:
            path = ghost["history"][:max(2, int(math.ceil(len(ghost["history"]) * fraction)))]
            d.line([(float(px) * PX, float(py) * PX) for px, py, _ in path], fill=(190, 150, 255, 150), width=3)
            self._car(d, *path[-1], (190, 150, 255, 200), 1.1)
        glow = trails.filter(ImageFilter.GaussianBlur(3))
        image = Image.alpha_composite(image, glow)
        image = Image.alpha_composite(image, trails)
        d = ImageDraw.Draw(image, "RGBA")
        champion = history[:upto, 0]
        points = [(float(px) * PX, float(py) * PX) for px, py, _ in champion]
        d.line(points, fill=(255, 196, 92, 110), width=9)
        d.line(points, fill=(255, 236, 170, 255), width=3)
        self._car(d, *champion[-1], (255, 200, 80, 255), 1.4)
        return image.convert("RGB")

    # ---- output helpers ------------------------------------------------
    def select_record(self, result, show):
        """Champion first, then the next best cars, as NumPy arrays for rendering."""
        fitness = to_numpy(result["fitness"])
        order = np.argsort(-fitness, kind="stable")[:show]
        history = to_numpy(result["history"][:, self.ctx.xp.asarray(order)])
        crash = to_numpy(result["crash_step"])[order]
        record_every = self.record_every
        return {
            "order": order, "history": history,
            "rank": np.linspace(0, 1, len(order)),
            "crash_sample": np.where(crash >= 0, np.minimum(history.shape[0] - 1, crash // record_every), -1),
        }

    def arena_generation(self, generation, record, result, ghosts, population=None):
        order = record["order"]
        lap = to_numpy(result["lap_step"])[order]
        crash = to_numpy(result["crash_step"])[order]
        progress = to_numpy(result["progress"])[order]
        cars = []
        for j in range(len(order)):
            h = record["history"][:, j]
            cars.append({"x": np.round(h[:, 0] * 100).astype(int).tolist(),
                         "y": np.round(h[:, 1] * 100).astype(int).tolist(),
                         "h": np.round(np.degrees(h[:, 2])).astype(int).tolist(),
                         "crash": int(record["crash_sample"][j]), "lap_s": round(float(lap[j]) * DT, 2) if lap[j] > 0 else None,
                         "laps": round(float(progress[j]) / self.track["length"], 3)})
        brains = []
        if population is not None:
            # The champion's real activations along its drive, for the live diagram.
            champion = int(order[0])
            brains.append(brain_payload(self.brain, self.catalogue_data, population.genome[champion:champion + 1],
                                        result["inputs"][:, champion], f"Generation {generation} champion"))
        return {"kind": "racers", "generation": generation, "sample_dt": DT * self.record_every,
                "cars": cars, "brains": brains, "ghosts": [{"name": g["name"], "lap_s": g["lap_s"],
                                           "x": np.round(g["history"][:, 0] * 100).astype(int).tolist(),
                                           "y": np.round(g["history"][:, 1] * 100).astype(int).tolist(),
                                           "h": np.round(np.degrees(g["history"][:, 2])).astype(int).tolist()}
                                          for g in ghosts]}

    def write_arena(self, catalogue):
        folder = self.ctx.run_dir / "interactive"
        folder.mkdir(exist_ok=True)
        t = self.track
        to_list = lambda a: np.round(a * 100).astype(int).tolist()
        (folder / "arena.json").write_text(json.dumps({
            "kind": "racers", "world": [WORLD_W, WORLD_H], "track": t["name"],
            "centre": to_list(t["centre"][::4]), "left": to_list(t["left"][::4]), "right": to_list(t["right"][::4]),
            "start": [to_list(t["left"][0]), to_list(t["right"][0])],
        }))

    def network_overlay(self, population, champion_row, result, index, fraction, generation):
        genome = population.genome[champion_row:champion_row + 1]
        weights = [to_numpy(w[0]) for w, _ in population.layers(genome)]
        samples = result["inputs"].shape[0]
        sample = min(samples - 1, int(fraction * (samples - 1)))
        x = result["inputs"][sample, index:index + 1]
        _, activations = population.forward(x, genome, return_hidden=True)
        activations = [to_numpy(a[0]) for a in activations]
        catalogue = self.catalogue_data
        return draw_brain(weights, activations, input_labels(self.brain, catalogue), action_labels(self.brain, catalogue),
                          title="YOUR BRAIN · CHAMPION CAR",
                          subtitle=f"{self.brain['parameters']} evolved weights · generation {generation}",
                          footer=f"{self.brain['points']:g} / {self.brain['budget']:g} LEGO points")

    def load_ghosts(self, catalogue):
        ghosts = []
        for path in self.ctx.params.get("_ghosts") or []:
            try:
                genome, brain, meta = load_champion(Path(path) / "champion.npz")
            except Exception:
                continue
            if int(meta.get("track", -1)) != self.track["id"]:
                continue
            sim = RaceSim(self.ctx.xp, self.track, brain, catalogue)
            pop = Population(self.ctx.xp, 2, brain["layer_sizes"], seed=0)
            result = sim.run(pop, self.steps, self.record_every, genome=self.ctx.xp.asarray(genome))
            lap = int(to_numpy(result["lap_step"])[0])
            ghosts.append({"name": meta.get("name") or Path(path).name, "run": Path(path).name,
                           "history": to_numpy(result["history"][:, 0]),
                           "lap_s": round(lap * DT, 2) if lap > 0 else None,
                           "laps": round(float(to_numpy(result["progress"])[0]) / self.track["length"], 3)})
        return ghosts

    # ---- the run -------------------------------------------------------
    def run(self):
        xp, ctx, s = self.ctx.xp, self.ctx, self.settings
        catalogue = self.catalogue_data = self.catalogue()
        self.brain = self.brain_spec(catalogue)
        self.track = track_geometry(int(ctx.params.get("track", 0)))
        seed = int(ctx.params.get("seed", 7))
        sigma = float(ctx.params.get("mutation", .5))
        population_size = int(s.get("population", 160))
        generations = max(1, int(s.get("generations", 25)))
        self.steps = int(s.get("sim_steps", 900))
        self.record_every = max(1, int(s.get("record_every", 3)))
        show = max(1, min(population_size, int(s.get("record_cars", 32))))
        frames = ctx.frames

        sim = RaceSim(xp, self.track, self.brain, catalogue)
        population = Population(xp, population_size, self.brain["layer_sizes"], seed=seed, sigma=sigma)
        background = self.background(self.track)
        self.write_arena(catalogue)
        # Frame -> (generation, time fraction).  With more frames than
        # generations the extra frames replay the latest generation's drive.
        plan, done = [], 0
        for i in range(frames):
            target = int(round(generations * (i + 1) / frames))
            plan.append(max(1, target))
        blocks = {}
        for i, g in enumerate(plan):
            blocks.setdefault(g, []).append(i)
        fractions = [0.0] * frames
        for g, members in blocks.items():
            for j, i in enumerate(members):
                fractions[i] = (j + 1) / len(members)
        ctx.write_meta({"brain": self.brain, "track": {"id": self.track["id"], "name": self.track["name"]},
                        "lab": {"checkpoints": "checkpoints", "generations": generations, "compare": True},
                        "arena_view": {"folder": "interactive", "kind": "racers", "arena": "interactive/arena.json",
                                       "frames": [[g, round(f, 4)] for g, f in zip(plan, fractions)]}})
        (ctx.run_dir / "overlays" / "network").mkdir(parents=True, exist_ok=True)

        best_ever, best_lap, car_seconds = -1e9, None, 0.0
        champion_genome = None
        history_stats = []
        current = None
        ghosts = []
        for i in range(frames):
            while done < plan[i]:
                done += 1
                if done > 1:
                    population.evolve(current["result"]["fitness"])
                result = self.simulate(sim, population, self.steps, self.record_every)
                car_seconds += population.size * (result["history"].shape[0] - 1) * self.record_every * DT
                record = self.select_record(result, show)
                champion = int(record["order"][0])
                fitness = to_numpy(result["fitness"])
                laps = to_numpy(result["lap_step"])
                crashes = to_numpy(result["crash_step"])
                lap_s = float(laps[champion]) * DT if laps[champion] > 0 else None
                finished = laps[laps > 0]
                if finished.size:
                    fastest = float(finished.min()) * DT
                    best_lap = fastest if best_lap is None else min(best_lap, fastest)
                if fitness[champion] > best_ever:
                    best_ever = float(fitness[champion])
                    champion_genome = population.genome[champion:champion + 1].copy()
                if done == generations:
                    ghosts = self.load_ghosts(catalogue)
                stats = {"generation": done, "best_laps": float(to_numpy(result["progress"])[champion]) / self.track["length"],
                         "lap_rate": float((laps > 0).mean()), "crash_rate": float((crashes >= 0).mean()),
                         "fastest_lap_s": round(float(finished.min()) * DT, 2) if finished.size else None}
                history_stats.append(stats)
                current = {"result": result, "record": record, "champion": champion, "stats": stats, "lap_s": lap_s}
                save_checkpoint(ctx.run_dir / "checkpoints" / f"gen_{done:04d}.npz",
                                champion=population.genome[champion:champion + 1])
                (ctx.run_dir / "interactive" / f"gen_{done:04d}.json").write_text(
                    json.dumps(self.arena_generation(done, record, result, ghosts if done == generations else [], population)))
            fraction = fractions[i]
            image = self.render_frame(background, current["record"], fraction,
                                      ghosts if done == generations else ())
            ctx.save_frame(image, ctx.frame_path(i))
            graph = self.network_overlay(population, current["champion"], current["result"], current["champion"],
                                         fraction, done)
            ctx.save_frame(graph, ctx.run_dir / "overlays" / "network" / f"frame_{i:04d}.jpg")
            stats = current["stats"]
            overlay = {
                "generation": f"{done} / {generations}",
                "cars per generation": f"{population.size:,}",
                "champion distance": f"{stats['best_laps']:.2f} laps",
                "fastest lap": f"{stats['fastest_lap_s']:.1f} s" if stats["fastest_lap_s"] else "no full lap yet",
                "cars completing a lap": f"{100 * stats['lap_rate']:.0f}%",
                "cars that crashed": f"{100 * stats['crash_rate']:.0f}%",
                "brain": f"{self.brain['parameters']} weights · {self.brain['points']:g}/{self.brain['budget']:g} pts",
                "car-seconds simulated": f"{car_seconds:,.0f}",
            }
            if ghosts and done == generations:
                for ghost in ghosts:
                    overlay[f"ghost {ghost['name']}"] = f"{ghost['lap_s']:.1f} s lap" if ghost["lap_s"] else f"{ghost['laps']:.2f} laps"
            ctx.write_status(i, f"generation {done} of {generations}", overlay)

        save_champion(ctx.run_dir / "champion.npz", champion_genome, self.brain,
                      {"track": self.track["id"], "name": ctx.params.get("_name") or ctx.run_dir.name[-5:]})
        reveal = self.reveal(catalogue, seed, sigma)
        ctx.write_meta({"summary": {"track_id": self.track["id"], "track": self.track["name"],
                                    "best_lap_s": round(best_lap, 2) if best_lap else None,
                                    "best_laps": round(max(h["best_laps"] for h in history_stats), 3),
                                    "brain_points": self.brain["points"], "parameters": self.brain["parameters"],
                                    "generations": generations, "population": population.size},
                        "generation_stats": history_stats})
        ctx.finish(reveal)

    # ---- generation lab ------------------------------------------------
    @classmethod
    def replay(cls, run_dir: Path, meta: dict, gens, seed: int):
        """Re-drive saved champions from one random start, on the CPU.

        Every requested generation's champion starts from the same random pose
        (anywhere on the track, slightly off-line and off-angle), so the
        visitor can compare how the network drove at different points of its
        training.  Returns arena JSON plus each network's live activations.
        """
        specs = json.loads((Path(__file__).resolve().parents[2] / "config" / "demo_specs.json").read_text(encoding="utf-8"))
        catalogue = brain_catalogue(specs, cls.id)
        brain = meta["brain"]
        track = track_geometry(int((meta.get("track") or {}).get("id", meta.get("params", {}).get("track", 0))))
        steps = int((meta.get("settings") or {}).get("sim_steps", 900))
        every = 2
        sim = RaceSim(np, track, brain, catalogue)
        pose = sim.start_pose(np.random.default_rng(int(seed)))
        genomes = np.concatenate([load_checkpoint(Path(run_dir) / "checkpoints" / f"gen_{int(g):04d}.npz")["champion"]
                                  for g in gens])
        pop = Population(np, max(2, len(gens)), brain["layer_sizes"], seed=0)
        result = sim.run(pop, steps, every, genome=genomes, pose=pose)
        cars, brains = [], []
        for j, g in enumerate(gens):
            h = result["history"][:, j]
            lap, crash = int(result["lap_step"][j]), int(result["crash_step"][j])
            cars.append({"x": np.round(h[:, 0] * 100).astype(int).tolist(), "y": np.round(h[:, 1] * 100).astype(int).tolist(),
                         "h": np.round(np.degrees(h[:, 2])).astype(int).tolist(), "label": f"gen {int(g)}", "gen": int(g),
                         "crash": crash // every if crash >= 0 else -1, "lap_s": round(lap * DT, 2) if lap > 0 else None,
                         "laps": round(float(result["progress"][j]) / track["length"], 3)})
            brains.append(brain_payload(brain, catalogue, genomes[j:j + 1], result["inputs"][:, j], f"Generation {int(g)} champion"))
        return {"kind": "racers", "generation": [int(g) for g in gens], "sample_dt": DT * every,
                "replay": {"gens": [int(g) for g in gens], "seed": int(seed)}, "cars": cars, "brains": brains, "ghosts": []}

    # ---- the reveal ----------------------------------------------------
    def reveal(self, catalogue, seed, sigma):
        """Independent evolutions of the same brain from different random starts.

        All islands run as one batched population; individuals only compete
        within their own island, so every tile is a genuinely separate search.
        """
        xp, s = self.ctx.xp, self.settings
        islands = max(1, int(s.get("ensemble", 16)))
        size = max(4, int(s.get("reveal_population", 48)))
        generations = max(1, int(s.get("reveal_generations", 10)))
        sim = RaceSim(xp, self.track, self.brain, catalogue)
        population = Population(xp, size, self.brain["layer_sizes"], seed=seed * 1000 + 17, groups=islands, sigma=sigma)
        result = None
        for g in range(generations):
            if result is not None:
                population.evolve(result["fitness"])
            result = sim.run(population, self.steps, self.record_every)
            self.ctx.write_status(self.ctx.frames - 1, f"scale reveal: {islands} independent evolutions, generation {g + 1}/{generations}")
        fitness = to_numpy(result["fitness"]).reshape(islands, size)
        laps = to_numpy(result["lap_step"]).reshape(islands, size)
        progress = to_numpy(result["progress"]).reshape(islands, size)
        history = to_numpy(result["history"])
        background = self.background(self.track)
        cols = max(1, int(math.ceil(math.sqrt(islands))))
        rows = int(math.ceil(islands / cols))
        gap = 8
        tile_w = (1280 - gap * (cols + 1)) // cols
        tile_h = (720 - gap * (rows + 1)) // rows
        out = Image.new("RGB", (1280, 720), (3, 6, 15))
        d = ImageDraw.Draw(out, "RGBA")
        tiles = []
        label_size = max(10, min(18, tile_h // 7))
        for g in range(islands):
            best = int(np.argmax(fitness[g]))
            index = g * size + best
            record = {"history": history[:, [index]], "rank": np.zeros(1), "crash_sample": np.array([-1])}
            tile = self.render_frame(background, record, 1.0).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            r, c = divmod(g, cols)
            x0, y0 = gap + c * (tile_w + gap), gap + r * (tile_h + gap)
            out.paste(tile, (x0, y0))
            lap = float(laps[g, best]) * DT if laps[g, best] > 0 else None
            text = f"search {g + 1} · " + (f"lap {lap:.1f} s" if lap else f"{progress[g, best] / self.track['length']:.2f} laps")
            d.rectangle((x0, y0 + tile_h - label_size - 8, x0 + tile_w, y0 + tile_h), fill=(2, 5, 15, 200))
            d.text((x0 + 6, y0 + tile_h - label_size - 5), text, font=font(label_size, True), fill=(255, 226, 160))
            tiles.append({"island": g + 1, "lap_s": round(lap, 2) if lap else None,
                          "laps": round(float(progress[g, best]) / self.track["length"], 3)})
        path = self.ctx.run_dir / "reveal.jpg"
        self.ctx.save_frame(out, path)
        # The same result as animation data: every search's champion drive, so
        # the viewer can play the grid instead of showing a still image.
        crash_all = to_numpy(result["crash_step"]).reshape(-1)
        lap_all = to_numpy(result["lap_step"]).reshape(-1)
        boxes = []
        for g in range(islands):
            best = int(np.argmax(fitness[g])); index = g * size + best
            h = history[:, index]; crash = int(crash_all[index]); lap = int(lap_all[index])
            lap_s = round(lap * DT, 2) if lap > 0 else None
            boxes.append({"label": f"search {g + 1}", "lap_s": lap_s,
                          "laps": round(float(progress[g, best]) / self.track["length"], 3),
                          "gen": {"kind": "racers", "generation": generations, "sample_dt": DT * self.record_every,
                                  "cars": [{"x": np.round(h[:, 0] * 100).astype(int).tolist(),
                                            "y": np.round(h[:, 1] * 100).astype(int).tolist(),
                                            "h": np.round(np.degrees(h[:, 2])).astype(int).tolist(),
                                            "crash": crash // self.record_every if crash >= 0 else -1,
                                            "lap_s": lap_s, "laps": round(float(progress[g, best]) / self.track["length"], 3)}],
                                  "brains": [], "ghosts": []}})
        folder = self.ctx.run_dir / "interactive"; folder.mkdir(exist_ok=True)
        (folder / "reveal.json").write_text(json.dumps({"kind": "racers", "shared_arena": "interactive/arena.json",
                                                        "boxes": boxes}))
        self.ctx.write_meta({"reveal_tiles": tiles, "reveal_population": size * islands,
                             "reveal_generations": generations,
                             "reveal_view": {"file": "interactive/reveal.json", "boxes": islands}})
        return path
