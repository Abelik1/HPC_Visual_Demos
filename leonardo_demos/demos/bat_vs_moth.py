"""Bat vs Moth: a sonar arms race between two visitor-built brains.

Many caves run in parallel.  Each holds one bat and a few moths; every bat has
the bat visitor's architecture and every moth the moth visitor's, each with
its own weights.  Both species evolve at once: bats are rewarded for catches,
moths for staying alive.  Moths may evolve jamming clicks that put false
echoes into a nearby bat's ears, as real tiger moths do.

The sonar is a deliberately reduced exhibition model, not acoustics: an echo's
loudness falls with distance and with a cardioid ear pattern, its delay is
proportional to distance, rock does not occlude, and there is no Doppler
physics beyond the closing-speed input.  See docs/SCIENTIFIC_NOTES.md.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..backend import to_numpy
from ..base import Demo
from ..neuroevo import Population, action_labels, brain_catalogue, input_labels, validate_brains
from ..neuro_render import draw_brain
from ..render import font

WORLD_W, WORLD_H = 16.0, 9.0
PX = 80.0
SDF_RES = 1 / 24
DT = 1 / 30
BAT_VMIN, BAT_VMAX, BAT_TURN = 0.9, 3.2, 3.4
MOTH_VMIN, MOTH_VMAX, MOTH_TURN = 0.3, 1.4, 4.5
DIVE_SPEED, DIVE_STEPS, DIVE_REST = 2.7, 10, 45
CHIRP_REST = 5                  # steps between calls
ECHO_RANGE = 6.0                # a moth echo is audible to the bat within this
HEAR_RANGE = 9.0                # a bat call is audible to a moth within this
JAM_RANGE = 3.0
EAR_OFFSET = math.radians(40)
CATCH_RADIUS = 0.3
WALL_MARGIN = 0.12
WHISKER_RANGE, FEELER_RANGE = 0.9, 0.6
WHISKER_ANGLES = (-45, 0, 45)
FEELER_ANGLES = (-50, 50)
ECHO_DECAY, HEAR_DECAY = 0.93, 0.97
# Jam and dive fire above this output, so an untrained moth rarely does either.
TRIGGER = 0.5
# Survival lost by jamming for the whole hunt; clicks are cheap but not free.
JAM_COST = 0.08
# Selection settings tuned for sparse, noisy hunting rewards: stronger
# tournaments and a larger initial weight scale than the racers need.
EVOLUTION = {"tournament": 5, "init_gain": 2.0}
# The hunt step is hundreds of small array operations, so on CUDA it is
# kernel-launch bound (~20 ms per step whatever the size).  Below this many
# caves NumPy is faster, and Auto uses it; an explicit GPU request is honoured.
GPU_MIN_CAVES = 4096
# A bat calls whenever its chirp output is above this (and it has rested).
CHIRP_TRIGGER = -0.3
BAT_COLOUR, MOTH_COLOUR, ECHO_COLOUR, FAKE_COLOUR = (255, 190, 80), (110, 240, 200), (120, 225, 255), (255, 90, 200)


def cave_geometry(seed: int):
    """Procedural rock pillars and hanging rock, as circles plus an SDF grid."""
    rng = np.random.default_rng(int(seed) * 7919 + 3)
    for _ in range(40):
        rocks = []
        for _ in range(rng.integers(8, 12)):     # free-standing pillars
            rocks.append((rng.uniform(1.4, WORLD_W - 1.4), rng.uniform(1.4, WORLD_H - 1.4), rng.uniform(.35, .85)))
        for _ in range(rng.integers(4, 7)):      # stalactites and stalagmites
            rocks.append((rng.uniform(.8, WORLD_W - .8), rng.choice([-.15, WORLD_H + .15]), rng.uniform(.6, 1.2)))
        ny, nx = int(round(WORLD_H / SDF_RES)), int(round(WORLD_W / SDF_RES))
        gx = (np.arange(nx) + .5) * SDF_RES
        gy = (np.arange(ny) + .5) * SDF_RES
        X, Y = np.meshgrid(gx, gy)
        sdf = np.minimum(np.minimum(X, WORLD_W - X), np.minimum(Y, WORLD_H - Y))
        for cx, cy, r in rocks:
            sdf = np.minimum(sdf, np.hypot(X - cx, Y - cy) - r)
        if (sdf > .45).mean() > .62:
            break
    edges = []
    for cx, cy, r in rocks:
        t = np.linspace(0, 2 * math.pi, max(12, int(r * 40)), endpoint=False)
        pts = np.stack([cx + r * np.cos(t), cy + r * np.sin(t)], axis=1)
        ix = np.clip((pts[:, 0] / SDF_RES).astype(int), 0, nx - 1)
        iy = np.clip((pts[:, 1] / SDF_RES).astype(int), 0, ny - 1)
        inside_world = (pts[:, 0] > 0) & (pts[:, 0] < WORLD_W) & (pts[:, 1] > 0) & (pts[:, 1] < WORLD_H)
        keep = inside_world & (sdf[iy, ix] > -.05)
        edges.append(pts[keep])
    return {"rocks": rocks, "sdf": sdf.astype(np.float32), "edges": np.concatenate(edges) if edges else np.zeros((0, 2))}


class CaveSim:
    """Many caves in lockstep: one bat and ``moths`` moths in each."""

    def __init__(self, xp, cave, bat_brain, moth_brain, moths):
        self.xp, self.cave, self.bat, self.moth, self.k = xp, cave, bat_brain, moth_brain, int(moths)
        self.sdf = xp.asarray(cave["sdf"])
        self.ny, self.nx = cave["sdf"].shape
        self.free = np.argwhere(cave["sdf"] > .6)

    def distance(self, x, y):
        xp = self.xp
        ix = xp.clip((x / SDF_RES).astype(xp.int32), 0, self.nx - 1)
        iy = xp.clip((y / SDF_RES).astype(xp.int32), 0, self.ny - 1)
        inside = (x >= 0) & (x < WORLD_W) & (y >= 0) & (y < WORLD_H)
        return xp.where(inside, self.sdf[iy, ix], -1.0)

    def march(self, x, y, heading, angles_deg, reach, iters=8):
        """Sphere-traced distance to rock along each angle, divided by reach."""
        xp = self.xp
        angles = heading[..., None] + xp.asarray(np.radians(angles_deg).astype(np.float32))
        dx, dy = xp.cos(angles), xp.sin(angles)
        t = xp.full(angles.shape, .02, dtype=xp.float32)
        for _ in range(iters):
            d = self.distance(x[..., None] + t * dx, y[..., None] + t * dy)
            t = xp.minimum(xp.where(d > .01, t + xp.maximum(d, .03), t), reach)
        return t / reach

    def starts(self, rng, count):
        cells = self.free[rng.integers(0, len(self.free), count)]
        return ((cells[:, 1] + .5) * SDF_RES).astype(np.float32), ((cells[:, 0] + .5) * SDF_RES).astype(np.float32)

    def run(self, bats, moths, moth_index, steps, seed, record_every=2, shared_starts=False):
        """Hunt for ``steps`` steps in every cave.

        ``moth_index`` (arenas, K) says which moth genome flies in which slot.
        Each cave gets its own random start positions, so no brain is selected
        merely for suiting one particular layout; ``shared_starts`` gives every
        cave the same positions (used for like-for-like tests).
        """
        xp, k = self.xp, self.k
        arenas = bats.size
        rng = np.random.default_rng(seed)
        caves = 1 if shared_starts else arenas
        bx0, by0 = self.starts(rng, caves)
        mx0, my0 = self.starts(rng, caves * k)
        bx = xp.asarray(np.broadcast_to(bx0, (arenas,)).copy()); by = xp.asarray(np.broadcast_to(by0, (arenas,)).copy())
        bh = xp.asarray(np.broadcast_to(rng.uniform(0, 2 * math.pi, caves).astype(np.float32), (arenas,)).copy())
        bv = xp.full(arenas, BAT_VMIN, dtype=xp.float32)
        mx = xp.asarray(np.broadcast_to(mx0.reshape(caves, k), (arenas, k)).copy())
        my = xp.asarray(np.broadcast_to(my0.reshape(caves, k), (arenas, k)).copy())
        mh = xp.asarray(np.broadcast_to(rng.uniform(0, 2 * math.pi, (caves, k)).astype(np.float32), (arenas, k)).copy())
        mvx = xp.zeros((arenas, k), dtype=xp.float32); mvy = xp.zeros((arenas, k), dtype=xp.float32)
        alive = xp.ones((arenas, k), dtype=bool)
        catch_step = xp.full((arenas, k), -1, dtype=xp.int32)
        dive = xp.zeros((arenas, k), dtype=xp.int32); rest = xp.zeros((arenas, k), dtype=xp.int32)
        refractory = xp.zeros(arenas, dtype=xp.int32)
        ear = xp.zeros((arenas, 2, 2), dtype=xp.float32); ear[:, :, 1] = 1.0      # (loudness, delay) per ear
        closing = xp.zeros(arenas, dtype=xp.float32)
        heard = xp.zeros((arenas, k), dtype=xp.float32)
        bearing = xp.zeros((arenas, k, 2), dtype=xp.float32)
        bat_actions, moth_actions = self.bat["actions"], self.moth["actions"]
        last = xp.zeros((arenas, len(bat_actions)), dtype=xp.float32)
        moth_genome = moths.genome[xp.asarray(moth_index.reshape(-1))]
        dive_noise = xp.asarray(rng.uniform(-.6, .6, (steps, arenas, k)).astype(np.float32))
        chirps = xp.zeros(arenas, dtype=xp.float32); jammed_chirps = xp.zeros(arenas, dtype=xp.float32)
        wall_hits = xp.zeros(arenas, dtype=xp.float32)
        jam_steps = xp.zeros((arenas, k), dtype=xp.float32); dives = xp.zeros((arenas, k), dtype=xp.float32)
        near_steps = xp.zeros((arenas, k), dtype=xp.float32); near_jam = xp.zeros((arenas, k), dtype=xp.float32)
        far_steps = xp.zeros((arenas, k), dtype=xp.float32); far_jam = xp.zeros((arenas, k), dtype=xp.float32)
        # Shaping: how close the bat stays to moths, summed over the hunt.
        pursuit = xp.zeros(arenas, dtype=xp.float32); exposure = xp.zeros((arenas, k), dtype=xp.float32)
        record = {"bat": [], "moth": [], "alive": [], "jam": []}
        chirp_log, fake_log = [], []
        has = {name: name in bat_actions or name in moth_actions for name in ("dive", "jam")}
        for step in range(steps):
            # ---- senses --------------------------------------------------
            columns = []
            for sensor in self.bat["sensors"]:
                block = sensor["block"]
                if block == "ear_left": columns.append(ear[:, 0, :])
                elif block == "ear_right": columns.append(ear[:, 1, :])
                elif block == "whiskers": columns.append(self.march(bx, by, bh, WHISKER_ANGLES, WHISKER_RANGE))
                elif block == "doppler": columns.append(closing[:, None])
                elif block == "memory": columns.append(last)
            bat_in = xp.concatenate(columns, axis=1) if columns else xp.ones((arenas, 1), dtype=xp.float32)
            out = bats.forward(bat_in)
            last = out
            columns = []
            for sensor in self.moth["sensors"]:
                block = sensor["block"]
                if block == "tympanum": columns.append(heard.reshape(-1, 1))
                elif block == "direction": columns.append(bearing.reshape(-1, 2))
                elif block == "feelers": columns.append(self.march(mx, my, mh, FEELER_ANGLES, FEELER_RANGE).reshape(-1, 2))
            moth_in = xp.concatenate(columns, axis=1) if columns else xp.ones((arenas * k, 1), dtype=xp.float32)
            mout = moths.forward(moth_in, moth_genome).reshape(arenas, k, -1)
            # ---- bat flight ----------------------------------------------
            bh = bh + out[:, bat_actions.index("steer")] * BAT_TURN * DT
            bv = BAT_VMIN + (out[:, bat_actions.index("speed")] + 1) * .5 * (BAT_VMAX - BAT_VMIN)
            nx_, ny_ = bx + bv * xp.cos(bh) * DT, by + bv * xp.sin(bh) * DT
            hit = self.distance(nx_, ny_) < WALL_MARGIN
            bx, by = xp.where(hit, bx, nx_), xp.where(hit, by, ny_)
            bh = xp.where(hit, bh + math.pi, bh)
            wall_hits = wall_hits + hit
            bvx, bvy = bv * xp.cos(bh), bv * xp.sin(bh)
            # ---- chirp and echoes ----------------------------------------
            chirp = (out[:, bat_actions.index("chirp")] > CHIRP_TRIGGER) & (refractory <= 0)
            refractory = xp.where(chirp, CHIRP_REST, refractory - 1)
            chirps = chirps + chirp
            dx, dy = mx - bx[:, None], my - by[:, None]
            dist = xp.sqrt(dx * dx + dy * dy) + 1e-6
            rel = xp.arctan2(dy, dx) - bh[:, None]
            fade = xp.clip(1 - dist / ECHO_RANGE, 0, 1) * alive      # perceived loudness, linear in distance
            jam_on = (mout[:, :, moth_actions.index("jam")] > TRIGGER) & alive if "jam" in moth_actions else xp.zeros((arenas, k), dtype=bool)
            near = alive & (dist < JAM_RANGE)
            # A jamming moth's *own* echo is replaced by a phantom with random
            # loudness and delay in each ear, as tiger-moth clicks protect the
            # clicker; other moths' echoes are unaffected.
            phantom = jam_on & near
            noise = xp.asarray(rng.random((2, arenas, k, 2), dtype=np.float32))
            rows = xp.arange(arenas)
            new_ear, fake_ear = [], []
            for side, offset in ((0, -EAR_OFFSET), (1, EAR_OFFSET)):
                loud = xp.where(phantom, .3 + .7 * noise[side, :, :, 0], .5 * (1 + xp.cos(rel - offset)) * fade)
                delay = xp.where(phantom, .1 + .8 * noise[side, :, :, 1], dist / ECHO_RANGE)
                best = xp.argmax(loud, axis=1)
                heard_any = loud[rows, best] > 0
                new_ear.append(xp.stack([loud[rows, best], xp.where(heard_any, delay[rows, best], 1.0)], axis=1))
                fake_ear.append(xp.where(phantom[rows, best] & heard_any, delay[rows, best], -1.0))
            new_ear = xp.stack(new_ear, axis=1)                                   # (A, 2, 2)
            fake_ear = xp.stack(fake_ear, axis=1)                                 # (A, 2): phantom delay or -1
            jammed = chirp & (fake_ear >= 0).any(axis=1)
            ear = xp.where(chirp[:, None, None], new_ear, ear * xp.asarray([ECHO_DECAY, 1.0], dtype=xp.float32))
            target = xp.argmax(fade, axis=1); rows = xp.arange(arenas)
            approach = -(dx[rows, target] * (mvx[rows, target] - bvx) + dy[rows, target] * (mvy[rows, target] - bvy)) / dist[rows, target]
            closing = xp.where(chirp, xp.where(fade[rows, target] > 0, xp.clip(approach / 4, -1, 1), 0), closing * ECHO_DECAY)
            jammed_chirps = jammed_chirps + jammed
            # ---- moths hear the call -------------------------------------
            loud_at_moth = xp.clip(1 - dist / HEAR_RANGE, 0, 1)
            heard = xp.where(chirp[:, None], loud_at_moth, heard * HEAR_DECAY)
            towards = xp.arctan2(-dy, -dx) - mh
            bearing = xp.where((chirp[:, None] & (loud_at_moth > 0))[..., None],
                               xp.stack([xp.sin(towards), xp.cos(towards)], axis=2), bearing)
            # ---- moth flight ---------------------------------------------
            mh = mh + mout[:, :, moth_actions.index("steer")] * MOTH_TURN * DT
            speed = MOTH_VMIN + (mout[:, :, moth_actions.index("speed")] + 1) * .5 * (MOTH_VMAX - MOTH_VMIN)
            if "dive" in moth_actions:
                start = (mout[:, :, moth_actions.index("dive")] > TRIGGER) & (rest <= 0) & alive
                dive = xp.where(start, DIVE_STEPS, xp.maximum(dive - 1, 0))
                rest = xp.where(start, DIVE_REST, rest - 1)
                dives = dives + start
                speed = xp.where(dive > 0, DIVE_SPEED, speed)
                mh = mh + xp.where(dive > 0, dive_noise[step], 0)
            nmx, nmy = mx + speed * xp.cos(mh) * DT, my + speed * xp.sin(mh) * DT
            moth_hit = self.distance(nmx, nmy) < WALL_MARGIN
            moving = alive & ~moth_hit
            mvx = xp.where(moving, (nmx - mx) / DT, 0); mvy = xp.where(moving, (nmy - my) / DT, 0)
            mx, my = xp.where(moving, nmx, mx), xp.where(moving, nmy, my)
            mh = xp.where(moth_hit, mh + math.pi, mh)
            jam_steps = jam_steps + jam_on
            near_steps = near_steps + near
            near_jam = near_jam + (jam_on & near)
            far = alive & ~near
            far_steps = far_steps + far
            far_jam = far_jam + (jam_on & far)
            # ---- catches -------------------------------------------------
            cdx, cdy = mx - bx[:, None], my - by[:, None]
            caught = alive & (cdx * cdx + cdy * cdy < CATCH_RADIUS ** 2)
            gap = xp.sqrt(cdx * cdx + cdy * cdy)
            closeness = xp.clip(1 - gap / ECHO_RANGE, 0, 1) * alive
            exposure = exposure + closeness
            # Pursuit: how directly the bat flies at the nearest moth it could
            # hear.  Unlike closeness this does not reward a lucky start.
            nearest = xp.argmin(xp.where(alive, gap, 1e9), axis=1)
            audible = closeness[rows, nearest] > 0
            aim = xp.cos(xp.arctan2(cdy[rows, nearest], cdx[rows, nearest]) - bh)
            pursuit = pursuit + xp.where(audible, aim, 0)
            catch_step = xp.where(caught, step + 1, catch_step)
            alive = alive & ~caught
            chirp_log.append(chirp)
            fake_log.append(xp.where(chirp[:, None], fake_ear, -1.0))
            if step % record_every == 0 or step == steps - 1:
                record["bat"].append(xp.stack([bx, by, bh], axis=1))
                record["moth"].append(xp.stack([mx, my], axis=2))
                record["alive"].append(alive)
                record["jam"].append(jam_on)
        caught_any = catch_step >= 0
        timing = xp.where(caught_any, 1 - catch_step / steps, 0)
        bat_fitness = caught_any.sum(axis=1) + .5 * timing.sum(axis=1) + 1.5 * pursuit / steps - .002 * chirps - .05 * wall_hits
        survival = xp.where(caught_any, catch_step / steps, 1.0)
        slot_fitness = survival - .5 * exposure / steps - JAM_COST * jam_steps / steps - .02 * dives
        return {
            "bat_fitness": bat_fitness, "slot_fitness": slot_fitness, "catch_step": catch_step,
            "chirps": chirps, "jammed_chirps": jammed_chirps, "near_steps": near_steps, "near_jam": near_jam, "far_steps": far_steps, "far_jam": far_jam,
            "record": {key: xp.stack(value) for key, value in record.items()},
            "chirp_log": xp.stack(chirp_log), "fake_log": xp.stack(fake_log), "record_every": record_every,
            "bat_inputs": bat_in, "moth_inputs": moth_in,
        }


def quiet_start(population, brain, bias=-1.0):
    """Start moths not yet knowing how to jam or dive.

    Their jam and dive outputs begin strongly negative, so these behaviours
    appear only if evolution finds them worth their energy.  Without this an
    untrained moth jams by chance often enough that no bat can ever learn.
    """
    actions = brain["actions"]
    for name in ("jam", "dive"):
        if name in actions:
            column = population.genome.shape[1] - len(actions) + actions.index(name)
            population.genome[:, column] = bias
    return population


def jamming_rates(result):
    """Share of moth-steps spent jamming with a bat near, and with none near.

    Selective jamming (much more often near a bat than far from one) is the
    evolved behaviour; blanket jamming is just a noisy moth.
    """
    near = float(to_numpy(result["near_jam"]).sum()) / max(1.0, float(to_numpy(result["near_steps"]).sum()))
    far = float(to_numpy(result["far_jam"]).sum()) / max(1.0, float(to_numpy(result["far_steps"]).sum()))
    return near, far


def selective(near, far):
    return near > .35 and near - far > .15


def moth_fitness(xp, slot_fitness, moth_index, moth_count):
    """Mean slot fitness of every moth genome over all caves it flew in."""
    flat = to_numpy(moth_index).reshape(-1)
    values = to_numpy(slot_fitness).reshape(-1)
    total = np.bincount(flat, weights=values, minlength=moth_count)
    count = np.maximum(np.bincount(flat, minlength=moth_count), 1)
    return total / count


def assign_moths(rng, groups, bats_per_group, moths_per_group, k):
    """Deal every island's moths into its caves, each moth to about equally many slots."""
    rows = []
    for g in range(groups):
        need = bats_per_group * k
        deck = np.concatenate([rng.permutation(moths_per_group) for _ in range(int(math.ceil(need / moths_per_group)))])[:need]
        rows.append(g * moths_per_group + deck.reshape(bats_per_group, k))
    return np.concatenate(rows)


class BatVsMothDemo(Demo):
    id = "bat_vs_moth"
    title = "Bat vs Moth"
    timing_methods = {"hunt": "simulation", "render_frame": "render"}
    cpu_only_methods = frozenset({"render_frame"})

    def catalogues(self):
        specs = json.loads((Path(__file__).resolve().parents[2] / "config" / "demo_specs.json").read_text())
        return brain_catalogue(specs, self.id)

    def hunt(self, sim, bats, moths, moth_index, steps, seed):
        return sim.run(bats, moths, moth_index, steps, seed, self.record_every)

    # ---- rendering -----------------------------------------------------
    def rock_layer(self, lit):
        image = Image.new("RGB", (1280, 720), (1, 3, 8))
        field = np.asarray(Image.fromarray(self.cave["sdf"], mode="F").resize((1280, 720), Image.Resampling.BILINEAR))
        base = np.zeros((720, 1280, 3), dtype=np.float32); base[:] = (2, 4, 10)
        if lit:
            base[:] = (10, 16, 28)
            base[field < 0] = (46, 52, 68)
            edge = np.exp(-(field / .03) ** 2)[..., None]
            base = base * (1 - edge) + np.array((120, 130, 150), dtype=np.float32) * edge
        image = Image.fromarray(np.clip(base, 0, 255).astype(np.uint8), "RGB")
        return image

    @staticmethod
    def _bat(d, x, y, heading, scale=1.0):
        c, s = math.cos(heading), math.sin(heading)
        pts = [(.22, 0), (-.05, .26), (-.14, .08), (-.1, 0), (-.14, -.08), (-.05, -.26)]
        d.polygon([((x + (px * c - py * s) * scale) * PX, (y + (px * s + py * c) * scale) * PX) for px, py in pts],
                  fill=(*BAT_COLOUR, 255), outline=(255, 245, 220, 255))

    def render_frame(self, episode, fraction, lit):
        """One moment of the champion cave: the bat's senses, or the lit cave."""
        image = (self.lit_bg if lit else self.dark_bg).copy().convert("RGBA")
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        g = ImageDraw.Draw(glow, "RGBA")
        bat, moths, alive, jam = episode["bat"], episode["moth"], episode["alive"], episode["jam"]
        every, samples = episode["record_every"], bat.shape[0]
        full = fraction >= 1.0
        now = samples - 1 if full else max(1, int(round(fraction * (samples - 1))))
        now_step = now * every
        chirps = np.flatnonzero(episode["chirps"][:now_step + 1])
        window = now_step + 1 if full else int(1.6 / DT)
        edges = self.cave["edges"]
        if full and len(edges) and len(chirps):
            # Summary "sonar map": every rock edge the hunt's calls reached,
            # brighter the more often it echoed.
            spots = bat[np.minimum(samples - 1, chirps[::max(1, len(chirps) // 150)] // every), :2]
            reach = np.hypot(edges[:, None, 0] - spots[None, :, 0], edges[:, None, 1] - spots[None, :, 1]) < 3.0
            for (ex, ey), hits in zip(edges, reach.sum(axis=1)):
                if hits:
                    g.ellipse((ex * PX - 2, ey * PX - 2, ex * PX + 2, ey * PX + 2), fill=(170, 200, 255, int(min(210, 50 + 25 * hits))))
        for step in chirps[chirps >= now_step - window]:
            s = min(samples - 1, step // every)
            x, y = bat[s, 0], bat[s, 1]
            age = (now_step - step) * DT
            fade = 1.0 if full else max(0.0, 1 - age / 1.6)
            if not full:
                radius = min(ECHO_RANGE, age * 5.5)
                if radius < ECHO_RANGE:
                    g.ellipse(((x - radius) * PX, (y - radius) * PX, (x + radius) * PX, (y + radius) * PX),
                              outline=(*BAT_COLOUR, int(120 * fade)), width=2)
                    if len(edges):
                        lit_edge = np.abs(np.hypot(edges[:, 0] - x, edges[:, 1] - y) - radius) < .22
                        for ex, ey in edges[lit_edge]:
                            g.ellipse((ex * PX - 2, ey * PX - 2, ex * PX + 2, ey * PX + 2), fill=(170, 200, 255, int(200 * fade)))
            # The echoes the bat's ears actually registered for this call.
            dx, dy = moths[s, :, 0] - x, moths[s, :, 1] - y
            dist = np.hypot(dx, dy)
            rel = np.arctan2(dy, dx) - bat[s, 2]
            gain = np.maximum(.5 * (1 + np.cos(rel + EAR_OFFSET)), .5 * (1 + np.cos(rel - EAR_OFFSET)))
            loud = gain * np.clip(1 - dist / ECHO_RANGE, 0, 1) * alive[s]
            for j in np.flatnonzero(loud > .02):
                r = 3 + 9 * loud[j]
                ex, ey = moths[s, j, 0] * PX, moths[s, j, 1] * PX
                g.ellipse((ex - r, ey - r, ex + r, ey + r), fill=(*ECHO_COLOUR, int((60 + 190 * loud[j]) * fade * (.5 if full else 1))))
            fakes = episode["fakes"][step]
            for side, delay in enumerate(fakes):
                if delay >= 0:
                    angle = bat[s, 2] + (EAR_OFFSET if side else -EAR_OFFSET) + (delay - .5) * .8
                    fx, fy = (x + math.cos(angle) * delay * ECHO_RANGE) * PX, (y + math.sin(angle) * delay * ECHO_RANGE) * PX
                    g.ellipse((fx - 9, fy - 9, fx + 9, fy + 9), outline=(*FAKE_COLOUR, int(230 * fade)), width=3)
        blurred = glow.filter(ImageFilter.GaussianBlur(4))
        image = Image.alpha_composite(Image.alpha_composite(image, blurred), glow)
        d = ImageDraw.Draw(image, "RGBA")
        if lit:
            for j in range(moths.shape[1]):
                path = moths[:now + 1, j]
                d.line([(float(px) * PX, float(py) * PX) for px, py in path[-60:]], fill=(*MOTH_COLOUR, 90), width=2)
                mx_, my_ = moths[now, j] * PX
                if alive[now, j]:
                    if jam[now, j]:
                        d.ellipse((mx_ - 16, my_ - 16, mx_ + 16, my_ + 16), outline=(*FAKE_COLOUR, 220), width=2)
                    d.ellipse((mx_ - 6, my_ - 6, mx_ + 6, my_ + 6), fill=(*MOTH_COLOUR, 255), outline=(230, 255, 245, 255))
        # Catches are the one thing both views show.
        for j, step in enumerate(episode["catch_step"]):
            if 0 <= step <= now_step:
                s = min(samples - 1, step // every)
                cx, cy = moths[s, j] * PX
                d.line((cx - 7, cy - 7, cx + 7, cy + 7), fill=(255, 255, 255, 220), width=2)
                d.line((cx - 7, cy + 7, cx + 7, cy - 7), fill=(255, 255, 255, 220), width=2)
        trail = bat[max(0, now - (samples if full else 60)):now + 1]
        d.line([(float(px) * PX, float(py) * PX) for px, py, _ in trail], fill=(*BAT_COLOUR, 150 if full else 120), width=2)
        self._bat(d, *bat[now])
        return image.convert("RGB")

    def arms_race_chart(self, history, generation):
        """Catch rate and moth jamming per generation: the arms race itself."""
        w, h = 560, 300
        image = Image.new("RGB", (w, h), (4, 12, 27))
        d = ImageDraw.Draw(image, "RGBA")
        d.rounded_rectangle((0, 0, w - 1, h - 1), radius=16, outline=(255, 150, 200, 170), width=2)
        d.text((20, 14), "ARMS RACE", font=font(16, True), fill=(236, 246, 255))
        d.text((20, 38), f"generation {generation}", font=font(11), fill=(150, 192, 224))
        left, top, right, bottom = 48, 70, w - 20, h - 40
        d.line((left, bottom, right, bottom), fill=(80, 100, 130), width=1)
        d.line((left, top, left, bottom), fill=(80, 100, 130), width=1)
        for frac in (.5, 1.0):
            y = bottom - (bottom - top) * frac
            d.line((left, y, right, y), fill=(40, 60, 90), width=1)
            d.text((14, y - 6), f"{int(frac * 100)}%", font=font(10), fill=(140, 160, 190))
        series = [("moths caught", "catch_rate", BAT_COLOUR), ("jamming, bat near", "jam_near", FAKE_COLOUR),
                  ("jamming, bat far", "jam_far", (130, 110, 170))]
        total = max(2, self.generations)
        # Shade the head start, when only the bats were evolving.
        head = min(self.freeze, total - 1)
        if head > 0:
            d.rectangle((left + 1, top, left + (right - left) * (head / (total - 1)), bottom - 1), fill=(40, 55, 80, 120))
            d.text((left + 6, top + 4), "moths not yet evolving", font=font(10), fill=(150, 170, 200))
        for label, key, colour in series:
            pts = [(left + (right - left) * (i / (total - 1)), bottom - (bottom - top) * float(np.clip(row[key], 0, 1)))
                   for i, row in enumerate(history)]
            if len(pts) > 1:
                d.line(pts, fill=(*colour, 255), width=3)
            elif pts:
                d.ellipse((pts[0][0] - 3, pts[0][1] - 3, pts[0][0] + 3, pts[0][1] + 3), fill=(*colour, 255))
        x = 20
        for label, _, colour in series:
            d.rectangle((x, h - 24, x + 12, h - 14), fill=colour)
            d.text((x + 17, h - 26), label, font=font(11), fill=(200, 215, 235))
            x += 30 + d.textbbox((0, 0), label, font=font(11))[2]
        return image

    def brains_overlay(self, bats, moths, bat_row, moth_row, result, generation):
        roles = self.catalogues_data["roles"]
        images = []
        for role, pop, row, x, colour in (("bat", bats, bat_row, result["bat_inputs"], BAT_COLOUR),
                                           ("moth", moths, moth_row, result["moth_inputs"], MOTH_COLOUR)):
            genome = pop.genome[row:row + 1]
            weights = [to_numpy(w[0]) for w, _ in pop.layers(genome)]
            source = int(self.champion_cave) if role == "bat" else int(self.champion_cave) * self.k
            _, acts = pop.forward(x[source:source + 1], genome, return_hidden=True)
            brain = self.brains[role]
            images.append(draw_brain(weights, [to_numpy(a[0]) for a in acts], input_labels(brain, roles[role]),
                                     action_labels(brain, roles[role]), title=f"{role.upper()} BRAIN",
                                     subtitle=f"{brain['parameters']} evolved weights · generation {generation}",
                                     footer=f"{brain['points']:g} / {brain['budget']:g} LEGO points",
                                     accent=colour, size=(520, 460)))
        out = Image.new("RGB", (1052, 460), (3, 6, 15))
        out.paste(images[0], (0, 0)); out.paste(images[1], (532, 0))
        return out

    # ---- helpers -------------------------------------------------------
    def episode(self, result, cave):
        """Champion cave's recorded hunt as NumPy arrays."""
        r = result["record"]
        return {"bat": to_numpy(r["bat"][:, cave]), "moth": to_numpy(r["moth"][:, cave]),
                "alive": to_numpy(r["alive"][:, cave]), "jam": to_numpy(r["jam"][:, cave]),
                "chirps": to_numpy(result["chirp_log"][:, cave]), "fakes": to_numpy(result["fake_log"][:, cave]),
                "catch_step": to_numpy(result["catch_step"][cave]), "record_every": result["record_every"]}

    def arena_json(self, generation, ep):
        q = lambda a: np.round(np.asarray(a) * 100).astype(int).tolist()
        every = ep["record_every"]
        chirps = [[int(s // every), [round(float(v), 3) for v in ep["fakes"][s]]] for s in np.flatnonzero(ep["chirps"])]
        catches = [int(s // every) if s >= 0 else -1 for s in ep["catch_step"]]
        jam_mask = (ep["jam"] * (1 << np.arange(ep["jam"].shape[1]))).sum(axis=1).astype(int).tolist()
        return {"kind": "batmoth", "generation": generation, "sample_dt": DT * every,
                "cars": [{"x": q(ep["bat"][:, 0]), "y": q(ep["bat"][:, 1]), "h": np.round(np.degrees(ep["bat"][:, 2])).astype(int).tolist(),
                          "crash": -1, "lap_s": None}],
                "moths": [{"x": q(ep["moth"][:, j, 0]), "y": q(ep["moth"][:, j, 1]), "caught": catches[j]}
                          for j in range(ep["moth"].shape[1])],
                "chirps": chirps, "jam": jam_mask}

    # ---- the run -------------------------------------------------------
    def run(self):
        ctx, s = self.ctx, self.settings
        if ctx.backend_requested.lower() == "auto" and ctx.xp is not np and int(s.get("population", 160)) < GPU_MIN_CAVES:
            ctx.xp = np
            ctx.set_backend_name(f"numpy (auto: under {GPU_MIN_CAVES:,} caves the CUDA step is launch-bound)")
        xp = ctx.xp
        self.catalogues_data = self.catalogues()
        spec = ctx.params.get("_brain")
        self.brains = spec if spec and "bat" in spec else validate_brains(spec or {}, self.catalogues_data)
        self.cave = cave_geometry(int(ctx.params.get("cave", 11)))
        self.k = k = int(ctx.params.get("moths", 4))
        sigma = float(ctx.params.get("mutation", 1.0))
        size = int(s.get("population", 160))
        self.generations = generations = max(1, int(s.get("generations", 30)))
        steps = int(s.get("hunt_steps", 600))
        self.record_every = max(1, int(s.get("record_every", 2)))
        self.freeze = freeze = int(s.get("moth_head_start", 3))
        frames = ctx.frames
        rng = np.random.default_rng(int(ctx.params.get("cave", 11)) * 31 + 5)
        sim = CaveSim(xp, self.cave, self.brains["bat"], self.brains["moth"], k)
        bats = Population(xp, size, self.brains["bat"]["layer_sizes"], seed=rng.integers(1 << 30), sigma=sigma, **EVOLUTION)
        moths = quiet_start(Population(xp, size, self.brains["moth"]["layer_sizes"], seed=rng.integers(1 << 30), sigma=sigma, **EVOLUTION), self.brains["moth"])
        self.dark_bg, self.lit_bg = self.rock_layer(False), self.rock_layer(True)
        folder = ctx.run_dir / "interactive"; folder.mkdir(exist_ok=True)
        (folder / "arena.json").write_text(json.dumps({
            "kind": "batmoth", "world": [WORLD_W, WORLD_H], "rocks": [[round(x, 3), round(y, 3), round(r, 3)] for x, y, r in self.cave["rocks"]],
            "echo_range": ECHO_RANGE, "ear_offset_deg": math.degrees(EAR_OFFSET), "jam_range": JAM_RANGE, "moths": k}))
        plan = [max(1, int(round(generations * (i + 1) / frames))) for i in range(frames)]
        fractions, blocks = [0.0] * frames, {}
        for i, g in enumerate(plan):
            blocks.setdefault(g, []).append(i)
        for g, members in blocks.items():
            for j, i in enumerate(members):
                fractions[i] = (j + 1) / len(members)
        ctx.write_meta({"brain": self.brains, "cave": int(ctx.params.get("cave", 11)),
                        "arena_view": {"folder": "interactive", "kind": "batmoth", "arena": "interactive/arena.json",
                                       "frames": [[g, round(f, 4)] for g, f in zip(plan, fractions)]},
                        "view_modes": [{"id": "frames", "label": "Bat's senses", "folder": "frames"},
                                       {"id": "lit", "label": "Lit cave", "folder": "modes/lit"}],
                        "default_view_mode": "frames"})
        for sub in ("overlays/network", "overlays/arms_race", "modes/lit"):
            (ctx.run_dir / sub).mkdir(parents=True, exist_ok=True)
        done, history, current, streak, evolved_at, evaluations = 0, [], None, 0, None, 0
        for i in range(frames):
            while done < plan[i]:
                done += 1
                if current is not None:
                    bats.evolve(current["result"]["bat_fitness"])
                    if done > freeze:
                        moths.evolve(current["moth_fitness"])
                moth_index = assign_moths(rng, 1, size, size, k)
                result = self.hunt(sim, bats, moths, moth_index, steps, seed=int(rng.integers(1 << 30)))
                fit_m = moth_fitness(xp, result["slot_fitness"], moth_index, size)
                bat_fit = to_numpy(result["bat_fitness"])
                self.champion_cave = cave = int(np.argmax(bat_fit))
                catches = to_numpy(result["catch_step"]) >= 0
                chirps = float(to_numpy(result["chirps"]).sum())
                jam_near, jam_far = jamming_rates(result)
                stats = {"generation": done, "catch_rate": float(catches.mean()),
                         "jammed_calls": float(to_numpy(result["jammed_chirps"]).sum()) / max(1.0, chirps),
                         "jam_near": jam_near, "jam_far": jam_far,
                         "chirps_per_s": chirps / size / (steps * DT)}
                has_jam = "jam" in self.brains["moth"]["actions"]
                streak = streak + 1 if has_jam and selective(jam_near, jam_far) else 0
                if streak >= 3 and evolved_at is None:
                    evolved_at = done
                stats["jamming_evolved"] = evolved_at is not None
                history.append(stats)
                evaluations += size * steps
                moth_row = int(np.argmax(fit_m))
                current = {"result": result, "moth_fitness": fit_m, "cave": cave, "moth_row": moth_row,
                           "episode": self.episode(result, cave), "stats": stats}
                (folder / f"gen_{done:04d}.json").write_text(json.dumps(self.arena_json(done, current["episode"])))
            ep, stats = current["episode"], current["stats"]
            ctx.save_frame(self.render_frame(ep, fractions[i], False), ctx.frame_path(i))
            ctx.save_frame(self.render_frame(ep, fractions[i], True), ctx.run_dir / "modes" / "lit" / f"frame_{i:04d}.jpg")
            ctx.save_frame(self.brains_overlay(bats, moths, current["cave"], current["moth_row"], current["result"], done),
                           ctx.run_dir / "overlays" / "network" / f"frame_{i:04d}.jpg")
            ctx.save_frame(self.arms_race_chart(history, done), ctx.run_dir / "overlays" / "arms_race" / f"frame_{i:04d}.jpg")
            ctx.write_status(i, f"generation {done} of {generations}", {
                "generation": f"{done} / {generations}" + (" · moths waiting" if done <= freeze else ""),
                "caves hunting at once": f"{size:,}",
                "moths caught": f"{100 * stats['catch_rate']:.0f}%",
                "bat calls per second": f"{stats['chirps_per_s']:.1f}",
                "calls jammed": f"{100 * stats['jammed_calls']:.0f}%",
                "moths jamming · bat near / far": f"{100 * stats['jam_near']:.0f}% / {100 * stats['jam_far']:.0f}%",
                "selective jamming evolved": f"yes, generation {evolved_at}" if evolved_at else ("not yet" if "jam" in self.brains["moth"]["actions"] else "moths have no jam block"),
                "bat brain": f"{self.brains['bat']['parameters']} weights · {self.brains['bat']['points']:g}/{self.brains['bat']['budget']:g} pts",
                "moth brain": f"{self.brains['moth']['parameters']} weights · {self.brains['moth']['points']:g}/{self.brains['moth']['budget']:g} pts",
                "bat-steps simulated": f"{evaluations:,}",
            })
        reveal = self.reveal(sigma, steps, bats, moths)
        ctx.write_meta({"generation_stats": history,
                        "summary": {"cave": int(ctx.params.get("cave", 11)), "catch_rate": round(history[-1]["catch_rate"], 3),
                                    "jamming_evolved_at": evolved_at, "bat_points": self.brains["bat"]["points"],
                                    "moth_points": self.brains["moth"]["points"], "generations": generations}})
        ctx.finish(reveal)

    # ---- the reveal ----------------------------------------------------
    def reveal(self, sigma, steps, bats_final, moths_final):
        """The evolved bats and moths released into new, unseen caves.

        Each island starts from a random sample of the final populations and
        keeps co-evolving on its own in a procedurally different cave.  Tiles
        show whether the hunting and the jamming carry over to new terrain.
        Islands never exchange genomes; each cave needs its own distance field,
        so islands run one after another, each fully batched.
        """
        xp, s = self.ctx.xp, self.settings
        islands = max(1, int(s.get("ensemble", 9)))
        size = max(4, int(s.get("reveal_population", 40)))
        generations = max(1, int(s.get("reveal_generations", 10)))
        cols = max(1, int(math.ceil(math.sqrt(islands)))); rows = int(math.ceil(islands / cols)); gap = 8
        tile_w, tile_h = (1280 - gap * (cols + 1)) // cols, (720 - gap * (rows + 1)) // rows
        out = Image.new("RGB", (1280, 720), (3, 6, 15)); d = ImageDraw.Draw(out, "RGBA")
        label_size = max(10, min(18, tile_h // 7)); tiles = []
        main_cave, main_bg = self.cave, (self.dark_bg, self.lit_bg)
        base = int(self.ctx.params.get("cave", 11))
        for island in range(islands):
            seed = base * 101 + island + 1
            self.cave = cave_geometry(seed)
            self.dark_bg, self.lit_bg = self.rock_layer(False), self.rock_layer(True)
            rng = np.random.default_rng(seed)
            sim = CaveSim(xp, self.cave, self.brains["bat"], self.brains["moth"], self.k)
            bats = Population(xp, size, self.brains["bat"]["layer_sizes"], seed=seed, sigma=sigma, **EVOLUTION)
            moths = Population(xp, size, self.brains["moth"]["layer_sizes"], seed=seed + 7, sigma=sigma, **EVOLUTION)
            bats.genome = bats_final.genome[xp.asarray(rng.integers(0, bats_final.size, size))].copy()
            moths.genome = moths_final.genome[xp.asarray(rng.integers(0, moths_final.size, size))].copy()
            result, fit_m, jam_history = None, None, []
            for g in range(generations):
                if result is not None:
                    bats.evolve(result["bat_fitness"])
                    moths.evolve(fit_m)
                index = assign_moths(rng, 1, size, size, self.k)
                result = sim.run(bats, moths, index, steps, int(rng.integers(1 << 30)), self.record_every)
                fit_m = moth_fitness(xp, result["slot_fitness"], index, size)
                jam_history.append(jamming_rates(result))
                self.ctx.write_status(self.ctx.frames - 1, f"scale reveal: new cave {island + 1}/{islands}, generation {g + 1}/{generations}")
            catch = float((to_numpy(result["catch_step"]) >= 0).mean())
            jammed = "jam" in self.brains["moth"]["actions"] and selective(*np.mean(jam_history[-3:], axis=0))
            ep = self.episode(result, int(np.argmax(to_numpy(result["bat_fitness"]))))
            tile = self.render_frame(ep, 1.0, True).resize((tile_w, tile_h), Image.Resampling.LANCZOS)
            r, c = divmod(island, cols); x0, y0 = gap + c * (tile_w + gap), gap + r * (tile_h + gap)
            out.paste(tile, (x0, y0))
            text = f"cave {island + 1} · {100 * catch:.0f}% caught" + (" · jamming evolved" if jammed else "")
            d.rectangle((x0, y0 + tile_h - label_size - 8, x0 + tile_w, y0 + tile_h), fill=(2, 5, 15, 200))
            d.text((x0 + 6, y0 + tile_h - label_size - 5), text, font=font(label_size, True), fill=(255, 210, 150))
            tiles.append({"cave": seed, "catch_rate": round(catch, 3), "jamming": bool(jammed)})
        self.cave, (self.dark_bg, self.lit_bg) = main_cave, main_bg
        path = self.ctx.run_dir / "reveal.jpg"
        self.ctx.save_frame(out, path)
        self.ctx.write_meta({"reveal_tiles": tiles, "reveal_population": size * islands, "reveal_generations": generations})
        return path
