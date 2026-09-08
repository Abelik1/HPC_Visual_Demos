from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..backend import to_numpy
from ..base import Demo
from ..render import add_progress, add_title, font, mosaic


ATOM_COLOURS = np.array(
    [
        [92, 226, 255],
        [255, 158, 76],
        [212, 105, 255],
        [116, 244, 170],
    ],
    dtype=np.uint8,
)


class MolecularDynamicsDemo(Demo):
    timing_methods={"initialise":"initialization","step":"simulation","hero":"render"}
    """Coarse-grained polymer dynamics with all-pairs non-bonded forces."""

    id = "molecular_dynamics"
    title = "Molecular Machine"

    def budget(self):
        if "total_steps" in self.settings:
            return max(1, int(self.settings["total_steps"]))
        return max(1, int(self.settings.get("steps_per_frame", 10)) * self.ctx.frames)

    def initialise(self, particles, seed=7, sequence=0, temperature=310.0):
        rng = np.random.default_rng(seed)
        n = int(particles)
        t = np.linspace(0, 5.5 * np.pi, n)
        # A loose helical chain gives the dynamics room to compact without
        # beginning in a numerically dangerous straight-line configuration.
        pos = np.stack(
            [1.65 * np.cos(t), 1.65 * np.sin(t), np.linspace(-1.45, 1.45, n)],
            axis=1,
        )
        pos += rng.normal(0, 0.035, pos.shape)
        speed = 0.18 * math.sqrt(float(temperature) / 310.0)
        vel = rng.normal(0, speed, pos.shape)
        vel -= vel.mean(axis=0, keepdims=True)
        seq = int(sequence) % 4
        if seq == 0:
            types = np.arange(n) % 4
        elif seq == 1:
            types = (np.arange(n) // max(2, n // 12)) % 4
        elif seq == 2:
            types = rng.integers(0, 4, n)
        else:
            types = (np.arange(n) * 3 + (np.arange(n) // 5)) % 4
        xp = self.ctx.xp
        return (
            xp.asarray(pos.astype(np.float32)),
            xp.asarray(vel.astype(np.float32)),
            xp.asarray(types.astype(np.int32)),
        )

    def _force_masks(self, n):
        xp = self.ctx.xp
        cache = getattr(self, "_mask_cache", {})
        key = (n, self.ctx.backend_name)
        if key not in cache:
            ii = np.arange(n)[:, None]
            jj = np.arange(n)[None, :]
            nonbond = np.abs(ii - jj) > 1
            cache[key] = xp.asarray(nonbond)
            self._mask_cache = cache
        return cache[key]

    def forces(self, pos, types, attraction, solvent_quality):
        """Harmonic bonds plus a softened, type-dependent Lennard-Jones force."""
        xp = self.ctx.xp
        n = pos.shape[0]
        delta = pos[:, None, :] - pos[None, :, :]
        dist2 = xp.sum(delta * delta, axis=-1)
        safe2 = xp.maximum(dist2, 0.105)
        sigma = 0.34
        sr2 = sigma * sigma / safe2
        sr6 = xp.minimum(sr2 * sr2 * sr2, 18.0)
        hydrophobic = ((types[:, None] + types[None, :]) % 3 == 0).astype(xp.float32)
        epsilon = float(attraction) * (0.065 + 0.075 * float(solvent_quality) * hydrophobic)
        coeff = 24.0 * epsilon * (2.0 * sr6 * sr6 - sr6) / safe2
        coeff *= self._force_masks(n)
        force = xp.sum(coeff[..., None] * delta, axis=1)

        bonds = pos[1:] - pos[:-1]
        lengths = xp.sqrt(xp.sum(bonds * bonds, axis=1) + 1e-8)
        rest = 0.405
        bond_force = 36.0 * (lengths - rest)[:, None] * bonds / lengths[:, None]
        force[:-1] += bond_force
        force[1:] -= bond_force

        # Weak centring emulates a finite solvent volume and prevents an
        # uninteresting centre-of-mass drift across the camera.
        force -= 0.025 * pos
        xp.clip(force, -75, 75, out=force)
        return force

    def step(self, pos, vel, types, temperature, attraction, solvent_quality, steps):
        xp = self.ctx.xp
        dt = 0.006
        target_speed = 0.18 * math.sqrt(float(temperature) / 310.0)
        damping = 0.996 - 0.0015 * float(solvent_quality)
        for _ in range(steps):
            force = self.forces(pos, types, attraction, solvent_quality)
            vel = damping * vel + dt * force
            current = xp.sqrt(xp.mean(vel * vel) + 1e-9)
            # A gentle deterministic thermostat avoids backend-specific random
            # streams while still holding the requested kinetic temperature.
            vel *= 0.985 + 0.015 * target_speed / current
            pos += dt * vel
            pos -= xp.mean(pos, axis=0, keepdims=True)
        return pos, vel

    @staticmethod
    def diagnostics(pos):
        p = np.asarray(pos, dtype=np.float32)
        centre = p.mean(axis=0)
        rg = float(np.sqrt(np.mean(np.sum((p - centre) ** 2, axis=1))))
        delta = p[:, None, :] - p[None, :, :]
        dist = np.sqrt(np.sum(delta * delta, axis=-1) + np.eye(len(p)) * 1e6)
        contacts = int(np.sum((dist < 0.62) & (dist > 0.01)) // 2)
        return rg, contacts

    @staticmethod
    def rotate(pos, angle):
        ca, sa = math.cos(angle), math.sin(angle)
        cb, sb = math.cos(0.38), math.sin(0.38)
        ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]], dtype=np.float32)
        rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]], dtype=np.float32)
        return np.asarray(pos) @ ry.T @ rx.T

    def molecule_image(self, pos, types, size=(820, 650), angle=0.0, compact=False):
        p = self.rotate(np.asarray(pos), angle)
        types = np.asarray(types, dtype=int)
        w, h = size
        scale = min(w, h) * (0.19 if not compact else 0.20)
        px = w * 0.5 + p[:, 0] * scale
        py = h * 0.5 - p[:, 1] * scale
        order = np.argsort(p[:, 2])
        base = Image.new("RGB", size, (2, 5, 14))
        glow = Image.new("RGBA", size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        atom_r = max(2, int(min(w, h) * (0.017 if not compact else 0.022)))

        # Bonds are drawn first; atom depth sorting then gives a readable 3-D
        # ball-and-stick structure without requiring an OpenGL context.
        for i in range(len(p) - 1):
            depth = (p[i, 2] + p[i + 1, 2]) * 0.5
            alpha = int(np.clip(105 + 32 * depth, 55, 205))
            gd.line((px[i], py[i], px[i + 1], py[i + 1]), fill=(170, 210, 235, alpha), width=max(1, atom_r // 3))
        for j in order:
            colour = ATOM_COLOURS[types[j] % len(ATOM_COLOURS)]
            rr = atom_r * (0.85 + 0.12 * np.clip(p[j, 2], -1, 1))
            gd.ellipse((px[j] - rr * 1.8, py[j] - rr * 1.8, px[j] + rr * 1.8, py[j] + rr * 1.8), fill=(*map(int, colour), 72))
        halo = glow.filter(ImageFilter.GaussianBlur(max(2, atom_r)))
        base = Image.alpha_composite(base.convert("RGBA"), halo)
        sharp = Image.new("RGBA", size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp, "RGBA")
        for i in range(len(p) - 1):
            sd.line((px[i], py[i], px[i + 1], py[i + 1]), fill=(194, 220, 238, 155), width=max(1, atom_r // 3))
        for j in order:
            colour = ATOM_COLOURS[types[j] % len(ATOM_COLOURS)]
            depth_light = float(np.clip(0.66 + 0.22 * p[j, 2], 0.42, 1.0))
            c = tuple(int(np.clip(v * depth_light, 0, 255)) for v in colour)
            rr = atom_r * (0.85 + 0.12 * np.clip(p[j, 2], -1, 1))
            sd.ellipse((px[j] - rr, py[j] - rr, px[j] + rr, py[j] + rr), fill=(*c, 245), outline=(235, 248, 255, 165), width=1)
            if not compact and rr >= 5:
                sd.ellipse((px[j] - rr * 0.42, py[j] - rr * 0.48, px[j] - rr * 0.05, py[j] - rr * 0.11), fill=(255, 255, 255, 145))
        return Image.alpha_composite(base, sharp).convert("RGB")

    def hero(self, pos, types, i, done, total, temperature, attraction, solvent, sequence):
        pn, tn = to_numpy(pos), to_numpy(types)
        canvas = self.molecule_image(pn, tn, size=(1280,720), angle=0.012 * i)
        canvas = add_title(
            canvas,
            "Molecular Machine",
            f"coarse-grained 3-D molecular dynamics · {len(pn)} particles · all-pairs forces · {self.ctx.backend_name}",
            badge="LIVE MOLECULAR DYNAMICS",
        )
        add_progress(canvas, done / total, "LOOSE CHAIN", "FOLDED ENSEMBLE")
        return canvas

    def run(self):
        n = int(self.settings["particles"])
        total = self.budget()
        temperature = float(self.ctx.params.get("temperature", 310.0))
        attraction = float(self.ctx.params.get("attraction", 1.0))
        solvent = float(self.ctx.params.get("solvent", 0.65))
        sequence = int(self.ctx.params.get("sequence", 0))
        pos, vel, types = self.initialise(n, sequence=sequence, temperature=temperature)
        done = 0
        for i in range(self.ctx.frames):
            target = int(round(total * (i + 1) / self.ctx.frames))
            pos, vel = self.step(pos, vel, types, temperature, attraction, solvent, max(1, target - done))
            done = target
            image = self.hero(pos, types, i, done, total, temperature, attraction, solvent, sequence)
            self.ctx.save_frame(image, self.ctx.frame_path(i))
            rg, contacts = self.diagnostics(to_numpy(pos)); pair_evals=done*n*(n-1)//2
            self.ctx.write_status(i, f"molecular step {done:,} · Rg {rg:.3f}",{
                "sequence":chr(65+sequence%4),"temperature":f"{temperature:.0f} K","attraction":f"{attraction:.2f} ε",
                "solvent quality":f"{solvent:.2f}","radius of gyration":f"{rg:.3f}",
                "close contacts":f"{contacts:,}","integration step":f"{done:,} / {total:,}","pair evaluations":f"{pair_evals:,}"})

        ens = max(1, int(self.settings.get("ensemble", 16)))
        side = max(1, int(math.ceil(math.sqrt(ens))))
        sweep_particles = int(self.settings.get("sweep_particles", max(28, n // 2)))
        sweep_steps = int(self.settings.get("sweep_steps", max(220, total // 2)))
        temperatures = np.linspace(270, 390, side)
        attractions = np.linspace(0.55, 1.65, side)
        tiles, labels = [], []
        for j in range(ens):
            row,col=divmod(j,side); temp=temperatures[row]; attr=attractions[col]
            pj, vj, tj = self.initialise(
                sweep_particles,
                seed=70 + j,
                sequence=(sequence + row + col) % 4,
                temperature=float(temp),
            )
            pj, vj = self.step(pj, vj, tj, float(temp), float(attr), solvent, sweep_steps)
            pn, tn = to_numpy(pj), to_numpy(tj)
            rg, _ = self.diagnostics(pn)
            tiles.append(self.molecule_image(pn, tn, size=(260, 146), angle=0.45, compact=True))
            labels.append(f"{temp:.0f}K · ε{attr:.2f} · Rg{rg:.2f}")
        reveal = mosaic(
            tiles,
            side,
            title="Actually… we ran a virtual molecular laboratory",
            subtitle="Temperature increases downward; attraction increases to the right. Every structure is an independent trajectory.",
            labels=labels,
            label_fill=(199, 233, 255),
        )
        rp = self.ctx.run_dir / "reveal.jpg"
        self.ctx.save_frame(reveal, rp)
        self.ctx.finish(rp)
