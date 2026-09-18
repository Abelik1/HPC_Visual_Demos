from __future__ import annotations

import base64
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..backend import to_numpy
from ..base import Demo
from ..colors import palette
from ..plasma_control import (
    AXIS_GAIN, INPUT_NAMES, OUTPUT_NAMES, START_STATE, ConfinedParticles, FrozenController,
    make_trainer, risk_of, sample_periodic, save_controller, transition_numpy,
)
from ..render import add_progress, add_title, font, mosaic
from ..pipeline import FramePipeline


# AXIS_GAIN lives in plasma_control so the renderer and the training objective
# cannot drift apart.  A fully diverged uncontrolled column therefore sits at
# 0.8 of the way to the wall and loses nearly every marker.
SPARK_LIFE = 9.0

# Radius-to-colour ramp for the confined markers: cold core, hot edge.
PARTICLE_RAMP = (
    (0.00, (255, 96, 64)),     # touching the wall
    (0.09, (255, 166, 72)),
    (0.19, (168, 255, 206)),
    (0.34, (118, 224, 255)),
    (1.00, (152, 216, 255)),   # deep in the core
)


class FusionPlasmaDemo(Demo):
    """Two views of the same magnetic bottle.

    ``passive`` integrates a complex Ginzburg--Landau amplitude field on a
    periodic lattice, maps it onto a torus and advects passive tracers through
    the drift derived from that field.  The equation is an exhibition-scale
    amplitude model, not a reactor prediction, but every lattice point takes
    part in a nonlinear diffusive update and its ordered-wave to
    defect-turbulence transition is fast and visually unmistakable.

    ``guardian`` keeps that field as the turbulence source and adds a reduced,
    differentiable control environment: a small neural policy trained through
    batches of virtual shots commands three aggregate coil banks, those
    commands shape the confining field inside the torus, and a population of
    guiding-centre-like markers is confined - or lost to the wall - as a
    result.  See docs/SCIENTIFIC_NOTES.md for the boundaries of both models.
    """

    id = "fusion_plasma"
    title = "Star in a Bottle"
    aspect = 16 / 9

    methods = ("passive", "guardian")
    default_method = "passive"
    method_labels = {
        "passive": "Mode 1 · Passive confinement",
        "guardian": "Mode 2 · AI plasma guardian (3D)",
    }
    method_descriptions = {
        "passive": "The plasma-wave field alone: tracers circle the torus on the drift it produces.",
        "guardian": "A neural policy commands the coils that shape the field inside the torus; markers it fails to hold reach the wall and spark.",
    }
    timing_methods = {"initialise": "initialization", "step": "simulation",
                      "train": "simulation", "advance_particles": "simulation",
                      "hero": "render", "guardian_hero": "render"}

    # ---- shared field solver -------------------------------------------
    def mode(self):
        method = getattr(self.ctx, "method", self.default_method)
        return method if method in self.methods else self.default_method

    def grid_shape(self, n):
        ny = max(20, int(n))
        return ny, max(32, int(round(ny * self.aspect)))

    def budget(self):
        if "total_steps" in self.settings:
            return max(1, int(self.settings["total_steps"]))
        return max(1, int(self.settings.get("steps_per_frame", 8)) * self.ctx.frames)

    def initialise(self, ny, nx, seed=12):
        rng = np.random.default_rng(seed)
        y, x = np.mgrid[0:ny, 0:nx]
        phase = 2 * np.pi * (x / nx * 2.0 + y / ny)
        envelope = 0.48 + 0.18 * np.cos(2 * np.pi * y / ny)
        real = envelope * np.cos(phase) + rng.normal(0, 0.055, (ny, nx))
        imag = envelope * np.sin(phase) + rng.normal(0, 0.055, (ny, nx))
        xp = self.ctx.xp
        return (
            xp.asarray(real.astype(np.float32)),
            xp.asarray(imag.astype(np.float32)),
        )

    def step(self, real, imag, magnetic_field, heating, density, steps):
        """Integrate a complex Ginzburg--Landau amplitude equation.

        Magnetic confinement shifts the dispersion coefficients and reduces
        the effective drive. Heating increases nonlinear drive; density adds
        damping. The coefficients are deliberately dimensionless.
        """
        xp = self.ctx.xp
        b = float(magnetic_field)
        heat = float(heating)
        dens = float(density)
        c1 = 0.55 + 2.1 / (b + 1.2)
        c3 = 0.45 + 0.032 * heat / max(0.45, dens)
        drive = 0.74 + 0.015 * heat
        damping = 0.36 + 0.22 * dens + 0.08 * b
        dt = 0.075
        for _ in range(steps):
            lap_r = 0.25 * (
                xp.roll(real, 1, 0) + xp.roll(real, -1, 0)
                + xp.roll(real, 1, 1) + xp.roll(real, -1, 1)
                - 4 * real
            )
            lap_i = 0.25 * (
                xp.roll(imag, 1, 0) + xp.roll(imag, -1, 0)
                + xp.roll(imag, 1, 1) + xp.roll(imag, -1, 1)
                - 4 * imag
            )
            amp2 = real * real + imag * imag
            dr = drive * real + lap_r - c1 * lap_i - amp2 * real + c3 * amp2 * imag
            di = drive * imag + lap_i + c1 * lap_r - amp2 * imag - c3 * amp2 * real
            real += dt * (dr - 0.12 * damping * real)
            imag += dt * (di - 0.12 * damping * imag)
            # A tiny poloidal shear represents the E x B rotation that winds
            # structures around a tokamak. It is part of the state update,
            # not a rendering effect.
            shear = 0.012 * (heat / 25.0) / max(0.6, b / 4.0)
            real += shear * (xp.roll(real, 1, 1) - xp.roll(real, -1, 1))
            imag += shear * (xp.roll(imag, 1, 1) - xp.roll(imag, -1, 1))
            xp.clip(real, -2.5, 2.5, out=real)
            xp.clip(imag, -2.5, 2.5, out=imag)
        return real, imag

    @staticmethod
    def turbulence(real, imag):
        xp = _xp_of(real)
        amp = xp.sqrt(real * real + imag * imag)
        phase = xp.arctan2(imag, real)
        wrap = lambda d: xp.arctan2(xp.sin(d), xp.cos(d))
        dx = wrap(xp.roll(phase, 1, 1) - phase)
        dy = wrap(xp.roll(phase, 1, 0) - phase)
        amp_grad = xp.abs(xp.roll(amp, 1, 0) - amp) + xp.abs(xp.roll(amp, 1, 1) - amp)
        return float(xp.mean(xp.abs(dx) + xp.abs(dy)) / np.pi + 0.25 * xp.mean(amp_grad))

    @staticmethod
    def turbulence_map(real, imag):
        """Local incoherence of the solved field, normalised to [0, 1].

        Defects and phase slips in the amplitude equation become the regions
        that transport markers outward in guardian mode, so the confinement
        losses are driven by the field the solver actually produced.
        """
        real, imag = to_numpy(real), to_numpy(imag)
        amp = np.sqrt(real * real + imag * imag)
        phase = np.arctan2(imag, real)
        dx = np.abs(np.angle(np.exp(1j * (np.roll(phase, -1, 1) - phase))))
        dy = np.abs(np.angle(np.exp(1j * (np.roll(phase, -1, 0) - phase))))
        grad = np.abs(np.roll(amp, -1, 1) - amp) + np.abs(np.roll(amp, -1, 0) - amp)
        raw = (dx + dy) / np.pi + 0.9 * grad
        high = float(np.percentile(raw, 97)) or 1.0
        return np.clip(raw / max(1e-3, high), 0.0, 1.0).astype(np.float32)

    # ---- passive tracers ------------------------------------------------
    @staticmethod
    def initialise_tracers(count, trail, seed=91):
        """Seed passive tracer histories in periodic toroidal coordinates."""
        rng = np.random.default_rng(seed)
        head = rng.random((max(1, int(count)), 2)).astype(np.float32)
        return np.repeat(head[:, None, :], max(2, int(trail)), axis=1)

    @staticmethod
    def flow_field(real, imag, magnetic_field, heating):
        """Derive a passive drift field from the simulated complex amplitude.

        Phase gradients provide wave propagation while the rotated amplitude
        gradient supplies an E x B-like drift around coherent structures. The
        result is measured in turns of the torus per solver step.
        """
        # Computed wherever the field lives: copying a 1536^2 complex field to
        # the host every frame cost more than the solver itself.
        xp = _xp_of(real)
        amp = xp.sqrt(real * real + imag * imag)
        phase = xp.arctan2(imag, real)
        wrap = lambda d: xp.arctan2(xp.sin(d), xp.cos(d))
        phase_x = 0.5 * wrap(xp.roll(phase, -1, 1) - xp.roll(phase, 1, 1))
        phase_y = 0.5 * wrap(xp.roll(phase, -1, 0) - xp.roll(phase, 1, 0))
        amp_x = 0.5 * (xp.roll(amp, -1, 1) - xp.roll(amp, 1, 1))
        amp_y = 0.5 * (xp.roll(amp, -1, 0) - xp.roll(amp, 1, 0))
        confinement = max(0.55, math.sqrt(float(magnetic_field) / 5.0))
        drive = max(0.25, float(heating) / 25.0)
        toroidal = 0.00105 * drive / confinement
        flow_u = toroidal + 0.0024 * phase_x / np.pi - 0.0042 * amp_y / confinement
        flow_v = 0.0019 * phase_y / np.pi + 0.0042 * amp_x / confinement
        return flow_u.astype(xp.float32), flow_v.astype(xp.float32)

    @staticmethod
    def _sample_periodic(field, uv):
        """Bilinearly sample a 2-D periodic field at normalised (u, v)."""
        xp = _xp_of(field)
        if xp is np:
            return sample_periodic(field, uv[:, 0], uv[:, 1])
        ny, nx = field.shape
        fx = xp.mod(uv[:, 0], 1.0) * nx
        fy = xp.mod(uv[:, 1], 1.0) * ny
        x0 = xp.floor(fx).astype(xp.int32) % nx
        y0 = xp.floor(fy).astype(xp.int32) % ny
        x1, y1 = (x0 + 1) % nx, (y0 + 1) % ny
        ax, ay = fx - xp.floor(fx), fy - xp.floor(fy)
        return (field[y0, x0] * (1 - ax) * (1 - ay) + field[y0, x1] * ax * (1 - ay)
                + field[y1, x0] * (1 - ax) * ay + field[y1, x1] * ax * ay)

    def advance_tracers(self, trails, real, imag, magnetic_field, heating, solver_steps):
        """Advect passive tracers through the current simulated flow field."""
        flow_u, flow_v = self.flow_field(real, imag, magnetic_field, heating)
        xp = _xp_of(flow_u)
        trails = xp.asarray(trails)
        head = trails[:, -1, :].copy()
        substeps = max(1, min(8, int(math.ceil(max(1, solver_steps) / 8))))
        dt = float(max(1, solver_steps)) / substeps
        sampled_speed = 0.0
        for _ in range(substeps):
            du = self._sample_periodic(flow_u, head)
            dv = self._sample_periodic(flow_v, head)
            head[:, 0] = xp.mod(head[:, 0] + dt * du, 1.0)
            head[:, 1] = xp.mod(head[:, 1] + dt * dv, 1.0)
            sampled_speed += float(xp.mean(xp.sqrt(du * du + dv * dv)))
        trails = xp.roll(trails, -1, axis=1)
        trails[:, -1, :] = head
        return trails, sampled_speed / substeps

    # ---- geometry -------------------------------------------------------
    @staticmethod
    def _project_points(u, a, b, size, angle, minor=0.40):
        """Project toroidal coordinates to screen x/y and depth.

        ``u`` is the toroidal angle in turns; ``a`` and ``b`` place the point
        inside the poloidal cross-section in units of the wall minor radius,
        ``a`` outward along the major radius and ``b`` along the machine axis.
        """
        phi = np.asarray(u) * 2 * np.pi + angle
        major = 1.0
        radius = major + minor * np.asarray(a)
        x = radius * np.cos(phi)
        y = radius * np.sin(phi)
        z = minor * np.asarray(b)
        tilt = 0.92
        yy = y * math.cos(tilt) - z * math.sin(tilt)
        zz = y * math.sin(tilt) + z * math.cos(tilt)
        w, h = size
        scale = min(w / 3.15, h / 2.25)
        return w * 0.5 + x * scale, h * 0.52 - yy * scale, zz

    @classmethod
    def _project_torus(cls, uv, size, angle):
        """Project normalised toroidal surface coordinates (u, v)."""
        uv = np.asarray(uv)
        v = uv[..., 1] * 2 * np.pi
        return cls._project_points(uv[..., 0], np.cos(v), np.sin(v), size, angle)

    @staticmethod
    def _ramp(values, stops=PARTICLE_RAMP):
        """Piecewise-linear RGB ramp over normalised values."""
        values = np.clip(np.asarray(values, dtype=np.float32), 0, 1)
        positions = np.array([stop[0] for stop in stops], dtype=np.float32)
        colours = np.array([stop[1] for stop in stops], dtype=np.float32)
        index = np.clip(np.searchsorted(positions, values, side="right") - 1, 0, len(stops) - 2)
        span = np.maximum(1e-6, positions[index + 1] - positions[index])
        t = ((values - positions[index]) / span)[:, None]
        return colours[index] * (1 - t) + colours[index + 1] * t

    # ---- torus surface --------------------------------------------------
    def _shell_points(self, real, imag, size, angle, compact):
        """Depth-sorted coloured points sampling the plasma surface."""
        ny, nx = real.shape
        # Only the sampled lattice points are drawn; copy just those.
        pre = max(1, int(math.ceil(ny / (80 if not compact else 45))))
        real, imag = to_numpy(real[::pre, ::pre]), to_numpy(imag[::pre, ::pre])
        ny, nx = real.shape
        amp = np.sqrt(real * real + imag * imag)
        phase = (np.arctan2(imag, real) + np.pi) / (2 * np.pi)
        texture = np.clip(0.62 * amp / 1.25 + 0.38 * phase, 0, 1)
        rgb = palette(texture, "plasma", normalize_input=False)
        # Sampling every second lattice row keeps the point renderer quick at
        # Leonardo resolutions while still deriving every colour from state.
        stride = 1
        vv, uu = np.mgrid[0:ny:stride, 0:nx:stride]
        vv, uu = vv.ravel(), uu.ravel()
        v = 2 * np.pi * vv / ny
        px, py, zz = self._project_points(uu / nx, np.cos(v), np.sin(v), size, angle)
        cols = rgb[vv, uu].astype(np.float32)
        light = np.clip(0.45 + 0.55 * (zz + 1.1) / 2.2, 0.35, 1.0)[:, None]
        cols = np.clip(cols * light + np.array([12, 5, 25]), 0, 255).astype(np.uint8)
        return px, py, zz, cols

    def _paint_shell(self, glow_draw, sharp_draw, points, size, compact,
                     alpha=1.0, order=None):
        px, py, zz, cols = points
        w, h = size
        radius = max(1, int(min(w, h) / (260 if compact else 285)))
        order = np.argsort(zz) if order is None else order
        for j in order:
            c = tuple(int(q) for q in cols[j])
            rr = radius + (1 if zz[j] > 0.25 and not compact else 0)
            glow_draw.ellipse((px[j] - rr * 2, py[j] - rr * 2, px[j] + rr * 2, py[j] + rr * 2),
                              fill=(*c, max(4, int(38 * alpha))))
            sharp_draw.ellipse((px[j] - rr, py[j] - rr, px[j] + rr, py[j] + rr),
                               fill=(*c, max(6, int((145 if not compact else 190) * alpha))))
        return radius

    def _draw_tracers(self, image, trails, size, angle):
        """Draw luminous, depth-aware tracer ribbons over the torus field."""
        if trails is None or not len(trails):
            return image
        px, py, depth = self._project_torus(trails, size, angle)
        w, h = size
        glow = Image.new("RGBA", size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        sharp = Image.new("RGBA", size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp, "RGBA")
        colours = ((100, 239, 255), (255, 171, 83), (225, 126, 255), (188, 255, 228))
        steps = trails.shape[1]
        for particle in range(len(trails)):
            colour = colours[particle % len(colours)]
            for k in range(1, steps):
                x0, y0, x1, y1 = px[particle, k-1], py[particle, k-1], px[particle, k], py[particle, k]
                if abs(x1-x0) > w * 0.22 or abs(y1-y0) > h * 0.22:
                    continue
                age = k / max(1, steps - 1)
                front = np.clip(0.40 + 0.60 * (depth[particle, k] + 0.65) / 1.3, 0.28, 1.0)
                alpha = int((25 + 180 * age) * front)
                gd.line((x0, y0, x1, y1), fill=(*colour, max(12, alpha // 2)), width=7)
                sd.line((x0, y0, x1, y1), fill=(*colour, alpha), width=2 if front < 0.7 else 3)
            hx, hy = px[particle, -1], py[particle, -1]
            front = np.clip(0.45 + 0.55 * (depth[particle, -1] + 0.65) / 1.3, 0.32, 1.0)
            radius = 2.4 + 2.2 * front
            gd.ellipse((hx-radius*2.4, hy-radius*2.4, hx+radius*2.4, hy+radius*2.4), fill=(*colour, 105))
            sd.ellipse((hx-radius, hy-radius, hx+radius, hy+radius), fill=(245, 253, 255, int(225*front)), outline=(*colour, 245), width=1)
        glow = glow.filter(ImageFilter.GaussianBlur(4))
        return Image.alpha_composite(Image.alpha_composite(image.convert("RGBA"), glow), sharp).convert("RGB")

    def torus_image(self, real, imag, size=(820, 650), angle=0.0, compact=False, trails=None):
        """Project the simulated periodic field onto a luminous 3-D torus."""
        # Rendering is deliberately the CPU boundary; CuPy forbids implicit
        # conversion, so make the transfer explicit for GPU runs.
        points = self._shell_points(real, imag, size, angle, compact)
        base = Image.new("RGB", size, (2, 4, 12)).convert("RGBA")
        glow = Image.new("RGBA", size, (0, 0, 0, 0))
        sharp = Image.new("RGBA", size, (0, 0, 0, 0))
        gd, sd = ImageDraw.Draw(glow, "RGBA"), ImageDraw.Draw(sharp, "RGBA")
        radius = self._paint_shell(gd, sd, points, size, compact)
        halo = glow.filter(ImageFilter.GaussianBlur(max(2, radius * 3)))
        result = Image.alpha_composite(Image.alpha_composite(base, halo), sharp).convert("RGB")
        return self._draw_tracers(result, trails, size, angle) if not compact else result

    # ---- guardian rendering ---------------------------------------------
    @staticmethod
    def _flux_shift(axis, minor):
        """Axis displacement carried by a flux surface of the given radius."""
        weight = 1.0 - 0.35 * min(1.0, minor / 0.42)
        return axis[0] * weight, axis[1] * weight

    def _draw_field_lines(self, glow_draw, sharp_draw, size, angle, count, pitch,
                          axis=(0.0, 0.0), action=(0.0, 0.0, 0.0), samples=150):
        """Helical confinement geometry, shifted and tightened by the coils.

        The curves are an explanatory picture of the field the commands are
        shaping. They are not field lines traced through a solved equilibrium.
        """
        if int(count) <= 0:
            return
        turns = 1.6
        radii = (0.34, 0.56, 0.78)
        effort = float(np.mean(np.abs(np.asarray(action, dtype=np.float32))))
        segments = []
        for line in range(max(1, int(count))):
            minor = radii[line % len(radii)]
            offset = line / max(1, count)
            shift = self._flux_shift(axis, minor)
            t = np.linspace(0.0, turns, samples)
            theta = 2 * np.pi * (offset + pitch * t)
            px, py, zz = self._project_points(
                t, shift[0] + minor * np.cos(theta), shift[1] + minor * np.sin(theta),
                size, angle)
            for k in range(1, samples):
                segments.append((px[k-1], py[k-1], px[k], py[k],
                                 0.5 * (zz[k-1] + zz[k]), line))
        segments.sort(key=lambda s: s[4])
        hues = ((88, 235, 255), (112, 155, 255), (179, 115, 255))
        warm = np.array((255, 168, 92), dtype=np.float32)
        for x0, y0, x1, y1, z, line in segments:
            front = float(np.clip(0.28 + 0.72 * (z + 0.65) / 1.3, 0.22, 1.0))
            hue = np.array(hues[line % len(hues)], dtype=np.float32)
            colour = tuple(int(q) for q in (hue * (1 - 0.55 * effort) + warm * (0.55 * effort)))
            glow_draw.line((x0, y0, x1, y1), fill=(*colour, int(16 * front)), width=5)
            sharp_draw.line((x0, y0, x1, y1), fill=(*colour, int(30 + 116 * front)),
                            width=1 if front < 0.72 else 2)

    def _coil_segments(self, size, angle, action, samples=110, radius=1.15):
        """Poloidal field coils: rings encircling the machine, outside the wall.

        Each ring sits at the poloidal angle its bank occupies in the
        cross-section, so the hardware in the two views is the same hardware.
        """
        segments = []
        turn = np.linspace(0.0, 1.0, samples)
        for (_name, colour, angles), command in zip(self.COIL_BANKS, action):
            level = .25 + .75 * abs(float(np.clip(command, -1, 1)))
            for degrees in angles:
                theta = math.radians(degrees)
                px, py, zz = self._project_points(
                    turn,
                    np.full(samples, radius * math.cos(theta)),
                    np.full(samples, radius * math.sin(theta)),
                    size, angle)
                for k in range(1, samples):
                    segments.append((px[k-1], py[k-1], px[k], py[k],
                                     0.5 * (zz[k-1] + zz[k]), colour, level))
        return segments

    @staticmethod
    def _draw_coils(glow_draw, sharp_draw, segments, half):
        """Draw the half of every coil ring that belongs on this side of the torus."""
        for x0, y0, x1, y1, z, colour, level in segments:
            if (z < 0) != (half == "back"):
                continue
            front = float(np.clip(0.30 + 0.70 * (z + 0.65) / 1.3, 0.22, 1.0))
            # Hardware, not the subject: the rings stay quiet until a bank is
            # actually commanded, then brighten and thicken with the command.
            glow_draw.line((x0, y0, x1, y1), fill=(*colour, int((6 + 26 * level) * front)),
                           width=int(4 + 6 * level))
            sharp_draw.line((x0, y0, x1, y1),
                            fill=(*colour, int((22 + 92 * level) * front)),
                            width=max(1, int(1 + 2 * level)))

    def _draw_particles(self, glow_draw, sharp_draw, particles, axis, size, angle,
                        compact=False, scale=1.0):
        """Confined markers with short trails, coloured by their wall clearance."""
        trail = particles.trail
        steps = trail.shape[1]
        theta = 2 * np.pi * trail[:, :, 1]
        radial = trail[:, :, 2]
        px, py, zz = self._project_points(
            trail[:, :, 0], axis[0] + radial * np.cos(theta), axis[1] + radial * np.sin(theta),
            size, angle)
        clearance = particles.clearance(axis)
        cols = self._ramp(clearance)
        w, h = size
        head_order = np.argsort(zz[:, -1])
        width = 1 if compact else 2
        for index in head_order:
            colour = tuple(int(q) for q in cols[index])
            for k in range(1, steps):
                x0, y0, x1, y1 = px[index, k-1], py[index, k-1], px[index, k], py[index, k]
                if abs(x1 - x0) > w * 0.22 or abs(y1 - y0) > h * 0.22:
                    continue
                age = k / max(1, steps - 1)
                front = float(np.clip(0.36 + 0.64 * (zz[index, k] + 0.65) / 1.3, 0.24, 1.0))
                sharp_draw.line((x0, y0, x1, y1),
                                fill=(*colour, int((10 + 180 * age * age) * front)),
                                width=width if age > 0.6 else max(1, width - 1))
            hx, hy = px[index, -1], py[index, -1]
            front = float(np.clip(0.42 + 0.58 * (zz[index, -1] + 0.65) / 1.3, 0.30, 1.0))
            radius = (1.9 + 2.3 * front) * (0.62 if compact else 1.0) * scale
            glow_draw.ellipse((hx - radius * 3, hy - radius * 3, hx + radius * 3, hy + radius * 3),
                              fill=(*colour, int(105 * front)))
            sharp_draw.ellipse((hx - radius, hy - radius, hx + radius, hy + radius),
                               fill=(*colour, int(245 * front)))

    def _draw_sparks(self, glow_draw, sharp_draw, particles, size, angle, compact=False,
                     spark_life=SPARK_LIFE):
        """Wall contacts: a marker that escaped confinement hitting the vessel."""
        sparks = particles.spark_list(spark_life)
        if not len(sparks):
            return
        theta = 2 * np.pi * sparks[:, 1]
        px, py, zz = self._project_points(sparks[:, 0], np.cos(theta), np.sin(theta),
                                          size, angle)
        for index in np.argsort(zz):
            age = float(sparks[index, 2])
            energy = float(sparks[index, 3])
            front = float(np.clip(0.35 + 0.65 * (zz[index] + 0.65) / 1.3, 0.22, 1.0))
            fade = (1.0 - age) ** 1.5
            if fade <= 0.02:
                continue
            radius = (2.6 + 8.5 * energy * fade) * (0.38 if compact else 1.0)
            hot = int(255 * min(1.0, fade * 1.4))
            colour = (255, int(120 + 135 * fade), int(40 + 150 * fade * fade))
            bloom = radius * (2.2 if compact else 3.2)
            glow_draw.ellipse((px[index] - bloom, py[index] - bloom,
                               px[index] + bloom, py[index] + bloom),
                              fill=(*colour, int((95 if compact else 165) * fade * front)))
            sharp_draw.ellipse((px[index] - radius, py[index] - radius,
                                px[index] + radius, py[index] + radius),
                               fill=(255, hot, int(190 * fade), int(245 * front)))
            if not compact:
                for spoke in range(4):
                    a = 2 * np.pi * (spoke / 4 + 0.12 * age)
                    length = radius * (1.8 + 1.4 * fade)
                    sharp_draw.line((px[index], py[index],
                                     px[index] + length * math.cos(a),
                                     py[index] + length * math.sin(a)),
                                    fill=(*colour, int(150 * fade * front)), width=1)

    def guardian_image(self, real, imag, particles, axis, action, size=(1280, 720),
                       angle=0.0, compact=False, field_lines=14, pitch=0.7,
                       shell_alpha=None, spark_life=SPARK_LIFE, marker_scale=1.0,
                       coils=True):
        """The magnetic bottle in 3-D with the markers the controller is holding."""
        points = self._shell_points(real, imag, size, angle, compact)
        px, py, zz, cols = points
        order = np.argsort(zz)
        back = order[zz[order] < 0]
        front = order[zz[order] >= 0]

        base = Image.new("RGB", size, (2, 4, 12)).convert("RGBA")
        glow = Image.new("RGBA", size, (0, 0, 0, 0))
        sharp = Image.new("RGBA", size, (0, 0, 0, 0))
        gd, sd = ImageDraw.Draw(glow, "RGBA"), ImageDraw.Draw(sharp, "RGBA")

        # The far half of the shell first, so the interior really reads as
        # interior; the near half returns at low opacity as a glass wall.
        # The coil rings encircle the machine, so the far half belongs behind
        # the vessel and the near half in front of everything inside it.
        coil_segments = self._coil_segments(size, angle, action) if coils else []
        self._draw_coils(gd, sd, coil_segments, "back")
        back_alpha, front_alpha = shell_alpha or (0.92, 0.24)
        radius = self._paint_shell(gd, sd, points, size, compact, alpha=back_alpha, order=back)
        self._draw_field_lines(gd, sd, size, angle, field_lines, pitch, axis, action,
                               samples=70 if compact else 150)
        self._draw_particles(gd, sd, particles, axis, size, angle, compact, marker_scale)
        self._draw_sparks(gd, sd, particles, size, angle, compact, spark_life)
        self._paint_shell(gd, sd, points, size, compact, alpha=front_alpha, order=front)
        self._draw_coils(gd, sd, coil_segments, "front")

        halo = glow.filter(ImageFilter.GaussianBlur(max(2, radius * 3)))
        return Image.alpha_composite(Image.alpha_composite(base, halo), sharp).convert("RGB")

    # ---- guardian diagnostic overlays -----------------------------------
    @staticmethod
    def _ellipse_points(cx, cy, rx, ry, wobble, count=96):
        theta = np.linspace(0, math.tau, count)
        r = 1 + wobble * np.sin(5 * theta + 1.7) + wobble * .45 * np.sin(9 * theta - .4)
        return [(cx + rx * r0 * math.cos(t), cy + ry * r0 * math.sin(t)) for t, r0 in zip(theta, r)]

    # Where each aggregate coil bank sits around the vessel, in degrees measured
    # from the outboard midplane.  Each bank is drawn as a symmetric set of coil
    # cross-sections - what you would actually cut through in this plane - so no
    # bank looks like it is missing a member.
    COIL_BANKS = (
        ("radial coils", (80, 225, 255), (0.0, 180.0)),
        ("vertical coils", (255, 166, 89), (90.0, 270.0)),
        ("shape coils", (198, 126, 255), (45.0, 135.0, 225.0, 315.0)),
    )

    @staticmethod
    def _legend_entry(draw, x, y, colour, text, marker="dot"):
        """Draw one swatch plus caption and return the x to continue from."""
        if marker == "line":
            draw.line((x, y + 5, x + 13, y + 5), fill=(*colour, 230), width=2)
        else:
            draw.ellipse((x + 2, y + 2, x + 10, y + 10), fill=(*colour, 235))
        draw.text((x + 18, y), text, font=font(9), fill=(168, 198, 224, 240))
        return x + 24 + draw.textbbox((0, 0), text, font=font(9))[2]

    def draw_poloidal(self, state, reference, action, particles, axis, size=(520, 556)):
        """Vessel cross-section: coils, controlled column, island, wall losses.

        This is one slice through the torus. Every marker in the volume is drawn
        at its own poloidal angle and minor radius, so all toroidal positions
        collapse onto this single plane.
        """
        image = Image.new("RGBA", size, (4, 11, 26, 240))
        d = ImageDraw.Draw(image, "RGBA")
        d.rounded_rectangle((0, 0, size[0]-1, size[1]-1), radius=18,
                            outline=(65, 155, 213, 190), width=2)
        d.text((20, 15), "POLOIDAL CROSS-SECTION", font=font(14, True), fill=(231, 246, 255, 255))
        d.text((20, 35), "one slice through the torus, seen end-on",
               font=font(10), fill=(142, 191, 224, 255))

        cx, cy = size[0] / 2, 288.0
        vessel = 138.0
        d.ellipse((cx-vessel, cy-vessel, cx+vessel, cy+vessel), outline=(79, 133, 183, 210), width=3)
        d.ellipse((cx-vessel+11, cy-vessel+11, cx+vessel-11, cy+vessel-11),
                  outline=(24, 63, 105, 180), width=2)

        # The three aggregate coil banks are the policy's three outputs. Each is
        # drawn as its coil cross-sections; radius and brightness are |command|.
        coil_radius = vessel + 24
        for (name, colour, angles), command in zip(self.COIL_BANKS, action):
            level = .25 + .75 * abs(float(command))
            for index, degrees in enumerate(angles):
                angle = math.radians(degrees)
                x = cx + coil_radius * math.cos(angle)
                y = cy - coil_radius * math.sin(angle)
                r = 6 + 5 * level
                d.ellipse((x-r, y-r, x+r, y+r), fill=(*colour, int(40 + 120 * level)),
                          outline=(*colour, int(120 + 135 * level)), width=2)
                if index:
                    continue
                label = f"{name}  {float(command):+.2f}"
                width = d.textbbox((0, 0), label, font=font(9))[2]
                if degrees == 90.0:
                    anchor = [x - width / 2, y - r - 15]
                else:
                    anchor = [x + r + 7, y - 5 if degrees == 0.0 else y - 15]
                # Keep every caption inside the card rather than clipping the
                # command value off the right edge.
                anchor[0] = min(anchor[0], size[0] - 16 - width)
                d.text(tuple(anchor), label, font=font(9), fill=(*colour, 245))

        px = cx + float(np.clip(state[0], -.95, .95)) * vessel * AXIS_GAIN
        py = cy + float(np.clip(state[1], -.95, .95)) * vessel * AXIS_GAIN
        mode = float(np.clip(state[4], 0, 1))
        pressure = float(np.clip(state[5], .45, 1.35))
        rx = vessel * 0.44 * (1 + .10 * (pressure - .9))
        ry = vessel * 0.33 * (1 - .08 * (pressure - .9))
        # The confined column is a soft backdrop: the markers in front of it
        # are the thing the controller is actually holding.
        column = Image.new("RGBA", size, (0, 0, 0, 0))
        cd = ImageDraw.Draw(column, "RGBA")
        cd.polygon(self._ellipse_points(px, py, rx, ry, .024 + .13 * mode), fill=(60, 190, 255, 58))
        column = column.filter(ImageFilter.GaussianBlur(7))
        image.alpha_composite(column)
        d.polygon(self._ellipse_points(px, py, rx, ry, .024 + .13 * mode),
                  outline=(185, 243, 255, 190))

        # A smooth amber island is part of the simulated state: its size is the
        # tearing-risk proxy.
        island_angle = particles.phase * .19
        ix = px + (rx * .60) * math.cos(island_angle)
        iy = py + (ry * .42) * math.sin(island_angle)
        island = vessel * (0.04 + 0.18 * mode)
        d.ellipse((ix-island, iy-island*.62, ix+island, iy+island*.62),
                  fill=(255, 139, 76, int(45 + 120 * mode)), outline=(255, 221, 153, 200), width=2)

        # The ghost is an uncontrolled reference trajectory, not a second plasma.
        # Once it has diverged it is held against the wall rather than drawn
        # outside the vessel: it disrupted there, it did not leave the machine.
        bw = vessel * (0.05 + 0.17 * min(1, float(reference[4])))
        brx, bry = vessel * 0.27 + bw, vessel * 0.20 + bw * .7
        ox = float(np.clip(reference[0], -.95, .95)) * AXIS_GAIN
        oy = float(np.clip(reference[1], -.95, .95)) * AXIS_GAIN
        offset = math.hypot(ox, oy)
        limit = max(0.0, 0.98 - max(brx, bry) / vessel)
        if offset > limit and offset > 1e-6:
            ox, oy = ox * limit / offset, oy * limit / offset
        d.line(self._ellipse_points(cx + ox * vessel, cy + oy * vessel, brx, bry,
                                    .018 + float(reference[4]) * .025),
               fill=(255, 80, 91, 130), width=3, joint="curve")

        # Markers, at their true poloidal position relative to the magnetic axis.
        angle = 2 * np.pi * particles.theta
        mx = cx + (axis[0] + particles.r * np.cos(angle)) * vessel
        my = cy + (axis[1] + particles.r * np.sin(angle)) * vessel
        cols = self._ramp(particles.clearance(axis))
        for index in range(len(mx)):
            colour = tuple(int(q) for q in cols[index])
            d.ellipse((mx[index]-1.7, my[index]-1.7, mx[index]+1.7, my[index]+1.7),
                      fill=(*colour, 225))
        # Recent wall contacts sit on the vessel boundary itself.
        sparks = particles.spark_list(SPARK_LIFE)
        for spark in sparks:
            fade = (1.0 - float(spark[2])) ** 1.5
            if fade <= 0.03:
                continue
            wall_angle = 2 * np.pi * float(spark[1])
            sx = cx + vessel * math.cos(wall_angle)
            sy = cy + vessel * math.sin(wall_angle)
            r = 2.0 + 5.0 * float(spark[3]) * fade
            d.ellipse((sx-r, sy-r, sx+r, sy+r),
                      fill=(255, int(150 + 100 * fade), 90, int(235 * fade)))

        d.ellipse((px-5, py-5, px+5, py+5), fill=(250, 255, 255, 250))

        # A legend, because none of the above is guessable from colour alone.
        top = size[1] - 54
        d.line((20, top - 10, size[0] - 20, top - 10), fill=(40, 78, 120, 190), width=1)
        self._legend_entry(d, 20, top, (118, 224, 255), "marker deep inside")
        self._legend_entry(d, 190, top, (255, 96, 64), "marker at the wall")
        self._legend_entry(d, 352, top, (255, 190, 110), "wall contact")
        self._legend_entry(d, 20, top + 19, (255, 139, 76), "tearing island")
        self._legend_entry(d, 190, top + 19, (255, 80, 91), "uncontrolled reference", "line")
        self._legend_entry(d, 352, top + 19, (250, 255, 255), "magnetic axis")
        return image.convert("RGB")

    def draw_shots(self, history, current, shots, size=(520, 300)):
        """Scoreboard: wall losses per finished shot, against the no-control run.

        ``history`` holds the finished shots, ``current`` the one in progress.
        Bars are counted losses, not an illustration of them.
        """
        image = Image.new("RGBA", size, (4, 11, 26, 240))
        d = ImageDraw.Draw(image, "RGBA")
        d.rounded_rectangle((0, 0, size[0]-1, size[1]-1), radius=18,
                            outline=(65, 155, 213, 190), width=2)
        d.text((20, 15), "TRAINING PROGRESS", font=font(14, True), fill=(231, 246, 255, 255))
        d.text((20, 35), "markers lost to the wall in each virtual shot"
               " \u00b7 bar height scales as \u221alosses",
               font=font(10), fill=(142, 191, 224, 255))

        rows = list(history) + ([current] if current else [])
        baseline = max((row.get("baseline", 0) for row in rows), default=0)
        # A square-root height keeps the first, catastrophic shot from flattening
        # every later bar into an unreadable sliver. The printed value on each
        # bar is the raw count.
        scale = math.sqrt
        peak = scale(max([row["lost"] for row in rows] + [baseline, 1]))
        left, right = 44.0, size[0] - 26.0
        floor, ceiling = size[1] - 46.0, 78.0
        span = (right - left) / max(1, shots)

        # The no-control reference is the line the policy has to beat.
        if baseline:
            y = floor - (floor - ceiling) * scale(baseline) / peak
            for x in range(int(left), int(right), 9):
                d.line((x, y, x + 5, y), fill=(255, 96, 106, 190), width=2)
            caption = f"no control  {baseline:,}"
            width = d.textbbox((0, 0), caption, font=font(9))[2]
            d.text((right - width, y - 15), caption, font=font(9), fill=(255, 140, 148, 235))

        best = min((row["lost"] for row in history), default=None)
        for index, row in enumerate(rows):
            lost = int(row["lost"])
            height = (floor - ceiling) * scale(lost) / peak
            x0 = left + index * span + span * 0.22
            x1 = left + (index + 1) * span - span * 0.22
            running = row is current and history is not rows
            fraction = scale(lost) / max(1e-6, peak)
            colour = (255, 104, 76) if fraction > .55 else (255, 186, 92) if fraction > .25 else (108, 232, 178)
            d.rectangle((x0, floor - height, x1, floor),
                        fill=(*colour, 110 if running else 215),
                        outline=(*colour, 245), width=2 if running else 1)
            d.text((x0, floor + 6), f"{row['shot']}", font=font(10, True),
                   fill=(186, 212, 234, 235))
            label = f"{lost:,}"
            width = d.textbbox((0, 0), label, font=font(10, True))[2]
            d.text(((x0 + x1) / 2 - width / 2, floor - height - 15), label,
                   font=font(10, True), fill=(*colour, 250))
            if best is not None and not running and lost == best:
                mark = d.textbbox((0, 0), "best", font=font(9, True))[2]
                d.text(((x0 + x1) / 2 - mark / 2, floor - height - 29), "best",
                       font=font(9, True), fill=(150, 240, 205, 235))

        d.line((left, floor, right, floor), fill=(60, 104, 150, 210), width=1)
        d.text((20, size[1] - 26), "shot", font=font(9), fill=(142, 191, 224, 220))
        note = ("policy frozen during each shot, trained between them"
                if len(rows) > 1 else "first shot: the policy has not trained yet")
        d.text((70, size[1] - 26), note, font=font(9), fill=(142, 191, 224, 220))
        return image.convert("RGB")

    @staticmethod
    def _node_positions(x, top, bottom, n):
        return [(x, top + (bottom-top)*(i+.5)/n) for i in range(n)]

    def draw_network(self, weights, action, loss, training, progress):
        image = Image.new("RGBA", (530, 520), (4, 12, 27, 235))
        d = ImageDraw.Draw(image, "RGBA")
        d.rounded_rectangle((0,0,529,519),radius=18,outline=(65,155,213,190),width=2)
        d.text((22,18), "NEURAL FEEDBACK POLICY", font=font(17, True), fill=(231,246,255,255))
        subtitle = "training through virtual plasma shots" if training else "analytical fallback — not learning"
        d.text((22,45), subtitle, font=font(11), fill=(142,191,224,255))
        xs=(54,190,332,472); layers=(self._node_positions(xs[0],94,402,6), self._node_positions(xs[1],118,378,7), self._node_positions(xs[2],132,364,6), self._node_positions(xs[3],178,318,3))
        compact=(weights[0][:, :7], weights[1][:7, :6], weights[2][:6, :])
        for stage, matrix in enumerate(compact):
            scale=max(.02,float(np.percentile(np.abs(matrix),90)))
            for a, start in enumerate(layers[stage]):
                for b, end in enumerate(layers[stage+1]):
                    value=float(matrix[a,b]); strength=min(1,abs(value)/scale)
                    colour=(78,226,255,int(18+150*strength)) if value>=0 else (250,92,177,int(18+150*strength))
                    d.line((*start,*end),fill=colour,width=1+int(2*strength))
        for layer, nodes in enumerate(layers):
            for idx, (x,y) in enumerate(nodes):
                if layer==3:
                    magnitude=abs(float(action[idx])); colour=((83,232,255) if idx==0 else (255,170,88) if idx==1 else (206,126,255))
                    r=9+5*magnitude
                else:
                    value=.55+.45*math.sin(progress*12 + idx*1.8 + layer)
                    colour=(95,207,255); r=6+2*value
                d.ellipse((x-r,y-r,x+r,y+r),fill=(*colour,235),outline=(230,250,255,245),width=1)
            if layer==0:
                for idx,(x,y) in enumerate(nodes): d.text((x-43,y-5),INPUT_NAMES[idx],font=font(9),fill=(156,196,224,240))
            if layer==3:
                for idx,(x,y) in enumerate(nodes): d.text((x+16,y-5),OUTPUT_NAMES[idx],font=font(9),fill=(196,222,245,245))
        d.text((22,470), f"policy loss  {loss:.4f}",font=font(13,True),fill=(127,239,255,255))
        d.text((305,470), "cyan + / pink − weight",font=font(10),fill=(170,197,224,220))
        return image.convert("RGB")

    # ---- hero frames ----------------------------------------------------
    def hero(self, real, imag, frame, done, total, trails=None):
        canvas = self.torus_image(real, imag, size=(1280, 720), angle=frame * 0.010, trails=trails)
        shape = to_numpy(real).shape
        canvas = add_title(
            canvas,
            "Star in a Bottle",
            f"field-driven plasma tracers · {shape[1]}×{shape[0]} periodic lattice · solver step {done:,}/{total:,} · {self.ctx.backend_name}",
            badge="LIVE FIELD + FLOW",
        )
        add_progress(canvas, done / total, "COHERENT WAVES", "TURBULENT PLASMA")
        return canvas

    def guardian_hero(self, real, imag, particles, axis, action, frame, updates,
                      total_updates, pitch, field_lines, training):
        canvas = self.guardian_image(real, imag, particles, axis, action,
                                     size=(1280, 720), angle=frame * 0.010,
                                     field_lines=field_lines, pitch=pitch)
        badge = "NEURAL CONTROL" if training else "ANALYTICAL FALLBACK"
        label = "policy training" if training else "analytical fallback — not learning"
        canvas = add_title(
            canvas,
            "AI Plasma Guardian",
            f"{particles.count:,} confined markers · {label} · update {updates:,}/{total_updates:,} · {self.ctx.backend_name}",
            badge=badge,
        )
        add_progress(canvas, updates / max(1, total_updates), "UNTRAINED POLICY", "HELD CONFINEMENT")
        return canvas

    # ---- interactive state ----------------------------------------------
    def _texture_payload(self, real, imag):
        ny, nx = real.shape
        stride = max(1, int(math.ceil(ny / 96)))
        real, imag = to_numpy(real[::stride, ::stride]), to_numpy(imag[::stride, ::stride])
        amp = np.sqrt(real * real + imag * imag)
        phase = (np.arctan2(imag, real) + np.pi) / (2 * np.pi)
        texture = np.clip(0.62 * amp / 1.25 + 0.38 * phase, 0, 1)
        return texture

    @staticmethod
    def field_pitch(magnetic_field):
        """Illustrative helical pitch used by the magnetic-geometry view only.

        This is a simple monotone relationship, not a solved equilibrium or a
        safety-factor profile.
        """
        return float(np.clip(0.48 + 0.045 * float(magnetic_field), 0.5, 0.9))

    def _write_view(self, manifest, frame):
        if frame is None:
            path = self.ctx.run_dir / "fusion_view.json"
        else:
            path = self.ctx.run_dir / "modes" / "fusion3d" / f"frame_{int(frame):04d}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
        if frame == 0:
            self.ctx.write_meta({"fusion_view": {"folder": "modes/fusion3d",
                                                 "fallback": "fusion_view.json"},
                                 "default_view_mode": "fusion3d"})
        return path

    def _passive_manifest(self, real, imag, trails, magnetic_field, heating, density):
        manifest = self._passive_manifest_body(self._texture_payload(real, imag), trails, magnetic_field,
                                               heating, density)
        return manifest

    def write_interactive_view(self, real, imag, trails, magnetic_field, heating,
                               density, frame=None):
        """Persist a compact state for the browser's live rotatable canvas."""
        manifest = self._passive_manifest_body(self._texture_payload(real, imag), to_numpy(trails),
                                               magnetic_field, heating, density)
        return self._write_view(manifest, frame)

    def _passive_manifest_body(self, texture, trails, magnetic_field, heating, density):
        manifest = {
            "version": 2,
            "kind": "fusion-torus",
            "mode": "passive",
            "shape": [int(texture.shape[0]), int(texture.shape[1])],
            "texture": np.rint(texture * 255).astype(np.uint8).ravel().tolist(),
            "trails": (np.round(np.asarray(trails, dtype=np.float32), 5).tolist() if trails is not None else None),
            "magnetic_field": float(magnetic_field),
            "heating": float(heating),
            "density": float(density),
            "field_lines": 14,
            "field_pitch": self.field_pitch(magnetic_field),
            "note": "Magnetic lines are illustrative helical confinement geometry, not a solved equilibrium.",
        }
        return manifest

    def write_guardian_view(self, real, imag, particles, axis, state, action,
                            magnetic_field, heating, density, stats, frame=None):
        """Persist the controlled 3-D confinement state for the browser."""
        texture = self._texture_payload(real, imag)
        trail = np.asarray(particles.trail, dtype=np.float32)
        if trail.shape[1] > 5:
            trail = trail[:, -5:, :]
        manifest = {
            "version": 2,
            "kind": "fusion-torus",
            "mode": "guardian",
            "shape": [int(texture.shape[0]), int(texture.shape[1])],
            "texture": np.rint(texture * 255).astype(np.uint8).ravel().tolist(),
            "particles": np.round(trail[:, -1, :], 4).tolist(),
            "particle_trails": np.round(trail, 4).tolist(),
            "clearance": np.round(particles.clearance(axis).astype(np.float32), 3).tolist(),
            "sparks": np.round(particles.spark_list(SPARK_LIFE), 4).tolist(),
            "axis": [round(float(axis[0]), 4), round(float(axis[1]), 4)],
            "commands": [round(float(v), 4) for v in np.asarray(action).ravel()[:3]],
            "island": round(float(state[4]), 4),
            "pressure": round(float(state[5]), 4),
            "risk": round(risk_of(state), 4),
            "lost_total": int(stats["lost_total"]),
            "loss_rate": round(float(stats["loss_rate"]), 5),
            "magnetic_field": float(magnetic_field),
            "heating": float(heating),
            "density": float(density),
            "field_lines": 14,
            "field_pitch": self.field_pitch(magnetic_field),
            "note": "Coil commands come from the trained policy; the helical lines are illustrative confinement geometry, not a solved equilibrium.",
        }
        return self._write_view(manifest, frame)

    # ---- solver-side helpers timed by the harness ------------------------
    def train(self, trainer, updates, drive, confine=1.0, experience=None):
        return trainer.train(updates,
                             int(self.settings.get("batch", 128)),
                             int(self.settings.get("horizon", 28)),
                             drive, confine, experience)

    def advance_particles(self, particles, turbulence, axis, island, pressure,
                          shape_command, magnetic_field, heating, drive, substeps=4):
        return particles.advance(turbulence, axis, island, pressure, shape_command,
                                 magnetic_field, heating, drive, substeps=substeps,
                                 spark_life=SPARK_LIFE)

    # ---- runs ------------------------------------------------------------
    def run(self):
        if self.mode() == "guardian":
            return self.run_guardian()
        return self.run_passive()

    def run_passive(self):
        ny, nx = self.grid_shape(self.settings["n"])
        total = self.budget()
        b = float(self.ctx.params.get("magnetic_field", 5.0))
        heating = float(self.ctx.params.get("heating", 25.0))
        density = float(self.ctx.params.get("density", 1.0))
        real, imag = self.initialise(ny, nx)
        trails = self.initialise_tracers(
            int(self.settings.get("tracers", 180)),
            int(self.settings.get("trail", 12)),
        )
        done = 0
        # Physics and the frame's reductions on the device; drawing the torus,
        # the tracer ribbons and the 3-D view file on the allocated CPU cores.
        workers = max(1, min(self.ctx.cpu_workers - 1, 16, max(1, self.ctx.frames // 6)))
        with FramePipeline(self.ctx, workers=workers) as frames:
            for i in range(self.ctx.frames):
                target = int(round(total * (i + 1) / self.ctx.frames))
                solver_steps = max(1, target - done)
                real, imag = self.step(real, imag, b, heating, density, solver_steps)
                done = target
                with self.ctx.stage("visualization"):
                    trails, tracer_speed = self.advance_tracers(trails, real, imag, b, heating, solver_steps)
                    score = self.turbulence(real, imag)
                    angle = i * 0.010
                    payload = {
                        "points": self._shell_points(real, imag, (1280, 720), angle, False),
                        "trails": to_numpy(trails).astype(np.float32), "angle": angle,
                        "subtitle": (f"field-driven plasma tracers · {real.shape[1]}×{real.shape[0]} periodic lattice · "
                                     f"solver step {done:,}/{total:,} · {self.ctx.backend_name}"),
                        "progress": done / total,
                        "view": {"path": str(self.ctx.run_dir / "modes" / "fusion3d" / f"frame_{i:04d}.json"),
                                 "manifest": self._passive_manifest(real, imag, None, b, heating, density)},
                    }
                if i == 0:
                    self._write_view({**payload["view"]["manifest"], "trails": []}, 0)
                regime = "phase turbulence" if score > .075 else "coherent confinement"
                frames.submit(i, draw_passive_frame, payload, self.ctx.frame_path(i),
                              f"plasma lattice step {done:,} · {len(trails):,} tracers", {
                    "mode":"passive confinement",
                    "magnetic field":f"{b:.1f} T","heating power":f"{heating:.0f} MW","density":f"{density:.2f} n₀",
                    "regime":regime,"turbulence index":f"{score:.3f}","passive tracers":f"{len(trails):,}",
                    "mean drift":f"{tracer_speed*360:.3f}° / step","solver step":f"{done:,} / {total:,}"})
        trails = to_numpy(trails)

        # Retain one final-state file so older viewers can still open new runs.
        self.write_interactive_view(real, imag, trails, b, heating, density)

        ens = max(1, int(self.ctx.params.get("_parallel_count", self.settings.get("ensemble", 16))))
        side = max(1, int(math.ceil(math.sqrt(ens))))
        sweep_n = int(self.settings.get("sweep_n", max(36, ny // 2)))
        my, mx = self.grid_shape(sweep_n)
        sweep_steps = int(self.settings.get("sweep_steps", max(160, total // 3)))
        tiles, labels = [], []
        b_values = np.linspace(2.5, 7.5, side)
        h_values = np.linspace(10.0, 45.0, side)
        for j in range(ens):
            row,col=divmod(j,side); hj=h_values[row]; bj=b_values[col]
            rr, ii = self.initialise(my, mx, seed=40 + j)
            rr, ii = self.step(rr, ii, float(bj), float(hj), density, sweep_steps)
            tiles.append(self.torus_image(rr, ii, size=(260, 146), angle=0.35, compact=True))
            labels.append(f"B {bj:.1f}T · {hj:.0f}MW")
        reveal = mosaic(
            tiles,
            side,
            title="Actually… we tested a whole reactor operating map",
            subtitle="Magnetic field increases left to right; heating increases top to bottom. Every torus is a separate run.",
            labels=labels,
            label_fill=(180, 238, 255),
        )
        rp = self.ctx.run_dir / "reveal.jpg"
        self.ctx.save_frame(reveal, rp)
        self.ctx.finish(rp)

    def shot_plan(self, frames):
        """Frame index boundaries of each virtual shot.

        A shot needs enough frames to show an instability develop, so a short
        run gets fewer shots rather than one-frame ones.
        """
        shots = max(1, int(self.settings.get("shots", 6)))
        shots = max(1, min(shots, frames // 3 or 1))
        edges = [int(round(frames * k / shots)) for k in range(shots + 1)]
        return [(edges[k], edges[k + 1]) for k in range(shots) if edges[k + 1] > edges[k]]

    def run_guardian(self):
        ny, nx = self.grid_shape(self.settings["n"])
        total = self.budget()
        frames = self.ctx.frames
        b = float(self.ctx.params.get("magnetic_field", 5.0))
        heating = float(self.ctx.params.get("heating", 25.0))
        density = float(self.ctx.params.get("density", 1.0))
        drive = float(self.ctx.params.get("instability", 1.0))
        confine = max(0.55, math.sqrt(b / 5.0))
        pitch = self.field_pitch(b)
        field_lines = 9

        trainer = make_trainer(self.ctx, float(self.settings.get("learning_rate", 0.004)))
        total_updates = max(1, int(self.settings.get("train_updates", 120)))
        shot_steps = max(4, int(self.settings.get("display_steps", 96)))
        count = int(self.settings.get("particles", 260))
        trail = max(3, min(5, int(self.settings.get("trail", 12)) // 3))
        plan = self.shot_plan(frames)
        shots = len(plan)
        # Each shot gets an identical slice of the field budget, so the total
        # numerical work is still the profile's and every shot faces the same
        # turbulence history.
        shot_budget = max(1, total // shots)
        idle = np.zeros(3, dtype=np.float32)
        disturbance = np.zeros(2, dtype=np.float32)

        for name in ("network", "poloidal", "shots"):
            (self.ctx.run_dir / "overlays" / name).mkdir(parents=True, exist_ok=True)
        # Every shot's controller is saved as it flew, so a finished run can be
        # tested again in conditions of the visitor's choosing (see replay).
        checkpoints = self.ctx.run_dir / "checkpoints"
        checkpoints.mkdir(exist_ok=True)
        self.ctx.write_meta({"overlays": ["network", "poloidal", "shots"],
                             "simulation_mode": "guardian",
                             "shots": shots,
                             "shot_steps": shot_steps,
                             "training": "between shots, on the states each shot visited",
                             "trained_world": {"magnetic_field": b, "heating": heating,
                                               "instability": drive, "density": density},
                             "lab": {"checkpoints": "checkpoints", "generations": shots,
                                     "kind": "controller", "compare": True}})

        history = []
        loss = 0.0
        trained = 0
        for shot, (first, last) in enumerate(plan):
            save_controller(checkpoints / f"gen_{shot + 1:04d}.npz", trainer)
            # Every shot is the same experiment: same field seed, same marker
            # seed, same start state, same disturbance sequence. The only thing
            # that differs between shots is the policy, so the scoreboard is a
            # controlled comparison rather than a drifting one.
            real, imag = self.initialise(ny, nx)
            particles = ConfinedParticles(count, trail, seed=404)
            reference = ConfinedParticles(count, trail, seed=404)
            state = np.array(START_STATE, dtype=np.float32)
            reference_state = state.copy()
            action = trainer.act(state)
            visited = [state.copy()]
            shot_frames = last - first
            done = stepped = 0
            disrupted_at = None
            stats = reference_stats = None
            peak_risk = 0.0

            for index in range(shot_frames):
                target = int(round(shot_budget * (index + 1) / shot_frames))
                real, imag = self.step(real, imag, b, heating, density, max(1, target - done))
                done = target

                # The policy is frozen for the whole shot. Nothing the network
                # does mid-shot can perturb the physics being displayed.
                control_target = int(round(shot_steps * (index + 1) / shot_frames))
                for _ in range(max(1, control_target - stepped)):
                    action = trainer.act(state)
                    state = transition_numpy(state, action, disturbance, stepped, drive)
                    reference_state = transition_numpy(reference_state, idle, disturbance,
                                                       stepped, drive)
                    stepped += 1
                    visited.append(state.copy())
                peak_risk = max(peak_risk, risk_of(state))
                if disrupted_at is None and float(np.hypot(reference_state[0], reference_state[1])) >= 1.0:
                    disrupted_at = stepped

                turbulence = self.turbulence_map(real, imag)
                axis = np.clip(state[:2], -1.0, 1.0) * AXIS_GAIN
                reference_axis = np.clip(reference_state[:2], -1.0, 1.0) * AXIS_GAIN
                stats = self.advance_particles(particles, turbulence, axis, state[4], state[5],
                                               action[2], b, heating, drive)
                reference_stats = self.advance_particles(reference, turbulence, reference_axis,
                                                         reference_state[4], reference_state[5],
                                                         0.0, b, heating, drive)

                frame = first + index
                image = self.guardian_hero(real, imag, particles, axis, action, frame, trained,
                                           total_updates, pitch, field_lines, trainer.training)
                self.ctx.save_frame(image, self.ctx.frame_path(frame))
                weights = trainer.weights()
                self.ctx.save_frame(
                    self.draw_network(weights, action, loss, trainer.training, (frame + 1) / frames),
                    self.ctx.run_dir / "overlays" / "network" / f"frame_{frame:04d}.jpg")
                self.ctx.save_frame(
                    self.draw_poloidal(state, reference_state, action, particles, axis),
                    self.ctx.run_dir / "overlays" / "poloidal" / f"frame_{frame:04d}.jpg")
                progress = {"shot": shot + 1, "lost": int(stats["lost_total"]),
                            "baseline": int(reference_stats["lost_total"]), "count": count}
                self.ctx.save_frame(self.draw_shots(history, progress, shots),
                                    self.ctx.run_dir / "overlays" / "shots" / f"frame_{frame:04d}.jpg")
                self.write_guardian_view(real, imag, particles, axis, state, action,
                                         b, heating, density, stats, frame=frame)

                best = min([row["lost"] for row in history] + [int(stats["lost_total"])])
                self.ctx.write_status(frame, f"shot {shot + 1} of {shots} - policy frozen", {
                    "mode": "AI plasma guardian",
                    "phase": f"shot {shot + 1} of {shots} running, policy frozen",
                    "control model": "neural policy" if trainer.training else "analytical fallback",
                    "training so far": (f"{trained:,} updates after shot {shot}" if shot
                                        else "none yet"),
                    "policy loss": f"{loss:.4f}" if shot else "untrained",
                    "confined markers": f"{count:,}",
                    "wall losses this shot": f"{stats['lost_total']:,}",
                    "wall losses this frame": f"{stats['lost']:,}",
                    "previous shot": f"{history[-1]['lost']:,}" if history else "-",
                    "best shot so far": f"{best:,}",
                    "no-control reference": (f"{reference_stats['lost_total']:,}"
                                             + ("" if disrupted_at is None
                                                else f" - disrupted at step {disrupted_at}")),
                    "tearing-risk proxy": f"{risk_of(state):.2f}",
                    "coil command norm": f"{np.linalg.norm(action):.2f}",
                    "magnetic field": f"{b:.1f} T",
                    "instability drive": f"{drive:.2f}",
                    "solver step": f"{done:,} / {shot_budget:,} this shot",
                })

            history.append({
                "shot": shot + 1,
                "lost": int(stats["lost_total"]),
                "baseline": int(reference_stats["lost_total"]),
                "count": count,
                "peak_risk": round(peak_risk, 3),
                "disrupted_at": disrupted_at,
            })

            # Between shots, and only between shots: score what happened and
            # optimize the policy against it before the next one starts.
            if shot < shots - 1:
                update_target = int(round(total_updates * (shot + 1) / max(1, shots - 1)))
                loss = self.train(trainer, max(1, update_target - trained), drive, confine, visited)
                trained = min(total_updates, update_target)
                history[-1]["trained_to"] = trained
                history[-1]["loss_after"] = round(float(loss), 5)
            self.ctx.write_meta({"shot_history": history})

        self.write_guardian_view(real, imag, particles, axis, state, action,
                                 b, heating, density, stats)
        reveal_path = self.ctx.run_dir / "reveal.jpg"
        self.ctx.save_frame(self.guardian_reveal(trainer, b, heating, density, ny), reveal_path)
        self.ctx.finish(reveal_path)

    # ---- testing a saved controller ---------------------------------------
    TEST_FRAMES = 40
    TEST_GRID = 40

    @classmethod
    def replay(cls, run_dir: Path, meta: dict, gens, seed: int, env: dict | None = None):
        """Fly saved controllers through a fresh plasma, in conditions of choice.

        ``gens`` are shot numbers: the controller that flew each shot, frozen.
        ``env`` may change the magnetic field, heating and instability drive;
        nothing about the networks changes.  Every controller and an
        uncontrolled reference fly the same field from the same start, on the
        CPU, at a small test resolution (``TEST_GRID`` rows) so a visitor's
        request comes back in about a second.  Returns one texture per frame
        (shared) and, per controller, the confined markers frame by frame.
        Bulk arrays are packed as base64 bytes to keep a test to a few hundred
        kilobytes: textures as uint8 (frames x rows x cols), marker positions
        (toroidal turn, poloidal turn, minor radius, each in [0, 1]) as uint16
        scaled by 65535 (frames x markers x 3), and wall sparks the same way
        (toroidal, poloidal, age fraction, energy / 1.6), with ``spark_counts``
        per frame.  The browser rebuilds the short trails from earlier frames.
        """
        env = env or {}
        settings = meta.get("settings") or {}
        params = meta.get("params") or {}
        trained = meta.get("trained_world") or {
            "magnetic_field": float(params.get("magnetic_field", 5.0)), "heating": float(params.get("heating", 25.0)),
            "instability": float(params.get("instability", 1.0)), "density": float(params.get("density", 1.0))}
        world = {key: float(env.get(key, trained[key])) for key in ("magnetic_field", "heating", "instability", "density")}
        b, heating, drive, density = (world[k] for k in ("magnetic_field", "heating", "instability", "density"))

        demo = cls.__new__(cls)
        demo.ctx = type("TestContext", (), {"xp": np})()
        rng = np.random.default_rng(int(seed))
        ny, nx = demo.grid_shape(min(cls.TEST_GRID, int(settings.get("sweep_n", cls.TEST_GRID))))
        real, imag = demo.initialise(ny, nx, seed=140 + int(seed) % 997)
        real, imag = demo.step(real, imag, b, heating, density, 120)
        count = max(48, min(260, int(settings.get("particles", 220))))
        frames = cls.TEST_FRAMES
        control_per_frame = max(1, int(round(int(settings.get("display_steps", 48)) * 1.5 / frames)))
        start = np.array(START_STATE, dtype=np.float32)
        start[:4] += rng.uniform(-.06, .06, 4).astype(np.float32)
        particle_seed = 404 + int(seed) % 997
        disturbance = np.zeros(2, dtype=np.float32)

        pilots = [(f"shot {int(g)}", int(g), FrozenController(Path(run_dir) / "checkpoints" / f"gen_{int(g):04d}.npz"))
                  for g in gens]
        flights = [{"label": label, "shot": shot, "pilot": pilot, "state": start.copy(), "step": 0,
                    "markers": ConfinedParticles(count, 4, seed=particle_seed), "frames": [], "positions": [], "sparks": []}
                   for label, shot, pilot in pilots]
        flights.append({"label": "no control", "shot": None, "pilot": None, "state": start.copy(), "step": 0,
                        "markers": ConfinedParticles(count, 4, seed=particle_seed), "frames": [], "positions": [], "sparks": []})
        textures = []
        idle = np.zeros(3, dtype=np.float32)
        for frame in range(frames):
            real, imag = demo.step(real, imag, b, heating, density, 3)
            turbulence = demo.turbulence_map(real, imag)
            textures.append(np.rint(demo._texture_payload(real, imag) * 255).astype(np.uint8))
            for f in flights:
                action = idle
                for _ in range(control_per_frame):
                    action = idle if f["pilot"] is None else f["pilot"].act(f["state"])
                    f["state"] = transition_numpy(f["state"], action, disturbance, f["step"], drive)
                    f["step"] += 1
                state = f["state"]
                axis = np.clip(state[:2], -1.0, 1.0) * AXIS_GAIN
                stats = f["markers"].advance(turbulence, axis, state[4], state[5], action[2], b, heating, drive,
                                             substeps=3, spark_life=SPARK_LIFE)
                markers = f["markers"]
                f["positions"].append(np.stack([markers.u, markers.theta, markers.r], axis=-1))
                f["sparks"].append(markers.spark_list(SPARK_LIFE))
                f["frames"].append({
                    "axis": [round(float(axis[0]), 4), round(float(axis[1]), 4)],
                    "commands": [round(float(v), 4) for v in np.asarray(action).ravel()[:3]],
                    "risk": round(risk_of(state), 4),
                    "lost_total": int(stats["lost_total"]),
                })
        shape = [int(v) for v in textures[0].shape]
        pack = lambda a: base64.b64encode(np.ascontiguousarray(a).tobytes()).decode("ascii")
        runs = [{"label": f["label"], "shot": f["shot"], "controlled": f["pilot"] is not None,
                 "lost": int(f["markers"].lost_total), "frames": f["frames"],
                 "positions": pack(np.rint(np.clip(np.asarray(f["positions"]), 0, 1) * 65535).astype("<u2")),
                 "spark_counts": [len(k) for k in f["sparks"]],
                 "sparks": pack(np.rint(np.clip(np.concatenate(f["sparks"] + [np.zeros((0, 4), np.float32)])
                                                / np.array([1, 1, 1, 1.6], np.float32), 0, 1) * 65535).astype("<u2"))}
                for f in flights]
        return {"kind": "fusion-test", "shape": shape, "textures": pack(np.asarray(textures)),
                "count": count, "frames": frames,
                "world": world, "trained_world": trained, "seed": int(seed),
                "field_lines": 9, "field_pitch": cls.field_pitch(b),
                "runs": runs[:-1], "reference": runs[-1]}

    def guardian_reveal(self, trainer, b, heating, density, ny):
        """Re-test the trained policy against instability drives it never saw.

        Each tile integrates its own plasma field and then runs an independent
        closed-loop confinement evaluation, so the mosaic is real additional
        computation rather than a decorated copy of the last frame.
        """
        ens = max(1, int(self.ctx.params.get("_parallel_count", self.settings.get("ensemble", 16))))
        side = max(1, int(math.ceil(math.sqrt(ens))))
        sweep_n = int(self.settings.get("sweep_n", max(36, ny // 2)))
        my, mx = self.grid_shape(sweep_n)
        sweep_steps = int(self.settings.get("sweep_steps", 220))
        tile_count = max(48, int(self.settings.get("particles", 260)) // 4)
        evaluation = max(24, int(self.settings.get("display_steps", 96)) // 2)
        # Contacts persist through the whole evaluation so each tile shows the
        # wall load it accumulated, not just its final instant.
        tile_spark_life = float(4 * evaluation)
        drives = np.linspace(0.6, 1.5, ens) if ens > 1 else np.array([1.0])
        disturbance = np.zeros(2, dtype=np.float32)
        tiles, labels = [], []
        for j in range(ens):
            drive = float(drives[j])
            rr, ii = self.initialise(my, mx, seed=140 + j)
            rr, ii = self.step(rr, ii, b, heating, density, sweep_steps)
            turbulence = self.turbulence_map(rr, ii)
            markers = ConfinedParticles(tile_count, 3, seed=900 + j)
            state = np.array(START_STATE, dtype=np.float32)
            action = trainer.act(state)
            for k in range(evaluation):
                action = trainer.act(state)
                state = transition_numpy(state, action, disturbance, k, drive)
                axis = np.clip(state[:2], -1.0, 1.0) * AXIS_GAIN
                markers.advance(turbulence, axis, state[4], state[5], action[2],
                                b, heating, drive, substeps=2, spark_life=tile_spark_life)
            axis = np.clip(state[:2], -1.0, 1.0) * AXIS_GAIN
            tiles.append(self.guardian_image(rr, ii, markers, axis, action, size=(260, 146),
                                             angle=0.35, compact=True, field_lines=0,
                                             pitch=self.field_pitch(b),
                                             shell_alpha=(0.34, 0.10), coils=False,
                                             spark_life=tile_spark_life, marker_scale=1.7))
            labels.append(f"drive {drive:.2f} · {100 * markers.lost_total / tile_count:.0f} hits/100")
        # The browser owns reveal text, so publish what each tile varied.
        self.ctx.write_meta({"reveal_sweep": {"variable": "instability drive", "labels": labels}})
        return mosaic(
            tiles,
            side,
            title="Actually… we stress-tested the controller across the whole instability range",
            subtitle="Every tile is an independent plasma field plus its own closed-loop evaluation. "
                     "Wall hits per 100 markers are counted, not illustrated.",
            labels=labels,
            label_fill=(180, 238, 255),
        )


def _xp_of(array):
    """NumPy or CuPy, whichever module owns this array."""
    module = type(array).__module__
    if module.startswith("cupy"):
        import cupy
        return cupy
    return np


def draw_passive_frame(p):
    """Draw one passive-mode frame and write its 3-D view file (worker process)."""
    import json as _json
    demo = FusionPlasmaDemo.__new__(FusionPlasmaDemo)   # drawing helpers only; no solver state
    size = (1280, 720)
    base = Image.new("RGB", size, (2, 4, 12)).convert("RGBA")
    glow = Image.new("RGBA", size, (0, 0, 0, 0))
    sharp = Image.new("RGBA", size, (0, 0, 0, 0))
    gd, sd = ImageDraw.Draw(glow, "RGBA"), ImageDraw.Draw(sharp, "RGBA")
    radius = demo._paint_shell(gd, sd, p["points"], size, False)
    halo = glow.filter(ImageFilter.GaussianBlur(max(2, radius * 3)))
    image = Image.alpha_composite(Image.alpha_composite(base, halo), sharp).convert("RGB")
    image = demo._draw_tracers(image, p["trails"], size, p["angle"])
    image = add_title(image, "Star in a Bottle", p["subtitle"], badge="LIVE FIELD + FLOW")
    add_progress(image, p["progress"], "COHERENT WAVES", "TURBULENT PLASMA")
    view = p.get("view")
    if view:
        manifest = dict(view["manifest"])
        manifest["trails"] = np.round(np.asarray(p["trails"], dtype=np.float32), 5).tolist()
        target = Path(view["path"])
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp = target.with_suffix(".tmp")
        tmp.write_text(_json.dumps(manifest, separators=(",", ":")), encoding="utf-8")
        tmp.replace(target)
    return image

