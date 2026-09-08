from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..backend import to_numpy
from ..base import Demo
from ..render import add_progress, add_title, font, mosaic, save_frame


class WeatherEnsembleDemo(Demo):
    timing_methods={"initialise":"initialization","step":"simulation","hero":"render"}
    """Reduced global atmosphere with a genuine initial-condition ensemble."""

    id = "weather_ensemble"
    title = "Storm Factory"

    def grid_shape(self, n):
        ny = max(24, int(n))
        return ny, ny * 2

    def budget(self):
        if "total_steps" in self.settings:
            return max(1, int(self.settings["total_steps"]))
        return max(1, int(self.settings.get("steps_per_frame", 5)) * self.ctx.frames)

    @staticmethod
    def _wrapped_delta(a, b):
        return (a - b + np.pi) % (2 * np.pi) - np.pi

    def initialise(self, ny, nx, seed=23, perturbation=0.0, warming=1.5):
        rng = np.random.default_rng(seed)
        lat = np.linspace(-np.pi / 2, np.pi / 2, ny, dtype=np.float32)[:, None]
        lon = np.linspace(-np.pi, np.pi, nx, endpoint=False, dtype=np.float32)[None, :]
        zeta = (
            0.18 * np.sin(3 * lon + 1.4 * np.sin(2 * lat)) * np.cos(lat) ** 2
            + 0.10 * np.sin(6 * lon - 2.2 * lat) * np.cos(lat)
        )
        # Three synoptic vortices, including an Atlantic storm that the camera
        # follows. They are initial vorticity, not painted-on graphics.
        storm_lon = -48 + rng.normal(0, max(0.15, 0.12 * perturbation))
        storm_lat = 26 + rng.normal(0, max(0.08, 0.06 * perturbation))
        for lo, la, strength, radius in [
            (storm_lon, storm_lat, -1.65 - 0.10 * warming, 13),
            (38, 48, 1.05, 17),
            (142, -28, -0.90, 16),
        ]:
            dx = self._wrapped_delta(lon, math.radians(lo)) * np.cos(math.radians(la))
            dy = lat - math.radians(la)
            zeta += strength * np.exp(-(dx * dx + dy * dy) / (2 * math.radians(radius) ** 2))
        noise = rng.normal(0, 1, (ny, nx)).astype(np.float32)
        # Smooth the perturbation so it represents uncertain large-scale
        # observations rather than pixel noise.
        for _ in range(4):
            noise = 0.25 * (
                np.roll(noise, 1, 0) + np.roll(noise, -1, 0)
                + np.roll(noise, 1, 1) + np.roll(noise, -1, 1)
            )
        zeta += (0.006 + 0.0012 * perturbation) * noise
        humidity = (
            0.44 + 0.27 * np.cos(lat) ** 2
            + 0.11 * np.sin(2 * lon - lat)
            + 0.14 * np.exp(-((lat - math.radians(22)) / 0.32) ** 2)
        )
        storm_dx = self._wrapped_delta(lon, math.radians(storm_lon)) * np.cos(math.radians(storm_lat))
        storm_dy = lat - math.radians(storm_lat)
        storm_r = np.sqrt(storm_dx * storm_dx + storm_dy * storm_dy)
        storm_theta = np.arctan2(storm_dy, storm_dx)
        spiral_bands = np.exp(-(storm_r / math.radians(23)) ** 2) * (
            0.5 + 0.5 * np.cos(4 * storm_theta - 18 * storm_r)
        )
        humidity += 0.32 * spiral_bands
        humidity += rng.normal(0, 0.012, (ny, nx))
        xp = self.ctx.xp
        return (
            xp.asarray(zeta.astype(np.float32)),
            xp.asarray(np.clip(humidity, 0, 1).astype(np.float32)),
        )

    def _spectral_grids(self, ny, nx):
        xp = self.ctx.xp
        cache = getattr(self, "_fft_cache", {})
        key = (ny, nx, self.ctx.backend_name)
        if key not in cache:
            # Wavenumbers are in radians per full model domain. Omitting the
            # grid-size factor makes k too small, the inverse Laplacian too
            # large and the advecting velocity numerically explosive.
            ky = xp.fft.fftfreq(ny) * 2 * xp.pi * ny
            kx = xp.fft.rfftfreq(nx) * 2 * xp.pi * nx
            kxg, kyg = xp.meshgrid(kx, ky)
            k2 = kxg * kxg + kyg * kyg
            k2[0, 0] = 1.0
            lat = xp.linspace(-xp.pi / 2, xp.pi / 2, ny, dtype=xp.float32)[:, None]
            lon = xp.linspace(-xp.pi, xp.pi, nx, endpoint=False, dtype=xp.float32)[None, :]
            cache[key] = (k2, lat, lon)
            self._fft_cache = cache
        return cache[key]

    def step(self, zeta, humidity, warming, jet_strength, steps):
        """Advance a damped barotropic-vorticity and moisture system."""
        xp = self.ctx.xp
        ny, nx = zeta.shape
        k2, lat, lon = self._spectral_grids(ny, nx)
        dt = 0.045
        beta = 0.42 + 0.035 * float(warming)
        jet = 0.18 + 0.055 * float(jet_strength)
        viscosity = 0.014
        drag = 0.0035
        # Only the top few polar rows need a numerical sponge. Applying a
        # cosine taper everywhere erased mid-latitude storms over long runs.
        pole_taper = xp.clip(xp.cos(lat) / 0.28, 0, 1)
        for _ in range(steps):
            zk = xp.fft.rfft2(zeta)
            psik = -zk / k2
            psik[0, 0] = 0
            psi = xp.fft.irfft2(psik, s=(ny, nx)).real
            dpsi_dx = 0.5 * (xp.roll(psi, -1, 1) - xp.roll(psi, 1, 1))
            dpsi_dy = 0.5 * (xp.roll(psi, -1, 0) - xp.roll(psi, 1, 0))
            u = -dpsi_dy + jet * xp.cos(lat) ** 2
            v = dpsi_dx
            q = zeta + beta * xp.sin(lat)
            dqdx = 0.5 * (xp.roll(q, -1, 1) - xp.roll(q, 1, 1))
            dqdy = 0.5 * (xp.roll(q, -1, 0) - xp.roll(q, 1, 0))
            lap = 0.25 * (
                xp.roll(zeta, 1, 0) + xp.roll(zeta, -1, 0)
                + xp.roll(zeta, 1, 1) + xp.roll(zeta, -1, 1) - 4 * zeta
            )
            self.weather_time = getattr(self, "weather_time", 0.0) + dt
            forcing = 0.0015 * (1 + 0.22 * warming) * xp.sin(4 * lon - 0.8 * self.weather_time) * xp.cos(lat) ** 3
            zeta += dt * (-(u * dqdx + v * dqdy) + viscosity * lap - drag * zeta + forcing)
            zeta *= pole_taper

            dhdx = 0.5 * (xp.roll(humidity, -1, 1) - xp.roll(humidity, 1, 1))
            dhdy = 0.5 * (xp.roll(humidity, -1, 0) - xp.roll(humidity, 1, 0))
            hlap = 0.25 * (
                xp.roll(humidity, 1, 0) + xp.roll(humidity, -1, 0)
                + xp.roll(humidity, 1, 1) + xp.roll(humidity, -1, 1) - 4 * humidity
            )
            ocean_source = (0.005 + 0.0015 * warming) * xp.cos(lat) ** 2
            condensation = 0.004 * xp.maximum(0, -zeta - 0.35) * humidity
            humidity += dt * (-(u * dhdx + v * dhdy) + 0.08 * hlap + ocean_source - condensation - 0.002 * humidity)
            xp.clip(zeta, -3.5, 3.5, out=zeta)
            xp.clip(humidity, 0, 1, out=humidity)
        return zeta, humidity

    @staticmethod
    def land_mask(lat, lon):
        """Low-resolution procedural geography for orientation only."""
        land = np.zeros(np.broadcast_shapes(lat.shape, lon.shape), dtype=bool)
        latg = np.broadcast_to(lat, land.shape)
        long = np.broadcast_to(lon, land.shape)
        continents = [
            (-105, 45, 46, 25), (-83, 18, 19, 13), (-60, -18, 20, 35),
            (55, 47, 72, 24), (20, 5, 23, 34), (135, -25, 20, 13),
            (-42, 72, 13, 12), (48, -19, 7, 13),
        ]
        for lo, la, rx, ry in continents:
            dx = (np.degrees(long) - lo + 180) % 360 - 180
            dy = np.degrees(latg) - la
            land |= (dx / rx) ** 2 + (dy / ry) ** 2 < 1
        return land

    def atmosphere_map(self, zeta, humidity, warming):
        z = np.asarray(zeta, dtype=np.float32)
        h = np.asarray(humidity, dtype=np.float32)
        ny, nx = z.shape
        lat = np.linspace(-np.pi / 2, np.pi / 2, ny)[:, None]
        lon = np.linspace(-np.pi, np.pi, nx, endpoint=False)[None, :]
        land = self.land_mask(lat, lon)
        rgb = np.zeros((ny, nx, 3), dtype=np.float32)
        rgb[:] = (8, 29, 58)
        # Warm and cold vorticity lobes expose the atmospheric circulation.
        positive = np.clip(z / 1.7, 0, 1)[..., None]
        negative = np.clip(-z / 1.7, 0, 1)[..., None]
        rgb += positive * np.array([235, 82, 28]) + negative * np.array([20, 112, 225])
        latitude_heat = np.clip(np.cos(lat) ** 2 * (0.35 + 0.06 * warming), 0, 0.75)
        rgb += latitude_heat[..., None] * np.array([42, 26, 4])
        land_colour = np.stack([
            48 + 45 * np.cos(lat) ** 2,
            75 + 42 * np.cos(lat) ** 2,
            48 + 10 * np.cos(lat) ** 2,
        ], axis=-1)
        rgb = np.where(land[..., None], 0.55 * rgb + 0.45 * land_colour, rgb)
        cloud = np.clip((h - 0.64) * 2.05 + 0.16 * np.abs(z), 0, 0.72)[..., None]
        rgb = rgb * (1 - cloud) + np.array([238, 245, 252]) * cloud
        return np.clip(rgb, 0, 255).astype(np.uint8)

    @staticmethod
    def storm_position(zeta):
        z = np.asarray(zeta)
        ny, nx = z.shape
        lo = int(ny * 0.50)
        hi = int(ny * 0.82)
        sub = z[lo:hi]
        iy, ix = np.unravel_index(np.argmin(sub), sub.shape)
        iy += lo
        lon = -180.0 + 360.0 * ix / nx
        lat = -90.0 + 180.0 * iy / (ny - 1)
        return lon, lat, float(z[iy, ix])

    def globe(self, zeta, humidity, warming, size=(760, 620), central_lon=-25.0, compact=False):
        world = self.atmosphere_map(zeta, humidity, warming)
        w, h = size
        diameter = int(min(h * 0.92, w * (0.92 if compact else 0.86)))
        diameter = max(80, diameter)
        yy, xx = np.mgrid[0:diameter, 0:diameter]
        gx = (xx - (diameter - 1) / 2) / (diameter / 2)
        gy = ((diameter - 1) / 2 - yy) / (diameter / 2)
        rr = gx * gx + gy * gy
        mask = rr <= 1
        gz = np.sqrt(np.clip(1 - rr, 0, 1))
        lat = np.arcsin(np.clip(gy, -1, 1))
        lon = math.radians(central_lon) + np.arctan2(gx, gz)
        iy = np.clip(((lat + np.pi / 2) / np.pi * (world.shape[0] - 1)).astype(int), 0, world.shape[0] - 1)
        ix = np.mod(((lon + np.pi) / (2 * np.pi) * world.shape[1]).astype(int), world.shape[1])
        disc = world[iy, ix].astype(np.float32)
        shade = (0.34 + 0.66 * np.clip(gz, 0, 1))[..., None]
        disc *= shade
        disc[~mask] = 0
        alpha = (mask * 255).astype(np.uint8)
        rgba = np.dstack([np.clip(disc, 0, 255).astype(np.uint8), alpha])
        globe = Image.fromarray(rgba, "RGBA")
        glow = Image.new("RGBA", size, (0, 0, 0, 0))
        x0, y0 = (w - diameter) // 2, (h - diameter) // 2
        halo = Image.new("RGBA", size, (0, 0, 0, 0))
        hd = ImageDraw.Draw(halo, "RGBA")
        hd.ellipse((x0 - 7, y0 - 7, x0 + diameter + 7, y0 + diameter + 7), fill=(48, 142, 255, 75))
        halo = halo.filter(ImageFilter.GaussianBlur(max(6, diameter // 35)))
        glow = Image.alpha_composite(glow, halo)
        glow.alpha_composite(globe, (x0, y0))
        gd = ImageDraw.Draw(glow, "RGBA")
        gd.ellipse((x0, y0, x0 + diameter, y0 + diameter), outline=(164, 224, 255, 125), width=max(1, diameter // 180))
        storm_lon, storm_lat, _ = self.storm_position(zeta)
        rel = math.radians(storm_lon - central_lon)
        lat_r = math.radians(storm_lat)
        visible = math.cos(lat_r) * math.cos(rel) > 0
        if visible:
            gx_eye = math.cos(lat_r) * math.sin(rel)
            gy_eye = math.sin(lat_r)
            eye_x = x0 + diameter * 0.5 * (1 + gx_eye)
            eye_y = y0 + diameter * 0.5 * (1 - gy_eye)
            eye_r = max(3, diameter // (38 if compact else 52))
            gd.ellipse((eye_x-eye_r*1.8, eye_y-eye_r*1.8, eye_x+eye_r*1.8, eye_y+eye_r*1.8),
                       outline=(255, 154, 68, 95), width=max(1, eye_r // 3))
            gd.ellipse((eye_x-eye_r, eye_y-eye_r, eye_x+eye_r, eye_y+eye_r),
                       outline=(255, 215, 128, 245), width=max(1, eye_r // 3))
            gd.ellipse((eye_x-2, eye_y-2, eye_x+2, eye_y+2), fill=(255, 244, 216, 255))
        return glow.convert("RGB"), (x0, y0, diameter)

    def hero(self, zeta, humidity, i, done, total, warming, jet, uncertainty):
        zn, hn = to_numpy(zeta), to_numpy(humidity)
        centre = -32 + 18 * done / total
        canvas, bounds = self.globe(zn, hn, warming, size=(1280,720), central_lon=centre)
        canvas = add_title(
            canvas,
            "Storm Factory",
            f"barotropic-vorticity atmosphere + advected moisture · {zn.shape[1]}×{zn.shape[0]} globe · {self.ctx.backend_name}",
            badge="LIVE FORECAST",
        )
        add_progress(canvas, done / total, "OBSERVATIONS NOW", "+120 H FORECAST")
        return canvas

    def run(self):
        ny, nx = self.grid_shape(self.settings["n"])
        total = self.budget()
        warming = float(self.ctx.params.get("warming", 1.5))
        jet = float(self.ctx.params.get("jet_stream", 1.0))
        uncertainty = float(self.ctx.params.get("uncertainty", 25.0))
        self.weather_time = 0.0
        zeta, humidity = self.initialise(ny, nx, perturbation=uncertainty, warming=warming)
        done = 0
        for i in range(self.ctx.frames):
            target = int(round(total * (i + 1) / self.ctx.frames))
            zeta, humidity = self.step(zeta, humidity, warming, jet, max(1, target - done))
            done = target
            image = self.hero(zeta, humidity, i, done, total, warming, jet, uncertainty)
            self.ctx.save_frame(image, self.ctx.frame_path(i))
            zn=to_numpy(zeta); lon,lat,strength=self.storm_position(zn)
            self.ctx.write_status(i, f"forecast +{120 * done / total:.1f} h",{
                "forecast time":f"+{120*done/total:.1f} h","ocean warming":f"+{warming:.1f} °C",
                "jet stream":f"{jet:.2f}×","initial uncertainty":f"{uncertainty:.0f}%",
                "cyclone centre":f"{abs(lat):.1f}°{'N' if lat>=0 else 'S'}, {abs(lon):.1f}°{'E' if lon>=0 else 'W'}",
                "relative vorticity":f"{strength:+.3f}"})

        ens = max(1, int(self.settings.get("ensemble", 16)))
        side = max(1, int(math.ceil(math.sqrt(ens))))
        sy, sx = self.grid_shape(int(self.settings.get("sweep_n", max(36, ny // 2))))
        sweep_steps = int(self.settings.get("sweep_steps", max(160, total // 2)))
        tiles, labels = [], []
        for j in range(ens):
            self.weather_time = 0.0
            # The perturbation amplitude is tied to the visitor's uncertainty
            # control, while the seed represents a different admissible set of
            # observations inside that uncertainty.
            zj, hj = self.initialise(sy, sx, seed=100 + j, perturbation=uncertainty, warming=warming)
            zj, hj = self.step(zj, hj, warming, jet, sweep_steps)
            zn, hn = to_numpy(zj), to_numpy(hj)
            lon, lat, _ = self.storm_position(zn)
            tile, _ = self.globe(zn, hn, warming, size=(260, 146), central_lon=-20, compact=True)
            tiles.append(tile)
            labels.append(f"member {j + 1:02d} · {lat:.0f}°, {lon:.0f}°")
        reveal = mosaic(
            tiles,
            side,
            title="One forecast is not enough",
            subtitle="Tiny observational differences produce different storm positions. Every globe is an independent forecast.",
            labels=labels,
            label_fill=(190, 232, 255),
        )
        rp = self.ctx.run_dir / "reveal.jpg"
        self.ctx.save_frame(reveal, rp)
        self.ctx.finish(rp)
