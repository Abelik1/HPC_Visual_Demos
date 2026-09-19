from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from ..backend import to_numpy
from ..base import Demo
from ..pipeline import FramePipeline
from ..render import mosaic


HUBBLE_DEEP_FIELD = Path(__file__).resolve().parents[2] / "web" / "assets" / "hubble_deep_field.jpg"


class BlackHoleDemo(Demo):
    """Black-hole lensing: exact Schwarzschild ray tracing of the real Gaia sky,
    with the original weak-field Hubble-image lens kept as a comparison."""

    id = "black_hole"
    title = "Black-hole lensing"
    methods = ("schwarzschild", "weak_field")
    default_method = "schwarzschild"
    method_labels = {
        "schwarzschild": "Exact ray tracing · real Gaia sky",
        "weak_field": "Weak-field lens on the Hubble image (legacy)",
    }
    method_descriptions = {
        "schwarzschild": "Every pixel follows the exact Schwarzschild photon orbit from a camera held beside one "
                         "of the real Gaia black holes, into the 3-D Gaia DR3 sky seen from that place.",
        "weak_field": "The original demonstration: a weak-field point-lens formula distorting the Hubble Deep "
                      "Field photograph. Kept for comparison; it is an image warp, not ray tracing.",
    }
    timing_methods = {"background": "initialization", "lens": "simulation",
                      "integrate_rays": "simulation", "render_3d": "render"}

    def background(self, width, height, phase=0.0):
        """A moving crop of NASA's recorded Hubble Deep Field source plane."""
        if not HUBBLE_DEEP_FIELD.exists():
            raise FileNotFoundError("Missing Hubble Deep Field asset: web/assets/hubble_deep_field.jpg")
        with Image.open(HUBBLE_DEEP_FIELD) as source:
            source = source.convert("RGB")
            sw, sh = source.size
            crop_w = max(2, int(sw * .78))
            crop_h = max(2, int(crop_w * height / max(1, width)))
            if crop_h > sh:
                crop_h = max(2, int(sh * .78))
                crop_w = max(2, int(crop_h * width / max(1, height)))
            crop_w, crop_h = min(crop_w, sw), min(crop_h, sh)
            dx, dy = sw - crop_w, sh - crop_h
            p = float(phase) * 2 * math.pi
            left = int(round(dx * (.50 + .34 * math.sin(p))))
            top = int(round(dy * (.50 + .30 * math.cos(.83 * p))))
            image = source.crop((left, top, left + crop_w, top + crop_h))
            image = image.resize((int(width), int(height)), Image.Resampling.LANCZOS)
        return np.asarray(image, dtype=np.float32) / 255.0

    @staticmethod
    def lens_wells(params):
        count = max(1, min(3, int(round(float(params.get("lens_count", 1))))))
        x = float(params.get("lens_x", 0.0)) * 1.25
        y = float(params.get("lens_y", 0.0)) * .82
        separation = float(params.get("lens_separation", .62))
        angle = math.radians(float(params.get("lens_angle", 0.0)))
        wells = [(x, y, 1.0)]
        if count == 2:
            wells.append((x + separation * math.cos(angle), y + separation * math.sin(angle), .62))
        elif count == 3:
            for offset in (-math.pi / 3, math.pi / 3):
                wells.append((x + separation * math.cos(angle + offset),
                              y + separation * math.sin(angle + offset), .52))
        return wells

    def lens(self, background, mass, spin, progress=1.0, wells=None):
        """Map observer pixels through one or more weak-field point lenses."""
        xp = self.ctx.xp
        source = xp.asarray(background)
        height, width = background.shape[:2]
        y, x = xp.mgrid[0:height, 0:width]
        xx = (x - width * .5) / (height * .5)
        yy = (y - height * .5) / (height * .5)
        amount = .15 + .85 * float(progress)
        mapped_x, mapped_y = xx.copy(), yy.copy()
        capture = xp.zeros((height, width), dtype=bool)
        ring = xp.zeros((height, width), dtype=source.dtype)
        wells = wells or [(0.0, 0.0, 1.0)]
        for well_x, well_y, relative_mass in wells:
            dx, dy = xx - well_x, yy - well_y
            radius2 = dx * dx + dy * dy + 1e-4
            local_mass = float(mass) * float(relative_mass)
            einstein = .16 * local_mass * amount
            mapped_x -= einstein * einstein * dx / radius2
            mapped_y -= einstein * einstein * dy / radius2
            # Qualitative spin-like twist; this is not Kerr geodesic tracing.
            shear = .025 * spin * relative_mass * amount / (radius2 + .06)
            mapped_x -= shear * dy
            mapped_y += shear * dx
            radius = xp.sqrt(radius2)
            capture |= radius < (.055 + .018 * local_mass) * amount
            ring += xp.exp(-((radius - (.105 + .020 * local_mass)) / .018) ** 2) * amount
        if not getattr(self,"show_disk",True):
            ring *= 0
        sx = xp.clip((mapped_x * (height * .5) + width * .5).astype(xp.int32), 0, width - 1)
        sy = xp.clip((mapped_y * (height * .5) + height * .5).astype(xp.int32), 0, height - 1)
        output = source[sy, sx]
        output = xp.where(capture[..., None], 0, output)
        colour = xp.stack((1.5 * ring, .52 * ring, .12 * ring), axis=-1)
        return to_numpy(xp.clip(output + colour, 0, 1))

    @staticmethod
    def integrate_rays(mass, spin, wells=None, count=81, samples=150):
        """Advance photon directions through a reduced 3-D deflection field.

        This is intentionally labelled as a weak-field educational integrator,
        not a Schwarzschild/Kerr null-geodesic solver.  The resulting lines are
        nevertheless computed trajectories, including capture at a finite
        radius, rather than authored curves.
        """
        wells = wells or [(0.0, 0.0, 1.0)]
        side = max(3, int(math.ceil(math.sqrt(count))))
        axis = np.linspace(-1.55, 1.55, side)
        target = [(x, y, 4.2) for y in axis for x in axis]
        for well_x, well_y, _ in wells:
            target.extend((well_x + radius * math.cos(angle), well_y + radius * math.sin(angle), 4.2)
                          for radius in (.20, .43, .70) for angle in np.linspace(0, 2 * math.pi, 12, endpoint=False))
        target = np.asarray(target[:count], dtype=np.float64)
        position = np.repeat(np.array([[0.0, 0.0, -4.2]]), len(target), axis=0)
        direction = target - position
        direction /= np.linalg.norm(direction, axis=1, keepdims=True)
        paths = np.empty((samples, len(target), 3), dtype=np.float32)
        captured = np.zeros(len(target), dtype=bool)
        step = 8.8 / max(1, samples - 1)
        for sample in range(samples):
            paths[sample] = position
            active = ~captured
            if not active.any():
                paths[sample:] = position
                break
            p = position[active]
            d = direction[active]
            bend = np.zeros_like(p)
            hit = np.zeros(len(p), dtype=bool)
            for well_x, well_y, relative_mass in wells:
                offset = p - np.array((well_x, well_y, 0.0))
                radius2 = np.sum(offset * offset, axis=1) + .035
                # Only acceleration perpendicular to the current photon
                # direction changes it in this reduced weak-field model.
                radial = offset - d * np.sum(offset * d, axis=1, keepdims=True)
                local_mass = float(mass) * float(relative_mass)
                bend += -.055 * local_mass * radial / radius2[:, None] ** 1.5
                hit |= np.sqrt(np.sum(offset * offset, axis=1)) < (.16 + .035 * local_mass)
            radius2 = np.sum(p * p, axis=1) + .035
            transverse = np.column_stack((-p[:, 1], p[:, 0], np.zeros(len(p))))
            bend += .006 * float(spin) * transverse / (radius2[:, None] + .15)
            d += bend * step
            d /= np.maximum(np.linalg.norm(d, axis=1, keepdims=True), 1e-9)
            p += d * step
            direction[active] = d
            position[active] = p
            newly = active.copy()
            newly[active] = hit
            captured |= newly
        return paths, captured

    @staticmethod
    def _project(points):
        yaw, pitch = math.radians(-31), math.radians(18)
        cy, sy = math.cos(yaw), math.sin(yaw)
        cp, sp = math.cos(pitch), math.sin(pitch)
        x = points[..., 0] * cy + points[..., 2] * sy
        z = -points[..., 0] * sy + points[..., 2] * cy
        y = points[..., 1] * cp - z * sp
        depth = points[..., 1] * sp + z * cp
        scale = 98.0 / np.maximum(1.0, 1.0 + .035 * depth)
        return 640 + x * scale, 360 - y * scale

    def draw_source_plane(self, draw, background):
        """Map the same Hubble crop used by 2-D onto the 3-D source plane."""
        rows, cols = 16, 24
        height, width = background.shape[:2]
        for row in range(rows):
            for col in range(cols):
                x0, x1 = -1.8 + 3.6 * col / cols, -1.8 + 3.6 * (col + 1) / cols
                y0, y1 = -1.8 + 3.6 * row / rows, -1.8 + 3.6 * (row + 1) / rows
                corners = np.array(((x0, y0, 4.2), (x1, y0, 4.2),
                                    (x1, y1, 4.2), (x0, y1, 4.2)))
                px, py = self._project(corners)
                sample = background[min(height - 1, int((row + .5) * height / rows)),
                                    min(width - 1, int((col + .5) * width / cols))]
                draw.polygon(tuple(zip(px, py)), fill=(*[int(value * 255) for value in sample], 230))
        corners = np.array(((-1.8, -1.8, 4.2), (1.8, -1.8, 4.2),
                            (1.8, 1.8, 4.2), (-1.8, 1.8, 4.2), (-1.8, -1.8, 4.2)))
        px, py = self._project(corners)
        draw.line(tuple(zip(px, py)), fill=(135, 213, 255, 180), width=2)

    def render_3d(self, paths, captured, mass, wells, background, progress):
        image = Image.new("RGB", (1280, 720), (2, 5, 15))
        draw = ImageDraw.Draw(image, "RGBA")
        self.draw_source_plane(draw, background)
        # The observer grid makes the shared source-plane geometry legible.
        for z, colour in ((-4.2, (90, 230, 205, 85)),):
            grid=[]
            for value in np.linspace(-1.8, 1.8, 7):
                grid.extend((np.array(((-1.8, value, z), (1.8, value, z))),
                             np.array(((value, -1.8, z), (value, 1.8, z)))))
            for line in grid:
                xx, yy = self._project(line)
                draw.line(tuple(zip(xx, yy)), fill=colour, width=1)
        glow = Image.new("RGBA", image.size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        for well_x, well_y, relative_mass in wells:
            centre_x, centre_y = self._project(np.array([[well_x, well_y, 0.0]]))
            radius = (18 + 7 * mass) * relative_mass
            cx, cy = float(centre_x[0]), float(centre_y[0])
            gd.ellipse((cx-radius*2.2, cy-radius*2.2, cx+radius*2.2, cy+radius*2.2),
                       fill=(255, 104, 35, 72))
        glow = glow.filter(ImageFilter.GaussianBlur(17))
        image = Image.alpha_composite(image.convert("RGBA"), glow).convert("RGB")
        draw = ImageDraw.Draw(image, "RGBA")
        for well_x, well_y, relative_mass in wells:
            centre_x, centre_y = self._project(np.array([[well_x, well_y, 0.0]]))
            radius = (18 + 7 * mass) * relative_mass
            cx, cy = float(centre_x[0]), float(centre_y[0])
            draw.ellipse((cx-radius, cy-radius, cx+radius, cy+radius), fill=(0, 0, 0, 255),
                         outline=(255, 142, 62, 225), width=3)
        visible = max(2, min(len(paths), int(2 + progress * (len(paths) - 2))))
        for ray in range(paths.shape[1]):
            xx, yy = self._project(paths[:visible, ray])
            colour = (255, 116, 70, 145) if captured[ray] else (86, 220, 255, 135)
            draw.line(tuple(zip(xx, yy)), fill=colour, width=2)
            packet = min(visible - 1, max(0, int((progress * 1.9 % 1) * visible)))
            draw.ellipse((xx[packet]-3, yy[packet]-3, xx[packet]+3, yy[packet]+3),
                         fill=(255, 244, 190, 230))
        return image

    def run(self):
        if self.ctx.method == "weak_field":
            return self.run_weak_field()
        return self.run_schwarzschild()

    # ---- exact Schwarzschild ray tracing of the Gaia sky ------------------
    @staticmethod
    def camera_frame(target_position_pc, orbit_rad):
        """Camera axis (hole -> camera), forward, right and up, as unit ICRS vectors.

        The camera starts on the far side of the hole from the Galactic centre,
        so the bright core of the Milky Way sits behind the shadow, and circles
        the hole about the Galactic pole as the run advances.
        """
        from ..gaia_sky import GALACTIC_CENTRE, GALACTIC_CENTRE_PC, GALACTIC_NORTH
        to_centre = GALACTIC_CENTRE * GALACTIC_CENTRE_PC - np.asarray(target_position_pc)
        to_centre = to_centre / np.linalg.norm(to_centre)
        n = GALACTIC_NORTH / np.linalg.norm(GALACTIC_NORTH)
        c0 = -to_centre
        c = (c0 * math.cos(orbit_rad) + np.cross(n, c0) * math.sin(orbit_rad)
             + n * np.dot(n, c0) * (1 - math.cos(orbit_rad)))
        c = c / np.linalg.norm(c)
        forward = -c
        right = np.cross(forward, n); right = right / np.linalg.norm(right)
        up = np.cross(right, forward)
        return c, forward, right, up

    def render_camera(self, fine, glow, table, axis, forward, right, up, r_camera):
        from ..gaia_sky import sample_cube, sample_equirect
        from ..schwarzschild import lens_directions
        xp = self.ctx.xp
        width, height = int(self.settings["width"]), int(self.settings["height"])
        ss = max(1, int(self.settings.get("supersample", 1)))
        half = math.tan(math.radians(float(self.settings.get("fov_deg", 80))) / 2)
        f, rt, upv, ax = (xp.asarray(v, dtype=xp.float64) for v in (forward, right, up, axis))
        px = xp.arange(width, dtype=xp.float64)
        py = xp.arange(height, dtype=xp.float64)
        stars = xp.zeros((height, width, 3), dtype=xp.float64)
        diffuse = xp.zeros((height, width, 3), dtype=xp.float64)
        captured_share = 0.0
        for j in range(ss):
            for i in range(ss):
                x = ((px[None, :] + (i + .5) / ss) / width * 2 - 1) * half
                y = (1 - (py[:, None] + (j + .5) / ss) / height * 2) * half * height / width
                d = f + x[..., None] * rt + y[..., None] * upv
                d = d / xp.linalg.norm(d, axis=-1, keepdims=True)
                sky, captured = lens_directions(d, ax, r_camera, table, xp=xp)
                stars += xp.where(captured[..., None], 0.0, sample_cube(fine, sky, xp=xp))
                diffuse += xp.where(captured[..., None], 0.0, sample_equirect(glow, sky, xp=xp))
                captured_share += float(xp.mean(captured))
        # A static observer deep in the potential sees starlight blueshifted by
        # g = 1/sqrt(1 - 2M/r); bolometric specific intensity scales as g^4.
        boost = (1.0 / (1.0 - 2.0 / r_camera)) ** 2 / (ss * ss)
        return stars * boost, diffuse * boost, captured_share / (ss * ss)

    # Display calibration. Stars and diffuse light get separate curves, as in a
    # long-exposure photograph: resolved stars stay linear against a fixed flux
    # scale (only the bright ones rise above the black point), while the diffuse
    # light is given a contrast curve anchored to the brightest part of each
    # sky's own band, so a view from inside the Galactic disc (bright almost
    # everywhere) and one from BH2 (a narrow band) both keep a dark sky.
    STAR_GAIN = 0.006 / 1634.8          # calibrated on the Gaia BH3 sky
    DIFFUSE_PEAK = 1.0                  # display value of the sky's 99.9th percentile
    DIFFUSE_FLOOR = 0.02                # display value of the sky's median
    DIFFUSE_SATURATION = 0.6

    @classmethod
    def sky_reference(cls, glow, xp):
        """The diffuse sky's 99.9th percentile and the contrast exponent that
        maps its median to DIFFUSE_FLOOR and that percentile to DIFFUSE_PEAK.

        Skies differ a lot: from inside the disc the band is broad (the 99.5th
        percentile is ~3.6x the median), from BH2 it is narrow (~10x at the 99.5th). Solving
        for the exponent per sky gives each the same dark background and the
        same bright band.
        """
        from ..gaia_sky import sample_equirect
        rng = np.random.default_rng(0)
        d = rng.normal(size=(200_000, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        lum = sample_equirect(glow, xp.asarray(d), xp=xp) @ xp.asarray([0.2126, 0.7152, 0.0722])
        median, peak = (float(v) for v in xp.percentile(lum, xp.asarray([50.0, 99.9])))
        contrast = math.log(cls.DIFFUSE_PEAK / cls.DIFFUSE_FLOOR) / math.log(max(peak / max(median, 1e-12), 1.05))
        return {"peak": peak, "median": median, "contrast": contrast}

    def expose(self, stars, diffuse, reference):
        """Tone map to an 8-bit frame, with a soft glow around bright stars."""
        xp = self.ctx.xp
        exposure = float(self.settings.get("exposure", 1.0))
        diffuse_gain = float(self.settings.get("glow", 1.0))
        # Blackbody colours of ordinary stars are subtle; like a colour-boosted
        # astrophotograph, star_colour scales each pixel's distance from grey
        # (1 = true blackbody colour, 0 = greyscale).
        colour = max(0.0, float(self.settings.get("star_colour", 1.0)))
        weights = xp.asarray([0.2126, 0.7152, 0.0722])
        scaled = stars * exposure * self.STAR_GAIN
        grey = (scaled @ weights)[..., None]
        scaled = xp.maximum(grey + colour * (scaled - grey), 0.0)
        luminance = xp.maximum(diffuse @ weights, 1e-12)
        chroma = diffuse / luminance[..., None]
        chroma = xp.maximum(1.0 + self.DIFFUSE_SATURATION * min(colour, 1.0) * (chroma - 1.0), 0.0)
        sky = chroma * (exposure * diffuse_gain * self.DIFFUSE_PEAK
                        * (luminance / reference["peak"]) ** reference["contrast"])[..., None]
        height, width = scaled.shape[:2]
        fy = xp.fft.fftfreq(height)[:, None]
        fx = xp.fft.fftfreq(width)[None, :]
        glow = xp.zeros_like(scaled)
        # Bloom widths are tuned in 1280-wide pixels; keep the look at any size.
        for sigma, weight in ((1.4 * width / 1280, .45), (6.0 * width / 1280, .12)):
            kernel = xp.exp(-2 * (math.pi ** 2) * sigma * sigma * (fx * fx + fy * fy))
            for channel in range(3):
                glow[..., channel] += weight * xp.fft.ifft2(xp.fft.fft2(scaled[..., channel]) * kernel).real
        # Compress by the brightest channel so a bright star keeps its hue
        # instead of every channel saturating to white.
        total = scaled + glow + sky
        peak = xp.maximum(total.max(axis=-1, keepdims=True), 1e-12)
        mapped = total * (1.0 - xp.exp(-peak)) / peak
        image = xp.clip(mapped, 0, 1) ** (1 / 2.2)
        # Frames keep the rendered resolution, so a showcase run made for a
        # large screen is not thrown away at 1280x720.
        return Image.fromarray((to_numpy(image) * 255).astype(np.uint8))

    # Escaping rays are coloured by how far gravity turned them, from nearly
    # straight (blue) to rays that looped the hole (magenta); captured rays are red.
    BEND_STOPS = ((0.0, (0.25, 0.50, 1.00)), (30.0, (0.15, 0.85, 1.00)), (90.0, (0.25, 1.00, 0.45)),
                  (180.0, (1.00, 0.85, 0.20)), (360.0, (1.00, 0.30, 0.85)))
    CAPTURED_COLOUR = (1.00, 0.22, 0.14)

    @classmethod
    def bend_colour(cls, degrees):
        stops = cls.BEND_STOPS
        degrees = min(max(float(degrees), stops[0][0]), stops[-1][0])
        for (a, ca), (b, cb) in zip(stops, stops[1:]):
            if degrees <= b:
                f = (degrees - a) / (b - a)
                return tuple(x + f * (y - x) for x, y in zip(ca, cb))
        return stops[-1][1]

    RAY_VIEW_PITCH = math.radians(24)
    BACKDROP_FOV_DEG = 55.0      # about the camera view's pixel scale, so stars look alike
    BACKDROP_LEVEL = 0.8         # dimmed so the rays stay the brightest thing in the frame
    BACKDROP_GLOW = 0.45         # Milky Way glow turned down, so it does not wash out the rays
    SOURCE_CONE_DEG = 1.5        # a ray's source star: the brightest Gaia star this close to its exact direction

    @staticmethod
    def ray_view_to_world(axis, right, up):
        """Rotation from the ray diagram's local frame to ICRS (row vectors).

        The local frame has x along the camera axis; its y and z are mapped onto
        the camera's right and up. render_rays draws that frame mirrored, so y
        is mapped onto -right to show the sky the right way round.
        """
        return np.stack([np.asarray(axis), -np.asarray(right), np.asarray(up)])

    def ray_view_backdrop(self, fine, glow, reference, axis, right, up, view_angle, size=(1280, 720)):
        """The same Gaia sky as the camera view, seen past the hole from the
        ray-path view's own vantage point (not lensed), as display values in [0, 1].
        """
        from ..gaia_sky import sample_cube, sample_equirect
        xp = self.ctx.xp
        W, H = size
        cy, sy = math.cos(view_angle), math.sin(view_angle)
        cp, sp = math.cos(self.RAY_VIEW_PITCH), math.sin(self.RAY_VIEW_PITCH)
        screen_right = np.array([cy, sy, 0.0])
        screen_up = np.array([sy * sp, -cy * sp, cp])
        toward_viewer = np.array([-sy * cp, cy * cp, sp])
        to_world = self.ray_view_to_world(axis, right, up)
        half = math.tan(math.radians(self.BACKDROP_FOV_DEG) / 2)
        x = ((np.arange(W) + .5) / W * 2 - 1) * half
        y = (1 - (np.arange(H) + .5) / H * 2) * half * H / W
        d = (-toward_viewer + x[None, :, None] * screen_right + y[:, None, None] * screen_up) @ to_world
        d = xp.asarray(d / np.linalg.norm(d, axis=-1, keepdims=True))
        diffuse = sample_equirect(glow, d, xp=xp) * self.BACKDROP_GLOW
        image = self.expose(sample_cube(fine, d, xp=xp), diffuse, reference)
        return np.asarray(image, dtype=np.float32) / 255.0 * self.BACKDROP_LEVEL

    def source_stars(self, bundle, stars, to_world):
        """The real star each escaping ray's light came from, per ray.

        A ray traced back from the camera leaves along the exact escape angle
        phi: direction cos(phi) e1 + sin(phi) e2 in the diagram's frame. Its
        source is taken as the brightest catalogue star within SOURCE_CONE_DEG
        of that direction (the nearest one if none is that close). Returns
        {(plane, ray): (magnitude, rgb)} for rays that escape.
        """
        from ..gaia_sky import REFERENCE_MAG
        direction, flux, colour = stars
        cone = math.cos(math.radians(self.SOURCE_CONE_DEG))
        sources = {}
        for plane, ((e1, e2), rays, escape) in enumerate(bundle):
            for index, ((_, _, fate, _, _), phi) in enumerate(zip(rays, escape)):
                if fate != "escaped" or not np.isfinite(phi):
                    continue
                target = (math.cos(phi) * e1 + math.sin(phi) * e2) @ to_world
                closeness = direction @ target
                near = closeness > cone
                best = int(np.argmax(np.where(near, flux, -1.0))) if near.any() else int(np.argmax(closeness))
                sources[plane, index] = (REFERENCE_MAG - 2.5 * math.log10(flux[best]), colour[best])
        return sources

    def render_rays(self, r_camera, bundle, view_angle, backdrop=None, sources=None):
        """Exact geodesics around the hole, seen from outside, orthographically.

        backdrop, from ray_view_backdrop, is the star field drawn behind the
        rays; sources, from source_stars, puts each ray's own star at its end.
        """
        from ..schwarzschild import HORIZON, PHOTON_SPHERE
        W, H = 1280, 720
        extent = r_camera * 1.45
        scale = min(W, H) / (2.0 * extent)
        yaw, pitch = view_angle, self.RAY_VIEW_PITCH
        cy, sy, cp, sp = math.cos(yaw), math.sin(yaw), math.cos(pitch), math.sin(pitch)

        def project(points):
            x = points[:, 0] * cy + points[:, 1] * sy
            yy = -points[:, 0] * sy + points[:, 1] * cy
            z = points[:, 2]
            return W / 2 + x * scale, H / 2 - (z * cp - yy * sp) * scale, z * sp + yy * cp

        # Rays are drawn on two light layers, behind and in front of the
        # horizon, each given a soft bloom before the layers are stacked.
        layers = {True: Image.new("RGB", (W, H)), False: Image.new("RGB", (W, H))}
        draws = {k: ImageDraw.Draw(v) for k, v in layers.items()}

        def polyline(points, colour, width, dashed=False):
            xs, ys, depth = project(points)
            behind = (np.hypot(xs - W / 2, ys - H / 2) < HORIZON * scale) & (depth < 0)
            fill = tuple(int(255 * c) for c in colour)
            for k in range(len(xs) - 1):
                if dashed and k % 2:
                    continue
                draws[bool(behind[k])].line([(float(xs[k]), float(ys[k])), (float(xs[k + 1]), float(ys[k + 1]))],
                                            fill=fill, width=width)

        ring = np.array([[PHOTON_SPHERE * math.cos(a), PHOTON_SPHERE * math.sin(a), 0.0]
                         for a in np.linspace(0, 2 * math.pi, 181)])
        polyline(ring, (0.75, 0.62, 0.30), 2, dashed=True)
        ends, arrows = [], []
        for plane, ((e1, e2), rays, _) in enumerate(bundle):
            for index, (phi, r, fate, near_edge, bend) in enumerate(rays):
                pts = r[:, None] * (np.cos(phi)[:, None] * e1 + np.sin(phi)[:, None] * e2)
                colour = self.CAPTURED_COLOUR if fate == "captured" else self.bend_colour(bend)
                strength = 0.9 if near_edge else 0.6
                polyline(pts, tuple(c * strength for c in colour), 3 if near_edge else 2)
                if sources is not None and (plane, index) in sources:
                    ends.append((pts, sources[plane, index]))
                    # Light runs from the star to the camera: back along the traced
                    # path. Every third ray carries an arrow, to keep the fan legible.
                    if index % 3 == 0:
                        k = max(1, int(len(pts) * .55))
                        arrows.append((pts[k], pts[k - 1], colour))

        def lit(layer):
            sharp = np.asarray(layer, dtype=np.float32) / 255.0
            wide = np.asarray(layer.filter(ImageFilter.GaussianBlur(9)), dtype=np.float32) / 255.0
            near = np.asarray(layer.filter(ImageFilter.GaussianBlur(2.5)), dtype=np.float32) / 255.0
            return 0.8 * sharp + 0.5 * near + 0.9 * wide

        yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
        distance = np.hypot(xx - W / 2, yy - H / 2) / scale          # in units of M
        base = (0.008, 0.014, 0.035) if backdrop is None else (0.0, 0.0, 0.0)
        canvas = np.zeros((H, W, 3), dtype=np.float32) + np.array(base, dtype=np.float32)
        canvas += np.array([0.10, 0.07, 0.03], dtype=np.float32) * np.exp(-((distance - PHOTON_SPHERE) / 1.2) ** 2)[..., None]
        canvas += lit(layers[True])
        canvas[distance < HORIZON] = 0.0                                # the horizon hides what is behind it
        rim = np.exp(-((distance - HORIZON) / 0.12) ** 2) * (distance >= HORIZON)
        canvas += np.array([0.35, 0.18, 0.08], dtype=np.float32) * rim[..., None]
        canvas += lit(layers[False])
        # Overlapping rays add up; compress by the brightest channel so dense
        # bundles stay coloured instead of burning out to white.
        peak = np.maximum(canvas.max(axis=-1, keepdims=True), 1e-6)
        canvas = canvas * (1.0 - np.exp(-1.6 * peak)) / (1.6 * peak) * 1.6
        display = np.clip(canvas, 0, 1) ** (1 / 1.4)
        if backdrop is not None:
            # Screen the stars in after the curve, so they keep their camera-view
            # look; the horizon hides them like everything else behind it.
            # The sky also fades out behind the legend at the bottom.
            legend = np.clip((H - 150 - yy) / 60, 0.25, 1.0)
            stars = np.asarray(backdrop, dtype=np.float32) * ((distance >= HORIZON) * legend)[..., None]
            display = 1.0 - (1.0 - display) * (1.0 - stars)
        image = Image.fromarray((display * 255).astype(np.uint8))
        if ends:
            camera_xy = [float(v[0]) for v in project(np.array([[r_camera, 0.0, 0.0]]))[:2]]
            image = self.draw_sources(image, project, ends, arrows, camera_xy)

        draw = ImageDraw.Draw(image)
        cx, cyy, _ = project(np.array([[r_camera, 0.0, 0.0]]))
        draw.ellipse((cx[0] - 7, cyy[0] - 7, cx[0] + 7, cyy[0] + 7), fill=(255, 255, 255))
        try:
            font = ImageFont.load_default(size=20)
        except TypeError:                            # Pillow < 10.1 has one fixed size
            font = ImageFont.load_default()
        draw.text((cx[0] + 14, cyy[0] - 12), "camera", fill=(225, 232, 242), font=font)
        muted = (160, 170, 186)
        if ends:
            draw.text((24, H - 114), "star at a ray's end: the real Gaia star its light came from",
                      fill=muted, font=font)
        elif backdrop is not None:
            draw.text((24, H - 114), "behind: the camera's Gaia sky, where the escaping rays end (not lensed here)",
                      fill=muted, font=font)
        draw.text((24, H - 88), "black disc: horizon    dashed ring: photon sphere, where light can orbit",
                  fill=muted, font=font)
        draw.text((24, H - 36), "how far gravity bent the ray", fill=muted, font=font)
        bar_x, bar_y, bar_w = 310, H - 40, 360
        for i in range(bar_w):
            c = self.bend_colour(360.0 * i / (bar_w - 1))
            draw.line([(bar_x + i, bar_y + 6), (bar_x + i, bar_y + 22)], fill=tuple(int(255 * v) for v in c))
        for degrees in (0, 90, 180, 360):
            x = bar_x + bar_w * degrees / 360
            draw.line([(x, bar_y + 22), (x, bar_y + 28)], fill=muted)
        for degrees, text in ((0, "0°"), (90, "90°"), (180, "180°"), (360, "360°+")):
            draw.text((bar_x + bar_w * degrees / 360 - 8, bar_y - 18), text, fill=muted, font=font)
        red = tuple(int(255 * c) for c in self.CAPTURED_COLOUR)
        draw.line([(bar_x + bar_w + 50, bar_y + 14), (bar_x + bar_w + 90, bar_y + 14)], fill=red, width=4)
        draw.text((bar_x + bar_w + 100, bar_y + 3), "falls into the hole", fill=muted, font=font)
        return image

    @staticmethod
    def draw_sources(image, project, ends, arrows, camera_xy):
        """Each ray's source star at its far end, and an arrowhead on the ray
        pointing the way the light travels, towards the camera."""
        glow = Image.new("RGB", image.size)
        gd = ImageDraw.Draw(glow)
        cores = []
        width, height = image.size
        for path, (magnitude, colour) in ends:
            # The star goes at the path's far end, or where it leaves the picture.
            xs, ys, _ = project(path)
            bottom = height - 140                                           # clear of the legend
            inside = np.flatnonzero((xs > 16) & (xs < width - 16) & (ys > 16) & (ys < bottom))
            if not len(inside):
                continue
            last = inside[-1]
            x, y = float(xs[last]), float(ys[last])
            if last < len(xs) - 1:
                # Straight stretches are sampled sparsely: cut the segment that
                # leaves the picture at the frame edge.
                ex, ey = float(xs[last + 1]) - x, float(ys[last + 1]) - y
                reach = [(bound - start) / step for start, step, bound in
                         ((x, ex, 16 if ex < 0 else width - 16), (y, ey, 16 if ey < 0 else bottom))
                         if abs(step) > 1e-9]
                t = min([1.0] + reach)
                x, y = x + t * ex, y + t * ey
            # Rays aimed near the viewer end, foreshortened, on top of the camera.
            if math.hypot(x - camera_xy[0], y - camera_xy[1]) < 45:
                continue
            size = min(max((8.5 - magnitude) / 6.0, 0.0), 1.0)          # G ~ 2.5 biggest, G >= 8.5 smallest
            hue = np.asarray(colour, dtype=float) / max(float(np.max(colour)), 1e-9)
            radius = 4 + 7 * size
            gd.ellipse((x - radius, y - radius, x + radius, y + radius),
                       fill=tuple(int(255 * (.35 + .35 * size) * c) for c in hue))
            cores.append((x, y, size, hue))
        glow = np.asarray(glow.filter(ImageFilter.GaussianBlur(6)), dtype=np.float32) / 255.0
        base = np.asarray(image, dtype=np.float32) / 255.0
        image = Image.fromarray(((1 - (1 - base) * (1 - glow)) * 255).astype(np.uint8))
        draw = ImageDraw.Draw(image)
        for x, y, size, hue in cores:
            tint = tuple(int(255 * (.55 + .45 * c)) for c in hue)
            if size > .4:
                spike = 3 + 8 * size
                draw.line([(x - spike, y), (x + spike, y)], fill=tint, width=1)
                draw.line([(x, y - spike), (x, y + spike)], fill=tint, width=1)
            core = 1.5 + 2 * size
            draw.ellipse((x - core, y - core, x + core, y + core), fill=tint)
        for tail, head, colour in arrows:
            x, y, _ = project(np.stack([tail, head]))
            dx, dy = float(x[1] - x[0]), float(y[1] - y[0])
            length = math.hypot(dx, dy)
            if length < 1e-6:
                continue
            dx, dy = dx / length, dy / length
            tip = (float(x[0]) + 4 * dx, float(y[0]) + 4 * dy)
            fill = tuple(int(255 * min(1.0, .35 + .8 * c)) for c in colour)
            draw.polygon([tip, (tip[0] - 9 * dx + 4.5 * dy, tip[1] - 9 * dy - 4.5 * dx),
                          (tip[0] - 9 * dx - 4.5 * dy, tip[1] - 9 * dy + 4.5 * dx)], fill=fill)
        return image

    @staticmethod
    def ray_bundle(r_camera, table=None):
        """Fans of exact geodesics in three planes through the camera axis.

        Each ray carries its bend: the angle between the direction it leaves
        the camera along and the direction it finally escapes to. Each plane
        also carries the rays' exact escape angles (NaN if captured).
        """
        from ..schwarzschild import EscapeTable, shadow_half_angle, trace_fan
        edge = shadow_half_angle(r_camera)
        psis = np.unique(np.concatenate([np.linspace(0.02, math.pi - 0.02, 23),
                                         edge + np.array([-0.02, -0.004, 0.004, 0.02])]))
        rays = trace_fan(r_camera, psis, r_max=r_camera * 1.45, max_phi=3 * math.pi)
        e1 = np.array([1.0, 0.0, 0.0])
        # The exact escape angle, not the end of the drawn (truncated) path.
        escape, _ = (table or EscapeTable(r_camera, samples=2000)).escape_angle(psis)
        described = []
        for psi, (phi, r, fate), angle in zip(psis, rays, escape):
            bend = math.degrees(max(0.0, angle - (math.pi - psi))) if np.isfinite(angle) else 0.0
            described.append((phi, r, fate, abs(psi - edge) < 0.03, bend))
        bundle = []
        for chi in np.radians([90, 150, 210]):
            e2 = np.array([0.0, math.cos(chi), math.sin(chi)])
            bundle.append(((e1, e2), described, escape))
        return bundle

    def run_schwarzschild(self):
        from ..gaia_sky import BACKGROUND_PC, TARGET_KEYS, bright_stars, build_sky
        from ..schwarzschild import EscapeTable, KM_PER_SCHWARZSCHILD_RADIUS_PER_MSUN, shadow_half_angle
        params = self.ctx.params
        self.ctx.jpeg_subsampling = 0           # keep star colour in single pixels
        target_key = TARGET_KEYS[int(round(float(params.get("target", 0)))) % len(TARGET_KEYS)]
        start_rs = float(params.get("camera_distance", 10.0))
        dive = float(params.get("dive", 0.0))
        orbit = math.radians(float(params.get("orbit", 120.0)))
        with self.ctx.stage("initialization"):
            fine, glow, sky = build_sky(target_key, face=int(self.settings.get("sky_face", 2048)))
            fine, glow = self.ctx.xp.asarray(fine), self.ctx.xp.asarray(glow)
            reference = self.sky_reference(glow, self.ctx.xp)
            source_catalogue = bright_stars(target_key)
        target = sky["target"]
        mass = float(target["mass_msun"])
        km_per_rs = KM_PER_SCHWARZSCHILD_RADIUS_PER_MSUN * mass
        width, height = int(self.settings["width"]), int(self.settings["height"])
        ss = max(1, int(self.settings.get("supersample", 1)))
        rays_per_frame = width * height * ss * ss
        mode_dir = self.ctx.run_dir / "modes" / "3d"
        mode_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.write_meta({
            "view_modes": [{"id": "frames", "label": "Camera view", "folder": "frames"},
                           {"id": "3d", "label": "Exact ray paths", "folder": "modes/3d"}],
            "default_view_mode": "frames",
            "black_hole": {"key": target_key, **target},
            "sky": sky,
            "physics": {
                "metric": "Schwarzschild (non-rotating); exact null geodesics",
                "method": "escape angle per ray angle by Gauss-Legendre quadrature of the exact orbit equation, "
                          "checked against Runge-Kutta integration; better than 1e-5 rad",
                "observer": "static camera held in place (not orbiting); static-frame aberration included",
                "brightness": "gravitational blueshift g^4 for a static observer; lensing conserves surface brightness",
                "not_modelled": ["black-hole spin (Kerr)", "accretion disc", "the companion star",
                                 "an orbiting camera's own motion",
                                 "dust extinction re-derived for the new line of sight",
                                 "stars brighter than G = 3 (saturated in Gaia) and stars beyond the catalogue cuts",
                                 f"stars without a good parallax are placed on a {BACKGROUND_PC:g} pc shell"],
            },
        })
        tables, bundles = {}, {}
        # Ray tracing and exposure on the device; the ray-path diagram (tens of
        # thousands of PIL polyline segments per frame) is drawn by worker
        # processes on the allocated cores while the next frame is traced.
        workers = max(1, min(self.ctx.cpu_workers - 1, 16, max(1, self.ctx.frames // 6)))
        with FramePipeline(self.ctx, workers=workers) as diagrams:
            for frame in range(self.ctx.frames):
                progress = frame / max(1, self.ctx.frames - 1)
                r_rs = max(1.6, start_rs * (1 - dive * progress))
                r_camera = 2.0 * r_rs
                key = round(r_camera, 4)
                if key not in tables:
                    if len(tables) > 4:
                        tables.clear()
                    tables[key] = EscapeTable(r_camera, samples=4000)
                axis, forward, right, up = self.camera_frame(sky["viewpoint_pc"], orbit * progress)
                with self.ctx.stage("simulation"):
                    stars, diffuse, captured = self.render_camera(fine, glow, tables[key], axis, forward, right, up, r_camera)
                with self.ctx.stage("visualization"):
                    self.ctx.save_frame(self.expose(stars, diffuse, reference), self.ctx.frame_path(frame))
                    bundle_key = round(r_camera, 2)
                    if bundle_key not in bundles:
                        if len(bundles) > 4:
                            bundles.clear()
                        bundles[bundle_key] = self.ray_bundle(r_camera, tables[key])
                    view_angle = .6 + 1.2 * progress
                    backdrop = self.ray_view_backdrop(fine, glow, reference, axis, right, up, view_angle)
                    sources = self.source_stars(bundles[bundle_key], source_catalogue,
                                                self.ray_view_to_world(axis, right, up))
                half = math.degrees(shadow_half_angle(r_camera))
                diagrams.submit(frame, draw_ray_diagram,
                                (r_camera, bundles[bundle_key], view_angle, backdrop, sources),
                                mode_dir / f"frame_{frame:04d}.jpg",
                                f"tracing {rays_per_frame:,} exact photon orbits", {
                    "black hole": f"{target['label']} · {mass:g} M☉",
                    "camera distance": f"{r_rs:.1f} r_s · {r_rs * km_per_rs:,.0f} km",
                    "shadow": f"{2 * half:.1f}° across",
                    "starlight blueshift": f"×{1 / math.sqrt(1 - 2 / r_camera):.3f}",
                    "stars in the sky": f"{sky['stars']:,} (Gaia DR3)",
                    "rays this frame": f"{rays_per_frame:,}",
                    "frame in shadow": f"{100 * captured:.1f}%",
                    "distance from Earth": "here" if target_key == "sun" else f"{target['distance_pc']:,.0f} pc",
                })
        # Camera frames go through the context's own encoder threads; make sure
        # they are all on disk before the run is marked complete.
        self.ctx.flush_frames()
        self.ctx.finish(None)

    def run_weak_field(self):
        width = int(self.settings["width"])
        height = int(self.settings["height"])
        mass = float(self.ctx.params.get("mass", 1.35))
        spin = float(self.ctx.params.get("spin", .55))
        wells = self.lens_wells(self.ctx.params)
        self.show_disk = bool(round(float(self.ctx.params.get("disk",1))))
        paths, captured = self.integrate_rays(mass, spin, wells=wells)
        mode_dir = self.ctx.run_dir / "modes" / "3d"
        mode_dir.mkdir(parents=True, exist_ok=True)
        self.ctx.write_meta({
            "view_modes": [
                {"id": "frames", "label": "2D observer image", "folder": "frames"},
                {"id": "3d", "label": "3D ray space", "folder": "modes/3d"},
            ],
            "default_view_mode": "frames",
            "source_plane": {"name":"Hubble Deep Field (PIA12110)",
                             "file":"web/assets/hubble_deep_field.jpg",
                             "credit":"NASA/JPL-Caltech/STScI"},
            "lens_wells": [{"x":x,"y":y,"relative_mass":weight}
                           for x,y,weight in wells],
            "physics": {"3d": "reduced weak-field numerical photon integration",
                        "2d": "parallel thin-lens image mapping",
                        "limitation": "not a Kerr geodesic or GRMHD solver"},
        })
        for frame in range(self.ctx.frames):
            progress = (frame + 1) / self.ctx.frames
            background = self.background(width,height,phase=.22*progress)
            observer = Image.fromarray((self.lens(background,mass,spin,progress,wells) * 255).astype(np.uint8))
            observer = observer.resize((1280, 720), Image.Resampling.LANCZOS)
            observer = Image.blend(observer, observer.filter(ImageFilter.GaussianBlur(9)), .10)
            self.ctx.save_frame(observer, self.ctx.frame_path(frame))
            self.ctx.save_frame(self.render_3d(paths,captured,mass,wells,background,progress),
                                mode_dir / f"frame_{frame:04d}.jpg")
            self.ctx.write_status(frame, "Integrating photon paths through 3-D space", {
                "lens mass": f"{mass:.2f}", "dimensionless spin": f"{spin:+.2f}",
                "observer sight lines": f"{width * height:,}", "3D rays": f"{paths.shape[1]}",
                "integration samples": f"{paths.shape[0]}",
                "captured rays": f"{captured.sum()} / {len(captured)}",
            })
        ensemble = max(1, int(self.ctx.params.get("_parallel_count", self.settings.get("ensemble", 16))))
        side = max(1, int(math.ceil(math.sqrt(ensemble))))
        tiles=[]
        small_background = self.background(260, 150, phase=.42)
        for j in range(ensemble):
            trial_mass = max(.45, mass * (.65 + .7 * (j % side) / max(1, side - 1)))
            trial_spin = -.9 + 1.8 * (j // side) / max(1, side - 1)
            tiles.append(Image.fromarray((self.lens(small_background,trial_mass,trial_spin,wells=wells)*255).astype(np.uint8)))
        reveal = mosaic(tiles, side, title="That was one observer. Leonardo can explore many.")
        reveal_path = self.ctx.run_dir / "reveal.jpg"
        self.ctx.save_frame(reveal, reveal_path)
        self.ctx.finish(reveal_path)


def draw_ray_diagram(args):
    """The exact-ray-paths view for one frame (runs in a worker process)."""
    demo = BlackHoleDemo.__new__(BlackHoleDemo)   # class constants and drawing helpers only
    return demo.render_rays(*args)

