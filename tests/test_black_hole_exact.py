"""Exact Schwarzschild ray tracing and the Gaia sky it looks at."""
import json
import math
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from leonardo_demos import gaia_sky
from leonardo_demos.base import RunContext
from leonardo_demos.demos.black_hole import BlackHoleDemo
from leonardo_demos.schwarzschild import (B_CRITICAL, EscapeTable, _sweep_to, impact_parameter,
                                          lens_directions, shadow_half_angle, trace_fan, trace_path,
                                          turning_point)


def exact_escape(r, psi, nodes=96):
    """Escape angle by direct quadrature, without the table's interpolation."""
    b = impact_parameter(r, psi)
    if math.cos(psi) > 0:
        u0 = turning_point(np.array([b]))
        return float(2 * _sweep_to(u0, np.array([b]), nodes, turning=np.array([True]))[0]
                     - _sweep_to(np.minimum([1 / r], u0), np.array([b]), nodes)[0])
    return float(_sweep_to(np.array([1 / r]), np.array([b]), nodes)[0])


class GeodesicTests(unittest.TestCase):
    def test_weak_field_limit_matches_the_second_order_deflection(self):
        # Far from the hole: alpha = 4M/b + 15 pi M^2 / (4 b^2) + O(M^3/b^3).
        for b in (200.0, 1000.0):
            u0 = turning_point(np.array([b]))
            alpha = 2 * _sweep_to(u0, np.array([b]), 96, turning=np.array([True]))[0] - math.pi
            series = 4 / b + 15 * math.pi / (4 * b * b)
            self.assertLess(abs(alpha - series) / series, 2e-3)

    def test_rays_split_exactly_at_the_critical_impact_parameter(self):
        r = 20.0
        table = EscapeTable(r)
        edge = shadow_half_angle(r)
        _, captured = table.escape_angle(np.array([edge - 1e-7, edge + 1e-7]))
        self.assertEqual(captured.tolist(), [True, False])
        self.assertAlmostEqual(impact_parameter(r, edge), B_CRITICAL, places=9)

    def test_winding_diverges_logarithmically_near_the_photon_sphere(self):
        # Strong-deflection limit for Schwarzschild: d(phi)/d ln(b - b_c) = -1.
        gaps = np.geomspace(1e-9, 1e-5, 12)
        b = B_CRITICAL + gaps
        u0 = turning_point(b)
        phi = 2 * _sweep_to(u0, b, 96, turning=np.ones(len(b), bool))
        slope = np.polyfit(np.log(gaps), phi, 1)[0]
        self.assertAlmostEqual(slope, -1.0, places=3)

    def test_table_agrees_with_quadrature_and_with_runge_kutta(self):
        rng = np.random.default_rng(3)
        for r in (3.4, 20.0, 80.0):
            table = EscapeTable(r)
            psis = np.concatenate([rng.uniform(0, math.pi, 60),
                                   table.psi_edge + np.geomspace(1e-9, 1e-3, 10),
                                   math.pi / 2 + rng.normal(0, 1e-4, 6)])
            phi, captured = table.escape_angle(psis)
            for p, value, cap in zip(psis, phi, captured):
                if not cap:
                    self.assertLess(abs(value - exact_escape(r, p)), 2e-5, (r, p))
        table = EscapeTable(20.0)
        for degrees in (20, 60, 95, 150):
            psi = math.radians(degrees)
            phis, radii, fate = trace_path(20.0, psi, step=2e-5, r_escape=1e7)
            self.assertEqual(fate, "escaped")
            rk4 = phis[-1] + math.asin(impact_parameter(20.0, psi) / radii[-1])
            self.assertLess(abs(rk4 - table.escape_angle(np.array([psi]))[0][0]), 5e-5)

    def test_fan_tracer_matches_the_single_ray_tracer(self):
        psis = np.radians([10, 14.2, 14.4, 40, 150])
        for (phi, r, fate), psi in zip(trace_fan(20.0, psis, r_max=60), psis):
            phi1, r1, fate1 = trace_path(20.0, psi, step=4e-3, r_escape=60)
            self.assertEqual(fate, fate1)
            self.assertAlmostEqual(phi[-1], phi1[-1], places=6)

    def test_without_gravity_directions_are_unchanged(self):
        d = np.random.default_rng(1).normal(size=(50, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        sky, captured = lens_directions(d, np.array([0, 0, 1.0]), 1e9)
        self.assertLess(np.abs(sky[~captured] - d[~captured]).max(), 1e-4)

    def test_shadow_has_the_analytic_angular_size(self):
        # Rays across the frame's centre row: the captured band must span
        # exactly twice the analytic half-angle.
        r = 16.0
        angles = np.linspace(-0.6, 0.6, 24001)
        d = np.stack([np.sin(angles), np.zeros_like(angles), -np.cos(angles)], axis=1)
        _, captured = lens_directions(d, np.array([0, 0, 1.0]), r)
        width = angles[captured].max() - angles[captured].min()
        self.assertAlmostEqual(width, 2 * shadow_half_angle(r), delta=2 * (angles[1] - angles[0]))

    def test_camera_inside_the_photon_sphere_is_refused(self):
        with self.assertRaises(ValueError):
            EscapeTable(2.9)


class SkyTests(unittest.TestCase):
    def test_cube_map_puts_a_star_where_it_is_and_conserves_flux(self):
        rng = np.random.default_rng(0)
        directions = rng.normal(size=(400, 3))
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        flux = rng.uniform(.5, 2, 400)
        colour = np.ones((400, 3))
        size = 64
        cube = gaia_sky.splat_cube(directions, flux, colour, size)
        centres = (np.arange(size) + .5) / size * 2 - 1
        a, b = np.meshgrid(centres, centres, indexing="xy")
        solid = (2.0 / size) ** 2 / (1 + a * a + b * b) ** 1.5
        self.assertAlmostEqual(float((cube[..., 0] * solid).sum()), float(flux.sum()), delta=flux.sum() * 1e-4)
        one = gaia_sky.splat_cube(directions[:1], flux[:1], colour[:1], size)
        at_star = gaia_sky.sample_cube(one, directions[:1])[0, 0]
        opposite = gaia_sky.sample_cube(one, -directions[:1])[0, 0]
        self.assertGreater(at_star, 0)
        self.assertEqual(opposite, 0)

    def test_cube_faces_cover_every_direction_once(self):
        d = np.random.default_rng(2).normal(size=(20000, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        face, u, v = gaia_sky.cube_coordinates(d)
        self.assertEqual(sorted(set(face.tolist())), list(range(6)))
        self.assertTrue(((u >= 0) & (u <= 1) & (v >= 0) & (v <= 1)).all())

    def test_moving_the_viewpoint_applies_parallax_and_inverse_square_brightness(self):
        stars = {"ra": np.array([0.0, 90.0]), "dec": np.array([0.0, 0.0]),
                 "parallax": np.array([10.0, 10.0]), "parallax_error": np.array([.1, .1]),
                 "phot_g_mean_mag": np.array([5.0, 5.0]), "bp_rp": np.array([1.0, 1.0])}
        with patch.object(gaia_sky, "load_catalogue", return_value=(stars, {})):
            # Both stars are 100 pc from Earth. From 50 pc towards the first,
            # it is half as far (1.5 mag brighter) and still straight ahead.
            viewpoint = np.array([50.0, 0.0, 0.0])
            direction, flux, _, summary = gaia_sky.stars_from(viewpoint)
        self.assertTrue(np.allclose(direction[0], [1, 0, 0]))
        self.assertAlmostEqual(-2.5 * math.log10(flux[0] / 10 ** (-0.4 * (5 - gaia_sky.REFERENCE_MAG))),
                               5 * math.log10(50 / 100), places=6)
        expected = np.array([-50.0, 100.0, 0.0]) / math.hypot(50, 100)
        self.assertTrue(np.allclose(direction[1], expected))
        self.assertEqual(summary["stars"], 2)

    def test_companion_star_is_left_out(self):
        stars = {"ra": np.array([294.82786, 294.9]), "dec": np.array([14.93098, 14.93098]),
                 "parallax": np.array([1.64, 1.64]), "parallax_error": np.array([.07, .07]),
                 "phot_g_mean_mag": np.array([11.2, 11.2]), "bp_rp": np.array([1.2, 1.2])}
        with patch.object(gaia_sky, "load_catalogue", return_value=(stars, {})):
            _, _, _, summary = gaia_sky.stars_from(np.zeros(3), exclude_direction=gaia_sky.radec_unit(294.82786, 14.93098))
        self.assertEqual(summary["stars"], 1)

    def test_colour_runs_from_blue_hot_to_red_cool(self):
        rgb = gaia_sky.rgb_from_teff(gaia_sky.teff_from_bp_rp(np.array([-0.2, 0.8, 2.5])))
        self.assertGreater(rgb[0, 2] / rgb[0, 0], 1.0)   # hot star bluer than red
        self.assertLess(rgb[2, 2] / rgb[2, 0], rgb[1, 2] / rgb[1, 0])

    def test_typical_catalogue_star_is_shown_white(self):
        # Balanced like an astrophotograph: the median star is neutral, so
        # both blue and orange stars appear rather than an all-orange sky.
        white = gaia_sky.rgb_from_teff(gaia_sky.WHITE_POINT_K)
        self.assertTrue(np.allclose(white, 1.0))
        rgb = gaia_sky.rgb_from_teff(gaia_sky.teff_from_bp_rp(np.array([0.3, 2.0])))
        self.assertGreater(rgb[0, 2], rgb[0, 0])
        self.assertGreater(rgb[1, 0], rgb[1, 2])


class BlackHoleDemoTests(unittest.TestCase):
    def test_exact_run_writes_camera_and_ray_views(self):
        stars = {"ra": np.random.default_rng(0).uniform(0, 360, 3000),
                 "dec": np.degrees(np.arcsin(np.random.default_rng(1).uniform(-1, 1, 3000))),
                 "parallax": np.full(3000, 2.0), "parallax_error": np.full(3000, .1),
                 "phot_g_mean_mag": np.random.default_rng(2).uniform(4, 11, 3000),
                 "bp_rp": np.full(3000, 1.0)}
        metadata = {"source": "test", "credit": "test", "selection": {}, "retrieved": "today",
                    "black_holes": {"gaia_bh3": {"label": "Gaia BH3", "mass_msun": 32.7, "distance_pc": 590.0,
                                                 "ra_deg": 294.82786, "dec_deg": 14.93098}}}
        with tempfile.TemporaryDirectory() as t, patch.object(gaia_sky, "load_catalogue", return_value=(stars, metadata)):
            ctx = RunContext(Path(t), "black_hole", "local", 2,
                             {"target": 0, "camera_distance": 6, "orbit": 30, "dive": .2}, "numpy", method="schwarzschild")
            BlackHoleDemo(ctx, {"width": 96, "height": 54, "supersample": 1, "sky_face": 64,
                                "fov_deg": 80, "exposure": 1.0, "glow": 1.0}).run()
            root = Path(t)
            for path in ("frames/frame_0001.jpg", "modes/3d/frame_0001.jpg", "frame_data/frame_0001.json"):
                self.assertTrue((root / path).exists(), path)
            meta = json.loads((root / "meta.json").read_text())
            self.assertEqual(meta["status"], "complete")
            self.assertEqual(meta["method"], "schwarzschild")
            self.assertEqual(meta["black_hole"]["mass_msun"], 32.7)
            self.assertEqual([m["id"] for m in meta["view_modes"]], ["frames", "3d"])
            values = json.loads((root / "frame_data/frame_0001.json").read_text())["values"]
            # Dive 0.2 from 6 r_s ends at 4.8 r_s; the readout uses the real mass.
            self.assertIn("4.8 r_s", values["camera distance"])
            self.assertIn(f"{4.8 * 2.95325 * 32.7:,.0f} km", values["camera distance"])

    def test_ray_view_colours_rays_by_their_exact_bend(self):
        r = 20.0
        bundle = BlackHoleDemo.ray_bundle(r, EscapeTable(r))
        bends = [(fate, near, bend) for _, _, fate, near, bend in bundle[0][1]]
        escaped = [bend for fate, near, bend in bends if fate == "escaped"]
        grazing = [bend for fate, near, bend in bends if fate == "escaped" and near]
        # Rays leaving straight away from the hole barely turn; rays that
        # skim the photon sphere turn by more than half a circle.
        self.assertLess(min(escaped), 1.0)
        self.assertGreater(max(grazing), 180.0)
        self.assertEqual(BlackHoleDemo.bend_colour(0), BlackHoleDemo.BEND_STOPS[0][1])
        self.assertTrue(np.allclose(BlackHoleDemo.bend_colour(1e4), BlackHoleDemo.BEND_STOPS[-1][1]))
        image = BlackHoleDemo(RunContext(Path("."), "black_hole", "local", 1, {}, "numpy",
                                         method="schwarzschild"), {}).render_rays(r, bundle, 0.6)
        pixels = np.asarray(image, dtype=float)
        self.assertGreater((pixels.max(axis=-1) - pixels.min(axis=-1)).max(), 120)   # not greyscale

    def test_missing_catalogue_explains_how_to_get_it(self):
        gaia_sky.load_catalogue.cache_clear()
        with self.assertRaisesRegex(gaia_sky.CatalogueMissing, "fetch_gaia_sky.py"):
            gaia_sky.load_catalogue(str(Path(tempfile.gettempdir()) / "no_such_gaia_sky.npz"))


if __name__ == "__main__":
    unittest.main()
