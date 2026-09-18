"""The Gaia DR3 sky as seen from a chosen point in the Galaxy.

Every star is placed in 3-D from its position and parallax, then re-seen from
the viewpoint: its direction changes by parallax and its brightness by the
inverse-square law (magnitude + 5 log10(new distance / distance from Earth)).
The result is written into a cube map of radiance that the black-hole ray
tracer samples for every escaping ray.

Deliberate simplifications, recorded in each run's metadata:
* interstellar extinction is the one measured towards Earth, not re-derived
  for the new sightline;
* stars without a parallax measured to better than 20 % are placed on a
  distant background shell rather than at a guessed depth;
* Gaia is incomplete for the very brightest stars (G < ~3), which only matters
  when the viewpoint is the Solar System;
* the diffuse glow is the catalogue's own starlight heavily blurred, standing
  in for the unresolved stars fainter than the catalogue.
"""
from __future__ import annotations

import json
import math
from functools import lru_cache
from pathlib import Path

import numpy as np

CATALOGUE = Path(__file__).resolve().parents[1] / "data" / "gaia_sky.npz"
BACKGROUND_PC = 3000.0          # stars with poor parallaxes
REFERENCE_MAG = 6.0             # a G = 6 star has unit flux
GALACTIC_CENTRE_PC = 8178.0     # GRAVITY Collaboration 2019

# ICRS unit vectors of the Galactic centre and the north Galactic pole.
def radec_unit(ra_deg, dec_deg):
    ra, dec = np.radians(ra_deg), np.radians(dec_deg)
    return np.stack([np.cos(dec) * np.cos(ra), np.cos(dec) * np.sin(ra), np.sin(dec)], axis=-1)


GALACTIC_CENTRE = radec_unit(266.40499, -28.93617)
GALACTIC_NORTH = radec_unit(192.85948, 27.12825)

SUN_TARGET = {"label": "A black hole in the Solar System", "mass_msun": 10.0, "mass_err_msun": None,
              "distance_pc": 0.0, "ra_deg": 0.0, "dec_deg": 0.0,
              "reference": "hypothetical: a 10 solar-mass black hole placed where the Sun is"}
TARGET_KEYS = ("gaia_bh3", "gaia_bh1", "gaia_bh2", "sun")


class CatalogueMissing(FileNotFoundError):
    pass


@lru_cache(maxsize=1)
def load_catalogue(path: str = str(CATALOGUE)):
    if not Path(path).exists():
        raise CatalogueMissing(
            "The Gaia sky catalogue is not downloaded yet. Run: python tools/fetch_gaia_sky.py "
            f"(writes {Path(path).name}, about 36 MB).")
    data = np.load(path)
    metadata = json.loads(str(data["metadata"]))
    stars = {key: data[key].astype(np.float64) for key in data.files if key != "metadata"}
    return stars, metadata


def target_info(key: str, metadata: dict) -> dict:
    if key == "sun":
        return dict(SUN_TARGET)
    return dict(metadata["black_holes"][key])


def target_position_pc(target: dict) -> np.ndarray:
    return radec_unit(target["ra_deg"], target["dec_deg"]) * float(target["distance_pc"])


def teff_from_bp_rp(bp_rp):
    """Effective temperature from Gaia BP-RP (dwarf relation of Mucciarelli &
    Bellazzini 2020, theta = 5040/T); unknown colours are given 5800 K."""
    c = np.clip(np.nan_to_num(bp_rp, nan=0.82), -0.3, 3.5)
    theta = 0.4929 + 0.5092 * c - 0.0353 * c * c
    return np.clip(5040.0 / theta, 2600.0, 30000.0)


# Colour balance as in astrophotography, where the average star is set to
# white: the median catalogue star (BP-RP 1.14, about 4900 K; most bright Gaia
# stars are reddened K giants) is shown neutral, so the hotter half reads blue
# and the cooler half orange instead of nearly everything looking orange.
WHITE_POINT_K = 4900.0


def _planck_rgb(teff):
    wavelengths = np.array([610e-9, 550e-9, 465e-9])
    h, c, k = 6.62607015e-34, 2.99792458e8, 1.380649e-23
    t = np.asarray(teff, dtype=np.float64)[..., None]
    return 1.0 / (wavelengths ** 5 * (np.exp(h * c / (wavelengths * k * t)) - 1.0))


def rgb_from_teff(teff):
    """Display colour of a blackbody, normalised to unit luminance.

    Planck spectra sampled at three wavelengths near the sRGB primaries and
    balanced to WHITE_POINT_K: a display approximation, not colorimetry.
    """
    spectrum = _planck_rgb(teff) / _planck_rgb(WHITE_POINT_K)
    luminance = spectrum @ np.array([0.2126, 0.7152, 0.0722])
    return spectrum / luminance[..., None]


def stars_from(viewpoint_pc: np.ndarray, exclude_direction=None):
    """Directions, fluxes and colours of every catalogue star seen from a point.

    exclude_direction drops the star within 2 arcsec of that Earth-frame
    direction: the black hole's own companion (see build_sky).
    """
    stars, _ = load_catalogue()
    earth_dir = radec_unit(stars["ra"], stars["dec"])
    good = (stars["parallax"] > 0) & (stars["parallax"] > 5 * stars["parallax_error"])
    d_earth = np.where(good, 1000.0 / np.maximum(stars["parallax"], 1e-9), BACKGROUND_PC)
    offset = earth_dir * d_earth[:, None] - np.asarray(viewpoint_pc)[None, :]
    d_view = np.linalg.norm(offset, axis=1)
    keep = np.ones(len(d_view), dtype=bool)
    if exclude_direction is not None:
        keep &= earth_dir @ np.asarray(exclude_direction) < math.cos(math.radians(2.0 / 3600))
    # No catalogue star is closer than a parsec to any viewpoint used here
    # except a black hole's own companion, which is excluded; the floor only
    # stops a mis-measured parallax from producing an impossible magnitude.
    d_view = np.maximum(d_view, 1.0)
    magnitude = stars["phot_g_mean_mag"] + 5.0 * np.log10(d_view / d_earth)
    flux = 10.0 ** (-0.4 * (magnitude - REFERENCE_MAG))
    colour = rgb_from_teff(teff_from_bp_rp(stars["bp_rp"]))
    direction = offset / d_view[:, None]
    return direction[keep], flux[keep], colour[keep], {
        "stars": int(keep.sum()), "background_shell": int((~good & keep).sum()),
        "brightest_magnitude": float(magnitude[keep].min()),
    }


# ---- cube map --------------------------------------------------------------
# Faces +X, -X, +Y, -Y, +Z, -Z. For each face, the direction's major axis m,
# and the two in-face axes (u along a, v along b), with texel (0, 0) at uv = -1.
_FACES = [(0, 1.0, (2, -1.0), (1, -1.0)), (0, -1.0, (2, 1.0), (1, -1.0)),
          (1, 1.0, (0, 1.0), (2, 1.0)), (1, -1.0, (0, 1.0), (2, -1.0)),
          (2, 1.0, (0, 1.0), (1, -1.0)), (2, -1.0, (0, -1.0), (1, -1.0))]


def cube_coordinates(direction, xp=np):
    """Face index and continuous texel-space uv in [0, 1] for unit directions."""
    d = xp.asarray(direction)
    ax = xp.abs(d)
    major = xp.argmax(ax, axis=-1)
    sign = xp.take_along_axis(d, major[..., None], axis=-1)[..., 0] >= 0
    face = major * 2 + xp.where(sign, 0, 1)
    u = xp.zeros(d.shape[:-1], dtype=d.dtype)
    v = xp.zeros(d.shape[:-1], dtype=d.dtype)
    for index, (m, s, (ua, us), (va, vs)) in enumerate(_FACES):
        on = face == index
        denom = xp.abs(d[..., m])
        u = xp.where(on, (us * d[..., ua] / xp.maximum(denom, 1e-12) + 1) / 2, u)
        v = xp.where(on, (vs * d[..., va] / xp.maximum(denom, 1e-12) + 1) / 2, v)
    return face, xp.clip(u, 0, 1), xp.clip(v, 0, 1)


def splat_cube(direction, weight, colour, size: int) -> np.ndarray:
    """Radiance cube map (6, size, size, 3): each star's flux shared bilinearly
    among the four nearest texels and divided by the texel's solid angle, so
    total flux is conserved and brightness does not depend on resolution."""
    face, u, v = cube_coordinates(direction)
    x = u * size - 0.5
    y = v * size - 0.5
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    fx, fy = x - x0, y - y0
    cube = np.zeros((6 * size * size, 3), dtype=np.float64)
    for dx, dy, share in ((0, 0, (1 - fx) * (1 - fy)), (1, 0, fx * (1 - fy)),
                          (0, 1, (1 - fx) * fy), (1, 1, fx * fy)):
        xi = np.clip(x0 + dx, 0, size - 1)
        yi = np.clip(y0 + dy, 0, size - 1)
        index = (face * size + yi) * size + xi
        np.add.at(cube, index, (weight * share)[:, None] * colour)
    cube = cube.reshape(6, size, size, 3)
    # Solid angle of each texel: a cube face texel at (a, b) covers
    # dA / (1 + a^2 + b^2)^(3/2) steradians.
    centres = (np.arange(size) + 0.5) / size * 2 - 1
    a, b = np.meshgrid(centres, centres, indexing="xy")
    solid = (2.0 / size) ** 2 / (1 + a * a + b * b) ** 1.5
    return (cube / solid[None, :, :, None]).astype(np.float32)


GLOW_FAINTEST_RESOLVED_MAG = 8.0   # stars brighter than this stay out of the glow


def glow_map(direction, flux, colour, width: int = 720, blur_deg: float = 1.2) -> np.ndarray:
    """Diffuse starlight on an equirectangular map (height, width, 3).

    Built from the fainter stars only (bright ones are already sharp points)
    and blurred with a kernel that wraps in right ascension, so there are no
    seams. Radiance per steradian, like the star cube.
    """
    height = width // 2
    faint = flux < 10.0 ** (-0.4 * (GLOW_FAINTEST_RESOLVED_MAG - REFERENCE_MAG))
    d = direction[faint]
    ra = np.arctan2(d[:, 1], d[:, 0]) % (2 * np.pi)
    dec = np.arcsin(np.clip(d[:, 2], -1, 1))
    xi = np.minimum((ra / (2 * np.pi) * width).astype(np.int64), width - 1)
    yi = np.minimum(((np.pi / 2 - dec) / np.pi * height).astype(np.int64), height - 1)
    image = np.zeros((height * width, 3))
    np.add.at(image, yi * width + xi, flux[faint, None] * colour[faint])
    image = image.reshape(height, width, 3)
    rows = (np.arange(height) + 0.5) / height * np.pi
    solid = (2 * np.pi / width) * (np.pi / height) * np.sin(rows)
    image /= np.maximum(solid, 1e-12)[:, None, None]
    # Gaussian blur in Fourier space along RA (periodic) and a reflected pad in Dec.
    sigma_x = blur_deg / 360.0 * width
    sigma_y = blur_deg / 180.0 * height
    fx = np.fft.fftfreq(width)
    image = np.fft.ifft(np.fft.fft(image, axis=1) * np.exp(-2 * (np.pi * fx * sigma_x) ** 2)[None, :, None], axis=1).real
    padded = np.concatenate([image[::-1], image, image[::-1]], axis=0)
    fy = np.fft.fftfreq(3 * height)
    padded = np.fft.ifft(np.fft.fft(padded, axis=0) * np.exp(-2 * (np.pi * fy * sigma_y) ** 2)[:, None, None], axis=0).real
    # Near the poles a pixel is a sliver; average each row towards its mean so
    # the map does not pinch into a star there.
    image = padded[height:2 * height]
    weight = np.sin(rows)[:, None, None] ** 0.5
    image = weight * image + (1 - weight) * image.mean(axis=1, keepdims=True)
    return np.maximum(image, 0).astype(np.float32)


def sample_equirect(image, direction, xp=np):
    image = xp.asarray(image)
    height, width = image.shape[:2]
    d = xp.asarray(direction)
    ra = xp.arctan2(d[..., 1], d[..., 0]) % (2 * math.pi)
    dec = xp.arcsin(xp.clip(d[..., 2], -1, 1))
    x = ra / (2 * math.pi) * width - 0.5
    y = xp.clip((math.pi / 2 - dec) / math.pi * height - 0.5, 0, height - 1)
    x0 = xp.floor(x).astype(xp.int64)
    y0 = xp.floor(y).astype(xp.int64)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0w, x1w = x0 % width, (x0 + 1) % width
    y1 = xp.minimum(y0 + 1, height - 1)
    return ((1 - fx) * (1 - fy) * image[y0, x0w] + fx * (1 - fy) * image[y0, x1w]
            + (1 - fx) * fy * image[y1, x0w] + fx * fy * image[y1, x1w])


def sample_cube(cube, direction, xp=np):
    """Bilinear radiance lookup for unit directions."""
    cube = xp.asarray(cube)
    size = cube.shape[1]
    face, u, v = cube_coordinates(direction, xp=xp)
    x = u * size - 0.5
    y = v * size - 0.5
    x0 = xp.floor(x).astype(xp.int64)
    y0 = xp.floor(y).astype(xp.int64)
    fx, fy = (x - x0)[..., None], (y - y0)[..., None]
    x0c, x1c = xp.clip(x0, 0, size - 1), xp.clip(x0 + 1, 0, size - 1)
    y0c, y1c = xp.clip(y0, 0, size - 1), xp.clip(y0 + 1, 0, size - 1)
    return ((1 - fx) * (1 - fy) * cube[face, y0c, x0c] + fx * (1 - fy) * cube[face, y0c, x1c]
            + (1 - fx) * fy * cube[face, y1c, x0c] + fx * fy * cube[face, y1c, x1c])


def _viewpoint(target_key: str):
    """The target, its position and the companion direction to leave out."""
    _, metadata = load_catalogue()
    target = target_info(target_key, metadata)
    # The companion star orbits within tens of AU of the hole, far below
    # Gaia's depth precision: DR3's single-star parallax would put a bright
    # impostor tens to hundreds of parsecs along the line of sight. Leave it out.
    exclude = None if target_key == "sun" else radec_unit(target["ra_deg"], target["dec_deg"])
    return target, target_position_pc(target), exclude


def bright_stars(target_key: str, count: int = 60000):
    """Directions, fluxes and colours of the brightest stars seen from the target."""
    _, viewpoint, exclude = _viewpoint(target_key)
    direction, flux, colour, _ = stars_from(viewpoint, exclude_direction=exclude)
    keep = np.argsort(flux)[::-1][:count]
    return direction[keep], flux[keep], colour[keep]


def build_sky(target_key: str, face: int = 2048, glow_width: int = 720):
    """Fine star cube, coarse glow cube and a description of what went in."""
    _, metadata = load_catalogue()
    target, viewpoint, exclude = _viewpoint(target_key)
    direction, flux, colour, summary = stars_from(viewpoint, exclude_direction=exclude)
    fine = splat_cube(direction, flux, colour, face)
    glow = glow_map(direction, flux, colour, width=glow_width)
    summary.update({"target": target, "viewpoint_pc": viewpoint.tolist(), "cube_face_texels": face,
                    "catalogue": {k: metadata[k] for k in ("source", "credit", "selection", "retrieved")},
                    "companion_excluded": exclude is not None})
    return fine, glow, summary
