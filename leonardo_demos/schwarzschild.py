"""Exact light propagation around a non-rotating (Schwarzschild) black hole.

Geometric units throughout: G = c = M = 1, so the event horizon is at r = 2,
the photon sphere at r = 3 and the Schwarzschild radius r_s = 2.

A photon's path lies in one plane through the black hole. With u = 1/r and
impact parameter b, the orbit obeys

    (du/dphi)^2 = F(u) = 1/b^2 - u^2 + 2 u^3        (exact, not weak-field)

A camera held static at radius r_c sees a ray leaving at angle psi from the
direction to the hole with b = r_c sin(psi) / sqrt(1 - 2/r_c) (the local frame
of a static observer). Rays aimed inward with b below the critical value
3*sqrt(3) fall through the horizon: that set is the shadow. Every other ray
escapes, and the total angle phi it sweeps on the way out fixes the direction
of the star it sees. Stars are parsecs away and the hole is kilometres across,
so they are at infinity to within ~1e-13.

The escape angle depends on b alone for a given camera radius, so it is
tabulated once per radius by Gauss-Legendre quadrature and interpolated for
every pixel. `trace_path` integrates the same geodesic step by step, as an
independent check and for the 3-D view.
"""
from __future__ import annotations

import math

import numpy as np

B_CRITICAL = 3.0 * math.sqrt(3.0)
HORIZON = 2.0
PHOTON_SPHERE = 3.0
KM_PER_SCHWARZSCHILD_RADIUS_PER_MSUN = 2.95325  # r_s = 2GM/c^2 for one solar mass


def _nodes(per_interval: int):
    """Composite Gauss-Legendre nodes on [0, 1], log-spaced towards 0.

    Near the critical impact parameter the integrand has a logarithmic peak at
    s = 0 whose width shrinks without limit; geometric sub-intervals resolve it
    at every scale.
    """
    edges = np.concatenate([[0.0], np.geomspace(1e-10, 1.0, 21)])
    x, w = np.polynomial.legendre.leggauss(per_interval)
    nodes, weights = [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        nodes.append(lo + (x + 1) / 2 * (hi - lo))
        weights.append(w / 2 * (hi - lo))
    return np.concatenate(nodes), np.concatenate(weights)


def _sweep_to(u_end: np.ndarray, b: np.ndarray, per_interval: int = 24, turning=None) -> np.ndarray:
    """Angle swept between u = 0 and u = u_end: integral du / sqrt(F(u)).

    With u = u_end (1 - s^2) and F factored about its value at the endpoint,

        F(u) = F(u_end) + u_end s^2 G(u),  G = (u + u_end) - 2 (u^2 + u u_end + u_end^2),

    nothing is computed as a difference of nearly equal numbers. At a turning
    point F(u_end) = 0 exactly and the s cancels analytically, so the integrand
    stays finite; evaluating F directly there lost every digit to round-off.
    """
    s, w = _nodes(per_interval)
    u_end = np.asarray(u_end, dtype=np.float64)[:, None]
    b = np.asarray(b, dtype=np.float64)[:, None]
    u = u_end * (1 - s * s)
    g = (u + u_end) - 2.0 * (u * u + u * u_end + u_end * u_end)
    f_end = np.maximum(0.0, 1.0 / (b * b) - u_end * u_end + 2.0 * u_end ** 3)
    if turning is not None:
        f_end = np.where(np.asarray(turning)[:, None], 0.0, f_end)
    integrand = 2.0 * u_end * s / np.sqrt(np.maximum(f_end + u_end * s * s * g, 1e-300))
    return (integrand * w).sum(axis=1)


def turning_point(b: np.ndarray) -> np.ndarray:
    """Periapsis u0 = 1/r0 for b > B_CRITICAL: the root of F in (0, 1/3).

    F is monotonic there, so bisection is exact and never picks a wrong root.
    """
    b = np.asarray(b, dtype=np.float64)
    c = 1.0 / (b * b)
    lo, hi = np.zeros_like(b), np.full_like(b, 1.0 / 3.0)
    for _ in range(80):
        mid = (lo + hi) / 2
        positive = 2 * mid ** 3 - mid * mid + c > 0
        lo, hi = np.where(positive, mid, lo), np.where(positive, hi, mid)
    return (lo + hi) / 2


def impact_parameter(r_camera: float, psi):
    """b for a ray leaving a static camera at angle psi from the inward radial."""
    return r_camera * np.sin(psi) / math.sqrt(1.0 - 2.0 / r_camera)


def shadow_half_angle(r_camera: float) -> float:
    """Angular radius of the shadow for a static camera (radians)."""
    return math.asin(min(1.0, B_CRITICAL * math.sqrt(1.0 - 2.0 / r_camera) / r_camera))


class EscapeTable:
    """Escape angle as a function of ray angle psi, for one camera radius.

    Tabulated in psi rather than b: near a sideways ray b(psi) has zero slope,
    so phi(b) is infinitely steep there and a b-table interpolates badly, while
    phi(psi) stays smooth. Near the shadow edge phi grows like -ln(psi - psi_c),
    so that side is sampled and interpolated logarithmically.
    """

    def __init__(self, r_camera: float, samples: int = 6000):
        if r_camera <= PHOTON_SPHERE:
            raise ValueError("the camera must be outside the photon sphere (r > 3M = 1.5 r_s)")
        self.r = float(r_camera)
        self.u = 1.0 / self.r
        self.scale = self.r / math.sqrt(1.0 - 2.0 / self.r)       # b = scale * sin(psi)
        self.psi_edge = shadow_half_angle(self.r)                  # inward rays below this are captured
        # Outward rays: psi from sideways (pi/2) to straight out (pi).
        self.psi_out = np.linspace(math.pi / 2, math.pi, samples)
        b_out = self.scale * np.sin(self.psi_out)
        self.phi_out = np.where(b_out > 1e-12,
                                _sweep_to(np.full(samples, self.u), np.maximum(b_out, 1e-12)), 0.0)
        # Inward rays that escape: psi between the shadow edge and sideways.
        span = math.pi / 2 - self.psi_edge
        gap = np.unique(np.concatenate([np.geomspace(1e-11, span, samples // 2),
                                        np.linspace(span / samples, span, samples // 2)]))
        self.log_gap = np.log(gap)
        b_in = np.maximum(self.scale * np.sin(self.psi_edge + gap), B_CRITICAL * (1 + 1e-15))
        u0 = turning_point(b_in)
        self.phi_in = (2.0 * _sweep_to(u0, b_in, turning=np.ones(len(b_in), dtype=bool))
                       - _sweep_to(np.minimum(np.full(len(b_in), self.u), u0), b_in))

    def escape_angle(self, psi, xp=np):
        """Swept angle for rays at angle psi from the inward radial; NaN if captured."""
        psi = xp.asarray(psi)
        inward = psi < math.pi / 2
        captured = psi <= self.psi_edge
        phi_out = xp.interp(psi, xp.asarray(self.psi_out), xp.asarray(self.phi_out))
        gap = xp.maximum(psi - self.psi_edge, 1e-11)
        phi_in = xp.interp(xp.log(gap), xp.asarray(self.log_gap), xp.asarray(self.phi_in))
        phi = xp.where(inward, phi_in, phi_out)
        return xp.where(captured, xp.nan, phi), captured


def lens_directions(directions, camera_axis, r_camera: float, table: EscapeTable | None = None, xp=np):
    """Map camera ray directions to the sky directions they see.

    directions: (..., 3) unit vectors in the static camera's local frame,
    expressed in the same coordinates as camera_axis (unit vector from the hole
    to the camera). Returns (sky_directions, captured).
    """
    table = table or EscapeTable(r_camera)
    d = xp.asarray(directions)
    c = xp.asarray(camera_axis, dtype=d.dtype)
    along = (d * c).sum(axis=-1)
    psi = xp.arccos(xp.clip(-along, -1.0, 1.0))      # angle from the direction to the hole
    phi, captured = table.escape_angle(psi, xp=xp)
    perp = d - along[..., None] * c
    norm = xp.linalg.norm(perp, axis=-1, keepdims=True)
    e2 = xp.where(norm > 1e-12, perp / xp.maximum(norm, 1e-12), 0.0)
    phi = xp.nan_to_num(phi)
    sky = xp.cos(phi)[..., None] * c + xp.sin(phi)[..., None] * e2
    return sky, captured


def trace_path(r_camera: float, psi: float, step: float = 2e-3, r_escape: float = 400.0, max_turns: float = 4.0):
    """Integrate one photon from the camera: returns (phi, r) samples and fate.

    Fourth-order Runge-Kutta on u'' = 3u^2 - u, the second-order form of the
    exact orbit equation. Independent of the quadrature in EscapeTable.
    """
    u = 1.0 / r_camera
    if abs(math.sin(psi)) < 1e-12:
        inward = math.cos(psi) > 0
        r_end = HORIZON if inward else r_escape
        return np.array([0.0, 0.0]), np.array([r_camera, r_end]), ("captured" if inward else "escaped")
    v = math.sqrt(1.0 - 2.0 * u) * u * math.cos(psi) / math.sin(psi)   # du/dphi
    phi = 0.0
    phis, radii = [0.0], [r_camera]

    def rhs(uu, vv):
        return vv, 3.0 * uu * uu - uu

    while phi < max_turns * 2 * math.pi:
        k1u, k1v = rhs(u, v)
        k2u, k2v = rhs(u + step / 2 * k1u, v + step / 2 * k1v)
        k3u, k3v = rhs(u + step / 2 * k2u, v + step / 2 * k2v)
        k4u, k4v = rhs(u + step * k3u, v + step * k3v)
        u += step / 6 * (k1u + 2 * k2u + 2 * k3u + k4u)
        v += step / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        phi += step
        if u >= 1.0 / HORIZON:
            phis.append(phi); radii.append(HORIZON)
            return np.array(phis), np.array(radii), "captured"
        if u <= 1.0 / r_escape:
            phis.append(phi); radii.append(r_escape)
            return np.array(phis), np.array(radii), "escaped"
        if len(phis) % 8 == 0:
            phis.append(phi); radii.append(1.0 / u)
    return np.array(phis), np.array(radii), "orbiting"


def trace_fan(r_camera: float, psis, step: float = 4e-3, r_max: float | None = None,
              max_phi: float = 4 * math.pi, keep_every: int = 6):
    """Vectorised form of trace_path for a fan of rays in one plane.

    Returns a list of (phi, r, fate) per ray; rays stop at the horizon, at
    r_max, or after max_phi radians.
    """
    psis = np.asarray(psis, dtype=np.float64)
    r_max = r_max or 3.0 * r_camera
    u = np.full(len(psis), 1.0 / r_camera)
    sin = np.sin(psis)
    straight = np.abs(sin) < 1e-9
    v = np.where(straight, 0.0, math.sqrt(1.0 - 2.0 / r_camera) / r_camera * np.cos(psis) / np.where(straight, 1.0, sin))
    alive = ~straight
    fate = np.array(["orbiting"] * len(psis), dtype=object)
    fate[straight] = np.where(np.cos(psis[straight]) > 0, "captured", "escaped")
    phis, radii = [np.zeros(len(psis))], [np.full(len(psis), float(r_camera))]
    end = np.full(len(psis), -1)
    steps = int(max_phi / step)

    def rhs(uu, vv):
        return vv, 3.0 * uu * uu - uu

    for n in range(1, steps + 1):
        if not alive.any():
            break
        k1u, k1v = rhs(u, v)
        k2u, k2v = rhs(u + step / 2 * k1u, v + step / 2 * k1v)
        k3u, k3v = rhs(u + step / 2 * k2u, v + step / 2 * k2v)
        k4u, k4v = rhs(u + step * k3u, v + step * k3v)
        u = np.where(alive, u + step / 6 * (k1u + 2 * k2u + 2 * k3u + k4u), u)
        v = np.where(alive, v + step / 6 * (k1v + 2 * k2v + 2 * k3v + k4v), v)
        hit = alive & (u >= 1.0 / HORIZON)
        out = alive & (u <= 1.0 / r_max)
        fate[hit], fate[out] = "captured", "escaped"
        u = np.where(hit, 1.0 / HORIZON, np.where(out, 1.0 / r_max, u))
        stopped = hit | out
        if n % keep_every == 0 or stopped.any():
            phis.append(np.full(len(psis), n * step))
            radii.append(1.0 / u)
            end = np.where(stopped, len(phis) - 1, end)
        alive &= ~stopped
    phis, radii = np.array(phis), np.array(radii)
    rays = []
    for i in range(len(psis)):
        if straight[i]:
            r_end = HORIZON if fate[i] == "captured" else r_max
            rays.append((np.array([0.0, 0.0]), np.array([r_camera, r_end]), fate[i]))
            continue
        last = end[i] if end[i] >= 0 else len(phis) - 1
        rays.append((phis[:last + 1, i], radii[:last + 1, i], fate[i]))
    return rays
