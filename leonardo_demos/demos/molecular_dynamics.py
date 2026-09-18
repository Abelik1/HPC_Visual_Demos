"""Molecular Machine: coarse-grained molecular dynamics in two modes.

**Fold** - "fold your own protein". A chain of beads, each one of four kinds:
water-avoiding (H), water-loving (P), positive (+) and negative (-). It is the
classic HP lattice idea taken off the lattice: harmonic bonds, a bending
stiffness, excluded volume between every pair, an H-H attraction scaled by how
strongly water pushes oily groups together, and screened (Debye-Huckel)
attraction/repulsion between charges. A Langevin thermostat supplies the
thermal kicks of the surrounding water. Visitors write the sequence
themselves (``_chain``), or pick a preset; the chain buries its H beads in a
core, zips opposite charges together, or stays floppy, depending on what they
wrote.

**Shuttle** - a molecular machine of the kind the 2016 Chemistry Nobel was
awarded for: a ring threaded on an axle (a rotaxane) with two binding
stations and bulky stoppers at each end. Only one station is sticky at a time;
a switch (in the lab: light, acid or an electrode) flips which one. The ring is
never pushed. It diffuses along the axle on thermal kicks alone and is caught
when it reaches the sticky station, so flipping the switch makes it shuttle.

Both are reduced-unit models (bead diameter 1, energies in units of the
thermal energy at 310 K / 0.5): illustrative, not a force field.
"""
from __future__ import annotations

import json
import math
import re

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from ..backend import to_numpy
from ..base import Demo
from ..render import add_progress, add_title

# H, P, +, - : amber oil, cyan water-loving, blue positive, rose negative.
FOLD_PALETTE = [(255, 170, 64), (96, 214, 255), (120, 132, 255), (255, 92, 136)]
FOLD_LABELS = ["water-avoiding (H)", "water-loving (P)", "positive (+)", "negative (-)"]
CHAIN_LETTERS = {"H": 0, "P": 1, "+": 2, "-": 3}
CHAIN_PATTERN = re.compile(r"^[HP+\-]{6,400}$")
# Presets for the "Sequence" control, built to whatever chain length the
# quality preset asks for.
PRESET_NAMES = ["Oily core", "Hairpin", "Charge zipper", "Soluble"]


def preset_text(index: int, length: int) -> str:
    n = max(6, int(length))
    index = int(index) % len(PRESET_NAMES)
    if index == 0:      # oily beads spaced so they can all meet in the middle
        motif = "PHPPHHPHPPHPHHPPHPHH"
        return (motif * (n // len(motif) + 1))[:n]
    if index == 1:      # two alternating strands joined by a water-loving turn
        turn = 6 if n >= 20 else 2
        strand = (n - turn) // 2
        return ("HP" * n)[:strand] + "P" * (n - 2 * strand) + ("PH" * n)[:strand][::-1]
    if index == 2:      # + near one end, - near the other, oil in between
        end = max(2, n // 4)
        middle = ("PHHPH" * n)[:n - 2 * end]
        return ("P+" * n)[:end] + middle + ("-P" * n)[:end]
    return "".join("H" if i % 9 == 4 else "P" for i in range(n))   # soluble

# Shuttle palette: axle, stopper, station (idle), station (sticky), ring.
SHUTTLE_PALETTE = [(120, 140, 168), (182, 196, 216), (70, 120, 96), (120, 255, 170), (255, 196, 70)]
SHUTTLE_LABELS = ["axle", "stopper", "station", "sticky station", "ring"]
# Drawn bead size per type; the stoppers are fat in the physics too.
SHUTTLE_SIZES = [1.0, 2.2, 1.0, 1.0, 1.0]

DT = 0.005
# The shuttle's stiffest term (ring bonds, k=120) is stable at twice the step.
SHUTTLE_DT = 0.01
MAX_KERNEL_BEADS = 1024

# The whole fold, fused: one thread block holds the chain, one thread per bead,
# and runs many Langevin (BAOAB) steps per launch with __syncthreads between
# them. Each thread sums the non-bonded force from every other bead, then its
# bond and bending terms, then updates itself. At a few hundred beads a step is
# a few microseconds; the unfused CuPy version spent milliseconds launching
# ~50 tiny kernels per step. Same force field as fold_forces() below.
FOLD_KERNEL = r"""
__device__ __forceinline__ unsigned int mix(unsigned int x){
  x ^= x >> 16; x *= 0x7feb352dU; x ^= x >> 15; x *= 0x846ca68bU; x ^= x >> 16; return x;}
__device__ __forceinline__ float uniform01(unsigned int a, unsigned int b, unsigned int c){
  return (mix(a ^ mix(b ^ mix(c))) >> 8) * (1.0f / 16777216.0f) + (0.5f / 16777216.0f);}
__device__ __forceinline__ void bend_terms(float3 a, float3 b, float3 c, float k, float c0,
                                           float3* fa, float3* fb, float3* fc){
  float3 b1 = make_float3(b.x-a.x, b.y-a.y, b.z-a.z), b2 = make_float3(c.x-b.x, c.y-b.y, c.z-b.z);
  float l1 = sqrtf(b1.x*b1.x + b1.y*b1.y + b1.z*b1.z + 1e-9f), l2 = sqrtf(b2.x*b2.x + b2.y*b2.y + b2.z*b2.z + 1e-9f);
  float3 u1 = make_float3(b1.x/l1, b1.y/l1, b1.z/l1), u2 = make_float3(b2.x/l2, b2.y/l2, b2.z/l2);
  float cs = u1.x*u2.x + u1.y*u2.y + u1.z*u2.z, g = 2.0f*k*(cs - c0);
  float3 d1 = make_float3((u2.x-cs*u1.x)/l1, (u2.y-cs*u1.y)/l1, (u2.z-cs*u1.z)/l1);
  float3 d2 = make_float3((u1.x-cs*u2.x)/l2, (u1.y-cs*u2.y)/l2, (u1.z-cs*u2.z)/l2);
  *fa = make_float3(g*d1.x, g*d1.y, g*d1.z);
  *fb = make_float3(-g*(d1.x-d2.x), -g*(d1.y-d2.y), -g*(d1.z-d2.z));
  *fc = make_float3(-g*d2.x, -g*d2.y, -g*d2.z);}

extern "C" __global__ void fold_steps(float* pos, float* vel, const float* eps, const float* qq,
    const int n, const int steps, const float dt, const float c1, const float c2,
    const float bend_k, const float bend_c0, const float debye, const int use_charge,
    const unsigned int seed, const unsigned int step0){
  extern __shared__ float3 sp[];
  const int i = threadIdx.x;
  const bool live = i < n;
  float3 x = make_float3(0.f, 0.f, 0.f), v = x, f = x;
  if (live){ x = make_float3(pos[3*i], pos[3*i+1], pos[3*i+2]); v = make_float3(vel[3*i], vel[3*i+1], vel[3*i+2]); }
  for (int s = -1; s < steps; ++s){
    if (s >= 0 && live){
      v.x += 0.5f*dt*f.x; v.y += 0.5f*dt*f.y; v.z += 0.5f*dt*f.z;
      x.x += 0.5f*dt*v.x; x.y += 0.5f*dt*v.y; x.z += 0.5f*dt*v.z;
      unsigned int t = step0 + (unsigned int)s;
      float u1 = uniform01(seed, t, 6*i), u2 = uniform01(seed, t, 6*i+1), u3 = uniform01(seed, t, 6*i+2), u4 = uniform01(seed, t, 6*i+3);
      float r1 = sqrtf(-2.f*logf(u1)), r2 = sqrtf(-2.f*logf(u3));
      v.x = c1*v.x + c2*r1*cosf(6.2831853f*u2);
      v.y = c1*v.y + c2*r1*sinf(6.2831853f*u2);
      v.z = c1*v.z + c2*r2*cosf(6.2831853f*u4);
      x.x += 0.5f*dt*v.x; x.y += 0.5f*dt*v.y; x.z += 0.5f*dt*v.z;
    }
    __syncthreads();
    if (live) sp[i] = x;
    __syncthreads();
    if (live){
      float fx = 0.f, fy = 0.f, fz = 0.f;
      for (int j = 0; j < n; ++j){
        if (j - i <= 2 && i - j <= 2) continue;
        float dx = x.x - sp[j].x, dy = x.y - sp[j].y, dz = x.z - sp[j].z;
        float r2 = fmaxf(dx*dx + dy*dy + dz*dz, 0.64f), inv2 = 1.f/r2, s6 = inv2*inv2*inv2;
        float lj = 24.f*(2.f*s6*s6 - s6)*inv2, coeff = 0.f;
        if (r2 < 6.25f) coeff += eps[i*n+j]*lj;
        if (r2 < 1.2599f) coeff += lj;
        if (use_charge){ float q = qq[i*n+j]; if (q != 0.f){ float r = sqrtf(r2); coeff += q*expf(-r/debye)*(1.f/r + 1.f/debye)*inv2; } }
        fx += coeff*dx; fy += coeff*dy; fz += coeff*dz;
      }
      if (i + 1 < n){ float3 o = sp[i+1]; float dx = o.x-x.x, dy = o.y-x.y, dz = o.z-x.z; float l = sqrtf(dx*dx+dy*dy+dz*dz+1e-9f), k = 180.f*(l-1.f)/l; fx += k*dx; fy += k*dy; fz += k*dz; }
      if (i > 0){ float3 o = sp[i-1]; float dx = x.x-o.x, dy = x.y-o.y, dz = x.z-o.z; float l = sqrtf(dx*dx+dy*dy+dz*dz+1e-9f), k = 180.f*(l-1.f)/l; fx -= k*dx; fy -= k*dy; fz -= k*dz; }
      float3 fa, fb, fc;
      if (i > 0 && i + 1 < n){ bend_terms(sp[i-1], x, sp[i+1], bend_k, bend_c0, &fa, &fb, &fc); fx += fb.x; fy += fb.y; fz += fb.z; }
      if (i + 2 < n){ bend_terms(x, sp[i+1], sp[i+2], bend_k, bend_c0, &fa, &fb, &fc); fx += fa.x; fy += fa.y; fz += fa.z; }
      if (i >= 2){ bend_terms(sp[i-2], sp[i-1], x, bend_k, bend_c0, &fa, &fb, &fc); fx += fc.x; fy += fc.y; fz += fc.z; }
      fx -= 0.002f*x.x; fy -= 0.002f*x.y; fz -= 0.002f*x.z;
      f = make_float3(fminf(fmaxf(fx,-400.f),400.f), fminf(fmaxf(fy,-400.f),400.f), fminf(fmaxf(fz,-400.f),400.f));
      if (s >= 0){ v.x += 0.5f*dt*f.x; v.y += 0.5f*dt*f.y; v.z += 0.5f*dt*f.z; }
    }
  }
  if (live){ pos[3*i] = x.x; pos[3*i+1] = x.y; pos[3*i+2] = x.z; vel[3*i] = v.x; vel[3*i+1] = v.y; vel[3*i+2] = v.z; }
}
"""


def parse_chain(text: str) -> list[int]:
    text = str(text).strip().upper()
    if not CHAIN_PATTERN.match(text):
        raise ValueError("a chain is 6-400 beads written with H, P, + and -")
    return [CHAIN_LETTERS[c] for c in text]


def preset_chain(index: int, length: int) -> list[int]:
    return [CHAIN_LETTERS[c] for c in preset_text(index, length)]


class MolecularDynamicsDemo(Demo):
    """Coarse-grained Langevin dynamics with all-pairs non-bonded forces."""

    id = "molecular_dynamics"
    title = "Molecular Machine"
    methods = ("fold", "shuttle")
    default_method = "fold"
    method_labels = {
        "fold": "Fold your own protein",
        "shuttle": "Molecular shuttle (a machine)",
    }
    method_descriptions = {
        "fold": "A chain of water-avoiding, water-loving and charged beads folds under thermal kicks; every bead feels every other one.",
        "shuttle": "A ring threaded on an axle hops between two binding stations when a switch changes which one is sticky.",
    }
    timing_methods = {}   # stages are timed inline with ctx.stage

    # ------------------------------------------------------------ common --
    @property
    def xp(self):
        return getattr(self, "_xp", None) or self.ctx.xp

    @property
    def mode(self) -> str:
        method = getattr(self.ctx, "method", self.default_method)
        return method if method in self.methods else self.default_method

    def kT(self) -> float:
        return 0.5 * float(self.ctx.params.get("temperature", 310.0)) / 310.0

    def budget(self) -> int:
        if "total_steps" in self.settings:
            return max(1, int(self.settings["total_steps"]))
        return max(1, int(self.settings.get("steps_per_frame", 200)) * self.ctx.frames)

    def _rng(self, seed):
        return self.xp.random.default_rng(seed)

    def langevin(self, pos, vel, force_fn, steps, kT, gamma, dt=DT):
        """BAOAB Langevin integration (unit masses)."""
        xp = self.xp
        c1 = math.exp(-gamma * dt)
        c2 = math.sqrt(max(0.0, 1.0 - c1 * c1) * kT)
        force = force_fn(pos)
        for _ in range(steps):
            vel += 0.5 * dt * force
            pos += 0.5 * dt * vel
            vel *= c1
            vel += c2 * self.rng.standard_normal(vel.shape, dtype=xp.float32)
            pos += 0.5 * dt * vel
            force = force_fn(pos)
            vel += 0.5 * dt * force
        return pos, vel

    @staticmethod
    def _pair_terms(xp, pos):
        delta = pos[:, None, :] - pos[None, :, :]
        r2 = xp.sum(delta * delta, axis=-1)
        return delta, r2

    @staticmethod
    def _bend_forces(xp, pos, force, k, c0, ring=False):
        """E = k (cos(theta) - c0)^2 on every consecutive bond pair."""
        if ring:
            prev, nxt = xp.roll(pos, 1, axis=0), xp.roll(pos, -1, axis=0)
            b1, b2 = pos - prev, nxt - pos
        else:
            b1, b2 = pos[1:-1] - pos[:-2], pos[2:] - pos[1:-1]
        l1 = xp.sqrt(xp.sum(b1 * b1, axis=1, keepdims=True) + 1e-9)
        l2 = xp.sqrt(xp.sum(b2 * b2, axis=1, keepdims=True) + 1e-9)
        u1, u2 = b1 / l1, b2 / l2
        c = xp.sum(u1 * u2, axis=1, keepdims=True)
        g = 2.0 * k * (c - c0)
        d1 = (u2 - c * u1) / l1
        d2 = (u1 - c * u2) / l2
        f_prev, f_mid, f_next = g * d1, -g * (d1 - d2), -g * d2
        if ring:
            force += f_mid
            force += xp.roll(f_prev, -1, axis=0)
            force += xp.roll(f_next, 1, axis=0)
        else:
            force[:-2] += f_prev
            force[1:-1] += f_mid
            force[2:] += f_next
        return force

    @staticmethod
    def _bond_forces(xp, pos, force, k, r0, ring=False):
        nxt = xp.roll(pos, -1, axis=0) if ring else pos[1:]
        cur = pos if ring else pos[:-1]
        b = nxt - cur
        length = xp.sqrt(xp.sum(b * b, axis=1, keepdims=True) + 1e-9)
        f = k * (length - r0) * b / length
        if ring:
            force += f
            force -= xp.roll(f, 1, axis=0)
        else:
            force[:-1] += f
            force[1:] -= f
        return force

    # -------------------------------------------------------------- fold --
    def fold_chain(self) -> list[int]:
        custom = self.ctx.params.get("_chain")
        if custom:
            return parse_chain(custom)
        return preset_chain(int(self.ctx.params.get("sequence", 0)), int(self.settings.get("particles", 40)))

    def fold_initial(self, n, seed=7):
        """A loose, self-avoiding random coil with unit bonds."""
        rng = np.random.default_rng(seed)
        pos = np.zeros((n, 3))
        direction = np.array([1.0, 0.0, 0.0])
        for i in range(1, n):
            for _ in range(60):
                trial = direction + rng.normal(0, 0.55, 3)
                trial /= np.linalg.norm(trial)
                candidate = pos[i - 1] + trial
                if i < 3 or np.min(np.linalg.norm(pos[:i - 1] - candidate, axis=1)) > 1.05:
                    break
            direction = trial
            pos[i] = candidate
        pos -= pos.mean(axis=0)
        return pos.astype(np.float32)

    def fold_setup(self, types):
        xp = self.xp
        n = len(types)
        t = np.asarray(types)
        attraction = float(self.ctx.params.get("attraction", 1.0))
        water = float(self.ctx.params.get("solvent", 0.65))
        salt = float(self.ctx.params.get("salt", 0.3))
        hydro = (t[:, None] == 0) & (t[None, :] == 0)
        # H-H: the hydrophobic effect; everything else sticks only faintly.
        eps = np.where(hydro, attraction * (0.35 + 1.45 * water), 0.12 * attraction)
        charge = np.where(t == 2, 1.0, np.where(t == 3, -1.0, 0.0))
        qq = charge[:, None] * charge[None, :]
        ii, jj = np.arange(n)[:, None], np.arange(n)[None, :]
        nonbond = np.abs(ii - jj) > 2
        self.fold = {
            "eps": xp.asarray((eps * nonbond).astype(np.float32)),
            "qq": xp.asarray((2.4 * qq * nonbond).astype(np.float32)),
            "wca": xp.asarray(nonbond.astype(np.float32)),
            "debye": 0.6 + 3.4 * (1.0 - salt),
            "bend": float(self.settings.get("bend", 2.5)),
            "any_charge": bool(np.any(charge)),
        }

    def fold_forces(self, pos):
        xp, f = self.xp, self.fold
        delta, r2 = self._pair_terms(xp, pos)
        r2 = xp.maximum(r2, 0.64)
        inv2 = 1.0 / r2
        s6 = inv2 * inv2 * inv2
        # Attractive LJ for the pair's epsilon (cut at 2.5), plus a purely
        # repulsive WCA core for every non-bonded pair.
        lj = xp.where(r2 < 6.25, 24.0 * f["eps"] * (2.0 * s6 * s6 - s6) * inv2, 0.0)
        wca = xp.where(r2 < 1.2599, 24.0 * (2.0 * s6 * s6 - s6) * inv2, 0.0) * f["wca"]
        coeff = lj + wca
        if f["any_charge"]:
            r = xp.sqrt(r2)
            coeff += f["qq"] * xp.exp(-r / f["debye"]) * (1.0 / r + 1.0 / f["debye"]) * inv2
        force = xp.sum(coeff[..., None] * delta, axis=1)
        force = self._bond_forces(xp, pos, force, 180.0, 1.0)
        force = self._bend_forces(xp, pos, force, f["bend"], 0.35)
        force -= 0.002 * pos          # a very weak tether keeps it on camera
        return xp.clip(force, -400.0, 400.0)

    def fold_kernel_steps(self, pos, vel, steps, kT, done):
        """Advance the fold `steps` Langevin steps in fused CUDA launches."""
        xp, f = self.xp, self.fold
        n = pos.shape[0]
        if not hasattr(self, "_fold_kernel"):
            self._fold_kernel = xp.RawKernel(FOLD_KERNEL, "fold_steps")
            threads = 32 * ((n + 31) // 32)
            self._fold_launch = threads
            self.ctx.record_kernel("molecular_fold_fused", {
                "threads_per_block": threads, "blocks": 1, "beads": n,
                "steps_per_launch": "up to 2000", "shared_memory_bytes": 12 * n,
                "design": "one block holds the chain; all-pairs forces and BAOAB update fused, many steps per launch"})
        c1 = math.exp(-DT)
        c2 = math.sqrt(max(0.0, 1.0 - c1 * c1) * kT)
        seed = int(self.ctx.params.get("seed", 11)) * 2654435761 % (1 << 32)
        remaining, t = steps, done
        while remaining > 0:
            chunk = min(2000, remaining)
            self._fold_kernel((1,), (self._fold_launch,),
                              (pos, vel, f["eps"], f["qq"], np.int32(n), np.int32(chunk), np.float32(DT),
                               np.float32(c1), np.float32(c2), np.float32(f["bend"]), np.float32(0.35),
                               np.float32(f["debye"]), np.int32(1 if f["any_charge"] else 0),
                               np.uint32(seed), np.uint32(t % (1 << 32))),
                              shared_mem=12 * n)
            remaining -= chunk
            t += chunk
        return pos, vel

    def fold_diagnostics(self, pos, types):
        p = np.asarray(pos, dtype=np.float64)
        t = np.asarray(types)
        rg = float(np.sqrt(np.mean(np.sum((p - p.mean(axis=0)) ** 2, axis=1))))
        d = np.sqrt(np.sum((p[:, None] - p[None]) ** 2, axis=-1))
        n = len(p)
        far = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) > 2
        close = (d < 1.6) & far
        hh = int(np.sum(close & (t[:, None] == 0) & (t[None, :] == 0)) // 2)
        salt = int(np.sum(close & (((t[:, None] == 2) & (t[None, :] == 3)))))
        h = t == 0
        buried = float(np.mean(np.sum(close[h], axis=1) >= 3)) if h.any() else 0.0
        compact = 0.62 * n ** (1 / 3)          # Rg of a tightly packed globule
        return {"rg": rg, "hh": hh, "salt": salt, "buried": buried, "compactness": compact / max(rg, 1e-6)}

    def run_fold(self):
        types = self.fold_chain()
        n = len(types)
        total = self.budget()
        kT = self.kT()
        self.fold_setup(types)
        xp = self.xp
        pos = xp.asarray(self.fold_initial(n))
        vel = xp.asarray(np.random.default_rng(3).normal(0, math.sqrt(kT), (n, 3)).astype(np.float32))
        pos, vel = xp.ascontiguousarray(pos), xp.ascontiguousarray(vel)
        text = "".join("HP+-"[t] for t in types)
        self.ctx.write_meta({"chain": text, "chain_length": n,
                             "physics": {"model": "off-lattice HP chain with charges, Langevin (BAOAB) dynamics",
                                         "units": "reduced: bead diameter 1, kT(310 K) = 0.5",
                                         "time_step": DT, "pair_interactions": n * (n - 1) // 2}})
        self._begin_3d("fold", FOLD_PALETTE, FOLD_LABELS)
        done = 0
        best = None
        for i in range(self.ctx.frames):
            target = int(round(total * (i + 1) / self.ctx.frames))
            with self.ctx.stage("simulation"):
                if self.ctx.on_gpu and n <= MAX_KERNEL_BEADS:
                    pos, vel = self.fold_kernel_steps(pos, vel, max(1, target - done), kT, done)
                else:
                    pos, vel = self.langevin(pos, vel, self.fold_forces, max(1, target - done), kT, 1.0)
            done = target
            pn = to_numpy(pos)
            diag = self.fold_diagnostics(pn, types)
            best = diag if best is None or diag["compactness"] > best["compactness"] else best
            with self.ctx.stage("render"):
                image = self.render(pn, types, FOLD_PALETTE, bonds=None, angle=0.01 * i,
                                    subtitle=f"{n} beads · every bead feels every other bead · {self.ctx.backend_name}",
                                    progress=done / total, left="UNFOLDED CHAIN", right="FOLDED?")
                self.ctx.save_frame(image, self.ctx.frame_path(i))
                self._save_3d(i, pn, types, bonds=None, step=done)
            self.ctx.write_status(i, f"step {done:,} · Rg {diag['rg']:.2f}", {
                "sequence": text if len(text) <= 40 else text[:37] + "…",
                "temperature": f"{float(self.ctx.params.get('temperature', 310)):.0f} K",
                "radius of gyration": f"{diag['rg']:.2f}",
                "oily contacts": f"{diag['hh']}",
                "buried oil": f"{100 * diag['buried']:.0f}%",
                "salt bridges": f"{diag['salt']}",
                "time steps": f"{done:,}",
                "pair evaluations": f"{done * n * (n - 1) // 2:,}"})
        self.ctx.write_meta({"summary": {"mode": "fold", "chain": text, "radius_of_gyration": round(diag["rg"], 3),
                                         "oily_contacts": diag["hh"], "buried_oil": round(diag["buried"], 3),
                                         "salt_bridges": diag["salt"]}})
        self.ctx.finish()

    # ----------------------------------------------------------- shuttle --
    def shuttle_build(self):
        """Axle with stoppers and two stations; a ring threaded around it."""
        axle_beads = int(self.settings.get("axle_beads", 13))
        half = (axle_beads - 1) / 2
        axle = np.zeros((axle_beads, 3), np.float32)
        axle[:, 0] = np.arange(axle_beads) - half
        separation = float(self.settings.get("station_gap", 5.0))
        kinds = np.zeros(axle_beads, int)
        kinds[[0, -1]] = 1
        station_a = np.where(np.abs(axle[:, 0] + separation / 2) < 1.1)[0]
        station_b = np.where(np.abs(axle[:, 0] - separation / 2) < 1.1)[0]
        ring_n = int(self.settings.get("ring_beads", 12))
        radius = 1.0 / (2 * math.sin(math.pi / ring_n))       # unit bonds
        theta = np.arange(ring_n) * 2 * math.pi / ring_n
        ring = np.stack([np.full(ring_n, -separation / 2), radius * np.cos(theta), radius * np.sin(theta)], 1)
        return axle, kinds, station_a, station_b, ring.astype(np.float32), radius

    def shuttle_forces_factory(self, axle, station_a, station_b, radius):
        xp = self.xp
        axle_x = xp.asarray(axle)
        n_axle = len(axle)
        stopper = np.zeros(n_axle, np.float32)
        stopper[[0, -1]] = 1
        stopper = xp.asarray(stopper)
        in_a = np.zeros(n_axle, np.float32)
        in_a[station_a] = 1
        in_b = np.zeros(n_axle, np.float32)
        in_b[station_b] = 1
        in_a, in_b = xp.asarray(in_a), xp.asarray(in_b)
        # Per-pair well depths, normalised so a whole station holds the ring by
        # roughly `drive` x 4 kT when sticky and under 1 kT when idle: strong
        # enough to catch it, weak enough that it lets go after a flip.
        pairs = len(station_a) * int(self.settings.get("ring_beads", 12))
        drive = float(self.ctx.params.get("drive", 3.0)) * 4.8 * self.kT() / pairs
        idle = 0.08 * drive
        sigma_bind = radius / 2 ** (1 / 6)        # LJ minimum at the ring radius
        state = {"sticky": 0}

        def forces(ring):
            eps = in_a * (drive if state["sticky"] == 0 else idle) + in_b * (drive if state["sticky"] == 1 else idle)
            delta = ring[:, None, :] - axle_x[None, :, :]
            r2 = xp.maximum(xp.sum(delta * delta, axis=-1), 0.25)
            inv2 = 1.0 / r2
            # Station binding: LJ well at the ring radius, cut at 1.6 radii.
            s2 = sigma_bind * sigma_bind * inv2
            s6 = s2 * s2 * s2
            bind = xp.where(r2 < (1.6 * radius) ** 2, 24.0 * eps[None, :] * (2 * s6 * s6 - s6) * inv2, 0.0)
            # Axle core: WCA at sigma 1; stoppers are fat (sigma 2.6) so the
            # ring can never slide off the ends.
            sig = 1.0 + 1.6 * stopper[None, :]
            c2 = sig * sig * inv2
            c6 = c2 * c2 * c2
            core = xp.where(r2 < (1.1225 * sig) ** 2, 24.0 * (2 * c6 * c6 - c6) * inv2, 0.0)
            force = xp.sum((bind + core)[..., None] * delta, axis=1)
            force = self._bond_forces(xp, ring, force, 120.0, 1.0, ring=True)
            force = self._bend_forces(xp, ring, force, 40.0, math.cos(2 * math.pi / len(ring)), ring=True)
            # A real macrocycle is stiff: keep the ring roughly perpendicular
            # to the axle rather than letting it fold flat along it.
            force[:, 0] -= 30.0 * (ring[:, 0] - xp.mean(ring[:, 0]))
            return xp.clip(force, -400.0, 400.0)
        return forces, state

    def run_shuttle(self):
        if self.ctx.on_gpu:
            # Twelve mobile beads are far too few to occupy a GPU; kernel
            # launches would cost more than the arithmetic.
            self._xp = np
            self.rng = np.random.default_rng(int(self.ctx.params.get("seed", 11)))
            self.ctx.backend_name = "numpy"
            self.ctx.write_meta({"backend": "numpy", "compute_note": "The shuttle's 12-bead ring is integrated with NumPy on the CPU; "
                                                 "a GPU gives no benefit at this size."})
        xp = self.xp
        axle, kinds, station_a, station_b, ring0, radius = self.shuttle_build()
        forces, state = self.shuttle_forces_factory(axle, station_a, station_b, radius)
        kT = self.kT()
        # The ring needs many more steps than a fold to diffuse between stations.
        total = max(1, int(self.settings.get("shuttle_steps", 100000)))
        every = int(round(float(self.ctx.params.get("switch_every", 20))))
        ring = xp.asarray(ring0)
        vel = xp.zeros_like(ring)
        n_axle, n_ring = len(axle), len(ring0)
        bonds = [[i, i + 1] for i in range(n_axle - 1)] + \
                [[n_axle + i, n_axle + (i + 1) % n_ring] for i in range(n_ring)]
        gap = float(self.settings.get("station_gap", 5.0))
        self.ctx.write_meta({"physics": {"model": "rotaxane shuttle: rigid axle, flexible ring, Langevin dynamics",
                                         "units": "reduced: bead diameter 1, kT(310 K) = 0.5",
                                         "time_step": SHUTTLE_DT, "station_gap": gap}})
        self._begin_3d("shuttle", SHUTTLE_PALETTE, SHUTTLE_LABELS, SHUTTLE_SIZES)
        done, trips, flips, at, arrived_at = 0, 0, 0, 0, None
        last_frame_flip = 0
        for i in range(self.ctx.frames):
            if every > 0 and i > 0 and i - last_frame_flip >= every:
                state["sticky"] = 1 - state["sticky"]
                flips += 1
                last_frame_flip = i
                arrived_at = None
            target = int(round(total * (i + 1) / self.ctx.frames))
            with self.ctx.stage("simulation"):
                ring, vel = self.langevin(ring, vel, forces, max(1, target - done), kT, 0.15, dt=SHUTTLE_DT)
            done = target
            rn = to_numpy(ring)
            x = float(rn[:, 0].mean())
            side = -1 if x < -gap / 4 else 1 if x > gap / 4 else 0
            if side and side != at:
                if at:
                    trips += 1
                at = side
            sticky_side = -1 if state["sticky"] == 0 else 1
            if side == sticky_side and arrived_at is None:
                arrived_at = i - last_frame_flip
            types = kinds.copy()
            types[station_a] = 3 if state["sticky"] == 0 else 2
            types[station_b] = 3 if state["sticky"] == 1 else 2
            all_types = np.concatenate([types, np.full(n_ring, 4)])
            pos = np.concatenate([axle, rn])
            with self.ctx.stage("render"):
                image = self.render(pos, all_types, SHUTTLE_PALETTE, bonds=bonds, angle=0.3 + 0.003 * i,
                                    subtitle=f"a ring on an axle · the switch picks the sticky station · {self.ctx.backend_name}",
                                    progress=done / total, left="SWITCH", right="SHUTTLE",
                                    scale_to=(len(axle) - 1) / 2 + 1.5)
                self.ctx.save_frame(image, self.ctx.frame_path(i))
                self._save_3d(i, pos, all_types, bonds=bonds, step=done)
            self.ctx.write_status(i, f"step {done:,} · ring at {x:+.1f}", {
                "sticky station": "left" if state["sticky"] == 0 else "right",
                "ring position": f"{x:+.1f}",
                "trips along the axle": f"{trips}",
                "switch flips": f"{flips}",
                "temperature": f"{float(self.ctx.params.get('temperature', 310)):.0f} K",
                "time steps": f"{done:,}"})
        self.ctx.write_meta({"summary": {"mode": "shuttle", "trips": trips, "flips": flips}})
        self.ctx.finish()

    # ------------------------------------------------------------ output --
    def _begin_3d(self, mode, palette, labels, sizes=None):
        (self.ctx.run_dir / "interactive").mkdir(exist_ok=True)
        self._palette, self._labels, self._mode = palette, labels, mode
        self._sizes = list(sizes or [1.0] * len(palette))
        self.ctx.write_meta({"galaxy3d_view": {"kind": "molecule-3d", "folder": "interactive",
                                               "frames": self.ctx.frames, "mode": mode}})

    def _save_3d(self, i, pos, types, bonds, step):
        p = np.asarray(pos, dtype=np.float64)
        data = {"kind": "molecule-3d", "mode": self._mode, "step": int(step),
                "positions": np.round(p, 3).tolist(), "types": [int(t) for t in types],
                "palette": [list(c) for c in self._palette], "labels": self._labels, "sizes": self._sizes,
                "chain": bonds is None, "bonds": bonds,
                "extent": float(max(4.0, np.max(np.abs(p - p.mean(axis=0))) + 1.5))}
        path = self.ctx.run_dir / "interactive" / f"frame_{i:04d}.json"
        path.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")

    @staticmethod
    def rotate(pos, yaw, pitch=0.38):
        ca, sa = math.cos(yaw), math.sin(yaw)
        cb, sb = math.cos(pitch), math.sin(pitch)
        ry = np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])
        rx = np.array([[1, 0, 0], [0, cb, -sb], [0, sb, cb]])
        return np.asarray(pos, dtype=np.float64) @ ry.T @ rx.T

    def render(self, pos, types, palette, bonds, angle, subtitle, progress, left, right, scale_to=None,
               size=(1280, 720)):
        sizes = getattr(self, "_sizes", None) or [1.0] * len(palette)
        w, h = size
        p = self.rotate(np.asarray(pos) - np.asarray(pos).mean(axis=0), angle)
        if scale_to:
            # The shuttle is long and thin: fit the axle to the frame width.
            extent = scale_to
            scale = w * 0.45 / extent
        else:
            extent = max(3.0, float(np.max(np.abs(p[:, :2]))) + 1.0)
            # Ease the zoom so the camera does not jump as the chain collapses.
            prev = getattr(self, "_extent", extent)
            extent = self._extent = 0.55 * prev + 0.45 * extent
            scale = min(w, h) * 0.47 / extent
        px, py = w * 0.5 + p[:, 0] * scale, h * 0.5 - p[:, 1] * scale
        depth = p[:, 2] / max(extent, 1e-6)
        radius = max(3.0, 0.42 * scale)
        types = np.asarray(types, dtype=int)
        base = Image.new("RGB", size, (2, 5, 14))
        glow = Image.new("RGBA", size, (0, 0, 0, 0))
        gd = ImageDraw.Draw(glow, "RGBA")
        for j in range(len(p)):
            c = palette[types[j]]
            r = radius * 1.9 * sizes[types[j]]
            gd.ellipse((px[j] - r, py[j] - r, px[j] + r, py[j] + r), fill=(*c, 46))
        base = Image.alpha_composite(base.convert("RGBA"), glow.filter(ImageFilter.GaussianBlur(radius)))
        sharp = Image.new("RGBA", size, (0, 0, 0, 0))
        sd = ImageDraw.Draw(sharp, "RGBA")
        pairs = bonds if bonds is not None else [[i, i + 1] for i in range(len(p) - 1)]
        # Painter's algorithm over bonds and atoms together, back to front.
        items = [(float((p[a, 2] + p[b, 2]) / 2) - 1e-3, 0, a, b) for a, b in pairs]
        items += [(float(p[j, 2]), 1, j, j) for j in range(len(p))]
        items.sort()
        for z, kind, a, b in items:
            shade = float(np.clip(0.72 + 0.28 * z / max(extent, 1e-6), 0.38, 1.05))
            if kind == 0:
                col = tuple(int(np.clip(v * shade, 0, 255)) for v in (176, 200, 226))
                sd.line((px[a], py[a], px[b], py[b]), fill=(*col, 230), width=max(2, int(radius * 0.42)))
                continue
            c = palette[types[a]]
            rr = radius * sizes[types[a]] * (0.9 + 0.1 * float(np.clip(depth[a], -1, 1)))
            # A few concentric discs fake a lit sphere without OpenGL.
            for k, f in enumerate((1.0, 0.82, 0.6, 0.36)):
                lift = 0.55 + 0.2 * k
                col = tuple(int(np.clip(v * shade * lift + 40 * k * shade, 0, 255)) for v in c)
                off = rr * (1 - f) * 0.45
                sd.ellipse((px[a] - rr * f - off, py[a] - rr * f - off, px[a] + rr * f - off, py[a] + rr * f - off),
                           fill=(*col, 255))
        image = Image.alpha_composite(base, sharp).convert("RGB")
        image = add_title(image, "Molecular Machine", subtitle,
                          badge="LIVE MOLECULAR DYNAMICS" if self.mode == "fold" else "MOLECULAR MACHINE")
        add_progress(image, progress, left, right)
        return image

    def run(self):
        self.rng = self._rng(int(self.ctx.params.get("seed", 11)))
        if self.mode == "shuttle":
            self.run_shuttle()
        else:
            self.run_fold()
