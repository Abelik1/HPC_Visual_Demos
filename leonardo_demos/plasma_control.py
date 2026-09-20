from __future__ import annotations

"""Reduced, differentiable plasma-control environment and torus confinement.

Two things live here, both deliberately small:

* A trainable feedback-control environment inspired by the loop used in real
  tokamaks - noisy diagnostics -> neural policy -> coil commands -> new plasma
  state.  The policy is genuinely optimized by back-propagating through the
  simulated dynamics.  It is **not** an equilibrium, transport, tearing-mode or
  disruption solver.
* A guiding-centre-like particle population living inside the torus volume.
  Its confinement is driven by the control state above, so the commands the
  network issues really do decide which particles stay inside the magnetic
  bottle and which reach the wall.

Both are exhibition-scale reduced models.  See docs/SCIENTIFIC_NOTES.md.
"""

import math

import numpy as np

from .backend import torch_device


INPUT_NAMES = ("radial", "vertical", "radial v", "vertical v", "tearing", "pressure")
OUTPUT_NAMES = ("radial coils", "vertical coils", "shape coils")

# The controlled state starts displaced and already slightly unstable so the
# uncontrolled reference has somewhere to fall from.
START_STATE = (0.34, -0.28, 0.08, -0.06, 0.28, 0.88)

# How far the controller's dimensionless displacement moves the magnetic axis,
# in units of the wall minor radius.  The renderer and the training objective
# must agree on this or the policy optimizes a different machine than the one
# on screen.
AXIS_GAIN = 0.80


def sample_periodic(field, x, y):
    """Bilinearly sample a 2-D periodic field at normalised (x, y)."""
    field = np.asarray(field)
    ny, nx = field.shape
    fx = np.mod(np.asarray(x, dtype=np.float32), 1.0) * nx
    fy = np.mod(np.asarray(y, dtype=np.float32), 1.0) * ny
    x0 = np.floor(fx).astype(np.int32) % nx
    y0 = np.floor(fy).astype(np.int32) % ny
    x1, y1 = (x0 + 1) % nx, (y0 + 1) % ny
    ax, ay = fx - np.floor(fx), fy - np.floor(fy)
    return (
        field[y0, x0] * (1 - ax) * (1 - ay)
        + field[y0, x1] * ax * (1 - ay)
        + field[y1, x0] * (1 - ax) * ay
        + field[y1, x1] * ax * ay
    )


# ---------------------------------------------------------------------------
# reduced control environment
# ---------------------------------------------------------------------------

def transition_numpy(state, action, disturbance, step, drive):
    """One step of the reduced unstable plasma state space.

    State is radial/vertical position and velocity, a tearing-risk proxy and a
    pressure proxy.  The positive position feedback and growing tearing proxy
    create an unstable open loop.  Three aggregate coil banks counter it.
    """
    x, y, vx, vy, mode, pressure = state
    ar, av, ashape = action
    phase = .21 * step
    kick_r = drive * (.038 * math.sin(phase * 1.7) + .020 * math.sin(phase * 3.1))
    kick_v = drive * (.034 * math.cos(phase * 1.3) - .018 * math.sin(phase * 2.4))
    vx = vx + .075 * (0.78 * x - .48 * vx + 1.52 * ar + kick_r)
    vy = vy + .075 * (0.82 * y - .44 * vy + 1.52 * av + kick_v)
    x = x + .075 * vx
    y = y + .075 * vy
    wall = x * x + y * y
    pressure = np.clip(pressure + .075 * (.09 * drive - .075 * (ashape + 1) / 2 - .045 * wall), .45, 1.35)
    mode = np.clip(mode + .075 * (.22 * drive + 1.22 * wall + .15 * (vx * vx + vy * vy) - .94 * (ashape + 1) / 2), 0, 1.6)
    return np.array([x, y, vx, vy, mode, pressure], dtype=np.float32)


def rollout_numpy(start, steps, drive, policy=None):
    state = np.asarray(start, dtype=np.float32).copy()
    states, actions = [], []
    for k in range(steps):
        action = np.zeros(3, dtype=np.float32) if policy is None else np.asarray(policy(state), dtype=np.float32)
        action = np.clip(action, -1, 1)
        states.append(state.copy()); actions.append(action)
        state = transition_numpy(state, action, np.zeros(2), k, drive)
    return np.asarray(states), np.asarray(actions)


def risk_of(state):
    """Dimensionless instability proxy used for the readouts, in [0, 1]."""
    state = np.asarray(state, dtype=np.float32)
    return float(np.clip(.35 * (state[0] ** 2 + state[1] ** 2) + .70 * state[4] ** 2
                         + .11 * float(state[2:4] @ state[2:4]), 0, 1.0))


def wall_load_terms(gate, confine, drive, island, pressure):
    """Coefficients of the marker radial balance, as a quadratic in r.

    ``ConfinedParticles.advance`` balances an edge-weighted outward drift
    against a restoring term set by the shaping command and the field.  Writing
    that balance as ``A r^2 - B r - C = 0`` makes the equilibrium radius of the
    marker population - and therefore how close it sits to the wall - a
    closed-form, differentiable function of what the controller is doing.
    ``turb`` is taken at its mean, so this is the population's expected
    behaviour rather than any individual marker's.
    """
    mean_turb = 0.30 + 0.90 * 0.5
    a = 0.0118 * (0.30 + 1.00 * gate) * confine
    b = 0.0070 * drive * mean_turb * 1.10
    c = 0.0070 * drive * mean_turb * 0.45 + 0.0125 * island * 0.6 + 0.0040 * (pressure - 0.90)
    return a, b, c


class AnalyticSafetyPolicy:
    """Explicit fallback when PyTorch is unavailable; it never claims training."""

    device = "analytical fallback"
    training = False

    def train(self, updates, batch, horizon, drive, confine=1.0, experience=None):
        return 0.0

    @staticmethod
    def actions(state):
        state = np.asarray(state, dtype=np.float32)
        return np.clip(np.stack((
            -1.20 * state[..., 0] - 0.56 * state[..., 2],
            -1.20 * state[..., 1] - 0.56 * state[..., 3],
            1.0 * state[..., 4] + 0.18 * state[..., 5],
        ), axis=-1), -1, 1)

    def act(self, state):
        return np.asarray(self.actions(state), dtype=np.float32)

    def evaluate(self, steps, drive, controlled):
        state = np.array(START_STATE, dtype=np.float32)
        return rollout_numpy(state, steps, drive, self.actions if controlled else None)

    def weights(self):
        rng = np.random.default_rng(7)
        return (rng.normal(0, .4, (6, 14)), rng.normal(0, .32, (14, 10)), rng.normal(0, .48, (10, 3)))


class TorchPolicy:
    """Small neural policy optimized through batches of virtual plasma shots."""

    training = True

    def __init__(self, requested, learning_rate=0.004):
        import torch

        self.torch = torch
        self.device = torch_device(requested)
        torch.manual_seed(23)
        self.policy = torch.nn.Sequential(
            torch.nn.Linear(6, 14), torch.nn.Tanh(),
            torch.nn.Linear(14, 10), torch.nn.Tanh(),
            torch.nn.Linear(10, 3), torch.nn.Tanh(),
        ).to(self.device)
        # Paced so the policy improves over several shots instead of solving
        # the task in the first training burst: the exhibition needs the
        # learning to be watchable, and a plateau afterwards is honest.
        self.optim = torch.optim.Adam(self.policy.parameters(), lr=float(learning_rate))
        # Trigger the backend's first matmul while construction is still inside
        # make_trainer.  A CUDA library can report a device then fail only when
        # cuBLAS initialises; Auto mode can safely retry on CPU in that case.
        with torch.no_grad():
            self.policy(torch.zeros((1, 6), device=self.device)).sum().item()
        self.last_loss = 0.0

    def _transition(self, state, step, drive):
        torch = self.torch
        action = self.policy(state)
        x, y, vx, vy, mode, pressure = state.unbind(-1)
        ar, av, ashape = action.unbind(-1)
        phase = .21 * step
        kick_r = drive * (.038 * math.sin(phase * 1.7) + .020 * math.sin(phase * 3.1))
        kick_v = drive * (.034 * math.cos(phase * 1.3) - .018 * math.sin(phase * 2.4))
        vx = vx + .075 * (.78 * x - .48 * vx + 1.52 * ar + kick_r)
        vy = vy + .075 * (.82 * y - .44 * vy + 1.52 * av + kick_v)
        x = x + .075 * vx
        y = y + .075 * vy
        wall = x.square() + y.square()
        pressure = torch.clamp(pressure + .075 * (.09 * drive - .075 * (ashape + 1) / 2 - .045 * wall), .45, 1.35)
        mode = torch.clamp(mode + .075 * (.22 * drive + 1.22 * wall + .15 * (vx.square() + vy.square()) - .94 * (ashape + 1) / 2), 0, 1.6)
        return torch.stack((x, y, vx, vy, mode, pressure), -1), action

    def _wall_load(self, state, action, confine, drive):
        """Differentiable estimate of how much of the marker population is lost.

        The policy is optimized against the same balance the visible markers
        obey, so "keep the plasma centred" is not a stand-in for "keep the
        plasma off the wall" - the second is what is actually minimized.
        """
        torch = self.torch
        gate = (action[:, 2] + 1) / 2
        island = state[:, 4]
        radius = wall_load_terms(gate, confine, drive, island, state[:, 5])
        a, b, c = radius
        c = torch.clamp(c, min=1e-5)
        equilibrium = (b + torch.sqrt(b * b + 4 * a * c)) / (2 * a)
        displacement = AXIS_GAIN * torch.sqrt(state[:, 0].square() + state[:, 1].square() + 1e-9)
        spread = 0.16 + 0.45 * island
        return torch.nn.functional.softplus(
            (displacement + equilibrium + spread - 1.0) * 6.0) / 6.0

    def _seed_batch(self, batch, experience):
        """Start states for one update: half replayed from the last shot."""
        torch = self.torch
        state = torch.empty((batch, 6), device=self.device).uniform_(-1, 1)
        state[:, :2] *= .46; state[:, 2:4] *= .14
        state[:, 4] = torch.empty(batch, device=self.device).uniform_(.08, .52)
        state[:, 5] = torch.empty(batch, device=self.device).uniform_(.70, 1.18)
        if experience is None or len(experience) < 4:
            return state
        # Replaying the states the shot actually visited is what makes this
        # learning *from the shot*; the random half keeps the policy from
        # overfitting to one trajectory.
        buffer = torch.as_tensor(np.asarray(experience, dtype=np.float32), device=self.device)
        take = batch // 2
        picked = buffer[torch.randint(0, buffer.shape[0], (take,), device=self.device)]
        jitter = torch.randn((take, 6), device=self.device)
        jitter[:, :2] *= .08; jitter[:, 2:4] *= .03; jitter[:, 4] *= .06; jitter[:, 5] *= .04
        picked = picked + jitter
        picked[:, 4] = picked[:, 4].clamp(0.0, 1.6)
        picked[:, 5] = picked[:, 5].clamp(0.45, 1.35)
        state[:take] = picked
        return state

    def train(self, updates, batch, horizon, drive, confine=1.0, experience=None):
        torch = self.torch
        losses = []
        for _ in range(max(1, int(updates))):
            state = self._seed_batch(batch, experience)
            loss = torch.zeros((), device=self.device)
            for step in range(horizon):
                state, action = self._transition(state, step, drive)
                wall = state[:, 0].square() + state[:, 1].square()
                loss = loss + (
                    2.4 * self._wall_load(state, action, confine, drive)
                    + 1.4 * wall
                    + 1.3 * state[:, 4].square()
                    + .22 * state[:, 2:4].square().sum(-1)
                    + .015 * action.square().sum(-1)
                ).mean()
            loss = loss / horizon
            self.optim.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 1.0)
            self.optim.step()
            losses.append(float(loss.detach().cpu()))
        self.last_loss = float(np.mean(losses))
        return self.last_loss

    def act(self, state):
        torch = self.torch
        with torch.no_grad():
            tensor = torch.tensor(np.asarray(state, dtype=np.float32)[None], device=self.device)
            return np.clip(self.policy(tensor)[0].detach().cpu().numpy(), -1, 1)

    def evaluate(self, steps, drive, controlled):
        torch = self.torch
        state = torch.tensor([list(START_STATE)], device=self.device)
        states, actions = [], []
        with torch.no_grad():
            for k in range(steps):
                if controlled:
                    state, action = self._transition(state, k, drive)
                else:
                    value = state[0].detach().cpu().numpy()
                    nxt = transition_numpy(value, np.zeros(3, dtype=np.float32), np.zeros(2), k, drive)
                    action = torch.zeros((1, 3), device=self.device)
                    state = torch.tensor(nxt[None], device=self.device)
                states.append(state[0].detach().cpu().numpy()); actions.append(action[0].detach().cpu().numpy())
        return np.asarray(states), np.asarray(actions)

    def weights(self):
        layers = [m.weight.detach().cpu().numpy().T for m in self.policy if hasattr(m, "weight")]
        return tuple(layers)


def save_controller(path, trainer):
    """Save the policy as it is now, as plain arrays, so it can fly again later.

    The network is only ever used frozen after this, so NumPy is enough to
    run it: no PyTorch and no GPU are needed to test a saved controller.
    """
    policy = getattr(trainer, "policy", None)
    if policy is None:
        np.savez_compressed(path, kind=np.array("analytic"))
        return
    arrays = {"kind": np.array("mlp")}
    for i, layer in enumerate(m for m in policy if hasattr(m, "weight")):
        arrays[f"w{i}"] = layer.weight.detach().cpu().numpy().astype(np.float32)
        arrays[f"b{i}"] = layer.bias.detach().cpu().numpy().astype(np.float32)
    np.savez_compressed(path, **arrays)


def load_controller(path, trainer) -> int:
    """Put a saved controller back into a trainable policy, to carry on training.

    Returns the number of layers loaded, or 0 if the file holds the analytic
    fallback or does not fit this network. A run that continues an earlier one
    must not silently start from scratch, so the caller checks the count.
    """
    policy = getattr(trainer, "policy", None)
    if policy is None:
        return 0
    layers = [m for m in policy if hasattr(m, "weight")]
    with np.load(path, allow_pickle=False) as data:
        if str(data["kind"]) != "mlp" or f"w{len(layers) - 1}" not in data:
            return 0
        shapes = [(data[f"w{i}"].shape, data[f"b{i}"].shape) for i in range(len(layers))]
        if any(tuple(layer.weight.shape) != w or tuple(layer.bias.shape) != b
               for layer, (w, b) in zip(layers, shapes)):
            return 0
        torch = trainer.torch
        with torch.no_grad():
            for i, layer in enumerate(layers):
                layer.weight.copy_(torch.as_tensor(data[f"w{i}"], device=trainer.device))
                layer.bias.copy_(torch.as_tensor(data[f"b{i}"], device=trainer.device))
    # Adam's moments belong to the old optimizer; start it clean on the loaded
    # weights rather than pretending the old momentum still applies.
    trainer.optim = trainer.torch.optim.Adam(policy.parameters(),
                                             lr=trainer.optim.param_groups[0]["lr"])
    return len(layers)


class FrozenController:
    """A saved controller, flown with NumPy exactly as the network computed it."""

    training = False

    def __init__(self, path):
        with np.load(path, allow_pickle=False) as data:
            self.kind = str(data["kind"])
            self.layers = []
            i = 0
            while f"w{i}" in data:
                self.layers.append((data[f"w{i}"], data[f"b{i}"]))
                i += 1

    def act(self, state):
        if self.kind != "mlp":
            return AnalyticSafetyPolicy.actions(state).astype(np.float32)
        x = np.asarray(state, dtype=np.float32)
        for weight, bias in self.layers:
            x = np.tanh(weight @ x + bias)
        return np.clip(x, -1, 1).astype(np.float32)

    def weights(self):
        return tuple(w.T for w, _ in self.layers) if self.layers else AnalyticSafetyPolicy().weights()


def make_trainer(ctx, learning_rate=0.004):
    """Build the policy, honouring the requested backend and never lying about it."""
    requested = getattr(ctx, "backend_requested", "auto")
    base = getattr(ctx, "backend_name", "numpy")
    try:
        trainer = TorchPolicy(requested, learning_rate)
        ctx.set_backend_name(f"{base} + torch·{trainer.device}")
        # The array backend is recorded by RunContext; the policy runs on its
        # own device, so record that separately rather than losing it.
        ctx.write_meta({"torch_device": str(trainer.device)})
        return trainer
    except Exception as exc:
        if str(requested).lower() == "auto":
            try:
                trainer = TorchPolicy("cpu", learning_rate)
                ctx.set_backend_name(f"{base} + torch·cpu (CUDA unavailable at runtime)")
                return trainer
            except Exception:
                pass
        requested = str(requested).lower()
        if requested in {"gpu", "cuda", "cupy", "hybrid", "cpu+gpu", "cpu_gpu"}:
            raise RuntimeError(
                f"GPU requested for the guardian mode but PyTorch could not start: {exc}") from exc
        ctx.set_backend_name(f"{base} + analytical fallback — PyTorch unavailable")
        return AnalyticSafetyPolicy()


# ---------------------------------------------------------------------------
# confined particle population
# ---------------------------------------------------------------------------

# Sparks are cheap but unbounded growth would eventually dominate the manifest.
MAX_SPARKS = 260


class ConfinedParticles:
    """Guiding-centre-like markers moving inside the torus volume.

    Each particle carries a toroidal angle ``u``, a poloidal angle ``theta``
    (both in turns, so in [0, 1)) and a minor radius ``r`` measured from the
    magnetic axis in units of the wall minor radius.  The axis itself is the
    controller's displacement, so a policy that lets the column drift pushes
    the outer markers into the wall.  A marker that reaches the wall is
    recorded as a loss, leaves a spark at the contact point and is recycled
    back into the core, which keeps the population - and therefore the
    rendering cost - constant.

    This is a transport-flavoured exhibition model, not a gyrokinetic or
    full-orbit particle code: there is no gyromotion, no collision operator and
    no self-consistent field response.
    """

    def __init__(self, count, trail=6, seed=0):
        count = max(1, int(count))
        trail = max(2, int(trail))
        self.count = count
        self.rng = np.random.default_rng(seed)
        self.u = self.rng.random(count).astype(np.float32)
        self.theta = self.rng.random(count).astype(np.float32)
        # sqrt keeps the seeded population uniform over the cross-section area.
        self.r = (0.12 + 0.62 * np.sqrt(self.rng.random(count))).astype(np.float32)
        self.trail = np.repeat(
            np.stack([self.u, self.theta, self.r], axis=-1)[:, None, :], trail, axis=1)
        self.sparks = np.zeros((0, 4), dtype=np.float32)
        self.phase = 0.0
        self.lost_total = 0
        self.last_lost = 0
        self.steps = 0

    # -- geometry -----------------------------------------------------------
    def offsets(self, axis):
        """Poloidal-plane position of every marker, in wall-minor-radius units."""
        angle = 2 * np.pi * self.theta
        return (axis[0] + self.r * np.cos(angle), axis[1] + self.r * np.sin(angle))

    def clearance(self, axis):
        a, b = self.offsets(axis)
        return np.clip(1.0 - np.sqrt(a * a + b * b), 0.0, 1.0)

    # -- evolution ----------------------------------------------------------
    def advance(self, turbulence, axis, island, pressure, shape_command,
                magnetic_field, heating, drive, substeps=4, spark_life=9.0):
        """Advance every marker through the field the controller is shaping.

        ``turbulence`` is a normalised map derived from the live plasma-wave
        solver, sampled at the marker's own (toroidal, poloidal) angle, so the
        radial transport really is driven by the simulated field rather than by
        an independent random process.
        """
        b = max(0.1, float(magnetic_field))
        heat = float(heating)
        drive = float(drive)
        island = float(np.clip(island, 0.0, 1.6))
        pressure = float(pressure)
        gate = 0.5 * (float(np.clip(shape_command, -1, 1)) + 1.0)
        axis = (float(axis[0]), float(axis[1]))

        confine = max(0.55, math.sqrt(b / 5.0))
        pitch = 0.48 + 0.045 * b
        v_tor = 0.0085 * max(0.25, heat / 25.0) / confine
        v_pol = v_tor * pitch * (0.72 + 0.55 * gate)
        substeps = max(1, int(substeps))
        lost = 0
        contacts = []

        for _ in range(substeps):
            self.phase += 0.11
            turb = sample_periodic(turbulence, self.u, self.theta).astype(np.float32)
            angle = 2 * np.pi * self.theta
            # Resonant island kick: a m=2 structure that rotates with the field.
            resonance = np.cos(2 * angle - 2 * np.pi * self.u + self.phase)
            # Turbulent transport is edge-weighted, as it is in a real
            # device: the further out a marker already sits, the harder the
            # fluctuations push it.  That is what makes the balance below
            # sensitive to the instability drive instead of self-correcting.
            outward = (0.0070 * drive * (0.30 + 0.90 * turb) * (0.45 + 1.10 * self.r)
                       + 0.0125 * island * resonance
                       + 0.0040 * (pressure - 0.90))
            # Shaping and field strength decide how hard the coils pull the
            # outer markers back onto a closed surface.  The balance leaves a
            # well-controlled population sitting at roughly 0.7 of the wall
            # radius, so turbulent edge losses never stop entirely - they only
            # become rare.
            restore = 0.0118 * (0.30 + 1.00 * gate) * confine * (self.r ** 2)
            noise = self.rng.normal(0.0, 0.0030 + 0.0115 * island, self.count)
            self.r = self.r + (outward - restore + noise).astype(np.float32)
            # A rare large-angle scattering channel, standing in for the
            # collisional and charge-exchange losses that keep the particle
            # confinement time finite even in a well-held discharge.  It scales
            # with the drive and inversely with the field, so the exhibition's
            # physics controls still matter when the policy is doing well.
            scattered = self.rng.random(self.count) < (0.0022 * drive / confine)
            if np.any(scattered):
                self.r[scattered] += self.rng.uniform(0.14, 0.46, int(scattered.sum())).astype(np.float32)
            self.u = np.mod(self.u + v_tor * (0.85 + 0.30 * self.r), 1.0).astype(np.float32)
            self.theta = np.mod(self.theta + v_pol * (0.55 + 0.75 * self.r), 1.0).astype(np.float32)

            # A marker pushed through the magnetic axis reappears on the far side.
            through = self.r < 0
            if np.any(through):
                self.r[through] = -self.r[through]
                self.theta[through] = np.mod(self.theta[through] + 0.5, 1.0)

            a, c = self.offsets(axis)
            radius = np.sqrt(a * a + c * c)
            hit = radius >= 1.0
            if np.any(hit):
                impact = np.arctan2(c[hit], a[hit]) / (2 * np.pi)
                energy = np.clip(0.35 + 0.9 * (radius[hit] - 1.0) + 0.5 * island, 0.3, 1.6)
                contacts.append(np.stack([
                    self.u[hit], np.mod(impact, 1.0),
                    np.zeros(int(hit.sum()), dtype=np.float32), energy], axis=-1).astype(np.float32))
                count = int(hit.sum())
                lost += count
                # Recycling: a lost marker is refuelled near the magnetic axis
                # so the population, and therefore the render cost, is steady.
                self.r[hit] = (0.06 + 0.30 * np.sqrt(self.rng.random(count))).astype(np.float32)
                self.theta[hit] = self.rng.random(count).astype(np.float32)
            np.clip(self.r, 0.0, 1.0, out=self.r)

        self.trail = np.roll(self.trail, -1, axis=1)
        self.trail[:, -1, 0] = self.u
        self.trail[:, -1, 1] = self.theta
        self.trail[:, -1, 2] = self.r

        if self.sparks.size:
            self.sparks[:, 2] += 1.0
            self.sparks = self.sparks[self.sparks[:, 2] < spark_life]
        if contacts:
            self.sparks = np.concatenate([self.sparks] + contacts, axis=0)[-MAX_SPARKS:]

        self.last_lost = lost
        self.lost_total += lost
        self.steps += substeps
        clearance = self.clearance(axis)
        return {
            "lost": lost,
            "lost_total": self.lost_total,
            "loss_rate": lost / max(1, self.count),
            "mean_radius": float(np.mean(self.r)),
            "edge_fraction": float(np.mean(clearance < 0.18)),
            "clearance": clearance,
        }

    def spark_list(self, spark_life=9.0):
        """Active wall sparks as (toroidal, poloidal, age fraction, energy)."""
        if not self.sparks.size:
            return np.zeros((0, 4), dtype=np.float32)
        out = self.sparks.copy()
        out[:, 2] = np.clip(out[:, 2] / max(1.0, spark_life), 0.0, 1.0)
        return out
