"""Visitor-built brains and batched neuroevolution.

A visitor assembles a network from blocks (sensors, hidden-layer bricks and
actions) in the viewer.  That brain spec is validated here against the demo's
block catalogue in ``config/demo_specs.json`` so the API and the solver apply
exactly the same rules.

``Population`` then evolves many copies of that architecture at once.  Every
individual's parameters are one row of a flat genome matrix, and the forward
pass is a batched multiply-and-sum over per-individual weight tensors, so a
population of 8192 costs a few array operations per layer rather than a
Python loop.
Evolution needs no gradients, which keeps the whole engine on ``ctx.xp``
(NumPy or CuPy).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from .backend import to_numpy

MAX_HIDDEN_LAYERS = 3
MAX_LAYER_WIDTH = 16


class BrainError(ValueError):
    """A brain spec the visitor built that the catalogue does not allow."""


def brain_catalogue(specs: dict, demo: str, role: str | None = None) -> dict | None:
    """Return a demo's block catalogue, or None if the demo has no builder.

    Co-evolution demos carry one catalogue per species under ``roles``.
    """
    brain = (specs.get(demo) or {}).get("brain")
    if not brain:
        return None
    if role is not None:
        return (brain.get("roles") or {}).get(role)
    return brain


def validate_brain(spec, catalogue: dict) -> dict:
    """Check a brain spec against its catalogue and return a normalised copy.

    The result is canonical (blocks in catalogue order, rays sorted by angle)
    so the same design always produces the same network layout, and it carries
    the derived ``layer_sizes``, ``points`` and ``parameters``.
    """
    if not isinstance(spec, dict):
        raise BrainError("brain must be an object with sensors, hidden and actions")
    blocks = catalogue["blocks"]
    order = list(blocks)
    budget = float(catalogue["budget"])

    sensors_in = spec.get("sensors", [])
    if not isinstance(sensors_in, list) or len(sensors_in) > 32:
        raise BrainError("sensors must be a list of at most 32 blocks")
    sensors, seen, counts, points = [], set(), {}, 0.0
    for item in sensors_in:
        if not isinstance(item, dict) or item.get("block") not in blocks:
            raise BrainError(f"unknown sensor block: {item!r}")
        name = item["block"]
        block = blocks[name]
        if block["kind"] != "sensor":
            raise BrainError(f"{block['label']} is not a sensor")
        entry = {"block": name}
        if "angles" in block:
            try:
                angle = int(item.get("angle"))
            except (TypeError, ValueError):
                raise BrainError(f"{block['label']} needs an angle") from None
            if angle not in block["angles"]:
                raise BrainError(f"{block['label']} cannot point at {angle}°")
            entry["angle"] = angle
        key = (name, entry.get("angle"))
        if key in seen:
            raise BrainError(f"{block['label']} is already placed there")
        seen.add(key)
        counts[name] = counts.get(name, 0) + 1
        if counts[name] > int(block.get("max", 1)):
            raise BrainError(f"at most {block.get('max', 1)} × {block['label']}")
        sensors.append(entry)
        points += float(block["cost"])
    sensors.sort(key=lambda s: (order.index(s["block"]), s.get("angle", 0)))

    hidden_rule = catalogue["hidden"]
    hidden = spec.get("hidden", [])
    if not isinstance(hidden, list) or len(hidden) > min(MAX_HIDDEN_LAYERS, int(hidden_rule["max_layers"])):
        raise BrainError(f"at most {hidden_rule['max_layers']} hidden layers")
    clean_hidden = []
    for width in hidden:
        try:
            width = int(width)
        except (TypeError, ValueError):
            raise BrainError("hidden layer widths must be whole numbers") from None
        if str(width) not in hidden_rule["costs"] or width > MAX_LAYER_WIDTH:
            raise BrainError(f"hidden layers can have {', '.join(hidden_rule['costs'])} neurons")
        clean_hidden.append(width)
        points += float(hidden_rule["costs"][str(width)])

    actions_in = spec.get("actions", [])
    if not isinstance(actions_in, list):
        raise BrainError("actions must be a list")
    actions = []
    for name in actions_in:
        if name not in blocks or blocks[name]["kind"] != "action":
            raise BrainError(f"unknown action block: {name!r}")
        if name not in actions:
            actions.append(name)
            points += float(blocks[name]["cost"])
    for name, block in blocks.items():
        if block["kind"] == "action" and block.get("required") and name not in actions:
            raise BrainError(f"the brain needs a {block['label']} action")
    actions.sort(key=order.index)

    if points > budget + 1e-9:
        raise BrainError(f"this brain costs {points:g} points but the budget is {budget:g}")

    inputs = sum(block_inputs(blocks[s["block"]], actions) for s in sensors)
    # A brain with no sensors still gets a constant bias input, so a "blind"
    # design is a legitimate, testable (and usually hopeless) network.
    layer_sizes = [max(1, inputs), *clean_hidden, len(actions)]
    return {
        "sensors": sensors,
        "hidden": clean_hidden,
        "actions": actions,
        "points": round(points, 3),
        "budget": budget,
        "inputs": inputs,
        "layer_sizes": layer_sizes,
        "parameters": parameter_count(layer_sizes),
    }


def block_inputs(block: dict, actions) -> int:
    """Network inputs a sensor block adds.

    ``"inputs": "actions"`` is a memory block: it feeds back the previous
    value of every action, so its width follows the chosen actions.
    """
    value = block.get("inputs", 1)
    return len(actions) if value == "actions" else int(value)


def validate_brains(spec, brain: dict):
    """Validate a request for a demo with one brain, or one brain per role.

    Co-evolution demos carry ``roles`` (e.g. bat and moth); a role the visitor
    did not design falls back to that role's default preset.
    """
    roles = brain.get("roles")
    if not roles:
        return validate_brain(spec, brain)
    if not isinstance(spec, dict) or set(spec) - set(roles):
        raise BrainError(f"brains must be given per role: {', '.join(roles)}")
    out = {}
    for role, catalogue in roles.items():
        chosen = spec.get(role)
        if chosen is None:
            preset = catalogue["presets"][catalogue["default_preset"]]
            chosen = {k: preset[k] for k in ("sensors", "hidden", "actions")}
        try:
            out[role] = validate_brain(chosen, catalogue)
        except BrainError as error:
            raise BrainError(f"{role}: {error}") from None
    return out


def parameter_count(layer_sizes) -> int:
    return int(sum(a * b + b for a, b in zip(layer_sizes[:-1], layer_sizes[1:])))


def input_labels(brain: dict, catalogue: dict) -> list[str]:
    """Human-readable name of every network input, in network order."""
    labels = []
    for sensor in brain["sensors"]:
        block = catalogue["blocks"][sensor["block"]]
        names = block.get("input_names") or [block.get("short", block["label"])]
        if block.get("inputs") == "actions":
            names = [f"{names[0]} {label}" for label in action_labels(brain, catalogue)]
        for name in names:
            if "angle" in sensor:
                # Negative angles turn toward the agent's left.
                angle = sensor["angle"]
                side = "ahead" if angle == 0 else f"{'L' if angle < 0 else 'R'}{abs(angle)}°"
                labels.append(f"{name} {side}")
            else:
                labels.append(name)
    return labels or ["bias"]


def action_labels(brain: dict, catalogue: dict) -> list[str]:
    return [catalogue["blocks"][a].get("short", catalogue["blocks"][a]["label"]) for a in brain["actions"]]


class Population:
    """Many individuals of one architecture, evolved in ``groups`` islands.

    Individuals only ever compete inside their own group, so a reveal can run
    sixteen genuinely independent evolutions as one batched population.
    """

    def __init__(self, xp, size: int, layer_sizes, seed: int = 0, groups: int = 1,
                 sigma: float = 0.5, sigma_decay: float = 0.97, sigma_min: float = 0.08,
                 mutation_rate: float = 0.2, elite: float = 0.06, tournament: int = 3,
                 init_gain: float = 1.0):
        self.xp = xp
        self.groups = max(1, int(groups))
        self.group_size = max(2, int(size))
        self.size = self.groups * self.group_size
        self.layer_sizes = [int(v) for v in layer_sizes]
        self.shapes = list(zip(self.layer_sizes[:-1], self.layer_sizes[1:]))
        self.length = parameter_count(self.layer_sizes)
        self.rng = np.random.default_rng(seed)
        self.sigma, self.sigma_decay, self.sigma_min = float(sigma), float(sigma_decay), float(sigma_min)
        self.mutation_rate, self.elite, self.tournament = float(mutation_rate), float(elite), int(tournament)
        # Initial weights are N(0, init_gain / sqrt(fan_in)).  A gain above 1 keeps
        # a signal alive through random hidden layers, which matters when the
        # useful cue is small (a bat's left-right loudness difference).
        self.init_gain = float(init_gain)
        self.generation = 0
        self.genome = xp.asarray(self._initial(self.size))

    def _initial(self, count):
        parts = []
        for fan_in, fan_out in self.shapes:
            parts.append(self.rng.normal(0.0, self.init_gain / math.sqrt(fan_in), (count, fan_in * fan_out)))
            parts.append(np.zeros((count, fan_out)))
        return np.concatenate(parts, axis=1).astype(np.float32)

    def layers(self, genome=None):
        """Per-layer (weights, bias) views of the genome, weights (P, in, out)."""
        genome = self.genome if genome is None else genome
        out, offset, count = [], 0, genome.shape[0]
        xp = self.xp
        for fan_in, fan_out in self.shapes:
            w = xp.ascontiguousarray(genome[:, offset:offset + fan_in * fan_out]).reshape(count, fan_in, fan_out)
            offset += fan_in * fan_out
            b = genome[:, offset:offset + fan_out]
            offset += fan_out
            out.append((w, b))
        return out

    def forward(self, x, genome=None, return_hidden=False):
        """Batched forward pass; x is (P, inputs) and returns (P, actions) in [-1, 1]."""
        xp = self.xp
        h = x.astype(xp.float32)
        activations = [h]
        for w, b in self.layers(genome):
            # Broadcast multiply-sum rather than a batched GEMM: layers are at
            # most 16x16, and CuPy's cuBLAS path crashes (access violation)
            # in a worker thread once PyTorch's CUDA libraries are also loaded
            # in the process, which is exactly the viewer's situation.
            h = xp.tanh((h[:, :, None] * w).sum(axis=1) + b)
            activations.append(h)
        return (h, activations) if return_hidden else h

    def evolve(self, fitness):
        """Replace the population with the next generation.

        Per group: the best ``elite`` fraction survives unchanged; the rest are
        children of two tournament winners with uniform crossover, then a
        fraction of genes gets Gaussian noise.  Sigma decays so early
        generations explore and later ones refine.
        """
        xp, rng = self.xp, self.rng
        groups, size = self.groups, self.group_size
        fit = to_numpy(fitness).astype(np.float64).reshape(groups, size)
        fit = np.where(np.isfinite(fit), fit, -1e9)
        keep = max(1, int(round(size * self.elite)))
        children = size - keep
        elite_idx, parent_a, parent_b = [], [], []
        for g in range(groups):
            base = g * size
            elite_idx.append(base + np.argsort(-fit[g], kind="stable")[:keep])
            for parents in (parent_a, parent_b):
                picks = rng.integers(0, size, (children, self.tournament))
                winners = picks[np.arange(children), np.argmax(fit[g][picks], axis=1)]
                parents.append(base + winners)
        genome = self.genome
        width = genome.shape[1]
        elites = genome[xp.asarray(np.concatenate(elite_idx))]
        a = genome[xp.asarray(np.concatenate(parent_a))]
        b = genome[xp.asarray(np.concatenate(parent_b))]
        kids = xp.where(xp.asarray(rng.random(a.shape) < 0.5), a, b)
        mutate = rng.random(kids.shape) < self.mutation_rate
        noise = rng.standard_normal(kids.shape).astype(np.float32) * np.float32(self.sigma)
        kids = kids + xp.asarray(np.where(mutate, noise, np.float32(0)))
        # Row g*size is always the previous best of group g (see elite_rows).
        self.genome = xp.concatenate([elites.reshape(groups, keep, width),
                                      kids.reshape(groups, children, width)], axis=1).reshape(self.size, width)
        self.sigma = max(self.sigma_min, self.sigma * self.sigma_decay)
        self.generation += 1

    def elite_rows(self):
        """Index of the first (best-surviving) row of every group after evolve()."""
        return [g * self.group_size for g in range(self.groups)]


def save_champion(path: Path, genome_row, brain: dict, meta: dict | None = None):
    path = Path(path)
    np.savez_compressed(path, genome=to_numpy(genome_row).astype(np.float32),
                        brain=json.dumps(brain), meta=json.dumps(meta or {}))


def load_champion(path: Path):
    """Return (genome[1, D], brain spec, meta) saved by ``save_champion``."""
    with np.load(Path(path), allow_pickle=False) as data:
        genome = np.asarray(data["genome"], dtype=np.float32).reshape(1, -1)
        brain = json.loads(str(data["brain"]))
        meta = json.loads(str(data["meta"]))
    if genome.shape[1] != parameter_count(brain["layer_sizes"]):
        raise ValueError("champion genome does not match its brain spec")
    return genome, brain, meta
