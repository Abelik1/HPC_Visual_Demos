"""Measured CUDA launch-configuration selection.

The best block size and per-thread work for a kernel differ between an RTX
workstation card, a Leonardo A100 and a Discoverer B200, and between FP32 and
FP64.  Rather than hard-coding one card's optimum, each solver offers a small
set of candidate configurations and this module times them once on the device
actually running the job.  The winner is cached per device, kernel, precision
and problem-size bucket, both in memory and in a JSON file, so later runs on
the same hardware start immediately.

Controls:
  LEONARDO_DEMO_AUTOTUNE=0        use each solver's default configuration
  LEONARDO_DEMO_TUNING_CACHE=path cache file (default: <checkout>/tmp/kernel_tuning.json)
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Tuple

_ROOT = Path(__file__).resolve().parents[1]
_LOCK = threading.Lock()
_MEMORY: Dict[str, Dict[str, Any]] = {}


def enabled() -> bool:
    return os.getenv("LEONARDO_DEMO_AUTOTUNE", "1").strip().lower() not in {"0", "off", "false", "no"}


def cache_path() -> Path:
    return Path(os.getenv("LEONARDO_DEMO_TUNING_CACHE") or _ROOT / "tmp" / "kernel_tuning.json")


def size_bucket(n: int) -> int:
    """Power-of-two bucket, so N=200,000 and N=210,000 share one tuning."""
    return 1 << max(0, int(n) - 1).bit_length()


def device_key(cp) -> str:
    device = cp.cuda.Device()
    props = cp.cuda.runtime.getDeviceProperties(device.id)
    name = props.get("name", "unknown")
    if isinstance(name, bytes):
        name = name.decode(errors="replace").rstrip("\0")
    return f"{name}|sm{props.get('major', 0)}{props.get('minor', 0)}"


def _label(candidate: Dict[str, Any]) -> str:
    return ",".join(f"{k}={candidate[k]}" for k in sorted(candidate))


def _read_file() -> Dict[str, Any]:
    try:
        return json.loads(cache_path().read_text(encoding="utf-8"))
    except Exception:
        return {}


def _write_file(key: str, entry: Dict[str, Any]) -> None:
    # Best effort: a read-only checkout or a race between two jobs must never
    # fail a simulation, it only costs a re-tune next time.
    try:
        path = cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        data = _read_file()
        data[key] = entry
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
        temporary.write_text(json.dumps(data, indent=1, sort_keys=True), encoding="utf-8")
        temporary.replace(path)
    except Exception:
        pass


def select(cp, kernel: str, problem: str, candidates: Iterable[Dict[str, Any]],
           launch: Callable[[Dict[str, Any]], None], default: Dict[str, Any],
           repeats: int = 3) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Return ``(configuration, report)`` for ``kernel`` on the current device.

    ``launch(candidate)`` must run the kernel once and be free of lasting side
    effects (write to scratch buffers, or be idempotent).  Candidates that fail
    to compile or launch, for example by exceeding a register or shared-memory
    limit on a smaller GPU, are skipped.
    """
    candidates = [dict(c) for c in candidates]
    if not enabled():
        return dict(default), {"mode": "default (autotuning disabled)", "config": dict(default)}
    key = f"{device_key(cp)}|{kernel}|{problem}"
    labels = {_label(c): c for c in candidates}
    with _LOCK:
        entry = _MEMORY.get(key) or _read_file().get(key)
    if entry and entry.get("best") in labels:
        _MEMORY[key] = entry
        return dict(labels[entry["best"]]), {"mode": "cached", "config": labels[entry["best"]],
                                              "key": key}
    timings: Dict[str, float | None] = {}
    started = time.perf_counter()
    for candidate in candidates:
        label = _label(candidate)
        try:
            launch(candidate)  # compile + warm caches
            cp.cuda.Device().synchronize()
            begin, end = cp.cuda.Event(), cp.cuda.Event()
            begin.record()
            for _ in range(repeats):
                launch(candidate)
            end.record()
            end.synchronize()
            timings[label] = cp.cuda.get_elapsed_time(begin, end) / repeats
        except Exception:
            timings[label] = None
    measured = {k: v for k, v in timings.items() if v is not None}
    if not measured:
        return dict(default), {"mode": "default (no candidate launched)", "config": dict(default)}
    best = min(measured, key=measured.get)
    entry = {"best": best, "timings_ms": timings,
             "tuned_seconds": round(time.perf_counter() - started, 3),
             "tuned_at": time.strftime("%Y-%m-%dT%H:%M:%S")}
    with _LOCK:
        _MEMORY[key] = entry
    _write_file(key, entry)
    return dict(labels[best]), {"mode": "measured", "config": labels[best], "key": key,
                                "timings_ms": timings}
