from __future__ import annotations
import os
import numpy as np


def choose_backend(requested: str = "auto"):
    """Return (xp, name). xp is numpy or cupy.

    The demos deliberately use a NumPy-like subset so the same source can run
    on CPU and CUDA. Set LEONARDO_DEMO_BACKEND=numpy/cupy to force a choice.
    """
    requested = os.getenv("LEONARDO_DEMO_BACKEND", requested).lower()
    if requested in {"numpy", "cpu"}:
        return np, "numpy"
    # ``hybrid`` leaves numerical arrays on CUDA while RunContext overlaps
    # JPEG encoding and disk I/O on CPU workers.
    hybrid = requested in {"hybrid", "cpu+gpu", "cpu_gpu"}
    if requested in {"cupy", "cuda", "gpu", "auto"} or hybrid:
        try:
            import cupy as cp
            device_count = cp.cuda.runtime.getDeviceCount()
            if device_count > 0:
                return cp, "cupy + CPU frame workers" if hybrid else "cupy"
            if requested != "auto":
                raise RuntimeError("GPU requested but CuPy reports no visible CUDA devices")
        except Exception as exc:
            if requested != "auto":
                raise RuntimeError(f"GPU requested but CUDA/CuPy is unavailable: {exc}") from exc
    return np, "numpy"


# Numerical precision of a run.  ``state`` is the dtype of the evolving
# solution (positions, velocities, lattice populations); ``compute`` is the
# dtype of the hot arithmetic inside the CUDA kernels.  ``mixed`` keeps an
# FP64 state and FP64 force accumulation around FP32 pair arithmetic: on
# consumer GPUs FP64 runs at 1/32–1/64 of FP32, while A100/H100/B200-class HPC
# GPUs run it at half rate, so the choice is hardware dependent.
PRECISIONS = {
    "fp32": {"state": np.float32, "compute": np.float32, "accumulate": np.float32,
             "label": "FP32"},
    "mixed": {"state": np.float64, "compute": np.float32, "accumulate": np.float64,
              "label": "mixed (FP64 state and sums, FP32 pair maths)"},
    "fp64": {"state": np.float64, "compute": np.float64, "accumulate": np.float64,
             "label": "FP64"},
}


def resolve_precision(name: str = "fp32"):
    key = str(name or "fp32").lower()
    aliases = {"single": "fp32", "float32": "fp32", "float": "fp32",
               "double": "fp64", "float64": "fp64"}
    key = aliases.get(key, key)
    if key not in PRECISIONS:
        raise ValueError(f"unknown precision {name!r}; choices: {', '.join(PRECISIONS)}")
    return key, PRECISIONS[key]


def to_numpy(a):
    try:
        import cupy as cp
        if isinstance(a, cp.ndarray):
            return cp.asnumpy(a)
    except Exception:
        pass
    return np.asarray(a)


def torch_device(requested: str = "auto"):
    """Device string for the PyTorch demos, honouring the same request.

    The array demos pick NumPy or CuPy through `choose_backend`, but the
    neural-network wall runs on PyTorch and used to ignore the request
    entirely, silently taking CUDA whenever it was present. Forcing CPU has to
    force it everywhere or the reported backend is a lie.
    """
    requested = os.getenv("LEONARDO_DEMO_BACKEND", requested).lower()
    if requested in {"numpy", "cpu"}:
        return "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    if requested in {"cupy", "cuda", "gpu", "hybrid", "cpu+gpu", "cpu_gpu"}:
        raise RuntimeError("GPU requested but no CUDA device is available to PyTorch")
    return "cpu"


def probe():
    """Report which compute backends this machine can actually use.

    The viewer needs this to offer a CPU/GPU choice honestly: an option that
    silently falls back to the CPU is worse than one shown as unavailable with
    the reason why.
    """
    out = {
        "cpu": {"available": True, "label": "CPU (NumPy)", "detail": f"NumPy {np.__version__}"},
        "gpu": {"available": False, "label": "GPU (CUDA)", "detail": "no CUDA array backend found"},
    }
    try:
        import cupy as cp
        n = cp.cuda.runtime.getDeviceCount()
        if n > 0:
            name = cp.cuda.runtime.getDeviceProperties(0)["name"]
            if isinstance(name, bytes):
                name = name.decode(errors="replace")
            out["gpu"] = {"available": True, "label": "GPU (CuPy)",
                          "detail": f"{n}x {name}"}
        else:
            out["gpu"]["detail"] = "CuPy installed but no CUDA device visible"
    except Exception as exc:
        out["gpu"]["detail"] = f"CuPy unavailable ({type(exc).__name__})"
    try:
        import torch
        if torch.cuda.is_available():
            out["torch"] = {"available": True, "label": "PyTorch CUDA",
                            "detail": torch.cuda.get_device_name(0)}
            if not out["gpu"]["available"]:
                out["gpu"] = {"available": True, "label": "GPU (CUDA)",
                              "detail": f"PyTorch only: {torch.cuda.get_device_name(0)}"}
        else:
            out["torch"] = {"available": True, "label": "PyTorch CPU",
                            "detail": f"torch {torch.__version__} (no CUDA)"}
    except Exception:
        out["torch"] = {"available": False, "label": "PyTorch",
                        "detail": "not installed - neural wall uses its surrogate"}
    out["hybrid"] = {
        "available": out["gpu"]["available"],
        "label": "GPU + CPU pipeline",
        "detail": ("CUDA solver with CPU frame encoding and disk I/O in parallel"
                   if out["gpu"]["available"] else "requires a working CUDA backend"),
    }
    return out
