"""Several GPUs of one node, driven from one process.

A supercomputer node has four GPUs (A100 on Leonardo's Booster, GB200 on
Discoverer), joined by NVLink.  The demos spread one simulation over them the
way MUrB's four-GPU backend does, only without MPI: each GPU gets a thread,
every thread issues its CuPy work inside ``with cupy.cuda.Device(id)``, and
CuPy releases the GIL while kernels run, so the four GPUs compute at once.
Data moves between them with device-to-device copies (NVLink where present).

The decomposition code is written once, here, and used by every demo:

* ``split``       balanced contiguous index ranges, one per GPU;
* ``Devices.run`` call ``fn(i, part)`` on every GPU concurrently;
* ``Devices.to`` / ``gather`` / ``allgather`` / ``halo_exchange``
                  move arrays between GPUs.

``Devices`` also accepts NumPy and repeated device ids, so the same
decomposition runs on a laptop in the unit tests: two "GPUs" that are really
one device (or the host) must give the same answer as one.
"""
from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from typing import Any, Callable, Sequence

import numpy as np


def split(length: int, parts: int) -> list[slice]:
    """``parts`` contiguous slices covering ``range(length)``, sizes within one."""
    parts = max(1, min(int(parts), max(1, int(length))))
    base, extra = divmod(int(length), parts)
    out, start = [], 0
    for k in range(parts):
        stop = start + base + (1 if k < extra else 0)
        out.append(slice(start, stop))
        start = stop
    return out


def requested_gpus(params: dict | None = None) -> int:
    """GPUs a run asked for: the ``_gpus`` parameter, else LEONARDO_DEMO_GPUS, else 1."""
    raw = (params or {}).get("_gpus") or os.getenv("LEONARDO_DEMO_GPUS") or 1
    try:
        return max(1, int(float(raw)))
    except (TypeError, ValueError):
        return 1


class Devices:
    """The GPUs one simulation is spread over.

    ``ids`` are CUDA device ordinals as this process sees them (Slurm's
    ``--gres=gpu:N`` already restricts visibility to the allocated ones).
    With ``xp`` = NumPy the ids only count "parts"; everything stays on the host.
    """

    def __init__(self, xp, ids: Sequence[int]):
        self.xp = xp
        self.ids = list(ids) or [0]
        self._pool = None
        if self.cupy:
            self._enable_peer_access()

    def _enable_peer_access(self):
        """Let every pair of these GPUs copy directly (NVLink) rather than via the host.

        Without it cudaMemcpyPeer stages each copy through host memory, which
        on a GB200 or A100 node is several times slower than NVLink.
        """
        cp = self.xp
        distinct = sorted(set(self.ids))
        for a in distinct:
            for b in distinct:
                if a == b or not cp.cuda.runtime.deviceCanAccessPeer(a, b):
                    continue
                with cp.cuda.Device(a):
                    try:
                        cp.cuda.runtime.deviceEnablePeerAccess(b)
                    except cp.cuda.runtime.CUDARuntimeError:
                        pass                                  # already enabled in this process

    @property
    def count(self) -> int:
        return len(self.ids)

    @property
    def cupy(self) -> bool:
        return self.xp is not np

    def device(self, i: int):
        """Context manager making part ``i``'s GPU current (no-op on NumPy)."""
        return self.xp.cuda.Device(self.ids[i]) if self.cupy else nullcontext()

    def synchronize(self):
        if not self.cupy:
            return
        for i in range(self.count):
            with self.device(i):
                self.xp.cuda.Device().synchronize()

    def run(self, fn: Callable[[int], Any]) -> list:
        """``fn(i)`` on every part concurrently, each on its own GPU; results in order."""
        if self.count == 1:
            with self.device(0):
                return [fn(0)]
        if self._pool is None:
            self._pool = ThreadPoolExecutor(max_workers=self.count, thread_name_prefix="gpu")

        def call(i):
            with self.device(i):
                value = fn(i)
                if self.cupy:
                    self.xp.cuda.get_current_stream().synchronize()
                return value

        return [f.result() for f in [self._pool.submit(call, i) for i in range(self.count)]]

    def to(self, array, i: int):
        """A copy of ``array`` on part ``i``'s GPU (peer copy over NVLink if possible)."""
        if not self.cupy:
            return np.array(array, copy=True)
        cp = self.xp
        dst = self.ids[i]
        if not isinstance(array, cp.ndarray):
            with cp.cuda.Device(dst):
                return cp.asarray(array)
        src = array.device.id
        with cp.cuda.Device(src):
            array = cp.ascontiguousarray(array)
        with cp.cuda.Device(dst):
            out = cp.empty(array.shape, dtype=array.dtype)
            if src == dst:
                out[...] = array
            else:
                cp.cuda.runtime.memcpyPeer(out.data.ptr, dst, array.data.ptr, src, array.nbytes)
        return out

    def scatter(self, array, axis: int = 0) -> list:
        """Split ``array`` along ``axis`` into one piece per part, each on its GPU."""
        slices = split(array.shape[axis], self.count)
        index = [slice(None)] * array.ndim
        pieces = []
        for i, s in enumerate(slices):
            index[axis] = s
            pieces.append(self.to(array[tuple(index)], i))
        return pieces

    def gather(self, parts: Sequence, axis: int = 0, on: int = 0):
        """Concatenate the parts (one per GPU) along ``axis`` onto part ``on``'s GPU.

        Along axis 0 every part is copied straight into its place in one
        output array (one peer copy each, no second pass over the data).
        """
        if not self.cupy or axis != 0:
            moved = [self.to(p, on) for p in parts]
            with self.device(on):
                return self.xp.concatenate(moved, axis=axis)
        cp, dst = self.xp, self.ids[on]
        shape = (sum(int(p.shape[0]) for p in parts),) + tuple(parts[0].shape[1:])
        with cp.cuda.Device(dst):
            out = cp.empty(shape, dtype=parts[0].dtype)
        row = 0
        for p in parts:
            src = p.device.id
            with cp.cuda.Device(src):
                p = cp.ascontiguousarray(p)
            n = int(p.shape[0])
            target = out[row:row + n]
            with cp.cuda.Device(dst):
                if src == dst:
                    target[...] = p
                else:
                    cp.cuda.runtime.memcpyPeer(target.data.ptr, dst, p.data.ptr, src, p.nbytes)
            row += n
        return out

    def allgather(self, parts: Sequence, axis: int = 0) -> list:
        """Every part receives the concatenation of all parts."""
        whole = self.gather(parts, axis=axis, on=0)
        return [whole] + [self.to(whole, i) for i in range(1, self.count)]

    def halo_exchange(self, strips: Sequence, halo: int = 1, axis: int = 0, periodic: bool = True):
        """Fill each strip's ghost rows from its neighbours' edge rows, in place.

        Every strip carries ``halo`` ghost rows at both ends of ``axis``:
        ``[ghost | own rows | ghost]``.  After the exchange a strip's leading
        ghosts hold the last own rows of the strip before it and its trailing
        ghosts the first own rows of the strip after it (wrapping round when
        ``periodic``), exactly what a whole-domain periodic ``roll`` would see.
        """
        n = len(strips)

        def rows(a, start, stop):
            index = [slice(None)] * a.ndim
            index[axis] = slice(start, stop)
            return tuple(index)

        edges = []
        for s in strips:
            size = s.shape[axis]
            edges.append((s[rows(s, halo, 2 * halo)], s[rows(s, size - 2 * halo, size - halo)]))
        for k, s in enumerate(strips):
            size = s.shape[axis]
            before, after = k - 1, k + 1
            if periodic:
                before %= n
                after %= n
            with self.device(k):
                if 0 <= before < n:
                    s[rows(s, 0, halo)] = self.to(edges[before][1], k)
                if 0 <= after < n:
                    s[rows(s, size - halo, size)] = self.to(edges[after][0], k)

    def close(self):
        if self._pool is not None:
            self._pool.shutdown(wait=True)
            self._pool = None


def devices_for(ctx, ids: Sequence[int] | None = None) -> Devices:
    """The GPUs a run may use: ``ids`` if given, else the first ``ctx.gpus`` visible ones.

    A CPU run gets a single host "part", so callers fall back to their
    one-device path.
    """
    if ids is not None:
        return Devices(ctx.xp, ids)
    if not ctx.on_gpu:
        return Devices(np, [0])
    visible = int(ctx.xp.cuda.runtime.getDeviceCount())
    return Devices(ctx.xp, range(max(1, min(ctx.gpus, visible))))
