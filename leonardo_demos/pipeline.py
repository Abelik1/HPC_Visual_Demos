"""Draw frames on the CPU cores in parallel while the GPU keeps simulating.

On the clusters most physics demos were drawing-bound: the wind tunnel on a
GB200 spent 22 s on physics and 213 s drawing, one frame after another on one
core, while the GPU waited and the node's other cores idled. A FramePipeline
separates the two:

* the solver loop does the physics, reduces what a frame needs to a small
  payload (an output-sized image, tracer positions, a few numbers) and submits
  it with a picklable, module-level ``draw(payload) -> PIL.Image``;
* a pool of worker *processes* (Python drawing holds the GIL, so threads would
  not scale) draws and JPEG-encodes frames concurrently;
* a frame is published to the viewer (``write_status``) only once it and every
  earlier frame are on disk, so a live viewer never requests a missing frame.

With one worker (a laptop run, or a 1-core allocation) it draws inline, in
order, exactly as before.
"""
from __future__ import annotations

import io
import multiprocessing
import os
import threading
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path


def _draw_and_save(draw, payload, path: str, quality: int, subsampling):
    started = time.perf_counter()
    image = draw(payload)
    drawn = time.perf_counter()
    buffer = io.BytesIO()
    options = {} if subsampling is None else {"subsampling": subsampling}
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, **options)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f".{target.stem}.{os.getpid()}.{time.time_ns()}.tmp{target.suffix}")
    temporary.write_bytes(buffer.getvalue())
    temporary.replace(target)
    return drawn - started, time.perf_counter() - drawn


class FramePipeline:
    def __init__(self, ctx, workers: int | None = None, max_in_flight: int | None = None):
        self.ctx = ctx
        self.workers = max(1, int(workers if workers is not None else ctx.cpu_workers))
        self.max_in_flight = max_in_flight or 2 * self.workers
        self._pool = None
        if self.workers > 1:
            # spawn: a forked child of a CUDA process must never touch CUDA.
            self._pool = ProcessPoolExecutor(max_workers=self.workers,
                                             mp_context=multiprocessing.get_context("spawn"))
        self._lock = threading.Lock()
        self._futures = {}
        self._done = {}
        self._next = 0
        self._error = None
        ctx.write_meta({"frame_pipeline": {"workers": self.workers, "mode": "processes" if self._pool else "inline"}})

    def submit(self, index: int, draw, payload, path, message: str = "", overlay=None):
        if self._error:
            raise self._error
        quality, subsampling = 92, getattr(self.ctx, "jpeg_subsampling", None)
        if self._pool is None:
            seconds = _draw_and_save(draw, payload, str(path), quality, subsampling)
            self._record(seconds)
            self.ctx.write_status(index, message, overlay)
            return
        while True:
            with self._lock:
                if len(self._futures) < self.max_in_flight:
                    break
                oldest = self._futures[min(self._futures)]
            with self.ctx.stage("frame_wait", synchronize=False):
                try:
                    oldest.result()
                except BaseException:
                    pass          # recorded by its callback, raised below
            if self._error:
                raise self._error
        future = self._pool.submit(_draw_and_save, draw, payload, str(path), quality, subsampling)
        with self._lock:
            self._futures[index] = future
        future.add_done_callback(lambda f, i=index, m=message, o=overlay: self._finished(i, f, m, o))

    def _record(self, seconds):
        draw_s, encode_s = seconds
        self.ctx._record_timing("render", draw_s)
        self.ctx._record_timing("jpeg_encode", encode_s)

    def _finished(self, index, future, message, overlay):
        with self._lock:
            self._futures.pop(index, None)
            try:
                self._record(future.result())
            except BaseException as error:  # surfaced on the next submit/close
                self._error = error
                return
            self._done[index] = (message, overlay)
            while self._next in self._done:
                m, o = self._done.pop(self._next)
                self.ctx.write_status(self._next, m, o)
                self._next += 1

    def close(self):
        if self._pool is not None:
            for index in sorted(list(self._futures)):
                future = self._futures.get(index)
                if future is not None:
                    try:
                        future.result()
                    except BaseException:
                        pass
            self._pool.shutdown(wait=True)
            self._pool = None
        if self._error:
            raise self._error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if exc[0] is not None and self._pool is not None:
            self._pool.shutdown(wait=False, cancel_futures=True)
            self._pool = None
            return False
        self.close()
        return False
