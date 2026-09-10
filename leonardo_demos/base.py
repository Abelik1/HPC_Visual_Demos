from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
import io, json, math, os, platform, socket, threading, time, traceback
from typing import Any, Callable, Dict
import numpy as np
from .backend import choose_backend


def _cpu_worker_count() -> int:
    """Return the CPU parallelism assigned to this process.

    SLURM_CPUS_PER_TASK is the allocation boundary on Leonardo.  The explicit
    override is useful for local benchmarking and for launchers that do not
    populate SLURM's environment.
    """
    raw = os.getenv("LEONARDO_DEMO_CPU_WORKERS") or os.getenv("SLURM_CPUS_PER_TASK")
    try:
        requested = int(raw) if raw else (os.cpu_count() or 1)
    except ValueError:
        requested = 1
    return max(1, min(requested, os.cpu_count() or requested))


def _environment_value(name: str):
    """Read an optional scheduler variable without recording empty values."""
    value = os.getenv(name)
    return value if value else None


def _decode_device_name(value: Any) -> str:
    if isinstance(value, bytes):
        return value.decode(errors="replace").rstrip("\0")
    return str(value)

@dataclass
class RunContext:
    run_dir: Path
    demo: str
    profile: str
    frames: int
    params: Dict[str, Any]
    backend_requested: str = "auto"
    backend_kind: str = "array"
    method: str = "default"
    timings_enabled: bool = False
    xp: Any = field(init=False, repr=False)
    backend_name: str = field(init=False)
    cpu_workers: int = field(init=False)
    _frame_pool: ThreadPoolExecutor | None = field(init=False, default=None, repr=False)
    _frame_futures: list[Future] = field(init=False, default_factory=list, repr=False)
    _compute_pool: ThreadPoolExecutor | None = field(init=False, default=None, repr=False)
    _timings: Dict[str, Dict[str, float]] = field(init=False, default_factory=dict, repr=False)
    _timing_lock: threading.Lock = field(init=False, default_factory=threading.Lock, repr=False)
    started: float = field(default_factory=time.time)
    _started_monotonic: float = field(default_factory=time.perf_counter, init=False, repr=False)

    def __post_init__(self):
        # On Windows the CUDA runtimes bundled by PyTorch and CuPy must be
        # loaded in a consistent order.  The neural wall uses PyTorch for its
        # batched training; importing it before the generic CuPy probe prevents
        # a first CUDA call from hanging when the GPU option is selected.
        if self.demo in {"neural_wall", "plasma_guardian"}:
            try:
                import torch  # noqa: F401
            except Exception:
                pass
        self.run_dir.mkdir(parents=True,exist_ok=True)
        (self.run_dir/'frames').mkdir(exist_ok=True)
        if self.backend_kind == "torch":
            # The neural wall owns its PyTorch device selection.  Requiring
            # CuPy here made a perfectly valid PyTorch/CUDA installation fail
            # before the demo could start.
            self.xp,self.backend_name=np,"torch (initialising)"
        elif self.backend_kind == "cpu":
            self.xp,self.backend_name=np,"numpy"
        else:
            self.xp,self.backend_name=choose_backend(self.backend_requested)
        self.cpu_workers=_cpu_worker_count()
        if self.backend_requested.lower() in {"hybrid", "cpu+gpu", "cpu_gpu"}:
            # GPU simulation and Pillow's JPEG encoder use different hardware.
            # Keep the queue short so the CPU overlaps the next CUDA update
            # without retaining a whole run's worth of images in memory.
            self._frame_pool=ThreadPoolExecutor(max_workers=2,
                                                 thread_name_prefix="frame-encode")
        self.write_meta({"status":"starting","demo":self.demo,"profile":self.profile,"frames":self.frames,"params":self.params,"backend":self.backend_name,"backend_requested":self.backend_requested,"method":self.method,"created":self.started,
                         "resources":self.resource_metadata()})

    def _cuda_devices(self):
        """Best-effort CUDA provenance; never make a completed run fail on it."""
        try:
            import cupy as cp
            devices=[]
            for index in range(int(cp.cuda.runtime.getDeviceCount())):
                properties=cp.cuda.runtime.getDeviceProperties(index)
                device={"visible_index":index,
                        "name":_decode_device_name(properties.get("name","unknown"))}
                for source,target in (("totalGlobalMem","memory_total_bytes"),
                                      ("major","compute_capability_major"),
                                      ("minor","compute_capability_minor")):
                    if properties.get(source) is not None:
                        device[target]=int(properties[source])
                devices.append(device)
            return {"api":"cupy","cuda_runtime_version":int(cp.cuda.runtime.runtimeGetVersion()),
                    "cuda_driver_version":int(cp.cuda.runtime.driverGetVersion()),
                    "visible_devices":devices,
                    "cuda_visible_devices":_environment_value("CUDA_VISIBLE_DEVICES")}
        except Exception as exc:
            return {"api":"cuda","probe_error":f"{type(exc).__name__}: {exc}",
                    "cuda_visible_devices":_environment_value("CUDA_VISIBLE_DEVICES")}

    def _torch_devices(self):
        try:
            import torch
            if not torch.cuda.is_available():
                return None
            return {"api":"torch","cuda_runtime_version":torch.version.cuda,
                    "visible_devices":[{"visible_index":index,
                                        "name":torch.cuda.get_device_name(index)}
                                       for index in range(torch.cuda.device_count())],
                    "cuda_visible_devices":_environment_value("CUDA_VISIBLE_DEVICES")}
        except Exception:
            return None

    def resource_metadata(self):
        """Describe the machine actually executing this run, not just its preset."""
        slurm={
            "job_id":_environment_value("SLURM_JOB_ID"),
            "job_name":_environment_value("SLURM_JOB_NAME"),
            "account":_environment_value("SLURM_JOB_ACCOUNT"),
            "partition":_environment_value("SLURM_JOB_PARTITION"),
            "node_list":_environment_value("SLURM_JOB_NODELIST"),
            "cpus_per_task":_environment_value("SLURM_CPUS_PER_TASK"),
            "tasks":_environment_value("SLURM_NTASKS"),
            "gpus_on_node":_environment_value("SLURM_GPUS_ON_NODE"),
        }
        result={"host":socket.gethostname(),"architecture":platform.machine(),
                "cpu_workers":self.cpu_workers,"host_cpu_count":os.cpu_count(),
                "slurm":{key:value for key,value in slurm.items() if value is not None}}
        # Keep the two legacy top-level fields so older dashboard consumers do
        # not need to know about the nested scheduler record.
        result["slurm_job_id"]=slurm["job_id"]
        result["slurm_partition"]=slurm["partition"]
        if self.backend_kind == "torch":
            gpu=self._torch_devices()
        elif self.xp is not np:
            gpu=self._cuda_devices()
        else:
            gpu=None
        if gpu is not None:
            result["gpu"]=gpu
        return result

    def elapsed_seconds(self) -> float:
        return max(0.0,time.perf_counter()-self._started_monotonic)

    def write_meta(self, update):
        p=self.run_dir/'meta.json'
        base={}
        if p.exists():
            try: base=json.loads(p.read_text())
            except Exception: pass
        base.update(update)
        p.write_text(json.dumps(base,indent=2))

    def frame_path(self,i): return self.run_dir/'frames'/f'frame_{i:04d}.jpg'
    def set_backend_name(self, name: str):
        self.backend_name=name
        self.write_meta({"backend":name})

    def _synchronize(self):
        if not self.timings_enabled:
            return
        if self.backend_kind == "torch":
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            except Exception:
                pass
        elif self.xp is not np:
            try: self.xp.cuda.get_current_stream().synchronize()
            except Exception: pass

    def _record_timing(self, name: str, seconds: float):
        if not self.timings_enabled:
            return
        with self._timing_lock:
            row=self._timings.setdefault(name,{"seconds":0.0,"count":0,"max_seconds":0.0})
            row["seconds"]+=float(seconds)
            row["count"]+=1
            row["max_seconds"]=max(row["max_seconds"],float(seconds))

    @contextmanager
    def stage(self, name: str, synchronize: bool = True):
        if not self.timings_enabled:
            yield
            return
        if synchronize: self._synchronize()
        started=time.perf_counter()
        try:
            yield
        finally:
            if synchronize: self._synchronize()
            self._record_timing(name,time.perf_counter()-started)

    def timing_summary(self):
        with self._timing_lock:
            return {name:{**row,"average_seconds":row["seconds"]/max(1,row["count"])}
                    for name,row in sorted(self._timings.items())}

    def save_frame(self, image, path: Path):
        """Write a frame, overlapping CPU encoding with CUDA work in hybrid mode."""
        if self._frame_pool is None:
            self._save_atomic(image,Path(path))
            return
        with self.stage("frame_copy",synchronize=False):
            copy=image.copy()
        self._frame_futures.append(self._frame_pool.submit(self._save_atomic,copy,Path(path)))
        if len(self._frame_futures)>2:
            self._frame_futures.pop(0).result()

    def _save_atomic(self,image,path: Path):
        path.parent.mkdir(parents=True,exist_ok=True)
        temporary=path.with_name(f'.{path.stem}.{time.time_ns()}.tmp{path.suffix}')
        buffer=io.BytesIO()
        with self.stage("jpeg_encode",synchronize=False):
            image.save(buffer,format="JPEG",quality=92)
        with self.stage("frame_write",synchronize=False):
            temporary.write_bytes(buffer.getvalue())
            temporary.replace(path)

    def flush_frames(self):
        while self._frame_futures:
            self._frame_futures.pop(0).result()
        if self._frame_pool is not None:
            self._frame_pool.shutdown(wait=True)
            self._frame_pool=None
    def parallel_slices(self, length: int, worker: Callable[[slice], Any],
                        min_items: int = 16_384):
        """Run NumPy work on disjoint slices using the allocated CPU cores.

        NumPy ufuncs release the GIL, so chunking the restricted N-body update
        gives genuine shared-memory CPU parallelism.  CuPy receives one whole
        slice: CUDA already supplies the parallel execution in that case.
        """
        if length <= 0:
            return []
        if self.xp is not np or self.cpu_workers <= 1 or length < min_items * 2:
            return [worker(slice(0,length))]
        workers=min(self.cpu_workers,max(1,math.ceil(length/min_items)))
        chunk=math.ceil(length/workers)
        if self._compute_pool is None:
            self._compute_pool=ThreadPoolExecutor(max_workers=self.cpu_workers,
                                                   thread_name_prefix="numpy-compute")
        futures=[self._compute_pool.submit(worker,slice(start,min(length,start+chunk)))
                 for start in range(0,length,chunk)]
        return [future.result() for future in futures]
    def shutdown_compute(self):
        if self._compute_pool is not None:
            self._compute_pool.shutdown(wait=True)
            self._compute_pool=None
    def write_status(self, frame, message="", overlay=None):
        """Publish live status and optional per-frame HTML-overlay values."""
        update={"status":"running","frame":frame,"message":message,
                "elapsed":self.elapsed_seconds()}
        if overlay is not None:
            clean={str(k):str(v) for k,v in dict(overlay).items()}
            update["overlay"]=clean
            directory=self.run_dir/'frame_data'; directory.mkdir(exist_ok=True)
            path=directory/f'frame_{int(frame):04d}.json'
            path.write_text(json.dumps({"message":message,"values":clean},ensure_ascii=False),encoding='utf-8')
            update["frame_data"]=True
        self.write_meta(update)
    def finish(self, reveal=None):
        self.flush_frames()
        self.shutdown_compute()
        elapsed=self.elapsed_seconds()
        self.write_meta({"status":"complete","frame":self.frames-1,"elapsed":elapsed,
                         "duration_seconds":elapsed,"finished":time.time(),
                         "resources":self.resource_metadata(),
                         "reveal":str(reveal.name) if reveal else None,
                         "timings":self.timing_summary()})
    def fail(self, exc):
        try: self.flush_frames()
        except Exception: pass
        try: self.shutdown_compute()
        except Exception: pass
        elapsed=self.elapsed_seconds()
        self.write_meta({"status":"failed","error":str(exc),"traceback":traceback.format_exc(),
                         "elapsed":elapsed,"duration_seconds":elapsed,"finished":time.time(),
                         "resources":self.resource_metadata()})

class Demo:
    id="base"
    title="Demo"
    backend_kind="array"
    supported_backends=("cpu","gpu","hybrid")
    default_method="default"
    methods=("default",)
    method_labels={"default":"Default solver"}
    method_descriptions={"default":"The demo's standard numerical method."}
    timing_methods={}
    def __init__(self, ctx, settings):
        self.ctx=ctx; self.settings=dict(settings)
        # The exhibition viewer can choose the actual width of an ensemble
        # independently of the machine profile.  Keep the override out of each
        # solver's public scientific parameters while still recording it in the
        # run metadata for exact replay.
        if ctx is not None and ctx.params.get('_parallel_count') is not None:
            count=max(1,int(ctx.params['_parallel_count']))
            if 'ensemble' in self.settings:
                self.settings['ensemble']=count
            if self.id == 'neural_wall':
                self.settings['networks']=count
        if ctx is not None and ctx.timings_enabled:
            self._install_timing_wrappers()
    def _install_timing_wrappers(self):
        for name,stage_name in self.timing_methods.items():
            original=getattr(self,name,None)
            if not callable(original):
                continue
            def measured(*args,_original=original,_stage=stage_name,**kwargs):
                with self.ctx.stage(_stage):
                    return _original(*args,**kwargs)
            setattr(self,name,measured)
    def run(self): raise NotImplementedError
