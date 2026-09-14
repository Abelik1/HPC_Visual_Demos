from __future__ import annotations
import atexit, base64, binascii, hashlib, io, json, math, multiprocessing, os, threading, time, uuid
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Literal
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from PIL import Image, UnidentifiedImageError
from run_demo import run, load_profiles, load_specs, profile_setting_schema, canonical_profile
from leonardo_demos.registry import DEMOS
from leonardo_demos.backend import probe as probe_backends
from leonardo_demos.neuroevo import BrainError, brain_catalogue, validate_brains

ROOT=Path(__file__).resolve().parent; RUNS=ROOT/'runs'; RUNS.mkdir(exist_ok=True)
app=FastAPI(title='Leonardo Visual Demos')
app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')
app.mount('/runs',StaticFiles(directory=RUNS),name='runs')
app.mount('/benchmarks',StaticFiles(directory=ROOT/'benchmarks',html=True),name='benchmarks')
# Default pictures for the image-compression demo; drop more files in here.
COMPRESSION_IMAGES=ROOT/'data'/'compression_images'; COMPRESSION_IMAGES.mkdir(parents=True,exist_ok=True)
app.mount('/compression_images',StaticFiles(directory=COMPRESSION_IMAGES),name='compression_images')

class RunReq(BaseModel):
    profile: str = 'local'
    # Desktop GPU runs are deliberately allowed to be longer than the default
    # exhibition loop.  The viewer streams frames as they arrive, so 300-frame
    # living-mathematics runs are valid rather than a request-validation error.
    frames: int = Field(default=70, ge=1, le=600)
    params: dict[str, float] = Field(default_factory=dict)
    settings: dict[str, float] = Field(default_factory=dict)
    backend: Literal['auto', 'numpy', 'cpu', 'cupy', 'cuda', 'gpu', 'hybrid'] = 'auto'
    method: str = Field(default='default', min_length=1, max_length=64)
    precision: Literal['fp32', 'mixed', 'fp64'] = 'fp32'
    numerical_substeps: int | None = Field(default=None, ge=1, le=32)
    target_image: str | None = Field(default=None, max_length=400_000)
    parallel_count: int | None = Field(default=None, ge=1, le=64)
    obstacle_grid: list[list[int]] | None = None
    # Visitor-built network for the AI game demos; validated against the
    # demo's block catalogue in config/demo_specs.json.
    brain: dict | None = None
    ghosts: list[str] | None = Field(default=None, max_length=5)

def save_target_image(data_url: str, destination: Path) -> None:
    """Validate a canvas PNG and save a bounded RGB target image."""
    prefix='data:image/png;base64,'
    if not data_url.startswith(prefix):
        raise HTTPException(422, 'target image must be a PNG drawing')
    try:
        raw=base64.b64decode(data_url[len(prefix):], validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            if image.width < 1 or image.height < 1 or image.width*image.height > 1_000_000:
                raise HTTPException(422, 'target image dimensions are invalid')
            image.convert('RGB').save(destination, format='PNG')
    except HTTPException:
        raise
    except (binascii.Error, OSError, UnidentifiedImageError, ValueError):
        raise HTTPException(422, 'target image could not be read')

@app.get('/',response_class=HTMLResponse)
def index(): return (ROOT/'web/index.html').read_text(encoding='utf-8')

# Demo-day viewer.  Same APIs and the same saved runs as the full dashboard,
# but a walk-up surface: a few large controls, no toggleable overlay dock, and
# every advanced knob behind one presenter panel.
@app.get('/demo',response_class=HTMLResponse)
def demo_mode(): return (ROOT/'web/demo.html').read_text(encoding='utf-8')

@app.get('/api/compression_images')
def compression_images():
    files=sorted(f for f in COMPRESSION_IMAGES.iterdir()
                 if f.is_file() and f.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.gif'})
    return [{'name':f.stem.replace('_',' ').replace('-',' ').strip().capitalize(),
             'url':f'/compression_images/{f.name}'} for f in files]

@app.get('/api/specs')
def specs():
    return {'demos':load_specs(),'profiles':load_profiles(),'available':list(DEMOS),
            'profile_setting_schema':profile_setting_schema(),
            'backends':probe_backends(),
            'capabilities':{name:{'backends':list(cls.supported_backends),
                                  'backend_kind':cls.backend_kind,
                                  'methods':list(cls.methods),
                                  'default_method':cls.default_method,
                                  'method_labels':dict(cls.method_labels),
                                  'method_descriptions':dict(cls.method_descriptions),
                                  'precisions':list(cls.precisions)}
                            for name,cls in DEMOS.items()}}

@app.get('/api/runs')
def list_runs(demo:str|None=None,limit:int=120):
    """Saved runs, newest first, with everything the viewer needs to replay one.

    Every run already persists to runs/<id>/; this simply makes them reachable
    so a finished simulation can be replayed instead of recomputed. That is the
    same path an exhibition uses for its playback fallback.
    """
    out=[]; favourites=set(load_library()['favourites'])
    for d in sorted(RUNS.glob('*'),key=lambda p:p.stat().st_mtime,reverse=True):
        if not d.is_dir() or d.name.startswith('_'): continue
        try: meta=json.loads((d/'meta.json').read_text(encoding="utf-8"))
        except Exception: continue
        if demo and meta.get('demo')!=demo: continue
        frames=sorted((d/'frames').glob('frame_*.jpg')) if (d/'frames').is_dir() else []
        if not frames: continue
        galaxy3d=meta.get('galaxy3d_view')
        if not (isinstance(galaxy3d,dict) and
                (d/str(galaxy3d.get('folder','_missing'))).is_dir()):
            galaxy3d=None
        fusion=meta.get('fusion_view')
        if isinstance(fusion,dict):
            if not (d/str(fusion.get('folder','_missing'))).is_dir(): fusion=None
        elif not (isinstance(fusion,str) and (d/fusion).exists()):
            fusion=None
        arena=meta.get('arena_view')
        if not (isinstance(arena,dict) and (d/str(arena.get('folder','_missing'))).is_dir()):
            arena=None
        out.append({
            'id':d.name,
            'demo':meta.get('demo'),
            'profile':canonical_profile(meta.get('profile')),
            'backend':meta.get('backend'),
            'method':meta.get('method'),
            # Runs recorded before precision selection existed were FP32.
            'precision':meta.get('precision','fp32'),
            'kernels':meta.get('kernels',{}),
            'status':meta.get('status'),
            'params':meta.get('params',{}),
            'settings':meta.get('settings',{}),
            'settings_override':meta.get('settings_override',{}),
            'resources':meta.get('resources',{}),
            'frames':len(frames),
            'elapsed':meta.get('elapsed'),
            'created':meta.get('created') or d.stat().st_mtime,
            'thumb':f'/runs/{d.name}/{"reveal.jpg" if (d/"reveal.jpg").exists() else "frames/"+frames[-1].name}',
            'has_reveal':(d/'reveal.jpg').exists(),
            'zoom':meta.get('zoom'),
            'fusion_view':fusion,
            'galaxy3d_view':galaxy3d,
            'arena_view':arena,
            'summary':meta.get('summary'),
            'lab':meta.get('lab') if (d/'checkpoints').is_dir() else None,
            'view_modes':meta.get('view_modes'),
            'default_view_mode':meta.get('default_view_mode'),
            'overlays':meta.get('overlays'),
            'frame_data':bool(meta.get('frame_data')),
            'favourite':d.name in favourites,
        })
        if len(out)>=max(1,min(500,limit)): break
    return out

# ---- run library: favourites and the demo-day showcase ------------------
# Kept beside the runs themselves (runs/ is not versioned, and a run id only
# means something on the machine that produced it). The leading underscore
# keeps list_runs from mistaking it for a run.
LIBRARY=RUNS/'_library.json'
LIBRARY_LOCK=threading.Lock()
SHOWCASE_MAX=3

def load_library():
    try: data=json.loads(LIBRARY.read_text(encoding='utf-8'))
    except (OSError,ValueError): data={}
    favourites=[r for r in data.get('favourites',[]) if isinstance(r,str)]
    showcase={k:[r for r in v if isinstance(r,str)][:SHOWCASE_MAX]
              for k,v in (data.get('showcase') or {}).items() if isinstance(v,list)}
    return {'favourites':favourites,'showcase':showcase}

def save_library(data):
    tmp=LIBRARY.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2),encoding='utf-8'); tmp.replace(LIBRARY)

def run_demo_name(rid:str):
    """The demo a saved run belongs to, or None if rid is not a real run."""
    rd=(RUNS/rid).resolve()
    if rd.parent!=RUNS.resolve() or rid.startswith('_'): return None
    try: return json.loads((rd/'meta.json').read_text(encoding="utf-8")).get('demo')
    except (OSError,ValueError): return None

class FavouriteReq(BaseModel):
    favourite: bool

class ShowcaseReq(BaseModel):
    runs: list[str] = Field(default_factory=list, max_length=SHOWCASE_MAX)

@app.get('/api/library')
def library():
    data=load_library()
    # Drop ids whose run directory has since been deleted.
    data['favourites']=[r for r in data['favourites'] if run_demo_name(r)]
    data['showcase']={k:[r for r in v if run_demo_name(r)==k] for k,v in data['showcase'].items()}
    return data

@app.post('/api/runs/{rid}/favourite')
def set_favourite(rid:str,req:FavouriteReq):
    if not run_demo_name(rid): raise HTTPException(404,'unknown run')
    with LIBRARY_LOCK:
        data=load_library(); favs=[r for r in data['favourites'] if r!=rid]
        if req.favourite: favs.insert(0,rid)
        data['favourites']=favs; save_library(data)
    return {'id':rid,'favourite':req.favourite}

@app.put('/api/showcase/{demo}')
def set_showcase(demo:str,req:ShowcaseReq):
    if demo not in DEMOS: raise HTTPException(404,'unknown demo')
    runs=list(dict.fromkeys(req.runs))
    for rid in runs:
        if run_demo_name(rid)!=demo: raise HTTPException(422,f'{rid} is not a saved {demo} run')
    with LIBRARY_LOCK:
        data=load_library()
        if runs: data['showcase'][demo]=runs
        else: data['showcase'].pop(demo,None)
        save_library(data)
    return {'demo':demo,'runs':runs}

@app.post('/api/run/{demo}')
def start(demo:str,req:RunReq):
    if demo not in DEMOS: raise HTTPException(404,'unknown demo')
    requested={'numpy':'cpu','cuda':'gpu','cupy':'gpu'}.get(req.backend,req.backend)
    if requested!='auto' and requested not in DEMOS[demo].supported_backends:
        allowed=', '.join(DEMOS[demo].supported_backends)
        raise HTTPException(422,f'{demo} supports these compute modes: {allowed}')
    req.profile=canonical_profile(req.profile)
    if req.profile not in load_profiles():
        raise HTTPException(422, 'unknown profile')
    method=DEMOS[demo].default_method if req.method=='default' else req.method
    if method not in DEMOS[demo].methods:
        allowed=', '.join(DEMOS[demo].methods)
        raise HTTPException(422,f'{demo} supports these solvers: {allowed}')
    if req.precision not in DEMOS[demo].precisions:
        allowed=', '.join(DEMOS[demo].precisions)
        raise HTTPException(422,f'{demo} supports these precisions: {allowed}')
    spec_params=load_specs()[demo]['params']
    unknown=set(req.params)-set(spec_params)
    if unknown:
        raise HTTPException(422, f"unknown parameter(s): {', '.join(sorted(unknown))}")
    for name,value in req.params.items():
        limits=spec_params[name]
        if not math.isfinite(value) or not limits['min'] <= value <= limits['max']:
            raise HTTPException(422, f"{name} must be between {limits['min']} and {limits['max']}")
    setting_limits=profile_setting_schema().get(demo,{})
    unknown_settings=set(req.settings)-set(setting_limits)
    if unknown_settings:
        raise HTTPException(422, f"unknown profile setting(s): {', '.join(sorted(unknown_settings))}")
    clean_settings={}
    for name,value in req.settings.items():
        limits=setting_limits[name]
        if not math.isfinite(value) or not limits['min'] <= value <= limits['max']:
            raise HTTPException(422, f"{name} must be between {limits['min']} and {limits['max']}")
        if limits['type']=='integer' and not float(value).is_integer():
            raise HTTPException(422, f'{name} must be a whole number')
        clean_settings[name]=int(value) if limits['type']=='integer' else float(value)
    rid=f'{demo}_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:5]}'; rd=RUNS/rid
    params=dict(req.params)
    if req.parallel_count is not None:
        if req.parallel_count not in {1,4,9,16,25,36,49,64}:
            raise HTTPException(422, 'parallel count must be 1 or form a square reveal grid: 4, 9, 16, 25, 36, 49 or 64')
        params['_parallel_count']=int(req.parallel_count)
    if req.obstacle_grid is not None:
        if demo != 'fluid':
            raise HTTPException(422, 'a custom obstacle grid is only supported by the wind tunnel')
        rows=req.obstacle_grid
        width=len(rows[0]) if rows else 0
        if not (6 <= len(rows) <= 24 and 8 <= width <= 40 and
                all(len(row)==width for row in rows) and
                all(cell in (0,1) for row in rows for cell in row)):
            raise HTTPException(422, 'obstacle grid must be a rectangular 0/1 grid between 8×6 and 40×24')
        params['_obstacle_grid']=rows
    if demo=='fluid' and int(round(req.params.get('obstacle',0)))==3 and not any(
            cell for row in (req.obstacle_grid or []) for cell in row):
        raise HTTPException(422, 'draw at least one block for a custom-shape-only wind tunnel')
    if req.brain is not None:
        catalogue=brain_catalogue(load_specs(),demo)
        if catalogue is None:
            raise HTTPException(422, 'a custom brain is only supported by the AI game demos')
        try: params['_brain']=validate_brains(req.brain,catalogue)
        except BrainError as error: raise HTTPException(422, str(error))
    if req.ghosts:
        if demo != 'neuro_racers':
            raise HTTPException(422, 'ghost races are only supported by Neuro-Racers')
        ghost_dirs=[]
        for ghost in req.ghosts:
            gd=(RUNS/ghost).resolve()
            if RUNS.resolve() not in gd.parents or not (gd/'champion.npz').exists():
                raise HTTPException(422, f'ghost run {ghost} has no saved champion')
            ghost_dirs.append(str(gd))
        params['_ghosts']=ghost_dirs
    if req.target_image is not None:
        if demo != 'neural_wall':
            raise HTTPException(422, 'a custom drawing is only supported by the neural-network wall')
        target_path=rd/'target.png'
        rd.mkdir(parents=True, exist_ok=True)
        save_target_image(req.target_image, target_path)
        params['_target_path']=str(target_path)
    kwargs=dict(demo=demo,profile=req.profile,frames=req.frames,params=params,backend=req.backend,
                run_dir=rd,method=method,numerical_substeps=req.numerical_substeps,
                settings_override=clean_settings,precision=req.precision)
    threading.Thread(target=_supervise_run,args=(rid,rd,kwargs),daemon=True).start(); return {'id':rid}

# Every simulation runs in its own spawned process rather than a thread of the
# viewer. PyTorch and CuPy each bundle a different build of cublasLt64_13.dll;
# Windows loads one DLL per name per process, so after a CuPy demo had used
# CuPy's copy, the next neural-wall run called into it through PyTorch and died
# with an access violation that took the whole viewer down. A fresh process
# per run keeps the libraries apart, returns GPU memory when the run ends, and
# turns any native crash into one failed run instead of a dead server.
ACTIVE_RUNS={}
ACTIVE_RUNS_LOCK=threading.Lock()

def _supervise_run(rid,rd,kwargs):
    # Not daemonic: crystal growth starts its own worker processes.
    proc=multiprocessing.get_context('spawn').Process(target=run,kwargs=kwargs,name=f'run-{rid}')
    proc.start()
    with ACTIVE_RUNS_LOCK: ACTIVE_RUNS[rid]=proc
    proc.join()
    with ACTIVE_RUNS_LOCK: ACTIVE_RUNS.pop(rid,None)
    if proc.exitcode: record_crashed_run(rd,proc.exitcode)

def record_crashed_run(rd,exitcode):
    """Mark a run whose process ended before reporting its own outcome."""
    p=rd/'meta.json'
    try: meta=json.loads(p.read_text(encoding="utf-8"))
    except (OSError,ValueError): meta={}
    if meta.get('status') in ('complete','failed'): return
    # multiprocessing reports a terminated process as a negative code on every OS.
    if exitcode<0: reason='terminated' if os.name=='nt' else f'signal {-exitcode}'
    else: reason=f'exit code 0x{exitcode:08X}'
    hint=' — a native crash inside a compute library' if exitcode & 0xFFFFFFFF==0xC0000005 else ''
    meta.update(status='failed',error=f'The simulation process ended without finishing ({reason}{hint}). '
                                      'The viewer is still running; you can start another run.')
    rd.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(meta,indent=2))

@atexit.register
def _stop_active_runs():
    # Registered after multiprocessing's own exit hook, so it runs first:
    # closing the viewer ends running simulations instead of waiting for them.
    with ACTIVE_RUNS_LOCK: procs=list(ACTIVE_RUNS.values())
    for proc in procs:
        if proc.is_alive(): proc.terminate()

ZOOMABLE={'crystal'}

# Deep-zoom levels are integers on a log2 ladder: level L means the model is
# rendered at 2**L tiles across. Quantising this way is what keeps magnification
# sharp. Rendering one image for the exact current view meant that between the
# render finishing and the next one arriving the browser was scaling a stale
# bitmap, so anything past a few hundred times looked like a blurred photograph.
# The renderer is deliberately finite.  Individual runs choose a lower cap in
# their zoom manifest; this hard ceiling keeps request indices and resource
# use bounded even for malformed requests.
MAX_ZOOM_LEVEL=32
# Tile generation is mostly Python tree traversal.  It needs processes (not
# threads) to use several CPU cores despite the GIL.  Eight workers give a
# fast first fill on a 64-GB desktop without reserving the whole machine.
TILE_WORKERS=max(2,min(8,(os.cpu_count() or 4)//2))
# Spawn prevents a CUDA context in the viewer process being inherited by the
# CPU geometry workers on Linux/Leonardo.
TILE_POOL=(ProcessPoolExecutor(max_workers=TILE_WORKERS,
                               mp_context=multiprocessing.get_context('spawn'))
           if multiprocessing.current_process().name=='MainProcess' else None)
TILE_LOCK=threading.RLock()
TILE_GPU_LOCK=threading.Lock()
TILE_PENDING={}
TILE_MEMORY=OrderedDict()
TILE_MEMORY_BYTES=0
try:
    TILE_MEMORY_LIMIT_MB=max(128,min(4096,int(os.getenv('LEONARDO_TILE_CACHE_MB','1024'))))
except ValueError:
    TILE_MEMORY_LIMIT_MB=1024
TILE_MEMORY_LIMIT=TILE_MEMORY_LIMIT_MB*1024*1024
DYNAMIC_TILE_VERSION=3
# A viewport is one coherent image, rather than a collection of independently
# budgeted branch traversals.  Bump this when its renderer changes.
DYNAMIC_VIEW_VERSION=2


def _deepzoom_raster_backend():
    # Spawned CPU geometry workers must never create a CUDA context.  Only the
    # serving process owns the optional GPU rasterizer.
    if multiprocessing.current_process().name != 'MainProcess':
        return 'cpu'
    requested=os.getenv('LEONARDO_DEEPZOOM_BACKEND','auto').lower()
    if requested in {'cpu','numpy'}: return 'cpu'
    # This CUDA path is available for profiling and for future larger vector
    # workloads, but the local benchmark selects CPU in auto mode: Pillow's
    # scanline renderer beats per-segment atomic writes for 256/512 px tiles.
    if requested=='auto': return 'cpu'
    try:
        from leonardo_demos.crystal_growth import gpu_raster_available
        if gpu_raster_available(): return 'gpu'
    except Exception:
        pass
    return 'cpu'


TILE_RASTER_BACKEND=_deepzoom_raster_backend()


def _crystal_geometry(meta,cx,cy,span,size_px,budget):
    from leonardo_demos.crystal_growth import generate, MODE_NAMES
    p=meta.get('params',{}) or {}
    return generate(cx,cy,span,size_px=size_px,
                    symmetry=int(p.get('symmetry',6)),
                    mode=MODE_NAMES[int(p.get('mode',0))%len(MODE_NAMES)],
                    undercooling=float(p.get('undercooling',.75)),
                    anisotropy=float(p.get('anisotropy',.055)),
                    seed=int(p.get('seed',3)),
                    max_segments=budget)


def _stable_crystal_geometry(meta,cx,cy,span,size_px,detail_depth,detail_max):
    from leonardo_demos.crystal_growth import generate_stable, MODE_NAMES
    p=meta.get('params',{}) or {}
    return generate_stable(cx,cy,span,size_px=size_px,
                           symmetry=int(p.get('symmetry',6)),
                           mode=MODE_NAMES[int(p.get('mode',0))%len(MODE_NAMES)],
                           undercooling=float(p.get('undercooling',.75)),
                           anisotropy=float(p.get('anisotropy',.055)),
                           seed=int(p.get('seed',3)),
                           detail_depth=detail_depth,max_depth=detail_max)


def _render_crystal_tile_jpeg(meta,tcx,tcy,sub,tile,budget):
    """CPU-process worker: make one deterministic native-resolution tile."""
    from leonardo_demos.crystal_growth import render_window
    geometry=_crystal_geometry(meta,tcx,tcy,sub,tile,budget)
    image=render_window(geometry,tcx,tcy,sub,size=(tile,tile),progress=1.0,supersample=2)
    buf=io.BytesIO(); image.save(buf,'JPEG',quality=88)
    return buf.getvalue()


def _generate_crystal_tile(meta,tcx,tcy,sub,tile,budget):
    """CPU-process worker for the hybrid CPU geometry -> GPU raster pipeline."""
    return _crystal_geometry(meta,tcx,tcy,sub,tile,budget)


def _render_crystal_view_jpeg(meta,cx,cy,span,width,height,detail_depth,detail_max):
    """CPU worker for one coherent deep-zoom viewport image."""
    from leonardo_demos.crystal_growth import render_window
    geometry=_stable_crystal_geometry(meta,cx,cy,span,max(width,height),
                                      detail_depth,detail_max)
    image=render_window(geometry,cx,cy,span,size=(width,height),progress=1.0,
                        supersample=1)
    buf=io.BytesIO(); image.save(buf,'JPEG',quality=90)
    return buf.getvalue()


def _memory_tile(key):
    with TILE_LOCK:
        data=TILE_MEMORY.get(key)
        if data is not None:
            TILE_MEMORY.move_to_end(key)
        return data


def _remember_tile(key,data):
    global TILE_MEMORY_BYTES
    with TILE_LOCK:
        old=TILE_MEMORY.pop(key,None)
        if old is not None: TILE_MEMORY_BYTES-=len(old)
        TILE_MEMORY[key]=data; TILE_MEMORY_BYTES+=len(data)
        while TILE_MEMORY and TILE_MEMORY_BYTES>TILE_MEMORY_LIMIT:
            _,evicted=TILE_MEMORY.popitem(last=False)
            TILE_MEMORY_BYTES-=len(evicted)


def _tile_bytes(key,cached,meta,tcx,tcy,sub,tile,budget):
    """Return a tile from RAM/disk or join exactly one in-flight render."""
    hit=_memory_tile(key)
    if hit is not None: return hit
    with TILE_LOCK:
        future=TILE_PENDING.get(key)
        if future is None:
            if cached.exists():
                data=cached.read_bytes()
                _remember_tile(key,data)
                return data
            worker=_generate_crystal_tile if TILE_RASTER_BACKEND=='gpu' else _render_crystal_tile_jpeg
            if TILE_POOL is None: raise RuntimeError('tile pool is unavailable in a worker process')
            future=TILE_POOL.submit(worker,meta,tcx,tcy,sub,tile,budget)
            TILE_PENDING[key]=future
    try:
        rendered=future.result()
        if TILE_RASTER_BACKEND=='gpu':
            from leonardo_demos.crystal_growth import render_window
            # One CUDA device renders the prepared tiles in a well-defined
            # stream while all CPU workers are already building later trees.
            with TILE_GPU_LOCK:
                image=render_window(rendered,tcx,tcy,sub,size=(tile,tile),progress=1.0,
                                    supersample=2,backend='gpu')
            buf=io.BytesIO(); image.save(buf,'JPEG',quality=88); data=buf.getvalue()
        else:
            data=rendered
        cached.parent.mkdir(parents=True,exist_ok=True)
        temporary=cached.with_name(f'.{cached.stem}.{time.time_ns()}.tmp{cached.suffix}')
        temporary.write_bytes(data); temporary.replace(cached)
        _remember_tile(key,data)
        return data
    finally:
        with TILE_LOCK:
            if TILE_PENDING.get(key) is future: TILE_PENDING.pop(key,None)


def _view_bytes(key,cached,meta,cx,cy,span,width,height,detail_depth,detail_max):
    """Return a cached viewport or render exactly one shared source image."""
    hit=_memory_tile(key)
    if hit is not None: return hit
    with TILE_LOCK:
        future=TILE_PENDING.get(key)
        if future is None:
            if cached.exists():
                data=cached.read_bytes()
                _remember_tile(key,data)
                return data
            if TILE_POOL is None: raise RuntimeError('viewport pool is unavailable in a worker process')
            future=TILE_POOL.submit(_render_crystal_view_jpeg,meta,cx,cy,span,
                                    width,height,detail_depth,detail_max)
            TILE_PENDING[key]=future
    try:
        data=future.result()
        cached.parent.mkdir(parents=True,exist_ok=True)
        temporary=cached.with_name(f'.{cached.stem}.{time.time_ns()}.tmp{cached.suffix}')
        temporary.write_bytes(data); temporary.replace(cached)
        _remember_tile(key,data)
        return data
    finally:
        with TILE_LOCK:
            if TILE_PENDING.get(key) is future: TILE_PENDING.pop(key,None)


def _run_dir(rid:str):
    rd=(RUNS/rid).resolve()
    if RUNS.resolve() not in rd.parents or not (rd/'meta.json').exists():
        raise HTTPException(404,'unknown run')
    return rd


def _run_zoom_limit(meta):
    """A run owns its finite zoom limit; old runs receive a safe default."""
    manifest=meta.get('zoom') or {}
    try: requested=int(manifest.get('max_level',24))
    except (TypeError,ValueError): requested=24
    return max(0,min(MAX_ZOOM_LEVEL,requested))


@app.get('/api/zoom_tile/{rid}')
def zoom_tile(rid:str,level:int=0,col:int=0,row:int=0,tile:int=256):
    """One tile of the deep-zoom pyramid, rendered on demand and cached.

    Levels beyond the pre-baked ones are generated the first time they are
    asked for and written into the same runs/<id>/zoom/L<level>/ layout, so the
    baked pyramid and the on-demand tiles are one continuous structure and a
    revisited region is served straight from disk.
    """
    rd=_run_dir(rid)
    meta=json.loads((rd/'meta.json').read_text(encoding="utf-8"))
    if meta.get('demo') not in ZOOMABLE: raise HTTPException(404,'no deep zoom for this demo')
    level=max(0,min(_run_zoom_limit(meta),int(level))); n=1<<level
    if not (0<=col<n and 0<=row<n): raise HTTPException(422,'tile out of range')
    tile=max(64,min(512,int(tile)))
    # Do not reuse tiles made by the pre-seam-fix renderer.  Baked levels are
    # still served directly from zoom/L*, while on-demand tiles live here.
    cached=rd/'zoom'/f'dynamic-v{DYNAMIC_TILE_VERSION}'/f'L{level}'/f'{col}_{row}.jpg'
    man=meta.get('zoom') or {}
    base_span=float(man.get('span') or 2.0); bcx=float(man.get('cx') or 0.0); bcy=float(man.get('cy') or 0.0)
    sub=base_span/n
    tcx=bcx-base_span/2+sub*(col+.5)
    tcy=bcy+base_span/2-sub*(row+.5)
    # Larger local budget prevents a dense visible branch from being cut off
    # midway through its detail.  Adjacent tiles share deterministic geometry,
    # so a segment crossing the seam is identical on both sides.
    budget=max(12000,int(meta.get('zoom_budget',12000)))
    key=f'v{DYNAMIC_TILE_VERSION}:{rid}:{level}:{col}:{row}:{tile}'
    data=_tile_bytes(key,cached,meta,tcx,tcy,sub,tile,budget)
    return Response(data,media_type='image/jpeg',
                    headers={'Cache-Control':'public, max-age=31536000, immutable',
                             'X-DeepZoom-Raster':TILE_RASTER_BACKEND})


@app.get('/api/zoom/{rid}')
def zoom(rid:str,cx:float=0.0,cy:float=0.0,span:float=1.0,w:int=960,h:int=540):
    """Render one arbitrary window of a run's model, live.

    A pre-baked pyramid runs out of levels after a handful of doublings; past
    that the viewer can only upscale its deepest tiles, which is why deep zoom
    still looked like magnifying an image. Regenerating the geometry for the
    requested window has no depth limit at all.
    """
    rd=(RUNS/rid).resolve()
    if RUNS.resolve() not in rd.parents or not (rd/'meta.json').exists():
        raise HTTPException(404,'unknown run')
    meta=json.loads((rd/'meta.json').read_text(encoding="utf-8"))
    demo=meta.get('demo')
    if demo not in ZOOMABLE: raise HTTPException(404,'live zoom unavailable for this demo')
    if not (math.isfinite(cx) and math.isfinite(cy) and math.isfinite(span)) or span<=0:
        raise HTTPException(422,'bad window')
    w=max(64,min(1600,int(w))); h=max(64,min(1000,int(h)))
    from leonardo_demos.crystal_growth import generate, render_window, MODE_NAMES
    p=meta.get('params',{}) or {}
    g=generate(cx,cy,span,size_px=max(w,h),
               symmetry=int(p.get('symmetry',6)),
               mode=MODE_NAMES[int(p.get('mode',0))%len(MODE_NAMES)],
               undercooling=float(p.get('undercooling',.75)),
               anisotropy=float(p.get('anisotropy',.055)),
               seed=int(p.get('seed',3)),
               max_segments=int(meta.get('zoom_budget',22000)))
    im=render_window(g,cx,cy,span,size=(w,h),progress=1.0,supersample=2)
    buf=io.BytesIO(); im.save(buf,'JPEG',quality=86)
    return Response(buf.getvalue(),media_type='image/jpeg',
                    headers={'Cache-Control':'no-store'})


@app.get('/api/zoom_view/{rid}')
def zoom_view(rid:str,cx:float=0.0,cy:float=0.0,span:float=1.0,
              w:int=960,h:int=540,level:int=0):
    """Cached coherent viewport for the responsive deep-zoom canvas.

    Generating one image for the whole screen is intentional.  The recursive
    generator has a finite branch budget, so generating neighbouring tiles
    separately can select different equally-valid branches at their seam.
    A viewport source has one traversal and therefore cannot contain a tile
    grid.  The client reprojects its last source synchronously while this
    endpoint builds a replacement in the background.
    """
    rd=_run_dir(rid)
    meta=json.loads((rd/'meta.json').read_text(encoding="utf-8"))
    if meta.get('demo') not in ZOOMABLE: raise HTTPException(404,'no deep zoom for this demo')
    if not all(math.isfinite(value) for value in (cx,cy,span)) or span<=0:
        raise HTTPException(422,'bad window')
    level=max(0,min(_run_zoom_limit(meta),int(level)))
    w=max(512,min(1600,int(w))); h=max(288,min(1000,int(h)))
    manifest=meta.get('zoom') or {}
    try: detail_base=int(manifest.get('detail_base',7))
    except (TypeError,ValueError): detail_base=7
    try: detail_max=int(manifest.get('detail_max',14))
    except (TypeError,ValueError): detail_max=14
    detail_max=max(detail_base,min(18,detail_max))
    level=max(0,min(detail_max-detail_base,int(level)))
    detail_depth=detail_base+level
    # The browser asks for a little more than its visible world window.  It
    # lets it pan and animate instantly inside this source before a new cached
    # view is needed, while preserving native-quality detail in the centre.
    token=f'{DYNAMIC_VIEW_VERSION}|{detail_depth}|{cx:.16g}|{cy:.16g}|{span:.16g}|{w}|{h}'
    digest=hashlib.sha256(token.encode()).hexdigest()[:24]
    cached=rd/'zoom'/f'view-v{DYNAMIC_VIEW_VERSION}'/f'L{level}'/f'{digest}.jpg'
    key=f'view:{rid}:{token}'
    data=_view_bytes(key,cached,meta,cx,cy,span,w,h,detail_depth,detail_max)
    return Response(data,media_type='image/jpeg',
                    headers={'Cache-Control':'public, max-age=31536000, immutable',
                             'X-DeepZoom-Source':'stable-density-viewport',
                             'X-DeepZoom-Detail':str(detail_depth)})

@app.get('/api/replay/{rid}')
def replay(rid:str,gens:str,seed:int=0):
    """Replay saved generations' champions from a fresh random start.

    Computed on the CPU in this process: one cave or a handful of cars is
    milliseconds of NumPy, and it never creates a CUDA context here.
    """
    rd=_run_dir(rid)
    meta=json.loads((rd/'meta.json').read_text(encoding="utf-8"))
    demo_class=DEMOS.get(meta.get('demo'))
    if not hasattr(demo_class,'replay') or not (rd/'checkpoints').is_dir():
        raise HTTPException(404,'this run has no saved generations to replay')
    try: wanted=[int(g) for g in gens.split(',') if g.strip()]
    except ValueError: raise HTTPException(422,'gens must be generation numbers')
    if not 1<=len(wanted)<=6: raise HTTPException(422,'choose between 1 and 6 generations')
    missing=[g for g in wanted if not (rd/'checkpoints'/f'gen_{g:04d}.npz').exists()]
    if missing: raise HTTPException(404,f'generation(s) not saved yet: {missing}')
    return demo_class.replay(rd,meta,wanted,int(seed)%(1<<31))

@app.get('/api/run/{rid}')
def status(rid:str):
    rd=RUNS/rid; p=rd/'meta.json'
    if not p.exists(): return {'status':'starting','frame':-1}
    return json.loads(p.read_text(encoding="utf-8"))

def _free_port(preferred=8000, tries=20):
    """First free port at or after `preferred`.

    Binding is not optional here: if 8000 is already held by an older viewer,
    starting silently fails and the browser keeps talking to that stale
    process. A stale server re-reads config/profiles.json from disk but still
    holds the previously imported demo modules, so every run dies with a
    confusing KeyError. Moving to a free port makes the new server reachable.
    """
    import socket
    for i in range(tries):
        port=preferred+i
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
            try:
                s.bind(('127.0.0.1',port)); return port,i>0
            except OSError:
                continue
    return preferred,False


if __name__=='__main__':
    import uvicorn, webbrowser
    port,moved=_free_port(8000)
    url=f'http://127.0.0.1:{port}'
    if moved:
        print('=' * 68)
        print(f'  Port 8000 is already in use by another program.')
        print(f'  Starting this viewer on {port} instead.')
        print(f'  If an older viewer window is still open, close it: a stale')
        print(f'  viewer runs old code against the current config and every')
        print(f'  demo it launches will fail.')
        print('=' * 68)
    print(f'Leonardo Visual Demos -> {url}')
    try: webbrowser.open(url)
    except Exception: pass
    uvicorn.run(app,host='127.0.0.1',port=port)
