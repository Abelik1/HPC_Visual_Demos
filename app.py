from __future__ import annotations
import atexit, base64, binascii, hashlib, io, json, math, multiprocessing, os, shutil, threading, time, uuid
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
from leonardo_demos import run_bundles, lineups, remote
from leonardo_demos.multigpu import MULTI_GPU_DEMOS

ROOT=Path(__file__).resolve().parent; RUNS=ROOT/'runs'; RUNS.mkdir(exist_ok=True)
app=FastAPI(title='Leonardo Visual Demos')
app.mount('/static',StaticFiles(directory=ROOT/'web'),name='static')
app.mount('/runs',StaticFiles(directory=RUNS),name='runs')
app.mount('/benchmarks',StaticFiles(directory=ROOT/'benchmarks',html=True),name='benchmarks')
# Default pictures for the image-compression demo; drop more files in here.
COMPRESSION_IMAGES=ROOT/'data'/'compression_images'; COMPRESSION_IMAGES.mkdir(parents=True,exist_ok=True)
app.mount('/compression_images',StaticFiles(directory=COMPRESSION_IMAGES),name='compression_images')
# Pre-recorded videos for the Videos page (e.g. downloaded from Google Drive on
# demo day). Drop files into videos/ or point LEONARDO_VIDEO_DIR at a folder.
VIDEOS=Path(os.getenv('LEONARDO_VIDEO_DIR') or ROOT/'videos'); VIDEOS.mkdir(parents=True,exist_ok=True)
VIDEO_TYPES={'.mp4','.m4v','.webm','.ogv','.ogg','.mov','.mkv'}
app.mount('/video_files',StaticFiles(directory=VIDEOS),name='video_files')

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
    # GPUs of one node to split the simulation across (demos in MULTI_GPU_DEMOS).
    gpus: int | None = Field(default=None, ge=1, le=8)
    obstacle_grid: list[list[int]] | None = None
    # Visitor-built network for the AI game demos; validated against the
    # demo's block catalogue in config/demo_specs.json.
    brain: dict | None = None
    ghosts: list[str] | None = Field(default=None, max_length=5)
    # The visitor's name tag: shown on their champion and when it races as a ghost.
    name: str | None = Field(default=None, max_length=24)
    # Molecular Machine: a visitor-written sequence of H, P, + and - beads.
    chain: str | None = Field(default=None, max_length=400)
    # Star in a Bottle: carry on training the controller an earlier guardian
    # run finished with, instead of starting from an untrained network.
    resume_from: str | None = Field(default=None, max_length=128)

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

@app.get('/videos',response_class=HTMLResponse)
def videos_page(): return (ROOT/'web/videos.html').read_text(encoding='utf-8')

@app.get('/api/videos')
def videos(folder:str|None=None):
    """Every video file under the video folder, subfolders included.

    `folder` narrows the list to one subfolder: a video demo in the lineup
    (the raytracer, say) plays only its own recordings.
    """
    from urllib.parse import quote
    base=VIDEOS
    if folder:
        if not lineups.FOLDER.match(folder): raise HTTPException(422,'invalid folder name')
        base=VIDEOS/folder; base.mkdir(exist_ok=True)
    files=sorted((f for f in base.rglob('*') if f.is_file() and f.suffix.lower() in VIDEO_TYPES),
                 key=lambda f:f.relative_to(VIDEOS).as_posix().lower())
    return {'folder':str(base),
            'videos':[{'name':f.stem.replace('_',' ').replace('-',' ').strip(),
                       'path':f.relative_to(VIDEOS).as_posix(),
                       'size':f.stat().st_size,
                       'url':'/video_files/'+quote(f.relative_to(VIDEOS).as_posix())} for f in files]}

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

_PLAYBACK_CACHE={}
def _playback_frames(d,meta,frames):
    """Replay length for runs whose tail is idle; None means play everything.

    Folds record it when they finish. Folds saved before that are measured
    once from their per-frame radius of gyration.
    """
    if meta.get('playback_frames'): return min(frames,int(meta['playback_frames']))
    if meta.get('demo')!='molecular_dynamics' or meta.get('method')!='fold' or meta.get('status')!='complete': return None
    key=(d.name,frames)
    if key not in _PLAYBACK_CACHE:
        from leonardo_demos.demos.molecular_dynamics import fold_playback_frames
        rgs=[]
        for i in range(frames):
            try: rgs.append(float(json.loads((d/'frame_data'/f'frame_{i:04d}.json').read_text(encoding='utf-8'))['values']['radius of gyration']))
            except (OSError,ValueError,KeyError,TypeError): break
        _PLAYBACK_CACHE[key]=fold_playback_frames(rgs) if len(rgs)==frames else None
    return _PLAYBACK_CACHE[key]

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
            # AI games: the visitor's name tag, for their champion and as a ghost.
            'name':meta.get('name') or (meta.get('params') or {}).get('_name'),
            # Neuro-Racers: a saved champion that can race in other runs' replays.
            'has_champion':(d/'champion.npz').exists(),
            # Frames worth replaying on a loop, where the rest adds nothing.
            'playback_frames':_playback_frames(d,meta,len(frames)),
            # Star in a Bottle guardian: the conditions its controller trained in.
            'trained_world':meta.get('trained_world'),
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

# ---- run bundles: saved runs carried between machines as one zip ---------
# tools/export_runs.py packs favourites, showcase picks and cluster runs into a
# zip. Drop it into runs/_import/ (or the project root) and the viewer unpacks
# it on start, favourites and showcase included.
IMPORT_DIR=RUNS/run_bundles.IMPORT_DIR_NAME; IMPORT_DIR.mkdir(exist_ok=True)

def update_library(change):
    with LIBRARY_LOCK: save_library(change(load_library()))

def import_run_bundles():
    try: run_bundles.import_pending(RUNS,[IMPORT_DIR,ROOT],update_library)
    except Exception as exc: print(f'Importing run bundles failed: {exc}')

_STARTED=threading.Event()

@app.on_event('startup')
def on_start():
    """Work that must happen however the viewer was launched.

    The stand may be started with `python app.py`, with uvicorn directly, or by
    a launcher script. Unpacking waiting run bundles and picking up cluster
    jobs again belong to the app, not to one entry point: a viewer started with
    uvicorn used to come up with an empty gallery because the zip on the desk
    was never unpacked."""
    if _STARTED.is_set():
        return
    _STARTED.set()
    # Large bundles take minutes to unpack; runs appear in the list as each lands.
    threading.Thread(target=import_run_bundles,name='run-bundle-import',daemon=True).start()
    # Cluster jobs submitted before a restart are still running; keep watching them.
    try:
        resumed=remote.resume_jobs(RUNS)
        if resumed: print(f'Watching {len(resumed)} cluster job(s) again: {", ".join(resumed)}')
    except Exception as exc:
        print(f'Could not resume cluster jobs: {exc}')

def new_run_id(demo:str)->str:
    return f'{demo}_{time.strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:5]}'

def prepare_run(demo:str,req:RunReq,rd:Path,*,remote:bool=False,dry:bool=False):
    """Validate a run request and return run() keyword arguments.

    Shared by local runs and cluster runs so both accept exactly the same
    requests. `dry` validates without writing anything (the confirmation
    dialog); `remote` rejects inputs that only exist on this PC.
    """
    if demo not in DEMOS: raise HTTPException(404,'unknown demo')
    requested={'numpy':'cpu','cuda':'gpu','cupy':'gpu'}.get(req.backend,req.backend)
    if requested!='auto' and requested not in DEMOS[demo].supported_backends:
        allowed=', '.join(DEMOS[demo].supported_backends)
        raise HTTPException(422,f'{demo} supports these compute modes: {allowed}')
    req.profile=canonical_profile(req.profile)
    if req.profile not in load_profiles():
        raise HTTPException(422, 'unknown profile')
    method=DEMOS[demo].default_method if req.method=='default' else req.method
    valid_methods=getattr(DEMOS[demo],'remote_methods',DEMOS[demo].methods) if remote else DEMOS[demo].methods
    if method not in valid_methods:
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
    params=dict(req.params)
    if req.parallel_count is not None:
        if req.parallel_count not in {1,4,9,16,25,36,49,64}:
            raise HTTPException(422, 'parallel count must be 1 or form a square reveal grid: 4, 9, 16, 25, 36, 49 or 64')
        params['_parallel_count']=int(req.parallel_count)
    if req.gpus is not None and req.gpus>1:
        if demo not in MULTI_GPU_DEMOS:
            raise HTTPException(422, f'{demo} runs on one GPU; only {", ".join(sorted(MULTI_GPU_DEMOS))} split across several')
        params['_gpus']=int(req.gpus)
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
    if req.chain:
        if demo != 'molecular_dynamics':
            raise HTTPException(422, 'a bead sequence is only used by the Molecular Machine')
        from leonardo_demos.demos.molecular_dynamics import parse_chain
        try: parse_chain(req.chain)
        except ValueError as error: raise HTTPException(422, str(error))
        params['_chain']=req.chain.strip().upper()
    if req.name and req.name.strip():
        if brain_catalogue(load_specs(),demo) is None:
            raise HTTPException(422, 'a name tag is only used by the AI game demos')
        params['_name']=' '.join(''.join(c for c in req.name if c.isprintable()).split())[:24]
    if req.ghosts:
        if remote:
            raise HTTPException(422, 'ghost races use champions saved on this PC; turn them off for a cluster run')
        if demo != 'neuro_racers':
            raise HTTPException(422, 'ghost races are only supported by Neuro-Racers')
        ghost_dirs=[]
        for ghost in req.ghosts:
            gd=(RUNS/ghost).resolve()
            if RUNS.resolve() not in gd.parents or not (gd/'champion.npz').exists():
                raise HTTPException(422, f'ghost run {ghost} has no saved champion')
            ghost_dirs.append(str(gd))
        params['_ghosts']=ghost_dirs
    if req.resume_from:
        if demo != 'fusion_plasma' or method != 'guardian':
            raise HTTPException(422, 'only the AI plasma guardian continues an earlier run')
        source=(RUNS/req.resume_from).resolve()
        if RUNS.resolve() not in source.parents or not source.is_dir():
            raise HTTPException(404, f'unknown run {req.resume_from}')
        saved=sorted(source.glob('checkpoints/gen_*.npz'))
        if not saved:
            raise HTTPException(422, f'{req.resume_from} saved no controller to continue from')
        latest=saved[-1]
        try: before=json.loads((source/'meta.json').read_text(encoding='utf-8'))
        except (OSError, ValueError): before={}
        done=[int(row.get('trained_to',0)) for row in (before.get('shot_history') or [])]
        params['_resume']='resume.npz'
        params['_resume_run']=req.resume_from
        params['_resume_shot']=int(latest.stem.split('_')[-1])
        params['_resume_updates']=max(done or [int((before.get('settings') or {}).get('train_updates',0) or 0)])
        if not dry:
            rd.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(latest, rd/'resume.npz')
    if req.target_image is not None:
        if demo != 'neural_wall':
            raise HTTPException(422, 'a custom drawing is only supported by the neural-network wall')
        target_path=rd/'target.png'
        if not dry:
            rd.mkdir(parents=True, exist_ok=True)
            save_target_image(req.target_image, target_path)
        params['_target_path']=str(target_path)
    return dict(demo=demo,profile=req.profile,frames=req.frames,params=params,backend=req.backend,
                run_dir=rd,method=method,numerical_substeps=req.numerical_substeps,
                settings_override=clean_settings,precision=req.precision)

@app.post('/api/run/{demo}')
def start(demo:str,req:RunReq):
    rid=new_run_id(demo); rd=RUNS/rid
    kwargs=prepare_run(demo,req,rd)
    threading.Thread(target=_supervise_run,args=(rid,rd,kwargs),daemon=True).start(); return {'id':rid}

# ---- "Why HPC": scaling tables from tools/scaling_sweep.py ----------------
SCALING=ROOT/'benchmarks'/'scaling'

@app.get('/api/scaling')
def scaling():
    """{demo: {machine: table}} for every merged table in benchmarks/scaling/<machine>/."""
    out={}
    for path in sorted(SCALING.glob('*/*.json')):
        if path.name.count('.')!=1: continue           # per-kind and pilot files
        try: table=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError): continue
        out.setdefault(table.get('demo') or path.stem,{})[path.parent.name]=table
    return out

# ---- demo-day lineups: which demos each machine's demo day shows ----------
class LineupReq(BaseModel):
    active: str = 'all'
    machines: dict = Field(default_factory=dict)
    archived: list[str] = Field(default_factory=list)
    extras: dict = Field(default_factory=dict)

class ActiveLineupReq(BaseModel):
    active: str

def _lineup_payload():
    data=lineups.load()
    for extra in data['extras'].values():
        if extra.get('kind')=='video':
            folder=VIDEOS/extra.get('folder','')
            extra['videos']=sum(1 for f in folder.rglob('*') if f.is_file() and f.suffix.lower() in VIDEO_TYPES) if folder.is_dir() else 0
    return data

@app.get('/api/lineups')
def get_lineups(): return _lineup_payload()

@app.put('/api/lineups')
def put_lineups(req:LineupReq):
    try: clean=lineups.validate(req.model_dump(),set(DEMOS))
    except ValueError as error: raise HTTPException(422,str(error))
    lineups.save(clean)
    for extra in clean['extras'].values(): (VIDEOS/extra['folder']).mkdir(exist_ok=True)
    return _lineup_payload()

@app.put('/api/lineups/active')
def put_active_lineup(req:ActiveLineupReq):
    data=lineups.load(); data['active']=req.active
    try: clean=lineups.validate(data,set(DEMOS))
    except ValueError as error: raise HTTPException(422,str(error))
    lineups.save(clean); return _lineup_payload()

# ---- cluster runs: submit to Discoverer / Leonardo, fetch the result ------
class RemoteRunReq(BaseModel):
    cluster: str
    walltime: str | None = Field(default=None, max_length=16)
    request: RunReq

class ClusterSettingsReq(BaseModel):
    values: dict = Field(default_factory=dict)

def _cluster_or_404(name):
    try: return remote.cluster(name)
    except KeyError: raise HTTPException(404,'unknown cluster')

@app.get('/api/clusters')
def clusters():
    out={}
    for name in remote.load_clusters():
        c=remote.cluster(name); view=remote.public_view(c)
        view['certificate']=remote.certificate_status(c); out[name]=view
    return {'clusters':out,'jobs':remote.active_jobs(RUNS)}

@app.put('/api/clusters/{name}')
def update_cluster(name:str,req:ClusterSettingsReq):
    try: c=remote.save_overrides(name,req.values)
    except KeyError: raise HTTPException(404,'unknown cluster')
    except ValueError as error: raise HTTPException(422,str(error))
    return remote.public_view(c)

@app.post('/api/clusters/{name}/check')
def check_cluster(name:str):
    return remote.check(_cluster_or_404(name))

@app.post('/api/clusters/{name}/certificate')
def refresh_certificate(name:str):
    try: remote.open_certificate_login(_cluster_or_404(name))
    except remote.RemoteError as error: raise HTTPException(422,str(error))
    return {'opened':True}

def _remote_plan(demo,body,*,dry):
    c=_cluster_or_404(body.cluster)
    walltime=body.walltime or c.get('walltime')
    if not remote.WALLTIME.match(walltime or ''): raise HTTPException(422,'time limit must look like HH:MM:SS')
    rid=new_run_id(demo)
    request=body.request
    if request.backend=='auto':
        # On a cluster "auto" means the device this demo is planned for
        # (config/hpc_plan.json), never whatever the node happens to have.
        method=DEMOS[demo].default_method if request.method=='default' else request.method
        planned=remote.demo_needs(demo,method).get('backend','auto')
        if planned!='auto': request=request.model_copy(update={'backend':planned})
    kwargs=prepare_run(demo,request,RUNS/rid,remote=True,dry=dry)
    return c,walltime,rid,kwargs

@app.post('/api/remote/plan/{demo}')
def remote_plan(demo:str,body:RemoteRunReq):
    """Everything the confirmation dialog lists before a cluster run starts."""
    c,walltime,_,kwargs=_remote_plan(demo,body,dry=True)
    spec=load_specs()[demo]; preset=load_profiles()[kwargs['profile']].get(demo,{})
    params=[{'key':k,'label':p.get('label') or k.replace('_',' '),'value':kwargs['params'].get(k,p.get('value')),
             'default':p.get('value'),'options':p.get('options')} for k,p in spec['params'].items()]
    overrides=kwargs['settings_override']
    settings=[{'key':k,'value':overrides.get(k,preset.get(k)),'preset':preset.get(k),
               'changed':k in overrides and overrides[k]!=preset.get(k)}
              for k in dict.fromkeys([*preset,*overrides])]
    extras=[]
    if '_parallel_count' in kwargs['params']: extras.append(f"Independent runs: {kwargs['params']['_parallel_count']}")
    if '_obstacle_grid' in kwargs['params']: extras.append('Your drawn obstacle shape')
    if '_brain' in kwargs['params']: extras.append('The brain built with the block builder')
    if '_target_path' in kwargs['params']: extras.append('Your own target picture')
    if '_resume' in kwargs['params']:
        extras.append(f"Continues {kwargs['params']['_resume_run']} "
                      f"({kwargs['params']['_resume_updates']:,} updates already trained)")
    if '_chain' in kwargs['params']: extras.append(f"Your own sequence: {kwargs['params']['_chain']}")
    gpus=kwargs['params'].get('_gpus')
    warnings=[]
    res=remote.resources(c,walltime,demo,kwargs['method'],gpus)
    if gpus: extras.append(f"Split across {res['gpus']} GPUs of one node")
    if remote.account_missing(c,res):
        warnings.append(f"No Slurm account is set for {c['label']} ({'CPU' if res['device']=='cpu' else 'GPU'} partition). Open HPC settings and enter it.")
    cert=remote.certificate_status(c)
    if cert and not cert.get('valid'):
        warnings.append(f"The {c['label']} SSH certificate: {cert['detail']}. Refresh it in HPC settings first.")
    elif cert and cert.get('seconds_left',1e9)<3600:
        warnings.append(f"The {c['label']} SSH certificate expires soon ({cert['detail']}); fetching may fail after that.")
    if kwargs['profile']!=c.get('default_profile','hpc'):
        warnings.append(f"Quality preset is '{kwargs['profile']}'. Pick HPC in the run settings to use the cluster at full scale.")
    return {'cluster':remote.public_view(c),'resources':res,
            'python':remote.python_for(c,demo,kwargs['method']),
            'demo':{'id':demo,'name':spec.get('name',demo)},
            'method':kwargs['method'],'method_label':DEMOS[demo].method_labels.get(kwargs['method'],kwargs['method']),
            'profile':kwargs['profile'],'frames':kwargs['frames'],'backend':kwargs['backend'],
            'precision':kwargs['precision'],'params':params,'settings':settings,'extras':extras,
            'warnings':warnings,'multi_gpu':demo in MULTI_GPU_DEMOS,
            'node_gpus':int((c.get('gpu_node') or {}).get('gpus',1))}

@app.post('/api/remote/run/{demo}')
def remote_run(demo:str,body:RemoteRunReq):
    c,walltime,rid,kwargs=_remote_plan(demo,body,dry=False)
    files={'target.png':RUNS/rid/'target.png'} if '_target_path' in kwargs['params'] else {}
    if '_resume' in kwargs['params']: files['resume.npz']=RUNS/rid/'resume.npz'
    remote.create_job(RUNS,rid,c,kwargs,walltime,files)
    return {'id':rid,'cluster':c['name']}

@app.post('/api/remote/cancel/{rid}')
def remote_cancel(rid:str):
    if not run_demo_name(rid): raise HTTPException(404,'unknown run')
    try: remote.cancel(RUNS,rid)
    except remote.RemoteError as error: raise HTTPException(422,str(error))
    return {'id':rid,'cancelling':True}

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
def replay(rid:str,gens:str,seed:int=0,track:int|None=None,cave:int|None=None,moths:int|None=None,ghosts:str|None=None,
           magnetic_field:float|None=None,heating:float|None=None,instability:float|None=None):
    """Replay saved generations' champions from a fresh random start.

    Computed on the CPU in this process: one cave or a handful of cars is
    milliseconds of NumPy, and it never creates a CUDA context here.
    ``track``, ``cave`` and ``moths`` (the AI games) or ``magnetic_field``,
    ``heating`` and ``instability`` (Star in a Bottle's saved controllers) test
    the frozen networks in a different world; ``ghosts`` (comma-separated run
    ids) races other saved champions.
    """
    rd=_run_dir(rid)
    meta=json.loads((rd/'meta.json').read_text(encoding="utf-8"))
    demo_class=DEMOS.get(meta.get('demo'))
    if not hasattr(demo_class,'replay') or not (rd/'checkpoints').is_dir():
        raise HTTPException(404,'this run has no saved generations to replay')
    try: wanted=[int(g) for g in gens.split(',') if g.strip()]
    except ValueError: raise HTTPException(422,'gens must be generation numbers')
    if not 1<=len(wanted)<=12: raise HTTPException(422,'choose between 1 and 12 generations')
    missing=[g for g in wanted if not (rd/'checkpoints'/f'gen_{g:04d}.npz').exists()]
    if missing: raise HTTPException(404,f'generation(s) not saved yet: {missing}')
    params=(load_specs().get(meta.get('demo')) or {}).get('params',{})
    env={}
    for key,value in (('track',track),('cave',cave),('moths',moths),
                      ('magnetic_field',magnetic_field),('heating',heating),('instability',instability)):
        if value is None: continue
        limits=params.get(key)
        if not limits or not math.isfinite(value) or not float(limits['min'])<=value<=float(limits['max']):
            raise HTTPException(422,f'{key} is not a setting of this demo, or is out of range')
        env[key]=int(value) if key in ('track','cave','moths') else float(value)
    if ghosts:
        if meta.get('demo')!='neuro_racers': raise HTTPException(422,'ghost races are only supported by Neuro-Racers')
        ids=[g for g in ghosts.split(',') if g.strip()]
        if len(ids)>5: raise HTTPException(422,'race at most 5 ghosts')
        env['ghosts']=[]
        for ghost in ids:
            gd=(RUNS/ghost).resolve()
            if RUNS.resolve() not in gd.parents or not (gd/'champion.npz').exists():
                raise HTTPException(422,f'ghost run {ghost} has no saved champion')
            env['ghosts'].append(str(gd))
    return demo_class.replay(rd,meta,wanted,int(seed)%(1<<31),env)

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
    # Bundles and cluster jobs are picked up by the startup hook above, so they
    # happen under uvicorn and any other launcher too.
    try: webbrowser.open(url)
    except Exception: pass
    uvicorn.run(app,host='127.0.0.1',port=port)
