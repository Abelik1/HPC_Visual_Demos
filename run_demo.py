from __future__ import annotations
import argparse, json, shutil, time, webbrowser
from numbers import Integral, Real
from pathlib import Path
from leonardo_demos.backend import PRECISIONS, resolve_precision
from leonardo_demos.base import RunContext
from leonardo_demos.registry import DEMOS

ROOT=Path(__file__).resolve().parent

def load_profiles(): return json.loads((ROOT/'config/profiles.json').read_text(encoding="utf-8"))
# The supercomputer-scale preset was called "leonardo" before it became "hpc".
# Saved runs and older job scripts still carry the old name, so keep accepting it.
PROFILE_ALIASES={'leonardo':'hpc'}
def canonical_profile(name): return PROFILE_ALIASES.get(name,name)
def load_specs(): return json.loads((ROOT/'config/demo_specs.json').read_text(encoding="utf-8"))

def profile_setting_schema(profiles=None):
    """Describe every numeric value carried by a compute profile.

    Profiles are safe starting presets, not sealed configurations. The
    editable range spans all shipped/tested presets. A setting whose value is
    identical in every preset receives a conservative range around that value
    so it is still genuinely editable.
    """
    profiles=profiles or load_profiles()
    demos={demo for profile in profiles.values() for demo in profile}
    result={}
    for demo in sorted(demos):
        keys={key for profile in profiles.values() for key in profile.get(demo,{})}
        fields={}
        for key in sorted(keys):
            values=[profile[demo][key] for profile in profiles.values()
                    if demo in profile and key in profile[demo]]
            if not values or not all(isinstance(value,Real) and not isinstance(value,bool)
                                     for value in values):
                continue
            integer=all(isinstance(value,Integral) for value in values)
            low=min(values); high=max(values)
            # Do not make the named presets the bounds. They are starting
            # points: visitors can deliberately run below benchmark or above
            # HPC. The finite profile-derived envelope still catches an
            # accidental extra zero before an O(N²) or high-resolution job is
            # launched.
            if integer:
                low=max(1,int(low)//4)
                high=max(int(high)+1,int(high)*2)
            else:
                low=max(.0001,float(low)/4)
                high=max(float(high)+.0001,float(high)*4)
            fields[key]={
                'type':'integer' if integer else 'number',
                'min':int(low) if integer else float(low),
                'max':int(high) if integer else float(high),
                # Decimal controls may have profile-derived minima that are
                # not an integer multiple of a display step (7.5 Gyr with a
                # derived 1.875 minimum, for example). ``any`` keeps exact
                # numeric editing valid; the API still enforces finite values
                # and the published min/max bounds.
                'step':1 if integer else 'any',
            }
        result[demo]=fields
    return result

def _normalise_backend(value):
    value=value.lower()
    if value in {'numpy','cpu'}: return 'cpu'
    if value in {'cupy','cuda','gpu'}: return 'gpu'
    if value in {'hybrid','cpu+gpu','cpu_gpu'}: return 'hybrid'
    return value

def run(demo,profile='local',frames=80,params=None,backend='auto',run_dir=None,
        method='default',timings=False,numerical_substeps=None,settings_override=None,
        precision='fp32',resume=None):
    profiles=load_profiles(); specs=load_specs(); profile=canonical_profile(profile)
    if demo not in DEMOS: raise SystemExit(f"Unknown demo {demo}. Choices: {', '.join(DEMOS)}")
    if profile not in profiles: raise SystemExit(f"Unknown profile {profile}")
    frames=int(frames)
    if frames < 1:
        raise ValueError('frames must be at least 1')
    defaults={k:v.get('value') for k,v in specs[demo]['params'].items()}
    defaults.update(params or {})
    stamp=time.strftime('%Y%m%d_%H%M%S'); run_dir=Path(run_dir or ROOT/'runs'/f'{demo}_{stamp}')
    demo_class=DEMOS[demo]
    mode=_normalise_backend(backend)
    if mode != 'auto' and mode not in demo_class.supported_backends:
        supported=', '.join(demo_class.supported_backends)
        raise ValueError(f"{demo} does not support the {mode} backend; supported: {supported}")
    if method == 'default':
        method=demo_class.default_method
    if method not in demo_class.methods:
        raise ValueError(f"{demo} does not support method {method!r}; choices: {', '.join(demo_class.methods)}")
    precision,_=resolve_precision(precision)
    if precision not in demo_class.precisions:
        raise ValueError(f"{demo} does not implement {precision} precision; "
                         f"choices: {', '.join(demo_class.precisions)}")
    settings=dict(profiles[profile][demo])
    settings.update(settings_override or {})
    if numerical_substeps is not None:
        if demo not in {'galaxy_collision','galaxy_collision_3d'}:
            raise ValueError('numerical substeps are only defined for the collision solvers')
        settings['substeps']=int(numerical_substeps)
    if resume:
        # Continue an earlier run's controller: the file is copied into this
        # run so the run stays self-contained (FusionPlasmaDemo.resume_controller).
        source=Path(resume)
        if source.is_dir():
            saved=sorted(source.glob('checkpoints/gen_*.npz'))
            if not saved: raise ValueError(f'{source} saved no controller to continue from')
            source=saved[-1]
        if not source.exists(): raise ValueError(f'no such controller: {source}')
        run_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, run_dir/'resume.npz')
        defaults['_resume']='resume.npz'
        defaults['_resume_run']=source.parent.parent.name
        defaults['_resume_shot']=int(source.stem.split('_')[-1]) if source.stem.startswith('gen_') else None
        meta=source.parent.parent/'meta.json'
        try: before=json.loads(meta.read_text(encoding='utf-8')) if meta.exists() else {}
        except ValueError: before={}
        done=[int(row.get('trained_to',0)) for row in (before.get('shot_history') or [])]
        defaults['_resume_updates']=max(done or [int((before.get('settings') or {}).get('train_updates',0) or 0)])
    ctx=RunContext(run_dir,demo,profile,frames,defaults,backend,
                   demo_class.backend_kind,method,timings,precision)
    ctx.write_meta({'settings':settings,
                    'settings_override':dict(settings_override or {})})
    if numerical_substeps is not None:
        ctx.write_meta({'numerical_substeps':int(numerical_substeps)})
    try:
        DEMOS[demo](ctx,settings).run()
    except Exception as e:
        ctx.fail(e); raise
    print(run_dir)
    return run_dir

if __name__=='__main__':
    ap=argparse.ArgumentParser(description='Generate a visual HPC demo run')
    ap.add_argument('demo',choices=sorted(DEMOS)); ap.add_argument('--profile',type=canonical_profile,choices=sorted(load_profiles()),default='local'); ap.add_argument('--frames',type=int,default=80); ap.add_argument('--backend',default='auto'); ap.add_argument('--method',default='default'); ap.add_argument('--precision',choices=sorted(PRECISIONS),default='fp32',help='fp32, mixed (FP64 state/sums, FP32 pair maths; 3-D galaxy only) or fp64'); ap.add_argument('--numerical-substeps',type=int); ap.add_argument('--timings',action='store_true'); ap.add_argument('--param',action='append',default=[],help='scientific key=value, repeatable'); ap.add_argument('--setting',action='append',default=[],help='profile/scale key=value, repeatable'); ap.add_argument('--run-dir'); ap.add_argument('--open',action='store_true')
    ap.add_argument('--resume',help='continue an earlier guardian run: its directory, or one gen_*.npz')
    ap.add_argument('--brain',help='JSON brain spec from the block builder (AI game demos)')
    a=ap.parse_args(); params={}; settings_override={}
    if a.brain:
        from leonardo_demos.neuroevo import brain_catalogue, validate_brains
        catalogue=brain_catalogue(load_specs(),a.demo)
        if catalogue is None: raise SystemExit(f'{a.demo} has no brain builder')
        params['_brain']=validate_brains(json.loads(Path(a.brain).read_text(encoding="utf-8")),catalogue)
    for kv in a.param:
        k,v=kv.split('=',1)
        try: v=float(v)
        except ValueError: pass
        params[k]=v
    for kv in a.setting:
        k,v=kv.split('=',1)
        try: v=float(v)
        except ValueError: pass
        settings_override[k]=v
    rd=run(a.demo,a.profile,a.frames,params,a.backend,a.run_dir,a.method,a.timings,
           a.numerical_substeps,settings_override,a.precision,a.resume)
    if a.open:
        webbrowser.open(f'file://{(rd/"frames"/"frame_0000.jpg").resolve()}')
