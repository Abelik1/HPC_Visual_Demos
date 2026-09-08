from __future__ import annotations
import math
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image, ImageDraw
from ..base import Demo
from ..render import add_title, add_progress, save_frame, mosaic, font
from ..crystal_growth import build_crystal, render_window, MODE_NAMES
from ..deepzoom import build_pyramid


def _crystal_reveal_worker(args):
    """Build one independent habit in a process (the generator is GIL-bound)."""
    mode,symmetry,depth,undercooling,anisotropy,seed=args
    crystal=build_crystal(symmetry=symmetry,depth=depth,mode=mode,
                          undercooling=undercooling,anisotropy=anisotropy,seed=seed)
    image=render_window(crystal,0,0,crystal['extent']*2.3,size=(240,240),
                        progress=1.0,supersample=1)
    return image.tobytes(),f"{mode} · {symmetry}-fold"


class CrystalDemo(Demo):
    backend_kind="cpu"
    supported_backends=("cpu",)
    timing_methods={"geometry":"initialization","frame":"render",
                    "build_reveal":"ensemble","build_zoom":"deep_zoom"}
    id="crystal"; title="Crystal growth"
    def geometry(self,depth=None):
        p=self.ctx.params
        mode=MODE_NAMES[int(p.get('mode',0))%len(MODE_NAMES)]
        return build_crystal(
            symmetry=int(p.get('symmetry',6)),
            depth=int(self.settings.get('depth',5) if depth is None else depth),
            mode=mode,
            undercooling=float(p.get('undercooling',.75)),
            anisotropy=float(p.get('anisotropy',.055)),
            seed=int(p.get('seed',3)),
        ),mode
    def frame(self,cr,progress,mode,i,frames):
        span=cr['extent']*2.35
        # The world-to-screen transform preserves scale, so a full 16:9 canvas
        # gains breathing room without distorting the crystal's symmetry.
        im=render_window(cr,0,0,span,size=(1280,720),progress=progress,supersample=2)
        im=add_title(im,"Crystal growth",f"recursive anisotropic growth · habit: {mode} · {self.ctx.backend_name}")
        add_progress(im,(i+1)/frames,"MICROSCOPIC SEED","BRANCHED CRYSTAL")
        return im
    def build_reveal(self):
        ens=max(1,int(self.settings.get('ensemble',25))); side=max(1,int(math.ceil(math.sqrt(ens))))
        base=self.ctx.params
        tasks=[]
        for j in range(ens):
            mode=MODE_NAMES[j%len(MODE_NAMES)]
            symmetry=3+(j//len(MODE_NAMES))%10
            tasks.append((mode,symmetry,max(3,int(self.settings.get('depth',5))-1),
                          float(base.get('undercooling',.75))*(.7+.6*((j%5)/4)),
                          float(base.get('anisotropy',.055)),3+j))
        if self.ctx.cpu_workers > 1 and len(tasks) > 1:
            workers=min(self.ctx.cpu_workers,len(tasks),8)
            with ProcessPoolExecutor(max_workers=workers) as pool:
                rows=list(pool.map(_crystal_reveal_worker,tasks))
        else:
            rows=[_crystal_reveal_worker(task) for task in tasks]
        ims=[Image.frombytes('RGB',(240,240),data) for data,_ in rows]
        labels=[label for _,label in rows]
        return mosaic(ims,side,title="One rule, many habits",
                      subtitle="Temperature and supersaturation pick the habit; the recursion is the same.",
                      labels=labels,label_fill=(190,226,255))
    def build_zoom(self,frames):
        levels=int(self.settings.get('zoom_levels',4))
        if levels<=0:
            return None
        p=self.ctx.params
        deep=build_crystal(
            symmetry=int(p.get('symmetry',6)),
            depth=int(self.settings.get('zoom_depth',7)),
            mode=MODE_NAMES[int(p.get('mode',0))%len(MODE_NAMES)],
            undercooling=float(p.get('undercooling',.75)),
            anisotropy=float(p.get('anisotropy',.055)),
            seed=int(p.get('seed',3)),
        )
        span=deep['extent']*2.35
        self.ctx.write_status(frames-1,"Baking deep-zoom tiles")
        manifest=build_pyramid(
            lambda cx,cy,sp,size: render_window(deep,cx,cy,sp,size=size,progress=1.0,supersample=2),
            self.ctx.run_dir/'zoom',cx=0,cy=0,span=span,
            levels=levels,tile=int(self.settings.get('zoom_tile',256)),
            progress_cb=lambda m,t: self.ctx.write_status(frames-1,f"Baking zoom tiles {m}/{t}"))
        # Stable deep zoom has a finite grammar depth.  These values describe
        # cumulative branch generations, not arbitrary image-pyramid levels.
        detail_base=int(self.settings.get('zoom_detail_base',7))
        detail_max=int(self.settings.get('zoom_detail_max',14))
        detail_max=max(detail_base,min(18,detail_max))
        manifest['detail_base']=detail_base
        manifest['detail_max']=detail_max
        manifest['max_level']=detail_max-detail_base
        self.ctx.write_meta({"zoom":manifest,"zoom_segments":int(len(deep['x0']))})
        return manifest
    def run(self):
        frames=self.ctx.frames
        cr,mode=self.geometry()
        for i in range(frames):
            progress=(i+1)/frames
            im=self.frame(cr,progress,mode,i,frames)
            self.ctx.save_frame(im,self.ctx.frame_path(i))
            self.ctx.write_status(i,f"Growing {mode} habit",{
                "habit":mode,"symmetry":f"{int(self.ctx.params.get('symmetry',6))}-fold",
                "undercooling":f"{float(self.ctx.params.get('undercooling',.75)):.2f}",
                "anisotropy":f"{float(self.ctx.params.get('anisotropy',.055)):.3f}",
                "segments":f"{len(cr['x0']):,}"})
        # Reveal: the same growth rule across habits and symmetries.
        rev=self.build_reveal()
        rp=self.ctx.run_dir/'reveal.jpg'; self.ctx.save_frame(rev,rp)
        # Deep-zoom pyramid, rendered from the geometry at a higher recursion
        # depth than the frames use, so zooming finds branches that were never
        # resolvable in the 1280x720 frames.
        self.build_zoom(frames)
        self.ctx.finish(rp)
