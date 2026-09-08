from __future__ import annotations
import math
import numpy as np
from PIL import ImageDraw
from ..base import Demo
from ..backend import to_numpy
from ..render import array_image, add_title, add_progress, save_frame, mosaic, font

class ReactionDiffusionDemo(Demo):
    timing_methods={"initialise":"initialization","step":"simulation"}
    id="reaction_diffusion"; title="Living mathematics"
    # The viewer frame is 16:9. Simulating a square grid and stretching it to
    # fill the frame turned every circular Turing spot into an ellipse, so the
    # domain itself now carries the display aspect ratio.
    aspect=16/9
    def grid_shape(self,n):
        ny=max(16,int(n)); return ny,max(16,int(round(ny*self.aspect)))
    def initialise(self,ny,nx,seed=2):
        xp=self.ctx.xp
        U=xp.ones((ny,nx),dtype=xp.float32); V=xp.zeros((ny,nx),dtype=xp.float32)
        rng=np.random.default_rng(seed)
        # Seed count follows the domain area. With a fixed nine seeds a large
        # grid stayed almost empty for the whole run, because the pattern front
        # only advances a few cells per hundred steps.
        seeds=max(6,int(round(9*(ny*nx)/(192*341))))
        r=max(3,ny//48)
        for _ in range(seeds):
            cy=int(rng.uniform(.12,.88)*ny); cx=int(rng.uniform(.12,.88)*nx)
            U[cy-r:cy+r,cx-r:cx+r]=0.48; V[cy-r:cy+r,cx-r:cx+r]=0.26+rng.uniform(0,.18)
        V += xp.asarray(rng.normal(0,0.01,(ny,nx)).astype(np.float32))
        return U,V
    def step(self,U,V,F,K,steps):
        xp=self.ctx.xp; Du,Dv=.16,.08; dt=1.0
        for _ in range(steps):
            lu=(xp.roll(U,1,0)+xp.roll(U,-1,0)+xp.roll(U,1,1)+xp.roll(U,-1,1)-4*U)
            lv=(xp.roll(V,1,0)+xp.roll(V,-1,0)+xp.roll(V,1,1)+xp.roll(V,-1,1)-4*V)
            uvv=U*V*V
            U += (Du*lu-uvv+F*(1-U))*dt
            V += (Dv*lv+uvv-(F+K)*V)*dt
            xp.clip(U,0,1,out=U); xp.clip(V,0,1,out=V)
        return U,V
    def budget(self):
        """Total solver steps for the run, independent of the frame count.

        Gray-Scott needs O(10^4) steps before local rules produce global
        structure. Deriving the budget from `frames * steps_per_frame` meant a
        70-frame run stopped at ~1000 steps, i.e. about 4% of the domain
        covered, which is why the screen looked empty.
        """
        s=self.settings
        if 'total_steps' in s: return max(1,int(s['total_steps']))
        return max(1,int(s.get('steps_per_frame',8))*self.ctx.frames)
    def run(self):
        ny,nx=self.grid_shape(self.settings['n'])
        F=float(self.ctx.params.get('feed',.0367)); K=float(self.ctx.params.get('kill',.0649))
        total=self.budget(); frames=self.ctx.frames
        U,V=self.initialise(ny,nx)
        done=0
        for i in range(frames):
            target=int(round(total*(i+1)/frames))
            U,V=self.step(U,V,F,K,max(1,target-done)); done=target
            field=to_numpy(V)
            im=array_image(field,'plasma',size=(1280,720))
            im=add_title(im,"Living mathematics",f"Gray–Scott reaction diffusion · F={F:.4f} · K={K:.4f} · {nx}×{ny} cells · {self.ctx.backend_name}")
            add_progress(im,(i+1)/frames,"DRAW / PERTURB","EMERGENT PATTERN")
            self.ctx.save_frame(im,self.ctx.frame_path(i)); self.ctx.write_status(i,f"Evolving reaction field · step {done:,}",{
                "solver step":f"{done:,} / {total:,}","feed F":f"{F:.4f}","kill K":f"{K:.4f}",
                "grid":f"{nx} × {ny}","backend":self.ctx.backend_name})
        # True parameter sweep (small independent simulations, matured far
        # enough that neighbouring F/K values are visually distinguishable).
        ens=max(1,int(self.settings.get('ensemble',25))); side=max(1,int(math.ceil(math.sqrt(ens))))
        sweep_steps=int(self.settings.get('sweep_steps',max(2500,total//4)))
        my,mx=self.grid_shape(max(72,ny//3)); ims=[]
        for j in range(ens):
            # Sweep spans labyrinth -> spots -> extinction. The kill span is
            # kept narrow: wider and the top rows are simply blank, because the
            # pattern cannot sustain itself there.
            fj=F + (j%side-(side-1)/2)*0.0020
            kj=K + (j//side-(side-1)/2)*0.0010 - 0.0006
            u,v=self.initialise(my,mx,seed=j+10)
            u,v=self.step(u,v,fj,kj,sweep_steps)
            ims.append(array_image(to_numpy(v),'plasma',size=(260,146)))
        rev=mosaic(ims,side,title="Actually… we evolved many mathematical worlds")
        rp=self.ctx.run_dir/'reveal.jpg'; self.ctx.save_frame(rev,rp); self.ctx.finish(rp)
