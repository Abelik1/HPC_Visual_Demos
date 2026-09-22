from __future__ import annotations
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .. import tuning
from ..base import Demo
from ..backend import to_numpy
from ..multigpu import split
from ..render import add_title, add_progress, save_frame, font
from ..colors import palette
from ..pipeline import FramePipeline

W,H=1280,720
LBM_TAU=.57
# Readout names for the obstacle parameter; index matches config/demo_specs.json.
OBSTACLE_NAMES=("single cylinder","twin cylinders","mixed bodies","custom shape only")

# One fused D2Q9 update per lattice step: pull-stream (with periodic wrap) and
# full-way bounce-back at solid cells, then BGK collision with the inlet
# velocity imposed on column 0, writing post-collision populations.  This is
# the NumPy path's collide -> roll -> bounce-back sequence reordered so that
# each population is read once and written once per step, instead of the ~60
# full-lattice array passes the array-expression version needs.
LBM_KERNEL=r'''
typedef REAL_T real;

extern "C" __global__ void __launch_bounds__(BX * BY)
lbm_fused(const real* __restrict__ src, real* __restrict__ dst,
          const unsigned char* __restrict__ solid, const int nx, const int ny,
          const real u0, const real omega, const int stream, const int write_macro,
          real* __restrict__ rho_out, real* __restrict__ ux_out, real* __restrict__ uy_out)
{
    const int CX[9] = {0, 1, 0, -1, 0, 1, -1, -1, 1};
    const int CY[9] = {0, 0, 1, 0, -1, 1, 1, -1, -1};
    const int OPP[9] = {0, 3, 4, 1, 2, 7, 8, 5, 6};
    const real W[9] = {(real)4/9, (real)1/9, (real)1/9, (real)1/9, (real)1/9,
                       (real)1/36, (real)1/36, (real)1/36, (real)1/36};
    const int x = blockIdx.x * BX + threadIdx.x;
    const int y = blockIdx.y * BY + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const size_t plane = (size_t)nx * ny;
    const size_t cell = (size_t)y * nx + x;
    const bool wall = solid[cell] != 0;
    real f[9];
#pragma unroll
    for (int q = 0; q < 9; ++q) {
        if (!stream) { f[q] = src[q * plane + cell]; continue; }
        /* streamed value g_q(x) = f*_q(x - c_q); a solid cell swaps
           directions, g_opp(q)(x) = f*_opp(q)(x + c_q). */
        const int qs = wall ? OPP[q] : q;
        int xs = wall ? x + CX[q] : x - CX[q];
        int ys = wall ? y + CY[q] : y - CY[q];
        xs += (xs < 0) ? nx : ((xs >= nx) ? -nx : 0);
        ys += (ys < 0) ? ny : ((ys >= ny) ? -ny : 0);
        f[q] = src[qs * plane + (size_t)ys * nx + xs];
    }
    real rho = 0, mx = 0, my = 0;
#pragma unroll
    for (int q = 0; q < 9; ++q) { rho += f[q]; mx += CX[q] * f[q]; my += CY[q] * f[q]; }
    real ux = mx / (rho + (real)1e-8), uy = my / (rho + (real)1e-8);
    if (write_macro) {
        rho_out[cell] = rho;
        ux_out[cell] = wall ? (real)0 : ux;
        uy_out[cell] = wall ? (real)0 : uy;
    }
    if (x == 0) { ux = u0; uy = 0; }
    const real usq = ux * ux + uy * uy;
#pragma unroll
    for (int q = 0; q < 9; ++q) {
        const real cu = 3 * (CX[q] * ux + CY[q] * uy);
        const real feq = W[q] * rho * (1 + cu + (real)0.5 * cu * cu - (real)1.5 * usq);
        dst[q * plane + cell] = f[q] + omega * (feq - f[q]);
    }
}
'''


class FluidDemo(Demo):
    timing_methods={"init":"initialization","step":"simulation","advect":"visualization"}
    precisions=("fp32","fp64")
    id="fluid"; title="Virtual wind tunnel"
    _lbm_kernels={}
    def build_obstacles(self,nx,ny,preset,custom_grid=None):
        """Build the actual LBM solid mask used by both solver and renderer."""
        yy,xx=np.mgrid[0:ny,0:nx]
        mask=np.zeros((ny,nx),dtype=bool)
        preset=int(round(preset))
        if preset==1:
            for cy in (ny*.35,ny*.65):
                mask|=(xx-nx*.30)**2+(yy-cy)**2 < (ny*.095)**2
        elif preset==2:
            # Two deliberately different bluff bodies make their interacting
            # wakes obvious: a streamlined ellipse followed by a small block.
            mask|=((xx-nx*.26)/(nx*.075))**2+((yy-ny*.42)/(ny*.065))**2 < 1
            mask|=(np.abs(xx-nx*.39)<nx*.026)&(np.abs(yy-ny*.62)<ny*.105)
        elif preset==3:
            # Custom shape only: the visitor's drawing is the whole body. Every
            # other preset is combined with the drawing, which is why a drawn
            # shape used to arrive with the single cylinder still inside it.
            pass
        else:
            mask|=(xx-nx*.28)**2+(yy-(ny*.5+.5))**2 < (ny*.15)**2
        if custom_grid:
            grid=np.asarray(custom_grid,dtype=bool)
            gh,gw=grid.shape
            for row,col in np.argwhere(grid):
                x0=max(3,int(round(col*nx/gw))); x1=min(nx-3,int(round((col+1)*nx/gw)))
                y0=max(1,int(round(row*ny/gh))); y1=min(ny-1,int(round((row+1)*ny/gh)))
                if x1>x0 and y1>y0: mask[y0:y1,x0:x1]=True
        if not mask.any():
            raise ValueError('the obstacle is empty: draw at least one solid block')
        ys,xs=np.where(mask)
        self.obstacle_centre=(float(xs.mean()),float(ys.mean()))
        self.obstacle_radius=math.sqrt(float(mask.sum())/math.pi)
        self.obstacle_bounds=(int(xs.min()),int(xs.max()),int(ys.min()),int(ys.max()))
        self.obstacle_mask=mask
        return mask

    def init(self,nx,ny,u0,preset=0,custom_grid=None):
        """Build the initial lattice.

        On the NumPy path ``f`` holds the pre-collision populations.  On CUDA
        it holds the post-collision populations instead, because the fused
        kernel's natural state is "collided, not yet streamed"; ``step``
        returns identical macroscopic fields either way.
        """
        xp=self.ctx.xp; dtype=self.ctx.state_dtype
        # D2Q9 LBM
        c=xp.asarray([[0,0],[1,0],[0,1],[-1,0],[0,-1],[1,1],[-1,1],[-1,-1],[1,-1]],dtype=xp.int32)
        w=xp.asarray([4/9,1/9,1/9,1/9,1/9,1/36,1/36,1/36,1/36],dtype=dtype)
        rho=xp.ones((ny,nx),dtype=dtype); ux=xp.full((ny,nx),u0,dtype=dtype); uy=xp.zeros((ny,nx),dtype=dtype)
        f=xp.empty((9,ny,nx),dtype=dtype)
        # A little transverse noise seeds the wake instability. A perfectly
        # uniform inlet leaves the flow symmetric, so the twin standing eddies
        # never break down into an alternating street.
        rng=np.random.default_rng(5)
        uy=uy+xp.asarray(rng.normal(0,u0*.05,(ny,nx)).astype(np.float32),dtype=dtype)
        usq=ux*ux+uy*uy
        for q in range(9):
            cu=3*(c[q,0]*ux+c[q,1]*uy); f[q]=w[q]*rho*(1+cu+.5*cu*cu-1.5*usq)
        # A half-cell vertical offset breaks the lattice-symmetric stagnation
        # point. Without it the wake stays perfectly symmetric for a very long
        # time and the von Karman street never appears.
        mask=xp.asarray(self.build_obstacles(nx,ny,preset,custom_grid))
        if xp is not np:
            f=self._gpu_collide_initial(f,mask,u0)
        return f,c,w,mask

    # ---- fused CUDA lattice update ------------------------------------------
    def _lbm_kernel(self,bx,by):
        key=(self.ctx.precision,int(bx),int(by))
        kernel=self._lbm_kernels.get(key)
        if kernel is None:
            real="double" if self.ctx.state_dtype==np.float64 else "float"
            source=(LBM_KERNEL.replace("REAL_T",real)
                    .replace("BX",str(int(bx))).replace("BY",str(int(by))))
            kernel=self.ctx.xp.RawKernel(source,"lbm_fused")
            self._lbm_kernels[key]=kernel
        return kernel

    def _gpu_launch(self,src,dst,u0,stream,macro,config,solid=None,dummy=None):
        xp=self.ctx.xp; real=self.ctx.state_dtype
        _,ny,nx=src.shape
        bx,by=int(config["bx"]),int(config["by"])
        rho,ux,uy=macro if macro is not None else ((self._macro_dummy if dummy is None else dummy),)*3
        self._lbm_kernel(bx,by)(((nx+bx-1)//bx,(ny+by-1)//by),(bx,by),
            (src,dst,self._solid if solid is None else solid,np.int32(nx),np.int32(ny),real(u0),real(1/LBM_TAU),
             np.int32(stream),np.int32(macro is not None),rho,ux,uy))

    HALO=16      # ghost rows per side of a strip = lattice steps between exchanges

    def _multi_gpu_step(self,f,u0,steps):
        """The fused lattice update split into horizontal strips, one per GPU.

        Each GPU owns ny/k rows plus HALO ghost rows above and below, and the
        same kernel updates the whole strip.  Rows that depend on ghosts go
        stale one row per step, from the outside in, so after HALO steps only
        the ghosts themselves are stale and every GPU's own rows are exact:
        the GPUs run HALO steps back to back, then swap HALO edge rows with
        their neighbours over NVLink (a few MB).  That is 16x fewer exchanges
        than swapping one row every step, for 2*HALO extra rows of work.
        Kernels are launched asynchronously from this thread.  The macroscopic
        fields are gathered onto the first GPU once per call, for the tracers
        and the drawing.  ``f`` stays the key; the live state is in the strips.
        """
        devices=self.ctx.devices; cp=self.ctx.xp
        _,ny,nx=f.shape
        m=getattr(self,"_strips",None)
        if m is None or m["source"] is not f:
            parts=split(ny,devices.count)
            halo=max(1,min(self.HALO,min(p.stop-p.start for p in parts)))
            src,dst,solid,dummy=[],[],[],[]
            for i,p in enumerate(parts):
                rows=cp.asarray(np.arange(p.start-halo,p.stop+halo)%ny)
                src.append(devices.to(cp.ascontiguousarray(f[:,rows,:]),i))
                solid.append(devices.to(cp.ascontiguousarray(self._solid[rows,:]),i))
                with devices.device(i):
                    dst.append(cp.empty_like(src[-1]))
                    dummy.append(cp.empty(1,dtype=f.dtype))
            m=self._strips={"source":f,"parts":parts,"halo":halo,"src":src,"dst":dst,"solid":solid,"dummy":dummy}
        src,dst,halo=m["src"],m["dst"],m["halo"]
        macro=[]
        for k in range(steps):
            last=k==steps-1
            for i in range(devices.count):
                with devices.device(i):
                    out=None
                    if last:
                        shape=src[i].shape[1:]
                        out=(cp.empty(shape,dtype=f.dtype),cp.empty(shape,dtype=f.dtype),cp.empty(shape,dtype=f.dtype))
                        macro.append(out)
                    self._gpu_launch(src[i],dst[i],u0,1,out,self._lbm_config,solid=m["solid"][i],dummy=m["dummy"][i])
            src,dst=dst,src
            if (k+1)%halo==0 or last:
                devices.synchronize()
                devices.halo_exchange(src,halo=halo,axis=1,periodic=True)
        m["src"],m["dst"]=src,dst
        rho,ux,uy=(devices.gather([mac[j][halo:-halo] for mac in macro]) for j in range(3))
        vort=cp.roll(uy,-1,1)-cp.roll(uy,1,1) - (cp.roll(ux,-1,0)-cp.roll(ux,1,0))
        return f,ux,uy,vort,rho

    def _gpu_collide_initial(self,f,mask,u0):
        xp=self.ctx.xp
        self._solid=xp.asarray(mask,dtype=xp.uint8)
        self._macro_dummy=xp.empty(1,dtype=self.ctx.state_dtype)
        _,ny,nx=f.shape
        post=xp.empty_like(f); self._spare=xp.empty_like(f)
        # Tune on the real lattice: the launch is idempotent (dst = F(src)).
        candidates=[{"bx":bx,"by":by} for bx,by in
                    ((32,2),(32,4),(32,8),(32,16),(64,2),(64,4),(64,8),(128,1),(128,2),(128,4),(256,1),(256,2))]
        config,report=tuning.select(
            xp,"fluid_lbm_d2q9",f"{self.ctx.precision}|{nx}x{ny}",candidates,
            lambda cfg:self._gpu_launch(f,self._spare,u0,1,None,cfg),default={"bx":128,"by":2})
        self._lbm_config=config
        self.ctx.record_kernel("fluid_lbm_d2q9",{**report,"precision":self.ctx.precision,
                                                 "lattice":f"{nx}x{ny}"})
        # Collide the initial state in place of streaming (stream=0).
        self._gpu_launch(f,post,u0,0,None,config)
        return post

    def _gpu_step(self,f,mask,u0,steps):
        xp=self.ctx.xp
        _,ny,nx=f.shape
        dtype=self.ctx.state_dtype
        rho=xp.empty((ny,nx),dtype=dtype); ux=xp.empty_like(rho); uy=xp.empty_like(rho)
        src,dst=f,self._spare
        for k in range(steps):
            self._gpu_launch(src,dst,u0,1,(rho,ux,uy) if k==steps-1 else None,self._lbm_config)
            src,dst=dst,src
        self._spare=dst
        vort=xp.roll(uy,-1,1)-xp.roll(uy,1,1) - (xp.roll(ux,-1,0)-xp.roll(ux,1,0))
        return src,ux,uy,vort,rho

    def step(self,f,c,w,mask,u0,steps):
        if self.ctx.xp is not np:
            if self.ctx.devices.count>1:
                return self._multi_gpu_step(f,u0,steps)
            return self._gpu_step(f,mask,u0,steps)
        xp=self.ctx.xp; tau=LBM_TAU; omega=1/tau
        opposite=[0,3,4,1,2,7,8,5,6]
        for _ in range(steps):
            rho=xp.sum(f,axis=0); ux=xp.sum(f*c[:,0,None,None],axis=0)/(rho+1e-8); uy=xp.sum(f*c[:,1,None,None],axis=0)/(rho+1e-8)
            ux[:,0]=u0; uy[:,0]=0
            usq=ux*ux+uy*uy
            for q in range(9):
                cu=3*(c[q,0]*ux+c[q,1]*uy); feq=w[q]*rho*(1+cu+.5*cu*cu-1.5*usq); f[q]+=omega*(feq-f[q])
            for q in range(9): f[q]=xp.roll(xp.roll(f[q],int(c[q,1]),axis=0),int(c[q,0]),axis=1)
            old=f.copy()
            for q in range(9): f[q][mask]=old[opposite[q]][mask]
        rho=xp.sum(f,axis=0); ux=xp.sum(f*c[:,0,None,None],axis=0)/(rho+1e-8); uy=xp.sum(f*c[:,1,None,None],axis=0)/(rho+1e-8)
        ux=xp.where(mask,0,ux); uy=xp.where(mask,0,uy)
        vort=xp.roll(uy,-1,1)-xp.roll(uy,1,1) - (xp.roll(ux,-1,0)-xp.roll(ux,1,0))
        return f,ux,uy,vort,rho

    # ---- tracers -------------------------------------------------------
    # Tracers live on the same device as the lattice, so a frame never has to
    # copy the full velocity field to the host just to move 3,000 points.
    def seed_tracers(self,n,nx,ny,rng):
        x=rng.uniform(0,nx,n); y=rng.uniform(0,ny,n)
        return np.stack([x,y],axis=1)
    def sample(self,field,pts,nx,ny):
        """Bilinear sample of a lattice field at particle positions."""
        xp=self.ctx.xp
        x=xp.clip(pts[:,0],0,nx-1.001); y=xp.clip(pts[:,1],0,ny-1.001)
        x0=x.astype(xp.int32); y0=y.astype(xp.int32)
        x1=xp.minimum(x0+1,nx-1); y1=xp.minimum(y0+1,ny-1)
        fx=x-x0; fy=y-y0
        return (field[y0,x0]*(1-fx)*(1-fy)+field[y0,x1]*fx*(1-fy)
                +field[y1,x0]*(1-fx)*fy+field[y1,x1]*fx*fy)
    def advect(self,trail,ux,uy,nx,ny,rng,dt,mask,substeps=4):
        """Advance the tracers and push the new position onto their trail.

        A single position per particle produced streaks under a pixel long at
        these lattice speeds, so the frame showed a static dot field. Carrying a
        short history and drawing it as a polyline turns each particle into a
        visible streakline, which is what makes the flow direction readable in a
        still frame.
        """
        xp=self.ctx.xp
        pts=trail[:,-1,:].copy()
        for _ in range(substeps):
            vx=self.sample(ux,pts,nx,ny); vy=self.sample(uy,pts,nx,ny)
            pts=pts+xp.stack([vx,vy],axis=1)*(dt/substeps)
        ix=xp.clip(pts[:,0].astype(xp.int32),0,nx-1); iy=xp.clip(pts[:,1].astype(xp.int32),0,ny-1)
        inside=mask[iy,ix]
        stalled=xp.hypot(pts[:,0]-trail[:,-1,0],pts[:,1]-trail[:,-1,1])<1e-3
        gone=(pts[:,0]>=nx-1)|(pts[:,0]<0)|(pts[:,1]<0)|(pts[:,1]>=ny-1)|inside|stalled
        trail=xp.concatenate([trail[:,1:,:],pts[:,None,:]],axis=1)
        k=int(gone.sum())
        if k:
            # Reinject at the inlet and collapse the trail so no streak is drawn
            # spanning the whole domain.
            nx0=rng.uniform(0,3.0,k); ny0=rng.uniform(0,ny-1,k)
            trail[gone]=xp.asarray(np.stack([nx0,ny0],axis=1),dtype=trail.dtype)[:,None,:]
        return trail

    # ---- rendering -----------------------------------------------------
    def frame_payload(self,ux,uy,vort,rho,trail,nx,ny,speed):
        """Everything one frame needs, reduced on the device to output size.

        The picture is 1280x720 whatever the lattice. Copying three full
        3840x2160 fields to the host every frame cost more than the physics;
        this ships a 1280x720 image, the tracer trails and the glyph samples.
        """
        xp=self.ctx.xp
        spd=xp.sqrt(ux*ux+uy*uy)
        # Background is flow SPEED on a cool palette; vorticity tints the wake
        # so shed vortices stay readable.
        s=xp.clip(_resample(xp,spd/max(1e-6,speed*1.9),W,H),0,1)
        v=_resample(xp,xp.clip(xp.log1p(xp.abs(vort)*220)/3.2,0,1),W,H)
        bg=xp.stack([xp.clip(0.08+0.75*s*s,0,1),xp.clip(0.12+0.88*s,0,1),xp.clip(0.28+0.72*xp.sqrt(s),0,1)],-1)
        tint=xp.stack([v,v*(90/255),v*(30/255)],-1)
        image=xp.clip((bg*0.8+tint*0.2)*255,0,255).astype(xp.uint8)
        stepx=max(1,nx//26); stepy=max(1,ny//14)
        gy=xp.arange(stepy//2,ny,stepy); gx=xp.arange(stepx//2,nx,stepx)
        return {"image":to_numpy(image),"trail":to_numpy(trail).astype(np.float32),
                "gux":to_numpy(ux[gy][:,gx]),"guy":to_numpy(uy[gy][:,gx]),
                "gx":to_numpy(gx),"gy":to_numpy(gy),"nx":nx,"ny":ny,"speed":speed}

    def panel(self,im,ux,uy,rho,nx,ny,speed,mach_note):
        d=ImageDraw.Draw(im,'RGBA')
        ox,oy=self.obstacle_centre; orad=self.obstacle_radius
        # Pressure from the LBM density: p = rho * cs^2, cs^2 = 1/3.
        p=(rho-1.0)/3.0
        front=float(p[int(oy),max(0,int(ox-orad-2))])
        back=float(p[int(oy),min(nx-1,int(ox+orad+2))])
        d.rounded_rectangle((26,112,470,246),radius=16,fill=(4,9,22,205))
        d.text((44,126),"AIRFLOW  →  left to right",font=font(17,True),fill=(160,230,255))
        d.text((44,154),f"inlet speed {speed:.3f} lattice units",font=font(15),fill=(206,224,248))
        d.text((44,178),f"stagnation pressure (front)  {front:+.5f}",font=font(15),fill=(255,206,150))
        d.text((44,200),f"wake pressure (behind)       {back:+.5f}",font=font(15),fill=(150,226,255))
        d.text((44,222),mach_note,font=font(14),fill=(150,170,200))
        return im
    def budget(self):
        """Total lattice updates for the run, independent of frame count.

        Vortex shedding needs a few thousand updates to grow out of the initial
        transient; a frame-derived budget stopped at a couple of hundred, so the
        wake was still perfectly steady when the run ended.
        """
        s=self.settings
        if 'total_steps' in s: return max(1,int(s['total_steps']))
        return max(1,int(s.get('steps_per_frame',6))*self.ctx.frames)
    def run(self):
        nx,ny=int(self.settings['nx']),int(self.settings['ny'])
        speed=float(self.ctx.params.get('speed',.06))
        total=self.budget(); done=0
        preset=int(self.ctx.params.get('obstacle',0))
        custom=self.ctx.params.get('_obstacle_grid')
        f,c,w,mask=self.init(nx,ny,speed,preset,custom)
        xp=self.ctx.xp
        rng=np.random.default_rng(11)
        ntr=int(self.settings.get('tracers',900))
        K=int(self.settings.get('trail',12))
        trail=xp.asarray(np.repeat(self.seed_tracers(ntr,nx,ny,rng)[:,None,:],K,axis=1).astype(np.float32))
        # Tracers are a visualisation of the same velocity field, advanced with
        # an amplified visual time step so the streaks span useful distances.
        boost=float(self.settings.get('tracer_boost',9.0))
        obstacle=_obstacle_layers(self.obstacle_mask)
        ox,oy=self.obstacle_centre; x0,x1,_,_=self.obstacle_bounds
        re=speed*(2*self.obstacle_radius)/((.57-.5)/3)
        subtitle_backend=self.ctx.backend_name
        # Physics on the device, drawing on the allocated CPU cores, overlapped.
        with FramePipeline(self.ctx,workers=_draw_workers(self.ctx)) as frames:
            for i in range(self.ctx.frames):
                step_target=int(round(total*(i+1)/self.ctx.frames))
                spf=max(1,step_target-done); done=step_target
                f,ux,uy,vort,rho=self.step(f,c,w,mask,speed,spf)
                # Several small advection sub-steps keep streaks smooth and stop
                # particles tunnelling through the cylinder.
                trail=self.advect(trail,ux,uy,nx,ny,rng,min(spf,14)*boost,mask)
                with self.ctx.stage("visualization"):
                    payload=self.frame_payload(ux,uy,vort,rho,trail,nx,ny,speed)
                    front=float((rho[int(oy),max(0,x0-2)]-1.0)/3.0); back=float((rho[int(oy),min(nx-1,x1+2)]-1.0)/3.0)
                payload.update(obstacle=obstacle,progress=(i+1)/self.ctx.frames,
                               subtitle=f"D2Q9 lattice-Boltzmann · {nx}×{ny} cells · {subtitle_backend}")
                frames.submit(i,draw_fluid_frame,payload,self.ctx.frame_path(i),"Updating lattice cells",{
                    "inlet speed":f"{speed:.4f}","Reynolds number":f"{re:,.0f}","grid":f"{nx} × {ny}",
                    "obstacle preset":OBSTACLE_NAMES[preset],
                    "custom blocks":f"{int(np.asarray(custom).sum()) if custom else 0}",
                    "front pressure":f"{front:+.5f}","wake pressure":f"{back:+.5f}","lattice step":f"{done:,} / {total:,}",
                    "precision":self.ctx.precision_spec["label"]})
        im=Image.open(self.ctx.frame_path(self.ctx.frames-1)).convert('RGB')
        rev=im.copy(); d=ImageDraw.Draw(rev,'RGBA')
        cols,rows=4,2
        for r in range(rows):
            for cc in range(cols):
                x0,x1=cc*W/cols,(cc+1)*W/cols; y0,y1=r*H/rows,(r+1)*H/rows
                d.rectangle((x0,y0,x1,y1),outline=(124,232,255,190),width=4)
        rp=self.ctx.run_dir/'reveal.jpg'; self.ctx.save_frame(rev,rp); self.ctx.finish(rp)


def _resample(xp,a,w,h):
    """Box-average by the whole-number factor, then bilinear to exactly w x h."""
    ny,nx=a.shape
    f=max(1,min(nx//w,ny//h))
    if f>1:
        a=a[:ny//f*f,:nx//f*f].reshape(ny//f,f,nx//f,f).mean(axis=(1,3))
        ny,nx=a.shape
    ys=xp.clip((xp.arange(h,dtype=xp.float32)+.5)*ny/h-.5,0,ny-1)
    xs=xp.clip((xp.arange(w,dtype=xp.float32)+.5)*nx/w-.5,0,nx-1)
    y0=xp.floor(ys).astype(xp.int32); x0=xp.floor(xs).astype(xp.int32)
    y1=xp.minimum(y0+1,ny-1); x1=xp.minimum(x0+1,nx-1)
    fy=(ys-y0)[:,None]; fx=(xs-x0)[None,:]
    top=a[y0][:,x0]*(1-fx)+a[y0][:,x1]*fx
    bottom=a[y1][:,x0]*(1-fx)+a[y1][:,x1]*fx
    return top*(1-fy)+bottom*fy


def _draw_workers(ctx):
    """Drawing processes: the allocated cores, minus one for the solver loop."""
    return max(1,min(ctx.cpu_workers-1,16,max(1,ctx.frames//6)))


def _obstacle_layers(mask):
    """The obstacle body and its outline at output size, as small PNGs."""
    import io
    solid=Image.fromarray((mask*255).astype(np.uint8),'L').resize((W,H),Image.Resampling.NEAREST)
    edge=mask & ~(np.roll(mask,1,0)&np.roll(mask,-1,0)&np.roll(mask,1,1)&np.roll(mask,-1,1))
    outline=Image.fromarray((edge*255).astype(np.uint8),'L').resize((W,H),Image.Resampling.NEAREST).filter(ImageFilter.MaxFilter(3))
    out=[]
    for layer in (solid,outline):
        buf=io.BytesIO(); layer.save(buf,format='PNG'); out.append(buf.getvalue())
    return out


def draw_fluid_frame(p):
    """Draw one wind-tunnel frame from a frame_payload (runs in a worker)."""
    import io
    nx,ny,speed=p["nx"],p["ny"],p["speed"]
    sx,sy=W/nx,H/ny
    im=Image.fromarray(p["image"],'RGB')
    d=ImageDraw.Draw(im,'RGBA')
    # Streaklines: the tail of each tracer's path, brightening toward the
    # head, so a single still frame already shows which way the air goes.
    trail=p["trail"]; K=trail.shape[1]
    for k in range(1,K):
        a=int(38+150*(k/(K-1))**1.7); wdt=1 if k<K*.6 else 2
        seg=np.stack([trail[:,k-1,:],trail[:,k,:]],axis=1)
        jump=np.hypot(seg[:,1,0]-seg[:,0,0],seg[:,1,1]-seg[:,0,1])>nx*.25
        for (p0,p1),bad in zip(seg,jump):
            if bad: continue
            d.line((p0[0]*sx,p0[1]*sy,p1[0]*sx,p1[1]*sy),fill=(214,244,255,a),width=wdt)
    for cx,cy in trail[:,-1,:]:
        d.ellipse((cx*sx-1.7,cy*sy-1.7,cx*sx+1.7,cy*sy+1.7),fill=(255,255,255,230))
    # Direction glyphs on a coarse grid.
    for jj,j in enumerate(p["gy"]):
        for ii,i in enumerate(p["gx"]):
            vx,vy=float(p["gux"][jj,ii]),float(p["guy"][jj,ii])
            m=math.hypot(vx,vy)
            if m<speed*.09: continue
            L=min(26,7+m/max(1e-6,speed)*13)
            x0,y0=i*sx,j*sy; dx,dy=vx/m*L,vy/m*L
            a=int(90+120*min(1,m/max(1e-6,speed*1.5)))
            d.line((x0-dx*.5,y0-dy*.5,x0+dx*.5,y0+dy*.5),fill=(150,225,255,a),width=2)
            ang=math.atan2(dy,dx)
            for s in (2.5,-2.5):
                d.line((x0+dx*.5,y0+dy*.5,
                        x0+dx*.5-6*math.cos(ang+s*.4),y0+dy*.5-6*math.sin(ang+s*.4)),
                       fill=(180,236,255,a),width=2)
    # Draw the exact bounce-back mask, including every custom grid block.
    solid=Image.open(io.BytesIO(p["obstacle"][0])); outline=Image.open(io.BytesIO(p["obstacle"][1]))
    body=Image.new('RGBA',(W,H),(7,12,24,0)); body.putalpha(solid)
    im=Image.alpha_composite(im.convert('RGBA'),body)
    stroke=Image.new('RGBA',(W,H),(190,232,255,0)); stroke.putalpha(outline.point(lambda v:int(v*.82)))
    im=Image.alpha_composite(im,stroke).convert('RGB')
    im=add_title(im,"Virtual wind tunnel",p["subtitle"])
    add_progress(im,p["progress"],"LAMINAR START","VORTEX WAKE")
    return im
