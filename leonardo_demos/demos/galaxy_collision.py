from __future__ import annotations
import math
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
from .. import tuning
from ..base import Demo
from ..backend import to_numpy
from ..render import add_title, add_progress, save_frame, mosaic, font

# Working units: kpc, km/s, solar masses. One time unit is then
# 1 kpc / (km/s) = 0.9778 Gyr, which keeps the displayed clock honest.
G=4.30091e-6           # kpc (km/s)^2 / Msun
TIME_UNIT_GYR=0.97779
# Plummer softening standing in for the extended dark-matter halo. The disc
# rotation curve is derived from this same potential; deriving it from a bare
# point mass instead gave inner stars ~2000 km/s and blew the discs apart.
EPS_GAL=30.0
M31_CATALOGUE=Path(__file__).resolve().parents[2]/"data"/"m31_catalog_reduced.npz"

# Published Local Group values. Masses are virial estimates and carry large
# uncertainties; the transverse velocity in particular is still debated, so
# these are presented as "a" plausible encounter, not "the" prediction.
# van der Marel et al. 2012 (ApJ 753, 8) and Gaia-era refinements.
MW_M31=dict(
    m1=1.5e12, m2=1.5e12,          # Msun, Milky Way and Andromeda
    separation=770.0,              # kpc
    v_radial=-109.0,               # km/s, approaching
    v_transverse=17.0,             # km/s
    r1=20.0, r2=30.0,              # disc scale radii, kpc
    label="Milky Way  ×  Andromeda (M31)",
    note="770 kpc apart, closing at 109 km/s",
)

# Force evaluations per substep for each tracer integrator, and the fused CUDA
# kernel that applies them.  ``table`` holds the tabulated galaxy-centre
# positions (c1x, c1y, c2x, c2y) for every stage of every substep.
TRACER_STAGES={"leapfrog":2,"symplectic_euler":1,"murb_kinematic":1,"rk4":4}
TRACER_METHOD_IDS={"leapfrog":0,"symplectic_euler":1,"murb_kinematic":2,"rk4":3}
TRACER_KERNEL=r'''
typedef REAL_T real;
#define S STAGES

__device__ __forceinline__ void pull(real x, real y, const real* __restrict__ c,
                                     real gm1, real gm2, real eps2, real& ax, real& ay)
{
    real dx = c[0] - x, dy = c[1] - y;
    real r2 = dx * dx + dy * dy + eps2;
    real w = gm1 / (r2 * SQRT(r2));
    ax = w * dx; ay = w * dy;
    dx = c[2] - x; dy = c[3] - y;
    r2 = dx * dx + dy * dy + eps2;
    w = gm2 / (r2 * SQRT(r2));
    ax += w * dx; ay += w * dy;
}

extern "C" __global__ void __launch_bounds__(BLOCK)
advance_tracers(real* __restrict__ p, real* __restrict__ v, const real* __restrict__ table,
                const int n, const int steps, const real dt,
                const real gm1, const real gm2, const real eps2)
{
    const int i = blockIdx.x * BLOCK + threadIdx.x;
    if (i >= n) return;
    real x = p[2 * i], y = p[2 * i + 1], vx = v[2 * i], vy = v[2 * i + 1];
    const real h = (real)0.5 * dt;
    real ax, ay;
#if METHOD == 0
    pull(x, y, table, gm1, gm2, eps2, ax, ay);
#endif
    for (int s = 0; s < steps; ++s) {
        const real* c = table + 4 * S * s;
#if METHOD == 0   /* leapfrog: kick, drift, kick; closing force reused */
        vx += h * ax; vy += h * ay;
        x += dt * vx; y += dt * vy;
        pull(x, y, c + 4, gm1, gm2, eps2, ax, ay);
        vx += h * ax; vy += h * ay;
#elif METHOD == 1 /* symplectic Euler */
        pull(x, y, c, gm1, gm2, eps2, ax, ay);
        vx += dt * ax; vy += dt * ay;
        x += dt * vx; y += dt * vy;
#elif METHOD == 2 /* MUrB constant acceleration */
        pull(x, y, c, gm1, gm2, eps2, ax, ay);
        x += dt * vx + h * dt * ax; y += dt * vy + h * dt * ay;
        vx += dt * ax; vy += dt * ay;
#else             /* classical RK4 */
        real a1x, a1y, a2x, a2y, a3x, a3y, a4x, a4y;
        pull(x, y, c, gm1, gm2, eps2, a1x, a1y);
        pull(x + h * vx, y + h * vy, c + 4, gm1, gm2, eps2, a2x, a2y);
        pull(x + h * (vx + h * a1x), y + h * (vy + h * a1y), c + 8, gm1, gm2, eps2, a3x, a3y);
        pull(x + dt * (vx + h * a2x), y + dt * (vy + h * a2y), c + 12, gm1, gm2, eps2, a4x, a4y);
        x += dt * (vx + dt * (a1x + a2x + a3x) / 6);
        y += dt * (vy + dt * (a1y + a2y + a3y) / 6);
        vx += dt * (a1x + 2 * a2x + 2 * a3x + a4x) / 6;
        vy += dt * (a1y + 2 * a2y + 2 * a3y + a4y) / 6;
#endif
    }
    p[2 * i] = x; p[2 * i + 1] = y; v[2 * i] = vx; v[2 * i + 1] = vy;
}
'''


class GalaxyCollisionDemo(Demo):
    default_method="leapfrog"
    methods=("leapfrog","symplectic_euler","murb_kinematic","rk4")
    method_labels={
        "leapfrog":"Leapfrog (recommended)",
        "symplectic_euler":"Symplectic Euler",
        "murb_kinematic":"MUrB constant-acceleration",
        "rk4":"Runge–Kutta 4",
    }
    method_descriptions={
        "leapfrog":"Second-order kick–drift–kick; symplectic and stable over orbital timescales.",
        "symplectic_euler":"One force evaluation per step; fast and symplectic, but first-order accurate.",
        "murb_kinematic":"The reference repository's standard x += v·dt + ½a·dt², v += a·dt update.",
        "rk4":"Four force evaluations per step; locally fourth-order, but not symplectic.",
    }
    timing_methods={"setup":"initialization","step":"simulation","render":"render"}
    precisions=("fp32","fp64")
    id="galaxy_collision"; title="Galaxy collision"
    _tracer_kernels={}

    def __init__(self,ctx,settings):
        super().__init__(ctx,settings)
        self._tracer_blocks={}

    def setup(self,N,preset,impact,speed,tilt,transverse_velocity=None,
              milky_way_mass=None,andromeda_mass=None):
        """Return positions, velocities and the two galaxy centres."""
        if preset>=0.5:
            p=MW_M31
            m1,m2=p['m1'],p['m2']; sep=p['separation']
            vr=p['v_radial']
            vt=p['v_transverse'] if transverse_velocity is None else float(transverse_velocity)
            r1,r2=p['r1'],p['r2']; label=p['label']; note=p['note']
        else:
            m1=1.5e12*max(.2,speed); m2=1.5e12*max(.2,2-speed)
            sep=400.0+500.0*impact
            vr=-60.0-160.0*speed; vt=10.0+120.0*impact
            r1,r2=20.0,26.0
            label="Custom encounter"; note=f"{sep:.0f} kpc apart, closing at {abs(vr):.0f} km/s"
        if milky_way_mass is not None:
            m1=float(milky_way_mass)*1e12
        if andromeda_mass is not None:
            m2=float(andromeda_mass)*1e12
        c1=np.array([-sep*.5,0.0]); c2=np.array([sep*.5,0.0])
        # Split the relative velocity between the two centres about the
        # centre of mass so the pair does not drift out of frame.
        mu1=m2/(m1+m2); mu2=m1/(m1+m2)
        # vrel is galaxy 2 relative to galaxy 1, and the separation vector
        # points along +x, so a negative radial component means approaching.
        # Splitting it about the centre of mass requires v2-v1 == vrel.
        vrel=np.array([vr,vt])
        vc1=-vrel*mu1; vc2=+vrel*mu2
        n1=N//2
        # Observationally motivated morphology: the Milky Way is drawn as a
        # four-arm disc, while M31 has two prominent arms plus its star-forming
        # ring. These are tracer-density structures, not literal catalog stars.
        p1,v1=self.disc(n1,c1,vc1,m1,r1,tilt,seed=2,arms=4,ring_radius=None)
        # A curator can place a reduced, deprojected M31 star catalogue in
        # data/m31_catalog_reduced.npz.  It then supplies the M31 tracer
        # positions; otherwise the explicitly labelled morphology model keeps
        # the demo self-contained.
        catalogued=self.catalogue_disc(N-n1,c2,vc2,m2,r2,seed=7)
        p2,v2=catalogued if catalogued is not None else self.disc(
            N-n1,c2,vc2,m2,r2,-tilt*.7,seed=7,arms=2,ring_radius=10.0)
        xp=self.ctx.xp; dtype=self.ctx.state_dtype
        return (xp.asarray(np.vstack([p1,p2]),dtype=dtype),
                xp.asarray(np.vstack([v1,v2]),dtype=dtype),
                xp.asarray(c1,dtype=dtype),xp.asarray(c2,dtype=dtype),
                xp.asarray(vc1,dtype=dtype),xp.asarray(vc2,dtype=dtype),
                m1,m2,label,note,sep)

    def disc(self,n,centre,vel,mass,scale,tilt,seed,arms=2,ring_radius=None):
        """A rotating exponential disc with visible arm/ring density structure."""
        rng=np.random.default_rng(seed)
        r=np.clip(rng.gamma(2.0,scale*.45,n),scale*.06,scale*2.6)
        a=rng.uniform(0,2*np.pi,n)
        # Most particles are concentrated around gently winding logarithmic-like
        # arms. The remaining diffuse particles retain a believable disc.
        arm_mask=rng.random(n)<.78
        arm_id=rng.integers(0,arms,n)
        winding=2*np.pi*1.28*r/(scale*2.6)
        a[arm_mask]=2*np.pi*arm_id[arm_mask]/arms+winding[arm_mask]+rng.normal(0,.20,n)[arm_mask]
        if ring_radius is not None:
            ring_mask=rng.random(n)<.24
            r[ring_mask]=np.clip(rng.normal(ring_radius,scale*.11,ring_mask.sum()),scale*.10,scale*2.6)
            a[ring_mask]=rng.uniform(0,2*np.pi,ring_mask.sum())
        ct=math.cos(math.radians(tilt))
        x=r*np.cos(a); y=r*np.sin(a)*ct
        pts=np.stack([x,y],axis=1)+np.asarray(centre)
        # Circular speed in the softened (Plummer) potential actually used by
        # accel(), so the discs start in equilibrium instead of flying apart.
        vcirc=r*np.sqrt(G*mass/np.power(r*r+EPS_GAL*EPS_GAL,1.5))
        tang=np.stack([-np.sin(a),np.cos(a)*ct],axis=1)
        disp=rng.normal(0,.06,(n,2))*vcirc[:,None]
        return pts,tang*vcirc[:,None]+disp+np.asarray(vel)

    def catalogue_disc(self,n,centre,vel,mass,scale,seed):
        """Sample a curator-reduced M31 catalogue, if one is installed.

        The reduction tool writes deprojected x/y positions in kpc and one
        representative mass/brightness per spatial cell.  Sampling from that
        distribution lets a large public catalogue preserve its actual arms
        and ring without pretending that every catalogue row is dynamically
        integrated.  The gravitational model remains the same restricted
        N-body approximation used by the synthetic fallback.
        """
        if not M31_CATALOGUE.exists():
            return None
        try:
            data=np.load(M31_CATALOGUE)
            xy=np.asarray(data['xy_kpc'],dtype=np.float64)
            weight=np.asarray(data.get('weight',np.ones(len(xy))),dtype=np.float64)
            good=np.isfinite(xy).all(axis=1)&np.isfinite(weight)&(weight>0)
            xy,weight=xy[good],weight[good]
            if len(xy)<32:
                return None
            rng=np.random.default_rng(seed)
            # Weighted resampling conserves the catalogue's luminous/mass
            # emphasis while keeping the number of dynamical bodies bounded.
            pick=rng.choice(len(xy),size=n,replace=len(xy)<n,p=weight/weight.sum())
            xy=xy[pick].copy()
            r=np.hypot(xy[:,0],xy[:,1])
            extent=max(float(np.percentile(r,95)),1e-3)
            xy*=scale*2.35/extent
            r=np.maximum(np.hypot(xy[:,0],xy[:,1]),scale*.04)
            a=np.arctan2(xy[:,1],xy[:,0])
            vcirc=r*np.sqrt(G*mass/np.power(r*r+EPS_GAL*EPS_GAL,1.5))
            tang=np.stack([-np.sin(a),np.cos(a)],axis=1)
            disp=rng.normal(0,.045,(n,2))*vcirc[:,None]
            return xy+np.asarray(centre),tang*vcirc[:,None]+disp+np.asarray(vel)
        except Exception:
            # A malformed optional asset must never stop exhibition playback.
            return None

    def accel(self,p,c1,c2,m1,m2,eps=EPS_GAL):
        xp=self.ctx.xp; a=xp.zeros_like(p)
        for c,m in ((c1,m1),(c2,m2)):
            d=c[None,:]-p; rr=xp.sum(d*d,axis=1)+eps*eps
            a+=G*m*d/(rr[:,None]**1.5)
        return a
    def centre_accel(self,c1,c2,m1,m2,eps=EPS_GAL):
        xp=self.ctx.xp; d=c2-c1; rr=xp.sum(d*d)+eps*eps
        a=G*d/(rr**1.5)
        return a*m2,-a*m1
    def particle_accel(self,p,c1,c2,m1,m2):
        """Evaluate tracer acceleration while honouring CPU chunk allocation."""
        xp=self.ctx.xp
        out=xp.empty_like(p)
        def evaluate(part):
            return part,self.accel(p[part],c1,c2,m1,m2)
        for part,value in self.ctx.parallel_slices(len(p),evaluate):
            out[part]=value
        return out
    def step(self,p,v,c1,c2,vc1,vc2,m1,m2,dt,steps):
        """Advance the coupled tracer/galaxy state with the selected solver.

        In this restricted N-body model the tracers feel the two galaxy centres
        but never pull on them, so the centres' own orbit is independent of the
        tracers.  It is integrated first, in FP64 on the host, with the same
        scheme, and every centre position a tracer force evaluation needs is
        stored in a small table.  The tracers are then advanced through all
        ``steps`` substeps at once: on CUDA by one fused kernel that keeps each
        tracer in registers, on CPU in disjoint chunks across the allocated
        cores.  This is mathematically the same coupled update as stepping
        everything together.
        """
        method=self.ctx.method
        method={"default":"leapfrog","euler":"symplectic_euler"}.get(method,method)
        if method not in TRACER_STAGES:
            raise ValueError(f"unknown galaxy-collision solver: {method}")
        steps=int(steps); dt=float(dt)
        table,final=self._centre_table(c1,c2,vc1,vc2,m1,m2,dt,steps,method)
        if self.ctx.xp is not np:
            self._advance_gpu(p,v,table,m1,m2,dt,method)
        else:
            dtype=p.dtype
            table=table.astype(dtype)
            self.ctx.parallel_slices(
                len(p),lambda part:self._advance_cpu(p[part],v[part],table,m1,m2,dt,method))
        xp=self.ctx.xp
        for array,value in zip((c1,c2,vc1,vc2),final):
            array[...]=xp.asarray(value,dtype=array.dtype)
        return p,v,c1,c2,vc1,vc2

    @staticmethod
    def _host_centre_accel(c1,c2,m1,m2,eps=EPS_GAL):
        d=c2-c1; rr=float(d@d)+eps*eps
        a=G*d/rr**1.5
        return a*m2,-a*m1

    def _centre_table(self,c1,c2,vc1,vc2,m1,m2,dt,steps,method):
        """Integrate the two centres and tabulate every force-evaluation point.

        Returns ``(table, final)``: ``table[s, k]`` holds (c1x, c1y, c2x, c2y)
        for force stage ``k`` of substep ``s``, and ``final`` the advanced
        centre positions and velocities.
        """
        c1,c2,vc1,vc2=(np.array(to_numpy(x),dtype=np.float64) for x in (c1,c2,vc1,vc2))
        acc=lambda a,b:self._host_centre_accel(a,b,m1,m2)
        table=np.empty((steps,TRACER_STAGES[method],4),dtype=np.float64)
        for s in range(steps):
            table[s,0]=np.concatenate((c1,c2))
            if method=="leapfrog":
                a1,a2=acc(c1,c2)
                vc1+=.5*dt*a1; vc2+=.5*dt*a2
                c1+=dt*vc1; c2+=dt*vc2
                a1,a2=acc(c1,c2)
                vc1+=.5*dt*a1; vc2+=.5*dt*a2
                table[s,1]=np.concatenate((c1,c2))
            elif method=="symplectic_euler":
                a1,a2=acc(c1,c2)
                vc1+=dt*a1; vc2+=dt*a2
                c1+=dt*vc1; c2+=dt*vc2
            elif method=="murb_kinematic":
                a1,a2=acc(c1,c2)
                c1+=dt*vc1+.5*dt*dt*a1; c2+=dt*vc2+.5*dt*dt*a2
                vc1+=dt*a1; vc2+=dt*a2
            else:  # classical RK4 on (c1, c2, vc1, vc2)
                k1=(vc1,vc2,*acc(c1,c2))
                s2=(c1+.5*dt*k1[0],c2+.5*dt*k1[1],vc1+.5*dt*k1[2],vc2+.5*dt*k1[3])
                k2=(s2[2],s2[3],*acc(s2[0],s2[1]))
                s3=(c1+.5*dt*k2[0],c2+.5*dt*k2[1],vc1+.5*dt*k2[2],vc2+.5*dt*k2[3])
                k3=(s3[2],s3[3],*acc(s3[0],s3[1]))
                s4=(c1+dt*k3[0],c2+dt*k3[1],vc1+dt*k3[2],vc2+dt*k3[3])
                k4=(s4[2],s4[3],*acc(s4[0],s4[1]))
                for k,stage in enumerate((s2,s3,s4),start=1):
                    table[s,k]=np.concatenate(stage[:2])
                c1=c1+dt*(k1[0]+2*k2[0]+2*k3[0]+k4[0])/6
                c2=c2+dt*(k1[1]+2*k2[1]+2*k3[1]+k4[1])/6
                vc1=vc1+dt*(k1[2]+2*k2[2]+2*k3[2]+k4[2])/6
                vc2=vc2+dt*(k1[3]+2*k2[3]+2*k3[3]+k4[3])/6
        return table,(c1,c2,vc1,vc2)

    def _advance_cpu(self,p,v,table,m1,m2,dt,method):
        """Advance one contiguous chunk of tracers in place (NumPy path)."""
        accel=lambda pp,c:self.accel(pp,c[0:2],c[2:4],m1,m2)
        a=None
        for stages in table:
            if method=="leapfrog":
                # A substep's closing force is the next one's opening force:
                # same positions and, exactly, the same tabulated centres.
                if a is None: a=accel(p,stages[0])
                v+=.5*dt*a; p+=dt*v
                a=accel(p,stages[1]); v+=.5*dt*a
            elif method=="symplectic_euler":
                v+=dt*accel(p,stages[0]); p+=dt*v
            elif method=="murb_kinematic":
                a=accel(p,stages[0]); p+=dt*v+.5*dt*dt*a; v+=dt*a
            else:
                a1=accel(p,stages[0])
                a2=accel(p+.5*dt*v,stages[1])
                a3=accel(p+.5*dt*(v+.5*dt*a1),stages[2])
                a4=accel(p+dt*(v+.5*dt*a2),stages[3])
                p+=dt*(v+dt*(a1+a2+a3)/6)
                v+=dt*(a1+2*a2+2*a3+a4)/6
        return None

    def _advance_gpu(self,p,v,table,m1,m2,dt,method):
        xp=self.ctx.xp; n=len(p)
        real=self.ctx.state_dtype
        table=xp.asarray(table,dtype=real)
        args=lambda pp,vv:(pp,vv,table,np.int32(n),np.int32(len(table)),real(dt),
                           real(G*m1),real(G*m2),real(EPS_GAL*EPS_GAL))
        key=(method,tuning.size_bucket(n))
        if key not in self._tracer_blocks:
            # The kernel updates in place, so candidates are timed on copies.
            sp,sv=p.copy(),v.copy()
            config,report=tuning.select(
                xp,f"galaxy2d_tracers_{method}",f"{self.ctx.precision}|n{key[1]}",
                [{"block":b} for b in (64,128,256,512,1024)],
                lambda c:self._tracer_kernel(method,c["block"])(
                    ((n+c["block"]-1)//c["block"],),(c["block"],),args(sp,sv)),
                default={"block":256})
            self._tracer_blocks[key]=int(config["block"])
            self.ctx.record_kernel(f"galaxy2d_tracers_n{n}",{**report,"precision":self.ctx.precision,
                                                             "method":method,"tracers":n})
        block=self._tracer_blocks[key]
        self._tracer_kernel(method,block)(((n+block-1)//block,),(block,),args(p,v))

    def _tracer_kernel(self,method,block):
        key=(self.ctx.precision,method,int(block))
        kernel=self._tracer_kernels.get(key)
        if kernel is None:
            real="double" if self.ctx.state_dtype==np.float64 else "float"
            source=(TRACER_KERNEL.replace("REAL_T",real)
                    .replace("SQRT",("sqrt" if real=="double" else "sqrtf"))
                    .replace("METHOD",str(TRACER_METHOD_IDS[method]))
                    .replace("STAGES",str(TRACER_STAGES[method]))
                    .replace("BLOCK",str(int(block))))
            kernel=self.ctx.xp.RawKernel(source,"advance_tracers")
            self._tracer_kernels[key]=kernel
        return kernel

    def render(self,p,half,n1,m1,m2,size=(1280,720)):
        pts=to_numpy(p); W,H=size
        im=Image.new('RGB',size,(2,5,13))
        # Square world window mapped to a 16:9 frame without distortion.
        halfx=half*W/H
        x=((pts[:,0]+halfx)/(2*halfx)*W).astype(int)
        y=((1-(pts[:,1]+half)/(2*half))*H).astype(int)
        good=(x>=0)&(x<W)&(y>=0)&(y<H)
        origin=(np.arange(len(pts))>=n1)[good]
        flat=(y[good]*W+x[good]).astype(np.int64)
        # Each tracer represents an equal share of its parent galaxy mass.
        # Accumulating those shares is a surface-mass estimate, so bright areas
        # genuinely correspond to many overlapping tracer masses, not overdraw.
        mass=np.where(origin,m2/max(1,len(pts)-n1),m1/max(1,n1))
        d_mw=np.bincount(flat[~origin],weights=mass[~origin],minlength=W*H).reshape(H,W)
        d_m31=np.bincount(flat[origin],weights=mass[origin],minlength=W*H).reshape(H,W)
        def glow_field(field):
            nz=field[field>0]
            if not len(nz): return np.zeros_like(field,dtype=np.float32)
            ref=max(float(np.percentile(nz,99.5)),1.0)
            v=np.log1p(field/(ref*.05))/np.log1p(ref/(ref*.05))
            v=np.clip(v,0,1).astype(np.float32)
            soft=np.asarray(Image.fromarray((v*255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(2.2)),dtype=np.float32)/255
            return np.maximum(v*.68,soft)
        mw,m31=glow_field(d_mw),glow_field(d_m31)
        overlap=np.minimum(mw,m31)
        base=np.zeros((H,W,3),dtype=np.float32)+np.array([2,5,13],dtype=np.float32)
        # Blue-white Milky Way and warm Andromeda components remain visible as
        # they interpenetrate; overlap blooms toward white rather than hiding.
        base+=mw[...,None]*np.array([62,154,255],dtype=np.float32)
        base+=m31[...,None]*np.array([255,105,54],dtype=np.float32)
        base+=overlap[...,None]*np.array([130,150,110],dtype=np.float32)
        im=Image.fromarray(np.clip(base,0,255).astype(np.uint8),'RGB')
        layer=Image.new('RGBA',size,(0,0,0,0)); d=ImageDraw.Draw(layer,'RGBA')
        stride=max(1,len(flat)//60000)
        for xx,yy,is_m31 in zip(x[good][::stride],y[good][::stride],origin[::stride]):
            color=(255,184,120,170) if is_m31 else (178,222,255,170)
            d.ellipse((xx-1,yy-1,xx+1,yy+1),fill=color)
        im=Image.alpha_composite(im.convert('RGBA'),layer).convert('RGB')
        return im

    def run(self):
        N=int(self.settings['particles'])
        preset=float(self.ctx.params.get('preset',1))
        impact=float(self.ctx.params.get('impact',.55))
        speed=float(self.ctx.params.get('speed',.75))
        tilt=float(self.ctx.params.get('tilt',18))
        mw_mass=float(self.ctx.params.get('milky_way_mass',1.5))
        m31_mass=float(self.ctx.params.get('andromeda_mass',1.5))
        p,v,c1,c2,vc1,vc2,m1,m2,label,note,sep=self.setup(
            N,preset,impact,speed,tilt,milky_way_mass=mw_mass,
            andromeda_mass=m31_mass)
        # Integrate far enough to pass through the merger (~6 Gyr for MW/M31).
        span_gyr=float(self.settings.get('span_gyr',7.5))
        total=span_gyr/TIME_UNIT_GYR
        substeps=int(self.settings.get('substeps',6))
        dt=total/(self.ctx.frames*substeps)
        half=sep*.85
        for i in range(self.ctx.frames):
            p,v,c1,c2,vc1,vc2=self.step(p,v,c1,c2,vc1,vc2,m1,m2,dt,substeps)
            t_gyr=(i+1)/self.ctx.frames*span_gyr
            pts=to_numpy(p)
            centres=to_numpy(self.ctx.xp.stack((c1,c2)))
            # Track the encounter: the frame follows the shrinking separation
            # so the merger does not happen inside two pixels.
            # Radial percentile, not per-coordinate: taking |x| and |y|
            # together is dominated by the many small y values and framed the
            # pair too tightly, clipping both galaxies off the edges. The 90th
            # percentile still ignores the few ejected tracers.
            reach=float(np.percentile(np.hypot(pts[:,0],pts[:,1]),92))
            half=float(np.clip(.85*half+.15*reach*1.45,70.0,sep*1.6))
            im=self.render(pts,half,N//2,m1,m2)
            cpu_note=f" · CPU×{self.ctx.cpu_workers}" if self.ctx.xp is np else ""
            solver=self.method_labels.get(self.ctx.method,self.ctx.method)
            im=add_title(im,"Galaxy collision",
                         f"restricted N-body · {solver} · M1={m1/1e12:.1f}e12 Msun · M2={m2/1e12:.1f}e12 Msun · {self.ctx.backend_name}")
            add_progress(im,(i+1)/self.ctx.frames,"FIRST APPROACH","MERGER / TIDAL DEBRIS")
            separation=float(np.linalg.norm(centres[1]-centres[0]))
            self.ctx.save_frame(im,self.ctx.frame_path(i)); self.ctx.write_status(i,f"t=+{t_gyr:.2f} Gyr",{
                "encounter":label,"time":f"+{t_gyr:.2f} Gyr","separation":f"{separation:.0f} kpc",
                "solver":solver,"tracers":f"{N:,}","view width":f"{2*half:,.0f} kpc",
                "saved-frame interval":f"{span_gyr/self.ctx.frames*1000:.1f} Myr",
                "solver step":f"{dt*TIME_UNIT_GYR*1000:.2f} Myr × {substeps}",
                "precision":self.ctx.precision_spec["label"],
                "compute":f"{self.ctx.backend_name}{cpu_note}"})
        # Reveal: the same encounter under the observational uncertainty on the
        # transverse velocity, which is what actually decides the outcome.
        ens=max(1,int(self.ctx.params.get('_parallel_count',self.settings.get('ensemble',16))))
        side=max(1,int(math.ceil(math.sqrt(ens)))); ims=[]; labels=[]
        reveal_frames=max(10,self.ctx.frames//2)
        reveal_dt=total/(reveal_frames*substeps)
        for j in range(ens):
            vt=0.0+j*(80.0/max(1,ens-1))
            # Rebuild the complete state with this measured transverse speed.
            # Updating only the galaxy-centre velocities would leave the disc
            # tracers moving with the old centres and invert the encounter.
            st=self.setup(max(2500,N//8),1,impact,speed,tilt,
                          transverse_velocity=vt,milky_way_mass=mw_mass,
                          andromeda_mass=m31_mass)
            pp,vv,cc1,cc2,vv1,vv2=st[0],st[1],st[2],st[3],st[4],st[5]
            for _ in range(reveal_frames):
                pp,vv,cc1,cc2,vv1,vv2=self.step(
                    pp,vv,cc1,cc2,vv1,vv2,st[6],st[7],reveal_dt,substeps)
            ims.append(self.render(pp,520,len(pp)//2,st[6],st[7],(260,146)))
            labels.append(f"vT {vt:.0f} km/s")
        rev=mosaic(ims,side,title="There was never only one possible collision",
                   subtitle="The transverse velocity is measured to ±tens of km/s — and it decides the outcome.",
                   labels=labels,label_fill=(190,226,255))
        rp=self.ctx.run_dir/'reveal.jpg'; self.ctx.save_frame(rev,rp); self.ctx.finish(rp)

    @staticmethod
    def _smoothstep(a,b,x):
        x=np.clip((x-a)/(b-a),0,1)
        return x*x*(3-2*x)
