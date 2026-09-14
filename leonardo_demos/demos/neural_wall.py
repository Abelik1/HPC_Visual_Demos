from __future__ import annotations
import io, math, warnings
import numpy as np
from PIL import Image, ImageDraw, ImageOps
from ..base import Demo
from ..backend import torch_device
from ..render import font

# Networks are laid out as a 2-D hyperparameter grid: learning rate varies
# along a row, hidden width varies down a column. That makes the reveal wall
# readable as an experiment rather than as a random scatter of thumbnails.
LR_LOG_RANGE=(-3.2,-1.4)
WIDTH_RANGE=(6,40)
# Fourier features: (x,y) is expanded into sin/cos waves at FOURIER_FEATURES
# random 2-D frequencies before it reaches the network. Plain tanh layers fed
# raw coordinates can only build smooth blobs ("spectral bias"); the waves give
# them ready-made high-frequency building blocks, so edges and texture appear.
# FOURIER_SIGMA is the spread of those frequencies in cycles per unit of the
# [-1,1] coordinate. Measured on a 131x161 photo with a width-40 network: plain
# x,y 22.6 dB PSNR, sigma 0.8-2.8 gave 31.4-32.2 dB, sigma 8 fell to 25.5 dB.
# Too wide a spread hands a small network waves it cannot combine and the
# reconstruction turns to noise. It depends on image content, not pixel count.
FOURIER_FEATURES=32
FOURIER_SIGMA=1.6


def fourier_matrix(features,seed=0):
    """Fixed random frequencies (cycles per unit of the [-1,1] coordinate)."""
    return (np.random.default_rng(seed).normal(size=(2,features))*FOURIER_SIGMA).astype(np.float32)


def encode(coords,B):
    """Map (..., 2) coordinates to their (..., 2*features) Fourier encoding."""
    if B is None: return coords.astype(np.float32)
    phase=2*np.pi*coords@B
    return np.concatenate([np.sin(phase),np.cos(phase)],-1).astype(np.float32)


def parameter_count(width,fourier=True):
    """Learned numbers in one network: its size as a compressed image."""
    inputs=2*FOURIER_FEATURES if fourier else 2
    return inputs*width+width + width*width+width + width*3+3


# ---- image compression accounting ------------------------------------------
# A coordinate network is a lossy image codec: its weights are the file. These
# helpers keep the demo's claims honest. The picture is counted as raw 8-bit
# RGB, the network as the float32 weights it actually stores (the Fourier
# frequencies come from a fixed seed, so a decoder would regenerate them).

def image_bytes(side):
    """Raw size of a side x side 8-bit RGB picture."""
    return int(side)*int(side)*3


def network_bytes(width,fourier=True):
    return parameter_count(int(width),fourier)*4


def psnr(mse):
    """Peak signal-to-noise ratio in dB for values in [0, 1]."""
    return 10*math.log10(1/max(float(mse),1e-10))


def quality_words(db):
    if db>=35: return "almost identical"
    if db>=30: return "very close"
    if db>=25: return "clearly recognisable"
    if db>=20: return "blurry"
    return "barely recognisable"


def format_bytes(n):
    n=float(n)
    return f"{n:,.0f} B" if n<1024 else f"{n/1024:,.1f} KB" if n<1024*1024 else f"{n/1024/1024:,.2f} MB"


def jpeg_at_size(rgb_uint8,max_bytes):
    """The best JPEG of the same picture that fits in max_bytes.

    Returns (bytes, quality setting, PSNR) or None when even the lowest
    quality is bigger. A network that loses to this is not a good codec, and
    the viewer says so rather than implying neural compression is magic.
    """
    image=Image.fromarray(rgb_uint8); reference=rgb_uint8.astype(np.float32)/255
    best=None; lo,hi=1,95
    while lo<=hi:
        q=(lo+hi)//2; buf=io.BytesIO(); image.save(buf,'JPEG',quality=q); size=buf.tell()
        if size<=max_bytes:
            buf.seek(0); decoded=np.asarray(Image.open(buf).convert('RGB'),dtype=np.float32)/255
            best=(size,q,psnr(np.mean((decoded-reference)**2))); lo=q+1
        else:
            hi=q-1
    return best


def compression_champion(losses,widths,side,fourier=True):
    """Best reconstruction among networks that are smaller than the picture.

    Returns (index, any_network_compresses). If no network is smaller, the
    overall best is returned and the caller must say it is not compressing.
    """
    sizes=np.array([network_bytes(w,fourier) for w in widths])
    pool=np.flatnonzero(sizes<image_bytes(side))
    compresses=bool(len(pool))
    if not compresses: pool=np.arange(len(losses))
    return int(pool[np.argmin(np.asarray(losses)[pool])]),compresses


def grid_shape(networks):
    cols=max(4,int(math.ceil(math.sqrt(networks))))
    return cols,int(math.ceil(networks/cols))


def hyperparameters(networks):
    cols,rows=grid_shape(networks)
    lrs=np.zeros(networks,dtype=np.float32); widths=np.zeros(networks,dtype=np.int64)
    lr_axis=10**np.linspace(*LR_LOG_RANGE,cols)
    w_axis=np.unique(np.round(np.linspace(*WIDTH_RANGE,rows)).astype(int))
    if len(w_axis)<rows: w_axis=np.resize(w_axis,rows)
    for i in range(networks):
        r,c=divmod(i,cols)
        lrs[i]=lr_axis[c]; widths[i]=w_axis[min(r,len(w_axis)-1)]
    return lrs,widths,cols,rows


class TorchWall:
    """All networks trained simultaneously as stacked weight tensors.

    Training N independent nn.Sequential models in a Python loop is what the
    demo used to do; it scales terribly and hides the point. Batching the
    weights into (N, in, out) tensors makes the ensemble a single set of
    matmuls, which is exactly the structure that maps onto a GPU and the story
    the exhibition is trying to tell.
    """
    def __init__(self,torch,networks,tile,target,seed=0,device=None,fourier=True):
        self.torch=torch; self.n=networks; self.tile=tile
        # The demo passes a tile x tile target, but any (h, w, 3) image works.
        self.shape=target.shape[:2]
        self.lrs,self.widths,_,_=hyperparameters(networks)
        self.B=fourier_matrix(FOURIER_FEATURES,seed) if fourier else None
        self.maxw=int(self.widths.max())
        # Honour the run's compute request. Taking CUDA whenever it happened to
        # be present meant a run explicitly asked to stay on the CPU reported a
        # CPU backend while training on the GPU.
        dev=device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.device=dev
        g=torch.Generator(device='cpu').manual_seed(seed)
        def par(*shape,gain=1.0):
            t=((torch.rand(*shape,generator=g)*2-1)*gain).to(dev)
            t.requires_grad_(True); return t
        w=self.maxw
        inputs=2 if self.B is None else 2*FOURIER_FEATURES
        # Keep first-layer pre-activations near unit scale whatever the input
        # width, otherwise 64 Fourier inputs would saturate every tanh.
        gain=.9 if self.B is None else math.sqrt(6/inputs)
        # The final layer produces red, green and blue rather than a single
        # brightness. A coordinate now maps (x,y) -> (r,g,b).
        self.params=[par(networks,inputs,w,gain=gain),par(networks,1,w,gain=.1),
                     par(networks,w,w,gain=.35),par(networks,1,w,gain=.1),
                     par(networks,w,3,gain=.6),par(networks,1,3,gain=.1)]
        # Unused hidden units are masked off, so a "width 6" network really has
        # the capacity of six neurons even though every network shares tensors.
        self.mask=(torch.arange(w,device=dev)[None,:]<torch.tensor(self.widths,device=dev)[:,None]).float()
        h,wd=self.shape
        coords=np.stack(np.meshgrid(np.linspace(-1,1,wd),np.linspace(-1,1,h),indexing='xy'),-1)
        # ``expand`` gives every network a zero-stride batch dimension.  That is
        # normally harmless, but CUDA's strided batched GEMM rejects it on some
        # driver/cuBLAS combinations.  Materialise the small teaching batch so
        # GPU and CPU runs follow the same path reliably.
        self.X=torch.tensor(encode(coords.reshape(-1,2),self.B),device=dev)[None].repeat(networks,1,1).contiguous()
        self.Y=torch.tensor(target.reshape(-1,3).astype(np.float32),device=dev)[None].repeat(networks,1,1).contiguous()
        self.lr=torch.tensor(self.lrs,device=dev)[:,None,None]
        self.m=[torch.zeros_like(p) for p in self.params]
        self.v=[torch.zeros_like(p) for p in self.params]
        self.t=0
    def forward(self):
        torch=self.torch; W1,b1,W2,b2,W3,b3=self.params; m=self.mask[:,None,:]
        h=torch.tanh(self.X@W1+b1)*m
        h=torch.tanh(h@W2+b2)*m
        return torch.sigmoid(h@W3+b3)
    def __call__(self,steps):
        torch=self.torch
        for _ in range(max(1,steps)):
            pred=self.forward()
            per=((pred-self.Y)**2).mean(dim=(1,2))
            per.sum().backward()
            self.t+=1
            with torch.no_grad():
                for k,p in enumerate(self.params):
                    g=p.grad
                    self.m[k]=.9*self.m[k]+.1*g
                    self.v[k]=.999*self.v[k]+.001*g*g
                    mh=self.m[k]/(1-.9**self.t); vh=self.v[k]/(1-.999**self.t)
                    p-=self.lr*mh/(vh.sqrt()+1e-8)
                    p.grad=None
        with torch.no_grad():
            out=self.forward().reshape(self.n,*self.shape,3).cpu().numpy()
            losses=((self.forward()-self.Y)**2).mean(dim=(1,2)).cpu().numpy()
        return list(out),np.asarray(losses),f"torch·{self.device}"
    def visual_state(self,index):
        """A compact snapshot of the winning network's real learned weights."""
        with self.torch.no_grad():
            w1=self.params[0][index].detach().cpu().numpy()
            w2=self.params[2][index].detach().cpu().numpy()
            w3=self.params[4][index].detach().cpu().numpy()
        if self.B is not None:
            # The diagram draws two input nodes, x and y. Through the Fourier
            # layer the true x->hidden link is the derivative of each hidden
            # pre-activation with respect to x (and y) at the image centre:
            # d/dx sum_k W_sin[k]*sin(2*pi*B_k.v) = 2*pi*sum_k B[0,k]*W_sin[k].
            w1=2*np.pi*self.B@w1[:FOURIER_FEATURES]
        return {'width':int(self.widths[index]),'w1':w1,'w2':w2,'w3':w3,'fourier':self.B is not None}


class SurrogateWall:
    """Deterministic stand-in used only when PyTorch is unavailable.

    Each network approaches its *own* quality ceiling. The previous version
    clipped a shared progress value at 1.0, so every network converged to a
    byte-identical image within a fifth of the run and the reveal became a wall
    of duplicates with the same loss printed on all of them.
    """
    def __init__(self,networks,tile,target):
        self.n=networks; self.tile=tile; self.target=target
        self.lrs,self.widths,_,_=hyperparameters(networks)
        self.f=np.fft.rfft2(target,axes=(0,1))
        yy=np.fft.fftfreq(tile)[:,None]; xx=np.fft.rfftfreq(tile)[None,:]
        self.kk=np.sqrt(xx*xx+yy*yy)
        lg=np.log10(self.lrs)
        peak=(LR_LOG_RANGE[0]+LR_LOG_RANGE[1])/2
        # Too small a learning rate converges slowly; too large never settles.
        # Both are permanent handicaps, not just slower starts.
        self.rate=.010+.055*np.exp(-((lg-peak)/.55)**2)
        capacity=(self.widths-WIDTH_RANGE[0])/max(1,WIDTH_RANGE[1]-WIDTH_RANGE[0])
        self.ceiling=np.clip(.30+.62*capacity-.22*np.abs(lg-peak),.12,.97)
        self.progress=np.zeros(networks)
    def __call__(self,steps):
        self.progress+=self.rate*max(1,steps)*(self.ceiling-self.progress).clip(0)
        outs=[];losses=[]
        for p in self.progress:
            cutoff=.015+.48*p
            filt=np.exp(-(self.kk/(cutoff+1e-6))**6)
            out=np.clip(np.fft.irfft2(self.f*filt[...,None],s=self.target.shape[:2],axes=(0,1)).real,0,1)
            outs.append(out); losses.append(float(np.mean((out-self.target)**2)))
        return outs,np.asarray(losses),'numpy-surrogate'
    def visual_state(self,index):
        return {'width':int(self.widths[index]),'w1':None,'w2':None,'w3':None}


class NeuralWallDemo(Demo):
    # The id stays neural_wall so saved runs and links keep working.
    id="neural_wall"; title="Neural image compression"
    backend_kind="torch"
    timing_methods={"compose_frame":"render","wall_image":"render"}
    def drawn_target(self,n,path):
        """Load a visitor's canvas drawing as the coordinate-network target."""
        with Image.open(path) as image:
            # Centre-crop to a square rather than squashing a non-square photo.
            image=ImageOps.fit(image.convert('RGB'),(n,n),Image.Resampling.LANCZOS)
            return np.asarray(image,dtype=np.float32)/255.0
    def target(self,n,kind=0,difficulty=1.0):
        y,x=np.mgrid[-1:1:complex(n),-1:1:complex(n)]
        difficulty=float(difficulty)
        if int(kind)%4==0:
            z=.5+.5*np.sin(8*difficulty*(x*x+y*y)+5*difficulty*np.arctan2(y,x)); z*=np.exp(-.4*(x*x+y*y));
            rgb=np.stack([.20+.80*z,.05+.55*z*z,.28+.72*(1-z)*np.exp(-.22*(x*x+y*y))],axis=-1)
        elif int(kind)%4==1:
            rgb=np.stack([.5+.5*np.sin(8*difficulty*x),.5+.5*np.cos(8*difficulty*y),.5+.5*np.sin(6*difficulty*(x+y))],axis=-1)
        elif int(kind)%4==2:
            z=np.exp(-5*difficulty*((np.sqrt(x*x+y*y)-.52)**2))*(.45+.55*np.cos(6*difficulty*np.arctan2(y,x))**2)
            rgb=np.stack([.95*z,.22+.68*z,.06+.62*(1-z)],axis=-1)
        else:
            z=.5+.5*np.sin(7*difficulty*x+4*np.sin(5*difficulty*y))
            rgb=np.stack([.12+.82*z,.12+.56*(1-z),.34+.62*np.sin(z*math.pi)**2],axis=-1)
        return np.clip(rgb,0,1).astype(np.float32)
    def make_trainer(self,networks,tile,target):
        try:
            import torch
            req=getattr(self.ctx,'backend_requested','auto') if self.ctx else 'auto'
            fourier=bool(int(self.ctx.params.get('fourier',1))) if self.ctx else True
            trainer=TorchWall(torch,networks,tile,target,device=torch_device(req),fourier=fourier)
            self.ctx.set_backend_name(f"torch·{trainer.device}")
            return trainer
        except Exception as e:
            requested=getattr(self.ctx,'backend_requested','auto').lower()
            if requested in {'cupy','cuda','gpu','hybrid','cpu+gpu','cpu_gpu'}:
                raise RuntimeError(f"GPU requested for neural_wall but PyTorch CUDA could not start: {e}") from e
            warnings.warn(f"PyTorch unavailable; using deterministic reconstruction surrogate: {e}")
            self.ctx.set_backend_name("numpy-surrogate")
            return SurrogateWall(networks,tile,target)
    @staticmethod
    def shade(a):
        a=np.asarray(a)
        if a.ndim==3 and a.shape[-1]==3:
            return Image.fromarray((np.clip(a,0,1)*255).astype(np.uint8),'RGB')
        rgb=np.stack([.15+.7*a,.08+.85*a**1.3,.28+.7*np.sqrt(a)],axis=-1)
        return Image.fromarray((np.clip(rgb,0,1)*255).astype(np.uint8))
    def wall_image(self,outs,losses,widths,net_bytes,picture_bytes,best):
        """Every network's reconstruction, labelled with its size and quality.

        A red label means that network is bigger than the picture it redraws,
        so it is not compressing anything at all.
        """
        networks=len(outs); cols,rows=grid_shape(networks); W,H=1280,720; gap=6
        tile_w=(W-gap*(cols+1))//cols; tile_h=(H-gap*(rows+1))//rows
        im=Image.new("RGB",(W,H),(3,6,15)); d=ImageDraw.Draw(im,"RGBA")
        size=max(9,min(15,tile_h//9))
        for i,a in enumerate(outs):
            r,c=divmod(i,cols); x=gap+c*(tile_w+gap); y=gap+r*(tile_h+gap)
            side=min(tile_w,tile_h)
            thumb=self.shade(a).resize((side,side),Image.Resampling.NEAREST)
            im.paste(thumb,(x+(tile_w-side)//2,y+(tile_h-side)//2))
            ratio=picture_bytes/net_bytes[i]
            label=f"w{int(widths[i])} · {ratio:.1f}× · {psnr(losses[i]):.0f} dB" if ratio>=1 else f"w{int(widths[i])} · bigger than picture"
            d.rectangle((x,y+tile_h-size-8,x+tile_w,y+tile_h),fill=(2,5,15,210))
            d.text((x+5,y+tile_h-size-5),label,font=font(size,True),fill=(170,235,205) if ratio>=1 else (255,130,120))
            if i==best:
                d.rectangle((x-3,y-3,x+tile_w+2,y+tile_h+2),outline=(120,255,190,255),width=3)
        return im
    def compose_frame(self,target_u8,recon_u8,side,net_size,picture_size,db):
        """The picture that goes in beside the one the network draws back out."""
        W,H=1280,720; panel=560; top=112
        im=Image.new("RGB",(W,H),(3,6,15)); d=ImageDraw.Draw(im,"RGBA")
        lx=(W//2-panel)//2+20; rx=W//2+(W//2-panel)//2-20
        for x,img in ((lx,target_u8),(rx,recon_u8)):
            im.paste(Image.fromarray(img).resize((panel,panel),Image.Resampling.NEAREST),(x,top))
            d.rectangle((x-1,top-1,x+panel,top+panel),outline=(60,90,120,255),width=2)
        ratio=picture_size/net_size
        d.text((lx,top-72),"IN",font=font(30,True),fill=(235,242,250))
        d.text((lx,top-34),f"{side} x {side} px · {format_bytes(picture_size)}",font=font(20),fill=(160,184,205))
        d.text((rx,top-72),"OUT",font=font(30,True),fill=(235,242,250))
        d.text((rx,top-34),(f"network {format_bytes(net_size)} · {ratio:.1f}x smaller · {db:.1f} dB" if ratio>=1
                            else f"network {format_bytes(net_size)} · bigger than the picture"),
               font=font(20),fill=(170,235,205) if ratio>=1 else (255,130,120))
        d.text((W//2-18,top+panel//2-26),"→",font=font(46,True),fill=(120,150,180))
        return im
    def draw_network_view(self,d,state,progress,x0,y0,x1,y1):
        """Draw the current winning MLP using its actual learned connection weights."""
        d.rounded_rectangle((x0,y0,x1,y1),radius=18,fill=(5,13,29,238),outline=(55,110,154,190),width=2)
        d.text((x0+18,y0+16),"LIVE NETWORK",font=font(15,True),fill=(121,231,255))
        caption=("x,y enter through Fourier waves · shown as local sensitivity" if state.get('fourier')
                 else "connection colour and brightness = learned weight")
        d.text((x0+18,y0+39),caption,font=font(11),fill=(145,169,203))
        width=state['width']; shown=min(12,width)
        indices=np.unique(np.round(np.linspace(0,width-1,shown)).astype(int))
        shown=len(indices)
        xs=[x0+48,x0+152,x0+270,x1-48]
        top,bottom=y0+96,y1-44
        def positions(count):
            return [top+(bottom-top)*(i+.5)/count for i in range(count)]
        layers=[positions(2),positions(shown),positions(shown),positions(1)]
        weights=(state['w1'],state['w2'],state['w3'])
        if weights[0] is None:
            # Explicit fallback view: animated, but not labelled as learned weights.
            a=np.arange(2*shown,dtype=float).reshape(2,shown)
            weights=(np.sin(a*.73+progress*4),np.cos(np.add.outer(np.arange(shown),np.arange(shown))*.31+progress*3),np.sin(np.arange(shown)[:,None]*.61-progress*2))
        else:
            final=weights[2][:width,:][indices,:]
            # The RGB output has three weights per hidden unit; compress only
            # this diagram to one signed connection strength per unit.
            final=np.mean(final,axis=1,keepdims=True)
            weights=(weights[0][:,:width][:,indices],weights[1][np.ix_(indices,indices)],final)
        def edges(left,right,weight):
            scale=max(.05,float(np.percentile(np.abs(weight),90)))
            for a,ya in enumerate(left):
                for b,yb in enumerate(right):
                    value=float(weight[a,b]); strength=min(1,abs(value)/scale)
                    alpha=int((28+180*strength)*(.35+.65*progress))
                    color=(83,232,255,alpha) if value >= 0 else (247,98,196,alpha)
                    d.line((xs[layer],ya,xs[layer+1],yb),fill=color,width=1+int(strength*2))
        for layer in range(3):
            edges(layers[layer],layers[layer+1],weights[layer])
        labels=['x','y']
        for layer,nodes in enumerate(layers):
            for i,y in enumerate(nodes):
                if layer in (1,2):
                    incoming=weights[layer-1][:,i] if layer==1 else weights[layer-1][:,i]
                    strength=min(1,float(np.mean(np.abs(incoming)))/(np.mean(np.abs(weights[layer-1]))+1e-6))
                else: strength=.75 if layer==0 else 1
                pulse=.5+.5*math.sin(progress*15+i*1.7+layer)
                radius=7+int(2*pulse*strength)
                fill=(41+int(55*strength),116+int(105*strength),185+int(55*strength),255)
                d.ellipse((xs[layer]-radius,y-radius,xs[layer]+radius,y+radius),fill=fill,outline=(190,246,255,240),width=1)
                if layer==0: d.text((xs[layer]-4,y-5),labels[i],font=font(11,True),fill='white')
                if layer==3: d.text((xs[layer]-4,y-5),'I',font=font(11,True),fill='white')
        d.text((x0+17,y1-28),f"{width} neurons per hidden layer · {shown} shown",font=font(11,True),fill=(197,218,240))
        d.line((x1-126,y1-21,x1-108,y1-21),fill=(83,232,255,210),width=2); d.text((x1-103,y1-27),'+',font=font(11,True),fill=(160,190,218))
        d.line((x1-70,y1-21,x1-52,y1-21),fill=(247,98,196,210),width=2); d.text((x1-47,y1-27),'−',font=font(11,True),fill=(160,190,218))
    def save_network_overlay(self,state,progress,frame):
        im=Image.new('RGB',(520,360),(3,7,17)); d=ImageDraw.Draw(im,'RGBA')
        self.draw_network_view(d,state,progress,10,8,510,350)
        self.ctx.save_frame(im,self.ctx.run_dir/'overlays'/'network'/f'frame_{frame:04d}.jpg')
    def budget(self):
        """Total optimiser steps for the run, independent of frame count.

        A coordinate network needs O(10^3) Adam steps before the target is
        recognisable. Deriving the budget from `frames * train_steps_per_frame`
        gave 80 steps for a 40-frame run, so every network was still a flat
        blur and the wall had nothing to compare.
        """
        s=self.settings
        if 'total_steps' in s: return max(1,int(s['total_steps']))
        return max(1,int(s.get('train_steps_per_frame',2))*self.ctx.frames)
    def run(self):
        networks=int(self.settings['networks']); side=int(self.settings['tile'])
        kind=int(self.ctx.params.get('target',0)); difficulty=float(self.ctx.params.get('difficulty',1.0))
        fourier=bool(int(self.ctx.params.get('fourier',1)))
        target_path=self.ctx.params.get('_target_path')
        target=self.drawn_target(side,target_path) if target_path else self.target(side,kind,difficulty)
        trainer=self.make_trainer(networks,side,target)
        total=self.budget(); frames=self.ctx.frames
        lrs,widths,_,_=hyperparameters(networks)
        picture=image_bytes(side)
        sizes=[network_bytes(w,fourier) for w in widths]
        rd=self.ctx.run_dir
        (rd/'recon').mkdir(parents=True,exist_ok=True); (rd/'modes'/'wall').mkdir(parents=True,exist_ok=True)
        (rd/'overlays'/'network').mkdir(parents=True,exist_ok=True)
        target_u8=(np.clip(target,0,1)*255).round().astype(np.uint8)
        Image.fromarray(target_u8).save(rd/'target_reduced.png')
        # A larger copy of what went in, before it was squeezed to `side`.
        if target_path:
            with Image.open(target_path) as source:
                ImageOps.fit(source.convert('RGB'),(256,256),Image.Resampling.LANCZOS).save(rd/'target_source.png')
        else:
            Image.fromarray((np.clip(self.target(256,kind,difficulty),0,1)*255).round().astype(np.uint8)).save(rd/'target_source.png')
        # Both views are recorded for every frame; which one is on screen is the
        # presenter's choice, never a switch baked into the run.
        self.ctx.write_meta({
            "view_modes":[{"id":"frames","label":"In → out","folder":"frames"},
                          {"id":"wall","label":"All networks","folder":"modes/wall"}],
            "default_view_mode":"frames",
            "compression":{"side":side,"picture_bytes":picture,"weights":"float32","fourier":fourier,
                           "input":"target_reduced.png","source":"target_source.png","recon":"recon",
                           "networks":[{"width":int(w),"learning_rate":float(lr),"numbers":parameter_count(int(w),fourier),
                                        "bytes":int(b)} for w,lr,b in zip(widths,lrs,sizes)]}})
        with self.ctx.stage("simulation"):
            outs,losses,device=trainer(1)
        done=0; history=[]; jpegs={}
        for i in range(frames):
            step_target=int(round(total*(i+1)/frames))
            with self.ctx.stage("simulation"):
                outs,losses,device=trainer(max(1,step_target-done))
            done=step_target
            best,compresses=compression_champion(losses,widths,side,fourier)
            mse=float(losses[best]); db=psnr(mse); ratio=picture/sizes[best]
            if sizes[best] not in jpegs: jpegs[sizes[best]]=jpeg_at_size(target_u8,sizes[best])
            jpeg=jpegs[sizes[best]]
            recon_u8=(np.clip(outs[best],0,1)*255).round().astype(np.uint8)
            Image.fromarray(recon_u8).save(rd/'recon'/f'frame_{i:04d}.png')
            self.ctx.save_frame(self.compose_frame(target_u8,recon_u8,side,sizes[best],picture,db),self.ctx.frame_path(i))
            self.ctx.save_frame(self.wall_image(outs,losses,widths,sizes,picture,best),rd/'modes'/'wall'/f'frame_{i:04d}.jpg')
            self.save_network_overlay(trainer.visual_state(best),done/max(1,total),i)
            history.append({"frame":i,"step":done,"best":best,"psnr":round(db,2)})
            self.ctx.write_status(i,f"quality {db:.1f} dB",{
                "picture in":f"{side} × {side} px · {format_bytes(picture)}",
                "network size":f"{parameter_count(int(widths[best]),fourier):,} numbers · {format_bytes(sizes[best])}",
                "compression":f"{ratio:.1f}× smaller" if ratio>=1 else f"{1/ratio:.1f}× bigger",
                "quality":f"{db:.1f} dB · {quality_words(db)}",
                "same-size JPEG":f"{jpeg[2]:.1f} dB" if jpeg else "cannot get this small",
                "training step":f"{done:,} / {total:,}",
                "networks":f"best of {networks:,}"+("" if compresses else " · none smaller than the picture"),
                "best loss":f"{mse:.5f}","winning width":f"{int(widths[best])}",
                "learning rate":f"{lrs[best]:.2e}","device":device})
        self.ctx.write_meta({"compression_history":history})
        best,_=compression_champion(losses,widths,side,fourier)
        rp=rd/'reveal.jpg'; self.ctx.save_frame(self.wall_image(outs,losses,widths,sizes,picture,best),rp); self.ctx.finish(rp)
