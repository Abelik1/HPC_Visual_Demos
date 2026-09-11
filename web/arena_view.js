// Smooth replay of the AI game demos' real simulated trajectories.
//
// The solver writes interactive/arena.json (track geometry, once) and one
// interactive/gen_NNNN.json per generation holding the recorded positions of
// the cars shown.  The JPEG frames remain the canonical, replayable output;
// this canvas only animates the same recorded states at display frame rate.
class ArenaView {
  constructor(canvas){
    this.canvas=canvas;this.ctx=canvas.getContext('2d');this.arena=null;this.gen=null;this.cache=new Map();
    this.dpr=1;this.follow=false;this.trails=true;this.t0=performance.now();this.running=false;this.loadSerial=0;
    // A generation's drive replays in ``replaySeconds`` / ``rate`` wall seconds.
    this.replaySeconds=9;this.rate=1;this.cycle=0;this.onloop=null;this.lit=false;
  }
  async fetchJson(url){if(this.cache.has(url))return this.cache.get(url);const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw new Error(`arena data returned ${r.status}`);const data=await r.json();this.cache.set(url,data);if(this.cache.size>90)this.cache.delete(this.cache.keys().next().value);return data;}
  async loadArena(url){const data=await this.fetchJson(url);if(data.kind!=='racers'&&data.kind!=='batmoth')throw new Error('arena description is invalid');this.arena=data;this.resize();}
  async showGeneration(url,restart=true){
    const serial=++this.loadSerial,data=await this.fetchJson(url);if(serial!==this.loadSerial)return;
    if(!Array.isArray(data.cars)||!data.cars.length)throw new Error('generation has no recorded cars');
    const changed=this.gen!==data;this.gen=data;if(changed&&restart){this.t0=performance.now();this.cycle=0;}this.start();
  }
  start(){
    if(this.running)return;this.running=true;const loop=()=>{if(!this.running)return;this.draw();this.raf=requestAnimationFrame(loop);};this.raf=requestAnimationFrame(loop);
    // Generation advancing must not depend on painting: a hidden or
    // throttled page gets no animation frames but should still move on.
    this.tick=setInterval(()=>this.cursor(),250);
  }
  stop(){this.running=false;if(this.raf)cancelAnimationFrame(this.raf);clearInterval(this.tick);}
  resize(){const rect=this.canvas.getBoundingClientRect();if(!rect.width||!rect.height)return;this.dpr=Math.min(window.devicePixelRatio||1,2);const w=Math.round(rect.width*this.dpr),h=Math.round(rect.height*this.dpr);if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;}this.draw();}

  samples(){return this.gen?Math.max(...this.gen.cars.map(c=>c.x.length)):0;}
  // Current sample position (fractional) with a short hold at the end.
  cursor(){
    const n=this.samples();if(n<2)return 0;
    const dur=this.replaySeconds/Math.max(.25,this.rate),hold=1.4,total=(performance.now()-this.t0)/1000;
    const cycle=Math.floor(total/(dur+hold));
    if(cycle>this.cycle){this.cycle=cycle;if(this.onloop)setTimeout(()=>this.onloop(),0);}
    return Math.min(n-1,(total%(dur+hold))/dur*(n-1));
  }
  view(){
    const w=this.canvas.width,h=this.canvas.height,[W,H]=this.arena?.world||[16,9];let scale=Math.min(w/W,h/H),ox=(w-W*scale)/2,oy=(h-H*scale)/2;
    if(this.follow&&this.gen){const c=this.gen.cars[0],i=Math.min(c.x.length-1,Math.floor(this.cursor()));const z=2.4;scale*=z;ox=w/2-c.x[i]/100*scale;oy=h/2-c.y[i]/100*scale;}
    // ``scale`` is canvas pixels per world unit; P() takes centi-units.
    return {scale,ox,oy,P:(x,y)=>[ox+x/100*scale,oy+y/100*scale]};
  }
  poly(points,v){const ctx=this.ctx;points.forEach(([x,y],i)=>{const [px,py]=v.P(x,y);i?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.closePath();}
  drawTrack(v){
    const ctx=this.ctx,a=this.arena,d=this.dpr;if(!a)return;
    ctx.beginPath();this.poly(a.left,v);this.poly(a.right,v);ctx.fillStyle='#161b26';ctx.fill('evenodd');
    ctx.shadowColor='rgba(96,220,255,.8)';ctx.shadowBlur=12*d;ctx.strokeStyle='#60dcff';ctx.lineWidth=2.2*d;
    for(const edge of [a.left,a.right]){ctx.beginPath();this.poly(edge,v);ctx.stroke();}
    ctx.shadowBlur=0;
    const [l,r]=a.start,[x1,y1]=v.P(...l),[x2,y2]=v.P(...r);ctx.setLineDash([6*d,6*d]);ctx.strokeStyle='#f5f8ff';ctx.lineWidth=5*d;ctx.beginPath();ctx.moveTo(x1,y1);ctx.lineTo(x2,y2);ctx.stroke();ctx.setLineDash([]);
  }
  drawCar(x,y,deg,fill,size){const ctx=this.ctx,r=deg*Math.PI/180;ctx.save();ctx.translate(x,y);ctx.rotate(r);ctx.beginPath();ctx.moveTo(size,0);ctx.lineTo(-size*.65,size*.45);ctx.lineTo(-size*.4,0);ctx.lineTo(-size*.65,-size*.45);ctx.closePath();ctx.fillStyle=fill;ctx.fill();ctx.strokeStyle='rgba(255,255,255,.85)';ctx.lineWidth=1*this.dpr;ctx.stroke();ctx.restore();}
  drawAgent(car,i,v,cursor,style){
    const ctx=this.ctx,d=this.dpr,n=car.x.length,k=Math.min(n-1,Math.floor(cursor)),f=Math.min(1,cursor-k);
    // ``crash`` is the recorded sample at which the car hit the wall, or -1.
    const crashed=car.crash>=0&&k>=car.crash;
    if(this.trails){const from=Math.max(0,k-(i===0?90:45));ctx.beginPath();for(let j=from;j<=k;j++){const [px,py]=v.P(car.x[j],car.y[j]);j===from?ctx.moveTo(px,py):ctx.lineTo(px,py);}ctx.strokeStyle=style.trail;ctx.lineWidth=style.width*d;ctx.stroke();}
    const k2=Math.min(n-1,k+1),x=car.x[k]+(car.x[k2]-car.x[k])*f,y=car.y[k]+(car.y[k2]-car.y[k])*f;const [px,py]=v.P(x,y);
    if(crashed){ctx.strokeStyle='rgba(255,92,92,.8)';ctx.lineWidth=2*d;const s=5*d;ctx.beginPath();ctx.moveTo(px-s,py-s);ctx.lineTo(px+s,py+s);ctx.moveTo(px-s,py+s);ctx.lineTo(px+s,py-s);ctx.stroke();return;}
    this.drawCar(px,py,car.h[k],style.fill,style.size*d*(this.follow?1.8:1));
    if(style.label){ctx.font=`700 ${11*d}px Arial`;ctx.fillStyle=style.fill;ctx.fillText(style.label,px+10*d,py-10*d);}
  }
  draw(){
    if(!this.canvas.width)return;const ctx=this.ctx,w=this.canvas.width,h=this.canvas.height,d=this.dpr;
    ctx.fillStyle='#030813';ctx.fillRect(0,0,w,h);
    ctx.strokeStyle='rgba(30,70,110,.25)';ctx.lineWidth=1;for(let x=0;x<w;x+=40*d){ctx.beginPath();ctx.moveTo(x,0);ctx.lineTo(x,h);ctx.stroke();}for(let y=0;y<h;y+=40*d){ctx.beginPath();ctx.moveTo(0,y);ctx.lineTo(w,y);ctx.stroke();}
    if(!this.arena||!this.gen)return;const v=this.view(),cursor=this.cursor(),cars=this.gen.cars;
    if(this.arena.kind==='batmoth'){this.drawBatMoth(v,cursor);return;}
    this.drawTrack(v);
    for(let i=cars.length-1;i>=1;i--){const top=i<cars.length/4;this.drawAgent(cars[i],i,v,cursor,{trail:top?'rgba(120,210,255,.35)':'rgba(140,160,200,.18)',width:1.4,fill:top?'rgba(120,210,255,.95)':'rgba(150,170,210,.75)',size:6});}
    (this.gen.ghosts||[]).forEach(g=>this.drawAgent({...g,crash:-1},1,v,cursor,{trail:'rgba(190,150,255,.55)',width:2.4,fill:'#be96ff',size:8,label:`ghost ${g.name}`}));
    this.drawAgent(cars[0],0,v,cursor,{trail:'rgba(255,214,120,.9)',width:3,fill:'#ffc450',size:9,label:'CHAMPION'});
    // Race time is the simulated time of the recorded sample, not wall time.
    const seconds=cursor*(this.gen.sample_dt||.1);
    // Top centre: the left edge belongs to the optional overlay cards.
    const bx=w/2-150*d;
    ctx.fillStyle='rgba(3,7,18,.82)';ctx.strokeStyle='rgba(255,196,80,.35)';ctx.lineWidth=d;ctx.beginPath();ctx.roundRect(bx,16*d,300*d,58*d,12*d);ctx.fill();ctx.stroke();
    ctx.textAlign='center';
    ctx.fillStyle='#f3f8ff';ctx.font=`800 ${17*d}px Arial`;ctx.fillText(`GENERATION ${this.gen.generation}`,w/2,42*d);
    ctx.fillStyle='#95b2d4';ctx.font=`${12*d}px Arial`;ctx.fillText(`race clock ${seconds.toFixed(1)} s · best ${cars.length} cars shown`,w/2,62*d);
    const lap=cars[0].lap_s;if(lap&&seconds>=lap){ctx.fillStyle='#ffc450';ctx.font=`800 ${15*d}px Arial`;ctx.fillText(`CHAMPION LAP ${lap.toFixed(1)} s`,w/2,96*d);}
    ctx.textAlign='left';
  }

  // ---- Bat vs Moth ------------------------------------------------------
  // "Senses" shows only what the champion bat's calls revealed: expanding
  // call rings, rock edges they touched, moth echoes and jamming phantoms.
  // "Lit" shows the whole cave.  Both replay the same recorded hunt.
  rockEdges(){if(this._edges&&this._edgesFor===this.arena)return this._edges;this._edgesFor=this.arena;this._edges=[];const [W,H]=this.arena.world;for(const [cx,cy,r] of this.arena.rocks){const n=Math.max(12,Math.round(r*40));for(let i=0;i<n;i++){const t=i/n*Math.PI*2,x=cx+r*Math.cos(t),y=cy+r*Math.sin(t);if(x>0&&x<W&&y>0&&y<H)this._edges.push([x,y]);}}return this._edges;}
  drawBatMoth(v,cursor){
    const ctx=this.ctx,d=this.dpr,a=this.arena,g=this.gen,w=this.canvas.width,bat=g.cars[0],k=Math.min(bat.x.length-1,Math.floor(cursor)),dt=g.sample_dt||.066;
    const P=(x,y)=>[v.ox+x*v.scale,v.oy+y*v.scale],range=a.echo_range||6,ear=(a.ear_offset_deg||40)*Math.PI/180;
    if(this.lit){ctx.fillStyle='#0a1020';ctx.fillRect(v.ox,v.oy,a.world[0]*v.scale,a.world[1]*v.scale);ctx.fillStyle='#2e3444';ctx.strokeStyle='#7c869c';ctx.lineWidth=1.5*d;for(const [cx,cy,r] of a.rocks){const [px,py]=P(cx,cy);ctx.beginPath();ctx.arc(px,py,r*v.scale,0,Math.PI*2);ctx.fill();ctx.stroke();}}
    else{ctx.fillStyle='#010308';ctx.fillRect(v.ox,v.oy,a.world[0]*v.scale,a.world[1]*v.scale);}
    const edges=this.rockEdges(),span=1.6/dt;
    ctx.globalCompositeOperation='lighter';
    for(const [s,fakes] of g.chirps){
      if(s>k||s<k-span)continue;const age=(cursor-s)*dt,fade=Math.max(0,1-age/1.6),bx=bat.x[s]/100,by=bat.y[s]/100,bh=bat.h[s]*Math.PI/180,radius=Math.min(range,age*5.5);
      if(radius<range){const [px,py]=P(bx,by);ctx.strokeStyle=`rgba(255,190,80,${.45*fade})`;ctx.lineWidth=2*d;ctx.beginPath();ctx.arc(px,py,radius*v.scale,0,Math.PI*2);ctx.stroke();
        if(!this.lit){ctx.fillStyle=`rgba(170,200,255,${.8*fade})`;for(const [ex,ey] of edges){if(Math.abs(Math.hypot(ex-bx,ey-by)-radius)<.22){const [qx,qy]=P(ex,ey);ctx.fillRect(qx-1.5*d,qy-1.5*d,3*d,3*d);}}}}
      g.moths.forEach(m=>{if(m.caught>=0&&m.caught<=s)return;const mx=m.x[Math.min(s,m.x.length-1)]/100,my=m.y[Math.min(s,m.y.length-1)]/100,dist=Math.hypot(mx-bx,my-by);if(dist>=range)return;const rel=Math.atan2(my-by,mx-bx)-bh,gain=Math.max(.5*(1+Math.cos(rel+ear)),.5*(1+Math.cos(rel-ear))),loud=gain*(1-dist/range);if(loud<.02)return;const [qx,qy]=P(mx,my),r=(4+10*loud)*d;const grad=ctx.createRadialGradient(qx,qy,0,qx,qy,r*2);grad.addColorStop(0,`rgba(120,225,255,${fade*(.3+.6*loud)})`);grad.addColorStop(1,'rgba(120,225,255,0)');ctx.fillStyle=grad;ctx.beginPath();ctx.arc(qx,qy,r*2,0,Math.PI*2);ctx.fill();});
      fakes.forEach((delay,side)=>{if(delay<0)return;const ang=bh+(side?ear:-ear)+(delay-.5)*.8,[qx,qy]=P(bx+Math.cos(ang)*delay*range,by+Math.sin(ang)*delay*range);ctx.strokeStyle=`rgba(255,90,200,${.9*fade})`;ctx.lineWidth=3*d;ctx.beginPath();ctx.arc(qx,qy,9*d,0,Math.PI*2);ctx.stroke();});
    }
    ctx.globalCompositeOperation='source-over';
    g.moths.forEach((m,j)=>{const caught=m.caught>=0&&k>=m.caught,s=caught?m.caught:Math.min(k,m.x.length-1),[qx,qy]=P(m.x[s]/100,m.y[s]/100);
      if(caught){ctx.strokeStyle='rgba(255,255,255,.9)';ctx.lineWidth=2*d;const r=6*d;ctx.beginPath();ctx.moveTo(qx-r,qy-r);ctx.lineTo(qx+r,qy+r);ctx.moveTo(qx-r,qy+r);ctx.lineTo(qx+r,qy-r);ctx.stroke();return;}
      if(!this.lit)return;
      if(this.trails){ctx.strokeStyle='rgba(110,240,200,.35)';ctx.lineWidth=1.5*d;ctx.beginPath();for(let i=Math.max(0,k-40);i<=k;i++){const [px,py]=P(m.x[i]/100,m.y[i]/100);i===Math.max(0,k-40)?ctx.moveTo(px,py):ctx.lineTo(px,py);}ctx.stroke();}
      if((g.jam[k]>>j)&1){ctx.strokeStyle='rgba(255,90,200,.85)';ctx.lineWidth=2*d;ctx.beginPath();ctx.arc(qx,qy,15*d,0,Math.PI*2);ctx.stroke();}
      ctx.fillStyle='#6ef0c8';ctx.beginPath();ctx.arc(qx,qy,6*d,0,Math.PI*2);ctx.fill();});
    if(this.trails){ctx.strokeStyle='rgba(255,190,80,.6)';ctx.lineWidth=2*d;ctx.beginPath();for(let i=Math.max(0,k-60);i<=k;i++){const [px,py]=P(bat.x[i]/100,bat.y[i]/100);i===Math.max(0,k-60)?ctx.moveTo(px,py):ctx.lineTo(px,py);}ctx.stroke();}
    const [bx,by]=P(bat.x[k]/100,bat.y[k]/100);this.drawCar(bx,by,bat.h[k],'#ffbe50',11*d);
    const caught=g.moths.filter(m=>m.caught>=0&&m.caught<=k).length,bx0=w/2-170*d;
    ctx.fillStyle='rgba(3,7,18,.82)';ctx.strokeStyle='rgba(255,90,200,.35)';ctx.lineWidth=d;ctx.beginPath();ctx.roundRect(bx0,16*d,340*d,58*d,12*d);ctx.fill();ctx.stroke();
    ctx.textAlign='center';ctx.fillStyle='#f3f8ff';ctx.font=`800 ${17*d}px Arial`;ctx.fillText(`GENERATION ${g.generation} · ${this.lit?'LIT CAVE':'BAT’S SENSES'}`,w/2,42*d);
    ctx.fillStyle='#95b2d4';ctx.font=`${12*d}px Arial`;ctx.fillText(`hunt clock ${(cursor*dt).toFixed(1)} s · ${caught} of ${g.moths.length} moths caught`,w/2,62*d);ctx.textAlign='left';
  }
}
window.ArenaView=ArenaView;
