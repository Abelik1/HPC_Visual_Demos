class FusionView {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.data = null;
    this.showPlasma = true;
    this.showMagnetic = false;
    this.showEscapes = true;
    this.particleFilter = 'all';
    this.yaw = -0.18;
    this.pitch = 0.92;
    this.zoom = 1;
    this.drag = null;
    this.dpr = 1;
    canvas.addEventListener('pointerdown', e => this.onDown(e));
    canvas.addEventListener('pointermove', e => this.onMove(e));
    canvas.addEventListener('pointerup', e => this.onUp(e));
    canvas.addEventListener('pointercancel', e => this.onUp(e));
    canvas.addEventListener('dblclick', () => this.reset());
    canvas.addEventListener('wheel', e => {
      e.preventDefault(); e.stopPropagation();
      this.zoom = Math.max(.62, Math.min(1.8, this.zoom * (e.deltaY < 0 ? 1.09 : .92)));
      this.draw();
    }, {passive: false});
  }

  async load(url, preserveView=false) {
    const request=(this.loadRequest||0)+1;this.loadRequest=request;
    const response = await fetch(url, {cache: 'no-store'});
    if (!response.ok) throw new Error(`interactive fusion state returned ${response.status}`);
    const data = await response.json();
    if (data.kind !== 'fusion-torus' || !Array.isArray(data.shape) || !Array.isArray(data.texture)) {
      throw new Error('interactive fusion state is invalid');
    }
    if (data.mode === 'guardian' && !Array.isArray(data.particle_trails)) {
      throw new Error('interactive fusion state is missing its confined markers');
    }
    if (data.texture.length !== data.shape[0] * data.shape[1]) {
      throw new Error('interactive fusion texture has the wrong size');
    }
    if(request!==this.loadRequest)return;
    this.data = data;
    if(preserveView)this.resize();else this.reset();
  }

  reset() {
    this.yaw = -0.18;
    this.pitch = 0.92;
    this.zoom = 1;
    this.resize();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    this.dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.round(rect.width * this.dpr), height = Math.round(rect.height * this.dpr);
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width; this.canvas.height = height;
    }
    this.draw();
  }

  setLayer(layer, enabled) {
    if (layer === 'magnetic') this.showMagnetic = Boolean(enabled);
    if (layer === 'plasma') this.showPlasma = Boolean(enabled);
    if (layer === 'escapes') this.showEscapes = Boolean(enabled);
    this.draw();
  }

  // The run decides which picture this canvas is showing.
  isGuardian() { return Boolean(this.data && this.data.mode === 'guardian'); }

  setParticleFilter(filter) {
    this.particleFilter = filter === 'all' ? 'all' : String(Math.max(0, Math.min(3, Number(filter) || 0)));
    this.draw();
  }

  onDown(event) {
    event.preventDefault(); event.stopPropagation();
    this.drag = {id: event.pointerId, x: event.clientX, y: event.clientY, yaw: this.yaw, pitch: this.pitch};
    this.canvas.setPointerCapture(event.pointerId);
    this.canvas.classList.add('isPanning');
  }

  onMove(event) {
    if (!this.drag) return;
    event.preventDefault(); event.stopPropagation();
    this.yaw = this.drag.yaw + (event.clientX - this.drag.x) * .009;
    this.pitch = Math.max(-1.42, Math.min(1.42, this.drag.pitch + (event.clientY - this.drag.y) * .008));
    this.draw();
  }

  onUp(event) {
    if (!this.drag) return;
    const id = this.drag.id; this.drag = null;
    this.canvas.classList.remove('isPanning');
    try { this.canvas.releasePointerCapture(id); } catch (_) {}
  }

  // `a` and `b` place a point inside the poloidal cross-section in units of
  // the wall minor radius: `a` outward along the major radius, `b` along the
  // machine axis. The surface case is simply (cos v, sin v).
  point(uNorm, a, b, minor=.40) {
    const u = uNorm * Math.PI * 2 + this.yaw;
    const radius = 1 + minor * a;
    const x = radius * Math.cos(u);
    const y = radius * Math.sin(u);
    const z = minor * b;
    const cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
    const yy = y * cp - z * sp;
    const zz = y * sp + z * cp;
    const w = this.canvas.width, h = this.canvas.height;
    const scale = Math.min(w / 3.15, h / 2.25) * this.zoom;
    return {x: w * .5 + x * scale, y: h * .52 - yy * scale, z: zz};
  }

  project(uNorm, vNorm, minor=.40) {
    const v = vNorm * Math.PI * 2;
    return this.point(uNorm, Math.cos(v), Math.sin(v), minor);
  }

  // Position of a confined marker given the magnetic axis the policy is holding.
  marker(u, theta, r, axis) {
    const angle = theta * Math.PI * 2;
    return this.point(u, axis[0] + r * Math.cos(angle), axis[1] + r * Math.sin(angle));
  }

  palette(value, light=1) {
    const x = Math.max(0, Math.min(1, value / 255));
    const r = Math.min(1, .08 + 1.3 * Math.pow(x, 1.6));
    const g = Math.min(1, .02 + 1.0 * Math.pow(x, 2.2));
    const b = Math.min(1, .18 + 1.2 * x * (1-x) + .6*x);
    return [Math.round(255*r*light), Math.round(255*g*light), Math.round(255*b*light)];
  }

  // Poloidal field coils: rings encircling the machine just outside the wall,
  // one per member of each bank. Mirrors COIL_BANKS in fusion_plasma.py, so the
  // hardware here is the hardware in the cross-section overlay.
  coilSegments(radius=1.15, samples=104) {
    const banks=[[[80,225,255],[0,180]],[[255,166,89],[90,270]],[[198,126,255],[45,135,225,315]]];
    const commands=this.data.commands||[0,0,0],segments=[];
    banks.forEach(([colour,angles],bank)=>{
      const level=.25+.75*Math.min(1,Math.abs(commands[bank]||0));
      for(const degrees of angles){
        const t=degrees*Math.PI/180,a=radius*Math.cos(t),b=radius*Math.sin(t);
        let previous=this.point(0,a,b);
        for(let k=1;k<samples;k++){
          const current=this.point(k/(samples-1),a,b);
          segments.push({a:previous,b:current,z:(previous.z+current.z)/2,colour,level});
          previous=current;
        }
      }
    });
    return segments;
  }

  drawCoils(segments,half) {
    const ctx=this.ctx;ctx.lineCap='round';ctx.globalCompositeOperation='lighter';
    for(const s of segments){
      if((s.z<0)!==(half==='back'))continue;
      const front=Math.max(.22,Math.min(1,.3+.7*(s.z+.65)/1.3));
      const c=s.colour;
      ctx.strokeStyle=`rgba(${c[0]},${c[1]},${c[2]},${(.09+.36*s.level)*front})`;
      ctx.lineWidth=(.8+1.6*s.level)*this.dpr;
      ctx.beginPath();ctx.moveTo(s.a.x,s.a.y);ctx.lineTo(s.b.x,s.b.y);ctx.stroke();
    }
    ctx.globalCompositeOperation='source-over';
  }

  // Mirrors PARTICLE_RAMP in leonardo_demos/demos/fusion_plasma.py so the
  // rendered frames and this canvas tell the same story about wall clearance.
  markerColour(clearance) {
    const stops=[[0,[255,96,64]],[.09,[255,166,72]],[.19,[168,255,206]],[.34,[118,224,255]],[1,[152,216,255]]];
    const x=Math.max(0,Math.min(1,clearance));
    let i=0; while(i<stops.length-2 && x>stops[i+1][0]) i++;
    const t=(x-stops[i][0])/Math.max(1e-6,stops[i+1][0]-stops[i][0]);
    const a=stops[i][1],b=stops[i+1][1];
    return [Math.round(a[0]+(b[0]-a[0])*t),Math.round(a[1]+(b[1]-a[1])*t),Math.round(a[2]+(b[2]-a[2])*t)];
  }

  clearanceOf(index) {
    const stored=this.data.clearance;
    if (Array.isArray(stored) && stored.length > index) return stored[index];
    const p=this.data.particle_trails[index], last=p[p.length-1], axis=this.data.axis||[0,0];
    const angle=last[1]*Math.PI*2;
    const a=axis[0]+last[2]*Math.cos(angle), b=axis[1]+last[2]*Math.sin(angle);
    return Math.max(0,Math.min(1,1-Math.hypot(a,b)));
  }

  drawMarkers() {
    const trails=this.data.particle_trails||[], axis=this.data.axis||[0,0];
    const ctx=this.ctx, segments=[], heads=[];
    for (let index=0; index<trails.length; index++) {
      if (this.particleFilter !== 'all' && index % 4 !== Number(this.particleFilter)) continue;
      const points=trails[index].map(p=>this.marker(p[0],p[1],p[2],axis));
      const colour=this.markerColour(this.clearanceOf(index));
      for (let k=1;k<points.length;k++) {
        const a=points[k-1],b=points[k];
        if (Math.abs(a.x-b.x)>this.canvas.width*.22||Math.abs(a.y-b.y)>this.canvas.height*.22) continue;
        segments.push({a,b,z:(a.z+b.z)/2,age:k/(points.length-1),colour});
      }
      heads.push({p:points[points.length-1],colour});
    }
    segments.sort((a,b)=>a.z-b.z);
    ctx.lineCap='round';ctx.globalCompositeOperation='lighter';
    for(const s of segments){
      const front=Math.max(.28,Math.min(1,.42+.58*(s.z+.65)/1.3));
      ctx.strokeStyle=`rgba(${s.colour[0]},${s.colour[1]},${s.colour[2]},${(.06+.74*s.age*s.age)*front})`;
      ctx.lineWidth=(.9+1.5*front)*this.dpr;
      ctx.beginPath();ctx.moveTo(s.a.x,s.a.y);ctx.lineTo(s.b.x,s.b.y);ctx.stroke();
    }
    heads.sort((a,b)=>a.p.z-b.p.z);
    for(const head of heads){
      const front=Math.max(.34,Math.min(1,.44+.56*(head.p.z+.65)/1.3));
      const c=head.colour;
      ctx.fillStyle=`rgba(${c[0]},${c[1]},${c[2]},${front})`;
      ctx.beginPath();ctx.arc(head.p.x,head.p.y,(1.7+2.1*front)*this.dpr,0,Math.PI*2);ctx.fill();
    }
    ctx.globalCompositeOperation='source-over';
  }

  drawSparks() {
    const sparks=this.data.sparks||[];
    if(!sparks.length)return;
    const ctx=this.ctx,items=[];
    for(const spark of sparks){
      const angle=spark[1]*Math.PI*2;
      items.push({p:this.point(spark[0],Math.cos(angle),Math.sin(angle)),
                  fade:Math.pow(Math.max(0,1-spark[2]),1.5),energy:spark[3]});
    }
    items.sort((a,b)=>a.p.z-b.p.z);
    ctx.globalCompositeOperation='lighter';
    for(const item of items){
      if(item.fade<=.03)continue;
      const front=Math.max(.3,Math.min(1,.4+.6*(item.p.z+.65)/1.3));
      const r=(1.8+6.5*item.energy*item.fade)*this.dpr;
      const halo=ctx.createRadialGradient(item.p.x,item.p.y,0,item.p.x,item.p.y,r*3.2);
      halo.addColorStop(0,`rgba(255,${Math.round(210*item.fade+40)},140,${.85*item.fade*front})`);
      halo.addColorStop(1,'rgba(255,110,40,0)');
      ctx.fillStyle=halo;
      ctx.beginPath();ctx.arc(item.p.x,item.p.y,r*3.2,0,Math.PI*2);ctx.fill();
      ctx.fillStyle=`rgba(255,255,${Math.round(190*item.fade)},${.95*item.fade*front})`;
      ctx.beginPath();ctx.arc(item.p.x,item.p.y,r*.55,0,Math.PI*2);ctx.fill();
    }
    ctx.globalCompositeOperation='source-over';
  }

  drawSurface(alpha, half='all') {
    const [ny, nx] = this.data.shape, texture = this.data.texture;
    const stride = Math.max(1, Math.ceil(ny / 64));
    const points = [];
    for (let row=0; row<ny; row+=stride) {
      for (let col=0; col<nx; col+=stride) {
        const p = this.project(col/nx, row/ny);
        if (half === 'back' && p.z >= 0) continue;
        if (half === 'front' && p.z < 0) continue;
        points.push({...p, value: texture[row*nx+col]});
      }
    }
    points.sort((a,b) => a.z-b.z);
    const ctx=this.ctx, radius=Math.max(1.1*this.dpr, Math.min(this.canvas.width,this.canvas.height)/350);
    ctx.globalCompositeOperation='lighter';
    for (const p of points) {
      const front=Math.max(.35,Math.min(1,.54+.46*(p.z+.65)/1.3));
      const [r,g,b]=this.palette(p.value,front);
      ctx.fillStyle=`rgba(${r},${g},${b},${alpha*front})`;
      ctx.beginPath();ctx.arc(p.x,p.y,radius*(.72+.35*front),0,Math.PI*2);ctx.fill();
    }
    ctx.globalCompositeOperation='source-over';
  }

  drawTrails() {
    const trails=this.data.trails||[], segments=[];
    for (let particle=0; particle<trails.length; particle++) {
      if (this.particleFilter !== 'all' && particle % 4 !== Number(this.particleFilter)) continue;
      const points=trails[particle].map(uv=>this.project(uv[0],uv[1]));
      for (let k=1;k<points.length;k++) {
        const a=points[k-1],b=points[k];
        if (Math.abs(a.x-b.x)>this.canvas.width*.22||Math.abs(a.y-b.y)>this.canvas.height*.22) continue;
        segments.push({a,b,z:(a.z+b.z)/2,age:k/(points.length-1),particle});
      }
    }
    segments.sort((a,b)=>a.z-b.z);
    const colours=[[104,239,255],[255,173,83],[224,126,255],[190,255,228]],ctx=this.ctx;
    ctx.lineCap='round';ctx.globalCompositeOperation='lighter';
    for(const s of segments){
      const c=colours[s.particle%colours.length];
      const front=Math.max(.3,Math.min(1,.46+.54*(s.z+.65)/1.3));
      ctx.strokeStyle=`rgba(${c[0]},${c[1]},${c[2]},${(.14+.72*s.age)*front})`;
      ctx.lineWidth=(1.2+1.8*front)*this.dpr;ctx.beginPath();ctx.moveTo(s.a.x,s.a.y);ctx.lineTo(s.b.x,s.b.y);ctx.stroke();
    }
    ctx.globalCompositeOperation='source-over';
    for(let particle=0;particle<trails.length;particle++){
      if (this.particleFilter !== 'all' && particle % 4 !== Number(this.particleFilter)) continue;
      const uv=trails[particle][trails[particle].length-1],p=this.project(uv[0],uv[1]);
      const c=colours[particle%colours.length],front=Math.max(.35,Math.min(1,.48+.52*(p.z+.65)/1.3));
      ctx.fillStyle=`rgba(248,255,255,${front})`;ctx.strokeStyle=`rgb(${c[0]},${c[1]},${c[2]})`;ctx.lineWidth=1.2*this.dpr;
      ctx.beginPath();ctx.arc(p.x,p.y,(2.1+2.5*front)*this.dpr,0,Math.PI*2);ctx.fill();ctx.stroke();
    }
  }

  drawMagneticField() {
    const count=this.data.field_lines||14,pitch=this.data.field_pitch||.7,segments=[];
    const guardian=this.isGuardian(),axis=guardian?(this.data.axis||[0,0]):[0,0];
    // In guardian mode the surfaces sit where the markers actually live and
    // follow the magnetic axis the policy is holding. They remain explanatory
    // confinement geometry, not field lines traced through a solved equilibrium.
    const radii=guardian?[.34,.56,.78]:[.475,.75,1];
    const commands=this.data.commands||[0,0,0];
    const effort=guardian?Math.min(1,(Math.abs(commands[0])+Math.abs(commands[1])+Math.abs(commands[2]))/3):0;
    const samples=220,turns=guardian?1.8:2.35;
    const shift=r=>{const w=1-.35*Math.min(1,r);return [axis[0]*w,axis[1]*w];};
    for(let line=0;line<count;line++){
      const r=radii[line%radii.length],offset=line/count,[sx,sy]=shift(r);
      const at=t=>{const a=(offset+pitch*t)*Math.PI*2;
                   return this.point(t,sx+r*Math.cos(a),sy+r*Math.sin(a));};
      let previous=at(0);
      for(let s=1;s<samples;s++){
        const current=at(s/(samples-1)*turns);
        segments.push({a:previous,b:current,z:(previous.z+current.z)/2,line});
        previous=current;
      }
    }
    // The magnetic axis is the centre of the nested flux surfaces.
    let previous=this.point(0,axis[0],axis[1]),axisLine=[];
    for(let s=1;s<180;s++){const current=this.point(s/179,axis[0],axis[1]);axisLine.push({a:previous,b:current,z:(previous.z+current.z)/2});previous=current;}
    segments.sort((a,b)=>a.z-b.z);axisLine.sort((a,b)=>a.z-b.z);
    const ctx=this.ctx;ctx.lineCap='round';ctx.globalCompositeOperation='lighter';
    const warm=[255,168,92];
    for(const s of segments){
      const front=Math.max(.28,Math.min(1,.46+.54*(s.z+.65)/1.3));
      const base=s.line%3===0?[88,235,255]:s.line%3===1?[112,155,255]:[179,115,255];
      const hue=base.map((v,i)=>Math.round(v*(1-.55*effort)+warm[i]*.55*effort));
      ctx.strokeStyle=`rgba(${hue[0]},${hue[1]},${hue[2]},${(guardian?.12:.22)+(guardian?.46:.70)*front})`;
      ctx.lineWidth=((guardian?.8:1.1)+1.8*front)*this.dpr;ctx.beginPath();ctx.moveTo(s.a.x,s.a.y);ctx.lineTo(s.b.x,s.b.y);ctx.stroke();
    }
    for(const s of axisLine){ctx.strokeStyle='rgba(255,210,112,.86)';ctx.lineWidth=3*this.dpr;ctx.beginPath();ctx.moveTo(s.a.x,s.a.y);ctx.lineTo(s.b.x,s.b.y);ctx.stroke();}
    ctx.globalCompositeOperation='source-over';
  }

  draw() {
    if(!this.data||!this.canvas.width||!this.canvas.height)return;
    const ctx=this.ctx,w=this.canvas.width,h=this.canvas.height;
    const gradient=ctx.createRadialGradient(w*.48,h*.40,0,w*.5,h*.5,Math.max(w,h)*.65);
    gradient.addColorStop(0,this.showMagnetic?'#07152b':'#0a0b28');gradient.addColorStop(.55,'#030711');gradient.addColorStop(1,'#010309');
    ctx.fillStyle=gradient;ctx.fillRect(0,0,w,h);
    if(this.isGuardian()){
      // The coil rings encircle the machine, so the far half belongs behind the
      // vessel and the near half in front of everything inside it.
      const coils=this.coilSegments();
      this.drawCoils(coils,'back');
      // The far half of the wall next, so the confined markers really do read
      // as being inside the bottle; the near half returns as a glass shell.
      if(this.showPlasma)this.drawSurface(this.showMagnetic?.34:.46,'back');
      if(this.showMagnetic)this.drawMagneticField();
      this.drawMarkers();
      if(this.showEscapes)this.drawSparks();
      if(this.showPlasma)this.drawSurface(.13,'front');
      this.drawCoils(coils,'front');
    } else {
      if(this.showPlasma){this.drawSurface(this.showMagnetic?.28:.42);this.drawTrails();}
      else if(this.showMagnetic){this.drawSurface(.10);}
      if(this.showMagnetic)this.drawMagneticField();
    }
    // Titles, parameter readouts and method notes belong to the HTML view
    // layers. The canvas remains a clean, rotatable scientific viewport.
  }
}

window.FusionView=FusionView;
