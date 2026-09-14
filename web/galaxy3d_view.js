class Galaxy3DView {
  constructor(canvas) {
    this.canvas=canvas;this.ctx=canvas.getContext('2d');this.data=null;
    this.yaw=-.34;this.pitch=.48;this.zoom=1;this.drag=null;this.showHalo=true;this.focus='all';this.dpr=1;this.loadSerial=0;
    canvas.addEventListener('pointerdown',event=>this.onDown(event));
    canvas.addEventListener('pointermove',event=>this.onMove(event));
    canvas.addEventListener('pointerup',event=>this.onUp(event));
    canvas.addEventListener('pointercancel',event=>this.onUp(event));
    canvas.addEventListener('dblclick',()=>this.reset());
    canvas.addEventListener('wheel',event=>{event.preventDefault();event.stopPropagation();this.zoom=Math.max(.3,Math.min(20,this.zoom*(event.deltaY<0?1.1:.91)));this.draw();},{passive:false});
  }

  async load(url) {
    const serial=++this.loadSerial;
    const response=await fetch(url,{cache:'no-store'});
    if(!response.ok)throw new Error(`3D particle frame returned ${response.status}`);
    const data=await response.json();
    if(serial!==this.loadSerial)return;
    const n=Array.isArray(data.positions)?data.positions.length:0;
    if(data.kind!=='nbody-galaxy-3d'||!n||data.origin?.length!==n||data.component?.length!==n)throw new Error('3D particle frame is invalid');
    this.data=data;this.resize();
  }

  reset(){this.yaw=-.34;this.pitch=.48;this.zoom=this.focus==='all'?1:14;this.resize();}
  setFocus(value){this.focus=['all','mw','m31'].includes(value)?value:'all';this.zoom=this.focus==='all'?1:14;if(this.focus==='mw'){this.yaw=0;this.pitch=0;}this.draw();}
  setHalo(value){this.showHalo=Boolean(value);this.draw();}
  resize(){const rect=this.canvas.getBoundingClientRect();if(!rect.width||!rect.height)return;this.dpr=Math.min(window.devicePixelRatio||1,2);const w=Math.round(rect.width*this.dpr),h=Math.round(rect.height*this.dpr);if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;}this.draw();}
  onDown(event){event.preventDefault();event.stopPropagation();this.drag={id:event.pointerId,x:event.clientX,y:event.clientY,yaw:this.yaw,pitch:this.pitch};this.canvas.setPointerCapture(event.pointerId);this.canvas.classList.add('isPanning');}
  onMove(event){if(!this.drag)return;event.preventDefault();event.stopPropagation();this.yaw=this.drag.yaw+(event.clientX-this.drag.x)*.009;this.pitch=Math.max(-1.5,Math.min(1.5,this.drag.pitch+(event.clientY-this.drag.y)*.008));this.draw();}
  onUp(){if(!this.drag)return;const id=this.drag.id;this.drag=null;this.canvas.classList.remove('isPanning');try{this.canvas.releasePointerCapture(id);}catch(_){}}

  rotated(point,centre=[0,0,0]){let x=point[0]-centre[0],y=point[1]-centre[1],z=point[2]-centre[2],cy=Math.cos(this.yaw),sy=Math.sin(this.yaw),cp=Math.cos(this.pitch),sp=Math.sin(this.pitch);let xx=x*cy+z*sy,zz=-x*sy+z*cy,yy=y*cp-zz*sp;return [xx,yy,y*sp+zz*cp];}
  galaxyCentres(){const p=this.data.positions,origin=this.data.origin,centres=[];for(const which of [0,1]){let x=0,y=0,z=0,n=0;for(let i=0;i<p.length;i++){if(origin[i]!==which||this.data.component[i]===2)continue;x+=p[i][0];y+=p[i][1];z+=p[i][2];n++;}centres.push(n?[x/n,y/n,z/n]:[0,0,0]);}return centres;}
  centre(){const centres=this.galaxyCentres();if(this.focus==='mw')return centres[0];if(this.focus==='m31')return centres[1];return [(centres[0][0]+centres[1][0])/2,(centres[0][1]+centres[1][1])/2,(centres[0][2]+centres[1][2])/2];}
  colour(origin,component,catalogue,stellarColour){
    if(component===2)return origin===0?[42,108,166,48]:[158,70,48,48];
    if(component===1)return origin===0?[225,240,255,225]:[255,222,178,225];
    if(origin===0)return catalogue?[91,211,255,225]:[74,166,235,210];
    const warm=Math.max(0,Math.min(1,(Number(stellarColour)-.2)/3.5));
    return [255,Math.round(174-54*warm),Math.round(104-45*warm),catalogue?225:205];
  }

  drawAxes(centre,scale){const ctx=this.ctx,d=this.dpr,w=this.canvas.width,h=this.canvas.height,base=[w-70*d,h-63*d],axes=[[[30,0,0],'x','#ff897d'],[[0,30,0],'y','#75e8b8'],[[0,0,30],'z','#7ebcff']];ctx.lineWidth=2*d;ctx.font=`700 ${11*d}px Arial`;for(const [vector,label,colour] of axes){const r=this.rotated(vector,[0,0,0]);ctx.strokeStyle=colour;ctx.fillStyle=colour;ctx.beginPath();ctx.moveTo(...base);ctx.lineTo(base[0]+r[0]*.72*d,base[1]-r[1]*.72*d);ctx.stroke();ctx.fillText(label,base[0]+r[0]*.78*d,base[1]-r[1]*.78*d);}}
  drawLabels(){const ctx=this.ctx,d=this.dpr,w=this.canvas.width,h=this.canvas.height,focus=this.focus==='all'?'WHOLE ENCOUNTER':this.focus==='mw'?'MILKY WAY FOCUS':'M31 FOCUS';ctx.fillStyle='rgba(3,7,18,.82)';ctx.strokeStyle='rgba(108,219,255,.35)';ctx.lineWidth=d;ctx.beginPath();ctx.roundRect(20*d,18*d,430*d,78*d,14*d);ctx.fill();ctx.stroke();ctx.fillStyle='#f3f8ff';ctx.font=`700 ${18*d}px Arial`;ctx.fillText('3D SELF-GRAVITATING ENCOUNTER',38*d,48*d);ctx.fillStyle='#95b2d4';ctx.font=`${12*d}px Arial`;ctx.fillText(`${focus} · ${Number(this.data.simulated_particles).toLocaleString()} super-particles · t +${Number(this.data.time_gyr).toFixed(2)} Gyr`,38*d,73*d);ctx.fillStyle='rgba(180,204,232,.9)';ctx.font=`${11*d}px Arial`;ctx.fillText('Gaia DR3 seeds · PHAT sky pattern + modelled depth · illustrative N-body, not a fitted prediction',22*d,h-22*d);}

  // Drawn to match the rendered 2D frames (galaxy_collision_3d.py render):
  // small sharp dots over a faint blurred halo, normal blending. The earlier
  // additive blend let overlapping particles pile up into a white glow, so the
  // same run looked different the moment you rotated it.
  flatStyle(origin,component){
    if(component===2)return {colour:origin===0?[42,82,125]:[120,61,45],radius:1.0,alpha:38};
    if(component===1)return {colour:origin===0?[225,238,255]:[255,220,174],radius:2.0,alpha:210};
    return {colour:origin===0?[92,194,255]:[255,128,66],radius:1.45,alpha:205};
  }
  draw(){
    if(!this.data||!this.canvas.width||!this.canvas.height)return;
    const ctx=this.ctx,w=this.canvas.width,h=this.canvas.height,centre=this.centre();
    const extent=Math.max(30,Number(this.data.extent_kpc)||500);
    const scale=Math.min(w/(2.15*extent),h/(1.45*extent))*this.zoom;
    // Radii are in pixels of the 1280-wide reference frame.
    const unit=w/1280,points=[];
    for(let i=0;i<this.data.positions.length;i++){
      const component=this.data.component[i];if(component===2&&!this.showHalo)continue;
      const r=this.rotated(this.data.positions[i],centre);
      points.push({i,x:w*.5+r[0]*scale,y:h*.5-r[1]*scale,z:r[2],component});
    }
    points.sort((a,b)=>a.z-b.z);
    ctx.globalCompositeOperation='source-over';ctx.filter='none';
    ctx.fillStyle=this.transparent?'rgba(0,0,0,0)':'rgb(1,3,10)';
    if(this.transparent)ctx.clearRect(0,0,w,h);else ctx.fillRect(0,0,w,h);
    if(!this.glow){this.glow=document.createElement('canvas');this.glowCtx=this.glow.getContext('2d');}
    if(this.glow.width!==w||this.glow.height!==h){this.glow.width=w;this.glow.height=h;}
    const g=this.glowCtx;g.clearRect(0,0,w,h);
    const visible=[];
    for(const p of points){
      if(p.x<-8||p.x>w+8||p.y<-8||p.y>h+8)continue;
      const s=this.flatStyle(this.data.origin[p.i],p.component);visible.push([p,s]);
      g.fillStyle=`rgba(${s.colour[0]},${s.colour[1]},${s.colour[2]},${(s.alpha/3/255).toFixed(3)})`;
      g.beginPath();g.arc(p.x,p.y,s.radius*3*unit,0,Math.PI*2);g.fill();
    }
    ctx.filter=`blur(${(3.2*unit).toFixed(2)}px)`;ctx.drawImage(this.glow,0,0);ctx.filter='none';
    for(const [p,s] of visible){
      ctx.fillStyle=`rgba(${s.colour[0]},${s.colour[1]},${s.colour[2]},${(s.alpha/255).toFixed(3)})`;
      ctx.beginPath();ctx.arc(p.x,p.y,Math.max(.6,s.radius*unit),0,Math.PI*2);ctx.fill();
    }
    this.drawAxes(centre,scale);
    if(this.showLabels!==false)this.drawLabels();
  }
}

window.Galaxy3DView=Galaxy3DView;
