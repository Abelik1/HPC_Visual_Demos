// Wind-tunnel obstacle grid, shared by the dashboard and demo mode.
//
// The grid is what the visitor paints; the preset body is what the solver adds
// on top of it (leonardo_demos/demos/fluid.py build_obstacles). Both are drawn
// here, the preset as a faint silhouette, because the solver combines them:
// painting without seeing the preset used to produce a run that still had the
// single cylinder inside the visitor's own shape. Preset 3 is "custom shape
// only", where the drawing is the entire body.
class ObstacleBuilder {
  static ROWS=12;static COLS=24;
  static SHAPES={cell:[[0,0]],block:[[0,0],[1,0],[0,1],[1,1]],hbar:[[0,0],[1,0],[2,0],[3,0]],vbar:[[0,0],[0,1],[0,2],[0,3]]};
  // Every profile's lattice is 16:9; the grid canvas is 2:1.
  static LATTICE_ASPECT=16/9;

  constructor(host,{preset=0,large=false,onchange=null}={}){
    this.rows=ObstacleBuilder.ROWS;this.cols=ObstacleBuilder.COLS;
    this.cells=Array.from({length:this.rows},()=>Array(this.cols).fill(0));
    this.tool='cell';this.preset=Number(preset);this.onchange=onchange;
    this.el=document.createElement('div');this.el.className='obstacleBuilder'+(large?' large':'');
    this.el.innerHTML=`<div><div class="builderHeading"><b>Draw your own obstacle</b><small class="builderNote"></small></div><canvas class="builderCanvas" width="720" height="360" aria-label="Wind tunnel obstacle grid"></canvas></div><div class="builderTools"><button type="button" data-tool="cell" class="selected">Cell</button><button type="button" data-tool="block">2×2 block</button><button type="button" data-tool="hbar">Horizontal bar</button><button type="button" data-tool="vbar">Vertical bar</button><button type="button" data-tool="erase">Eraser</button><button type="button" data-tool="clear">Clear grid</button><span>Drag to paint. The left and right columns stay open so air can enter and leave.</span></div>`;
    host.appendChild(this.el);
    this.canvas=this.el.querySelector('canvas');this.ctx=this.canvas.getContext('2d');
    let drawing=false;
    this.canvas.onpointerdown=e=>{drawing=true;this.canvas.setPointerCapture(e.pointerId);this.paint(e);};
    this.canvas.onpointermove=e=>{if(drawing)this.paint(e);};
    this.canvas.onpointerup=this.canvas.onpointercancel=()=>{drawing=false;};
    this.el.querySelectorAll('[data-tool]').forEach(button=>button.onclick=()=>{
      const tool=button.dataset.tool;
      if(tool==='clear'){this.clear();return;}
      this.tool=tool;this.el.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('selected',b===button));
    });
    this.setPreset(this.preset);
  }
  hasCells(){return this.cells.some(row=>row.some(Boolean));}
  clear(){this.cells.forEach(row=>row.fill(0));this.draw();this.changed();}
  setCells(grid){
    if(!Array.isArray(grid))return;
    this.cells.forEach((row,y)=>row.forEach((_,x)=>{row[x]=grid[y]?.[x]?1:0;}));this.draw();
  }
  setPreset(preset){
    this.preset=Number(preset);
    this.el.querySelector('.builderNote').textContent=this.preset===3
      ?'Custom shape only: your drawing is the entire obstacle.'
      :'The faint outline is the preset body. The solver combines it with every cell you paint; choose "Custom shape only" to remove it.';
    this.draw();
  }
  changed(){if(this.onchange)this.onchange(this);}
  paint(event){
    const r=this.canvas.getBoundingClientRect();
    const x=Math.floor((event.clientX-r.left)/r.width*this.cols),y=Math.floor((event.clientY-r.top)/r.height*this.rows);
    const shape=this.tool==='erase'?ObstacleBuilder.SHAPES.cell:ObstacleBuilder.SHAPES[this.tool];
    for(const [dx,dy] of shape||[]){
      const xx=x+dx,yy=y+dy;
      if(xx>=1&&xx<this.cols-1&&yy>=0&&yy<this.rows)this.cells[yy][xx]=this.tool==='erase'?0:1;
    }
    this.draw();this.changed();
  }
  // Mirrors build_obstacles: positions are fractions of the lattice, radii are
  // fractions of its height, so a circle is an ellipse on the 2:1 grid.
  drawPreset(){
    const {ctx,canvas}=this,W=canvas.width,H=canvas.height,sx=1/ObstacleBuilder.LATTICE_ASPECT;
    const ellipse=(cx,cy,rx,ry)=>{ctx.beginPath();ctx.ellipse(cx*W,cy*H,rx*W,ry*H,0,0,Math.PI*2);ctx.fill();ctx.stroke();};
    ctx.save();ctx.fillStyle='rgba(255,196,92,.16)';ctx.strokeStyle='rgba(255,196,92,.75)';ctx.lineWidth=2;ctx.setLineDash([7,5]);
    if(this.preset===1){ellipse(.30,.35,.095*sx,.095);ellipse(.30,.65,.095*sx,.095);}
    else if(this.preset===2){
      ellipse(.26,.42,.075,.065);
      ctx.beginPath();ctx.rect((.39-.026)*W,(.62-.105)*H,.052*W,.21*H);ctx.fill();ctx.stroke();
    }else if(this.preset!==3)ellipse(.28,.5,.15*sx,.15);
    ctx.restore();
  }
  draw(){
    const {ctx,canvas,rows,cols,cells}=this,cw=canvas.width/cols,ch=canvas.height/rows;
    ctx.fillStyle='#06101e';ctx.fillRect(0,0,canvas.width,canvas.height);
    ctx.fillStyle='rgba(255,255,255,.035)';ctx.fillRect(0,0,cw,canvas.height);ctx.fillRect(canvas.width-cw,0,cw,canvas.height);
    ctx.strokeStyle='rgba(91,145,188,.28)';ctx.lineWidth=1;
    for(let x=0;x<=cols;x++){ctx.beginPath();ctx.moveTo(x*cw,0);ctx.lineTo(x*cw,canvas.height);ctx.stroke();}
    for(let y=0;y<=rows;y++){ctx.beginPath();ctx.moveTo(0,y*ch);ctx.lineTo(canvas.width,y*ch);ctx.stroke();}
    this.drawPreset();
    ctx.fillStyle='#8ceaff';
    for(let y=0;y<rows;y++)for(let x=0;x<cols;x++)if(cells[y][x])ctx.fillRect(x*cw+1,y*ch+1,cw-2,ch-2);
  }
}
