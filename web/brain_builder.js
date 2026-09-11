// Drag-and-drop "LEGO" builder for a visitor's neural network.
//
// The block catalogue comes from config/demo_specs.json (the same table the
// server validates against), so the UI never invents a block or a price.
// Mouse and touch both use pointer events; HTML5 drag-and-drop is unreliable
// on large exhibition displays.
class BrainBuilder {
  static ICONS={ray:'↗',gauge:'⏱',skid:'〰',compass:'🧭',wheel:'⎈',pedal:'⏵',brake:'⏹',ear:'👂',whisker:'〃',doppler:'≋',memory:'⟲',wing:'✦',dive:'⤓',click:'✺',sonar:'◎',brick:'▦'};
  static RAY_ORDER=[0,-45,45,-90,90,-15,15,-30,30,-60,60];

  constructor(host,catalogue,{title='Build your brain',subtitle='',accent='#ffc45c',agent='car'}={}){
    this.catalogue=catalogue;this.accent=accent;this.agent=agent;this.onchange=null;
    this.el=document.createElement('section');this.el.className='brainBuilder';this.el.style.setProperty('--bb-accent',accent);
    this.el.innerHTML=`<header class="bbHead"><div><b>${escapeHtml(title)}</b><small>${escapeHtml(subtitle)}</small></div><div class="bbBudget" aria-live="polite"><div class="bbMeter"><i></i></div><span></span></div><div class="bbPresets"></div></header><div class="bbBody"><div class="bbPalette" aria-label="Blocks you can add"></div><div class="bbBoard"><div class="bbCol" data-col="sensor"><h5>1 · Sensors <small>what it can feel</small></h5><div class="bbFan"></div><div class="bbDrop"></div></div><div class="bbArrow" aria-hidden="true">→</div><div class="bbCol" data-col="hidden"><h5>2 · Hidden layers <small>how it thinks</small></h5><div class="bbDrop"></div></div><div class="bbArrow" aria-hidden="true">→</div><div class="bbCol" data-col="action"><h5>3 · Actions <small>what it can do</small></h5><div class="bbDrop"></div></div></div><aside class="bbSide"><canvas class="bbPreview" width="300" height="190" aria-label="Preview of the network you built"></canvas><p class="bbStats"></p><p class="bbHint"></p></aside></div>`;
    host.appendChild(this.el);
    this.renderPresets();this.renderPalette();
    const preset=catalogue.presets?.[catalogue.default_preset]||{sensors:[],hidden:[],actions:[]};
    this.setSpec(preset);
  }

  // ---- state ------------------------------------------------------------
  setSpec(spec){
    const blocks=this.catalogue.blocks;
    this.sensors=(spec.sensors||[]).filter(s=>blocks[s.block]?.kind==='sensor').map(s=>({...s}));
    this.hidden=(spec.hidden||[]).map(Number).filter(w=>this.catalogue.hidden.costs[String(w)]!==undefined);
    const required=Object.entries(blocks).filter(([,b])=>b.kind==='action'&&b.required).map(([k])=>k);
    this.actions=[...new Set([...required,...(spec.actions||[]).filter(a=>blocks[a]?.kind==='action')])];
    this.render();
  }
  getSpec(){return {sensors:this.sensors.map(s=>({...s})),hidden:[...this.hidden],actions:[...this.actions]};}
  cost(spec=this.getSpec()){
    const b=this.catalogue.blocks,h=this.catalogue.hidden.costs;
    return spec.sensors.reduce((t,s)=>t+b[s.block].cost,0)+spec.hidden.reduce((t,w)=>t+h[String(w)],0)+spec.actions.reduce((t,a)=>t+b[a].cost,0);
  }
  // A memory block ("inputs":"actions") feeds back one value per action.
  inputs(){return Math.max(1,this.sensors.reduce((t,s)=>{const n=this.catalogue.blocks[s.block].inputs;return t+(n==='actions'?this.actions.length:(n||1));},0));}
  layerSizes(){return [this.inputs(),...this.hidden,this.actions.length];}
  parameters(){const l=this.layerSizes();let n=0;for(let i=1;i<l.length;i++)n+=l[i-1]*l[i]+l[i];return n;}
  remaining(){return this.catalogue.budget-this.cost();}

  // Returns an error string, or '' if the change was applied.
  tryChange(mutator){
    const before=this.getSpec();mutator();
    const spec=this.getSpec(),blocks=this.catalogue.blocks;
    let error='';
    if(this.cost(spec)>this.catalogue.budget+1e-9)error=`Not enough LEGO points — this needs ${this.cost(spec)} of ${this.catalogue.budget}.`;
    const counts={};spec.sensors.forEach(s=>{counts[s.block]=(counts[s.block]||0)+1;if(counts[s.block]>(blocks[s.block].max||1))error=`At most ${blocks[s.block].max||1} × ${blocks[s.block].label}.`;});
    if(spec.hidden.length>this.catalogue.hidden.max_layers)error=`At most ${this.catalogue.hidden.max_layers} hidden layers.`;
    if(error){this.sensors=before.sensors;this.hidden=before.hidden;this.actions=before.actions;this.flash(error);return error;}
    this.render();return '';
  }
  flash(message){const hint=this.el.querySelector('.bbHint');hint.textContent=message;hint.classList.add('bbError');this.el.querySelector('.bbBudget').classList.add('bbShake');clearTimeout(this.flashTimer);this.flashTimer=setTimeout(()=>{hint.classList.remove('bbError');this.el.querySelector('.bbBudget').classList.remove('bbShake');this.renderHint();},1800);}

  addBlock(key,angle=null){
    const block=this.catalogue.blocks[key];
    if(key==='brick')return this.tryChange(()=>this.hidden.push(Number(angle||Object.keys(this.catalogue.hidden.costs)[0])));
    if(block.kind==='action')return this.actions.includes(key)?'':this.tryChange(()=>this.actions.push(key));
    if(block.angles){
      const taken=new Set(this.sensors.filter(s=>s.block===key).map(s=>s.angle));
      const free=angle!==null?Number(angle):BrainBuilder.RAY_ORDER.find(a=>block.angles.includes(a)&&!taken.has(a))??block.angles.find(a=>!taken.has(a));
      if(free===undefined||taken.has(free))return this.flash(`Every ${block.label.toLowerCase()} direction is already used.`);
      return this.tryChange(()=>this.sensors.push({block:key,angle:free}));
    }
    if(this.sensors.some(s=>s.block===key)&&(block.max||1)<=1)return this.flash(`${block.label} is already fitted.`);
    return this.tryChange(()=>this.sensors.push({block:key}));
  }
  removeAt(col,index){
    if(col==='sensor')this.sensors.splice(index,1);
    else if(col==='hidden')this.hidden.splice(index,1);
    else {const key=this.actions[index];if(this.catalogue.blocks[key]?.required)return this.flash(`${this.catalogue.blocks[key].label} is required.`);this.actions.splice(index,1);}
    this.render();
  }

  // ---- rendering --------------------------------------------------------
  renderPresets(){
    const host=this.el.querySelector('.bbPresets');host.innerHTML='';
    Object.entries(this.catalogue.presets||{}).forEach(([key,preset])=>{const b=document.createElement('button');b.type='button';b.textContent=preset.label||key;b.onclick=()=>this.setSpec(preset);host.appendChild(b);});
  }
  tile(key,{placed=false,label=null,extra=''}={}){
    const block=key==='brick'?{label:label||'Neuron brick',icon:'brick',kind:'hidden',desc:'A layer of neurons between the senses and the actions.'}:this.catalogue.blocks[key];
    const t=document.createElement('div');t.className=`bbTile bbKind-${block.kind}${placed?' placed':''}`;t.dataset.block=key;
    t.innerHTML=`<span class="bbIcon" aria-hidden="true">${BrainBuilder.ICONS[block.icon]||'■'}</span><span class="bbLabel">${escapeHtml(label||block.label)}${extra}</span>`;
    t.title=block.desc||'';return t;
  }
  renderPalette(){
    const host=this.el.querySelector('.bbPalette');host.innerHTML='<h5>Blocks <small>drag onto the board, or click</small></h5>';
    const groups=[['sensor','Sensors'],['hidden','Brain'],['action','Actions']];
    groups.forEach(([kind,name])=>{
      const group=document.createElement('div');group.className='bbGroup';group.innerHTML=`<span>${name}</span>`;
      if(kind==='hidden'){Object.entries(this.catalogue.hidden.costs).forEach(([width,cost])=>{const t=this.tile('brick',{label:`${width} neurons`});t.dataset.width=width;t.dataset.cost=cost;t.insertAdjacentHTML('beforeend',`<em>${cost}</em>`);group.appendChild(t);});}
      else Object.entries(this.catalogue.blocks).filter(([,b])=>b.kind===kind&&!b.required).forEach(([key,b])=>{const t=this.tile(key);t.dataset.cost=b.cost;t.insertAdjacentHTML('beforeend',`<em>${b.cost}</em>`);group.appendChild(t);});
      host.appendChild(group);
    });
    host.querySelectorAll('.bbTile').forEach(t=>this.makeDraggable(t,()=>({key:t.dataset.block,width:t.dataset.width})));
  }
  render(){
    const cost=this.cost(),budget=this.catalogue.budget,over=cost>budget;
    const meter=this.el.querySelector('.bbMeter i');meter.style.width=`${Math.min(100,100*cost/budget)}%`;meter.classList.toggle('full',cost>=budget);
    this.el.querySelector('.bbBudget span').textContent=`${cost} / ${budget} LEGO points`;
    this.el.querySelector('.bbBudget').classList.toggle('over',over);
    // Palette tiles that no longer fit are dimmed, not hidden.
    const remaining=this.remaining();
    this.el.querySelectorAll('.bbPalette .bbTile').forEach(t=>{const key=t.dataset.block;let blocked=Number(t.dataset.cost)>remaining;if(key!=='brick'){const b=this.catalogue.blocks[key];if(b.kind==='action'&&this.actions.includes(key))blocked=true;if(b.kind==='sensor'&&!b.angles&&this.sensors.some(s=>s.block===key))blocked=true;if(b.angles&&this.sensors.filter(s=>s.block===key).length>=(b.max||1))blocked=true;}else if(this.hidden.length>=this.catalogue.hidden.max_layers)blocked=true;t.classList.toggle('blocked',blocked);});
    const cols=Object.fromEntries([...this.el.querySelectorAll('.bbCol')].map(c=>[c.dataset.col,c.querySelector('.bbDrop')]));
    cols.sensor.innerHTML='';cols.hidden.innerHTML='';cols.action.innerHTML='';
    // Directional sensors live on the fan; the rest are tiles.
    this.sensors.forEach((s,i)=>{if(this.catalogue.blocks[s.block].angles)return;const t=this.tile(s.block,{placed:true});this.addRemove(t,'sensor',i);cols.sensor.appendChild(t);});
    const rays=this.sensors.filter(s=>this.catalogue.blocks[s.block].angles);
    if(!this.sensors.length)cols.sensor.innerHTML='<p class="bbEmpty">No senses: this brain is blind.</p>';
    else if(!cols.sensor.children.length&&rays.length)cols.sensor.innerHTML=`<p class="bbEmpty">${rays.length} direction sensor${rays.length===1?'':'s'} on the fan</p>`;
    this.hidden.forEach((width,i)=>{const t=this.tile('brick',{placed:true,label:`${width} neurons`,extra:`<i class="bbDots">${'•'.repeat(width)}</i>`});t.title='Click to change the number of neurons';t.onclick=e=>{if(e.target.closest('.bbX'))return;const widths=Object.keys(this.catalogue.hidden.costs).map(Number);const next=widths[(widths.indexOf(width)+1)%widths.length];this.tryChange(()=>{this.hidden[i]=next;});};this.addRemove(t,'hidden',i);cols.hidden.appendChild(t);});
    if(!this.hidden.length)cols.hidden.innerHTML='<p class="bbEmpty">No hidden layer: pure reflexes, senses wired straight to actions.</p>';
    this.actions.forEach((key,i)=>{const t=this.tile(key,{placed:true});if(this.catalogue.blocks[key].required){t.classList.add('locked');t.title='Required: every agent needs this control';}else this.addRemove(t,'action',i);cols.action.appendChild(t);});
    this.renderFan();this.renderPreview();this.renderHint();
    if(this.onchange)this.onchange(this.getSpec());
  }
  addRemove(tile,col,index){
    const x=document.createElement('button');x.type='button';x.className='bbX';x.setAttribute('aria-label','Remove block');x.textContent='×';x.onclick=e=>{e.stopPropagation();this.removeAt(col,index);};tile.appendChild(x);
    this.makeDraggable(tile,()=>({remove:{col,index}}));
  }
  renderFan(){
    const fan=this.el.querySelector('.bbFan'),rayBlocks=Object.entries(this.catalogue.blocks).filter(([,b])=>b.angles);
    if(!rayBlocks.length){fan.hidden=true;return;}
    const angles=[...new Set(rayBlocks.flatMap(([,b])=>b.angles))].sort((a,b)=>a-b);
    const w=260,h=150,cx=w/2,cy=h-26,r=[92,118];
    let svg=`<svg viewBox="0 0 ${w} ${h}" role="group" aria-label="Direction sensors: click a spoke to add or remove">`;
    angles.forEach(a=>{
      const placed=this.sensors.find(s=>s.angle===a&&this.catalogue.blocks[s.block].angles);
      const kind=placed?placed.block:null,long=kind&&kind!==rayBlocks[0][0];
      const len=long?r[1]:r[0],rad=(a-90)*Math.PI/180,x=cx+Math.cos(rad)*len,y=cy+Math.sin(rad)*len;
      svg+=`<g class="bbSpoke${kind?' on':''}${long?' long':''}" data-angle="${a}" tabindex="0"><line x1="${cx}" y1="${cy}" x2="${x}" y2="${y}"/><circle cx="${x}" cy="${y}" r="${kind?8:6}"/><title>${kind?this.catalogue.blocks[kind].label:'Empty'} · ${a===0?'straight ahead':`${Math.abs(a)}° ${a<0?'left':'right'}`}</title></g>`;
    });
    svg+=this.agent==='bat'?`<text x="${cx}" y="${cy+9}" class="bbAgent">🦇</text>`:`<path class="bbAgentShape" d="M${cx} ${cy-15} L${cx+9} ${cy+11} L${cx} ${cy+5} L${cx-9} ${cy+11} Z"/>`;
    fan.innerHTML=svg+'</svg><small>Click a spoke to cycle: off → '+rayBlocks.map(([,b])=>b.label.toLowerCase()).join(' → ')+'</small>';
    fan.querySelectorAll('.bbSpoke').forEach(g=>{const cycle=()=>{
      const a=Number(g.dataset.angle),i=this.sensors.findIndex(s=>s.angle===a&&this.catalogue.blocks[s.block].angles);
      const order=rayBlocks.filter(([,b])=>b.angles.includes(a)).map(([k])=>k);
      const current=i>=0?this.sensors[i].block:null,next=order[(order.indexOf(current)+1)]??null;
      this.tryChange(()=>{if(i>=0)this.sensors.splice(i,1);if(next)this.sensors.push({block:next,angle:a});});
    };g.onclick=cycle;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();cycle();}};});
  }
  renderPreview(){
    const canvas=this.el.querySelector('.bbPreview'),ctx=canvas.getContext('2d'),sizes=this.layerSizes(),w=canvas.width,h=canvas.height;
    ctx.clearRect(0,0,w,h);const xs=sizes.map((_,i)=>24+(w-48)*i/Math.max(1,sizes.length-1));
    const ys=sizes.map(n=>Array.from({length:n},(_,j)=>h/2+(j-(n-1)/2)*Math.min(12,(h-24)/Math.max(1,n))));
    ctx.lineWidth=.6;
    for(let i=1;i<sizes.length;i++)for(const a of ys[i-1])for(const b of ys[i]){ctx.strokeStyle='rgba(120,210,255,.22)';ctx.beginPath();ctx.moveTo(xs[i-1],a);ctx.lineTo(xs[i],b);ctx.stroke();}
    sizes.forEach((n,i)=>ys[i].forEach(y=>{ctx.fillStyle=i===0?'#78e2ff':i===sizes.length-1?this.accent:'#c9d7ea';ctx.beginPath();ctx.arc(xs[i],y,3.6,0,Math.PI*2);ctx.fill();}));
    this.el.querySelector('.bbStats').innerHTML=`<b>${this.parameters().toLocaleString()}</b> connections to evolve<br><span>${sizes.join(' → ')} neurons</span>`;
  }
  renderHint(){
    const hint=this.el.querySelector('.bbHint');if(hint.classList.contains('bbError'))return;
    const rays=this.sensors.filter(s=>this.catalogue.blocks[s.block].angles).map(s=>s.angle);
    let text='';const has=key=>this.sensors.some(s=>s.block===key),blocks=this.catalogue.blocks;
    if(blocks.ear_left&&has('ear_left')!==has('ear_right'))text='With one ear the bat hears a moth but cannot tell which side it is on.';
    else if(blocks.ear_left&&!has('ear_left'))text='No ears: the bat calls but cannot hear a single echo.';
    else if(blocks.tympanum&&!has('tympanum')&&!has('direction'))text='This moth cannot hear bats coming at all.';
    else if(blocks.jam&&this.actions.includes('jam'))text='Jamming clicks cost energy. Will evolution learn to click only when a bat is close?';
    else if(!this.sensors.length)text='With no sensors the network gets no information at all. Can evolution still find anything?';
    else if(rays.length&&rays.every(a=>a<0))text='Every direction sensor points left. Can it see the right-hand wall?';
    else if(rays.length&&rays.every(a=>a>0))text='Every direction sensor points right. Can it see the left-hand wall?';
    else if(this.hidden.length>=2)text='A big brain has more to tune, so evolution may need more generations.';
    else if(!this.hidden.length)text='A reflex brain: small, fast to evolve, but it cannot combine clues.';
    else text=`${this.remaining()} points left to spend.`;
    hint.textContent=text;
  }

  // ---- pointer drag -----------------------------------------------------
  makeDraggable(tile,payload){
    tile.addEventListener('pointerdown',event=>{
      if(event.button!==0||event.target.closest('.bbX'))return;
      const start={x:event.clientX,y:event.clientY},data=payload();let ghost=null;
      const move=e=>{if(!ghost&&Math.hypot(e.clientX-start.x,e.clientY-start.y)>6){ghost=tile.cloneNode(true);ghost.classList.add('bbGhost');document.body.appendChild(ghost);tile.classList.add('dragging');}
        if(ghost){ghost.style.left=`${e.clientX}px`;ghost.style.top=`${e.clientY}px`;this.el.querySelectorAll('.bbCol').forEach(c=>c.classList.toggle('bbTarget',c.contains(document.elementFromPoint(e.clientX,e.clientY))));}};
      const up=e=>{window.removeEventListener('pointermove',move);window.removeEventListener('pointerup',up);tile.classList.remove('dragging');this.el.querySelectorAll('.bbCol').forEach(c=>c.classList.remove('bbTarget'));
        if(!ghost){if(!data.remove)this.addBlock(data.key,data.width??null);return;}
        ghost.remove();const target=document.elementFromPoint(e.clientX,e.clientY),board=target?.closest?.('.bbBoard');
        if(data.remove){if(!board||!this.el.contains(board))this.removeAt(data.remove.col,data.remove.index);return;}
        if(board&&this.el.contains(board))this.addBlock(data.key,data.width??null);};
      window.addEventListener('pointermove',move);window.addEventListener('pointerup',up);
    });
  }
}
window.BrainBuilder=BrainBuilder;
