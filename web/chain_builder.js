// "Write your own protein": the Molecular Machine's sequence builder, shared by
// the dashboard and demo mode.
//
// Each bead is one of four kinds (leonardo_demos/demos/molecular_dynamics.py):
// H water-avoiding, P water-loving, + positive, - negative. Until the visitor
// touches it, the builder mirrors the chosen preset and the run uses that
// preset; once edited, getChain() returns the visitor's own sequence.
class ChainBuilder {
  static KINDS=[
    {k:'H',label:'Oily',help:'Water-avoiding: oily beads hide from water together',colour:'#ffaa40'},
    {k:'P',label:'Water-loving',help:'Happy in water: stays on the outside',colour:'#60d6ff'},
    {k:'+',label:'Plus',help:'Positive charge: attracts minus, repels plus',colour:'#7884ff'},
    {k:'-',label:'Minus',help:'Negative charge: attracts plus, repels minus',colour:'#ff5c88'}];
  static MIN=6;static MAX=80;
  // Same recipes as preset_text() in molecular_dynamics.py.
  static preset(index,length){
    const n=Math.max(6,length|0),rep=(s,k)=>s.repeat(Math.ceil(k/s.length)+1).slice(0,k);
    switch(((index|0)%4+4)%4){
      case 0:return rep('PHPPHHPHPPHPHHPPHPHH',n);
      case 1:{const turn=n>=20?6:2,strand=Math.floor((n-turn)/2);return rep('HP',strand)+'P'.repeat(n-2*strand)+[...rep('PH',strand)].reverse().join('');}
      case 2:{const end=Math.max(2,Math.floor(n/4));return rep('P+',end)+rep('PHHPH',n-2*end)+rep('-P',end);}
      default:return Array.from({length:n},(_,i)=>i%9===4?'H':'P').join('');
    }
  }
  static injectStyle(){
    if(document.getElementById('chainBuilderStyle'))return;
    const s=document.createElement('style');s.id='chainBuilderStyle';
    s.textContent=`.chainBuilder{grid-column:1/-1;min-width:0;border:1px solid #26436a;border-radius:16px;padding:14px 16px;background:#07101f;display:flex;flex-direction:column;gap:10px;color:#e6eefb}
.chainBuilder .cbHead{display:flex;justify-content:space-between;gap:12px;flex-wrap:wrap;align-items:baseline}
.chainBuilder .cbHead b{font-size:16px}.chainBuilder .cbHead small{color:#8ea3c4;font-size:13px}
.chainBuilder .cbPaint,.chainBuilder .cbTools{display:flex;gap:6px;flex-wrap:wrap;align-items:center}
.chainBuilder button{font:inherit;font-size:13px;color:#e6eefb;background:#0b1630;border:1px solid #26436a;border-radius:10px;padding:6px 11px;cursor:pointer}
.chainBuilder button:hover{border-color:#67f0d0}
.chainBuilder .cbPaint button[aria-pressed="true"]{border-color:#fff;background:#13284a}
.chainBuilder .cbSwatch{display:inline-block;width:13px;height:13px;border-radius:50%;vertical-align:-2px;margin-right:6px;box-shadow:inset -2px -2px 3px rgba(0,0,0,.35)}
.chainBuilder .cbBeads{display:flex;flex-wrap:wrap;gap:3px;align-items:center;padding:8px;border-radius:12px;background:#030813;min-height:40px;cursor:crosshair;user-select:none;touch-action:none}
.chainBuilder .cbBead{width:24px;height:24px;border-radius:50%;border:0;padding:0;font-size:11px;font-weight:700;color:#04111f;box-shadow:inset -3px -3px 5px rgba(0,0,0,.35),inset 2px 2px 4px rgba(255,255,255,.35);position:relative}
.chainBuilder .cbBead+.cbBead::before{content:"";position:absolute;left:-4px;top:11px;width:4px;height:2px;background:#8ea3c4}
.chainBuilder.large .cbBead{width:30px;height:30px;font-size:13px}.chainBuilder.large .cbBead+.cbBead::before{top:14px}
.chainBuilder .cbStatus{color:#8ea3c4;font-size:13px}.chainBuilder .cbStatus b{color:#67f0d0}
.chainBuilder select{font:inherit;font-size:13px;background:#07101f;color:#e6eefb;border:1px solid #26436a;border-radius:10px;padding:6px 8px}`;
    document.head.appendChild(s);
  }
  constructor(host,{preset=0,length=40,large=false,onchange=null}={}){
    ChainBuilder.injectStyle();
    this.presetIndex=Number(preset)||0;this.length=Math.max(ChainBuilder.MIN,Math.min(ChainBuilder.MAX,length|0));
    this.text=ChainBuilder.preset(this.presetIndex,this.length);this.custom=false;this.brush='H';this.onchange=onchange;
    this.el=document.createElement('div');this.el.className='chainBuilder'+(large?' large':'');
    this.el.innerHTML=`<div class="cbHead"><b>Write your own protein</b><small>Pick a kind of bead, then click or drag along the chain to paint it.</small></div>
      <div class="cbPaint" role="group" aria-label="Bead to paint">${ChainBuilder.KINDS.map(k=>`<button type="button" data-k="${k.k}" title="${k.help}" aria-pressed="${k.k==='H'}"><span class="cbSwatch" style="background:${k.colour}"></span>${k.label} (${k.k})</button>`).join('')}</div>
      <div class="cbBeads" aria-label="The chain, bead by bead"></div>
      <div class="cbTools"><button type="button" data-act="add">+ bead</button><button type="button" data-act="remove">− bead</button><button type="button" data-act="random">Random</button>
        <select data-act="preset" aria-label="Start from a preset"><option value="">Start from…</option><option value="0">Oily core</option><option value="1">Hairpin</option><option value="2">Charge zipper</option><option value="3">Soluble</option></select>
        <span class="cbStatus"></span></div>`;
    host.appendChild(this.el);
    this.beads=this.el.querySelector('.cbBeads');
    this.el.querySelectorAll('.cbPaint button').forEach(b=>b.onclick=()=>{this.brush=b.dataset.k;this.el.querySelectorAll('.cbPaint button').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));});
    let painting=false,last=null;
    // A fast drag skips beads between pointer samples; paint the whole run.
    const paintAt=e=>{const bead=document.elementFromPoint(e.clientX,e.clientY)?.closest?.('.cbBead');if(!bead||!this.beads.contains(bead))return;
      const i=Number(bead.dataset.i),from=last===null?i:last;last=i;
      const chars=[...this.text];let changed=false;
      for(let k=Math.min(from,i);k<=Math.max(from,i);k++)if(chars[k]!==this.brush){chars[k]=this.brush;changed=true;}
      if(changed){this.text=chars.join('');this.edited();}};
    this.beads.onpointerdown=e=>{painting=true;last=null;try{this.beads.setPointerCapture(e.pointerId);}catch(_){}paintAt(e);};
    this.beads.onpointermove=e=>{if(painting)paintAt(e);};
    this.beads.onpointerup=this.beads.onpointercancel=()=>{painting=false;};
    this.el.querySelector('[data-act=add]').onclick=()=>{if(this.text.length<ChainBuilder.MAX){this.text+=this.brush;this.edited();}};
    this.el.querySelector('[data-act=remove]').onclick=()=>{if(this.text.length>ChainBuilder.MIN){this.text=this.text.slice(0,-1);this.edited();}};
    this.el.querySelector('[data-act=random]').onclick=()=>{const w='HHHPPPP+-';this.text=Array.from(this.text,()=>w[Math.floor(Math.random()*w.length)]).join('');this.edited();};
    this.el.querySelector('[data-act=preset]').onchange=e=>{if(e.target.value==='')return;this.text=ChainBuilder.preset(Number(e.target.value),this.text.length);e.target.value='';this.edited();};
    this.render();
  }
  edited(){this.custom=true;this.render();this.onchange?.(this.text);}
  // Follow the Sequence preset and chain length until the visitor edits.
  setPreset(index,length=this.length){
    this.presetIndex=Number(index)||0;this.length=Math.max(ChainBuilder.MIN,Math.min(ChainBuilder.MAX,length|0));
    if(!this.custom){this.text=ChainBuilder.preset(this.presetIndex,this.length);this.render();}
  }
  reset(){this.custom=false;this.setPreset(this.presetIndex,this.length);}
  getChain(){return this.custom?this.text:null;}
  setVisible(on){this.el.style.display=on?'':'none';}
  render(){
    const colour=Object.fromEntries(ChainBuilder.KINDS.map(k=>[k.k,k.colour]));
    this.beads.innerHTML=[...this.text].map((c,i)=>`<span class="cbBead" data-i="${i}" style="background:${colour[c]}" title="Bead ${i+1}: ${c}">${c==='+'||c==='-'?c:''}</span>`).join('');
    const count=k=>[...this.text].filter(c=>c===k).length;
    this.el.querySelector('.cbStatus').innerHTML=`${this.text.length} beads · ${count('H')} oily · ${count('+')+count('-')} charged · ${this.custom?'<b>your own sequence</b>':'preset sequence'}${this.custom?' <button type="button" data-act="reset">Back to preset</button>':''}`;
    const reset=this.el.querySelector('[data-act=reset]');if(reset)reset.onclick=()=>this.reset();
  }
}
window.ChainBuilder=ChainBuilder;
