// Demo-day viewer.
//
// Same server, same APIs and the same saved runs as the full dashboard at "/".
// The difference is what a visitor is asked to decide: two or three large
// controls and one run button, with the explanation printed under the picture
// instead of hidden behind overlay toggles.  Everything else - quality preset,
// compute backend, frame count, solver, ensemble width and the raw profile
// values - lives in the presenter drawer behind the gear.
//
// The scale reveal is deliberately NOT offered for the physics demos.  For a
// large model the interesting compute is inside the single simulation, not in
// how many copies fit on screen, so those demos always run one simulation.
// Where a population genuinely IS the experiment - the evolving racers, the
// bat/moth arms race, the wall of networks - the population is a first-class
// visitor control instead: how many independent searches, and whether to watch
// them together in one arena or separately, one per box.

function escapeHtml(value){return String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
const $=s=>document.querySelector(s);
const pad=n=>String(n).padStart(4,'0');

// ---------------------------------------------------------------- config --
// Which demos the stand shows, in order, and the few knobs a visitor gets.
// `controls` entries are either a parameter key from config/demo_specs.json
// (optionally re-labelled, or forced to big segmented buttons with `choices`)
// or the special population control.
const KIOSK={
  fusion_plasma:{
    about:'A fusion plasma held by magnetic fields inside a torus-shaped vessel. Mode 2 hands the magnets to a neural network that learns, shot by shot, to keep the plasma off the wall.',
    tag:'Fusion',
    blurb:'Hold a plasma hotter than the Sun inside a magnetic bottle.',
    read:'Bright trails are particles carried by the computed magnetic field. In Guardian mode the colour is how close each one is to the wall.',
    story:[
      'A fusion plasma is hotter than the centre of the Sun, so no solid wall can hold it. Magnetic fields have to make the bottle.',
      'The luminous trails are markers carried by the field the simulation is solving.',
      'Heating feeds the plasma, waves interact, and smooth motion turns into turbulence.',
      'The physics never changed. Only the controller holding it did.'],
    controls:[
      {method:true,label:'What the machine is doing'},
      {key:'magnetic_field',label:'Magnetic field',unit:'T',decimals:1},
      {key:'heating',label:'Heating power',unit:'MW',decimals:0},
      {key:'instability',label:'Instability drive',decimals:2,onlyMethod:'guardian'}],
    views:[{id:'fusion',label:'3D torus',kind:'fusion',needs:'fusion_view'},
           {id:'frames',label:'Flat view',kind:'frames'}],
    // Guardian keys first, passive-mode keys after: a run only ever has one set.
    numbers:['wall losses this shot','confined markers','best shot so far',
             'regime','turbulence index','passive tracers']},

  galaxy_collision_3d:{
    about:'The Milky Way and Andromeda falling together, with every massive particle pulling on every other one in three dimensions.',
    tag:'Astrophysics',
    blurb:'Every particle pulls on every other one, in full 3D.',
    read:'Blue belongs to the Milky Way, orange to Andromeda. Faint points are dark-matter super-particles; all of them are in the force calculation. Seeded from Gaia DR3 and PHAT data; illustrative, not a fitted prediction. The twinkling background stars are decoration.',
    starfield:true,
    story:[
      'Two galaxies fall towards each other under nothing but their own gravity.',
      'Every massive particle is pulling on every other one, every step.',
      'Tidal tails fling stars far out of both discs.',
      'What is left over remembers the orbit that made it.'],
    controls:[
      {key:'impact',label:'How head-on',decimals:2},
      {key:'speed',label:'Approach speed',decimals:2},
      {key:'disc_tilt',label:'Disc tilt',unit:'°',decimals:0}],
    views:[{id:'frames',label:'Flat view',kind:'frames'},
           {id:'galaxy3d',label:'Rotate in 3D',kind:'galaxy3d',needs:'galaxy3d_view'}],
    numbers:['separation','massive particles','softening']},

  neuro_racers:{
    about:'Hundreds of cars share the brain a visitor built, each with different random weights. The best drivers of each generation become the parents of the next.',
    tag:'AI',ai:true,
    blurb:'Build a car brain from blocks, then evolve it until it can drive.',
    read:'Gold is the best car of this generation, blue trails are the runners-up and red crosses are crashes. Nobody wrote the driving.',
    readChampion:'Saved champion networks, frozen, replayed from a start they have never seen. Each colour is a different generation’s champion.',
    readLanes:'The same generation as Training, but every car drives its own copy of the track. Gold is the champion; a dimmed box has crashed.',
    story:[
      'Every car carries the brain you built, but each one starts with different random weights.',
      'Most crash or spin. The few that get furthest become the parents.',
      'Their children inherit mutated copies of those weights, and the driving improves generation by generation.',
      'Nobody programmed the driving. It was found by selection alone.'],
    controls:[
      {key:'track',label:'Track'},
      {key:'mutation',label:'Mutation strength',decimals:2},
      {ghosts:true,label:'Race the ghosts'},
      {name:true,label:'Your name',help:'Your name tag, on your champion and whenever it races as a ghost.'}],
    // Training: the population evolving, one full drive per generation.
    // One per box: that same training, each car on its own copy of the track.
    // Inference: a frozen champion network driving from a fresh start.
    views:[{id:'arena',label:'Training',kind:'arena',needs:'arena_view'},
           {id:'lanes',label:'One per box',kind:'arena',lanes:true,needs:'arena_view'},
           {id:'champion',label:'Champion drives',kind:'champion',needs:'lab'}],
    numbers:['generation','fastest lap','cars completing a lap']},

  bat_vs_moth:{
    about:'A bat that hunts by sonar and moths that learn to jam it, evolving against each other in a dark cave.',
    tag:'AI',ai:true,
    blurb:'One visitor builds a hunting bat. Another builds a moth that jams its sonar.',
    read:'Amber is the champion bat and its call rings. Cyan glows are moth echoes it heard; magenta rings are fake echoes from a jamming moth.',
    readChampion:'The final champion bat and moths, frozen, in a fresh random start. White crosses are catches; magenta rings are jamming.',
    readLanes:'The same generation as Training, each of its best caves in its own box. The first box is the champion’s cave; white crosses are catches.',
    story:[
      'The cave is dark. The screen shows only what the bat’s own calls reveal.',
      'At first the moths just flutter. The bats that learn to turn toward the louder ear catch the most.',
      'Now the moths evolve too. Magenta phantoms are moths clicking to fake their own echo.',
      'An arms race, with neither side designed by hand.'],
    controls:[
      {key:'cave',label:'Cave layout',decimals:0},
      {key:'moths',label:'Moths per cave',decimals:0},
      {name:true,label:'Your names',help:'Your name tag, shown on your champion.'}],
    // Training in the bat's senses or the lit cave; one per box is the same
    // training, each of the best caves in its own box; the champion hunts
    // (inference) can be watched either way too.
    views:[{id:'arena',label:'Training',kind:'arena',needs:'arena_view'},
           {id:'arenaLit',label:'Training · lit cave',kind:'arena',lit:true,needs:'arena_view'},
           {id:'lanes',label:'One per box',kind:'arena',lanes:true,lit:true,needs:'arena_lanes'},
           {id:'champion',label:'Champion hunts',kind:'champion',needs:'lab'},
           {id:'championLit',label:'Champion · lit cave',kind:'champion',lit:true,needs:'lab'}],
    numbers:['generation','moths caught','calls jammed']},

  neural_wall:{
    about:'A neural network learns to redraw a picture from nothing but pixel coordinates. Its weights are the compressed file: smaller than the picture, at the cost of some detail.',
    tag:'AI',ai:true,
    blurb:'Squeeze a picture into a tiny neural network and see what survives.',
    read:'Left is the picture going in, squeezed to the chosen size. Right is what the network draws back from its weights alone. Quality is in dB: higher is closer to the original.',
    story:[
      'The network is only ever told an x and a y and asked for a colour. It never sees the picture as a picture.',
      'Everything it knows is stored in its weights. Count them and you have the size of the compressed file.',
      'Squeeze harder, with fewer pixels or a smaller network, and fine detail is the first thing to go.',
      'A JPEG of the same size is the honest benchmark. Sometimes the network gets close; usually JPEG still wins.'],
    creator:'target',
    controls:[
      {setting:'tile',label:'Squeeze the picture to',values:[32,64,96,128],unit:'px',default:96,
       help:'Fewer pixels means less to remember, and less detail to keep.'},
      {key:'target',label:'Or learn a generated pattern',choices:['Nebula','RGB waves','Flower','Ribbon']},
      {population:true,label:'Networks trained at once',values:[1,4,16,36,64],
       help:'Each has a different size and learning rate. The best one that is smaller than the picture wins.'}],
    views:[{id:'frames',label:'In → out',kind:'frames'},
           {id:'wall',label:'All networks',kind:'frames',folder:'modes/wall',needs:'view_modes'}],
    numbers:['compression','quality','network size']},

  black_hole:{
    about:'A camera beside a real black hole found by the Gaia satellite, looking at the real sky of 1.8 million Gaia stars through the hole’s gravity.',
    tag:'Astrophysics',
    legacyMethod:'weak_field',
    blurb:'Fly beside a real black hole and see the Gaia sky bent around it.',
    read:'Every pixel’s light is traced backwards along its exact path around the hole to the real star it came from. The black disc is the shadow: those rays fell in.',
    story:[
      'This black hole is real. Gaia found it by watching a star orbit something invisible.',
      'Every star you see is a real Gaia star, moved to where it would appear from here.',
      'The shadow is 2.6 times wider than the horizon: light that comes too close is captured.',
      'Near the edge, light loops the hole before reaching us. Switch to the ray paths to see it.'],
    controls:[
      {key:'target',label:'Which black hole',onlyMethod:'schwarzschild',
       choices:['Gaia BH3','Gaia BH1','Gaia BH2','In our Solar System']},
      {key:'camera_distance',label:'How close',unit:'r_s',decimals:1,onlyMethod:'schwarzschild',
       help:'In Schwarzschild radii, the size of the horizon. Light can orbit at 1.5 r_s.'},
      {key:'orbit',label:'Circle around it',unit:'°',onlyMethod:'schwarzschild'},
      {key:'dive',label:'Dive in',decimals:2,onlyMethod:'schwarzschild'},
      {key:'mass',label:'Black-hole mass',decimals:2,onlyMethod:'weak_field'},
      {key:'lens_count',label:'How many black holes',onlyMethod:'weak_field'}],
    views:[{id:'frames',label:'Camera view',kind:'frames'},
           {id:'3d',label:'Light paths',kind:'frames',folder:'modes/3d',needs:'view_modes'}],
    numbers:['black hole','camera distance','shadow','starlight blueshift']},

  fluid:{
    about:'Air flowing past an obstacle in a lattice-Boltzmann wind tunnel. Draw your own shape to see what its wake does.',
    tag:'Engineering',
    blurb:'Put an obstacle in a wind tunnel and watch the wake go turbulent.',
    read:'Colour is speed. The streaks are tracer particles following the computed velocity field.',
    story:[
      'At first the flow is almost boring.',
      'The obstacle forces the fluid to organise itself into vortices.',
      'The vortices shed alternately, and the wake becomes complex.',
      'Underneath is a lattice of cells being updated over and over.'],
    // Drawing your own body is the heart of this demo, not an advanced option.
    creator:'obstacles',
    controls:[
      {key:'speed',label:'Wind speed',decimals:3},
      {key:'obstacle',label:'Obstacle shape'}],
    numbers:['Reynolds number','inlet speed','lattice step']},

  cosmic_web:{
    about:'An almost-smooth young universe in which gravity grows tiny differences into clusters, filaments and voids. Change the recipe and see whether a web can form at all.',
    tag:'Cosmology',
    blurb:'Choose what the universe is made of, then let gravity build the web.',
    read:'Brightness is density: knots are clusters, threads are filaments, dark gaps are voids. The recipes are a qualitative model, not a precision cosmology.',
    story:[
      'The early universe is almost uniform. Almost is the important word.',
      'In our universe matter dominates the expansion, and gravity amplifies the tiny differences into a web.',
      'Keep radiation in charge instead: space still expands, but radiation does not clump, so structure grows far more slowly.',
      'Take away dark matter and ordinary matter alone is too little, and started too smooth, to build the web in time.'],
    controls:[
      {key:'recipe',label:'Recipe of the Universe'},
      {key:'gravity',label:'Gravity strength',decimals:2},
      {key:'helium',label:'Helium fraction',decimals:2}],
    numbers:['universe','clumping density contrast','solver step']},

  galaxy_collision:{
    about:'Our actual future: the Milky Way and Andromeda meet, and gravity pulls out long tidal tails of stars.',
    tag:'Astrophysics',
    blurb:'Our actual future: the Milky Way meets Andromeda.',
    read:'Blue mass belongs to the Milky Way, warm mass to Andromeda. Where they overlap the picture brightens towards white. The twinkling background stars are decoration, not simulated.',
    starfield:true,
    story:[
      'Two calm galaxies approach each other.',
      'Their mutual gravity draws out long tidal tails.',
      'The tails are a memory of the orbit that made them.',
      'The final shape depends on how the two came in.'],
    controls:[
      {key:'impact',label:'How head-on',decimals:2},
      {key:'speed',label:'Approach speed',decimals:2},
      {key:'andromeda_mass',label:'Andromeda mass',unit:'T☉',decimals:1}],
    numbers:['time','separation','tracers']},
};
// The external NBody-EuroHPC code (MUrB), run through the same run API.
KIOSK.nbody_murb={
  about:'The C++ N-body code that was validated on Leonardo: every body pulls on every other body, every step. The dashboard runs the real program and draws it the way its own viewer does.',
  tag:'HPC code',
  blurb:'Run the real Leonardo N-body code and watch every body pull on every other.',
  read:'Each ring is one body, coloured by speed: deep blue is slow, cyan is fast, white is the fastest in that frame. The numbers alongside are the code’s own timing.',
  story:[
    'This is not a toy re-implementation: it is the same C++ code that runs on Leonardo’s A100 GPUs.',
    'With N bodies there are N × N forces every step. Double the bodies, four times the work.',
    'Same physics, different implementation: compare the naive reference with SIMD and OpenMP.',
    'On Leonardo the same code runs on one A100, or four at once.'],
  controls:[
    {method:true,label:'Implementation'},
    {key:'scheme',label:'Starting setup',choices:['Rotating galaxy','Random cloud']},
    {key:'dt',label:'Timestep',unit:'s',decimals:0}],
  views:[{id:'frames',label:'MUrB camera',kind:'frames'},
         {id:'side',label:'Side view',kind:'frames',folder:'modes/side',needs:'view_modes'},
         {id:'top',label:'Top view',kind:'frames',folder:'modes/top',needs:'view_modes'}],
  numbers:['simulated time','bodies','GFLOP/s','ms / iteration']};
KIOSK.molecular_dynamics={
  about:'Two things molecules do. Fold: a chain of oily, water-loving and charged beads, written by you, curls up while every bead pulls on every other one. Machine: a ring threaded on an axle that a switch sends from one end to the other.',
  tag:'Biophysics',
  blurb:'Write a protein and watch it fold, or run a machine made of one molecule.',
  read:'Each ball is a group of atoms; sticks are bonds.',
  readByMethod:{
    fold:'Amber beads avoid water, cyan beads like it, blue is plus and pink is minus. The oily beads hide together in the middle; opposite charges zip up.',
    shuttle:'The gold ring is threaded on the grey axle; the big end beads stop it falling off. The bright green station is the sticky one. Nothing pushes the ring: it jiggles along on heat alone until the sticky station catches it.'},
  story:[
    'At this scale nothing sits still. Water molecules kick every bead, billions of times a second.',
    'Fold: oily beads hide from water together. That single rule is why proteins have a core.',
    'Every bead feels every other bead, every step: that is where the computing goes, and real proteins have hundreds of thousands of atoms.',
    'Machine: the 2016 Nobel Prize in Chemistry was for molecules like this ring on an axle. The switch does not push; it only changes where the ring is caught.'],
  creator:'chain',
  controls:[
    {method:true,label:'What to build'},
    {key:'sequence',label:'Start from',onlyMethod:'fold'},
    {key:'temperature',label:'Temperature',unit:'K',decimals:0},
    {key:'solvent',label:'Water strength',decimals:2,onlyMethod:'fold'},
    {key:'drive',label:'Switch strength',decimals:2,onlyMethod:'shuttle'},
    {key:'switch_every',label:'Flip the switch every',unit:'frames',decimals:0,onlyMethod:'shuttle'}],
  views:[{id:'frames',label:'Picture',kind:'frames'},
         {id:'galaxy3d',label:'Rotate in 3D',kind:'galaxy3d',needs:'galaxy3d_view'}],
  // Fold readouts first, shuttle readouts after: a run only ever has one set.
  numbers:['radius of gyration','buried oil','salt bridges','sticky station','ring position','trips along the axle']};
// Any demo without its own entry above still works on the stand: its first
// three parameters become the controls and its tagline the caption.
function kioskFor(id){
  if(KIOSK[id])return KIOSK[id];
  const d=specs?.demos?.[id];if(!d)return null;
  KIOSK[id]={about:d.tagline,tag:'Simulation',blurb:d.tagline,read:'',story:[],
    controls:Object.keys(d.params||{}).slice(0,3).map(key=>({key})),numbers:[]};
  return KIOSK[id];
}
// Which demos the stand shows, in order: the active demo day's lineup
// (config/lineups.json), chosen with the Demo day buttons on the picker.
let ORDER=[];
function refreshOrder(){ORDER=HPC.items(Object.keys(specs.demos)).filter(id=>specs.demos[id]?kioskFor(id):HPC.extra(id));}

const FRAME_MS=140;
const IDLE_MS=180000;

// ----------------------------------------------------------------- state --
let specs=null,library=[],current=null,runId=null,meta={};
let pollTimer=null,playTimer=null,idleTimer=null;
let frame=0,total=0,playing=false,lastKnownFrame=-1;
let arena=null,fusion=null,galaxy=null,view=null,arenaGeneration=-1,failedViews=new Set();
let gridViews=[],champion={mode:'final',seed:0,data:null,results:[],env:{}};
// Full meta.json of a saved run: the library listing leaves out bulky fields
// (generation statistics, shot history) that only the instrument strip needs.
const fullMetaCache=new Map();
function fullMeta(id){
  if(!id)return Promise.resolve({});
  if(!fullMetaCache.has(id))fullMetaCache.set(id,fetch(`/api/run/${encodeURIComponent(id)}`).then(r=>r.json()).catch(()=>({})));
  return fullMetaCache.get(id);
}
let builder=null,builders=null,controlState={};
let libraryData={favourites:[],showcase:{}};
// A presentation is an ordered playlist of saved runs. "showcase" is one
// demo's picks; "slideshow" is every demo's picks back to back.
let present=null;
let obstacles=null,target=null,chain=null;
let historyFavsOnly=false,historyLimit=12,compareSel=[],starfield=null;
let compareTimer=null,compareProgress=0,comparePlaying=false,compareRuns=[];
// Display speed only: how quickly saved frames are shown.
const SPEEDS=[0.1,0.25,0.5,0.75,1,1.5,2,3,4];
let loopPlayback=true;
let numbers={};

const prefs=(()=>{
  try{return JSON.parse(localStorage.getItem('leonardo.demo')||'{}');}catch(_){return {};}
})();
function savePrefs(){try{localStorage.setItem('leonardo.demo',JSON.stringify(prefs));}catch(_){}}

// ------------------------------------------------------------------ init --
async function init(){
  specs=await (await fetch('/api/specs')).json();
  await HPC.load();refreshOrder();
  HPC.machineSwitch($('#machineSwitch'));
  HPC.onChange(()=>{refreshOrder();if(!current)renderPicker();});
  $('#hpcSettings').onclick=()=>HPC.openSettings(specs.demos);
  setupRunOn();
  $('#profile').value=prefs.profile||'local';
  $('#backend').value=prefs.backend||'auto';
  $('#frames').value=prefs.frames||70;
  $('#idleAction').value=prefs.idleAction||(prefs.idleReturn?'picker':'none');
  $('#slideDwell').value=String(prefs.slideDwell||20);
  setSpeedIndex(Number.isInteger(prefs.speedIndex)?prefs.speedIndex:4,false);
  setLoop(prefs.loop!==false,false);
  describeBackends();
  await loadLibrary();
  renderPicker();
  bindChrome();
  const wanted=new URLSearchParams(location.search).get('demo');
  if(wanted&&specs.demos[wanted]&&kioskFor(wanted))openDemo(wanted);
}
async function loadLibrary(){
  // Showcase picks can be old runs, so read the whole library rather than
  // only the newest handful.
  const [runs,lib]=await Promise.all([
    fetch('/api/runs?limit=500').then(r=>r.json()).catch(()=>[]),
    fetch('/api/library').then(r=>r.json()).catch(()=>({favourites:[],showcase:{}}))]);
  library=Array.isArray(runs)?runs:[];libraryData=lib||{favourites:[],showcase:{}};
}
function latestRun(id){return library.find(r=>r.demo===id&&r.status==='complete')||library.find(r=>r.demo===id)||null;}
function runById(id){return library.find(r=>r.id===id)||null;}
function demoRuns(id){return library.filter(r=>r.demo===id);}
// What plays when nobody is at the stand: the presenter's picks, else the
// favourites, else the newest finished run - so there is always something.
function showcaseRuns(id){
  const picked=(libraryData.showcase?.[id]||[]).map(runById).filter(Boolean);
  if(picked.length)return picked;
  const favs=demoRuns(id).filter(r=>r.favourite&&r.status==='complete').slice(0,3);
  if(favs.length)return favs;
  const latest=latestRun(id);return latest?[latest]:[];
}
function showcaseSource(id){
  if((libraryData.showcase?.[id]||[]).some(runById))return 'picked';
  return demoRuns(id).some(r=>r.favourite&&r.status==='complete')?'favourites':'latest';
}

// ------------------------------------------------------------------ speed --
function speed(){return SPEEDS[Number($('#speed').value)]??1;}
function setSpeedIndex(index,persist=true){
  index=Math.max(0,Math.min(SPEEDS.length-1,Math.round(index)));
  $('#speed').value=String(index);
  const value=SPEEDS[index];
  $('#speedValue').textContent=`${value}×`;
  $('#speedDown').disabled=index===0;$('#speedUp').disabled=index===SPEEDS.length-1;
  if(arena)arena.rate=value;
  gridViews.forEach(v=>{v.rate=value;});
  $('#fsSpeed').value=String(index);$('#fsSpeedValue').textContent=`${value}×`;
  if(playing)startPlay(frame);
  if(comparePlaying)startCompare();
  if(persist){prefs.speedIndex=index;savePrefs();}
}

function describeBackends(){
  const b=specs.backends||{},gpu=b.gpu||{};
  $('#pickBackend').textContent=gpu.available?`GPU READY · ${gpu.detail}`:'CPU ONLY · NO CUDA DEVICE';
  $('#backendHelp').textContent=gpu.available?`GPU: ${gpu.detail}`:'No CUDA device detected on this machine.';
}

// ---------------------------------------------------------------- picker --
function renderPicker(){
  const host=$('#pickGrid');host.innerHTML='';
  ORDER.forEach(id=>{
    const x=HPC.extra(id);
    if(x){
      // A recorded-video item: the tile opens the video player (or its link),
      // which links back here.
      const card=document.createElement('a');card.className='pickCard';
      card.href=x.url||HPC.videoHref(id,{fromDemo:true});if(x.url){card.target='_blank';card.rel='noopener';}
      card.innerHTML=`<span class="tag">Video</span>
        <img src="/static/previews/${encodeURIComponent(id)}.webp" alt="" loading="lazy" width="800" height="450">
        <span class="cardText"><h3>${escapeHtml(x.name)}</h3><p>${escapeHtml(x.tagline)}</p></span>`;
      card.querySelector('img').onerror=e=>{e.target.onerror=null;e.target.src='/static/previews/videos.webp';};
      host.appendChild(card);return;
    }
    const k=kioskFor(id),d=specs.demos[id];
    const card=document.createElement('button');card.className='pickCard';card.type='button';
    card.innerHTML=`<span class="tag${k.ai?' ai':''}">${escapeHtml(k.tag)}</span>
      <img src="/static/previews/${encodeURIComponent(id)}.webp" alt="" loading="lazy" width="800" height="450">
      <span class="cardText"><h3>${escapeHtml(d.name)}</h3><p>${escapeHtml(k.blurb)}</p></span>`;
    card.onclick=()=>openDemo(id);
    host.appendChild(card);
  });
  // Every video in videos/, for when the lineup has no video item of its own.
  if(ORDER.some(id=>HPC.extra(id)))return;
  const card=document.createElement('a');card.className='pickCard';card.href='/videos?from=demo';
  card.innerHTML=`<span class="tag">Videos</span>
    <img src="/static/previews/videos.webp" alt="" loading="lazy" width="800" height="450">
    <span class="cardText"><h3>Recorded videos</h3><p>Pre-recorded simulations, played from this computer.</p></span>`;
  host.appendChild(card);
}
// "Run on": this computer or a cluster from config/clusters.json.
function setupRunOn(){
  const select=$('#runOn');
  HPC.clusterList().forEach(c=>{const o=document.createElement('option');o.value=c.name;o.textContent=c.label;select.appendChild(o);});
  if(prefs.runOn&&[...select.options].some(o=>o.value===prefs.runOn))select.value=prefs.runOn;
  select.onchange=()=>{prefs.runOn=select.value;savePrefs();
    const c=HPC.clusterList().find(x=>x.name===select.value);
    if(c?.default_profile&&$('#profile').value!==c.default_profile){$('#profile').value=c.default_profile;$('#profile').dispatchEvent(new Event('change'));}};
}

// ----------------------------------------------------------- demo screen --
// `run` opens straight onto a particular saved run (presentations, history);
// otherwise the demo opens on the first showcase run.
function openDemo(id,{run=null}={}){
  const d=specs.demos[id];if(!d)return;
  const changed=id!==current;
  current=id;resetRun();
  history.replaceState(null,'',`/demo?demo=${encodeURIComponent(id)}`);
  const k=kioskFor(id);
  $('#pick').classList.add('hidden');$('#stage').classList.remove('hidden');
  $('#title').textContent=d.name;$('#subtitle').textContent=k.blurb;
  $('#preview').src=`/static/previews/${encodeURIComponent(id)}.webp`;
  $('#captionRead').textContent=k.read||'';
  $('#captionStory').textContent=k.about||k.blurb;
  // The narrative is the presenter's to tell: it lives in the drawer and never
  // advances by itself as frames play.
  $('#talkingList').innerHTML=(k.story||[]).map(line=>`<li>${escapeHtml(line)}</li>`).join('');
  $('#talkingPoints').classList.toggle('hidden',!(k.story||[]).length);
  $('#fsTitle').textContent=d.name;
  $('#settingsFor').textContent=d.name;
  if(changed||!$('#controls').children.length){
    applyMethods();
    renderControls();
    mountBuilders();
    renderAdvanced();
    applyParallelField();
    compareSel=[];historyLimit=12;
  }
  starfieldFor(id);
  const first=run||showcaseRuns(id)[0]||null;
  $('#replay').classList.toggle('hidden',!showcaseRuns(id).length);
  setAction('Ready when you are',first
    ? `Change anything above, then run a brand-new simulation ${$('#runOn').value==='local'?'on this machine':'on '+($('#runOn').selectedOptions[0]?.textContent||'the cluster')}.`
    : 'Nothing saved yet for this experiment — press Run it.');
  renderViewSwitch();
  renderHistory();
  if(first)openRun(first);
  if(!present)window.scrollTo(0,0);
  resetIdle();
}

function backToPicker(){
  stopPresenting({quiet:true});
  resetRun();current=null;starfieldFor(null);
  history.replaceState(null,'','/demo');
  $('#stage').classList.add('hidden');$('#pick').classList.remove('hidden');
  loadLibrary().then(renderPicker);
  window.scrollTo(0,0);
  resetIdle();
}

// ---------------------------------------------------------- presentation --
function startShowcase(){
  if(!current)return;
  const items=showcaseRuns(current).map(run=>({demo:current,run}));
  if(!items.length){setAction('No showcase yet','Run this experiment once, or pick showcase runs in the settings.');return;}
  beginPresentation('showcase',items,0);
}
function startSlideshow(){
  const items=[];
  ORDER.filter(id=>specs.demos[id]).forEach(id=>showcaseRuns(id).forEach(run=>items.push({demo:id,run})));
  if(!items.length)return;
  beginPresentation('slideshow',items,0);
}
function beginPresentation(kind,items,index){
  closeSettings();closeCompare();
  present={kind,items,index:-1,shownAt:0,loops:0};
  document.body.classList.add('presenting');
  $('#presentBar').classList.remove('hidden');
  presentGo(index);
}
function presentGo(index){
  if(!present)return;
  const n=present.items.length;
  present.index=((index%n)+n)%n;present.shownAt=performance.now();present.loops=0;
  const item=present.items[present.index];
  const d=specs.demos[item.demo];
  if(item.demo!==current)openDemo(item.demo,{run:item.run});else openRun(item.run);
  const slideshow=present.kind==='slideshow';
  $('#presentTitle').textContent=slideshow?`Slideshow · ${d.name}`:`Showcase · ${d.name}`;
  $('#presentMeta').textContent=`${present.index+1} of ${n} · ${runLabel(item.run)}`;
  $('#presentPrev').disabled=$('#presentNext').disabled=n<2;
  $('#presentTry').textContent=slideshow?'Try this one':'Try it yourself';
  setStatus(slideshow?'SLIDESHOW':'SHOWCASE','replay');
  window.scrollTo(0,0);
}
// Called when playback wraps. Returns true if it moved on to another run.
function presentLoopDone(){
  if(!present)return false;
  present.loops++;
  const dwell=(present.kind==='slideshow'?Number($('#slideDwell').value)||20:0)*1000;
  if(present.items.length<2||performance.now()-present.shownAt<dwell)return false;
  presentGo(present.index+1);return true;
}
function stopPresenting({quiet=false}={}){
  if(!present)return;
  present=null;
  document.body.classList.remove('presenting');
  $('#presentBar').classList.add('hidden');
  if(!quiet&&runId)setStatus('SAVED RUN','replay');
}
// A visitor walked up: keep the demo on screen, hand them the controls.
function tryItYourself(){
  stopPresenting();
  const deck=$('.deck');if(deck)deck.scrollIntoView({behavior:'smooth',block:'start'});
  resetIdle();
}

// ------------------------------------------------------ visitor controls --
function spec(){return specs.demos[current];}
function paramSpec(key){return spec()?.params?.[key];}
function controlDefs(){return (KIOSK[current]?.controls)||[];}
function methodValue(){return $('#method').value||'default';}

function renderControls(){
  const host=$('#controls');host.innerHTML='';controlState={};
  if(isGame())addTrainingHeader(host);
  controlDefs().forEach(def=>{
    if(def.onlyMethod&&def.onlyMethod!==methodValue())return;
    if(def.method)return addMethodBox(host,def);
    if(def.population)return addPopulationBox(host,def);
    if(def.ghosts)return addGhostBox(host,def);
    if(def.name)return addNameBox(host,def);
    if(def.setting)return addSettingBox(host,def);
    const p=paramSpec(def.key);if(!p)return;
    controlState[def.key]=Number(p.value);
    if(def.choices||p.kind==='choice')addChoiceBox(host,def,p);
    else if(p.kind==='toggle')addToggleBox(host,def,p);
    else addValueBox(host,def,p);
  });
  renderMoreParams();
}
// Every parameter without a big visitor control still exists: it lives in the
// presenter drawer instead of disappearing from demo mode.
function headlineKeys(){return new Set(controlDefs().map(d=>d.key).filter(Boolean));}
function renderMoreParams(){
  const host=$('#moreParamsFields');if(!host)return;host.innerHTML='';
  const params=spec()?.params||{},headline=headlineKeys();
  const extra=Object.entries(params).filter(([key,p])=>!headline.has(key)
    &&(!Array.isArray(p.methods)||p.methods.includes(methodValue())));
  $('#moreParams').classList.toggle('hidden',!extra.length);
  $('#moreParamsSummary').textContent=extra.length?`${extra.length} more`:'';
  extra.forEach(([key,p])=>{
    controlState[key]=Number(p.value);
    const field=document.createElement('label');
    const label=p.label||key.replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());
    if(p.kind==='toggle'){
      field.className='switchField';
      field.innerHTML=`<input type="checkbox" ${Number(p.value)?'checked':''}> ${escapeHtml(label)}`;
      field.querySelector('input').onchange=e=>{controlState[key]=e.target.checked?1:0;};
    }else if(p.kind==='choice'){
      field.innerHTML=`<span>${escapeHtml(label)}</span><select>${Object.entries(p.options||{}).map(([v,text])=>`<option value="${escapeHtml(v)}" ${Number(v)===Number(p.value)?'selected':''}>${escapeHtml(text)}</option>`).join('')}</select>`;
      field.querySelector('select').onchange=e=>{controlState[key]=Number(e.target.value);};
    }else{
      field.innerHTML=`<span>${escapeHtml(label)} <em>${escapeHtml(p.value)}</em></span><input type="range" min="${p.min}" max="${p.max}" step="${p.step}" value="${p.value}">`;
      const input=field.querySelector('input'),out=field.querySelector('em');
      input.oninput=()=>{controlState[key]=Number(input.value);out.textContent=input.value;};
    }
    host.appendChild(field);
  });
}
function box(host,label,wide){
  const el=document.createElement('div');el.className='box'+(wide?' wide':'');
  el.innerHTML=`<label>${escapeHtml(label)}</label>`;host.appendChild(el);return el;
}
// A slider becomes a value box: the number is large, the arrows are large, and
// the track underneath is still there for anyone who wants to sweep it.
function addValueBox(host,def,p){
  const el=box(host,def.label||p.label||def.key.replaceAll('_',' '));
  const dp=def.decimals??decimalsFor(p.step);
  const row=document.createElement('div');row.className='boxRow';
  row.innerHTML=`<button class="stepBtn" type="button" data-step="-1" aria-label="Decrease">−</button>
    <div class="boxValue"><span></span>${def.unit?`<small>${escapeHtml(def.unit)}</small>`:''}</div>
    <button class="stepBtn" type="button" data-step="1" aria-label="Increase">+</button>`;
  const slider=document.createElement('input');
  Object.assign(slider,{type:'range',min:p.min,max:p.max,step:p.step,value:p.value});
  el.append(row,slider);
  const out=row.querySelector('.boxValue span');
  const show=()=>{controlState[def.key]=Number(slider.value);out.textContent=Number(slider.value).toFixed(dp);
    row.querySelector('[data-step="-1"]').disabled=Number(slider.value)<=Number(p.min);
    row.querySelector('[data-step="1"]').disabled=Number(slider.value)>=Number(p.max);};
  slider.oninput=show;
  row.querySelectorAll('[data-step]').forEach(b=>b.onclick=()=>{
    const step=Number(p.step)*Number(b.dataset.step)*(def.coarse||1);
    slider.value=Math.min(Number(p.max),Math.max(Number(p.min),Number(slider.value)+step));show();});
  if(def.help){const hint=document.createElement('small');hint.className='boxHint';hint.textContent=def.help;el.appendChild(hint);}
  show();
}
function decimalsFor(step){const s=String(step??1);return s.includes('.')?Math.min(3,s.split('.')[1].length):0;}
// Big named buttons for anything with a small set of meaningful settings,
// including sliders whose values are really presets (the painting target).
function addChoiceBox(host,def,p){
  const names=def.choices||Object.values(p.options||{});
  const values=def.choices?names.map((_,i)=>Number(p.min)+i*Number(p.step||1)):Object.keys(p.options||{}).map(Number);
  const el=box(host,def.label||p.label||def.key.replaceAll('_',' '),names.length>2);el.dataset.key=def.key;
  const seg=document.createElement('div');seg.className='seg';el.appendChild(seg);
  const select=(v,fromUser=false)=>{controlState[def.key]=v;
    [...seg.children].forEach((b,i)=>{const on=values[i]===v;b.classList.toggle('on',on);b.setAttribute('aria-pressed',String(on));});
    if(def.key==='obstacle'&&obstacles)obstacles.setPreset(v);
    if(def.key==='target'&&fromUser&&target)target.usePreset();
    if(def.key==='track')refreshGhosts();
    if(def.key==='sequence'&&chain){if(fromUser)chain.reset();chain.setPreset(v,chainLength());}
  };
  names.forEach((name,i)=>{const b=document.createElement('button');b.type='button';b.textContent=name;
    b.onclick=()=>select(values[i],true);seg.appendChild(b);});
  select(Number(p.value));
}
function addToggleBox(host,def,p){
  const el=box(host,def.label||p.label||def.key.replaceAll('_',' '));el.classList.add('toggle');
  const seg=document.createElement('div');seg.className='seg';el.appendChild(seg);
  const select=v=>{controlState[def.key]=v;
    [...seg.children].forEach((b,i)=>{const on=i===v;b.classList.toggle('on',on);b.setAttribute('aria-pressed',String(on));});};
  ['Off','On'].forEach((name,i)=>{const b=document.createElement('button');b.type='button';b.textContent=name;
    b.onclick=()=>select(i);seg.appendChild(b);});
  select(Number(p.value)?1:0);
}
// Mode 1 / mode 2 on the demos that have two genuinely different questions.
function addMethodBox(host,def){
  const capability=specs.capabilities?.[current]||{},methods=capability.methods||[];
  if(methods.length<2)return;
  const el=box(host,def.label||'Mode',true);
  const seg=document.createElement('div');seg.className='seg';el.appendChild(seg);
  methods.forEach(m=>{const b=document.createElement('button');b.type='button';
    b.textContent=capability.method_labels?.[m]||m.replaceAll('_',' ');
    b.onclick=()=>{$('#method').value=m;renderControls();renderAdvanced();syncChain();showRunForMethod(m);};
    b.classList.toggle('on',m===methodValue());b.setAttribute('aria-pressed',String(m===methodValue()));
    seg.appendChild(b);});
}
// The population control replaces the old "show scale reveal" button, and only
// exists where the population is the science rather than a screen-filling trick.
function addPopulationBox(host,def){
  const el=box(host,def.label||'Independent runs',true);
  const seg=document.createElement('div');seg.className='seg';el.appendChild(seg);
  // Default to what the quality preset already asks for - the wall of networks
  // and the grid of searches are the point of these demos, not a trimmed-down
  // version of them - then remember whatever the stand last chose.
  const preset=specs?.profiles?.[$('#profile').value]?.[current]||{};
  const wanted=Number(preset.networks??preset.ensemble);
  const nearest=def.values.filter(v=>v<=wanted).pop();
  const stored=Number(prefs[`pop.${current}`]);
  const start=def.values.includes(stored)?stored
    :def.values.includes(wanted)?wanted
    :nearest??def.values[Math.min(1,def.values.length-1)];
  const select=v=>{controlState._population=v;prefs[`pop.${current}`]=v;savePrefs();
    [...seg.children].forEach((b,i)=>{const on=def.values[i]===v;b.classList.toggle('on',on);b.setAttribute('aria-pressed',String(on));});
    $('#parallelCount').value=String(v);};
  def.values.forEach(v=>{const b=document.createElement('button');b.type='button';
    b.textContent=v===1?'1':`${v}`;b.onclick=()=>select(v);seg.appendChild(b);});
  if(def.help){const hint=document.createElement('small');hint.style.color='var(--muted)';
    hint.style.fontSize='11.5px';hint.textContent=def.help;el.appendChild(hint);}
  select(start);
}

// A compute-profile value that is really a visitor's choice (how small to
// squeeze the picture). It overrides the preset for this run only.
function addSettingBox(host,def){
  const el=box(host,def.label,true);
  const seg=document.createElement('div');seg.className='seg';el.appendChild(seg);
  const key=`set.${current}.${def.setting}`,stored=Number(prefs[key]);
  const start=def.values.includes(stored)?stored:def.default??def.values[0];
  const select=v=>{controlState._settings={...(controlState._settings||{}),[def.setting]:v};prefs[key]=v;savePrefs();
    [...seg.children].forEach((b,i)=>{const on=def.values[i]===v;b.classList.toggle('on',on);b.setAttribute('aria-pressed',String(on));});};
  def.values.forEach(v=>{const b=document.createElement('button');b.type='button';b.textContent=`${v}${def.unit?` ${def.unit}`:''}`;
    b.onclick=()=>select(v);seg.appendChild(b);});
  if(def.help){const hint=document.createElement('small');hint.className='boxHint';hint.textContent=def.help;el.appendChild(hint);}
  select(start);
}

// The AI games keep two sets of controls apart: these (training, used by
// "Run it") and the test world under the picture (inference on a finished
// champion). Every visitor trains for the same number of generations, so
// brains compete on design alone.
function trainingGenerations(){
  return Number(controlState._settings?.generations??specs?.profiles?.[$('#profile').value]?.[current]?.generations)||null;
}
function addTrainingHeader(host){
  const el=document.createElement('div');el.className='trainingHead';
  const gens=trainingGenerations();
  el.innerHTML=`<span class="phaseTag train">Training</span><b>How your brain will be trained</b>
    <small>Used when you press <em>Run it</em>.${gens?` Everyone gets the same ${gens} generations: a bigger brain has more weights to tune in that time, and a brain that fits its training world too closely shows it when you test it in a new one.`:''}</small>`;
  host.appendChild(el);
}
function addNameBox(host,def){
  const el=box(host,def.label||'Your name');
  const input=document.createElement('input');
  Object.assign(input,{type:'text',maxLength:24,placeholder:'Name tag',autocomplete:'off',spellcheck:false,className:'nameInput'});
  input.oninput=()=>{controlState._name=input.value.trim();};
  el.appendChild(input);
  if(def.help){const hint=document.createElement('small');hint.className='boxHint';hint.textContent=def.help;el.appendChild(hint);}
}

// Ghost races: the fastest saved champions on the chosen track race beside
// the visitor's cars, re-simulated with their own brains.
function ghostRuns(track){
  return library.filter(r=>r.demo==='neuro_racers'&&r.status==='complete'&&r.summary&&Number(r.summary.track_id)===Number(track))
    .sort((a,b)=>(a.summary.best_lap_s??1e9)-(b.summary.best_lap_s??1e9)||(b.summary.best_laps-a.summary.best_laps)).slice(0,3);
}
function addGhostBox(host,def){
  const el=box(host,def.label||'Race the ghosts');el.classList.add('toggle');el.id='ghostBox';
  const seg=document.createElement('div');seg.className='seg';el.appendChild(seg);
  const hint=document.createElement('small');hint.className='boxHint';el.appendChild(hint);
  const select=v=>{controlState._ghosts=v;
    [...seg.children].forEach((b,i)=>{const on=i===v;b.classList.toggle('on',on);b.setAttribute('aria-pressed',String(on));});};
  ['Off','On'].forEach((name,i)=>{const b=document.createElement('button');b.type='button';b.textContent=name;
    b.onclick=()=>select(i);seg.appendChild(b);});
  select(0);refreshGhosts();
}
function refreshGhosts(){
  const el=$('#ghostBox');if(!el)return;
  const best=ghostRuns(controlState.track??0);
  const buttons=el.querySelectorAll('.seg button');
  buttons[1].disabled=!best.length;
  if(!best.length&&controlState._ghosts){controlState._ghosts=0;buttons.forEach((b,i)=>b.classList.toggle('on',i===0));}
  el.querySelector('.boxHint').textContent=best.length
    ?`The ${best.length} fastest saved car${best.length===1?'':'s'} on this track race beside yours.`
    :'No saved champions on this track yet: be the first ghost.';
}

// -------------------------------------------------------------- creators --
// What a visitor makes with their own hands: a wind-tunnel body, a picture for
// the network to learn, a brain. These are never hidden in demo mode.
// The sequence builder belongs to the fold mode only; its length follows the
// preset's chain length (or the value under Advanced) until the visitor edits.
function chainLength(){return Number($('#s_particles')?.value)||Number(specs?.profiles?.[$('#profile').value]?.molecular_dynamics?.particles)||40;}
function syncChain(){
  if(!chain)return;
  const fold=methodValue()!=='shuttle';
  chain.setVisible(fold);document.body.classList.toggle('hasBuilder',fold);
  chain.setPreset(controlState.sequence??0,chainLength());
}
function mountBuilders(){
  const host=$('#brainHost');host.innerHTML='';builder=null;builders=null;obstacles=null;chain=null;
  document.body.classList.remove('hasBrain');
  if(target)target.stop();target=null;
  const creator=KIOSK[current]?.creator;
  if(creator==='obstacles'){
    obstacles=new ObstacleBuilder(host,{preset:controlState.obstacle??0,large:true});
    document.body.classList.add('hasBuilder');return;
  }
  if(creator==='target'){target=new TargetPicture(host);document.body.classList.add('hasBuilder');return;}
  if(creator==='chain'){
    chain=new ChainBuilder(host,{preset:controlState.sequence??0,length:chainLength(),large:true});
    document.body.classList.add('hasBuilder');syncChain();return;
  }
  const catalogue=spec()?.brain;
  document.body.classList.toggle('hasBuilder',Boolean(catalogue));
  document.body.classList.toggle('hasBrain',Boolean(catalogue));
  if(!catalogue)return;
  if(catalogue.roles){
    const duo=document.createElement('div');duo.className='duoBuilders';host.appendChild(duo);
    builders={
      bat:new BrainBuilder(duo,catalogue.roles.bat,{title:'Visitor 1 · the bat',
        subtitle:`Up to ${catalogue.roles.bat.budget} points. Ears hear echoes of your calls; the bat is otherwise blind.`,
        accent:'#ffbe50',agent:'bat'}),
      moth:new BrainBuilder(duo,catalogue.roles.moth,{title:'Visitor 2 · the moth',
        subtitle:`Up to ${catalogue.roles.moth.budget} points. Hear the bat coming, dodge, or click to jam its sonar.`,
        accent:'#6ef0c8',agent:'moth'})};
    return;
  }
  builder=new BrainBuilder(host,catalogue,{title:'Build your car’s brain',
    subtitle:`Spend up to ${catalogue.budget} points on sensors, neurons and controls. Every car in the race gets exactly this brain.`,
    accent:'#ffc45c',agent:'car'});
}

// ----------------------------------------------------------- view switch --
// A view is offered only when this run has written the data behind it AND that
// data actually loaded. A streaming run publishes its arena manifest long
// before the generation file the viewer wants, so "the manifest exists" is not
// enough: without the second test the viewer would sit on a canvas that can
// never draw, showing nothing at all while the run computes.
function availableViews(){
  return (KIOSK[current]?.views||[]).filter(v=>{
    if(failedViews.has(v.id))return false;
    if(!v.needs)return true;
    if(v.needs==='has_reveal'||v.needs==='reveal')return Boolean(meta.has_reveal||meta.reveal);
    if(v.needs==='arena_lanes')return Number(meta.arena_view?.lanes)>1;
    return Boolean(meta[v.needs]);
  });
}
// Fall back to the next usable view - ultimately the plain rendered frames.
function markViewUnusable(id){
  failedViews.add(id);renderViewSwitch();
  const next=viewDef();
  if(next&&next.kind!=='frames')setView(next.id);else{renderInstruments();showFrame(frame);}
}
function viewSignature(){return availableViews().map(v=>v.id).join(',');}
function renderViewSwitch(){
  const host=$('#viewSwitch'),list=availableViews();
  host.innerHTML='';host.classList.toggle('hidden',list.length<2);
  // A view whose data this run has not written yet (no arena, no reveal) must
  // not stay selected, or the surface renders nothing at all.
  if(!list.some(v=>v.id===view))view=list.length?list[0].id:null;
  const fsHost=$('#fsViews');fsHost.innerHTML='';fsHost.classList.toggle('hidden',list.length<2);
  if(list.length<2)return;
  for(const target of [host,fsHost])list.forEach(v=>{const b=document.createElement('button');b.type='button';b.textContent=v.label;
    b.classList.toggle('on',v.id===view);b.setAttribute('aria-pressed',String(v.id===view));
    b.onclick=()=>setView(v.id);target.appendChild(b);});
}
// Only ever a view this run can actually show; null means the plain frames.
function viewDef(){return availableViews().find(v=>v.id===view)||null;}
// Switching view never changes what is playing, only how it is shown; it keeps
// the play/pause state the presenter chose.
async function setView(id,{autoplay=null}={}){
  const resume=autoplay??playing;
  stopPlay();
  view=id;renderViewSwitch();
  exitArena();exitFusion();exitGalaxy();exitGrid();
  const def=viewDef();
  renderInstruments();
  if(def?.kind==='arena')await enterArena(def);
  else if(def?.kind==='champion')await enterChampion(def);
  else if(def?.kind==='grid')await enterGrid();
  else if(def?.kind==='fusion')await enterFusion();
  else if(def?.kind==='galaxy3d')await enterGalaxy();
  else if(def?.kind==='reveal')showReveal();
  else showFrame(frame);
  if(id!==view)return;          // superseded by another switch meanwhile
  if(resume)startPlay(frame);
  updateOnScreen();
  const k=KIOSK[current]||{};
  $('#captionRead').textContent=def?.kind==='champion'?(k.readChampion||k.read||'')
    :def?.kind==='grid'?(k.readGrid||k.read||''):def?.lanes?(k.readLanes||k.read||''):(k.readByMethod?.[meta.method||methodValue()]||k.read||'');
  // The frame timeline belongs to the training run, not to a replay or grid.
  const timeline=!(def?.kind==='champion'||def?.kind==='grid');
  $('#seek').disabled=!timeline||!runId;$('#fsSeek').disabled=$('#seek').disabled;
  if(!timeline){$('#captionNumbers').innerHTML='';$('#frameLabel').textContent=def.kind==='grid'?'all boxes':'replay';}
  else if(runId)$('#frameLabel').textContent=`${frame+1} / ${total}`;
}

// --------------------------------------------------------------- surface --
function surfaceLive(on){
  $('#surface').classList.toggle('live',on);
  if(on){$('#idleTitle').textContent='Nothing has been run yet.';$('#idleText').innerHTML='Change something below, then press <em>Run it</em>.';}
}
function frameFolder(){const def=viewDef();return def&&def.kind==='frames'&&def.folder?def.folder:'frames';}
function showFrame(index,count=total){
  if(!runId||count<1)return;
  index=Math.max(0,Math.min(count-1,Math.trunc(index)));frame=index;
  const def=viewDef();
  if(def&&def.kind==='reveal')return;
  if(!def||def.kind==='frames'){
    const image=$('#screen'),fallback=`/runs/${runId}/frames/frame_${pad(index)}.jpg`;
    image.onerror=()=>{image.onerror=null;image.src=fallback;};
    image.src=`/runs/${runId}/${frameFolder()}/frame_${pad(index)}.jpg`;
  }
  surfaceLive(true);
  if(galaxy&&view==='galaxy3d'&&meta.galaxy3d_view)
    galaxy.load(`/runs/${runId}/${meta.galaxy3d_view.folder}/frame_${pad(index)}.json`).catch(()=>{});
  if(fusion&&view==='fusion'){const u=fusionFrameUrl(index);if(u)fusion.load(u,true).catch(()=>{});}
  // The arena animates one generation at a time; the frame cursor is what says
  // which generation the time-lapse has reached.
  if(arena&&def?.kind==='arena'&&meta.arena_view){
    const generation=generationAt(index);
    if(generation!==arenaGeneration){arenaGeneration=generation;
      arena.showGeneration(`/runs/${runId}/${meta.arena_view.folder}/gen_${pad(generation)}.json`).catch(()=>{});
      updateTrainingInstruments();}
  }
  if(def?.kind==='fusion'||(!def&&current==='fusion_plasma')||def?.kind==='frames')updateFusionInstruments(index);
  if(current==='neural_wall')updateCompressionInstruments(index);
  const progress=(index+1)/count;
  $('#progressBar').style.width=`${progress*100}%`;
  $('#seek').value=index;
  $('#frameLabel').textContent=`${index+1} / ${count}`;
  syncFullscreen();
  loadNumbers(index);
}
function showReveal(){
  stopPlay();
  if(!runId)return;
  $('#screen').onerror=null;
  $('#screen').src=`/runs/${runId}/reveal.jpg?t=${Date.now()}`;
  surfaceLive(true);
  $('#captionStory').textContent=current==='neural_wall'
    ? 'Every tile is a different network trained on the same picture: learning rate varies left to right, width top to bottom.'
    : 'Every box is a completely separate search, started from its own random population. None of them shared anything.';
}
function setAction(label,hint){$('#actionLabel').textContent=label;$('#runHint').textContent=hint||'';}
function setBusy(on,text){$('#busy').classList.toggle('hidden',!on);if(text)$('#busyText').textContent=text;}
function setStatus(text,state){const el=$('#status');el.textContent=text;if(state)el.dataset.state=state;else delete el.dataset.state;}

// Live readouts: from the poll while computing, from the run's per-frame JSON
// on replay.  Three at most - this is a caption, not an instrument panel.
async function loadNumbers(index){
  if(!runId)return;
  try{
    const r=await fetch(`/runs/${runId}/frame_data/frame_${pad(index)}.json`);
    if(!r.ok)return;
    const body=await r.json();
    if(index===frame)renderNumbers(body.values||{});
  }catch(_){}
}
function renderNumbers(values){
  numbers=values||{};
  const host=$('#captionNumbers'),wanted=KIOSK[current]?.numbers||[];
  const kind=viewDef()?.kind;
  if(kind==='champion'||kind==='grid'){host.innerHTML='';syncFullscreen();return;}
  const entries=[];
  wanted.forEach(key=>{const hit=Object.keys(numbers).find(k=>k.toLowerCase()===key.toLowerCase());
    if(hit)entries.push([hit,numbers[hit]]);});
  Object.entries(numbers).forEach(([k,v])=>{if(entries.length<3&&!entries.some(e=>e[0]===k))entries.push([k,v]);});
  host.innerHTML='';
  entries.slice(0,3).forEach(([k,v])=>{const cell=document.createElement('div');
    cell.innerHTML=`<span>${escapeHtml(k)}</span><b>${escapeHtml(v)}</b>`;host.appendChild(cell);});
  if(current==='neural_wall')updateCompressionInstruments(frame);
  syncFullscreen();
}

// -------------------------------------------------------------- playback --
function stopPlay(){
  if(playTimer)clearInterval(playTimer);playTimer=null;playing=false;$('#playPause').textContent='▶';
  const kind=canvasKind();
  if(kind==='arena'||kind==='champion')arena?.pause();
  if(kind==='grid'||kind==='champion')gridViews.forEach(v=>v.pause());
  syncFullscreen();
}
function setLoop(on,persist=true){
  loopPlayback=Boolean(on);
  for(const b of [$('#loopToggle'),$('#fsLoop')]){b.classList.toggle('on',loopPlayback);b.setAttribute('aria-pressed',String(loopPlayback));}
  if(persist){prefs.loop=loopPlayback;savePrefs();}
}
// The arena, the champion replay and the grid animate themselves at display
// frame rate. Driving them from the frame timer restarted each drive every
// few hundred milliseconds - the "jumping around". Instead each generation's
// drive plays to its end, and only then does the timeline move on.
function canvasKind(){const k=viewDef()?.kind;return k==='arena'&&!pollTimer?'arena':k==='champion'||k==='grid'?k:null;}
function recordedGenerations(){
  const f=meta.arena_view?.frames;if(!Array.isArray(f))return [];
  return [...new Set(f.map(e=>Array.isArray(e)?Number(e[0]):Number(e)))];
}
function lastFrameOfGeneration(g){
  let at=0;(meta.arena_view?.frames||[]).forEach((e,i)=>{if((Array.isArray(e)?Number(e[0]):Number(e))===g)at=i;});return at;
}
function generationDriveDone(){
  if(!playing||canvasKind()!=='arena')return;
  const gens=recordedGenerations(),at=gens.indexOf(arenaGeneration);
  if(at>=gens.length-1||at<0){
    if(present){if(presentLoopDone())return;}
    else if(!loopPlayback){stopPlay();return;}
    showFrame(lastFrameOfGeneration(gens[0]),total);return;
  }
  showFrame(lastFrameOfGeneration(gens[at+1]),total);
}
function startPlay(from=frame){
  if(!runId||total<1)return;
  const kind=canvasKind();
  if(kind){
    stopPlay();playing=true;$('#playPause').textContent='❚❚';
    if(kind==='arena'&&arena){showFrame(from,total);arena.onloop=generationDriveDone;arena.rate=speed();arena.resume();}
    if(kind==='champion'&&arena&&!gridViews.length){arena.onloop=()=>{if(present)presentLoopDone();};arena.rate=speed();arena.resume();}
    if(kind==='grid'||(kind==='champion'&&gridViews.length)){gridViews.forEach((v,i)=>{v.rate=speed();v.onloop=i?null:()=>{if(present)presentLoopDone();};v.resume();});}
    syncFullscreen();return;
  }
  if(!loopPlayback&&!present&&from>=total-1)from=0;
  stopPlay();showFrame(from,total);playing=true;$('#playPause').textContent='❚❚';syncFullscreen();
  playTimer=setInterval(()=>{
    const next=(frame+1)%total;
    // Only a finished run loops; a streaming one waits for its next frame.
    if(next===0&&present){if(presentLoopDone())return;}
    // A presentation always loops; otherwise honour the loop button.
    else if(next===0&&!loopPlayback&&!pollTimer){stopPlay();return;}
    showFrame(next,total);
  },Math.max(16,FRAME_MS/speed()));
}
function enableTransport(on){$('#playPause').disabled=!on;$('#seek').disabled=!on;}

// ------------------------------------------------------- interactive views --
function fusionFrameUrl(index){
  const m=meta.fusion_view;if(!m)return null;
  return typeof m==='string'?`/runs/${runId}/${m}`:`/runs/${runId}/${m.folder}/frame_${pad(index)}.json`;
}
function exitFusion(){if(fusion)fusion.stop?.();$('#fusionCanvas').classList.add('hidden');$('#screen').classList.remove('hidden');}
async function enterFusion(){
  if(!runId||!meta.fusion_view)return showFrame(frame);
  if(!fusion)fusion=new FusionView($('#fusionCanvas'));
  try{
    await fusion.load(fusionFrameUrl(frame));
    $('#screen').classList.add('hidden');$('#fusionCanvas').classList.remove('hidden');
    fusion.setLayer('plasma',true);fusion.setLayer('magnetic',false);fusion.setLayer('escapes',true);
    fusion.resize();surfaceLive(true);
  }catch(_){markViewUnusable('fusion');}
}
function exitGalaxy(){$('#galaxyCanvas').classList.add('hidden');$('#screen').classList.remove('hidden');}
async function enterGalaxy(){
  if(!runId||!meta.galaxy3d_view)return showFrame(frame);
  // No painted title box over the picture: the caption carries it.
  if(!galaxy){galaxy=new Galaxy3DView($('#galaxyCanvas'));galaxy.showLabels=false;}
  try{
    await galaxy.load(`/runs/${runId}/${meta.galaxy3d_view.folder}/frame_${pad(frame)}.json`);
    galaxy.setFocus('all');
    $('#screen').classList.add('hidden');$('#galaxyCanvas').classList.remove('hidden');
    galaxy.resize();surfaceLive(true);
  }catch(_){markViewUnusable('galaxy3d');}
}
function exitArena(){if(arena){arena.stop();arena.onloop=null;}$('#arenaCanvas').classList.add('hidden');$('#screen').classList.remove('hidden');}
function makeArena(){
  if(!arena){arena=new ArenaView($('#arenaCanvas'));arena.showBrain=false;}
  arena.attachBrainCanvas($('#brainCanvas'));arena.rate=speed();
  return arena;
}
async function enterArena(def){
  if(!runId||!meta.arena_view)return showFrame(frame);
  makeArena();
  try{
    await arena.loadArena(`/runs/${runId}/${meta.arena_view.arena||meta.arena_view.folder+'/arena.json'}`);
    arena.lit=Boolean(def?.lit);arena.lanes=Boolean(def?.lanes);arena.ownerName=meta.name||null;
    arenaGeneration=generationAt(frame);
    await arena.showGeneration(`/runs/${runId}/${meta.arena_view.folder}/gen_${pad(arenaGeneration)}.json`);
    if(!playing)arena.pause();
    $('#screen').classList.add('hidden');$('#arenaCanvas').classList.remove('hidden');
    arena.resize();surfaceLive(true);showFrame(frame);updateTrainingInstruments();
  }catch(_){markViewUnusable(def?.id||'arena');}
}

// --------------------------------------------------------------- champion ---
// Inference: saved champion networks, frozen, re-driven from a fresh random
// start on the server (the dashboard's generation lab uses the same endpoint).
//
// The test world is separate from the training settings: the same frozen
// network can be dropped onto another track or into another cave, with more or
// fewer moths, or into several worlds at once, one per box. Nothing about the
// network changes, so this is where over-fitting to the training world shows.
function trainedWorld(){
  const p=meta.params||{};
  return current==='neuro_racers'?{track:Number(meta.track?.id??p.track??0)}
    :{cave:Number(meta.cave??p.cave??11),moths:Number(p.moths??4)};
}
function championEnv(){return {...trainedWorld(),boxes:1,...champion.env};}
function envIsTrained(){const env=championEnv(),t=trainedWorld();return env.boxes===1&&Object.keys(t).every(k=>env[k]===t[k]);}
function championWorlds(env){
  if(current==='neuro_racers'){
    const tracks=Object.keys(paramSpec('track')?.options||{0:''}).map(Number);
    return env.boxes>1?tracks.map(track=>({track})):[{track:env.track}];
  }
  const p=paramSpec('cave')||{min:1,max:999},lo=Number(p.min),span=Number(p.max)-lo+1;
  return Array.from({length:env.boxes},(_,i)=>({cave:lo+(env.cave-lo+i)%span,moths:env.moths}));
}
function championGhosts(env){
  if(current!=='neuro_racers'||champion.mode!=='race')return [];
  return ghostRuns(env.track).filter(r=>r.id!==runId).map(r=>r.id);
}
async function enterChampion(def){
  if(!runId||!meta.lab)return markViewUnusable(def?.id||'champion');
  makeArena();
  const top=Math.max(1,Number(meta.lab.generations)||1),env=championEnv();
  const gens=champion.mode==='learn'&&current==='neuro_racers'
    ?[...new Set([1,Math.max(1,Math.round(top/2)),top])]:[top];
  if(!champion.seed)champion.seed=Math.floor(Math.random()*1e6);
  const worlds=championWorlds(env),ghosts=championGhosts(env);
  setBusy(true,worlds.length>1?`Testing the champion in ${worlds.length} ${current==='neuro_racers'?'tracks':'caves'}…`:'Replaying the champion…');
  let results;
  try{
    results=await Promise.all(worlds.map(w=>{
      const q=new URLSearchParams({gens:gens.join(','),seed:champion.seed});
      Object.entries(w).forEach(([k,v])=>q.set(k,v));if(ghosts.length)q.set('ghosts',ghosts.join(','));
      return fetch(`/api/replay/${encodeURIComponent(runId)}?${q}`).then(r=>{if(!r.ok)throw new Error(String(r.status));return r.json();});
    }));
  }catch(_){
    setBusy(false);
    // A test world that cannot be run goes back to the training world rather
    // than losing the view; only a run that cannot replay at all loses it.
    if(!envIsTrained()){champion.env={};$('#onScreen').textContent='That test world could not be run; back to the training world.';$('#onScreen').classList.remove('hidden');return setView(view);}
    return markViewUnusable(def?.id||'champion');
  }
  if(viewDef()?.kind!=='champion')return;      // switched away meanwhile
  champion.data=results[0];champion.results=results;champion.gens=gens;
  const lit=current==='bat_vs_moth'&&Boolean(def?.lit);
  $('#screen').classList.add('hidden');
  if(results.length===1){
    if(results[0].arena){arena.arena=results[0].arena;}
    else await arena.loadArena(`/runs/${runId}/${meta.arena_view?.arena||'interactive/arena.json'}`);
    arena.lit=lit;arena.lanes=false;arena.focus=null;arena.ownerName=meta.name||null;arena.showData(results[0]);
    if(!playing)arena.pause();
    $('#arenaCanvas').classList.remove('hidden');arena.resize();
  }else showChampionBoxes(results,lit);
  surfaceLive(true);setBusy(false);renderInstruments();
}
// Several test worlds at once: the frozen champion in each, one per box.
function showChampionBoxes(results,lit){
  const host=$('#gridView'),trained=trainedWorld();host.innerHTML='';
  host.style.gridTemplateColumns=`repeat(${Math.ceil(Math.sqrt(results.length))},minmax(0,1fr))`;
  gridViews=results.map(data=>{
    const cell=document.createElement('div');cell.className='gridCell';
    const canvas=document.createElement('canvas');cell.appendChild(canvas);
    const label=document.createElement('span');label.className='gridLabel';label.textContent=worldLabel(data,trained);
    cell.appendChild(label);host.appendChild(cell);
    const v=new ArenaView(canvas);v.arena=data.arena;v.showBrain=false;v.hud=false;v.lit=lit;v.trails=true;
    v.ownerName=meta.name||null;v.rate=speed();v.showData(data);if(!playing)v.pause();
    return v;
  });
  $('#arenaCanvas').classList.add('hidden');host.classList.remove('hidden');
  requestAnimationFrame(()=>gridViews.forEach(v=>v.resize()));
}
function worldLabel(data,trained){
  const rp=data.replay||{},own=data.cars.find(c=>!c.ghost)||data.cars[0];
  if(current==='neuro_racers'){
    const home=rp.track===trained.track?' · trained here':'';
    return `${data.arena?.track||'track'}${home} · ${own.lap_s?`lap ${own.lap_s.toFixed(1)} s`:own.crash>=0?'crashed':`${Number(own.laps).toFixed(2)} laps`}`;
  }
  const caught=data.moths.filter(m=>m.caught>=0).length;
  return `cave ${rp.cave}${rp.cave===trained.cave?' · trained here':''} · ${caught} of ${data.moths.length} caught`;
}
function rerollChampion(mode){
  if(mode)champion.mode=mode;
  champion.seed=Math.floor(Math.random()*1e6);
  if(viewDef()?.kind==='champion')setView(view);
}
// Test-world changes re-run the champion; steppers wait for the clicks to stop.
let envTimer=null;
function setChampionEnv(change,{now=false}={}){
  champion.env={...champion.env,...change};
  renderInstruments();
  clearTimeout(envTimer);
  envTimer=setTimeout(()=>{if(viewDef()?.kind==='champion')setView(view);},now?0:450);
}

// ------------------------------------------------------------------- grid ---
// One per box: every independent search's champion, animated side by side.
async function enterGrid(){
  if(!runId)return;
  const full=await fullMeta(runId),rv=meta.reveal_view||full.reveal_view;
  const host=$('#gridView');
  if(!rv){
    // Runs made before the animated grid only have the still image.
    host.classList.add('hidden');showReveal();
    $('#onScreen').innerHTML='Still image: <em>run this again to get animated boxes</em>';$('#onScreen').classList.remove('hidden');
    return;
  }
  try{
    const data=await fetch(`/runs/${runId}/${rv.file}`).then(r=>{if(!r.ok)throw new Error(r.status);return r.json();});
    const shared=data.shared_arena?await fetch(`/runs/${runId}/${data.shared_arena}`).then(r=>r.json()):null;
    if(viewDef()?.kind!=='grid')return;
    host.innerHTML='';
    const n=data.boxes.length,cols=Math.ceil(Math.sqrt(n));
    host.style.gridTemplateColumns=`repeat(${cols},minmax(0,1fr))`;
    gridViews=data.boxes.map(box=>{
      const cell=document.createElement('div');cell.className='gridCell';
      const canvas=document.createElement('canvas');cell.appendChild(canvas);
      const label=document.createElement('span');label.className='gridLabel';
      label.textContent=box.catch_rate!==undefined
        ?`${box.label} · ${Math.round(box.catch_rate*100)}% caught${box.jamming?' · jamming':''}`
        :`${box.label} · ${box.lap_s?`lap ${box.lap_s.toFixed(1)} s`:`${Number(box.laps).toFixed(2)} laps`}`;
      cell.appendChild(label);host.appendChild(cell);
      const v=new ArenaView(canvas);v.arena=box.arena||shared;v.showBrain=false;v.hud=false;v.lit=true;v.trails=true;v.rate=speed();
      v.showData(box.gen);if(!playing)v.pause();
      return v;
    });
    $('#screen').classList.add('hidden');host.classList.remove('hidden');surfaceLive(true);
    requestAnimationFrame(()=>gridViews.forEach(v=>v.resize()));
  }catch(_){markViewUnusable('grid');}
}
function exitGrid(){gridViews.forEach(v=>v.stop());gridViews=[];const host=$('#gridView');if(host){host.innerHTML='';host.classList.add('hidden');}}
// meta.arena_view.frames maps each saved frame to [generation, fraction].
function generationAt(index){
  const frames=meta.arena_view?.frames;
  if(!Array.isArray(frames)||!frames.length)return 0;
  const entry=frames[Math.max(0,Math.min(frames.length-1,index))];
  return Array.isArray(entry)?Number(entry[0]):Number(entry)||0;
}

// ------------------------------------------------------------ run/replay --
function resetRun(){
  if(pollTimer)clearInterval(pollTimer);pollTimer=null;
  stopPlay();exitArena();exitFusion();exitGalaxy();exitGrid();
  champion={mode:champion.mode,seed:0,data:null,results:[],env:{}};
  $('#onScreen').classList.add('hidden');$('#instruments').classList.add('hidden');
  runId=null;meta={};frame=0;total=0;lastKnownFrame=-1;numbers={};arenaGeneration=-1;view=null;
  failedViews=new Set();
  $('#screen').onerror=null;$('#screen').removeAttribute('src');
  surfaceLive(false);setBusy(false);enableTransport(false);
  $('#progressBar').style.width='0';$('#seek').max=0;$('#seek').value=0;
  $('#frameLabel').textContent='—';$('#captionNumbers').innerHTML='';
  setStatus('READY');
}
function openRun(r){
  resetRun();
  runId=r.id;meta=r;total=r.frames;
  $('#seek').max=Math.max(0,r.frames-1);
  enableTransport(true);
  setStatus('SAVED RUN','replay');
  const list=availableViews();
  view=list.length?list[0].id:'frames';
  renderViewSwitch();
  setView(view,{autoplay:true});
}
async function startRun(){
  if(!current)return;
  let settings;
  try{settings=collectSettings();}catch(error){setStatus('CHECK SETTINGS','failed');setAction('Check the settings',error.message);return;}
  Object.assign(settings,controlState._settings||{});
  const params={};
  Object.keys(spec().params).forEach(key=>{
    if(current==='neural_wall'&&key==='target')params[key]=controlState.target??Number(spec().params[key].value);
    else params[key]=controlState[key]??Number(spec().params[key].value);
  });
  const request={profile:$('#profile').value,frames:Number($('#frames').value),params,settings,
                 backend:$('#backend').value,method:methodValue()};
  const population=controlState._population;
  if(population!==undefined)request.parallel_count=Number(population);
  else if(supportsParallel())request.parallel_count=1;   // one model, run once
  if(current==='fluid'&&obstacles){
    if(Number(params.obstacle)===3&&!obstacles.hasCells()){
      setStatus('CHECK THE SHAPE','failed');setAction('Draw something first','"Custom shape only" needs at least one block in the grid.');return;}
    if(obstacles.hasCells()||Number(params.obstacle)===3)request.obstacle_grid=obstacles.cells;
  }
  if(current==='neural_wall'&&target?.custom)request.target_image=target.dataUrl();
  if(current==='molecular_dynamics'&&chain&&methodValue()!=='shuttle'){const own=chain.getChain();if(own)request.chain=own;}
  if(isGame()&&controlState._name)request.name=controlState._name;
  if(current==='neuro_racers'&&controlState._ghosts){const ghosts=ghostRuns(params.track).map(r=>r.id);if(ghosts.length)request.ghosts=ghosts;}
  if(builder)request.brain=builder.getSpec();
  if(builders)request.brain=Object.fromEntries(Object.entries(builders).map(([role,b])=>[role,b.getSpec()]));
  const cluster=$('#runOn').value;
  if(cluster!=='local'){
    const id=await HPC.confirmRun(current,request,cluster);
    if(!id)return;
    stopPresenting({quiet:true});resetRun();
    runId=id;total=request.frames;$('#seek').max=Math.max(0,total-1);
    setBusy(true,'Sending to the cluster…');setStatus('SUBMITTING','running');
    setAction('Sent to the cluster','It runs there and comes back here when it is finished.');
    pollTimer=setInterval(poll,1000);return;
  }
  stopPresenting({quiet:true});resetRun();
  setBusy(true,'Setting up…');setStatus('COMPUTING','running');
  const response=await fetch('/api/run/'+current,{method:'POST',headers:{'content-type':'application/json'},
                                                  body:JSON.stringify(request)});
  if(!response.ok){
    let detail='The simulation was rejected.';
    try{detail=(await response.json()).detail||detail;}catch(_){}
    setBusy(false);setStatus('FAILED','failed');setAction('Could not start',detail);return;
  }
  runId=(await response.json()).id;
  total=request.frames;$('#seek').max=Math.max(0,total-1);
  setAction('Computing…','Frames appear as the simulation writes them.');
  pollTimer=setInterval(poll,300);
}
function supportsParallel(){
  const schema=specs?.profile_setting_schema?.[current]||{};
  return 'ensemble' in schema||'networks' in schema;
}
async function poll(){
  if(!runId)return;
  let m;
  try{m=await (await fetch(`/api/run/${runId}?t=${Date.now()}`)).json();}catch(_){return;}
  const away=HPC.describe(m);
  if(away){setBusy(true,away.message);setStatus(away.badge,'running');setAction(`On ${away.label}`,away.message);return;}
  const before=viewSignature();
  meta={...meta,...m};
  if(m.overlay)renderNumbers(m.overlay);
  if(m.frame!==undefined&&m.frame>=0){
    setBusy(false);
    if(m.frame!==lastKnownFrame){lastKnownFrame=m.frame;enableTransport(true);showFrame(m.frame,total);}
  }
  // A streaming run writes its arena/torus/3D data partway through; switch to
  // the richer view the moment it exists rather than at the end.
  if(viewSignature()!==before){renderViewSwitch();if(viewDef())setView(view);}
  if(m.status==='complete'){
    clearInterval(pollTimer);pollTimer=null;
    total=Number(m.frames)||total;$('#seek').max=Math.max(0,total-1);
    setStatus('COMPLETE');setBusy(false);enableTransport(true);
    await loadLibrary();renderHistory();
    const saved=library.find(r=>r.id===runId);
    if(saved)meta={...saved,...meta,has_reveal:saved.has_reveal};
    failedViews=new Set();view=null;frame=0;fullMetaCache.delete(runId);
    renderViewSwitch();
    setView(view,{autoplay:true});
    $('#replay').classList.remove('hidden');
    setAction('Done','Change anything above and run it again.');
  }
  if(m.status==='failed'){
    clearInterval(pollTimer);pollTimer=null;
    setBusy(false);
    // Keep something on screen; the error stays in the action bar.
    const error=m.error||'Try a lower quality preset in the settings drawer.';
    const fallback=showcaseRuns(current)[0];
    if(fallback&&lastKnownFrame<0)openRun(fallback);
    setStatus('FAILED','failed');setAction('That run failed',error);
  }
}

// ------------------------------------------------------ presenter drawer --
const SETTING_LABELS={shots:'Virtual shots',learning_rate:'Policy learning rate',width:'Output width',
  height:'Output height',ensemble:'Independent runs',radial_points:'Radial grid points',nx:'Lattice width',
  ny:'Lattice height',steps_per_frame:'Steps per frame',tracers:'Tracer particles',trail:'Trail length',
  tracer_boost:'Tracer speed boost',total_steps:'Total simulation steps',grid:'Density grid',
  particles:'Particles / bodies',sweep_steps:'Grid-view steps',jeans_ref:'Jeans length reference',
  ic_amplitude:'Initial fluctuation amplitude',ic_velocity:'Initial velocity amplitude',substeps:'Solver substeps',
  span_gyr:'Simulated duration (Gyr)',softening:'Gravity softening',force_tile:'Force tile size',
  max_step:'Maximum gravity step',n:'Simulation grid',networks:'Networks trained',tile:'Network tile pixels',
  sweep_n:'Grid-view resolution',batch:'Training batch size',horizon:'Training horizon',
  train_updates:'Training updates',display_steps:'Control steps per shot',sweep_particles:'Particles per box',
  population:'Population size',generations:'Generations',sim_steps:'Steps per race',hunt_steps:'Steps per hunt',
  record_every:'Record every N steps',record_cars:'Agents drawn',moth_head_start:'Moth head start',
  reveal_population:'Population per box',reveal_generations:'Generations per box'};

function renderAdvanced(){
  const host=$('#advancedFields');host.innerHTML='';
  const schema=specs?.profile_setting_schema?.[current]||{};
  const preset=specs?.profiles?.[$('#profile').value]?.[current]||{};
  Object.entries(schema).forEach(([key,rule])=>{
    // Ensemble width is always carried by parallel_count - from the visitor's
    // population buttons, from the drawer's "Independent runs", or forced to 1
    // - and the solver lets parallel_count override this setting. Rendering it
    // here too would show a number the run then ignores.
    if(key==='ensemble'||key==='networks')return;
    // Owned by a big visitor button instead.
    if(controlDefs().some(d=>d.setting===key))return;
    const value=preset[key];
    const field=document.createElement('label');
    field.innerHTML=`<span>${escapeHtml(SETTING_LABELS[key]||key.replaceAll('_',' '))} <em>${escapeHtml(key)}</em></span>
      <input id="s_${key}" type="number" min="${rule.min}" max="${rule.max}" step="${rule.step}" value="${value}" required>`;
    const input=field.querySelector('input');
    input.oninput=()=>{field.classList.toggle('changed',Number(input.value)!==Number(preset[key]));countChanged();};
    host.appendChild(field);
  });
  countChanged();
}
function countChanged(){
  const n=document.querySelectorAll('#advancedFields .changed').length;
  $('#advancedSummary').textContent=n?`${n} custom value${n===1?'':'s'}`:`Matching the ${$('#profile').value} preset`;
}
function collectSettings(){
  const out={},schema=specs?.profile_setting_schema?.[current]||{};
  Object.keys(schema).forEach(key=>{
    const input=$('#s_'+key);if(!input)return;
    if(!input.checkValidity())
      throw new Error(`${SETTING_LABELS[key]||key} must be between ${schema[key].min} and ${schema[key].max}.`);
    out[key]=Number(input.value);
  });
  return out;
}
function applyMethods(){
  const capability=specs.capabilities?.[current]||{},methods=capability.methods||['default'];
  const select=$('#method');select.innerHTML='';
  methods.forEach(m=>{const option=document.createElement('option');option.value=m;
    option.textContent=capability.method_labels?.[m]||m.replaceAll('_',' ');select.appendChild(option);});
  select.value=capability.default_method||methods[0];
  // Where the mode is the headline choice it gets a big visitor control; the
  // drawer copy would then be a second, quieter way to set the same thing.
  const headline=controlDefs().some(d=>d.method);
  $('#methodField').classList.toggle('hidden',methods.length<2||headline);
  $('#methodHelp').textContent=capability.method_descriptions?.[select.value]||'';
  select.onchange=()=>{$('#methodHelp').textContent=capability.method_descriptions?.[select.value]||'';
    renderControls();renderAdvanced();syncChain();};
}
function applyParallelField(){
  // Demo mode only ever shows a population where a visitor control owns it;
  // every other demo runs one simulation, so there is no width to choose.
  $('#parallelField').classList.add('hidden');
}
function applyBackendOptions(){
  const allowed=specs.capabilities?.[current]?.backends||['cpu','gpu','hybrid'];
  const b=specs.backends||{},select=$('#backend');
  [['gpu',b.gpu],['hybrid',b.hybrid]].forEach(([id,info])=>{
    const option=[...select.options].find(o=>o.value===id);if(!option)return;
    option.disabled=!(info||{}).available||!allowed.includes(id);
  });
  if(select.selectedOptions[0]?.disabled)select.value='auto';
}

function openSettings(){applyBackendOptions();renderShowcasePicker();$('#settings').classList.remove('hidden');$('#settingsClose').focus();}
function closeSettings(){
  if($('#settings').classList.contains('hidden'))return;
  $('#settings').classList.add('hidden');$('#settingsOpen').focus();
}

// ----------------------------------------------------------------- idle --
function resetIdle(){
  clearTimeout(idleTimer);
  const action=$('#idleAction').value;
  if(action==='none'||present)return;
  if(action==='picker'&&!current)return;
  idleTimer=setTimeout(()=>{
    if(present||!$('#compare').classList.contains('hidden'))return;
    if(action==='slideshow')startSlideshow();else if(current)backToPicker();
  },IDLE_MS);
}



// ---------------------------------------------------------------- on screen --
// Which saved run is on the picture, and for Star in a Bottle which mode it
// was computed in: the mode buttons set up the NEXT run, so the picture needs
// its own label.
function methodHeadline(){return (KIOSK[current]?.controls||[]).some(c=>c.method);}
function methodName(m){return (specs.capabilities?.[current]?.method_labels?.[m]||m||'').replace(/ \(3D\)$/,'');}
function updateOnScreen(){
  const el=$('#onScreen');
  if(!methodHeadline()||!runId){if(!el.innerHTML.startsWith('Still'))el.classList.add('hidden');return;}
  const r=runById(runId);
  el.innerHTML=`On screen: <em>${escapeHtml(methodName(meta.method||r?.method))}</em>${pollTimer?' · computing now':r?` · ${escapeHtml(runLabel(r))}`:''}`;
  el.classList.remove('hidden');
}
function runsWithMethod(m){
  const picked=showcaseRuns(current).filter(r=>r.method===m);
  if(picked.length)return picked;
  return demoRuns(current).filter(r=>r.method===m&&r.status==='complete')
    .sort((a,b)=>(Boolean(b.favourite)-Boolean(a.favourite))||(b.created-a.created));
}
function showRunForMethod(m){
  if(pollTimer)return;
  const onScreen=runId&&(meta.method||runById(runId)?.method);
  if(onScreen===m)return updateOnScreen();
  const match=runsWithMethod(m)[0];
  if(match){stopPresenting({quiet:true});openRun(match);return;}
  resetRun();
  $('#idleTitle').textContent=`No ${methodName(m)} run saved yet.`;
  $('#idleText').innerHTML='Press <em>Run it</em> to compute one.';
}

// ------------------------------------------------------------- instruments --
// The strip under the picture: what the network looks like and whether it is
// learning (training) or only being used (inference). Nothing here overlays
// the simulation.
function isGame(){return ['neuro_racers','bat_vs_moth'].includes(current);}
function renderInstruments(){
  const host=$('#instruments'),kind=viewDef()?.kind;
  if(isGame()&&runId&&(kind==='arena'||kind==='champion'||kind==='grid')){
    const inference=kind==='champion',grid=kind==='grid';
    host.innerHTML=grid?`<div class="instPanel wide"><header><b>One per box</b><small>independent searches</small></header><p class="instText">Every box is a completely separate evolution of the same brain design, started from its own random population. Nothing is shared between boxes, so the differences you see are down to chance.</p></div>`
      :`${inference?testWorldPanel():''}<div class="instPanel brainPanel"><header><b>${inference?'The network, frozen':'This generation’s champion'}</b><small>live: node brightness is each neuron’s output; cyan wires push, pink wires pull</small></header><canvas id="brainCanvas"></canvas></div>
      <div class="instPanel"><header><span class="phaseTag ${inference?'infer':'train'}">${inference?'Inference':'Training'}</span><small>${inference?'the network is only being used; nothing is learning':'selection, crossover and mutation, generation by generation'}</small></header>
      ${inference?'':'<canvas id="trainChart" class="trainChart"></canvas>'}<p id="instText" class="instText"></p></div>`;
    host.classList.remove('hidden');
    if(inference)wireTestWorldPanel(host);
    const brainOwner=inference&&gridViews.length?gridViews[0]:arena;
    if(brainOwner)brainOwner.attachBrainCanvas($('#brainCanvas'));
    if(inference)updateInferenceText();else if(!grid)updateTrainingInstruments();
    return;
  }
  if(current==='neural_wall'&&runId&&(meta.view_modes||runById(runId)?.view_modes)){
    host.innerHTML=`<div class="instRow">
      <figure class="instPanel"><header><b>In</b><small>the picture, squeezed to the chosen size</small></header>
        <div class="pixPair"><img id="cmpSource" alt="Original picture"><span aria-hidden="true">→</span><img id="cmpIn" class="pixelated" alt="Squeezed picture"></div><p id="cmpInText" class="instText"></p></figure>
      <figure class="instPanel"><header><span class="phaseTag train">Training</span><small>every network adjusts its weights to redraw the picture</small></header>
        <canvas id="cmpChart" class="trainChart"></canvas><p id="cmpTrainText" class="instText"></p></figure>
      <figure class="instPanel"><header><b>Out</b><small>drawn from the winning network’s weights alone</small></header>
        <div class="pixPair"><img id="cmpOut" class="pixelated" alt="Network reconstruction"></div><p id="cmpOutText" class="instText"></p></figure></div>`;
    host.classList.remove('hidden');updateCompressionInstruments(frame);return;
  }
  if(current==='fusion_plasma'&&runId&&(meta.method==='guardian'||runById(runId)?.method==='guardian')){
    host.innerHTML=`<div class="instRow">
      <figure class="instPanel"><header><span class="phaseTag infer">Inference</span><small>the frozen network steering the coils during this shot</small></header><img id="instNet" alt="Policy network with live activations"></figure>
      <figure class="instPanel"><header><span class="phaseTag train">Training</span><small>between shots: wall losses per shot, before and after each update</small></header><img id="instShots" alt="Training scoreboard"></figure>
      <figure class="instPanel"><header><b>Cross-section</b><small>one slice through the torus, seen end-on</small></header><img id="instPol" alt="Poloidal cross-section"></figure></div>
      <div id="shotTimeline" class="shotTimeline"></div><p id="instPhase" class="instText"></p>`;
    host.classList.remove('hidden');updateFusionInstruments(frame);return;
  }
  host.classList.add('hidden');host.innerHTML='';
}
async function updateTrainingInstruments(){
  const text=$('#instText'),chart=$('#trainChart');if(!text||!runId)return;
  const id=runId,full=pollTimer?meta:await fullMeta(id);if(id!==runId)return;
  const stats=full.generation_stats||[],g=arenaGeneration,row=stats.find(s=>s.generation===g);
  const gens=Number(full.lab?.generations||stats.length||0);
  const pop=full.settings?.population;
  if(current==='neuro_racers'){
    text.textContent=`Generation ${g} of ${gens||'?'}. ${pop?`${pop} cars`:'Every car'} drove with this brain design and different weights; the ones that got furthest became the parents of the next generation.`
      +(row?` Best distance ${row.best_laps.toFixed(2)} laps${row.fastest_lap_s?`, fastest lap ${row.fastest_lap_s.toFixed(1)} s`:''}, ${Math.round(row.lap_rate*100)}% finished a lap.`:'');
    drawTrainChart(chart,stats,g,[{key:'best_laps',label:'best distance (laps)',colour:'#ffc45c'},{key:'lap_rate',label:'share finishing a lap',colour:'#5fd6ff',scale:Math.max(1,...stats.map(s=>s.best_laps||0))}]);
  }else{
    text.textContent=`Generation ${g} of ${gens||'?'}. Bats that caught more and moths that survived longer became the parents of the next generation.`
      +(row?` Catch rate ${Math.round(row.catch_rate*100)}%, ${Math.round(row.jammed_calls*100)}% of calls jammed${row.jamming_evolved?', jamming has evolved':''}.`:'');
    drawTrainChart(chart,stats,g,[{key:'catch_rate',label:'catch rate',colour:'#ffbe50'},{key:'jammed_calls',label:'calls jammed',colour:'#ff5ac8'}]);
  }
}
// The test world, right under the picture: where the frozen champion is run.
// These never change the network or start a training run - that is what the
// training settings further down and "Run it" are for.
function testWorldPanel(){
  const env=championEnv(),t=trainedWorld(),racers=current==='neuro_racers';
  const on=c=>c?' class="on" aria-pressed="true"':' aria-pressed="false"';
  const cells=[];
  if(racers){
    const opts=paramSpec('track')?.options||{};
    cells.push(`<div class="box wide"><label>Track</label><div class="seg">${Object.entries(opts).map(([v,n])=>
      `<button type="button" data-env-track="${v}"${on(env.boxes===1&&Number(v)===env.track)}>${escapeHtml(n)}${Number(v)===t.track?' ★':''}</button>`).join('')}
      <button type="button" data-env-boxes="4"${on(env.boxes>1)}>All four at once</button></div>
      <small class="boxHint">★ is the track it trained on.</small></div>`);
    const rivals=ghostRuns(env.track).filter(r=>r.id!==runId);
    const names=rivals.map(r=>r.name||r.id.slice(-5));
    cells.push(`<div class="box wide"><label>Who drives</label><div class="seg">
      <button type="button" data-champ="final"${on(champion.mode==='final')}>Final champion</button>
      <button type="button" data-champ="learn"${on(champion.mode==='learn')}>First · middle · final</button>
      <button type="button" data-champ="race"${on(champion.mode==='race')}${rivals.length?'':' disabled'}>Race the ghosts</button></div>
      <small class="boxHint">${rivals.length?`Race your champion against ${names.map(escapeHtml).join(', ')}: the best saved champions on this track, from the same start.`:'No other saved champions on this track yet.'}</small></div>`);
  }else{
    const stepper=(key,label,value,spec)=>`<div class="box"><label>${label}</label><div class="boxRow">
      <button class="stepBtn" type="button" data-env-step="${key}" data-dir="-1"${value<=Number(spec.min)?' disabled':''} aria-label="Decrease">−</button>
      <div class="boxValue"><span>${value}</span><small>${value===t[key]?'trained with this':'&nbsp;'}</small></div>
      <button class="stepBtn" type="button" data-env-step="${key}" data-dir="1"${value>=Number(spec.max)?' disabled':''} aria-label="Increase">+</button></div></div>`;
    cells.push(stepper('cave','Cave layout',env.cave,paramSpec('cave')||{min:1,max:999}));
    cells.push(stepper('moths','Moths per cave',env.moths,paramSpec('moths')||{min:2,max:6}));
    cells.push(`<div class="box wide"><label>Caves at once</label><div class="seg">${[1,4,9,16].map(n=>
      `<button type="button" data-env-boxes="${n}"${on(env.boxes===n)}>${n}</button>`).join('')}</div>
      <small class="boxHint">Each box is the next cave layout on from the one above.</small></div>`);
  }
  return `<div class="instPanel wide testWorld"><header><span class="phaseTag infer">Test world</span><small>change the world, not the network: the frozen champion is simply run again</small></header>
    <div class="testGrid">${cells.join('')}</div>
    <div class="instButtons"><button type="button" data-champ="reroll">New random start</button>${envIsTrained()?'':'<button type="button" data-env-reset>Back to the training world</button>'}</div></div>`;
}
function wireTestWorldPanel(host){
  host.querySelectorAll('[data-champ]').forEach(b=>b.onclick=()=>rerollChampion(b.dataset.champ==='reroll'?null:b.dataset.champ));
  host.querySelectorAll('[data-env-track]').forEach(b=>b.onclick=()=>setChampionEnv({track:Number(b.dataset.envTrack),boxes:1},{now:true}));
  host.querySelectorAll('[data-env-boxes]').forEach(b=>b.onclick=()=>setChampionEnv({boxes:Number(b.dataset.envBoxes)},{now:true}));
  host.querySelectorAll('[data-env-step]').forEach(b=>b.onclick=()=>{
    const key=b.dataset.envStep,spec=paramSpec(key)||{},env=championEnv();
    setChampionEnv({[key]:Math.max(Number(spec.min??1),Math.min(Number(spec.max??999),env[key]+Number(b.dataset.dir)))});});
  host.querySelector('[data-env-reset]')?.addEventListener('click',()=>{champion.env={};setChampionEnv({},{now:true});});
}
function updateInferenceText(){
  const text=$('#instText'),data=champion.data;if(!text||!data)return;
  const seed=data.replay?.seed,results=champion.results||[data],trained=trainedWorld();
  if(results.length>1){
    text.textContent=`The same frozen champion in ${results.length} ${current==='neuro_racers'?'tracks':'caves'}, from random start #${seed}. `
      +results.map(r=>worldLabel(r,trained)).join(' · ')+'.';
    return;
  }
  if(current==='neuro_racers'){
    const where=data.replay?.track!==undefined&&data.replay.track!==trained.track?` on the ${data.arena?.track||'new track'}, a track it never trained on,`:'';
    const parts=data.cars.map(c=>`${c.ghost?c.label:(meta.name?`${meta.name} ${c.label}`:c.label)}: ${c.lap_s?`lap ${c.lap_s.toFixed(1)} s`:c.crash>=0?'crashed':`${Number(c.laps).toFixed(2)} laps`}`);
    const own=data.cars.filter(c=>!c.ghost).length;
    text.textContent=`Saved champion network${own>1?'s':''}, frozen, driving${where} from a fresh random start (#${seed}) it has never seen. ${parts.join(' · ')}.`;
  }else{
    const caught=data.moths.filter(m=>m.caught>=0).length,rp=data.replay||{};
    const moved=rp.cave!==undefined&&(rp.cave!==trained.cave||rp.moths!==trained.moths)?` in cave ${rp.cave} with ${rp.moths} moths (it trained in cave ${trained.cave} with ${trained.moths})`:'';
    text.textContent=`The generation ${data.generation} champion bat and its moths, frozen${moved}, from a fresh random start (#${seed}). ${caught} of ${data.moths.length} moths caught.`;
  }
}
function drawTrainChart(canvas,stats,generation,series){
  if(!canvas)return;
  const rect=canvas.getBoundingClientRect(),d=Math.min(devicePixelRatio||1,2);if(!rect.width)return;
  canvas.width=Math.round(rect.width*d);canvas.height=Math.round(rect.height*d);
  const ctx=canvas.getContext('2d'),w=canvas.width,h=canvas.height,pad=10*d,bottom=h-18*d;
  ctx.clearRect(0,0,w,h);if(stats.length<2)return;
  const gens=stats.map(s=>s.generation),g0=Math.min(...gens),g1=Math.max(...gens);
  const x=g=>pad+(w-2*pad)*(g-g0)/Math.max(1,g1-g0);
  ctx.strokeStyle='rgba(255,255,255,.08)';ctx.lineWidth=d;ctx.beginPath();ctx.moveTo(pad,bottom);ctx.lineTo(w-pad,bottom);ctx.stroke();
  series.forEach((s,i)=>{
    const values=stats.map(r=>Number(r[s.key])||0),peak=s.scale?1:Math.max(1e-9,...values);
    const y=v=>bottom-(bottom-pad)*(s.scale?v:v/peak);
    ctx.strokeStyle=s.colour;ctx.lineWidth=2*d;ctx.beginPath();
    stats.forEach((r,j)=>{const px=x(r.generation),py=y(values[j]);j?ctx.lineTo(px,py):ctx.moveTo(px,py);});ctx.stroke();
    ctx.fillStyle=s.colour;ctx.font=`700 ${10.5*d}px system-ui`;ctx.fillText(s.label,pad+i*150*d,h-4*d);
  });
  if(generation>0){const px=x(generation);ctx.strokeStyle='rgba(255,255,255,.7)';ctx.setLineDash([4*d,4*d]);ctx.beginPath();ctx.moveTo(px,pad);ctx.lineTo(px,bottom);ctx.stroke();ctx.setLineDash([]);}
}
async function updateCompressionInstruments(index){
  const out=$('#cmpOut');if(!out||!runId)return;
  const id=runId;
  const set=(el,src)=>{if(el&&el.dataset.src!==src){el.dataset.src=src;el.src=src;}};
  set($('#cmpSource'),`/runs/${id}/target_source.png`);set($('#cmpIn'),`/runs/${id}/target_reduced.png`);
  set(out,`/runs/${id}/recon/frame_${pad(index)}.png`);
  const v=numbers||{};
  if(v['picture in'])$('#cmpInText').textContent=`${v['picture in']} of raw colour values to remember.`;
  if(v['training step'])$('#cmpTrainText').textContent=`Training step ${v['training step']} · ${v['networks']||''}`;
  if(v['network size'])$('#cmpOutText').innerHTML=`Network: <b>${escapeHtml(v['network size'])}</b> → <b>${escapeHtml(v['compression'])}</b>. Quality ${escapeHtml(v['quality'])}. A JPEG the same size: ${escapeHtml(v['same-size JPEG'])}.`;
  const full=pollTimer?meta:await fullMeta(id);if(id!==runId)return;
  const history=(full.compression_history||[]).map(h=>({generation:h.frame+1,psnr:h.psnr}));
  drawTrainChart($('#cmpChart'),history,index+1,[{key:'psnr',label:'quality of the best network (dB)',colour:'#b5e7cf'}]);
}
async function updateFusionInstruments(index){
  const net=$('#instNet');if(!net||!runId)return;
  const id=runId,n=pad(index);
  for(const [el,name] of [[net,'network'],[$('#instShots'),'shots'],[$('#instPol'),'poloidal']]){
    const src=`/runs/${id}/overlays/${name}/frame_${n}.jpg`;if(el.dataset.src!==src){el.dataset.src=src;el.src=src;}
  }
  const full=pollTimer?meta:await fullMeta(id);if(id!==runId||!$('#shotTimeline'))return;
  const shots=Math.max(1,Number(full.shots)||1),frames=Math.max(1,total);
  // Same shot plan as fusion_plasma.shot_plan: equal slices of the frames.
  const edges=Array.from({length:shots+1},(_,k)=>Math.round(frames*k/shots));
  const shot=Math.max(0,edges.findIndex((e,k)=>k<shots&&index>=e&&index<edges[k+1]));
  const history=full.shot_history||[];
  $('#shotTimeline').innerHTML=Array.from({length:shots},(_,k)=>
    `${k?`<span class="trainMark ${k<=shot?'done':''}" title="Training between shots ${k} and ${k+1}">⟳</span>`:''}<span class="shotSeg ${k<shot?'done':k===shot?'now':''}">Shot ${k+1}${history[k]?` · ${history[k].lost} lost`:''}</span>`).join('');
  const before=history[shot-1];
  $('#instPhase').textContent=`Shot ${shot+1} of ${shots}: the network is frozen and flying the plasma (inference). `
    +(shot===0?'It has not trained yet, so it starts out untrained.'
      :`Before this shot it trained on everything shot ${shot} did${before?.trained_to?` (${before.trained_to} updates so far${before.loss_after!==undefined?`, loss ${Number(before.loss_after).toFixed(3)}`:''})`:''}.`);
}

// ------------------------------------------------------------ run labels --
function runLabel(r){
  const d=new Date((r.created||0)*1000);
  return isNaN(d)?r.id:d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
}
function paramText(demo,key,value){
  const p=specs.demos[demo]?.params?.[key];if(!p)return String(value);
  const def=(KIOSK[demo]?.controls||[]).find(c=>c.key===key);
  if(def?.choices)return def.choices[Math.round((Number(value)-Number(p.min))/Number(p.step||1))]??String(value);
  if(p.kind==='choice')return p.options?.[String(Math.round(Number(value)))]??String(value);
  if(p.kind==='toggle')return Number(value)?'On':'Off';
  const dp=def?.decimals??decimalsFor(p.step);
  return `${Number(value).toFixed(dp)}${def?.unit?` ${def.unit}`:''}`;
}
function paramLabel(demo,key){
  const def=(KIOSK[demo]?.controls||[]).find(c=>c.key===key),p=specs.demos[demo]?.params?.[key];
  return def?.label||p?.label||key.replaceAll('_',' ').replace(/^./,c=>c.toUpperCase());
}
// One line a presenter can recognise a run by: its headline settings.
// Runs saved before a demo had a choice of solver recorded "default".
function solverOf(r){
  const m=r?.method;return !m||m==='default'?(KIOSK[r?.demo]?.legacyMethod||m):m;
}
function paramApplies(demo,key,method){
  const methods=specs.demos[demo]?.params?.[key]?.methods;
  return !Array.isArray(methods)||methods.includes(method);
}
function runSummary(r){
  const keys=(KIOSK[r.demo]?.controls||[]).filter(c=>!c.onlyMethod||c.onlyMethod===solverOf(r))
    .map(c=>c.key).filter(Boolean);
  const parts=keys.filter(k=>r.params?.[k]!==undefined).map(k=>`${paramLabel(r.demo,k)} ${paramText(r.demo,k,r.params[k])}`);
  const methodLabel=r.method&&specs.capabilities?.[r.demo]?.method_labels?.[r.method];
  // Only where the mode is a visitor choice; a solver name is noise elsewhere.
  if(methodLabel&&(KIOSK[r.demo]?.controls||[]).some(c=>c.method))parts.unshift(methodLabel.replace(/^Mode \d · /,''));
  if(r.params?._obstacle_grid)parts.push('custom shape');
  return parts.join(' · ');
}

// ------------------------------------------------------------ favourites --
async function toggleFavourite(r){
  const favourite=!r.favourite;
  try{
    const res=await fetch(`/api/runs/${encodeURIComponent(r.id)}/favourite`,{method:'POST',
      headers:{'content-type':'application/json'},body:JSON.stringify({favourite})});
    if(!res.ok)throw new Error(String(res.status));
    library.filter(x=>x.id===r.id).forEach(x=>{x.favourite=favourite;});
  }catch(error){setAction('Could not save the favourite',`The server answered ${error.message}.`);}
  renderHistory();renderShowcasePicker();
}
function starButton(r){
  const b=document.createElement('button');b.type='button';b.className='star'+(r.favourite?' on':'');
  b.textContent=r.favourite?'★':'☆';
  const label=r.favourite?'Remove from favourites':'Add to favourites';
  b.title=label;b.setAttribute('aria-label',label);b.setAttribute('aria-pressed',String(Boolean(r.favourite)));
  b.onclick=event=>{event.stopPropagation();toggleFavourite(r);};
  return b;
}

// --------------------------------------------------------- showcase picker --
function renderShowcasePicker(){
  const host=$('#showcasePicker');if(!host||!current)return;
  const picked=[...(libraryData.showcase?.[current]||[])].filter(runById);
  const runs=demoRuns(current).filter(r=>r.status==='complete');
  runs.sort((a,b)=>(picked.includes(b.id)-picked.includes(a.id))||(Boolean(b.favourite)-Boolean(a.favourite))||(b.created-a.created));
  const source=showcaseSource(current);
  $('#showcaseSummary').textContent=picked.length?`${picked.length} of ${SHOWCASE_MAX} picked`
    :source==='favourites'?'None picked: playing your favourites':runs.length?'None picked: playing the newest run':'No saved runs yet';
  host.innerHTML='';
  runs.slice(0,40).forEach(r=>{
    const order=picked.indexOf(r.id);
    const item=document.createElement('div');item.className='pickRun'+(order>=0?' on':'');
    item.innerHTML=`<button type="button" class="pickRunMain" aria-pressed="${order>=0}"><span class="pickOrder">${order>=0?order+1:'+'}</span><img src="${escapeHtml(r.thumb)}" alt="" loading="lazy"><span class="pickRunText"><b>${escapeHtml(runLabel(r))}</b><small>${escapeHtml(runSummary(r)||`${r.frames} frames`)}</small><small>${escapeHtml([r.profile,r.backend,`${r.frames} frames`].filter(Boolean).join(' · '))}</small></span></button>`;
    item.querySelector('.pickRunMain').onclick=()=>toggleShowcase(r.id);
    item.appendChild(starButton(r));
    host.appendChild(item);
  });
}
const SHOWCASE_MAX=3;
async function toggleShowcase(id){
  const picked=[...(libraryData.showcase?.[current]||[])].filter(runById);
  const at=picked.indexOf(id);
  if(at>=0)picked.splice(at,1);
  else if(picked.length>=SHOWCASE_MAX){$('#showcaseSummary').textContent=`Three at most: untick one first`;return;}
  else picked.push(id);
  try{
    const res=await fetch(`/api/showcase/${encodeURIComponent(current)}`,{method:'PUT',
      headers:{'content-type':'application/json'},body:JSON.stringify({runs:picked})});
    if(!res.ok)throw new Error(String(res.status));
    libraryData.showcase={...libraryData.showcase,[current]:picked};
  }catch(error){$('#showcaseSummary').textContent=`Could not save (${error.message})`;return;}
  renderShowcasePicker();
  $('#replay').classList.toggle('hidden',!showcaseRuns(current).length);
}

// ---------------------------------------------------------------- history --
function renderHistory(){
  const host=$('#historyList');if(!host||!current)return;
  const all=demoRuns(current),runs=historyFavsOnly?all.filter(r=>r.favourite):all;
  compareSel=compareSel.filter(id=>runs.some(r=>r.id===id));
  $('#historyCount').textContent=`${all.length} saved run${all.length===1?'':'s'} of this experiment${all.some(r=>r.favourite)?` · ${all.filter(r=>r.favourite).length} favourite`:''}`;
  $('#historyFavs').classList.toggle('on',historyFavsOnly);$('#historyFavs').setAttribute('aria-pressed',String(historyFavsOnly));
  $('#compareGo').disabled=compareSel.length!==2;
  $('#compareGo').textContent=compareSel.length===2?'Compare these 2 runs':`Compare 2 runs (${compareSel.length} ticked)`;
  host.innerHTML='';
  if(!runs.length){host.innerHTML=`<p class="historyEmpty">${historyFavsOnly?'No favourites yet. Star a run to keep it here.':'Nothing has been run yet.'}</p>`;}
  runs.slice(0,historyLimit).forEach(r=>{
    const ticked=compareSel.includes(r.id);
    const item=document.createElement('article');
    item.className='histItem'+(r.id===runId?' playing':'')+(ticked?' ticked':'');
    item.innerHTML=`<button type="button" class="histThumb" title="Play this run"><img src="${escapeHtml(r.thumb)}" alt="" loading="lazy"><span class="histPlay" aria-hidden="true">▶</span></button>
      <div class="histText"><b>${escapeHtml(runLabel(r))}${r.id===runId?' <em>on screen</em>':''}</b><span>${escapeHtml(runSummary(r)||'Default settings')}</span><small>${escapeHtml([r.profile==='hpc'?'HPC':r.profile,r.backend,`${r.frames} frames`,r.elapsed?`${Number(r.elapsed).toFixed(0)} s to compute`:null,r.status!=='complete'?r.status:null].filter(Boolean).join(' · '))}</small></div>
      <label class="histTick"><input type="checkbox" ${ticked?'checked':''}> Compare</label>`;
    item.querySelector('.histThumb').onclick=()=>{stopPresenting();openRun(r);renderHistory();window.scrollTo({top:0,behavior:'smooth'});};
    item.querySelector('input').onchange=e=>{
      compareSel=compareSel.filter(id=>id!==r.id);
      if(e.target.checked){compareSel.push(r.id);if(compareSel.length>2)compareSel.shift();}
      renderHistory();
    };
    item.insertBefore(starButton(r),item.querySelector('.histTick'));
    host.appendChild(item);
  });
  $('#historyMore').classList.toggle('hidden',runs.length<=historyLimit);
  $('#historyMore').textContent=`Show ${Math.min(24,runs.length-historyLimit)} more`;
}

// ---------------------------------------------------------------- compare --
function openCompare(){
  compareRuns=compareSel.map(runById).filter(Boolean);
  if(compareRuns.length!==2)return;
  stopPlay();
  // Oldest on the left, so "before" and "after" read in order.
  compareRuns.sort((a,b)=>(a.created||0)-(b.created||0));
  const panes=[...document.querySelectorAll('.comparePane')];
  compareRuns.forEach((r,i)=>{
    panes[i].querySelector('figcaption').innerHTML=`<b>${i?'B':'A'} · ${escapeHtml(runLabel(r))}</b><small>${escapeHtml(runSummary(r)||'Default settings')}</small>`;
  });
  renderCompareTable();
  compareProgress=0;$('#compare').classList.remove('hidden');
  showCompare();startCompare();
  $('#compareClose').focus();
}
function compareRows(){
  const [a,b]=compareRuns,demo=a.demo,rows=[];
  const caps=specs.capabilities?.[demo]||{};
  if((caps.methods||[]).length>1)rows.push(['Mode',caps.method_labels?.[a.method]||a.method,caps.method_labels?.[b.method]||b.method]);
  Object.keys(specs.demos[demo]?.params||{}).forEach(key=>{
    if(a.params?.[key]===undefined&&b.params?.[key]===undefined)return;
    if(!paramApplies(demo,key,solverOf(a))&&!paramApplies(demo,key,solverOf(b)))return;
    rows.push([paramLabel(demo,key),paramText(demo,key,a.params?.[key]??'—'),paramText(demo,key,b.params?.[key]??'—')]);
  });
  const count=r=>r.params?._parallel_count??r.settings?.networks??r.settings?.ensemble;
  if(count(a)!==undefined||count(b)!==undefined)rows.push(['Independent runs',String(count(a)??'—'),String(count(b)??'—')]);
  const blocks=r=>r.params?._obstacle_grid?`${r.params._obstacle_grid.flat().filter(Boolean).length} blocks`:'none';
  if(demo==='fluid')rows.push(['Drawn obstacle',blocks(a),blocks(b)]);
  const summaryKeys=[...new Set([...Object.keys(a.summary||{}),...Object.keys(b.summary||{})])];
  summaryKeys.forEach(key=>rows.push([`Result · ${key.replaceAll('_',' ')}`,String(a.summary?.[key]??'—'),String(b.summary?.[key]??'—')]));
  rows.push(['Quality preset',a.profile||'—',b.profile||'—']);
  rows.push(['Compute',a.backend||'—',b.backend||'—']);
  rows.push(['Saved frames',String(a.frames),String(b.frames)]);
  rows.push(['Time to compute',a.elapsed?`${Number(a.elapsed).toFixed(1)} s`:'—',b.elapsed?`${Number(b.elapsed).toFixed(1)} s`:'—']);
  return rows;
}
function renderCompareTable(){
  const rows=compareRows();
  $('#compareTable').innerHTML=`<thead><tr><th>Setting</th><th>A</th><th>B</th></tr></thead><tbody>${rows.map(([label,x,y])=>
    `<tr class="${x!==y?'differs':''}"><th scope="row">${escapeHtml(label)}</th><td>${escapeHtml(x)}</td><td>${escapeHtml(y)}</td></tr>`).join('')}</tbody>`;
}
// Both runs advance by the same fraction of their own length, so runs with
// different frame counts still line up start to finish.
function showCompare(){
  const panes=[...document.querySelectorAll('.comparePane')];
  compareRuns.forEach((r,i)=>{
    const index=Math.round(compareProgress*(r.frames-1));
    const img=panes[i].querySelector('img'),src=`/runs/${r.id}/frames/frame_${pad(index)}.jpg`;
    if(img.dataset.src!==src){img.dataset.src=src;img.src=src;}
  });
  $('#compareSeek').value=String(Math.round(compareProgress*1000));
  $('#compareLabel').textContent=`${Math.round(compareProgress*100)}%`;
}
function startCompare(){
  stopCompare();comparePlaying=true;$('#comparePlay').textContent='❚❚';
  const steps=Math.max(2,...compareRuns.map(r=>r.frames))-1;
  compareTimer=setInterval(()=>{compareProgress+=1/steps;if(compareProgress>1.0001)compareProgress=0;showCompare();},Math.max(16,FRAME_MS/speed()));
}
function stopCompare(){if(compareTimer)clearInterval(compareTimer);compareTimer=null;comparePlaying=false;$('#comparePlay').textContent='▶';}
function closeCompare(){
  if($('#compare').classList.contains('hidden'))return;
  stopCompare();$('#compare').classList.add('hidden');
  if(runId)startPlay(frame);
}

// -------------------------------------------------------- custom picture --
// The neural wall's own-picture target: draw, upload a photo, or take one with
// the webcam. Same 128 px canvas and PNG upload as the dashboard.
class TargetPicture{
  constructor(host){
    this.custom=false;this.stream=null;
    this.el=document.createElement('section');this.el.className='targetPicture';
    this.size=256;
    this.el.innerHTML=`<div class="targetIntro"><b>Pick a picture to compress</b><small>One of these, a photo, the camera, or draw your own with the mouse.</small><div class="targetDefaults" aria-label="Default pictures"></div><span class="targetStatus">Using a generated pattern</span></div>
      <div class="targetStage"><canvas width="256" height="256" aria-label="The picture the network will learn"></canvas><video class="hidden" autoplay muted playsinline></video></div>
      <div class="targetTools2"><button type="button" data-act="clear">Clear</button><label class="targetUpload2">Upload photo<input type="file" accept="image/*"></label><button type="button" data-act="camera">Camera</button><button type="button" data-act="capture" class="hidden primary">Take picture</button><button type="button" data-act="preset">Use preset instead</button></div>`;
    host.appendChild(this.el);
    this.canvas=this.el.querySelector('canvas');this.ctx=this.canvas.getContext('2d');
    this.video=this.el.querySelector('video');
    this.clear();this.loadDefaults();
    let drawing=false,last=null;
    const point=e=>{const r=this.canvas.getBoundingClientRect();return {x:(e.clientX-r.left)*this.size/r.width,y:(e.clientY-r.top)*this.size/r.height};};
    const paint=e=>{const p=point(e);const c=this.ctx;c.strokeStyle='#fff';c.lineCap='round';c.lineJoin='round';c.lineWidth=18;
      c.beginPath();c.moveTo(last.x,last.y);c.lineTo(p.x,p.y);c.stroke();last=p;this.useCustom('Your drawing');};
    this.canvas.addEventListener('pointerdown',e=>{drawing=true;last=point(e);this.canvas.setPointerCapture(e.pointerId);paint(e);});
    this.canvas.addEventListener('pointermove',e=>{if(drawing)paint(e);});
    ['pointerup','pointercancel'].forEach(type=>this.canvas.addEventListener(type,()=>{drawing=false;last=null;}));
    this.el.querySelector('[data-act="clear"]').onclick=()=>{this.clear();if(this.custom)this.useCustom('Blank canvas');};
    this.el.querySelector('[data-act="preset"]').onclick=()=>{this.usePreset();const box=document.querySelector('.box[data-key="target"]');box?.selectChoice?.(controlState.target??0);};
    this.el.querySelector('input[type=file]').onchange=e=>{
      const file=e.target.files?.[0];if(!file)return;
      const image=new Image();image.onload=()=>{this.crop(image,image.naturalWidth,image.naturalHeight);URL.revokeObjectURL(image.src);this.useCustom('Your photo');};
      image.src=URL.createObjectURL(file);
    };
    this.el.querySelector('[data-act="camera"]').onclick=()=>this.openCamera();
    this.el.querySelector('[data-act="capture"]').onclick=()=>{
      if(!this.video.videoWidth)return;
      this.crop(this.video,this.video.videoWidth,this.video.videoHeight);this.stop();this.useCustom('Your camera picture');
    };
  }
  clear(){this.ctx.fillStyle='#000';this.ctx.fillRect(0,0,this.size,this.size);}
  // Photos are rarely square: take the centred square rather than stretching.
  crop(source,w,h){const side=Math.min(w,h);this.clear();this.ctx.drawImage(source,(w-side)/2,(h-side)/2,side,side,0,0,this.size,this.size);}
  async loadDefaults(){
    const host=this.el.querySelector('.targetDefaults');
    let list=[];try{list=await (await fetch('/api/compression_images')).json();}catch(_){}
    host.innerHTML='';
    list.forEach(item=>{
      const b=document.createElement('button');b.type='button';b.className='targetDefault';b.title=item.name;
      b.innerHTML=`<img src="${escapeHtml(item.url)}" alt=""><span>${escapeHtml(item.name)}</span>`;
      b.onclick=()=>{const image=new Image();image.onload=()=>{this.crop(image,image.naturalWidth,image.naturalHeight);this.useCustom(item.name);
        host.querySelectorAll('.targetDefault').forEach(x=>x.classList.toggle('on',x===b));};image.src=item.url;};
      host.appendChild(b);
    });
  }
  async openCamera(){
    try{
      this.stop();
      this.stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});
      this.video.srcObject=this.stream;this.video.classList.remove('hidden');
      this.el.querySelector('[data-act="capture"]').classList.remove('hidden');
      this.status('Camera on: press Take picture');
    }catch(_){this.status('No camera available: upload a photo instead');}
  }
  stop(){
    if(this.stream){this.stream.getTracks().forEach(track=>track.stop());this.stream=null;}
    this.video?.classList.add('hidden');this.el?.querySelector('[data-act="capture"]')?.classList.add('hidden');
  }
  status(text){this.el.querySelector('.targetStatus').textContent=text;}
  useCustom(label){
    this.custom=true;this.el.classList.add('custom');this.status(`${label} will be compressed`);
    document.querySelectorAll('.box[data-key="target"] .seg button').forEach(b=>{b.classList.remove('on');b.setAttribute('aria-pressed','false');});
  }
  usePreset(){this.custom=false;this.el.classList.remove('custom');this.status('Using a generated pattern');
    this.el.querySelectorAll('.targetDefault').forEach(x=>x.classList.remove('on'));}
  dataUrl(){return this.canvas.toDataURL('image/png');}
}

// -------------------------------------------------------------- starfield --
// Background stars for the galaxy demos. Purely decorative - the caption says
// so - drawn over the frame with a screen blend so they only show in the dark.
function makeStarfield(canvas){
  const ctx=canvas.getContext('2d');
  let seed=7;const rand=()=>{seed=(seed*16807)%2147483647;return (seed-1)/2147483646;};
  const stars=Array.from({length:260},()=>{
    const bright=rand()**3;
    return {x:rand(),y:rand(),r:.35+bright*1.25,base:.18+bright*.6,speed:.6+rand()*2.2,phase:rand()*Math.PI*2,
      hue:rand()<.18?'255,222,190':rand()<.3?'190,215,255':'255,255,255',sparkle:bright>.55};
  });
  const still=matchMedia('(prefers-reduced-motion: reduce)').matches;
  let raf=0,running=false;
  function size(){
    const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);
    const w=Math.round(rect.width*dpr),h=Math.round(rect.height*dpr);
    if(w&&h&&(canvas.width!==w||canvas.height!==h)){canvas.width=w;canvas.height=h;}
    return dpr;
  }
  function draw(now){
    const dpr=size(),w=canvas.width,h=canvas.height,time=now/1000;
    ctx.clearRect(0,0,w,h);
    for(const s of stars){
      const twinkle=still?1:.55+.45*Math.sin(time*s.speed+s.phase);
      const a=s.base*twinkle,x=s.x*w,y=s.y*h,r=s.r*dpr;
      ctx.fillStyle=`rgba(${s.hue},${a.toFixed(3)})`;
      ctx.beginPath();ctx.arc(x,y,r,0,Math.PI*2);ctx.fill();
      if(s.sparkle&&twinkle>.85){
        const len=r*4.5*(twinkle-.85)/.15;
        ctx.strokeStyle=`rgba(${s.hue},${(a*.55).toFixed(3)})`;ctx.lineWidth=.6*dpr;
        ctx.beginPath();ctx.moveTo(x-len,y);ctx.lineTo(x+len,y);ctx.moveTo(x,y-len);ctx.lineTo(x,y+len);ctx.stroke();
      }
    }
    if(running&&!still)raf=requestAnimationFrame(draw);
  }
  return {
    start(){if(running)return;running=true;raf=requestAnimationFrame(draw);},
    stop(){running=false;cancelAnimationFrame(raf);ctx.clearRect(0,0,canvas.width,canvas.height);},
  };
}
function starfieldFor(id){
  const canvas=$('#starfield'),on=Boolean(id&&KIOSK[id]?.starfield);
  canvas.classList.toggle('hidden',!on);
  if(on){if(!starfield)starfield=makeStarfield(canvas);starfield.start();}
  else if(starfield)starfield.stop();
}


// ------------------------------------------------------------ full screen --
// Click the picture (without dragging it) or press F. In full screen only the
// picture shows; moving the mouse brings up play, loop, seek, speed and a
// little information along the bottom, which fades again when the mouse rests.
// Some embedded and kiosk browsers refuse the Fullscreen API without saying
// so; then the picture fills the window instead, with the same controls.
const screenEl=()=>document.querySelector('.screen');
function isFullscreen(){return screenEl().classList.contains('isFullscreen');}
function setFsState(on){
  const el=screenEl();el.classList.toggle('isFullscreen',on);document.body.classList.toggle('fsOpen',on);
  if(on){syncFullscreen();revealFsBar();}else el.classList.remove('fsActive');
  // Canvases size themselves from their box, which just changed.
  requestAnimationFrame(()=>{arena?.resize();fusion?.resize();galaxy?.resize();});
}
function toggleFullscreen(){
  if(isFullscreen()){
    if(document.fullscreenElement)document.exitFullscreen?.();else setFsState(false);
    return;
  }
  const request=screenEl().requestFullscreen?.();
  if(request)request.catch(()=>setFsState(true));
  setTimeout(()=>{if(!document.fullscreenElement&&!isFullscreen())setFsState(true);},350);
}
let fsHideTimer=0;
function revealFsBar(){
  const el=screenEl();el.classList.add('fsActive');
  clearTimeout(fsHideTimer);
  fsHideTimer=setTimeout(()=>{if(!$('#fsBar').matches(':hover'))el.classList.remove('fsActive');},2600);
}
function syncFullscreen(){
  const fsPlay=$('#fsPlay');if(!fsPlay)return;
  fsPlay.textContent=playing?'❚❚':'▶';fsPlay.disabled=$('#playPause').disabled;
  const seek=$('#fsSeek');seek.max=$('#seek').max;seek.value=$('#seek').value;seek.disabled=$('#seek').disabled;
  $('#fsFrame').textContent=$('#frameLabel').textContent;
  const r=runId&&runById(runId);
  $('#fsMeta').textContent=present?$('#presentMeta').textContent:r?`${runLabel(r)}${runSummary(r)?` · ${runSummary(r)}`:''}`:$('#status').textContent;
  $('#fsNumbers').innerHTML=$('#captionNumbers').innerHTML;
}
function bindFullscreen(){
  const surface=$('#surface');let down=null;
  // Capture phase: the 3D views stop propagation of their own drags.
  surface.addEventListener('pointerdown',e=>{down={x:e.clientX,y:e.clientY,t:performance.now()};},true);
  surface.addEventListener('pointerup',e=>{
    if(!down)return;const moved=Math.hypot(e.clientX-down.x,e.clientY-down.y),quick=performance.now()-down.t<450;down=null;
    if(e.target.closest('.fsBar,.fsEnter')||moved>6||!quick)return;
    if(isFullscreen()){playing?stopPlay():startPlay(frame);revealFsBar();}
    else if($('#surface').classList.contains('live'))toggleFullscreen();
  },true);
  $('#fsEnter').onclick=e=>{e.stopPropagation();toggleFullscreen();};
  $('#fsExit').onclick=()=>toggleFullscreen();
  $('#fsPlay').onclick=()=>{playing?stopPlay():startPlay(frame);};
  $('#fsLoop').onclick=()=>setLoop(!loopPlayback);
  $('#loopToggle').onclick=()=>setLoop(!loopPlayback);
  $('#fsSeek').oninput=e=>{stopPlay();showFrame(Number(e.target.value),total);};
  $('#fsSpeed').oninput=e=>setSpeedIndex(Number(e.target.value));
  screenEl().addEventListener('mousemove',()=>{if(isFullscreen())revealFsBar();});
  document.addEventListener('fullscreenchange',()=>setFsState(document.fullscreenElement===screenEl()));
}

// ---------------------------------------------------------------- wiring --
function bindChrome(){
  bindFullscreen();
  $('#back').onclick=backToPicker;
  $('#run').onclick=startRun;
  $('#replay').onclick=startShowcase;
  $('#showcaseStart').onclick=startShowcase;
  $('#slideshowStart').onclick=startSlideshow;
  $('#presentPrev').onclick=()=>present&&presentGo(present.index-1);
  $('#presentNext').onclick=()=>present&&presentGo(present.index+1);
  $('#presentStop').onclick=()=>{const wasSlideshow=present?.kind==='slideshow';stopPresenting();if(wasSlideshow)backToPicker();};
  $('#presentTry').onclick=tryItYourself;
  $('#speed').oninput=()=>setSpeedIndex(Number($('#speed').value));
  $('#speedDown').onclick=()=>setSpeedIndex(Number($('#speed').value)-1);
  $('#speedUp').onclick=()=>setSpeedIndex(Number($('#speed').value)+1);
  $('#historyFavs').onclick=()=>{historyFavsOnly=!historyFavsOnly;historyLimit=12;renderHistory();};
  $('#historyMore').onclick=()=>{historyLimit+=24;renderHistory();};
  $('#compareGo').onclick=openCompare;
  $('#compareClose').onclick=closeCompare;
  $('#comparePlay').onclick=()=>comparePlaying?stopCompare():startCompare();
  $('#compareSeek').oninput=e=>{stopCompare();compareProgress=Number(e.target.value)/1000;showCompare();};
  $('#slideDwell').onchange=()=>{prefs.slideDwell=Number($('#slideDwell').value);savePrefs();};
  $('#playPause').onclick=()=>playing?stopPlay():startPlay(frame);
  $('#seek').oninput=event=>{stopPlay();showFrame(Number(event.target.value),total);};
  $('#settingsOpen').onclick=openSettings;
  $('#settingsClose').onclick=closeSettings;
  $('#settingsDone').onclick=closeSettings;
  $('#settings').onclick=event=>{if(event.target===$('#settings'))closeSettings();};
  $('#settingsReset').onclick=()=>{renderAdvanced();};
  $('#profile').onchange=()=>{prefs.profile=$('#profile').value;savePrefs();renderAdvanced();};
  $('#backend').onchange=()=>{prefs.backend=$('#backend').value;savePrefs();};
  $('#frames').onchange=()=>{prefs.frames=Number($('#frames').value)||70;savePrefs();};
  $('#idleAction').onchange=()=>{prefs.idleAction=$('#idleAction').value;savePrefs();resetIdle();};
  $('#parallelCount').onchange=()=>{if(controlState._population!==undefined)controlState._population=Number($('#parallelCount').value);};
  document.addEventListener('keydown',event=>{
    if(event.key==='Escape'){
      if(document.fullscreenElement)return;
      if(isFullscreen()){setFsState(false);return;}
      if(!$('#compare').classList.contains('hidden'))closeCompare();
      else if(!$('#settings').classList.contains('hidden'))closeSettings();
      else if(present){const wasSlideshow=present.kind==='slideshow';stopPresenting();if(wasSlideshow)backToPicker();}
      else if(current)backToPicker();
    }
    if(event.key===' '&&current&&(event.target===document.body||isFullscreen())){event.preventDefault();playing?stopPlay():startPlay(frame);}
    if((event.key==='f'||event.key==='F')&&current&&!event.target.closest('input,textarea,select')&&$('#settings').classList.contains('hidden'))toggleFullscreen();
  });
  ['pointerdown','keydown','wheel'].forEach(type=>document.addEventListener(type,resetIdle,{passive:true}));
  window.addEventListener('resize',()=>{if(arena)arena.resize();if(fusion)fusion.resize();if(galaxy)galaxy.resize();});
}

init();
