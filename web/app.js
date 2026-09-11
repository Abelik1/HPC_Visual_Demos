let specs=null,current=null,runId=null,timer=null,lastFrame=-1;
let playbackTimer=null,playbackFrame=0,playbackTotal=0,playbackPlaying=false;
let zoom=1,panX=0,panY=0,dragState=null;
let deep=null,deepActive=false,deepManifest=null;
let fusion=null,fusionActive=false,fusionManifest=null,fusionEntering=false,preferFusion3d=true;
let galaxy3d=null,galaxy3dActive=false,galaxy3dManifest=null;
let neuralTarget={kind:0,custom:false};
let activeViewMode='frames',overlayEnabled=new Set(),frameOverlay={},currentStory='',currentMeta={};
let fluidBuilder=null;
const FRAME_INTERVAL_MS=140;
const $=s=>document.querySelector(s);
const stories={
 black_hole:["Start with a real Hubble Deep Field source plane.","Now bend those same galaxy sight lines around the chosen well or wells.","The 2D observer image scans over the source plane while the 3D rays advance through it.","Then pull back: one observer was only the beginning."],
 pbh:["The Universe is extremely young.","Increase one small density perturbation.","Below the threshold it disperses; above it, collapse accelerates.","Pull back and map the boundary using many universes."],
 fluid:["Begin with smooth flow.","Put an obstacle in the stream.","Watch coherent vortices form and interact.","Then reveal the enormous lattice being updated underneath."],
 cosmic_web:["Begin almost uniform.","Let gravity amplify tiny fluctuations.","Clusters, filaments and voids emerge.","Pull back: repeat the experiment with different initial universes."],
 galaxy_collision:["Two calm galaxies approach.","Their gravity creates long tidal tails.","The final shape remembers the orbit.","Then reveal an atlas of many possible encounters."],
 galaxy_collision_3d:["Two illustrative 3D galaxy models begin on a Local Group-scale approach.","Every disc, bulge and dark-halo super-particle pulls on every other one.","Rotate the particle state to inspect tidal tails and out-of-plane debris.","This is one all-pairs realisation with catalogue-conditioned structure, not a fitted Local Group prediction."],
 reaction_diffusion:["Start from a tiny disturbance.","Only local rules are applied.","Complex global structure appears.","Pull back and reveal the entire parameter space."],
 crystal:["One microscopic seed.","Six primary arms emerge from the seed.","Each branch creates smaller branches of its own.","Zoom in: the same growth rule repeats at several scales."],
 neural_wall:["This network is learning.","Its reconstruction becomes sharper.","Actually… something was left out.","We were training a wall of networks and searching for the best one."],
 fusion_plasma:["A coherent wave circles the magnetic bottle.","Luminous tracers follow drift derived from the evolving field.","Their trails expose changing toroidal and poloidal flow.","Pull back: we computed a complete reactor operating map."],
 plasma_guardian:["A disturbance pushes the plasma toward the vessel wall.","Noisy sensors report position, motion, pressure and a tearing-risk proxy.","A neural policy trains on many virtual shots and commands three coil banks.","Compare it with the uncontrolled reference: the policy keeps a safety margin."],
 weather_ensemble:["Begin from today’s global observations.","The atmosphere carries vorticity and moisture around the planet.","Tiny uncertainties grow as the forecast races five days ahead.","Pull back: one forecast was only one possible future."],
 molecular_dynamics:["Begin with one loose molecular chain.","Every particle attracts, repels and pulls on its bonded neighbours.","The chain rearranges while temperature and solvent compete.","Pull back: we ran a virtual laboratory of independent trajectories."]
};

const viewModes={
  black_hole:[
    {id:'3d',label:'3D ray space',folder:'modes/3d'},
    {id:'frames',label:'2D observer image',folder:'frames'}
  ]
};
const panelNames={
  black_hole:'Ray-tracing readout',pbh:'Collapse readout',fluid:'Wind-tunnel instruments',
  cosmic_web:'Cosmology readout',galaxy_collision:'Encounter readout',galaxy_collision_3d:'3D gravity readout',reaction_diffusion:'Pattern readout',
  crystal:'Growth readout',neural_wall:'Network training',fusion_plasma:'Tokamak control',plasma_guardian:'Neural control training',
  weather_ensemble:'Forecast clock',molecular_dynamics:'Molecular trajectory'
};
const legends={
  black_hole:'Both modes use the same NASA Hubble Deep Field crop: 2D maps it through the configured well(s); 3D places that crop on the source plane and advances a matching ray bundle. The spin-like term is qualitative, not Kerr tracing.',
  pbh:'Brighter central density means localisation; a spreading shell means dispersion.',
  fluid:'Colour shows speed. Tracer streaks show direction; alternating wake colours expose shed vortices.',
  cosmic_web:'Brightness is density: knots are clusters, threads are filaments and dark regions are voids.',
  galaxy_collision:'Blue mass belongs to the Milky Way, warm mass to Andromeda; overlap brightens toward white.',
  galaxy_collision_3d:'Blue bodies belong to the Milky Way and orange bodies to M31. Faint particles are massive dark-halo super-particles; every visible body participates in the direct force. This is a catalogue-conditioned illustrative super-particle calculation, not a fitted equilibrium prediction of the Local Group.',
  reaction_diffusion:'Colour represents the V chemical concentration in the Gray–Scott field.',
  crystal:'Every luminous segment is generated geometry. Deep zoom recomputes smaller branches.',
  neural_wall:'The image is the network output. The model learns RGB values from pixel coordinates; it does not recognise the subject.',
  fusion_plasma:'The torus texture is the evolving field; luminous trails are passive tracers following its derived drift.',
  plasma_guardian:'A reduced, differentiable control environment: coloured coils are three aggregate commands, cyan plasma is controlled, and the red outline is an uncontrolled reference. It is not a tokamak equilibrium or disruption solver.',
  weather_ensemble:'Cloud colour combines moisture and vorticity on the simulated globe; the bright marker follows the cyclone centre.',
  molecular_dynamics:'Atoms are depth-sorted; bonds and non-bonded forces evolve the coarse-grained chain in 3D.'
};
// A reveal is an independent sweep or ensemble. Keep this choice in the main
// run panel so a visitor can deliberately request a single fast test run.
const parallelDemos=new Set([
  'black_hole','pbh','cosmic_web','galaxy_collision','reaction_diffusion',
  'crystal','fusion_plasma','weather_ensemble','molecular_dynamics'
]);
const demoInformation={
  black_hole:['Light paths through the Hubble Deep Field','The 2D observer image and 3D source plane use the same credited Hubble Deep Field crop. Move the primary well, then use binary or triple mode to see multiple deflection centres in both views.','This is a weak-field educational model, not a full Kerr geodesic or GRMHD calculation. The 2D scan moves across recorded source data; the wells remain fixed in the chosen scene. NASA Hubble Deep Field image: PIA12110.'],
  pbh:['A threshold in the young Universe','A small spherical density enhancement either spreads out or concentrates rapidly. The interesting result is the sharp boundary between those outcomes.','This is a reduced radial collapse demonstrator. It visualises critical behaviour but does not replace the project’s validated numerical-relativity solver.'],
  fluid:['Airflow around solid geometry','A D2Q9 lattice-Boltzmann solver moves density and momentum through the grid. Bounce-back cells form the selected bodies; streaks are passive particles following the computed velocity.','Choose a preset and optionally paint extra solid cells with the grid builder. Every visible custom block becomes part of the solver mask, so it changes the wake rather than merely decorating the image.'],
  cosmic_web:['How gravity grows a cosmic web','Nearly uniform matter begins with a spectrum of tiny perturbations. Gravity amplifies them into knots, filaments and voids while gas pressure changes the smallest supported structure.','Toggle comoving expansion, a qualitative dark-energy acceleration term, and a warm-dark-matter small-scale cutoff. These are exhibition-scale theory comparisons, not precision cosmological parameter inference.'],
  galaxy_collision:['The Milky Way–Andromeda encounter','Massive galaxy centres and tracer stars evolve through their mutual gravity. Tidal tails and the final remnant remember the initial orbit.','The reveal reruns the encounter across transverse-velocity uncertainty. More parallel runs sample that uncertainty more finely.'],
  galaxy_collision_3d:['A direct 3D gravitational encounter','Disc, bulge and dark-halo super-particles interact in three dimensions. Rotate the saved particle state while playback advances.','The particles represent large groups of real stars and dark matter. Softening prevents unresolved close encounters from dominating the large-scale merger.'],
  reaction_diffusion:['Complexity from two local reactions','Two diffusing chemicals follow the Gray–Scott equations. A tiny disturbance grows into spots, waves or labyrinths without a central pattern designer.','Feed and kill rates choose the pattern regime. The parallel reveal runs neighbouring parameter pairs as independent simulations.'],
  crystal:['Branching growth from one seed','A deterministic anisotropic growth rule repeatedly creates side branches. Changing symmetry or growth conditions produces a different crystal habit.','Deep zoom regenerates geometry at the requested scale; it does not enlarge a finished bitmap. The model is deliberately geometric rather than molecular ice physics.'],
  neural_wall:['A neural network as an RGB painter','Each coordinate network receives only x and y and predicts red, green and blue. It recreates an image as a continuous function without recognising the person, object or digit in it.','The wall trains several widths and learning rates together. The optional policy graph is separate from the reconstruction, and the scan head is only a progress metaphor.'],
  fusion_plasma:['Flow around a magnetic bottle','A reduced nonlinear plasma-wave field evolves on a periodic lattice and is wrapped onto a torus. Passive tracers expose drift derived from that changing field.','The default live 3D canvas follows every playback frame and can be rotated throughout. Magnetic lines are explanatory confinement geometry, not a solved equilibrium.'],
  plasma_guardian:['A learned feedback controller','Noisy state measurements feed a small policy that commands three aggregate coil banks. Cyan is the controlled plasma; the faint red outline is the uncontrolled reference.','The smooth amber island inside the cyan plasma is the tearing-risk proxy—larger means less stable. Exact sensor values and neural weights live in optional view layers, not in the simulation frame.'],
  weather_ensemble:['Why forecasts become uncertain','A reduced rotating atmosphere advects vorticity and moisture around a globe. Small changes to the initial state grow into different storm tracks.','The reveal is a genuine initial-condition ensemble. Increasing parallel runs samples more plausible futures but also increases compute time.'],
  molecular_dynamics:['Forces reshape a molecular chain','Bonded and non-bonded particles move in 3D while temperature, attraction and solvent quality compete.','This is coarse-grained molecular dynamics: each sphere represents more than one atom. The reveal runs a virtual laboratory across conditions.']
};

function escapeHtml(value){return String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
const settingLabels={width:'Output width',height:'Output height',ensemble:'Reveal simulations',radial_points:'Radial grid points',nx:'Lattice width',ny:'Lattice height',steps_per_frame:'Steps per frame',tracers:'Tracer particles',trail:'Trail length',tracer_boost:'Tracer speed boost',total_steps:'Total simulation steps',grid:'Density grid',particles:'Particles / bodies',sweep_steps:'Reveal steps',jeans_ref:'Jeans length reference',ic_amplitude:'Initial fluctuation amplitude',ic_velocity:'Initial velocity amplitude',substeps:'Solver substeps',span_gyr:'Simulated duration',softening:'Gravity softening',force_tile:'Force tile size',max_step:'Maximum gravity step',n:'Simulation grid',depth:'Growth depth',zoom_levels:'Prebuilt zoom levels',zoom_depth:'Zoom growth depth',zoom_tile:'Zoom tile pixels',zoom_max_level:'Maximum zoom level',zoom_detail_base:'Base zoom detail',zoom_detail_max:'Maximum zoom detail',networks:'Networks trained',tile:'Network tile pixels',sweep_n:'Reveal grid size',batch:'Training batch size',horizon:'Training horizon',train_updates:'Training updates',display_steps:'Displayed rollout steps',sweep_particles:'Particles per reveal'};
const settingHelp={ensemble:'Independent runs shown in the scale reveal.',particles:'Actual simulated particle/body count.',total_steps:'Fixed numerical work distributed across the saved frames.',substeps:'Numerical integration steps inside each saved frame.',span_gyr:'Physical duration represented by the complete run, in billions of years.',softening:'Minimum gravity length scale in kpc; prevents singular super-particle forces.',force_tile:'Bodies processed together by the direct-force solver.',batch:'Virtual plasma shots evaluated per training update.',horizon:'Time steps in each virtual training shot.',networks:'Independent neural networks trained and compared.',tile:'Pixel width and height reconstructed by each network.',n:'Primary simulation grid resolution.',grid:'Cells per side in the density calculation.',nx:'Horizontal lattice cells.',ny:'Vertical lattice cells.',sweep_steps:'Numerical steps used by each reveal run.',sweep_n:'Grid resolution used by each reveal run.',sweep_particles:'Particle count used by each reveal run.',width:'Native render width in pixels.',height:'Native render height in pixels.'};
function renderProfileSettings(values=null){
  const host=$('#profileSettings');if(!host||!current)return;
  const presetName=$('#profile').value,preset=specs?.profiles?.[presetName]?.[current]||{};
  const schema=specs?.profile_setting_schema?.[current]||{};
  const chosen=values&&Object.keys(values).length?values:preset;
  host.innerHTML='';
  Object.entries(schema).forEach(([key,rule])=>{
    // Reveal cardinality has its own prominent, constrained selector in the
    // run panel. Rendering it here as well would allow two conflicting values.
    if(key==='ensemble'&&parallelDemos.has(current))return;
    const value=chosen[key]??preset[key],label=settingLabels[key]||key.replaceAll('_',' ');
    const field=document.createElement('label');field.className='profileSetting';
    field.innerHTML=`<span>${escapeHtml(label)} <em>${escapeHtml(key)}</em></span><input id="s_${key}" type="number" min="${rule.min}" max="${rule.max}" step="${rule.step}" value="${value}" required><small>${escapeHtml(settingHelp[key]||'Exact profile value used by the simulation.')}</small>`;
    const input=field.querySelector('input');
    input.oninput=()=>{field.classList.toggle('changed',Number(input.value)!==Number(preset[key]));updateProfileEditorSummary();};
    field.classList.toggle('changed',Number(value)!==Number(preset[key]));host.appendChild(field);
  });
  updateProfileEditorSummary();
}
function updateProfileEditorSummary(){
  const changed=document.querySelectorAll('#profileSettings .profileSetting.changed').length;
  $('#profileEditorSummary').textContent=changed?`${changed} custom value${changed===1?'':'s'}`:`Editable ${$('#profile').selectedOptions[0]?.textContent||$('#profile').value} preset`;
}
function collectProfileSettings(){
  const result={},schema=specs?.profile_setting_schema?.[current]||{};
  for(const key of Object.keys(schema)){
    if(key==='ensemble'&&parallelDemos.has(current))continue;
    const input=$('#s_'+key);
    if(!input||!input.checkValidity())throw new Error(`${settingLabels[key]||key} must be between ${schema[key].min} and ${schema[key].max}`);
    result[key]=Number(input.value);
  }
  return result;
}
function currentParameters(){let out={};if(!current||!specs?.demos?.[current])return out;Object.keys(specs.demos[current].params).forEach(k=>{let input=$('#p_'+k);if(input)out[k.replaceAll('_',' ')]=input.type==='checkbox'?(input.checked?'on':'off'):input.value;});return out;}
function overlayCard(id,title,kind='text'){
  let layer=$('#overlayLayer'),card=[...layer.children].find(node=>node.dataset.overlayCard===id);
  if(!card){card=document.createElement('section');card.className='overlayCard';card.dataset.overlayCard=id;let heading=document.createElement('h3');card.appendChild(heading);let body=document.createElement(kind==='rows'?'div':kind==='image'?'img':'p');if(kind==='rows')body.className='overlayRows';if(kind==='image'){card.classList.add('visualOverlay');body.alt='Live neural network connections';}card.appendChild(body);layer.appendChild(card);}
  card.querySelector('h3').textContent=title;return card;
}
function updateOverlayRows(host,values){
  let entries=Object.entries(values||{});if(!entries.length)entries=[['status','waiting for frame']];
  let wanted=new Set(entries.map(([key])=>key));[...host.children].forEach(row=>{if(!wanted.has(row.dataset.key))row.remove();});
  entries.forEach(([key,value])=>{let row=[...host.children].find(node=>node.dataset.key===key);if(!row){row=document.createElement('div');row.className='overlayRow';row.dataset.key=key;row.append(document.createElement('span'),document.createElement('b'));host.appendChild(row);}row.children[0].textContent=key;row.children[1].textContent=value;});
}
function renderOverlayCards(){
  let layer=$('#overlayLayer');if(!layer)return;let wanted=new Set();
  if(overlayEnabled.has('story')&&currentStory){wanted.add('story');let card=overlayCard('story','What is happening');card.querySelector('p').textContent=currentStory;}
  if(overlayEnabled.has('data')){wanted.add('data');let card=overlayCard('data',panelNames[current]||'Live data','rows');let values=Object.keys(frameOverlay||{}).length?frameOverlay:currentParameters();updateOverlayRows(card.querySelector('.overlayRows'),values);}
  if(overlayEnabled.has('legend')){wanted.add('legend');let card=overlayCard('legend','How to read this view');card.querySelector('p').textContent=legends[current]||'Scientific simulation output.';}
  if((current==='neural_wall'||current==='plasma_guardian')&&overlayEnabled.has('network')&&runId){wanted.add('network');let card=overlayCard('network',current==='plasma_guardian'?'Live policy graph':'Winning network','image');let img=card.querySelector('img'),src=`/runs/${runId}/overlays/network/frame_${String(playbackFrame).padStart(4,'0')}.jpg`;if(img.dataset.src!==src){img.dataset.src=src;img.src=src;}}
  window.GameDemos?.overlayCards(wanted);
  [...layer.children].forEach(card=>{if(!wanted.has(card.dataset.overlayCard))card.remove();});
}
function renderViewerDock(){let modes=$('#modeControls'),controls=$('#overlayControls');if(!modes||!controls)return;modes.innerHTML='';let defs=viewModes[current]||[];if(defs.length){modes.innerHTML='<h4>SIMULATION VIEW</h4>';defs.forEach(def=>{let b=document.createElement('button');b.textContent=def.label;b.setAttribute('aria-label',def.label);b.classList.toggle('selected',activeViewMode===def.id);b.setAttribute('aria-pressed',String(activeViewMode===def.id));b.onclick=()=>{activeViewMode=def.id;renderViewerDock();if(frameAvailable())showFrame(playbackFrame,playbackTotal);};modes.appendChild(b);});}controls.innerHTML='<h4>OPTIONAL OVERLAYS</h4>';let items=[['story','Explanation'],['data',panelNames[current]||'Live data'],['legend','Legend / method']];if(current==='neural_wall'||current==='plasma_guardian')items.push(['network',current==='plasma_guardian'?'Policy graph':'Network graph']);window.GameDemos?.dockItems(items);items.forEach(([id,label])=>{let b=document.createElement('button');b.textContent=label;b.setAttribute('aria-label',label);b.classList.toggle('selected',overlayEnabled.has(id));b.setAttribute('aria-pressed',String(overlayEnabled.has(id)));b.onclick=()=>{overlayEnabled.has(id)?overlayEnabled.delete(id):overlayEnabled.add(id);renderViewerDock();renderOverlayCards();};controls.appendChild(b);});renderOverlayCards();}
async function loadFrameOverlay(frame){if(!runId)return;try{let response=await fetch(`/runs/${runId}/frame_data/frame_${String(frame).padStart(4,'0')}.json?t=${Date.now()}`);if(!response.ok)return;let body=await response.json();if(frame===playbackFrame){frameOverlay=body.values||{};renderOverlayCards();}}catch(_){}}
function showUiMessage(message){currentStory=String(message);overlayEnabled.add('story');renderViewerDock();}

async function init(){specs=await (await fetch('/api/specs')).json();renderGallery();applyBackends();applyMethods();bindTimelineControls();await loadLibrary();let query=new URLSearchParams(location.search);let requestedRun=query.get('run'),requestedDemo=query.get('demo');let saved=requestedRun&&library.find(item=>item.id===requestedRun);if(saved)openRun(saved);else if(requestedDemo&&specs.demos[requestedDemo])openDemo(requestedDemo);}

// ---- compute backend ------------------------------------------------
function applyBackends(){
  let b=specs.backends||{},sel=$('#backend');
  let allowed=current&&specs.capabilities?.[current]?.backends||['cpu','gpu','hybrid'];
  let gpu=b.gpu||{available:false,detail:'unknown'};
  let opt=[...sel.options].find(o=>o.value==='gpu');
  if(opt){
    opt.disabled=!gpu.available||!allowed.includes('gpu');
    opt.textContent=!allowed.includes('gpu')?'GPU (not used by this demo)':gpu.available?'GPU':'GPU (unavailable)';
    opt.title=gpu.detail||'';
  }
  let hybrid=[...sel.options].find(o=>o.value==='hybrid'), h=b.hybrid||{available:false,detail:'unknown'};
  if(hybrid){
    hybrid.disabled=!h.available||!allowed.includes('hybrid');
    hybrid.textContent=!allowed.includes('hybrid')?'GPU + CPU pipeline (not used)':h.available?'GPU + CPU pipeline':'GPU + CPU pipeline (unavailable)';
    hybrid.title=h.detail||'';
  }
  sel.title=`CPU: ${(b.cpu||{}).detail||''}
GPU: ${gpu.detail||''}`;
  $('#backendPill').textContent=gpu.available?`GPU READY · ${gpu.detail}`:'CPU ONLY · NO CUDA DEVICE';
  if(sel.selectedOptions[0]?.disabled)sel.value='cpu';
}

// Solver selection is independent of where the array operations execute.
function applyMethods(){
  let capability=current&&specs.capabilities?.[current]||{};
  let methods=capability.methods||['default'],sel=$('#method'),control=$('#methodControl');
  sel.innerHTML='';
  methods.forEach(method=>{let option=document.createElement('option');option.value=method;option.textContent=capability.method_labels?.[method]||method.replaceAll('_',' ');sel.appendChild(option);});
  sel.value=capability.default_method||methods[0];
  control.classList.toggle('hidden',methods.length<2);
  let describe=()=>{let description=capability.method_descriptions?.[sel.value]||'';$('#methodHelp').textContent=description;sel.title=description;};
  sel.onchange=describe;describe();
}

function isCollisionDemo(){return current==='galaxy_collision'||current==='galaxy_collision_3d';}
function updateTimelineHelp(){
  const frames=Math.max(1,Number($('#frames').value)||70);
  let text=`${frames} saved frames`;
  if(isCollisionDemo()){
    const intervals=current==='galaxy_collision_3d'?Math.max(1,frames-1):frames;
    text=`≈${(7500/intervals).toFixed(frames>=200?0:1)} Myr between saved frames`;
  }
  $('#timelineHelp').textContent=text;
  const select=$('#timelineDetail'),match=[...select.options].find(option=>Number(option.value)===frames);
  if(match)select.value=match.value;
  else {let custom=[...select.options].find(option=>option.value==='custom');if(!custom){custom=document.createElement('option');custom.value='custom';custom.hidden=true;custom.textContent='Custom';select.appendChild(custom);}select.value='custom';}
}
function configureParallelControl(selectedValue=null){
  const control=$('#parallelControl'),select=$('#parallelCount');
  const enabled=parallelDemos.has(current);
  control.classList.toggle('hidden',!enabled);
  if(!enabled)return;
  const profileValue=Number(selectedValue??specs?.profiles?.[$('#profile').value]?.[current]?.ensemble);
  if([...select.options].some(option=>Number(option.value)===profileValue))select.value=String(profileValue);
}
function updateNumericalStepControl(){
  const control=$('#numericalStepControl');
  control.classList.add('hidden');
}
function bindTimelineControls(){
  $('#timelineDetail').onchange=event=>{$('#frames').value=event.target.value;updateTimelineHelp();};
  $('#frames').oninput=updateTimelineHelp;
  $('#profile').onchange=()=>{applyBackends();renderProfileSettings();configureParallelControl();};
  updateTimelineHelp();
}

// ---- saved run library ----------------------------------------------
let library=[],showAllRuns=false;
async function loadLibrary(){
  try{library=await (await fetch('/api/runs?limit=120')).json();}catch(e){library=[];}
  renderLibrary();renderStageRuns();
}
function runLabel(r){
  let d=new Date((r.created||0)*1000);
  return isNaN(d)?r.id:d.toLocaleString(undefined,{month:'short',day:'numeric',hour:'2-digit',minute:'2-digit'});
}
function resourceLabel(r){
  let resources=r.resources||{},gpu=resources.gpu?.visible_devices?.[0]?.name;
  return [resources.host,gpu].filter(Boolean).join(' · ');
}
function renderLibrary(){
  let el=$('#libList');if(!el)return;el.innerHTML='';
  if(!library.length){$('#libMore').classList.add('hidden');el.innerHTML='<p style="color:#7d93b6;font-size:14px">No saved runs yet. Run a simulation and it will appear here.</p>';return;}
  $('#libMore').classList.toggle('hidden',library.length<=8);
  $('#libMore').textContent=showAllRuns?'Show fewer saved runs':`Show all ${library.length} saved runs`;
  (showAllRuns?library:library.slice(0,8)).forEach(r=>{
    let name=(specs.demos[r.demo]||{}).name||r.demo;
    let card=document.createElement('article');card.className='runCard';
    let gpu=/cupy|cuda/i.test(r.backend||'');
    card.innerHTML=`<a class="runLink" href="/?run=${encodeURIComponent(r.id)}"><img loading="lazy" src="${escapeHtml(r.thumb)}" alt="">
      <div class=runMeta><b>${escapeHtml(name)}</b><span>${runLabel(r)} · ${r.frames} frames</span>
      <div class=runTags><i>${r.profile==='hpc'?'HPC':r.profile||'?'}</i><i class="${gpu?'gpu':''}">${r.backend||'?'}</i>${r.method?`<i>${r.method.replaceAll('_',' ')}</i>`:''}${r.zoom?'<i>zoom</i>':''}</div></div></a>`;
    card.querySelector('a').onclick=event=>{if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;event.preventDefault();openRun(r);};
    el.appendChild(card);
  });
}
function renderStageRuns(){
  let el=$('#stageRuns');if(!el)return;el.innerHTML='';
  let mine=library.filter(r=>r.demo===current);
  if(!mine.length)return;
  let head=document.createElement('span');head.className='chip';head.style.cursor='default';
  head.innerHTML='<em>Replay:</em>';el.appendChild(head);
  mine.slice(0,12).forEach(r=>{
    let c=document.createElement('button');c.className='chip'+(r.id===runId?' active':'');
    c.innerHTML=`${runLabel(r)} <em>${r.backend||''}</em>`;
    c.onclick=()=>openRun(r);el.appendChild(c);
  });
}

// Replay a finished run straight from disk: no recomputation, and every
// control (playback, reveal, deep zoom) behaves as it does after a live run.
function openRun(r){
  if((r.demo!==current||$('#stage').classList.contains('hidden'))&&openDemo(r.demo)===false)return;
  resetRunState();
  runId=r.id;lastFrame=r.frames-1;playbackTotal=r.frames;
  history.replaceState(null,'',`/?run=${encodeURIComponent(r.id)}`);
  currentMeta=r;frameOverlay={};
  deepManifest=r.zoom||null;
  fusionManifest=r.fusion_view||null;
  galaxy3dManifest=r.galaxy3d_view||null;
  $('#frameSeek').max=Math.max(0,r.frames-1);
  $('#status').textContent='REPLAY';$('#status').style.color='#ffc46b';
  $('#metric2').textContent=`elapsed ${(r.elapsed||0).toFixed(1)} s`;
  let machine=resourceLabel(r);
  $('#metric3').textContent=`backend ${r.backend||'—'}${machine?` · ${machine}`:''}`;
  if(r.profile&&[...$('#profile').options].some(option=>option.value===r.profile))$('#profile').value=r.profile;
  renderProfileSettings(r.settings||null);
  configureParallelControl(r.params?._parallel_count);
  Object.entries(r.params||{}).forEach(([k,v])=>{
    let inp=$('#p_'+k);if(inp){if(inp.type==='checkbox')inp.checked=Boolean(Number(v));else inp.value=v;let out=$('#v_'+k);if(out)out.textContent=v;}
  });
  $('#frames').value=r.frames;updateTimelineHelp();
  if(r.method&&[...$('#method').options].some(option=>option.value===r.method)){$('#method').value=r.method;$('#method').dispatchEvent(new Event('change'));}
  setPlaybackControls(true);
  $('#reveal').disabled=!r.has_reveal;
  startPlayback(0);
  if(current==='fusion_plasma'&&fusionManifest&&preferFusion3d)enterFusion();
  window.GameDemos?.onOpenRun(r);
  renderStageRuns();
}
$('#libRefresh').onclick=loadLibrary;
$('#libMore').onclick=()=>{showAllRuns=!showAllRuns;renderLibrary();if(!showAllRuns)$('#library').scrollIntoView({block:'start'});};
const demoCategories={black_hole:'Universe',pbh:'Universe',cosmic_web:'Universe',galaxy_collision:'Universe',galaxy_collision_3d:'Universe',fluid:'Physics',fusion_plasma:'Physics',reaction_diffusion:'Patterns & life',crystal:'Patterns & life',molecular_dynamics:'Patterns & life',weather_ensemble:'Physics',neural_wall:'AI & learning',plasma_guardian:'AI & learning'};
// Demos that still run but are not ready for visitors. They are hidden from
// the main gallery and listed only under the Archive tab; saved runs and
// direct ?demo= links keep working.
const archivedDemos=new Set(['pbh','crystal','molecular_dynamics','weather_ensemble','reaction_diffusion']);
const ARCHIVE='Archive';
let galleryCategory='All experiments';
function renderGallery(){
  const host=$('#gallery'),filters=$('#categoryFilters'),query=$('#demoSearch').value.trim().toLowerCase();
  const archiveView=galleryCategory===ARCHIVE;
  host.innerHTML='';filters.innerHTML='';
  // A category whose demos are all archived would only ever show an empty grid.
  const activeCategories=new Set(Object.keys(specs.demos).filter(id=>!archivedDemos.has(id)).map(id=>demoCategories[id]||'Physics'));
  ['All experiments','Universe','Physics','Patterns & life','AI & learning'].filter(category=>category==='All experiments'||activeCategories.has(category)).concat(ARCHIVE).forEach(category=>{
    const button=document.createElement('button');button.textContent=category;
    if(category===ARCHIVE){button.classList.add('archiveTab');button.title='Demos that are not ready for visitors yet';}
    button.classList.toggle('selected',category===galleryCategory);button.setAttribute('aria-pressed',String(category===galleryCategory));
    button.onclick=()=>{galleryCategory=category;renderGallery();[...filters.children].find(item=>item.textContent===category)?.focus();};filters.appendChild(button);
  });
  let count=0;
  Object.entries(specs.demos).forEach(([id,d])=>{
    const category=demoCategories[id]||'Physics',archived=archivedDemos.has(id);
    if(archived!==archiveView)return;
    if(!archiveView&&galleryCategory!=='All experiments'&&galleryCategory!==category)return;
    if(query&&!`${d.name} ${d.tagline} ${category}`.toLowerCase().includes(query))return;
    count++;
    const card=document.createElement('article');card.className='card'+(archived?' archived':'');
    card.innerHTML=`<a class="cardLink" href="/?demo=${encodeURIComponent(id)}"><div class="cardImage"><img src="/static/previews/${encodeURIComponent(id)}.webp" alt="" loading="lazy" width="800" height="450"><span class="num">${String(count).padStart(2,'0')}</span>${archived?'<span class="archiveBadge">Archived</span>':''}</div><div class="cardBody"><span class="cardCategory">${escapeHtml(category)}</span><h3>${escapeHtml(d.name)}</h3><p>${escapeHtml(d.tagline)}</p><div class="go">${archived?'Open work in progress':'Explore simulation'} <span aria-hidden="true">↗</span></div></div></a>`;
    card.querySelector('a').onclick=event=>{if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;event.preventDefault();openDemo(id);};host.appendChild(card);
  });
  $('#demoCount').textContent=Object.keys(specs.demos).filter(id=>!archivedDemos.has(id)).length;
  $('#galleryResults').textContent=archiveView?`${count} archived experiment${count===1?'':'s'} · not yet ready for visitors`:`${count} experiment${count===1?'':'s'}${galleryCategory==='All experiments'?' to explore':` in ${galleryCategory.toLowerCase()}`}`;
  if(!count)host.innerHTML=`<p class="galleryEmpty">${archiveView?'No archived experiments match.':'No experiments found. Try another search or category.'}</p>`;
}
$('#demoSearch').oninput=renderGallery;
$('#previewReplay').onclick=()=>{const latest=library.find(item=>item.demo===current);if(latest)openRun(latest);};
function hideReveal(){let sw=$('.screenWrap');sw.classList.remove('revealing');$('#scaleReveal').classList.remove('show');$('#screen').style.opacity=1;}
function stopPlayback(){if(playbackTimer)clearInterval(playbackTimer);playbackTimer=null;playbackPlaying=false;updatePlaybackButton();}
function updatePlaybackButton(){$('#playPause').textContent=playbackPlaying?'Pause':'Play';}
function setPlaybackControls(enabled){$('#playPause').disabled=!enabled;$('#playbackRate').disabled=!enabled;$('#frameSeek').disabled=!enabled;$('#reveal').disabled=!enabled;$('#deepZoom').disabled=!(enabled&&deepManifest);$('#fusionView').disabled=!(enabled&&fusionManifest&&current==='fusion_plasma');$('#galaxy3dView').disabled=!(enabled&&galaxy3dManifest&&current==='galaxy_collision_3d');updatePlaybackButton();}
function frameAvailable(){return Boolean(runId&&lastFrame>=0);}
function updateViewport(){let wrap=$('.screenWrap');wrap.style.setProperty('--view-zoom',zoom);wrap.style.setProperty('--view-pan-x',`${panX}px`);wrap.style.setProperty('--view-pan-y',`${panY}px`);wrap.classList.toggle('isZoomed',zoom>1);let enabled=frameAvailable()&&!deepActive&&!fusionActive&&!galaxy3dActive&&!window.GameDemos?.active;$('#zoomIn').disabled=!enabled;$('#zoomOut').disabled=!enabled||zoom<=1;$('#zoomReset').disabled=!enabled||zoom===1;$('#zoomReset').textContent=`${zoom.toFixed(zoom%1?1:0)}×`;}
function resetViewport(){zoom=1;panX=0;panY=0;updateViewport();}
function changeZoom(amount){if(deepActive||!frameAvailable())return;zoom=Math.max(1,Math.min(8,Math.round((zoom+amount)*10)/10));let wrap=$('.screenWrap');let limitX=wrap.clientWidth*(zoom-1)/2;let limitY=wrap.clientHeight*(zoom-1)/2;panX=Math.max(-limitX,Math.min(limitX,panX));panY=Math.max(-limitY,Math.min(limitY,panY));updateViewport();}
function fusionFrameUrl(frame){if(!fusionManifest)return null;if(typeof fusionManifest==='string')return `/runs/${runId}/${fusionManifest}`;return `/runs/${runId}/${fusionManifest.folder}/frame_${String(frame).padStart(4,'0')}.json`;}
function showFrame(frame,total=playbackTotal){if(!runId||total<1)return;frame=Math.max(0,Math.min(total-1,Math.trunc(frame)));playbackFrame=frame;hideReveal();$('.screenWrap').classList.remove('empty');let mode=(viewModes[current]||[]).find(item=>item.id===activeViewMode),folder=mode?.folder||'frames',image=$('#screen'),fallback=`/runs/${runId}/frames/frame_${String(frame).padStart(4,'0')}.jpg?t=${Date.now()}`;image.onerror=()=>{image.onerror=null;image.src=fallback;};image.src=`/runs/${runId}/${folder}/frame_${String(frame).padStart(4,'0')}.jpg?t=${Date.now()}`;if(galaxy3dActive&&galaxy3d)galaxy3d.load(`/runs/${runId}/${galaxy3dManifest.folder}/frame_${String(frame).padStart(4,'0')}.json?t=${Date.now()}`).catch(error=>showUiMessage(`3D frame unavailable: ${error.message}`));if(fusionActive&&fusion){let url=fusionFrameUrl(frame);if(url)fusion.load(`${url}?t=${Date.now()}`,true).catch(error=>showUiMessage(`3D plasma frame unavailable: ${error.message}`));}let p=(frame+1)/total;$('#bar').style.width=(p*100)+'%';$('#frameSeek').value=frame;$('#metric1').textContent=`frame ${frame+1}/${total}`;currentStory=stories[current][Math.min(3,Math.floor(p*4))];renderOverlayCards();loadFrameOverlay(frame);window.GameDemos?.onFrame(frame);updateViewport();}
function startPlayback(frame=playbackFrame){if(!runId||playbackTotal<1)return;stopPlayback();showFrame(frame,playbackTotal);playbackPlaying=true;updatePlaybackButton();let rate=Number($('#playbackRate').value);playbackTimer=setInterval(()=>showFrame((playbackFrame+1)%playbackTotal,playbackTotal),Math.max(40,FRAME_INTERVAL_MS/rate));}
function resetRunState(){window.GameDemos?.reset();if(timer)clearInterval(timer);timer=null;stopTargetCamera();stopPlayback();lastFrame=-1;playbackFrame=0;playbackTotal=0;exitFusion();exitGalaxy3d();exitDeep();runId=null;deepManifest=null;fusionManifest=null;galaxy3dManifest=null;currentMeta={};frameOverlay={};hideReveal();clearSimulationSurface();$('#frameSeek').max=0;$('#frameSeek').value=0;setPlaybackControls(false);resetViewport();renderOverlayCards();}
function stopTargetCamera(){if(neuralTarget.stream){neuralTarget.stream.getTracks().forEach(track=>track.stop());neuralTarget.stream=null;}}
function addNeuralTargetTools(host,defaultKind){
  stopTargetCamera();
  neuralTarget={kind:Number(defaultKind),custom:false};
  const el=document.createElement('div');el.className='targetTools';
  el.innerHTML=`<div class="targetHeading"><span>RGB training target</span><small>Use a colour preset, draw, upload a photo, or take a local webcam picture.</small></div><div class="targetBody"><div class="presetButtons"><button type="button" data-kind="0">Nebula</button><button type="button" data-kind="1">RGB waves</button><button type="button" data-kind="2">Flower</button><button type="button" data-kind="3">Ribbon</button></div><div class="drawing"><canvas id="targetCanvas" width="128" height="128" aria-label="Draw a target image"></canvas><video id="targetCamera" class="hidden" autoplay muted playsinline></video><div><button type="button" id="clearDrawing">Clear</button><label class="targetUpload">Upload<input id="targetUpload" type="file" accept="image/*"></label><button type="button" id="openCamera">Camera</button><button type="button" id="captureCamera" class="hidden">Capture</button><span id="targetMode">Preset target</span></div></div></div>`;
  host.appendChild(el);
  const canvas=el.querySelector('#targetCanvas'),ctx=canvas.getContext('2d');
  const clear=()=>{ctx.fillStyle='#000';ctx.fillRect(0,0,canvas.width,canvas.height);};
  function choose(kind){neuralTarget={kind:Number(kind),custom:false};el.querySelectorAll('[data-kind]').forEach(b=>b.classList.toggle('selected',Number(b.dataset.kind)===neuralTarget.kind));el.querySelector('#targetMode').textContent=`Preset ${neuralTarget.kind+1} selected`;}
  clear();choose(defaultKind);
  el.querySelectorAll('[data-kind]').forEach(button=>button.onclick=()=>choose(button.dataset.kind));
  el.querySelector('#clearDrawing').onclick=()=>{clear();choose(neuralTarget.kind);};
  function useCustom(label){neuralTarget.custom=true;el.querySelectorAll('[data-kind]').forEach(b=>b.classList.remove('selected'));el.querySelector('#targetMode').textContent=label;}
  // Photos are rarely square: take the largest centred square instead of
  // stretching the whole frame onto the square training canvas.
  const drawCentreCrop=(source,width,height)=>{const side=Math.min(width,height);clear();ctx.drawImage(source,(width-side)/2,(height-side)/2,side,side,0,0,canvas.width,canvas.height);};
  el.querySelector('#targetUpload').onchange=event=>{const file=event.target.files&&event.target.files[0];if(!file)return;const image=new Image();image.onload=()=>{drawCentreCrop(image,image.naturalWidth,image.naturalHeight);useCustom('Photo selected — centre-cropped RGB target');URL.revokeObjectURL(image.src);};image.src=URL.createObjectURL(file);};
  const video=el.querySelector('#targetCamera'),capture=el.querySelector('#captureCamera');
  el.querySelector('#openCamera').onclick=async()=>{try{stopTargetCamera();neuralTarget.stream=await navigator.mediaDevices.getUserMedia({video:{facingMode:'user',width:{ideal:640},height:{ideal:480}},audio:false});video.srcObject=neuralTarget.stream;video.classList.remove('hidden');capture.classList.remove('hidden');el.querySelector('#targetMode').textContent='Camera live — capture when ready';}catch(error){el.querySelector('#targetMode').textContent='Camera unavailable — upload a photo instead';}};
  capture.onclick=()=>{if(!video.videoWidth)return;drawCentreCrop(video,video.videoWidth,video.videoHeight);stopTargetCamera();video.classList.add('hidden');capture.classList.add('hidden');useCustom('Camera photo selected — RGB target');};
  let drawing=false,last=null;
  const point=event=>{const r=canvas.getBoundingClientRect();return {x:(event.clientX-r.left)*canvas.width/r.width,y:(event.clientY-r.top)*canvas.height/r.height};};
  function paint(event){const p=point(event);ctx.strokeStyle='#fff';ctx.lineCap='round';ctx.lineJoin='round';ctx.lineWidth=9;ctx.beginPath();ctx.moveTo(last.x,last.y);ctx.lineTo(p.x,p.y);ctx.stroke();last=p;useCustom('Your drawing selected — RGB target');}
  canvas.addEventListener('pointerdown',event=>{drawing=true;last=point(event);canvas.setPointerCapture(event.pointerId);paint(event);});
  canvas.addEventListener('pointermove',event=>{if(drawing)paint(event);});
  canvas.addEventListener('pointerup',()=>{drawing=false;last=null;});canvas.addEventListener('pointercancel',()=>{drawing=false;last=null;});
}
function addFluidBuilder(host){
  const rows=12,cols=24,cells=Array.from({length:rows},()=>Array(cols).fill(0));fluidBuilder={rows,cols,cells,tool:'cell'};
  let box=document.createElement('div');box.className='obstacleBuilder';box.innerHTML=`<div><div class="builderHeading"><b>Custom obstacle grid</b><small>Paint additional solid bounce-back cells. The preset and your blocks are combined in the solver.</small></div><canvas class="builderCanvas" width="720" height="360" aria-label="Wind tunnel obstacle grid"></canvas></div><div class="builderTools"><button data-tool="cell" class="selected">Cell</button><button data-tool="block">2×2 block</button><button data-tool="hbar">Horizontal bar</button><button data-tool="vbar">Vertical bar</button><button data-tool="erase">Eraser</button><button data-tool="clear">Clear grid</button><span>Drag to paint. Keep the left and right edges open so air can enter and leave.</span></div>`;host.appendChild(box);
  let canvas=box.querySelector('canvas'),ctx=canvas.getContext('2d');
  const draw=()=>{let cw=canvas.width/cols,ch=canvas.height/rows;ctx.fillStyle='#06101e';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.strokeStyle='rgba(91,145,188,.28)';for(let x=0;x<=cols;x++){ctx.beginPath();ctx.moveTo(x*cw,0);ctx.lineTo(x*cw,canvas.height);ctx.stroke();}for(let y=0;y<=rows;y++){ctx.beginPath();ctx.moveTo(0,y*ch);ctx.lineTo(canvas.width,y*ch);ctx.stroke();}ctx.fillStyle='#8ceaff';for(let y=0;y<rows;y++)for(let x=0;x<cols;x++)if(cells[y][x])ctx.fillRect(x*cw+1,y*ch+1,cw-2,ch-2);};
  const shapes={cell:[[0,0]],block:[[0,0],[1,0],[0,1],[1,1]],hbar:[[0,0],[1,0],[2,0],[3,0]],vbar:[[0,0],[0,1],[0,2],[0,3]]};
  const paint=event=>{let r=canvas.getBoundingClientRect(),x=Math.floor((event.clientX-r.left)/r.width*cols),y=Math.floor((event.clientY-r.top)/r.height*rows),shape=fluidBuilder.tool==='erase'?shapes.cell:shapes[fluidBuilder.tool];for(let [dx,dy] of shape||[]){let xx=x+dx,yy=y+dy;if(xx>=1&&xx<cols-1&&yy>=0&&yy<rows)cells[yy][xx]=fluidBuilder.tool==='erase'?0:1;}draw();};
  let drawing=false;canvas.onpointerdown=e=>{drawing=true;canvas.setPointerCapture(e.pointerId);paint(e);};canvas.onpointermove=e=>{if(drawing)paint(e);};canvas.onpointerup=canvas.onpointercancel=()=>{drawing=false;};
  box.querySelectorAll('[data-tool]').forEach(button=>button.onclick=()=>{let tool=button.dataset.tool;if(tool==='clear'){cells.forEach(row=>row.fill(0));draw();return;}fluidBuilder.tool=tool;box.querySelectorAll('[data-tool]').forEach(b=>b.classList.toggle('selected',b===button));});draw();
}
function renderDemoInfo(){let info=demoInformation[current]||['Scientific simulation',stories[current]?.[0]||'',''];$('#infoTitle').textContent=info[0];$('#infoSummary').textContent=info[1];$('#infoMethod').textContent=info[2];}
function clearSimulationSurface(){let wrap=$('.screenWrap'),image=$('#screen');image.onerror=null;image.onload=null;image.removeAttribute('src');image.style.opacity='';wrap.classList.add('empty');}
function addParameterControl(host,key,param){let el=document.createElement('div'),label=param.label||key.replaceAll('_',' ').replace(/^./,letter=>letter.toUpperCase());el.className='control';if(param.kind==='toggle'){el.classList.add('toggleControl');el.innerHTML=`<label for="p_${key}">${escapeHtml(label)}</label><input id="p_${key}" type="checkbox" ${Number(param.value)?'checked':''}>`;el.querySelector('input').onchange=renderOverlayCards;}else if(param.kind==='choice'){let options=Object.entries(param.options||{}).map(([value,text])=>`<option value="${escapeHtml(value)}" ${Number(value)===Number(param.value)?'selected':''}>${escapeHtml(text)}</option>`).join('');el.innerHTML=`<div class=row><label for="p_${key}">${escapeHtml(label)}</label></div><select id="p_${key}">${options}</select>`;el.querySelector('select').onchange=renderOverlayCards;}else{el.innerHTML=`<div class=row><label for="p_${key}">${escapeHtml(label)}</label><b id="v_${key}">${param.value}</b></div><input id="p_${key}" type=range min="${param.min}" max="${param.max}" step="${param.step}" value="${param.value}">`;el.querySelector('input').oninput=e=>{$('#v_'+key).textContent=e.target.value;renderOverlayCards();};}host.appendChild(el);}
function parameterValue(key){let input=$('#p_'+key);return input?.type==='checkbox'?Number(input.checked):Number(input?.value||0);}
function openDemo(id){let spec=specs&&specs.demos?specs.demos[id]:null;
  // The collision has rapidly changing pericentre frames; make smooth sampling
  // the default while preserving an explicit lower-cost option in the UI.
  if(id==='galaxy_collision'&&Number($('#frames').value)===70)$('#frames').value=140;
  // A saved run can name a demo this build no longer ships; refuse to open
  // it rather than throwing on a missing spec and leaving a dead stage.
  if(!spec){alert('This build has no demo called "'+id+'".');return false;}
  document.body.classList.add('demoOpen');
  history.replaceState(null,'',`/?demo=${encodeURIComponent(id)}`);
  $('#demoPreview').src=`/static/previews/${encodeURIComponent(id)}.webp`;
  $('#previewReplay').classList.toggle('hidden',!library.some(item=>item.demo===id));
  current=id;applyBackends();applyMethods();
  resetRunState();current=id;activeViewMode='frames';preferFusion3d=id==='fusion_plasma';overlayEnabled=new Set();currentStory=stories[id][0];$('#gallery').classList.add('hidden');$('#library').classList.add('hidden');$('#stage').classList.remove('hidden');let d=specs.demos[id];$('#fusionView').classList.toggle('hidden',id!=='fusion_plasma');$('#galaxy3dView').classList.toggle('hidden',id!=='galaxy_collision_3d');$('#stageTitle').textContent=d.name;$('#stageTag').textContent=d.tagline;$('#stageEyebrow').textContent=(demoCategories[id]||'Science')+' / INTERACTIVE SIMULATION';let s=$('#sliders');s.innerHTML='';fluidBuilder=null;Object.entries(d.params).forEach(([k,p])=>{if(id==='neural_wall'&&k==='target')return;addParameterControl(s,k,p);});if(id==='neural_wall')addNeuralTargetTools(s,d.params.target.value);if(id==='fluid')addFluidBuilder(s);window.GameDemos?.mount(s,id);renderProfileSettings();configureParallelControl();updateTimelineHelp();clearSimulationSurface();renderDemoInfo();renderStageRuns();renderViewerDock();$('#status').textContent='READY';$('#status').style.color='';$('#bar').style.width='0';$('#metric1').textContent='frame —';$('#metric2').textContent='elapsed —';$('#metric3').textContent='backend —';window.scrollTo(0,0);$('#back').focus({preventScroll:true});return true;}
$('#back').onclick=()=>{document.body.classList.remove('demoOpen');history.replaceState(null,'','/');resetRunState();$('#stage').classList.add('hidden');$('#gallery').classList.remove('hidden');$('#library').classList.remove('hidden');loadLibrary();window.scrollTo(0,0);$('#demoSearch').focus({preventScroll:true});};
$('#run').onclick=async()=>{if(!current)return;let settings;try{settings=collectProfileSettings();}catch(error){showUiMessage(error.message);return;}resetRunState();let ps={};Object.keys(specs.demos[current].params).forEach(k=>{if(current==='neural_wall'&&k==='target')ps[k]=neuralTarget.kind;else ps[k]=parameterValue(k);});let req={profile:$('#profile').value,frames:Number($('#frames').value),params:ps,settings,backend:$('#backend').value,method:$('#method').value};if(parallelDemos.has(current))req.parallel_count=Number($('#parallelCount').value);if(current==='fluid'&&fluidBuilder)req.obstacle_grid=fluidBuilder.cells;if(current==='neural_wall'&&neuralTarget.custom)req.target_image=$('#targetCanvas').toDataURL('image/png');window.GameDemos?.decorateRequest(req);let response=await fetch('/api/run/'+current,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(req)});if(!response.ok){let detail='Request rejected';try{let body=await response.json();detail=body.detail||detail;}catch(_){ }$('#status').textContent='FAILED TO START';showUiMessage(detail);return;}runId=(await response.json()).id;$('#status').textContent='COMPUTING';$('#status').style.color='#67f0d0';timer=setInterval(poll,300);};
async function poll(){if(!runId)return;let m=await (await fetch('/api/run/'+runId+'?t='+Date.now())).json();currentMeta=m;window.GameDemos?.onMeta(m);let total=Number($('#frames').value);if(m.fusion_view)fusionManifest=m.fusion_view;if(m.galaxy3d_view)galaxy3dManifest=m.galaxy3d_view;if(m.frame!==undefined&&m.frame>=0){lastFrame=m.frame;playbackTotal=total;frameOverlay=m.overlay||frameOverlay;$('#frameSeek').max=Math.max(0,total-1);showFrame(m.frame,total);renderOverlayCards();if(current==='fusion_plasma'&&fusionManifest&&preferFusion3d&&!fusionActive&&!fusionEntering)enterFusion();$('#metric2').textContent=`elapsed ${(m.elapsed||0).toFixed(1)} s`;$('#metric3').textContent=`backend ${m.backend||'—'}`;}if(m.status==='complete'){clearInterval(timer);timer=null;deepManifest=m.zoom||null;fusionManifest=m.fusion_view||fusionManifest;galaxy3dManifest=m.galaxy3d_view||galaxy3dManifest;playbackTotal=Number(m.frames)||total;$('#frameSeek').max=Math.max(0,playbackTotal-1);$('#status').textContent='COMPLETE';loadLibrary();$('#metric3').textContent=`backend ${m.backend||'—'}`;setPlaybackControls(true);startPlayback(0);if(current==='fusion_plasma'&&fusionManifest&&preferFusion3d&&!fusionActive)enterFusion();}if(m.status==='failed'){clearInterval(timer);timer=null;$('#status').textContent='FAILED';showUiMessage(m.error||'Simulation failed');}}
async function showReveal(){if(!runId)return;exitFusion();exitGalaxy3d();window.GameDemos?.exit();stopPlayback();let m=await (await fetch('/api/run/'+runId)).json();if(m.reveal){let sw=$('.screenWrap');sw.classList.add('revealing');$('#scaleReveal').classList.add('show');$('#screen').style.opacity=.15;setTimeout(()=>{$('#screen').src=`/runs/${runId}/${m.reveal}?t=${Date.now()}`;$('#screen').style.opacity=1;currentStory=stories[current][3];renderOverlayCards();},220);}}
const viewport=$('.screenWrap');
viewport.addEventListener('wheel',event=>{if(deepActive||fusionActive||galaxy3dActive||window.GameDemos?.active)return;if(!frameAvailable())return;event.preventDefault();let rect=viewport.getBoundingClientRect();viewport.style.setProperty('--view-origin-x',`${(event.clientX-rect.left)/rect.width*100}%`);viewport.style.setProperty('--view-origin-y',`${(event.clientY-rect.top)/rect.height*100}%`);changeZoom(event.deltaY<0?.5:-.5);},{passive:false});
viewport.addEventListener('dragstart',event=>event.preventDefault());
viewport.addEventListener('pointerdown',event=>{if(deepActive||fusionActive||galaxy3dActive||window.GameDemos?.active)return;if(zoom<=1||!frameAvailable())return;event.preventDefault();dragState={x:event.clientX,y:event.clientY,panX,panY};viewport.setPointerCapture(event.pointerId);viewport.classList.add('isPanning');});
viewport.addEventListener('pointermove',event=>{if(deepActive||fusionActive||galaxy3dActive||!dragState)return;event.preventDefault();let limitX=viewport.clientWidth*(zoom-1)/2;let limitY=viewport.clientHeight*(zoom-1)/2;panX=Math.max(-limitX,Math.min(limitX,dragState.panX+event.clientX-dragState.x));panY=Math.max(-limitY,Math.min(limitY,dragState.panY+event.clientY-dragState.y));updateViewport();});
viewport.addEventListener('pointerup',event=>{if(dragState)event.preventDefault();dragState=null;viewport.classList.remove('isPanning');});
viewport.addEventListener('pointercancel',()=>{dragState=null;viewport.classList.remove('isPanning');});
$('#playPause').onclick=()=>playbackPlaying?stopPlayback():startPlayback(playbackFrame);
$('#playbackRate').onchange=()=>{if(playbackPlaying)startPlayback(playbackFrame);};
$('#frameSeek').oninput=e=>{stopPlayback();showFrame(Number(e.target.value),playbackTotal);};
$('#zoomIn').onclick=()=>changeZoom(.5);
$('#zoomOut').onclick=()=>changeZoom(-.5);
$('#zoomReset').onclick=resetViewport;
function exitDeep(){if(deep&&deep.drag)deep.onUp({});deepActive=false;$('#deepCanvas').classList.add('hidden');$('#deepBadge').classList.add('hidden');$('#screen').classList.remove('hidden');$('#deepZoom').textContent='Deep zoom';}
function enterDeep(){
  if(!runId||!deepManifest)return;
  exitFusion();exitGalaxy3d();
  if(!deep){deep=new DeepZoom($('#deepCanvas'));deep.onstatus=(f,l,max)=>{
    let z=f<1000?f.toFixed(1):(f<1e6?(f/1e3).toFixed(1)+'k':(f<1e9?(f/1e6).toFixed(1)+'M':(f/1e9).toFixed(1)+'B'));
    $('#deepBadge').textContent=`DEEP ZOOM ${z}× · LEVEL ${l}/${max}`;};}
  stopPlayback();hideReveal();
  deepActive=true;
  $('#screen').classList.add('hidden');
  $('#deepCanvas').classList.remove('hidden');
  $('#deepBadge').classList.remove('hidden');
  $('#deepZoom').textContent='Exit deep zoom';
  currentStory='Zoom and pan are immediate; one coherent detail view refines in the background.';renderOverlayCards();
  deep.load(`/runs/${runId}/zoom`,deepManifest,`/api/zoom_view/${runId}`);
}
$('#deepZoom').onclick=()=>deepActive?exitDeep():enterDeep();
function updateFusionLayerButtons(){document.querySelectorAll('[data-fusion-layer]').forEach(button=>{let layer=button.dataset.fusionLayer;button.classList.toggle('selected',Boolean(fusion&&(layer==='plasma'?fusion.showPlasma:fusion.showMagnetic)));});}
function exitFusion(userChoice=false){if(userChoice)preferFusion3d=false;fusionActive=false;fusionEntering=false;$('#fusionCanvas').classList.add('hidden');$('#fusionTools').classList.add('hidden');$('#screen').classList.remove('hidden');$('#fusionView').textContent='3D live view';}
async function enterFusion(){
  if(!runId||!fusionManifest||current!=='fusion_plasma'||fusionEntering)return;
  fusionEntering=true;preferFusion3d=true;exitDeep();exitGalaxy3d();hideReveal();resetViewport();
  if(!fusion)fusion=new FusionView($('#fusionCanvas'));
  try{
    let url=fusionFrameUrl(playbackFrame);await fusion.load(`${url}?t=${Date.now()}`);
    fusionEntering=false;fusionActive=true;$('#screen').classList.add('hidden');$('#fusionCanvas').classList.remove('hidden');$('#fusionTools').classList.remove('hidden');$('#fusionView').textContent='2D frame';
    fusion.setLayer('plasma',true);fusion.setLayer('magnetic',false);fusion.setParticleFilter($('#fusionParticles').value);updateFusionLayerButtons();fusion.resize();
    currentStory='Drag to rotate the computed torus while playback continues. Turn on the magnetic overlay without hiding the plasma flow, and filter tracer families by colour.';renderOverlayCards();
  }catch(error){fusionEntering=false;exitFusion();showUiMessage(`Interactive view unavailable: ${error.message}`);}
}
$('#fusionView').onclick=()=>fusionActive?exitFusion(true):enterFusion();
document.querySelectorAll('[data-fusion-layer]').forEach(button=>button.onclick=()=>{if(!fusion)return;let layer=button.dataset.fusionLayer;fusion.setLayer(layer,layer==='plasma'?!fusion.showPlasma:!fusion.showMagnetic);updateFusionLayerButtons();currentStory=fusion.showMagnetic?'Helical lines now overlay the computed plasma flow. They are illustrative confinement geometry, not a solved tokamak equilibrium.':'Passive tracers follow drift derived from the computed plasma-wave field.';renderOverlayCards();});
$('#fusionParticles').onchange=()=>{if(!fusion)return;fusion.setParticleFilter($('#fusionParticles').value);currentStory=$('#fusionParticles').value==='all'?'All passive tracer families are visible.':'Only one colour family of passive tracers is visible; the underlying plasma state is unchanged.';renderOverlayCards();};
$('#fusionReset').onclick=()=>{if(fusion)fusion.reset();};
function exitGalaxy3d(){galaxy3dActive=false;$('#galaxy3dCanvas').classList.add('hidden');$('#galaxy3dTools').classList.add('hidden');$('#screen').classList.remove('hidden');$('#galaxy3dView').textContent='Rotate 3D';updateViewport();}
function updateGalaxyViewButtons(){document.querySelectorAll('[data-galaxy-focus]').forEach(button=>button.classList.toggle('selected',galaxy3d&&galaxy3d.focus===button.dataset.galaxyFocus));$('#galaxyHalo').classList.toggle('selected',Boolean(galaxy3d?.showHalo));}
async function enterGalaxy3d(){
  if(!runId||!galaxy3dManifest||current!=='galaxy_collision_3d')return;
  exitDeep();exitFusion();hideReveal();resetViewport();
  if(!galaxy3d)galaxy3d=new Galaxy3DView($('#galaxy3dCanvas'));
  try{
    await galaxy3d.load(`/runs/${runId}/${galaxy3dManifest.folder}/frame_${String(playbackFrame).padStart(4,'0')}.json?t=${Date.now()}`);
    galaxy3d.setFocus('all');galaxy3dActive=true;$('#screen').classList.add('hidden');$('#galaxy3dCanvas').classList.remove('hidden');$('#galaxy3dTools').classList.remove('hidden');$('#galaxy3dView').textContent='Exit 3D view';
    updateGalaxyViewButtons();updateViewport();galaxy3d.resize();
    currentStory='This is a real softened all-pairs super-particle calculation, conditioned by Gaia/PHAT morphology. It is illustrative rather than a fitted equilibrium prediction; use Milky Way or M31 focus to inspect the starting discs.';renderOverlayCards();
  }catch(error){exitGalaxy3d();showUiMessage(`Interactive 3D view unavailable: ${error.message}`);}
}
$('#galaxy3dView').onclick=()=>galaxy3dActive?exitGalaxy3d():enterGalaxy3d();
document.querySelectorAll('[data-galaxy-focus]').forEach(button=>button.onclick=()=>{if(!galaxy3d)return;galaxy3d.setFocus(button.dataset.galaxyFocus);updateGalaxyViewButtons();});
$('#galaxyHalo').onclick=()=>{if(!galaxy3d)return;galaxy3d.setHalo(!galaxy3d.showHalo);updateGalaxyViewButtons();};
$('#galaxy3dReset').onclick=()=>{if(galaxy3d)galaxy3d.reset();};
window.addEventListener('resize',()=>{if(deepActive&&deep)deep.resize();if(fusionActive&&fusion)fusion.resize();if(galaxy3dActive&&galaxy3d)galaxy3d.resize();});
$('#reveal').onclick=showReveal;
init();
