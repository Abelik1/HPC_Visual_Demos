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
 black_hole:["A camera parked beside a real black hole found by the Gaia satellite. Every star in the sky is a real Gaia star, placed where it would be seen from there.","The black disc is the shadow: every ray inside it falls through the horizon. It is 2.6 times wider than the horizon itself.","Just outside the shadow, light has circled the hole before reaching us, so the whole sky is squeezed into a thin ring, again and again.","Switch to the ray paths to see the exact routes the light took around the hole."],
 pbh:["The Universe is extremely young.","Increase one small density perturbation.","Below the threshold it disperses; above it, collapse accelerates.","The boundary between those two outcomes is extremely sharp."],
 fluid:["Begin with smooth flow.","Put an obstacle in the stream.","Watch coherent vortices form and interact.","Underneath is a lattice of cells being updated over and over."],
 cosmic_web:["Begin almost uniform.","Let gravity amplify tiny fluctuations.","Clusters, filaments and voids emerge.","This is the large-scale structure we actually observe."],
 galaxy_collision:["Two calm galaxies approach.","Their gravity creates long tidal tails.","The final shape remembers the orbit.","Change how they come in and the whole encounter changes."],
 galaxy_collision_3d:["Two illustrative 3D galaxy models begin on a Local Group-scale approach.","Every disc, bulge and dark-halo super-particle pulls on every other one.","Rotate the particle state to inspect tidal tails and out-of-plane debris.","This is one all-pairs realisation with catalogue-conditioned structure, not a fitted Local Group prediction."],
 nbody_murb:["A cloud of bodies orbits a heavy centre.","Every body pulls on every other body: N² forces per step.","This is the NBody-EuroHPC C++ code, the same one validated on Leonardo A100s.","Swap the implementation and compare how fast the same physics runs."],
 reaction_diffusion:["Start from a tiny disturbance.","Only local rules are applied.","Complex global structure appears.","No central designer: two chemicals and a local rule did all of it."],
 crystal:["One microscopic seed.","Six primary arms emerge from the seed.","Each branch creates smaller branches of its own.","Zoom in: the same growth rule repeats at several scales."],
 neural_wall:["A network learns to redraw a picture from pixel coordinates alone.","Its weights are the compressed file; the picture going in is raw colour values.","Squeeze harder and fine detail is the first thing to go.","A JPEG of the same size is the honest benchmark."],
 fusion_plasma:["A coherent wave circles the magnetic bottle.","Luminous tracers follow drift derived from the evolving field.","Their trails expose changing toroidal and poloidal flow.","Heating feeds the plasma until coherent motion turns into turbulence."],
 weather_ensemble:["Begin from today’s global observations.","The atmosphere carries vorticity and moisture around the planet.","Tiny uncertainties grow as the forecast races five days ahead.","A tiny change to the starting state sends the storm somewhere else."],
 molecular_dynamics:["At this scale nothing sits still: water kicks every bead, all the time.","Fold: oily beads hide from water together, which is why proteins have a core.","Every bead feels every other bead, every step.","Machine: the switch does not push the ring; it only changes where the ring is caught.","Motors: a walker on a lopsided track, or a rotor that every cell in you runs, turn thermal noise into directed motion."]
};

// Mode 2 replaces the story, the legend and the readout panel: it is the same
// magnetic bottle, but the question has changed from "how does it flow" to
// "can the controller hold it".
const methodStories={
 black_hole:{weak_field:["Start with a real Hubble Deep Field source plane.","Now bend those same galaxy sight lines around the chosen well or wells.","The 2D observer image scans over the source plane while the 3D rays advance through it.","Every pixel is one question: trace this sight line backwards and find where its light came from."]},
 fusion_plasma:{guardian:[
  "Shot 1. The same magnetic bottle, now filled with markers the field has to hold, and a policy that has never trained.",
  "Between shots the network is scored on what just happened and trained on it. Then the identical experiment runs again.",
  "Shot by shot the wall losses fall. The physics never changed; only the controller did.",
  "Pull back: the trained controller re-tested against instability drives it never saw."]}
};
const methodLegends={
 black_hole:{weak_field:'Both modes use the same NASA Hubble Deep Field crop: 2D maps it through the configured well(s); 3D places that crop on the source plane and advances a matching ray bundle. The spin-like term is qualitative, not Kerr tracing.'},
 fusion_plasma:{guardian:'Cyan markers are deep inside the bottle and amber ones are close to the wall; every orange burst is a marker that escaped confinement and hit the vessel. The eight rings encircling the machine are the coils: two radial (cyan), two vertical (orange) and four shaping (violet), brightening with the command the policy sends them. The helical lines are illustrative confinement geometry shaped by those commands, not a solved equilibrium, and the markers are transport tracers rather than kinetic particles. The run is a sequence of identical virtual shots: the policy is frozen for the whole of each one, then scored on its wall losses and trained on the states that shot visited before the next one starts. The training-progress overlay is that scoreboard. The cross-section overlay is one slice through the torus seen end-on, with all toroidal positions collapsed onto it; the eight ringed coils around it are the three output banks of the policy, and it carries its own legend.'}
};
const methodPanelNames={black_hole:{weak_field:'Lensing readout'},fusion_plasma:{guardian:'Shot and training readout'}};
// Runs saved before a demo had a choice of solver recorded "default".
const legacyMethods={black_hole:'weak_field'};
function runMethod(){let m=currentMeta&&currentMeta.method||$('#method')?.value||'';return m==='default'&&legacyMethods[current]||m;}
function storyLines(){return methodStories[current]?.[runMethod()]||stories[current]||[];}
function legendText(){return methodLegends[current]?.[runMethod()]||legends[current]||'Scientific simulation output.';}
function panelName(){return methodPanelNames[current]?.[runMethod()]||panelNames[current]||'Live data';}
// Optional diagnostic images a run actually wrote, published in its meta.
function runOverlays(){return Array.isArray(currentMeta?.overlays)?currentMeta.overlays:[];}
const overlayLabels={network:'Policy graph',poloidal:'Cross-section',shots:'Training progress'};
const overlayTitles={network:'Live policy graph',poloidal:'Poloidal cross-section',shots:'Training progress'};

const viewModes={
  nbody_murb:[
    {id:'frames',label:'MUrB camera',folder:'frames'},
    {id:'side',label:'Side view',folder:'modes/side'},
    {id:'top',label:'Top view',folder:'modes/top'}
  ],
  black_hole:[
    {id:'frames',label:'Camera view',folder:'frames'},
    {id:'3d',label:'Exact ray paths',folder:'modes/3d'}
  ],
  // Both recorded for every frame; the presenter picks, the run never switches.
  neural_wall:[
    {id:'frames',label:'In → out',folder:'frames'},
    {id:'wall',label:'All networks',folder:'modes/wall'}
  ]
};
const methodViewModes={black_hole:{weak_field:[
  {id:'3d',label:'3D ray space',folder:'modes/3d'},
  {id:'frames',label:'2D observer image',folder:'frames'}]}};
function viewModesFor(){return methodViewModes[current]?.[runMethod()]||viewModes[current]||[];}
const panelNames={
  black_hole:'Ray-tracing readout',pbh:'Collapse readout',fluid:'Wind-tunnel instruments',
  cosmic_web:'Cosmology readout',galaxy_collision:'Encounter readout',galaxy_collision_3d:'3D gravity readout',nbody_murb:'MUrB readout',reaction_diffusion:'Pattern readout',
  crystal:'Growth readout',neural_wall:'Compression readout',fusion_plasma:'Tokamak control',
  weather_ensemble:'Forecast clock',molecular_dynamics:'Molecular trajectory'
};
const legends={
  black_hole:'Camera view: the real Gaia DR3 sky, colour from each star\'s measured BP-RP colour, traced through exact Schwarzschild light paths. The black disc is the shadow, not the horizon. Ray paths: each escaping ray is coloured by how far gravity bent it, from blue (almost straight) through green and yellow to magenta (looped the hole); red rays fall in. The dashed ring is the photon sphere. Each escaping ray ends on the real Gaia star its light came from (the brightest within 1.5° of its exact escape direction; really far beyond the edge of the picture), and arrows show the light travelling to the camera. Behind is the same Gaia sky the camera sees, not lensed in this view.',
  pbh:'Brighter central density means localisation; a spreading shell means dispersion.',
  fluid:'Colour shows speed. Tracer streaks show direction; alternating wake colours expose shed vortices.',
  cosmic_web:'Brightness is density: knots are clusters, threads are filaments and dark regions are voids.',
  galaxy_collision:'Blue mass belongs to the Milky Way, warm mass to Andromeda; overlap brightens toward white.',
  galaxy_collision_3d:'Blue bodies belong to the Milky Way and orange bodies to M31. Faint particles are massive dark-halo super-particles; every visible body participates in the direct force. This is a catalogue-conditioned illustrative super-particle calculation, not a fitted equilibrium prediction of the Local Group.',
  nbody_murb:'Drawn the way the MUrB OpenGL viewer draws it: each body is a small sphere, coloured by speed from deep blue (slow) through cyan to white (fastest in that frame).',
  reaction_diffusion:'Colour represents the V chemical concentration in the Gray–Scott field.',
  crystal:'Every luminous segment is generated geometry. Deep zoom recomputes smaller branches.',
  neural_wall:'Left: the picture squeezed to the training size. Right: what the winning network redraws from its weights. Tile labels give each network\'s size ratio and quality in dB; red means bigger than the picture.',
  fusion_plasma:'The torus texture is the evolving field; luminous trails are passive tracers following its derived drift.',
  weather_ensemble:'Cloud colour combines moisture and vorticity on the simulated globe; the bright marker follows the cyclone centre.',
  molecular_dynamics:'Fold: amber beads avoid water, cyan like it, blue is plus, pink is minus. Machine: the gold ring rides the grey axle; the bright green station is the sticky one.'
};
// Running the same model many times over is only the experiment where a
// population is genuinely being trained, or a trained controller re-tested on
// conditions it never saw. For a large physics model the interesting compute is
// inside the single simulation, not in how many copies tile the screen, so
// those demos run one simulation: no width selector and no reveal button.
// Move an id between these two sets to change that for a demo.
const revealDemos=new Set(['neural_wall','neuro_racers','bat_vs_moth']);
// Demos whose solver honours a caller-chosen ensemble width. Any of these
// outside revealDemos is pinned to a single simulation, so it never spends
// time computing a reveal nobody can open.
const ensembleDemos=new Set([
  'black_hole','pbh','cosmic_web','galaxy_collision','reaction_diffusion',
  'crystal','fusion_plasma','weather_ensemble','neural_wall'
]);
// Mode 2 trains a policy and then re-tests it against instability drives it has
// not seen; mode 1 only re-runs the same passive field.
function revealAvailable(){
  if(!current)return false;
  if(current==='fusion_plasma')return runMethod()==='guardian';
  return revealDemos.has(current);
}
// Width is chosen in the run panel only where it is the visitor's decision.
const parallelDemos=new Set(['neuro_racers','bat_vs_moth']);
// Whether parallel_count will be sent, and so whether the profile editor must
// stay out of the way: two inputs for one number could only disagree.
function ensembleSetElsewhere(){
  return parallelDemos.has(current)||(ensembleDemos.has(current)&&!revealAvailable());
}
// "Scale reveal" was the wrong name once it only applies to populations.
const revealLabels={neural_wall:'Show every network',neuro_racers:'Show every search',
  bat_vs_moth:'Show every cave',fusion_plasma:'Re-test on unseen drives'};
const revealBanners={neural_wall:'THE WHOLE MODEL SEARCH',neuro_racers:'EVERY INDEPENDENT SEARCH',
  bat_vs_moth:'EVERY INDEPENDENT CAVE',fusion_plasma:'CONTROLLER RE-TEST'};
function configureRevealControl(){
  const button=$('#reveal'),available=revealAvailable();
  button.classList.toggle('hidden',!available);
  if(!available){button.disabled=true;return;}
  button.textContent=revealLabels[current]||'Show every run';
  $('#scaleReveal').textContent=revealBanners[current]||'EVERY INDEPENDENT RUN';
}
const demoInformation={
  black_hole:['Starlight around a real black hole','A virtual camera is parked beside one of the dormant black holes Gaia found by the wobble of a companion star. The sky it sees is built from 1.8 million real Gaia DR3 stars, moved to the camera\'s point of view with their measured parallaxes. Every pixel\'s light ray is followed backwards along the exact Schwarzschild geodesic to the star it came from, so the shadow, the Einstein ring and the nested images of the whole sky are all computed, not painted.','Exact non-spinning (Schwarzschild) light paths, solved by quadrature and checked against direct numerical integration. Not modelled: spin (Kerr), accretion light, the companion star, dust extinction along the new line of sight, and stars brighter than Gaia can measure (G < 3). Gaia BH3: Panuzzo et al. 2024; BH1/BH2: El-Badry et al. 2023. Star data: ESA/Gaia/DPAC.'],
  pbh:['A threshold in the young Universe','A small spherical density enhancement either spreads out or concentrates rapidly. The interesting result is the sharp boundary between those outcomes.','This is a reduced radial collapse demonstrator. It visualises critical behaviour but does not replace the project’s validated numerical-relativity solver.'],
  fluid:['Airflow around solid geometry','A D2Q9 lattice-Boltzmann solver moves density and momentum through the grid. Bounce-back cells form the selected bodies; streaks are passive particles following the computed velocity.','Choose a preset and optionally paint extra solid cells with the grid builder. Every visible custom block becomes part of the solver mask, so it changes the wake rather than merely decorating the image.'],
  cosmic_web:['How gravity grows a cosmic web','Nearly uniform matter begins with a spectrum of tiny perturbations. Gravity amplifies them into knots, filaments and voids while gas pressure changes the smallest supported structure.','The recipe compares our matter-dominated universe with one where radiation keeps dominating (it expands but does not clump) and one without dark matter (only the baryon fraction sources gravity, from smoother seeds). Toggle comoving expansion, a qualitative dark-energy term and a warm-dark-matter cutoff. These are exhibition-scale theory comparisons, not precision cosmological parameter inference.'],
  galaxy_collision:['The Milky Way–Andromeda encounter','Massive galaxy centres and tracer stars evolve through their mutual gravity. Tidal tails and the final remnant remember the initial orbit.','Transverse-velocity uncertainty makes the real encounter uncertain too; changing the impact parameter or approach speed changes the whole outcome.'],
  galaxy_collision_3d:['A direct 3D gravitational encounter','Disc, bulge and dark-halo super-particles interact in three dimensions. Rotate the saved particle state while playback advances.','The particles represent large groups of real stars and dark matter. Softening prevents unresolved close encounters from dominating the large-scale merger.'],
  nbody_murb:['MUrB: direct N-body on CPU and GPU','The external NBody-EuroHPC code (MUrB, Sorbonne University / LIP6, extended for EuroHPC) computes every pairwise gravitational force and records a .murbtraj trajectory. The dashboard launches it, then renders that trajectory.','Pick an implementation under Solver: the naive single-core reference, SIMD, OpenMP, or CUDA when the executable was built with it. Trajectories recorded on Leonardo can be imported with scripts/import_murbtraj.py.'],
  reaction_diffusion:['Complexity from two local reactions','Two diffusing chemicals follow the Gray–Scott equations. A tiny disturbance grows into spots, waves or labyrinths without a central pattern designer.','Feed and kill rates choose the pattern regime. Neighbouring values give spots, stripes, waves or labyrinths from the same two equations.'],
  crystal:['Branching growth from one seed','A deterministic anisotropic growth rule repeatedly creates side branches. Changing symmetry or growth conditions produces a different crystal habit.','Deep zoom regenerates geometry at the requested scale; it does not enlarge a finished bitmap. The model is deliberately geometric rather than molecular ice physics.'],
  neural_wall:['A neural network as an image codec','Each coordinate network receives only x and y and predicts red, green and blue. Everything it knows is stored in its weights, so their size is the size of the compressed picture.','Sizes count the picture as raw 8-bit RGB and the network as stored float32 weights. The winner is the best network that is actually smaller than the picture; a same-size JPEG is reported alongside so the comparison stays honest.'],
  fusion_plasma:['One magnetic bottle, two questions','Mode 1 evolves a reduced nonlinear plasma-wave field on a periodic lattice, wraps it onto a torus and lets passive tracers expose the drift it produces. Mode 2 keeps that field as the turbulence source and hands the coils to a small neural policy.','Mode 2 runs a series of virtual shots. Each shot is the identical experiment - same field, same markers, same disturbance - with the policy held frozen, so the physics on screen is never perturbed by training. Between shots the network is scored on the markers it lost to the wall and optimized against them. Magnetic lines are explanatory confinement geometry, not a solved equilibrium.'],
  weather_ensemble:['Why forecasts become uncertain','A reduced rotating atmosphere advects vorticity and moisture around a globe. Small changes to the initial state grow into different storm tracks.','Raising the initial uncertainty perturbs the starting state; small differences there grow into a different storm track by day five.'],
  molecular_dynamics:['Fold a protein, or run a molecular machine','Fold: write a chain of oily, water-loving and charged beads and watch it curl up. Machine: a ring threaded on an axle (a rotaxane) that a switch sends between two stations.','Coarse-grained Langevin dynamics in reduced units: each bead stands for a group of atoms, harmonic bonds and bending, all-pairs excluded volume, H-H attraction for the hydrophobic effect and screened charges. Illustrative, not a force field.']
};

function escapeHtml(value){return String(value??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));}
const settingLabels={bodies:'Bodies',iterations:'Iterations',warmup:'Warm-up iterations',shots:'Virtual shots',learning_rate:'Policy learning rate',width:'Output width',height:'Output height',ensemble:'Reveal simulations',radial_points:'Radial grid points',nx:'Lattice width',ny:'Lattice height',steps_per_frame:'Steps per frame',tracers:'Tracer particles',trail:'Trail length',tracer_boost:'Tracer speed boost',total_steps:'Total simulation steps',grid:'Density grid',particles:'Particles / bodies',sweep_steps:'Steps per independent run',jeans_ref:'Jeans length reference',ic_amplitude:'Initial fluctuation amplitude',ic_velocity:'Initial velocity amplitude',substeps:'Solver substeps',span_gyr:'Simulated duration',softening:'Gravity softening',force_tile:'Force tile size',max_step:'Maximum gravity step',n:'Simulation grid',depth:'Growth depth',zoom_levels:'Prebuilt zoom levels',zoom_depth:'Zoom growth depth',zoom_tile:'Zoom tile pixels',zoom_max_level:'Maximum zoom level',zoom_detail_base:'Base zoom detail',zoom_detail_max:'Maximum zoom detail',networks:'Networks trained',tile:'Network tile pixels',sweep_n:'Grid size per independent run',batch:'Training batch size',horizon:'Training horizon',train_updates:'Training updates',display_steps:'Control steps per shot',sweep_particles:'Particles per independent run'};
const settingHelp={bodies:'Number of bodies MUrB simulates; the work grows as N².',iterations:'Timed MUrB iterations; frames are sampled evenly across them.',warmup:'Untimed iterations run first so the timing excludes start-up.',shots:'Identical experiments run in sequence; the policy trains between them, never during one.',learning_rate:'Adam step size for the policy. Larger converges in fewer shots.',ensemble:'Independent runs compared side by side.',particles:'Actual simulated particle/body count.',total_steps:'Fixed numerical work distributed across the saved frames.',substeps:'Numerical integration steps inside each saved frame.',span_gyr:'Physical duration represented by the complete run, in billions of years.',softening:'Minimum gravity length scale in kpc; prevents singular super-particle forces.',force_tile:'Bodies processed together by the direct-force solver.',batch:'Imagined trajectories evaluated per training update, separate from the displayed shots.',horizon:'Time steps in each virtual training shot.',networks:'Independent neural networks trained and compared.',tile:'Pixel width and height reconstructed by each network.',n:'Primary simulation grid resolution.',grid:'Cells per side in the density calculation.',nx:'Horizontal lattice cells.',ny:'Vertical lattice cells.',sweep_steps:'Numerical steps used by each independent run.',sweep_n:'Grid resolution used by each independent run.',sweep_particles:'Particle count used by each independent run.',width:'Native render width in pixels.',height:'Native render height in pixels.'};
function renderProfileSettings(values=null){
  const host=$('#profileSettings');if(!host||!current)return;
  const presetName=$('#profile').value,preset=specs?.profiles?.[presetName]?.[current]||{};
  const schema=specs?.profile_setting_schema?.[current]||{};
  const chosen=values&&Object.keys(values).length?values:preset;
  host.innerHTML='';
  Object.entries(schema).forEach(([key,rule])=>{
    // Reveal cardinality has its own prominent, constrained selector in the
    // run panel. Rendering it here as well would allow two conflicting values.
    if((key==='ensemble'||key==='networks')&&ensembleSetElsewhere())return;
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
    if((key==='ensemble'||key==='networks')&&ensembleSetElsewhere())continue;
    const input=$('#s_'+key);
    if(!input||!input.checkValidity())throw new Error(`${settingLabels[key]||key} must be between ${schema[key].min} and ${schema[key].max}`);
    result[key]=Number(input.value);
  }
  return result;
}
function currentParameters(){let out={};if(!current||!specs?.demos?.[current])return out;Object.entries(specs.demos[current].params).forEach(([k,p])=>{if(Array.isArray(p.methods)&&!p.methods.includes($('#method')?.value))return;let input=$('#p_'+k);if(input)out[k.replaceAll('_',' ')]=input.type==='checkbox'?(input.checked?'on':'off'):input.value;});return out;}
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
  if(overlayEnabled.has('data')){wanted.add('data');let card=overlayCard('data',panelName(),'rows');let values=Object.keys(frameOverlay||{}).length?frameOverlay:currentParameters();updateOverlayRows(card.querySelector('.overlayRows'),values);}
  if(overlayEnabled.has('legend')){wanted.add('legend');let card=overlayCard('legend','How to read this view');card.querySelector('p').textContent=legendText();}
  if(current==='neural_wall'&&overlayEnabled.has('network')&&runId){wanted.add('network');let card=overlayCard('network','Winning network','image');let img=card.querySelector('img'),src=`/runs/${runId}/overlays/network/frame_${String(playbackFrame).padStart(4,'0')}.jpg`;if(img.dataset.src!==src){img.dataset.src=src;img.src=src;}}
  if(current!=='neural_wall'&&runId)runOverlays().forEach(name=>{if(!overlayEnabled.has(name))return;wanted.add(name);let card=overlayCard(name,overlayTitles[name]||name,'image');let img=card.querySelector('img'),src=`/runs/${runId}/overlays/${name}/frame_${String(playbackFrame).padStart(4,'0')}.jpg`;if(img.dataset.src!==src){img.dataset.src=src;img.src=src;}});
  window.GameDemos?.overlayCards(wanted);
  [...layer.children].forEach(card=>{if(!wanted.has(card.dataset.overlayCard))card.remove();});
}
function renderViewerDock(){let modes=$('#modeControls'),controls=$('#overlayControls');if(!modes||!controls)return;modes.innerHTML='';let defs=viewModesFor();if(defs.length){modes.innerHTML='<h4>SIMULATION VIEW</h4>';defs.forEach(def=>{let b=document.createElement('button');b.textContent=def.label;b.setAttribute('aria-label',def.label);b.classList.toggle('selected',activeViewMode===def.id);b.setAttribute('aria-pressed',String(activeViewMode===def.id));b.onclick=()=>{activeViewMode=def.id;renderViewerDock();if(frameAvailable())showFrame(playbackFrame,playbackTotal);};modes.appendChild(b);});}controls.innerHTML='<h4>OPTIONAL OVERLAYS</h4>';let items=[['story','Explanation'],['data',panelName()],['legend','Legend / method']];if(current==='neural_wall')items.push(['network','Network graph']);else runOverlays().forEach(name=>items.push([name,overlayLabels[name]||name]));window.GameDemos?.dockItems(items);items.forEach(([id,label])=>{let b=document.createElement('button');b.textContent=label;b.setAttribute('aria-label',label);b.classList.toggle('selected',overlayEnabled.has(id));b.setAttribute('aria-pressed',String(overlayEnabled.has(id)));b.onclick=()=>{overlayEnabled.has(id)?overlayEnabled.delete(id):overlayEnabled.add(id);renderViewerDock();renderOverlayCards();};controls.appendChild(b);});renderOverlayCards();}
async function loadFrameOverlay(frame){if(!runId)return;try{let response=await fetch(`/runs/${runId}/frame_data/frame_${String(frame).padStart(4,'0')}.json?t=${Date.now()}`);if(!response.ok)return;let body=await response.json();if(frame===playbackFrame){frameOverlay=body.values||{};renderOverlayCards();}}catch(_){}}
function showUiMessage(message){currentStory=String(message);overlayEnabled.add('story');renderViewerDock();}

async function init(){specs=await (await fetch('/api/specs')).json();await HPC.load();HPC.machineSwitch($('#machineSwitch'));HPC.onChange(()=>{if(galleryCategory!==ARCHIVE)galleryCategory='All experiments';renderGallery();});$('#hpcSettings').onclick=()=>HPC.openSettings(specs.demos);setupRunOn();renderGallery();applyBackends();applyMethods();bindTimelineControls();await loadLibrary();let query=new URLSearchParams(location.search);let requestedRun=query.get('run'),requestedDemo=query.get('demo');let saved=requestedRun&&library.find(item=>item.id===requestedRun);if(saved)openRun(saved);else if(requestedDemo&&specs.demos[requestedDemo])openDemo(requestedDemo);}

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
  sel.onchange=()=>{describe();configureRevealControl();applyParameterMethods();renderDemoInfo();renderViewerDock();if(current)renderProfileSettings();};describe();
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
// Favourites are stored by the server (runs/_library.json), so the demo-day
// viewer sees the same stars and can offer them first as showcase runs.
const STAR='★',NO_STAR='☆';
async function toggleFavourite(run){
  const favourite=!run.favourite;
  try{
    const response=await fetch(`/api/runs/${encodeURIComponent(run.id)}/favourite`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({favourite})});
    if(!response.ok)throw new Error(`favourite returned ${response.status}`);
    library.filter(item=>item.id===run.id).forEach(item=>{item.favourite=favourite;});
  }catch(error){showUiMessage(`Could not save the favourite: ${error.message}`);}
  renderLibrary();renderStageRuns();
}
function favouriteButton(run){
  const button=document.createElement('button');button.type='button';
  button.className='favStar'+(run.favourite?' on':'');button.textContent=run.favourite?STAR:NO_STAR;
  const label=run.favourite?'Remove from favourites':'Add to favourites';
  button.title=label;button.setAttribute('aria-label',label);button.setAttribute('aria-pressed',String(Boolean(run.favourite)));
  button.onclick=event=>{event.preventDefault();event.stopPropagation();toggleFavourite(run);};
  return button;
}
let libraryFavouritesOnly=false;
function renderLibrary(){
  let el=$('#libList');if(!el)return;el.innerHTML='';
  const filter=$('#libFavourites');
  if(filter){filter.classList.toggle('selected',libraryFavouritesOnly);filter.setAttribute('aria-pressed',String(libraryFavouritesOnly));
    filter.textContent=`${STAR} Favourites${library.some(r=>r.favourite)?` (${library.filter(r=>r.favourite).length})`:''}`;}
  const shown=libraryFavouritesOnly?library.filter(r=>r.favourite):library;
  if(libraryFavouritesOnly&&!shown.length){$('#libMore').classList.add('hidden');el.innerHTML='<p style="color:#7d93b6;font-size:14px">No favourites yet. Star a saved run to keep it here.</p>';return;}
  if(!library.length){$('#libMore').classList.add('hidden');el.innerHTML='<p style="color:#7d93b6;font-size:14px">No saved runs yet. Run a simulation and it will appear here.</p>';return;}
  $('#libMore').classList.toggle('hidden',shown.length<=8);
  $('#libMore').textContent=showAllRuns?'Show fewer saved runs':`Show all ${shown.length} saved runs`;
  (showAllRuns?shown:shown.slice(0,8)).forEach(r=>{
    let name=(specs.demos[r.demo]||{}).name||r.demo;
    let card=document.createElement('article');card.className='runCard';
    let gpu=/cupy|cuda/i.test(r.backend||'');
    card.innerHTML=`<a class="runLink" href="/?run=${encodeURIComponent(r.id)}"><img loading="lazy" src="${escapeHtml(r.thumb)}" alt="">
      <div class=runMeta><b>${escapeHtml(name)}</b><span>${runLabel(r)} · ${r.frames} frames</span>
      <div class=runTags><i>${r.profile==='hpc'?'HPC':r.profile||'?'}</i><i class="${gpu?'gpu':''}">${r.backend||'?'}</i>${r.method?`<i>${r.method.replaceAll('_',' ')}</i>`:''}${r.zoom?'<i>zoom</i>':''}</div></div></a>`;
    card.querySelector('a').onclick=event=>{if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;event.preventDefault();openRun(r);};
    card.classList.toggle('favourite',Boolean(r.favourite));card.appendChild(favouriteButton(r));
    el.appendChild(card);
  });
}
function renderStageRuns(){
  let el=$('#stageRuns');if(!el)return;el.innerHTML='';
  let mine=library.filter(r=>r.demo===current);
  if(!mine.length)return;
  let head=document.createElement('span');head.className='chip';head.style.cursor='default';
  head.innerHTML='<em>Replay:</em>';el.appendChild(head);
  // Favourites first: they are the runs worth coming back to.
  mine=[...mine.filter(r=>r.favourite),...mine.filter(r=>!r.favourite)];
  mine.slice(0,12).forEach(r=>{
    let group=document.createElement('span');group.className='chipGroup';
    let c=document.createElement('button');c.className='chip'+(r.id===runId?' active':'')+(r.favourite?' favourite':'');
    c.innerHTML=`${r.favourite?STAR+' ':''}${runLabel(r)} <em>${r.backend||''}</em>`;
    c.onclick=()=>openRun(r);group.append(c,favouriteButton(r));el.appendChild(group);
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
  renderViewerDock();
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
  if(current==='fluid'&&fluidBuilder){fluidBuilder.setPreset(r.params?.obstacle??0);fluidBuilder.setCells(r.params?._obstacle_grid);}
  $('#frames').value=r.frames;updateTimelineHelp();
  if(r.method&&[...$('#method').options].some(option=>option.value===r.method)){$('#method').value=r.method;$('#method').dispatchEvent(new Event('change'));}
  setPlaybackControls(true);
  configureRevealControl();
  $('#reveal').disabled=!(r.has_reveal&&revealAvailable());
  startPlayback(0);
  if(current==='fusion_plasma'&&fusionManifest&&preferFusion3d)enterFusion();
  window.GameDemos?.onOpenRun(r);
  renderStageRuns();
}
$('#libRefresh').onclick=loadLibrary;
$('#libFavourites').onclick=()=>{libraryFavouritesOnly=!libraryFavouritesOnly;showAllRuns=false;renderLibrary();};
$('#libMore').onclick=()=>{showAllRuns=!showAllRuns;renderLibrary();if(!showAllRuns)$('#library').scrollIntoView({block:'start'});};
const demoCategories={black_hole:'Universe',pbh:'Universe',cosmic_web:'Universe',galaxy_collision:'Universe',galaxy_collision_3d:'Universe',nbody_murb:'Universe',fluid:'Physics',fusion_plasma:'Physics',reaction_diffusion:'Patterns & life',crystal:'Patterns & life',molecular_dynamics:'Patterns & life',weather_ensemble:'Physics',neural_wall:'AI & learning',neuro_racers:'AI & learning',bat_vs_moth:'AI & learning'};
// Which demos appear, and in what order, comes from the active demo day's
// lineup (config/lineups.json, edited under "Lineups & HPC"). Archived demos
// are listed only under the Archive tab; saved runs and direct ?demo= links
// keep working for every demo.
const ARCHIVE='Archive';
let galleryCategory='All experiments';
function galleryItem(id){
  const x=HPC.extra(id);
  if(x)return {id,name:x.name,tagline:x.tagline,category:x.category||'Physics',video:true,href:x.url||HPC.videoHref(id),external:Boolean(x.url),count:x.videos};
  const d=specs.demos[id];if(!d)return null;
  return {id,name:d.name,tagline:d.tagline,category:demoCategories[id]||'Physics',href:`/?demo=${encodeURIComponent(id)}`};
}
function renderGallery(){
  const host=$('#gallery'),filters=$('#categoryFilters'),query=$('#demoSearch').value.trim().toLowerCase();
  const archiveView=galleryCategory===ARCHIVE;
  host.innerHTML='';filters.innerHTML='';
  const shown=HPC.items(Object.keys(specs.demos)).map(galleryItem).filter(Boolean);
  const archived=(HPC.lineup.archived||[]).map(galleryItem).filter(Boolean);
  // A category with nothing in today's lineup would only ever show an empty grid.
  const activeCategories=new Set(shown.map(item=>item.category));
  ['All experiments','Universe','Physics','Patterns & life','AI & learning'].filter(category=>category==='All experiments'||activeCategories.has(category)).concat(ARCHIVE).forEach(category=>{
    const button=document.createElement('button');button.textContent=category;
    if(category===ARCHIVE){button.classList.add('archiveTab');button.title='Demos that are not part of either demo day';}
    button.classList.toggle('selected',category===galleryCategory);button.setAttribute('aria-pressed',String(category===galleryCategory));
    button.onclick=()=>{galleryCategory=category;renderGallery();[...filters.children].find(item=>item.textContent===category)?.focus();};filters.appendChild(button);
  });
  let count=0;
  (archiveView?archived:shown).forEach(item=>{
    if(!archiveView&&galleryCategory!=='All experiments'&&galleryCategory!==item.category)return;
    if(query&&!`${item.name} ${item.tagline} ${item.category}`.toLowerCase().includes(query))return;
    count++;
    const machines=HPC.active()==='all'?HPC.machinesOf(item.id).map(HPC.machineLabel):[];
    const card=document.createElement('article');card.className='card'+(archiveView?' archived':'');
    card.innerHTML=`<a class="cardLink" href="${escapeHtml(item.href)}"${item.external?' target="_blank" rel="noopener"':''}><div class="cardImage"><img src="/static/previews/${encodeURIComponent(item.id)}.webp" alt="" loading="lazy" width="800" height="450"><span class="num">${String(count).padStart(2,'0')}</span>${archiveView?'<span class="archiveBadge">Archived</span>':item.video?'<span class="archiveBadge">Video</span>':''}</div><div class="cardBody"><span class="cardCategory">${escapeHtml(item.category)}${machines.length?' · '+escapeHtml(machines.join(' + ')):''}</span><h3>${escapeHtml(item.name)}</h3><p>${escapeHtml(item.tagline)}</p><div class="go">${item.video?(item.external?'Open link':`Play recorded video${item.count===0?' (none yet)':''}`):archiveView?'Open work in progress':'Explore simulation'} <span aria-hidden="true">↗</span></div></div></a>`;
    // Video items have no preview of their own until one is added.
    card.querySelector('img').onerror=e=>{e.target.onerror=null;e.target.src='/static/previews/videos.webp';};
    if(!item.video)card.querySelector('a').onclick=event=>{if(event.ctrlKey||event.metaKey||event.shiftKey||event.altKey)return;event.preventDefault();openDemo(item.id);};
    host.appendChild(card);
  });
  $('#demoCount').textContent=shown.length;
  const day=HPC.active()==='all'?'':` for the ${HPC.machineLabel(HPC.active())} demo day`;
  $('#galleryResults').textContent=archiveView?`${count} archived experiment${count===1?'':'s'} · not part of either demo day`:`${count} experiment${count===1?'':'s'}${galleryCategory==='All experiments'?day||' to explore':` in ${galleryCategory.toLowerCase()}`}`;
  if(!count)host.innerHTML=`<p class="galleryEmpty">${archiveView?'No archived experiments match.':'No experiments found. Try another search or category, or add demos to this day under Lineups & HPC.'}</p>`;
}
// "Run on": this computer, or one of the clusters in config/clusters.json.
function setupRunOn(){
  const select=$('#runOn');
  HPC.clusterList().forEach(c=>{const o=document.createElement('option');o.value=c.name;o.textContent=c.label;select.appendChild(o);});
  let saved=null;try{saved=localStorage.getItem('leonardo.runOn');}catch(_){}
  if(saved&&[...select.options].some(o=>o.value===saved))select.value=saved;
  const describe=()=>{const c=HPC.clusterList().find(x=>x.name===select.value);
    $('#runLabel').textContent=c?`Run on ${c.label}…`:'Run simulation';
    $('#runOnHelp').textContent=c?'You confirm every setting first; the run is fetched back here when it ends':'Runs here, frames appear as they are written';};
  select.onchange=()=>{try{localStorage.setItem('leonardo.runOn',select.value);}catch(_){}
    // A cluster run is only worth it at cluster scale: offer its preset.
    const c=HPC.clusterList().find(x=>x.name===select.value);
    if(c&&c.default_profile&&$('#profile').value!==c.default_profile&&[...$('#profile').options].some(o=>o.value===c.default_profile)){
      $('#profile').value=c.default_profile;$('#profile').dispatchEvent(new Event('change'));}
    describe();};
  describe();
}
$('#demoSearch').oninput=renderGallery;
$('#previewReplay').onclick=()=>{const latest=library.find(item=>item.demo===current);if(latest)openRun(latest);};
function hideReveal(){let sw=$('.screenWrap');sw.classList.remove('revealing');$('#scaleReveal').classList.remove('show');$('#screen').style.opacity=1;}
function stopPlayback(){if(playbackTimer)clearInterval(playbackTimer);playbackTimer=null;playbackPlaying=false;updatePlaybackButton();}
function updatePlaybackButton(){$('#playPause').textContent=playbackPlaying?'Pause':'Play';}
function setPlaybackControls(enabled){$('#playPause').disabled=!enabled;$('#playbackRate').disabled=!enabled;$('#frameSeek').disabled=!enabled;$('#reveal').disabled=!(enabled&&revealAvailable());$('#deepZoom').disabled=!(enabled&&deepManifest);$('#fusionView').disabled=!(enabled&&fusionManifest&&current==='fusion_plasma');$('#galaxy3dView').disabled=!(enabled&&galaxy3dManifest&&has3dView(current));updatePlaybackButton();}
function frameAvailable(){return Boolean(runId&&lastFrame>=0);}
function updateViewport(){let wrap=$('.screenWrap');wrap.style.setProperty('--view-zoom',zoom);wrap.style.setProperty('--view-pan-x',`${panX}px`);wrap.style.setProperty('--view-pan-y',`${panY}px`);wrap.classList.toggle('isZoomed',zoom>1);let enabled=frameAvailable()&&!deepActive&&!fusionActive&&!galaxy3dActive&&!window.GameDemos?.active;$('#zoomIn').disabled=!enabled;$('#zoomOut').disabled=!enabled||zoom<=1;$('#zoomReset').disabled=!enabled||zoom===1;$('#zoomReset').textContent=`${zoom.toFixed(zoom%1?1:0)}×`;}
function resetViewport(){zoom=1;panX=0;panY=0;updateViewport();}
function changeZoom(amount){if(deepActive||!frameAvailable())return;zoom=Math.max(1,Math.min(8,Math.round((zoom+amount)*10)/10));let wrap=$('.screenWrap');let limitX=wrap.clientWidth*(zoom-1)/2;let limitY=wrap.clientHeight*(zoom-1)/2;panX=Math.max(-limitX,Math.min(limitX,panX));panY=Math.max(-limitY,Math.min(limitY,panY));updateViewport();}
function fusionFrameUrl(frame){if(!fusionManifest)return null;if(typeof fusionManifest==='string')return `/runs/${runId}/${fusionManifest}`;return `/runs/${runId}/${fusionManifest.folder}/frame_${String(frame).padStart(4,'0')}.json`;}
function showFrame(frame,total=playbackTotal){if(!runId||total<1)return;frame=Math.max(0,Math.min(total-1,Math.trunc(frame)));playbackFrame=frame;hideReveal();$('.screenWrap').classList.remove('empty');let mode=viewModesFor().find(item=>item.id===activeViewMode),folder=mode?.folder||'frames',image=$('#screen'),fallback=`/runs/${runId}/frames/frame_${String(frame).padStart(4,'0')}.jpg?t=${Date.now()}`;image.onerror=()=>{image.onerror=null;image.src=fallback;};image.src=`/runs/${runId}/${folder}/frame_${String(frame).padStart(4,'0')}.jpg?t=${Date.now()}`;if(galaxy3dActive&&galaxy3d)galaxy3d.load(`/runs/${runId}/${galaxy3dManifest.folder}/frame_${String(frame).padStart(4,'0')}.json?t=${Date.now()}`).catch(error=>showUiMessage(`3D frame unavailable: ${error.message}`));if(fusionActive&&fusion){let url=fusionFrameUrl(frame);if(url)fusion.load(`${url}?t=${Date.now()}`,true).catch(error=>showUiMessage(`3D plasma frame unavailable: ${error.message}`));}let p=(frame+1)/total;$('#bar').style.width=(p*100)+'%';$('#frameSeek').value=frame;$('#metric1').textContent=`frame ${frame+1}/${total}`;currentStory=(storyLines()[Math.min(3,Math.floor(p*4))])||currentStory;renderOverlayCards();loadFrameOverlay(frame);window.GameDemos?.onFrame(frame);updateViewport();}
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
// Molecular Machine: write your own sequence (web/chain_builder.js). It follows
// the Sequence preset until edited, and belongs to the fold mode only.
let chainBuilder=null;
function addChainBuilder(host){
  const select=$('#p_sequence'),length=()=>Number($('#s_particles')?.value)||40;
  chainBuilder=new ChainBuilder(host,{preset:select?Number(select.value):0,length:length()});
  if(select)select.addEventListener('change',()=>{chainBuilder.reset();chainBuilder.setPreset(select.value,length());});
  const sync=()=>chainBuilder.setVisible(['fold','default'].includes($('#method').value));
  $('#method').addEventListener('change',sync);sync();
}
function addFluidBuilder(host){
  // The builder draws the chosen preset so the grid matches what the solver
  // will actually build; see web/obstacle_builder.js.
  const select=$('#p_obstacle');
  fluidBuilder=new ObstacleBuilder(host,{preset:select?Number(select.value):0});
  if(select)select.addEventListener('change',()=>fluidBuilder.setPreset(select.value));
}
const methodInformation={black_hole:{weak_field:['Light paths through the Hubble Deep Field','The 2D observer image and 3D source plane use the same credited Hubble Deep Field crop. Move the primary well, then use binary or triple mode to see multiple deflection centres in both views.','This is a weak-field educational model, not a full Kerr geodesic or GRMHD calculation. The 2D scan moves across recorded source data; the wells remain fixed in the chosen scene. NASA Hubble Deep Field image: PIA12110.']}};
function renderDemoInfo(){let info=methodInformation[current]?.[runMethod()]||demoInformation[current]||['Scientific simulation',stories[current]?.[0]||'',''];$('#infoTitle').textContent=info[0];$('#infoSummary').textContent=info[1];$('#infoMethod').textContent=info[2];}
function clearSimulationSurface(){let wrap=$('.screenWrap'),image=$('#screen');image.onerror=null;image.onload=null;image.removeAttribute('src');image.style.opacity='';wrap.classList.add('empty');}
function addParameterControl(host,key,param){let el=document.createElement('div'),label=param.label||key.replaceAll('_',' ').replace(/^./,letter=>letter.toUpperCase());el.className='control';if(Array.isArray(param.methods))el.dataset.methods=param.methods.join(' ');if(param.kind==='toggle'){el.classList.add('toggleControl');el.innerHTML=`<label for="p_${key}">${escapeHtml(label)}</label><input id="p_${key}" type="checkbox" ${Number(param.value)?'checked':''}>`;el.querySelector('input').onchange=renderOverlayCards;}else if(param.kind==='choice'){let options=Object.entries(param.options||{}).map(([value,text])=>`<option value="${escapeHtml(value)}" ${Number(value)===Number(param.value)?'selected':''}>${escapeHtml(text)}</option>`).join('');el.innerHTML=`<div class=row><label for="p_${key}">${escapeHtml(label)}</label></div><select id="p_${key}">${options}</select>`;el.querySelector('select').onchange=renderOverlayCards;}else{el.innerHTML=`<div class=row><label for="p_${key}">${escapeHtml(label)}</label><b id="v_${key}">${param.value}</b></div><input id="p_${key}" type=range min="${param.min}" max="${param.max}" step="${param.step}" value="${param.value}">`;el.querySelector('input').oninput=e=>{$('#v_'+key).textContent=e.target.value;renderOverlayCards();};}host.appendChild(el);}
// Some demos have solver-specific parameters; show only the ones the chosen solver reads.
function applyParameterMethods(){let method=$('#method')?.value||'';document.querySelectorAll('#sliders .control[data-methods]').forEach(el=>el.classList.toggle('hidden',!el.dataset.methods.split(' ').includes(method)));}
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
  resetRunState();current=id;activeViewMode='frames';preferFusion3d=id==='fusion_plasma';overlayEnabled=new Set();currentStory=(stories[id]||[''])[0];$('#gallery').classList.add('hidden');$('#library').classList.add('hidden');$('#stage').classList.remove('hidden');let d=specs.demos[id];$('#fusionView').classList.toggle('hidden',id!=='fusion_plasma');$('#galaxy3dView').classList.toggle('hidden',!has3dView(id));$('#stageTitle').textContent=d.name;$('#stageTag').textContent=d.tagline;$('#stageEyebrow').textContent=(demoCategories[id]||'Science')+' / INTERACTIVE SIMULATION';let s=$('#sliders');s.innerHTML='';fluidBuilder=null;Object.entries(d.params).forEach(([k,p])=>{if(id==='neural_wall'&&k==='target')return;addParameterControl(s,k,p);});applyParameterMethods();if(id==='neural_wall')addNeuralTargetTools(s,d.params.target.value);if(id==='fluid')addFluidBuilder(s);chainBuilder=null;if(id==='molecular_dynamics')addChainBuilder(s);window.GameDemos?.mount(s,id);renderProfileSettings();configureParallelControl();configureRevealControl();updateTimelineHelp();clearSimulationSurface();renderDemoInfo();renderStageRuns();renderViewerDock();$('#status').textContent='READY';$('#status').style.color='';$('#bar').style.width='0';$('#metric1').textContent='frame —';$('#metric2').textContent='elapsed —';$('#metric3').textContent='backend —';window.scrollTo(0,0);$('#back').focus({preventScroll:true});return true;}
$('#back').onclick=()=>{document.body.classList.remove('demoOpen');history.replaceState(null,'','/');resetRunState();$('#stage').classList.add('hidden');$('#gallery').classList.remove('hidden');$('#library').classList.remove('hidden');loadLibrary();window.scrollTo(0,0);$('#demoSearch').focus({preventScroll:true});};
function buildRunRequest(){let settings;try{settings=collectProfileSettings();}catch(error){showUiMessage(error.message);return null;}let ps={};Object.keys(specs.demos[current].params).forEach(k=>{if(current==='neural_wall'&&k==='target')ps[k]=neuralTarget.kind;else ps[k]=parameterValue(k);});let req={profile:$('#profile').value,frames:Number($('#frames').value),params:ps,settings,backend:$('#backend').value,method:$('#method').value};if(parallelDemos.has(current))req.parallel_count=Number($('#parallelCount').value);else if(ensembleSetElsewhere())req.parallel_count=1;if(current==='fluid'&&fluidBuilder)req.obstacle_grid=fluidBuilder.cells;if(current==='neural_wall'&&neuralTarget.custom)req.target_image=$('#targetCanvas').toDataURL('image/png');if(current==='molecular_dynamics'&&chainBuilder&&['fold','default'].includes($('#method').value)){let own=chainBuilder.getChain();if(own)req.chain=own;}window.GameDemos?.decorateRequest(req);return req;}
$('#run').onclick=async()=>{if(!current)return;let req=buildRunRequest();if(!req)return;let cluster=$('#runOn').value;
  if(cluster!=='local'){let id=await HPC.confirmRun(current,req,cluster);if(!id)return;resetRunState();runId=id;$('#status').textContent='SUBMITTING';$('#status').style.color='#ffd78a';timer=setInterval(poll,1000);return;}
  resetRunState();let response=await fetch('/api/run/'+current,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify(req)});if(!response.ok){let detail='Request rejected';try{let body=await response.json();detail=body.detail||detail;}catch(_){ }$('#status').textContent='FAILED TO START';showUiMessage(detail);return;}runId=(await response.json()).id;$('#status').textContent='COMPUTING';$('#status').style.color='#67f0d0';timer=setInterval(poll,300);};
async function poll(){if(!runId)return;let m=await (await fetch('/api/run/'+runId+'?t='+Date.now())).json();let away=HPC.describe(m);if(away){$('#status').textContent=away.badge;$('#status').style.color='#ffd78a';$('#metric1').textContent=away.message;if(away.progress!==null)$('#bar').style.width=`${Math.round(away.progress*100)}%`;return;}if($('#status').style.color)$('#status').style.color='#67f0d0';let overlaysChanged=String(m.overlays)!==String(currentMeta.overlays)||m.method!==currentMeta.method;currentMeta=m;if(overlaysChanged)renderViewerDock();window.GameDemos?.onMeta(m);let total=Number($('#frames').value);if(m.fusion_view)fusionManifest=m.fusion_view;if(m.galaxy3d_view)galaxy3dManifest=m.galaxy3d_view;if(m.frame!==undefined&&m.frame>=0){lastFrame=m.frame;playbackTotal=total;frameOverlay=m.overlay||frameOverlay;$('#frameSeek').max=Math.max(0,total-1);showFrame(m.frame,total);renderOverlayCards();if(current==='fusion_plasma'&&fusionManifest&&preferFusion3d&&!fusionActive&&!fusionEntering)enterFusion();$('#metric2').textContent=`elapsed ${(m.elapsed||0).toFixed(1)} s`;$('#metric3').textContent=`backend ${m.backend||'—'}`;}if(m.status==='complete'){clearInterval(timer);timer=null;deepManifest=m.zoom||null;fusionManifest=m.fusion_view||fusionManifest;galaxy3dManifest=m.galaxy3d_view||galaxy3dManifest;playbackTotal=Number(m.frames)||total;$('#frameSeek').max=Math.max(0,playbackTotal-1);$('#status').textContent='COMPLETE';loadLibrary();$('#metric3').textContent=`backend ${m.backend||'—'}`;setPlaybackControls(true);startPlayback(0);if(current==='fusion_plasma'&&fusionManifest&&preferFusion3d&&!fusionActive)enterFusion();}if(m.status==='failed'){clearInterval(timer);timer=null;$('#status').textContent='FAILED';showUiMessage(m.error||'Simulation failed');}}
async function showReveal(){if(!runId)return;exitFusion();exitGalaxy3d();window.GameDemos?.exit();stopPlayback();let m=await (await fetch('/api/run/'+runId)).json();if(m.reveal){let sw=$('.screenWrap');sw.classList.add('revealing');$('#scaleReveal').classList.add('show');$('#screen').style.opacity=.15;setTimeout(()=>{$('#screen').src=`/runs/${runId}/${m.reveal}?t=${Date.now()}`;$('#screen').style.opacity=1;currentStory=storyLines()[3]||currentStory;renderOverlayCards();},220);}}
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
// In guardian mode a marker's colour is its wall clearance, so the passive
// colour-family names would be a lie. The filter still selects the same subsets.
function updateFusionParticleLabels(){
  const guardian=Boolean(fusion&&fusion.isGuardian()),select=$('#fusionParticles');
  const names=guardian?['All markers','Group 1','Group 2','Group 3','Group 4']:['All particles','Cyan','Amber','Violet','Mint'];
  [...select.options].forEach((option,index)=>{option.textContent=names[index]||option.textContent;});
}
function fusionLayerState(layer){if(!fusion)return false;if(layer==='plasma')return fusion.showPlasma;if(layer==='magnetic')return fusion.showMagnetic;return fusion.showEscapes;}
function updateFusionLayerButtons(){let guardian=Boolean(fusion&&fusion.isGuardian());$('#fusionEscapes').classList.toggle('hidden',!guardian);document.querySelectorAll('[data-fusion-layer]').forEach(button=>{button.classList.toggle('selected',fusionLayerState(button.dataset.fusionLayer));});}
function exitFusion(userChoice=false){if(userChoice)preferFusion3d=false;fusionActive=false;fusionEntering=false;$('#fusionCanvas').classList.add('hidden');$('#fusionTools').classList.add('hidden');$('#screen').classList.remove('hidden');$('#fusionView').textContent='3D live view';}
async function enterFusion(){
  if(!runId||!fusionManifest||current!=='fusion_plasma'||fusionEntering)return;
  fusionEntering=true;preferFusion3d=true;exitDeep();exitGalaxy3d();hideReveal();resetViewport();
  if(!fusion)fusion=new FusionView($('#fusionCanvas'));
  try{
    let url=fusionFrameUrl(playbackFrame);await fusion.load(`${url}?t=${Date.now()}`);
    fusionEntering=false;fusionActive=true;$('#screen').classList.add('hidden');$('#fusionCanvas').classList.remove('hidden');$('#fusionTools').classList.remove('hidden');$('#fusionView').textContent='2D frame';
    fusion.setLayer('plasma',true);fusion.setLayer('magnetic',false);fusion.setLayer('escapes',true);fusion.setParticleFilter($('#fusionParticles').value);updateFusionLayerButtons();updateFusionParticleLabels();fusion.resize();
    currentStory='Drag to rotate the computed torus while playback continues. Turn on the magnetic overlay without hiding the plasma flow, and filter tracer families by colour.';renderOverlayCards();
  }catch(error){fusionEntering=false;exitFusion();showUiMessage(`Interactive view unavailable: ${error.message}`);}
}
$('#fusionView').onclick=()=>fusionActive?exitFusion(true):enterFusion();
document.querySelectorAll('[data-fusion-layer]').forEach(button=>button.onclick=()=>{if(!fusion)return;let layer=button.dataset.fusionLayer;fusion.setLayer(layer,!fusionLayerState(layer));updateFusionLayerButtons();let guardian=fusion.isGuardian();currentStory=layer==='escapes'?(fusion.showEscapes?'Every burst on the wall is a marker the confining field failed to hold.':'Wall losses are hidden; the markers still leave confinement in the simulation.'):fusion.showMagnetic?'Helical lines now overlay the computed plasma. They are illustrative confinement geometry shaped by the coil commands, not a solved tokamak equilibrium.':guardian?'Markers are coloured by how close they are to the wall; the policy is holding them with three coil commands.':'Passive tracers follow drift derived from the computed plasma-wave field.';renderOverlayCards();});
$('#fusionParticles').onchange=()=>{if(!fusion)return;fusion.setParticleFilter($('#fusionParticles').value);currentStory=$('#fusionParticles').value==='all'?'All tracer families are visible.':'Only one colour family is visible; the underlying plasma state is unchanged.';renderOverlayCards();};
$('#fusionReset').onclick=()=>{if(fusion)fusion.reset();};
function has3dView(id){return id==='galaxy_collision_3d'||id==='molecular_dynamics';}
function exitGalaxy3d(){galaxy3dActive=false;$('#galaxy3dCanvas').classList.add('hidden');$('#galaxy3dTools').classList.add('hidden');$('#screen').classList.remove('hidden');$('#galaxy3dView').textContent='Rotate 3D';updateViewport();}
function updateGalaxyViewButtons(){document.querySelectorAll('[data-galaxy-focus]').forEach(button=>button.classList.toggle('selected',galaxy3d&&galaxy3d.focus===button.dataset.galaxyFocus));$('#galaxyHalo').classList.toggle('selected',Boolean(galaxy3d?.showHalo));}
async function enterGalaxy3d(){
  if(!runId||!galaxy3dManifest||!has3dView(current))return;
  exitDeep();exitFusion();hideReveal();resetViewport();
  if(!galaxy3d)galaxy3d=new Galaxy3DView($('#galaxy3dCanvas'));
  try{
    await galaxy3d.load(`/runs/${runId}/${galaxy3dManifest.folder}/frame_${String(playbackFrame).padStart(4,'0')}.json?t=${Date.now()}`);
    galaxy3d.setFocus('all');galaxy3dActive=true;$('#screen').classList.add('hidden');$('#galaxy3dCanvas').classList.remove('hidden');$('#galaxy3dTools').classList.remove('hidden');$('#galaxy3dView').textContent='Exit 3D view';
    updateGalaxyViewButtons();updateViewport();galaxy3d.resize();
    // The galaxy's focus and halo buttons mean nothing for a molecule.
    document.querySelectorAll('[data-galaxy-focus],#galaxyHalo').forEach(b=>b.classList.toggle('hidden',current!=='galaxy_collision_3d'));
    if(current==='molecular_dynamics'){currentStory='Drag to turn the molecule; the wheel zooms. Each frame is the saved 3-D state of every bead.';renderOverlayCards();return;}
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
