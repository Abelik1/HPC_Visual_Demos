// Viewer wiring for the AI game demos: the block builder on the parameter
// panel, the animated arena replay, the brain graph and the leaderboard.
// app.js calls these hooks through window.GameDemos so the core viewer does
// not need to know what a brain is.
(function(){
const GAME_DEMOS=new Set(['neuro_racers','bat_vs_moth']);
Object.assign(stories,{
  neuro_racers:['Every car carries the brain you built, but each starts with different random weights.','Most cars crash or spin in circles. The ones that get furthest become the parents.','Their children inherit mutated copies of those weights, and the driving improves generation by generation.','Nobody programmed the driving. Pull back: independent searches of the same brain end in different places.'],
  bat_vs_moth:['The cave is dark. The screen shows only what the bat’s calls reveal: rings of sound, rock edges, and moth echoes.','At first the moths just flutter. The bats that learn to turn toward the louder ear catch the most.','Now the moths are evolving too. Watch for magenta phantoms: moths that click to fake their own echo.','Pull back: the same bats and moths released into new caves. Does the jamming survive?']
});
Object.assign(panelNames,{neuro_racers:'Evolution readout',bat_vs_moth:'Arms-race readout'});
Object.assign(legends,{neuro_racers:'Gold is the champion of the current generation, blue trails are the next-best cars and red crosses are crashes. Violet cars are ghosts: champions of earlier visitors, re-simulated on this track with their own brains.',
  bat_vs_moth:'Amber is the champion bat and its call rings. Blue dots are rock edges its calls reached and cyan glows are moth echoes it heard (bigger means louder). Magenta rings are phantom echoes from jamming moths. White crosses are catches. Switch to Lit cave to see everything.'});
Object.assign(demoInformation,{neuro_racers:['Evolving drivers from your own brain design','You choose the sensors, hidden neurons and controls. Hundreds of cars with exactly that network, each with different random weights, drive the track at once; the furthest become the parents of the next generation.','This is neuroevolution: no gradients and no human driving examples, only selection, crossover and mutation. The car is a reduced kinematic model with a grip limit, and its distance sensors are sphere-traced through a signed-distance field of the walls.'],
  bat_vs_moth:['Two brains, one arms race','One visitor designs the bat, another the moth. Hundreds of caves hunt at once; bats that catch more and moths that survive longer become the next generation’s parents. The moths only start evolving after a head start, so the bats have something to be countered.','A reduced sonar model, not acoustics: echo loudness falls with distance and a cardioid ear pattern, delay is proportional to distance, and rock does not block sound. A jamming moth replaces its own echo with a random phantom, as tiger-moth clicks protect the clicker.']});
Object.assign(demoCategories,{neuro_racers:'AI & learning',bat_vs_moth:'AI & learning'});
viewModes.bat_vs_moth=[{id:'frames',label:'Bat’s senses',folder:'frames'},{id:'lit',label:'Lit cave',folder:'modes/lit'}];
GAME_DEMOS.forEach(id=>parallelDemos.add(id));

const S={builder:null,builders:null,arena:null,active:false,manifest:null,prefer:true,entering:false,gen:null,pending:null,ghosts:null};
const pad=n=>String(n).padStart(4,'0');
const isGame=(id=current)=>GAME_DEMOS.has(id);
const frameGen=frame=>S.manifest?.frames?.[frame]?.[0]??null;

function topRuns(trackId){
  return library.filter(r=>r.demo==='neuro_racers'&&r.status==='complete'&&r.summary&&Number(r.summary.track_id)===Number(trackId))
    .sort((a,b)=>(a.summary.best_lap_s??1e9)-(b.summary.best_lap_s??1e9)||(b.summary.best_laps-a.summary.best_laps));
}

function mount(_sliders,id){
  S.builder=null;S.builders=null;S.ghosts=null;
  document.querySelectorAll('.gameStage').forEach(node=>node.remove());
  $('#arenaView').classList.toggle('hidden',!isGame(id));
  const catalogue=specs.demos[id]?.brain;if(!catalogue)return;
  // The builder is the main event of these demos, so it gets a full-width
  // row above the ordinary parameter and compute panels.
  const host=document.createElement('section');host.className='gameStage';host.setAttribute('aria-label','Brain builder');
  document.querySelector('#stage .controls').before(host);
  if(catalogue.roles){
    // Two visitors, two builders side by side.
    const duo=document.createElement('div');duo.className='duoBuilders';host.appendChild(duo);
    const bat=catalogue.roles.bat,moth=catalogue.roles.moth;
    S.builders={
      bat:new BrainBuilder(duo,bat,{title:'Visitor 1 · the bat',subtitle:`Up to ${bat.budget} points. Ears hear echoes of your calls; the bat is otherwise blind.`,accent:'#ffbe50',agent:'bat'}),
      moth:new BrainBuilder(duo,moth,{title:'Visitor 2 · the moth',subtitle:`Up to ${moth.budget} points. Hear the bat coming, dodge, or click to jam its sonar.`,accent:'#6ef0c8',agent:'moth'}),
    };
    return;
  }
  S.builder=new BrainBuilder(host,catalogue,{title:'Build your car’s brain',subtitle:`Spend up to ${catalogue.budget} LEGO points on sensors, neurons and controls. Every car in the race gets exactly this brain.`,accent:'#ffc45c',agent:'car'});
  if(id==='neuro_racers'){
    const options=document.createElement('div');options.className='gameOptions';
    options.innerHTML='<label><input type="checkbox" id="raceGhosts"> Race the ghosts</label><span id="ghostHint"></span>';
    host.appendChild(options);S.ghosts=options.querySelector('#raceGhosts');
    const refresh=()=>{const best=topRuns(parameterValue('track')).slice(0,3);S.ghosts.disabled=!best.length;if(!best.length)S.ghosts.checked=false;
      options.querySelector('#ghostHint').textContent=best.length?`The top ${best.length} saved car${best.length===1?'':'s'} on this track will be re-simulated beside yours.`:'No saved champions on this track yet: be the first ghost.';};
    $('#p_track')?.addEventListener('change',refresh);refresh();
  }
}

function decorateRequest(req){
  if(!isGame())return;
  if(S.builders){req.brain=Object.fromEntries(Object.entries(S.builders).map(([role,b])=>[role,b.getSpec()]));return;}
  if(!S.builder)return;
  req.brain=S.builder.getSpec();
  if(S.ghosts?.checked)req.ghosts=topRuns(req.params.track).slice(0,3).map(r=>r.id);
}

async function showGen(gen,{now=false}={}){
  if(!S.arena||!runId||gen===null)return;
  if(!now&&S.gen!==null&&gen!==S.gen){S.pending=gen;return;}
  try{await S.arena.showGeneration(`/runs/${runId}/${S.manifest.folder}/gen_${pad(gen)}.json`);S.gen=gen;S.pending=null;}
  catch(_){/* not written yet while a run is streaming */}
}

// One animation loop per generation: advance when the drive has been seen.
function onLoop(){
  if(!S.active)return;
  if(S.pending!==null){showGen(S.pending,{now:true});return;}
  if(!playbackPlaying||!S.manifest?.frames)return;
  const frames=S.manifest.frames,next=frames.findIndex(([g])=>g>S.gen);
  const available=next>=0&&next<=lastFrame;
  if(available){S.advancing=true;showFrame(next,playbackTotal);S.advancing=false;}
  else if(timer===null){S.advancing=true;showFrame(0,playbackTotal);S.advancing=false;}
}

async function enter(){
  if(!runId||!S.manifest||S.entering||!isGame())return;
  S.entering=true;S.prefer=true;exitDeep();exitFusion();exitGalaxy3d();hideReveal();resetViewport();
  if(!S.arena){S.arena=new ArenaView($('#arenaCanvas'));S.arena.onloop=onLoop;}
  try{
    await S.arena.loadArena(`/runs/${runId}/${S.manifest.arena}`);
    S.active=true;S.gen=null;S.pending=null;
    if(playbackTimer){clearInterval(playbackTimer);playbackTimer=null;}
    $('#screen').classList.add('hidden');$('#arenaCanvas').classList.remove('hidden');$('#arenaTools').classList.remove('hidden');$('#arenaView').textContent='Trail frames';
    S.arena.resize();
    await showGen(frameGen(Math.max(0,playbackFrame))??1,{now:true});
    const lit=$('#arenaLit');if(lit){lit.classList.toggle('hidden',current!=='bat_vs_moth');S.arena.lit=lit.classList.contains('selected');}
    currentStory=current==='bat_vs_moth'?'This is the champion bat’s real recorded hunt, animated. In Bat’s senses you only see what its calls reveal.':'This is the real recorded drive of this generation’s best cars, animated. The champion is gold.';renderOverlayCards();updateViewport();
  }catch(error){exit();showUiMessage(`Animated race unavailable: ${error.message}`);}
  finally{S.entering=false;}
}
function exit(user=false){
  if(user)S.prefer=false;S.active=false;S.gen=null;S.pending=null;S.arena?.stop();
  $('#arenaCanvas').classList.add('hidden');$('#arenaTools').classList.add('hidden');$('#screen').classList.remove('hidden');$('#arenaView').textContent='Animated race';
  if(user&&playbackPlaying){playbackPlaying=false;startPlayback(playbackFrame);}
}

function enableButton(){$('#arenaView').disabled=!(S.manifest&&frameAvailable()&&isGame());}
function onMeta(m){
  if(!isGame())return;
  if(m.arena_view)S.manifest=m.arena_view;
  if(S.manifest&&m.frame>=0){enableButton();if(S.prefer&&!S.active&&!S.entering)enter();}
}
function onFrame(frame){
  if(!S.active)return;
  const gen=frameGen(frame);if(gen===null)return;
  // Seeking, or a loop-driven advance, shows the generation at once; live
  // streaming queues it until the current drive has finished.
  showGen(gen,{now:S.advancing||!playbackPlaying&&timer===null});
}
function onOpenRun(r){
  if(!isGame(r.demo))return;
  S.manifest=r.arena_view||null;
  const brain=r.params?._brain;
  if(brain&&S.builders)Object.entries(S.builders).forEach(([role,b])=>{if(brain[role])b.setSpec(brain[role]);});
  else if(brain&&S.builder)S.builder.setSpec(brain);
  enableButton();if(S.manifest&&S.prefer)enter();
}
function reset(){exit();S.manifest=null;$('#arenaView').disabled=true;}

function dockItems(items){
  if(current==='neuro_racers')items.push(['network','Brain graph'],['leaderboard','Leaderboard']);
  if(current==='bat_vs_moth')items.push(['network','Both brains'],['arms','Arms race']);
}
function overlayCards(wanted){
  if(!isGame())return;
  if(overlayEnabled.has('arms')&&runId){wanted.add('arms');const card=overlayCard('arms','Arms race','image'),img=card.querySelector('img'),src=`/runs/${runId}/overlays/arms_race/frame_${pad(playbackFrame)}.jpg`;img.alt='Catch rate and moth jamming by generation';if(img.dataset.src!==src){img.dataset.src=src;img.src=src;}}
  if(overlayEnabled.has('network')&&runId){wanted.add('network');const card=overlayCard('network',current==='bat_vs_moth'?'Champion bat and moth brains':'Champion’s brain','image'),img=card.querySelector('img'),src=`/runs/${runId}/overlays/network/frame_${pad(playbackFrame)}.jpg`;if(img.dataset.src!==src){img.dataset.src=src;img.src=src;}}
  if(overlayEnabled.has('leaderboard')&&current==='neuro_racers'){
    wanted.add('leaderboard');const track=Number(currentMeta?.params?.track??parameterValue('track'));
    const name=specs.demos.neuro_racers.params.track.options[String(track)]||'this track';
    const card=overlayCard('leaderboard',`Leaderboard · ${name}`,'rows'),rows={};
    topRuns(track).slice(0,6).forEach((r,i)=>{const s=r.summary;rows[`${i+1}. ${r.id===runId?'this run':runLabel(r)}`]=`${s.best_lap_s?`${s.best_lap_s.toFixed(1)} s lap`:`${s.best_laps.toFixed(2)} laps`} · ${s.brain_points} pts`;});
    if(!Object.keys(rows).length)rows['no saved runs']='finish a run to set the first time';
    updateOverlayRows(card.querySelector('.overlayRows'),rows);
  }
}

// In arena mode the drive itself sets the pace, so Play means "keep
// advancing generations" rather than the fixed-rate frame timer.
const baseStartPlayback=startPlayback;
startPlayback=function(frame=playbackFrame){
  if(isGame()&&!S.active&&S.prefer&&S.manifest&&runId){enter();}
  if(S.active){if(playbackTimer){clearInterval(playbackTimer);playbackTimer=null;}playbackPlaying=true;updatePlaybackButton();S.advancing=true;showFrame(frame,playbackTotal);S.advancing=false;return;}
  return baseStartPlayback(frame);
};

$('#arenaView').onclick=()=>S.active?exit(true):enter();
$('#arenaFollow').onclick=()=>{if(!S.arena)return;S.arena.follow=!S.arena.follow;$('#arenaFollow').classList.toggle('selected',S.arena.follow);};
$('#arenaTrails').onclick=()=>{if(!S.arena)return;S.arena.trails=!S.arena.trails;$('#arenaTrails').classList.toggle('selected',S.arena.trails);};
// Senses/lit toggle for the Bat vs Moth arena, added beside the other tools.
(()=>{const b=document.createElement('button');b.id='arenaLit';b.className='hidden';b.title='Show the whole cave instead of only what the bat senses';b.textContent='Lit cave';
  b.onclick=()=>{if(!S.arena)return;S.arena.lit=!S.arena.lit;b.classList.toggle('selected',S.arena.lit);};$('#arenaTools').appendChild(b);})();
$('#playbackRate').addEventListener('change',()=>{if(S.arena)S.arena.rate=Number($('#playbackRate').value)||1;});
window.addEventListener('resize',()=>{if(S.active&&S.arena)S.arena.resize();});

window.GameDemos={mount,decorateRequest,onMeta,onFrame,onOpenRun,reset,exit,dockItems,overlayCards,get active(){return S.active;},debug:()=>({active:S.active,gen:S.gen,pending:S.pending,prefer:S.prefer,frames:S.manifest?.frames?.length,cycle:S.arena?.cycle})};
})();
