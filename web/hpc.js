// Demo-day lineups and cluster runs, shared by the dashboard (/) and demo mode (/demo).
//
// * Lineups: which demos the Discoverer day and the Leonardo day show, in
//   order, plus recorded-video "demos" such as the raytracer. Stored by the
//   server in config/lineups.json, so both front ends and every browser agree.
// * Cluster runs: the page builds its usual run request; HPC.confirmRun shows
//   every setting that will be used and, on confirmation, submits it to
//   Discoverer or Leonardo. The server syncs the code, runs sbatch, waits, and
//   fetches the finished run into runs/, so the page's ordinary polling of
//   /api/run/<id> simply sees it complete and plays it.
window.HPC=(()=>{
  const esc=v=>String(v??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
  let lineup={active:'all',machines:{},archived:[],extras:{}},clusters={},jobs=[];
  const listeners=new Set();

  async function load(){
    const [l,c]=await Promise.all([fetch('/api/lineups',{cache:'no-store'}).then(r=>r.json()).catch(()=>null),
                                   fetch('/api/clusters',{cache:'no-store'}).then(r=>r.json()).catch(()=>null)]);
    if(l)lineup=l;if(c){clusters=c.clusters||{};jobs=c.jobs||[];}
    return lineup;
  }
  function changed(){listeners.forEach(fn=>{try{fn(lineup);}catch(e){console.error(e);}});}
  function onChange(fn){listeners.add(fn);}
  // The lineup is usually edited in another window (the presenter dashboard)
  // while the stand is open, so re-read it now and then and pass on changes.
  function watch(ms=10000){
    setInterval(async()=>{
      const before=JSON.stringify(lineup);
      try{const r=await fetch('/api/lineups',{cache:'no-store'});if(r.ok)lineup=await r.json();}catch(_){return;}
      if(JSON.stringify(lineup)!==before)changed();
    },ms);
  }

  // ---------------------------------------------------------------- lineup --
  const active=()=>lineup.active||'all';
  const machineLabel=id=>lineup.machines?.[id]?.label||id;
  const isArchived=id=>(lineup.archived||[]).includes(id);
  const extra=id=>lineup.extras?.[id]||null;
  function machinesOf(id){return Object.entries(lineup.machines||{}).filter(([,m])=>(m.demos||[]).includes(id)).map(([k])=>k);}
  // The demos (and video items) to show, in order, for the active machine.
  // `demoIds` is every simulation the server knows, in its natural order.
  function items(demoIds){
    const known=new Set([...demoIds,...Object.keys(lineup.extras||{})]);
    if(active()==='all'){
      const ordered=[];
      Object.values(lineup.machines||{}).forEach(m=>(m.demos||[]).forEach(id=>{if(known.has(id)&&!ordered.includes(id))ordered.push(id);}));
      [...demoIds,...Object.keys(lineup.extras||{})].forEach(id=>{if(!ordered.includes(id))ordered.push(id);});
      return ordered.filter(id=>!isArchived(id));
    }
    return (lineup.machines[active()]?.demos||[]).filter(id=>known.has(id));
  }
  function videoHref(id,{fromDemo=false}={}){
    const x=extra(id);if(!x)return '/videos';
    const q=new URLSearchParams({folder:x.folder,title:x.name});if(fromDemo)q.set('from','demo');
    return '/videos?'+q.toString();
  }
  async function setActive(machine){
    const r=await fetch('/api/lineups/active',{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify({active:machine})});
    if(r.ok){lineup=await r.json();changed();}
  }
  // A segmented control: [Discoverer] [Leonardo] [All]
  function machineSwitch(host){
    const render=()=>{
      host.classList.add('hpcSwitch');host.setAttribute('role','group');host.setAttribute('aria-label','Demo day machine');
      const options=[...Object.keys(lineup.machines||{}),'all'];
      host.innerHTML='<span class="hpcSwitchLabel">Demo day</span>'+options.map(id=>
        `<button type="button" data-machine="${esc(id)}" aria-pressed="${id===active()}">${esc(id==='all'?'All demos':machineLabel(id))}</button>`).join('');
      host.querySelectorAll('button').forEach(b=>b.onclick=()=>setActive(b.dataset.machine));
    };
    render();onChange(render);return host;
  }

  // ----------------------------------------------------------------- modal --
  function modal(title,{wide=false}={}){
    const back=document.createElement('div');back.className='hpcModal';back.setAttribute('role','dialog');back.setAttribute('aria-modal','true');
    back.innerHTML=`<div class="hpcPanel${wide?' wide':''}"><header><b>${esc(title)}</b><button type="button" class="hpcClose" aria-label="Close">✕</button></header><div class="hpcBody"></div><footer></footer></div>`;
    document.body.appendChild(back);
    const close=()=>{back.remove();document.removeEventListener('keydown',key);};
    const key=e=>{if(e.key==='Escape')close();};
    document.addEventListener('keydown',key);
    back.querySelector('.hpcClose').onclick=close;
    back.addEventListener('mousedown',e=>{if(e.target===back)close();});
    return {el:back,body:back.querySelector('.hpcBody'),foot:back.querySelector('footer'),close};
  }

  // -------------------------------------------------------------- settings --
  // `demos` is specs.demos: {id:{name,...}}
  async function openSettings(demos,{tab='lineups'}={}){
    await load();
    const m=modal('Demo days & HPC',{wide:true});
    m.body.innerHTML=`<nav class="hpcTabs"><button data-tab="lineups">Demo-day lineups</button><button data-tab="clusters">Clusters</button><button data-tab="jobs">Cluster jobs</button></nav><div class="hpcTab"></div>`;
    const pane=m.body.querySelector('.hpcTab');
    const show=name=>{m.body.querySelectorAll('.hpcTabs button').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.tab===name)));m.foot.innerHTML='';
      ({lineups:lineupEditor,clusters:clusterEditor,jobs:jobList})[name](pane,m,demos);};
    m.body.querySelectorAll('.hpcTabs button').forEach(b=>b.onclick=()=>show(b.dataset.tab));
    show(tab);
  }

  function lineupEditor(pane,m,demos){
    const draft=JSON.parse(JSON.stringify(lineup));
    Object.values(draft.extras||{}).forEach(x=>delete x.videos);
    const name=id=>demos[id]?.name||draft.extras[id]?.name||id;
    const columns=()=>{
      const assigned=new Set([...Object.values(draft.machines).flatMap(x=>x.demos),...draft.archived]);
      const unassigned=[...Object.keys(demos),...Object.keys(draft.extras)].filter(id=>!assigned.has(id));
      return [...Object.entries(draft.machines).map(([k,x])=>({key:k,label:x.label,list:x.demos})),
              {key:'_none',label:'Not shown',list:unassigned,fixed:true},{key:'_archive',label:'Archive',list:draft.archived}];
    };
    const listOf=key=>key==='_archive'?draft.archived:draft.machines[key]?.demos;
    const move=(id,from,to)=>{
      const src=listOf(from);if(src){const i=src.indexOf(id);if(i>=0)src.splice(i,1);}
      if(to==='_archive'||to in draft.machines){
        // A demo can be in both machines' lineups, but never shown and archived at once.
        if(to==='_archive')Object.values(draft.machines).forEach(x=>{const i=x.demos.indexOf(id);if(i>=0)x.demos.splice(i,1);});
        else{const i=draft.archived.indexOf(id);if(i>=0)draft.archived.splice(i,1);}
        const dst=listOf(to);if(!dst.includes(id))dst.push(id);
      }
      render();
    };
    const render=()=>{
      const cols=columns();
      pane.innerHTML=`<p class="hpcHint">Each demo day shows its own list, in this order, in both the dashboard and demo mode. Switch the active day with the <b>Demo day</b> buttons. Video items play the files in their folder under <code>videos/</code>.</p>
      <div class="hpcColumns">${cols.map(col=>`<section class="hpcCol" data-col="${esc(col.key)}"><h4>${esc(col.label)} <small>${col.list.length}</small></h4>
        <ol>${col.list.map((id,i)=>`<li data-id="${esc(id)}"><span class="hpcItem">${esc(name(id))}${draft.extras[id]?' <i class="hpcTag">video</i>':''}</span>
          <span class="hpcItemTools">${col.fixed?'':`<button data-act="up" ${i===0?'disabled':''} title="Move up">↑</button><button data-act="down" ${i===col.list.length-1?'disabled':''} title="Move down">↓</button>`}
          <select data-act="to" aria-label="Move ${esc(name(id))}"><option value="">Move to…</option>${cols.filter(c=>c.key!==col.key).map(c=>c.key in draft.machines
            ?`<option value="${esc(c.key)}">Move to ${esc(c.label)}</option>${col.key in draft.machines?`<option value="+${esc(c.key)}">Also show on ${esc(c.label)}</option>`:''}`
            :`<option value="${esc(c.key)}">${esc(c.label)}</option>`).join('')}</select></span></li>`).join('')||'<li class="hpcEmpty">Nothing here</li>'}</ol></section>`).join('')}</div>
      <h4 class="hpcSub">Video items</h4>
      <div class="hpcExtras">${Object.entries(draft.extras).map(([id,x])=>`<div class="hpcExtra" data-extra="${esc(id)}">
        <label>Name <input data-f="name" value="${esc(x.name)}"></label>
        <label>Folder in videos/ <input data-f="folder" value="${esc(x.folder)}"></label>
        <label class="span2">One-line description <input data-f="tagline" value="${esc(x.tagline)}"></label>
        <label class="span2">Optional web link (opens instead of the videos) <input data-f="url" value="${esc(x.url||'')}" placeholder="https://…"></label>
        <span class="hpcHint">${(lineup.extras[id]?.videos??0)} video file(s) found</span><button data-remove="${esc(id)}" class="hpcLink">Remove</button></div>`).join('')}
        <button id="hpcAddExtra" class="hpcSecondary">+ Add a video item</button></div>`;
      pane.querySelectorAll('.hpcCol li[data-id]').forEach(li=>{
        const id=li.dataset.id,col=li.closest('.hpcCol').dataset.col;
        li.querySelectorAll('button[data-act]').forEach(b=>b.onclick=()=>{const list=listOf(col),i=list.indexOf(id),j=i+(b.dataset.act==='up'?-1:1);[list[i],list[j]]=[list[j],list[i]];render();});
        li.querySelector('select').onchange=e=>{const to=e.target.value;if(!to)return;
          // "Also show on" keeps it on this day too; every other choice moves it.
          if(to.startsWith('+')){const dst=draft.machines[to.slice(1)]?.demos;if(dst&&!dst.includes(id))dst.push(id);render();}
          else move(id,col,to);};
      });
      pane.querySelectorAll('.hpcExtra input').forEach(input=>input.oninput=()=>{draft.extras[input.closest('.hpcExtra').dataset.extra][input.dataset.f]=input.value;});
      pane.querySelectorAll('[data-remove]').forEach(b=>b.onclick=()=>{const id=b.dataset.remove;delete draft.extras[id];
        Object.values(draft.machines).forEach(x=>x.demos=x.demos.filter(d=>d!==id));draft.archived=draft.archived.filter(d=>d!==id);render();});
      pane.querySelector('#hpcAddExtra').onclick=()=>{let n=1;while(draft.extras['video_'+n])n++;
        draft.extras['video_'+n]={name:'New video demo',tagline:'',category:'Physics',kind:'video',folder:'video_'+n,url:''};render();};
    };
    render();
    m.foot.innerHTML='<span class="hpcStatus"></span><button class="hpcSecondary" data-close>Close</button><button class="hpcPrimary" data-save>Save lineups</button>';
    m.foot.querySelector('[data-close]').onclick=m.close;
    m.foot.querySelector('[data-save]').onclick=async()=>{
      const status=m.foot.querySelector('.hpcStatus');status.textContent='Saving…';
      const r=await fetch('/api/lineups',{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify(draft)});
      if(!r.ok){status.textContent=(await r.json().catch(()=>({}))).detail||'Could not save';status.classList.add('bad');return;}
      lineup=await r.json();status.classList.remove('bad');status.textContent='Saved';changed();
    };
  }

  const FIELDS=[['user','User'],['host','Login host'],['port','Port'],['account','Slurm account'],['qos','QoS'],['partition','Partition'],
                ['walltime','Default time limit'],['default_profile','Preset it expects'],['root','Code checkout'],['runs_root','Run folder'],
                ['python','Python'],['identity','SSH key (blank = default)'],['cert_email','CINECA UserDB e-mail']];
  function clusterEditor(pane){
    pane.innerHTML=`<p class="hpcHint">The dashboard connects with this PC's own <code>ssh</code>, exactly as in a terminal. Paths may use <code>$WORK</code> / <code>$FAST</code>; they expand on the cluster.</p>`+
      Object.values(clusters).map(c=>{const cert=c.certificate;return `<section class="hpcCluster" data-cluster="${esc(c.name)}">
        <h4>${esc(c.label)} <small>${esc(c.resources_note||'')}</small></h4>
        <div class="hpcFields">${FIELDS.filter(([k])=>k!=='cert_email'||cert).map(([k,l])=>`<label>${esc(l)}<input data-k="${k}" value="${esc(c[k]??'')}"></label>`).join('')}</div>
        ${cert?`<p class="hpcCert ${cert.valid?'good':'bad'}">SSH certificate: ${esc(cert.detail)} <button class="hpcSecondary" data-cert>Refresh certificate</button></p>`:''}
        <div class="hpcRow"><button class="hpcSecondary" data-save>Save</button><button class="hpcPrimary" data-check>Test connection</button><span class="hpcStatus"></span></div>
        <ul class="hpcChecks"></ul></section>`;}).join('');
    pane.querySelectorAll('.hpcCluster').forEach(sec=>{
      const name=sec.dataset.cluster,status=sec.querySelector('.hpcStatus');
      sec.querySelector('[data-save]').onclick=async()=>{
        const values={};sec.querySelectorAll('input[data-k]').forEach(i=>{if(String(clusters[name][i.dataset.k]??'')!==i.value)values[i.dataset.k]=i.value;});
        if(!Object.keys(values).length){status.textContent='Nothing changed';return;}
        const r=await fetch('/api/clusters/'+name,{method:'PUT',headers:{'content-type':'application/json'},body:JSON.stringify({values})});
        const body=await r.json().catch(()=>({}));
        if(!r.ok){status.textContent=body.detail||'Could not save';status.className='hpcStatus bad';return;}
        Object.assign(clusters[name],body);status.textContent='Saved';status.className='hpcStatus good';
      };
      sec.querySelector('[data-check]').onclick=async()=>{
        status.textContent='Connecting…';status.className='hpcStatus';const list=sec.querySelector('.hpcChecks');list.innerHTML='';
        const r=await fetch(`/api/clusters/${name}/check`,{method:'POST'});const body=await r.json().catch(()=>({items:[]}));
        list.innerHTML=(body.items||[]).map(i=>`<li class="${i.ok?'good':'bad'}"><b>${i.ok?'✓':'✗'} ${esc(i.name)}</b> ${esc(i.detail)}</li>`).join('');
        status.textContent=body.ok?'Ready to run':'Needs attention';status.className='hpcStatus '+(body.ok?'good':'bad');
      };
      const cert=sec.querySelector('[data-cert]');
      if(cert)cert.onclick=async()=>{
        const r=await fetch(`/api/clusters/${name}/certificate`,{method:'POST'});const body=await r.json().catch(()=>({}));
        status.textContent=r.ok?'A PowerShell window opened: sign in there (browser, password and one-time code), then press Test connection.':(body.detail||'Could not start the certificate helper');
        status.className='hpcStatus '+(r.ok?'':'bad');
      };
    });
  }

  function jobList(pane){
    const render=()=>{
      pane.innerHTML=jobs.length?`<table class="hpcTable"><thead><tr><th>Run</th><th>Where</th><th>Job</th><th>State</th><th></th></tr></thead><tbody>${jobs.map(j=>`<tr>
        <td>${esc(j.demo)}<small>${esc(j.id)}</small></td><td>${esc(j.label||j.cluster)}</td><td>${esc(j.job_id??'—')}</td>
        <td class="${j.status==='failed'?'bad':j.status==='complete'?'good':''}">${esc(j.status==='remote'?(j.message||j.stage):j.status==='failed'?(j.error||'failed'):j.message||j.status)}</td>
        <td>${j.status==='remote'?`<button class="hpcSecondary" data-cancel="${esc(j.id)}">Cancel</button>`:j.status==='complete'?`<a class="hpcLink" href="/?run=${encodeURIComponent(j.id)}">Open</a>`:''}</td></tr>`).join('')}</tbody></table>`
        :'<p class="hpcHint">No cluster runs in the last few hours.</p>';
      pane.querySelectorAll('[data-cancel]').forEach(b=>b.onclick=async()=>{b.disabled=true;await cancel(b.dataset.cancel);await load();render();});
    };
    render();
  }

  // ---------------------------------------------------------- cluster runs --
  const clusterList=()=>Object.values(clusters);
  async function cancel(id){
    const r=await fetch(`/api/remote/cancel/${encodeURIComponent(id)}`,{method:'POST'});
    if(!r.ok)alert((await r.json().catch(()=>({}))).detail||'Could not cancel');
  }
  // Show everything that will run, and submit on confirmation.
  // Resolves to the new run id, or null if the presenter backed out.
  function confirmRun(demo,request,cluster){
    return new Promise(async resolve=>{
      const label=clusters[cluster]?.label||cluster;
      const m=modal(`Run on ${label}?`,{wide:true});
      let done=false;const finish=v=>{if(done)return;done=true;m.close();resolve(v);};
      m.el.querySelector('.hpcClose').onclick=()=>finish(null);
      m.el.addEventListener('mousedown',e=>{if(e.target===m.el)finish(null);});
      document.addEventListener('keydown',function esc_(e){if(e.key==='Escape'){document.removeEventListener('keydown',esc_);finish(null);}});
      m.body.innerHTML='<p class="hpcHint">Checking the configuration…</p>';
      let walltime=null;
      const plan=async()=>{
        const r=await fetch(`/api/remote/plan/${encodeURIComponent(demo)}`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({cluster,walltime,request})});
        const body=await r.json().catch(()=>({}));
        if(!r.ok){m.body.innerHTML=`<p class="hpcWarn">${esc(body.detail||'This run cannot be sent to the cluster.')}</p>`;
          m.foot.innerHTML='<button class="hpcSecondary">Close</button>';m.foot.querySelector('button').onclick=()=>finish(null);return;}
        const p=body,res=p.resources;walltime=res.walltime;
        const fmt=v=>typeof v==='number'?(Number.isInteger(v)?v.toLocaleString():String(+v.toFixed(4))):esc(v);
        const optionText=(x,v)=>x.options?.[String(Math.round(v))]||fmt(v);
        m.body.innerHTML=`${p.warnings.map(w=>`<p class="hpcWarn">⚠ ${esc(w)}</p>`).join('')}
          <div class="hpcSummary">
            <div><small>Experiment</small><b>${esc(p.demo.name)}</b></div><div><small>Solver</small><b>${esc(p.method_label)}</b></div>
            <div><small>Quality preset</small><b>${esc(p.profile)}</b></div><div><small>Frames</small><b>${fmt(p.frames)}</b></div>
            <div><small>Compute</small><b>${esc(p.backend)}</b></div><div><small>Precision</small><b>${esc(p.precision)}</b></div></div>
          <h4 class="hpcSub">Experiment settings</h4>
          <table class="hpcTable"><tbody>${p.params.map(x=>`<tr class="${x.value!==x.default?'changed':''}"><td>${esc(x.label)}</td><td>${optionText(x,x.value)}</td><td><small>${x.value!==x.default?'changed · default '+optionText(x,x.default):'default'}</small></td></tr>`).join('')}</tbody></table>
          ${p.extras.length?`<ul class="hpcList">${p.extras.map(x=>`<li>${esc(x)}</li>`).join('')}</ul>`:''}
          <details class="hpcDetails"${p.settings.some(s=>s.changed)?' open':''}><summary>Simulation values (${p.settings.filter(s=>s.changed).length} changed from the ${esc(p.profile)} preset)</summary>
            <table class="hpcTable"><tbody>${p.settings.map(s=>`<tr class="${s.changed?'changed':''}"><td>${esc(s.key.replace(/_/g,' '))}</td><td>${fmt(s.value)}</td><td><small>${s.changed?'preset '+fmt(s.preset):''}</small></td></tr>`).join('')}</tbody></table></details>
          <h4 class="hpcSub">${esc(label)} resources</h4>
          <div class="hpcSummary">
            <div><small>Account</small><b>${esc(res.account||'—')}</b></div><div><small>QoS</small><b>${esc(res.qos||'—')}</b></div>
            <div><small>Partition</small><b>${esc(res.partition||'default')}</b></div>
            <div><small>Time limit</small><input id="hpcWalltime" value="${esc(res.walltime)}" size="10" aria-label="Time limit"></div></div>
          <p class="hpcHint">${esc(res.note)}<br><code>sbatch ${esc(res.sbatch.join(' '))}</code><br><code>srun ${esc(res.srun.join(' '))} ${esc(p.python)} tools/run_job.py</code></p>
          <p class="hpcHint">The code is synced first if it changed. When the job ends the run is copied into <code>runs/</code> and plays here automatically. You can close this page; the viewer keeps watching the job.</p>`;
        m.body.querySelector('#hpcWalltime').onchange=e=>{walltime=e.target.value.trim();plan();};
        m.foot.innerHTML=`<span class="hpcStatus"></span><button class="hpcSecondary" data-no>Cancel</button><button class="hpcPrimary" data-yes>Submit to ${esc(label)}</button>`;
        m.foot.querySelector('[data-no]').onclick=()=>finish(null);
        m.foot.querySelector('[data-yes]').onclick=async e=>{
          e.target.disabled=true;const status=m.foot.querySelector('.hpcStatus');status.textContent='Submitting…';
          const r=await fetch(`/api/remote/run/${encodeURIComponent(demo)}`,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({cluster,walltime,request})});
          const out=await r.json().catch(()=>({}));
          if(!r.ok){status.textContent=out.detail||'Rejected';status.className='hpcStatus bad';e.target.disabled=false;return;}
          finish(out.id);
        };
      };
      plan();
    });
  }
  // One line for a run that is waiting on a cluster, or null.
  function describe(meta){
    if(meta?.status!=='remote')return null;
    const r=meta.remote||{};
    return {stage:r.stage,label:r.label||r.cluster,message:r.message||`Waiting for ${r.label||r.cluster}`,
            badge:({queued:'QUEUED',running:'RUNNING',fetching:'FETCHING',syncing:'SYNCING',submitting:'SUBMITTING'}[r.stage]||'PREPARING')+' · '+String(r.label||r.cluster).toUpperCase(),
            progress:r.total&&r.frame>=0?(r.frame+1)/r.total:null};
  }
  return {load,onChange,watch,active,machineLabel,isArchived,extra,machinesOf,items,videoHref,setActive,machineSwitch,
          openSettings,confirmRun,describe,cancel,clusterList,get lineup(){return lineup;}};
})();
