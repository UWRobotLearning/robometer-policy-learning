'use strict';
const $=id=>document.getElementById(id);
const escape=v=>String(v??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
let latestData;
let inactiveControls;
let inactiveControlsRequest;
const mediaView={entries:[],overlayStep:null,overlayItem:null,videoItem:null,videoSignature:null};
let initialPhase = true;
function draw(){
 const cohortLabel=latestData.audit?.cohort_label??'SVM';
 const d=latestData,bc=$('stage').value==='bc',training=$('stage').value==='training',mode=$('mode').value,meanView=!bc&&!training&&$('cohort').value==='mean';
 $('mode').disabled=bc||training;$('cohort').disabled=bc||training;$('show-training').disabled=bc||training;
 $('show-inactive-controls').disabled=bc||training;
 $('chart-title').textContent=training?'Training success · last 20 completed episodes':bc?'BC success versus training steps':meanView?'RL seed mean and 95% confidence interval':'RL individual seed success';
 let rows=training?(d.training_episodes??[]).map(r=>({...r,step:r.physical_step,kind:'Training rolling success'})):bc?(d.bc_evaluations??[]).map(r=>({...r,kind:'BC '+r.mode})):
 [...d.baseline.filter(r=>r.mode===mode).map(r=>({...r,kind:'Historical sparse'})),...d.evaluations.filter(r=>r.mode===mode).map(r=>({...r,kind:cohortLabel}))];
 if(meanView)rows=Object.entries(d.summaries??{}).flatMap(([kind,points])=>points.filter(p=>p.mode===mode).map(p=>({...p,kind:kind==='SVM'?cohortLabel:kind,success:p.mean})));
 if(!bc&&!training&&$('show-inactive-controls').checked&&inactiveControls){
  const controls=meanView?inactiveControls.summaries:inactiveControls.evaluations;
  rows.push(...controls.filter(p=>p.mode===mode).map(p=>({...p,kind:inactiveControls.label,success:meanView?p.mean:p.success,inactiveControl:true})));
 }
 if(!bc&&!training&&$('show-training').checked)rows.push(...(d.training_episodes??[]).map(r=>({...r,step:r.physical_step,kind:'Training rolling success',rolling:true})));
 const groups=new Map();rows.forEach(r=>{const key=meanView&&!r.rolling?r.kind:r.kind+' · seed '+r.seed;if(!groups.has(key))groups.set(key,[]);groups.get(key).push(r)});
 const W=850,H=310,L=60,R=20,T=20,B=60,max=bc?(d.bc_budget??5000):100000;
 const x=v=>L+(W-L-R)*v/max,y=v=>T+(H-T-B)*(1-v);
 let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${training?'Running training success over the last20 completed episodes':bc?'BC ODE and SDE success versus optimizer training steps':mode+' RL success versus physical training steps'}"><title>${training?'Training success: last20 completed episodes':bc?'BC evaluations by training step':'Task 2: historical sparse and '+cohortLabel+' '+mode+' success'}</title>`;
 if(bc)svg+=`<rect x="${L}" y="${y(.5)}" width="${W-L-R}" height="${y(.25)-y(.5)}" fill="#e2f4ec"/><text x="${W-R-6}" y="${y(.5)+17}" text-anchor="end" fill="#33775a" font-size="13">ODE selection target: 25–50%</text>`;
 [0,.25,.5,.75,1].forEach(v=>svg+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#dce4ec"/><text x="${L-10}" y="${y(v)+4}" text-anchor="end" fill="#5b6d7e" font-size="14">${v*100}%</text>`);
 for(let i=0;i<=max;i+=max/5)svg+=`<text x="${x(i)}" y="${H-B+23}" text-anchor="middle" fill="#5b6d7e" font-size="14">${i>=1000?i/1000+'k':i}</text>`;
 svg+=`<text x="${W/2}" y="${H-9}" text-anchor="middle" fill="#5b6d7e" font-size="15">${bc?'BC training steps (optimizer updates)':'Physical training steps per environment'}</text>`;
 let legend='',table=meanView?'<table><thead><tr><th>Reward</th><th>Step</th><th>Mean success</th><th>95% t interval</th><th>Seeds</th></tr></thead><tbody>':'<table><thead><tr><th>Run / sampler</th><th>Success</th><th>Training steps</th></tr></thead><tbody>';
 groups.forEach((points,key)=>{
  points.sort((a,b)=>a.step-b.step);
  const isRolling=training||Boolean(points[0].rolling),groupMean=meanView&&!isRolling;
  const svm=key.startsWith(cohortLabel),color=points[0].inactiveControl?'#6953a4':bc?(points[0].mode==='ODE'?'#087e70':'#6953a4'):(svm?'#087e70':meanView||points[0].seed===100?'#b85b30':'#98a9bb');
  const width=isRolling?1.3:groupMean||bc||svm||points[0].seed===100?2.5:1.5,dash=isRolling?'4 3':bc&&points[0].mode==='SDE'?'6 4':'';
  if(groupMean){
   // Leave gaps where a seed interval is unavailable; do not invent an n=1 CI.
   let segment=[];
   const band=()=>{if(segment.length>1){const top=segment.map(p=>`${x(p.step)},${y(Math.min(1,p.high))}`);const bottom=[...segment].reverse().map(p=>`${x(p.step)},${y(Math.max(0,p.low))}`);svg+=`<polygon points="${[...top,...bottom].join(' ')}" fill="${color}" opacity=".16"/>`;}segment=[];};
   points.forEach(p=>{if(p.n>=2)segment.push(p);else band();});band();
  }
  const plotPoints=[];
  points.forEach((p,i)=>{if(isRolling&&i)plotPoints.push(`${x(p.step)},${y(points[i-1].success)}`);plotPoints.push(`${x(p.step)},${y(p.success)}`);});
  if(isRolling){const endpoint=d.runs?.find(r=>r.seed===points[0].seed)?.physical_steps??d.physical_steps;const last=points.at(-1);if(endpoint>last.step)plotPoints.push(`${x(endpoint)},${y(last.success)}`);}
  svg+=`<polyline points="${plotPoints.join(' ')}" fill="none" stroke="${color}" stroke-width="${width}" stroke-dasharray="${dash}"/>`;
  points.forEach(p=>{
   const chosen=bc&&p.mode==='ODE'&&d.bc_selection?.bc_updates===p.step;
   if(groupMean&&p.n>=2){const top=y(Math.min(1,p.high)),bottom=y(Math.max(0,p.low)),px=x(p.step);svg+=`<path d="M${px} ${top}V${bottom}M${px-3} ${top}H${px+3}M${px-3} ${bottom}H${px+3}" stroke="${color}" stroke-width="1" opacity=".55" fill="none"/>`;}

   svg+=`<circle cx="${x(p.step)}" cy="${y(p.success)}" r="${chosen?6:3.5}" fill="${color}" ${chosen?'stroke="#172c40" stroke-width="2"':''}><title>${escape(key)}: ${(p.success*100).toFixed(1)}% @ ${p.step} ${bc?'BC updates':'physical steps'}${groupMean?' · n='+p.n+' seeds'+(p.low==null?' · CI unavailable':` · 95% t CI [${(p.low*100).toFixed(1)}%, ${(p.high*100).toFixed(1)}%]`):''}${isRolling?' · '+p.window_successes+'/'+p.window_episodes+' in window · '+p.completed_episodes+' completed episodes':''}${chosen?' · selected checkpoint':''}</title></circle>`;
  });
  legend+=`<span><i class="swatch" style="background:${color}"></i>${escape(key)}${groupMean?(points.at(-1).n>=2?' · mean ± 95% CI (n='+points.at(-1).n+' at latest step)':' · mean; latest CI unavailable (n=1)'):''}${isRolling?' · rolling 20 (dashed)':''}${bc&&points[0].mode==='SDE'?' (dashed)':''}</span>`;
  const shown=bc||groupMean||training?points:[points.at(-1)];
  shown.forEach(p=>table+=groupMean?`<tr><td>${escape(key)}</td><td>${p.step.toLocaleString()}</td><td>${(100*p.mean).toFixed(1)}%</td><td>${p.low==null?'Unavailable (n=1)':(p.low*100).toFixed(1)+'% – '+(p.high*100).toFixed(1)+'%'}</td><td>${p.n} (${p.seeds.join(', ')})</td></tr>`:meanView?`<tr><td>${escape(key)}</td><td>${p.step.toLocaleString()}</td><td>${(100*p.success).toFixed(1)}%</td><td>Window: ${p.window_successes}/${p.window_episodes}</td><td>${p.completed_episodes} completed episodes</td></tr>`:`<tr><td>${escape(key)}</td><td>${(100*p.success).toFixed(1)}%</td><td>${p.step.toLocaleString()}</td></tr>`);
 });
 if(!rows.length)svg+=`<text x="${W/2}" y="${H/2}" text-anchor="middle" font-size="15" fill="#5b6d7e">${training?'No completed-training-episode logs for this cohort':('No '+(bc?'BC':'RL')+' evaluations logged yet')}</text>`;
 $('svm-chart').innerHTML=svg+'</svg>';$('svm-legend').innerHTML=legend;$('latest').innerHTML=table+'</tbody></table>';
}
async function refresh(){
 try{
  const [baseResponse,cohortResponse,auditResponse]=await Promise.all([fetch('svm-run.json'),fetch('svm-replicates.json'),fetch('svm-audit.json')]);
  if(!baseResponse.ok)throw Error('HTTP '+baseResponse.status);
  latestData=await baseResponse.json();const d=latestData;
  if(d.spec&&$('run-spec-link'))$('run-spec-link').href=d.spec;
  if(auditResponse.ok){const audit=await auditResponse.json();if(!d.campaign_id||audit.runs.some(r=>d.campaign_id.endsWith('/'+r.run)))d.audit=audit;}
  if(cohortResponse.ok){const cohort=await cohortResponse.json();if(cohort.campaign_id===d.campaign_id){d.evaluations=cohort.evaluations;d.summaries=cohort.summaries;d.runs=cohort.runs;d.updated_at=cohort.updated_at;d.training_episodes=cohort.training_episodes??d.training_episodes;d.eval_artifacts=cohort.eval_artifacts??d.eval_artifacts;}}
  if(initialPhase){$('stage').value=d.status==='bc_training'?'bc':'rl';initialPhase=false;}
  if(d.runs){
   const counts={};d.runs.forEach(r=>counts[r.status]=(counts[r.status]??0)+1);
   $('run-status').textContent=(d.audit?.cohort_label??'SVM')+' cohort: '+Object.entries(counts).map(([status,n])=>n+' '+status).join(' · ');
   $('progress').textContent=d.runs.map(r=>`Seed ${r.seed}: ${(r.physical_steps/1000).toFixed(0)}k / 100k (${r.status}${r.status==='running'?' · GPU '+r.gpu:''}${r.phase?' · '+r.phase:''}${r.svm_active?' · discriminator '+r.discriminator_updates+' updates':''})`).join(' · ')+' · updated '+d.updated_at;
  }else{
   $('run-status').textContent=d.status.replaceAll('_',' ')+'\n'+d.message;
   $('progress').textContent=`BC: ${(d.bc_updates??0).toLocaleString()} / 5,000 updates · RL: ${d.physical_steps.toLocaleString()} / 100,000 physical steps · updated ${d.updated_at}`;
  }
  drawBC(d);drawMedia(d);draw();document.body.dataset.status='ready';
 }catch(e){$('run-status').textContent='Could not load run status: '+e.message;document.body.dataset.status='error';}
}
$('show-training').onchange=draw;$('stage').onchange=draw;$('mode').onchange=draw;$('cohort').onchange=draw;refresh();setInterval(refresh,30000);
$('show-inactive-controls').onchange=async()=>{
 const status=$('inactive-controls-status');
 if(!$('show-inactive-controls').checked){status.textContent='';draw();return;}
 try{
  if(!inactiveControls){
   status.textContent='Loading sparse controls…';
   inactiveControlsRequest??=fetch('svm-inactive-controls.json').then(response=>{
    if(!response.ok)throw Error('HTTP '+response.status);
    return response.json();
   }).catch(error=>{inactiveControlsRequest=null;throw error;});
   inactiveControls=await inactiveControlsRequest;
  }
  status.textContent=$('show-inactive-controls').checked?'Five seeds (100–104), same 30-demo BC initialization; excluded from SVM statistics.':'';
  draw();
 }catch(error){status.textContent='Could not load sparse controls: '+error.message;}
};

// Local KaTeX assets render both display equations and inline notation.
renderMathInElement(document.querySelector('main'), {
  delimiters: [
    {left: '\\[', right: '\\]', display: true},
    {left: '\\(', right: '\\)', display: false}
  ],
  throwOnError: false
});

function drawBC(d){
 const selection=d.bc_selection;
 const rate=v=>v==null?'—':(100*v).toFixed(1)+'%';
 $('bc-history').textContent=selection?`Selected BC checkpoint: step ${selection.bc_updates}, ODE ${rate(selection.ode_success)}, SDE ${rate(selection.sde_success)}. ${selection.criterion}`:'BC: 30 demos · 5,000 training steps · 25 episodes per sampler every 500 steps.';
}

// Keep media elements mounted across live data polls. Only an explicit video
// checkpoint selection can replace the players; the overlay has separate controls.
function artifactKey(item){return JSON.stringify([item.seed,item.mode,item.physical_step,item.overlay_url]);}
function mediaOptions(select,options,fallback){
 const previous=select.value,signature=JSON.stringify(options);
 if(select.dataset.options!==signature){
  select.replaceChildren(...options.map(([value,label])=>{const option=document.createElement('option');option.value=value;option.textContent=label;return option;}));
  select.dataset.options=signature;
 }
 select.value=options.some(([v])=>v===previous)?previous:options.some(([v])=>v===fallback)?fallback:(options[0]?.[0]??'');
}
function initMedia(){
 if($('overlay-step'))return;
 $('eval-media').innerHTML=`<p id="media-empty" class="caption">No evaluation media logged yet.</p>
 <div id="media-content" hidden>
  <h3>Learned-reward trajectories</h3>
  <div class="media-controls"><label>Seed<select id="overlay-seed"></select></label><label>Evaluation sampler<select id="overlay-mode"></select></label><label>Trajectory outcome<select id="overlay-outcome"><option value="all">All trajectories</option><option value="success">Successful only</option><option value="failure">Unsuccessful only</option></select></label></div>
  <label for="overlay-step">Training checkpoint <output id="overlay-step-label" for="overlay-step"></output></label>
  <input id="overlay-step" type="range" min="0" max="0" step="1" value="0">
  <p id="overlay-caption" class="caption"></p>
  <a id="overlay-full"><img id="overlay-image" class="reward-overlay" alt="Calibrated end-effector trajectories colored by learned process reward"></a>
  <p><a id="overlay-download">Download selected trajectories and rewards</a></p>
  <h3>Evaluation recordings</h3>
  <p class="caption">Video selection is independent of the image slider. Live updates keep your playback position and open episodes.</p>
  <p id="videos-empty" class="caption" hidden>No recordings are available for this seed.</p>
  <div id="video-content"><div class="media-controls"><label>Recording seed<select id="video-seed"></select></label><label>Recorded checkpoint<select id="video-checkpoint"></select></label></div>
   <p id="video-caption" class="caption"></p><div id="video-players" class="media-videos"></div>
  </div>
 </div>`;
 $('overlay-seed').onchange=updateOverlay;
 $('overlay-mode').onchange=updateOverlay;
 $('overlay-outcome').onchange=updateOverlay;
 $('overlay-step').oninput=()=>{const items=overlayItems();mediaView.overlayStep=items[Number($('overlay-step').value)]?.physical_step??null;updateOverlay();};
 $('video-seed').onchange=updateVideos;
 $('video-checkpoint').onchange=updateVideos;
}
function overlayItems(){
 return mediaView.entries.filter(item=>String(item.seed)===$('overlay-seed').value&&item.mode===$('overlay-mode').value&&item.overlay_url)
  .sort((a,b)=>a.physical_step-b.physical_step);
}
function updateOverlay(){
 const entries=mediaView.entries;
 const seeds=[...new Set(entries.map(item=>item.seed))].sort((a,b)=>a-b);
 mediaOptions($('overlay-seed'),seeds.map(seed=>[String(seed),'Seed '+seed]));
 const modes=[...new Set(entries.filter(item=>String(item.seed)===$('overlay-seed').value).map(item=>item.mode))].sort();
 mediaOptions($('overlay-mode'),modes.map(mode=>[mode,mode]),'ODE');
 const items=overlayItems();if(!items.length)return;
 // Hold the chosen physical step when another checkpoint arrives or controls change.
 let index=mediaView.overlayStep==null?items.length-1:items.findIndex(item=>item.physical_step===mediaView.overlayStep);
 if(index<0){index=items.findLastIndex(item=>item.physical_step<=mediaView.overlayStep);if(index<0)index=0;}
 const item=items[index];mediaView.overlayStep=item.physical_step;mediaView.overlayItem=item;
 const slider=$('overlay-step');slider.max=String(items.length-1);slider.value=String(index);slider.disabled=items.length<2;
 slider.setAttribute('aria-valuetext',item.physical_step.toLocaleString()+' physical training steps');
 $('overlay-step-label').textContent=item.physical_step.toLocaleString()+' steps · '+(index+1)+' / '+items.length;
 const outcome=$('overlay-outcome').value;
 const label={all:'All',success:'Successful',failure:'Unsuccessful'}[outcome];
 const results=item.episode_results??[];
 const count=outcome==='all'?item.episodes:results.filter(result=>Boolean(result.success)===(outcome==='success')).length;
 const filename=item.overlays?.[outcome];
 const url=outcome==='all'?item.overlay_url:filename?new URL(filename,new URL(item.overlay_url,location.href)).pathname:null;
 $('overlay-caption').textContent=`Seed ${item.seed} · ${item.mode} · ${label}: ${count} / ${item.episodes} trajectories · background reset state ${item.background_init_state_id}. Same learned-reward color scale in all views.`+(url?'':' Filtered image is being prepared.');
 $('overlay-image').hidden=!url;$('overlay-full').hidden=!url;
 if(url){if($('overlay-image').getAttribute('src')!==url)$('overlay-image').src=url;$('overlay-full').href=url;}

 $('overlay-download').hidden=!item.data_url;if(item.data_url)$('overlay-download').href=item.data_url;
}
function updateVideos(){
 const entries=mediaView.entries.filter(item=>(item.video_urls??[]).some(Boolean));
 $('videos-empty').hidden=entries.length>0;$('video-content').hidden=!entries.length;
 if(!entries.length)return;
 const seeds=[...new Set(entries.map(item=>item.seed))].sort((a,b)=>a-b);
 mediaOptions($('video-seed'),seeds.map(seed=>[String(seed),'Seed '+seed]));
 const items=entries.filter(item=>String(item.seed)===$('video-seed').value).sort((a,b)=>a.physical_step-b.physical_step||a.mode.localeCompare(b.mode));
 const preferred=mediaView.videoItem??mediaView.overlayItem??items.at(-1);
 mediaOptions($('video-checkpoint'),items.map(item=>[artifactKey(item),item.physical_step.toLocaleString()+' steps · '+item.mode]),preferred?artifactKey(preferred):'');
 const item=items.find(item=>artifactKey(item)===$('video-checkpoint').value);if(!item)return;
 mediaView.videoItem=item;
 $('video-caption').textContent=`Seed ${item.seed} · ${item.mode} · ${item.physical_step.toLocaleString()} steps · recorded scored evaluation episodes`;
 const urls=item.video_urls.filter(Boolean),signature=JSON.stringify(urls);
 if(signature===mediaView.videoSignature)return; // Never detach a playing video during polling.
 mediaView.videoSignature=signature;
 $('video-players').querySelectorAll('video').forEach(video=>video.pause());
 $('video-players').replaceChildren(...urls.map((url,i)=>{
  const details=document.createElement('details'),summary=document.createElement('summary'),video=document.createElement('video');
  summary.textContent='Recorded evaluation episode '+(i+1);
  video.controls=true;video.preload='none';video.src=url;
  details.append(summary,video);return details;
 }));
}
function drawMedia(d){
 initMedia();
 const incoming=d.eval_artifacts??[];
 // Also retain the selected recording if the incoming dataset changes, so even
 // a new campaign cannot interrupt a video that is already open.
 const retained=[mediaView.videoItem,mediaView.overlayItem].filter(Boolean);
 const unique=new Map(incoming.map(item=>[artifactKey(item),item]));
 retained.forEach(item=>{if(!unique.has(artifactKey(item)))unique.set(artifactKey(item),item);});
 mediaView.entries=[...unique.values()];
 const available=mediaView.entries.length>0;
 $('media-empty').hidden=available;$('media-content').hidden=!available;
 if(!available)return;
 updateOverlay();updateVideos();
}
