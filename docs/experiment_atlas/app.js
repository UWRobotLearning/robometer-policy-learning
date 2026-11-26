'use strict';
const $ = id => document.getElementById(id);
const esc = v => String(v ?? '—').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const link = (url, title) => `<a href="${esc(url)}">${esc(title)}</a>`;
const pct = v => v == null ? '—' : (100 * v).toFixed(1) + '%';
const number = v => v == null ? '—' : Math.round(v).toLocaleString();
const short = v => String(v).replaceAll('_', ' ');
let data, filtered = [], selected, runLimit = 50, sourceLimit = 60;
const colors = ['#087e8b','#c25b25','#6953a4','#35804c','#b33968','#326ca8','#91701c','#48616d'];
function fill(id, values) {
  $(id).insertAdjacentHTML('beforeend', [...new Set(values.filter(v => v != null && v !== ''))].sort().map(v => `<option value="${esc(v)}">${esc(v)}</option>`).join(''));
}
function pointList(run) { return run.points.filter(p => p.mode === $('sampler').value); }
function filterRuns() {
  const q = $('search').value.toLowerCase();
  filtered = data.runs.filter(r => [r.name,r.task,r.seed,r.state,r.group].join(' ').toLowerCase().includes(q) && ['family','suite','reward','state'].every(k => !$(k).value || r[k] === $(k).value));
  const groups = [...new Set(filtered.map(r => r.group))];
  if (!groups.includes(selected)) selected = groups[0];
  $('run-count').textContent = `${filtered.length} runs · ${groups.length} experiment groups`;
  $('groups').innerHTML = groups.length ? groups.map((g,i) => `<button data-group="${i}" class="${g === selected ? 'selected' : ''}">${esc(short(g))}<small>${filtered.filter(r => r.group === g).length} runs</small></button>`).join('') : '<p class="empty">No matching experiments.</p>';
  $('groups').querySelectorAll('button').forEach(b => b.onclick = () => { selected = groups[Number(b.dataset.group)]; runLimit = 50; history.replaceState(null, '', '#experiment=' + encodeURIComponent(selected)); filterRuns(); });
  drawGroup();
}
function drawGroup() {
  const runs = filtered.filter(r => r.group === selected);
  const modes = [...new Set(runs.flatMap(r => r.points.map(p => p.mode)))];
  const oldMode = $('sampler').value;
  $('sampler').innerHTML = (modes.length ? modes : ['recorded']).map(m => `<option value="${esc(m)}">${m === 'recorded' ? 'Recorded protocol' : esc(m)}</option>`).join('');
  if (modes.includes(oldMode)) $('sampler').value = oldMode;
  $('plot-title').textContent = selected ? short(selected) : 'No matching experiment';
  $('plot-subtitle').textContent = [...new Set(runs.map(r => `${r.family} · ${r.suite} · task ${r.task_id ?? '?'} · ${r.model}`))].join(' / ');
  chart(runs);
  const rows = runs.slice(0,runLimit);
  $('run-table').innerHTML = rows.map(r => {
    const points = pointList(r).sort((a,b) => a.x-b.x), p = points.at(-1);
    return `<tr><td>${link(r.config,r.name)}<small>${esc(short(r.task))}</small><small>${link(r.source,'Evaluation data')}${r.status ? ' · ' + link(r.status,'Status record') : ''}</small></td><td>${esc(r.family)}</td><td>${esc(r.reward)}<small>seed ${esc(r.seed)}</small></td><td><span class="badge ${['complete','Completed'].includes(r.state) ? 'complete' : ''}">${esc(r.state)}</span><small>${esc(r.captured.slice(0,10))}</small></td><td>${pct(p?.y)}<small>${p ? `${p.successes} / ${p.episodes} episodes` : 'No evaluation for this sampler'}</small></td><td>${number(p?.x)}</td><td>${esc(r.envs)} / ${esc(r.horizon)}</td></tr>`;
  }).join('') || '<tr><td colspan="7">No matching runs.</td></tr>';
  $('more-runs').hidden = runs.length <= runLimit;
}
function chart(runs) {
  const series = runs.map(r => ({r, points:pointList(r).sort((a,b) => a.x-b.x)})).filter(s => s.points.length);
  if (!series.length) { $('chart').innerHTML='<p class="empty">No recorded evaluations for this selection. Check the source configuration and historical status.</p>'; $('legend').innerHTML=''; return; }
  const W=820,H=340,L=65,R=20,T=20,B=60,maxX=Math.max(1,...series.flatMap(s=>s.points.map(p=>p.x)));
  const x=v=>L+v/maxX*(W-L-R), y=v=>T+(1-v)*(H-T-B);
  let svg=`<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="Individual run success curves by physical training steps per environment"><title>${esc(selected)} — individual run success</title>`;
  for(let v=0;v<=1.001;v+=.25) svg+=`<line x1="${L}" x2="${W-R}" y1="${y(v)}" y2="${y(v)}" stroke="#e1e8ef"/><text x="${L-12}" y="${y(v)+5}" text-anchor="end" fill="#5b6d7e" font-size="13">${Math.round(v*100)}%</text>`;
  for(let i=0;i<5;i++){ const v=maxX*i/4; svg+=`<text x="${x(v)}" y="${H-B+24}" text-anchor="middle" fill="#5b6d7e" font-size="13">${v>=1000 ? (v/1000).toFixed(v%1000 ? 1 : 0)+'k' : Math.round(v)}</text>`; }
  svg+=`<text x="${W/2}" y="${H-8}" text-anchor="middle" fill="#5b6d7e" font-size="14">Physical training steps per environment</text>`;
  series.forEach(({r,points},i)=>{
    const color=colors[i%colors.length],dash=r.reward==='robometer'?'':'5 3';
    svg+=`<polyline points="${points.map(p=>`${x(p.x)},${y(p.y)}`).join(' ')}" fill="none" stroke="${color}" stroke-width="2" stroke-dasharray="${dash}" opacity=".85"/>`;
    points.forEach(p=>{svg+=`<circle cx="${x(p.x)}" cy="${y(p.y)}" r="3" fill="${color}" tabindex="0"><title>${esc(r.name)} | ${esc(p.mode)} | ${number(p.x)} steps/env | ${pct(p.y)} (${p.successes}/${p.episodes})</title></circle>`;});
  });
  $('chart').innerHTML=svg+'</svg>';
  $('legend').innerHTML=series.map(({r},i)=>`<span title="${esc(r.name)}"><i class="swatch" style="background:${colors[i%colors.length]}"></i>${esc(r.reward)} · seed ${esc(r.seed)} · ${esc(r.family)}</span>`).join('');
}
function sources() {
  const q=$('source-search').value.toLowerCase(), rows=data.inventory.filter(r=>r.path.toLowerCase().includes(q));
  $('source-count').textContent=`${rows.length} files · showing ${Math.min(sourceLimit,rows.length)}`;
  $('inventory').innerHTML=rows.slice(0,sourceLimit).map(r=>link(r.source,r.path)+`<span class="caption">${number(r.bytes)} bytes</span>`).join('') || 'No matching files.';
  $('more-sources').hidden=rows.length<=sourceLimit;
}
async function init() {
  const response=await fetch('data.json'); if(!response.ok) throw Error(`Data request returned ${response.status}`); data=await response.json();
  $('snapshot').textContent=`Saved evidence · ${data.captured.slice(0,10)} · Historical statuses, not live training`;
  const metrics=[[data.runs.length,'training run records'],[data.runs.reduce((n,r)=>n+r.points.length,0),'training evaluation points'],[new Set(data.runs.map(r=>r.group)).size,'experiment groups']];
  $('stats').innerHTML=metrics.map(([n,t])=>`<div class="stat"><b>${number(n)}</b><span>${t}</span></div>`).join('');
  ['family','suite','reward','state'].forEach(k=>{fill(k,data.runs.map(r=>r[k]));$(k).onchange=filterRuns;});
  $('search').oninput=filterRuns; $('sampler').onchange=drawGroup;
  $('more-runs').onclick=()=>{runLimit+=50;drawGroup();};
  $('source-search').oninput=()=>{sourceLimit=60;sources();};$('more-sources').onclick=()=>{sourceLimit+=120;sources();};
  const hash=location.hash;if(hash.startsWith('#experiment=')){try{selected=decodeURIComponent(hash.slice(12));}catch{selected=undefined;}}
  // Lead with the SAC sanity check when no experiment was linked directly.
  if(!selected)selected=data.runs.find(r=>r.group.includes('sac_task28_general'))?.group;
  filterRuns();sources();
  $('plans-summary').textContent=`Planned work · ${data.plans.length} specification / manifest files`;
  $('plans').innerHTML=data.plans.map(r=>link(r.source,r.path)).join('');
  $('inventory-summary').textContent=`Complete numerical source inventory · ${data.inventory.length.toLocaleString()} files`;
  $('snapshot-index').href='/'+data.snapshot+'/index.json';
  $('notes').innerHTML=data.notes.map(n=>`<article><span class="tag">${esc(n.kind)}</span><h3>${esc(n.title)}</h3><p>${esc(n.body)}</p><p>${link(n.source,'Read the evidence')}</p></article>`).join('');
  const reports=[['reward_learning_summary_20261028',['SAC · LIBERO-90 task 28','DSRL · π₀ · LIBERO-10 task 6','DSRL · π₀ · LIBERO-10 task 8','DSRL · π₀.₅ · LIBERO-90 task 28','DSRL · π₀.₅ · LIBERO-90 task 29','DSRL · task 29 · 1M steps','DSRL · task 29 · four environments','DSRL · task 8 · batched four environments']],['ogpo_success_summary_20261031',['OGPO MLP · LIBERO-10 task 8 · 50k','OGPO MLP · LIBERO-90 task 29 · 50k','OGPO MLP · LIBERO-90 task 2 · 50k','OGPO MLP · LIBERO-90 task 2 · 100k','OGPO native LoRA · LIBERO-90 task 29']]];
  $('figures').innerHTML=reports.flatMap(([dir,titles])=>titles.map((title,i)=>{const src=`/docs/reports/${dir}/curve_${String(i).padStart(2,'0')}.png`;return `<details><summary>${esc(title)}</summary><a href="${src}"><img loading="lazy" src="${src}" alt="${esc(title)} — saved report figure"></a><p class="caption">Report capture: ${dir.includes('20261028')?'October 28':'October 31'}, 2026. ${link(`/docs/reports/${dir}/report.md`,'Read the protocol and caveats')}</p></details>`;})).join('');
  $('warnings').textContent=data.warnings.length?'Export warnings: '+JSON.stringify(data.warnings):'Snapshot exporter reported no warnings.';
  $('build-date').textContent=`Page data built ${data.built_at.slice(0,19).replace('T',' ')} UTC · OGPO report captured ${data.ogpo_captured.slice(0,10)}`;
  document.body.dataset.status='ready';
  const observer=new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting)document.querySelectorAll('nav[aria-label="On this page"] a').forEach(a=>a.classList.toggle('active',a.hash==='#'+entry.target.id));}),{rootMargin:'-10% 0px -60% 0px'});
  document.querySelectorAll('main section').forEach(s=>observer.observe(s));
}
init().catch(error=>{ $('snapshot').textContent='Could not load experiment data: '+error.message; document.body.dataset.status='error'; console.error(error); });
