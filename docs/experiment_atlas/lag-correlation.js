'use strict';
// Separate from live training/media refreshes: switching this plot never resets videos.
(() => {
 const select=document.getElementById('lag-mode');
 const image=document.getElementById('lag-image');
 const status=document.getElementById('lag-status');
 let data;
 function draw(){
  const mode=select.value,file=`lag-correlation-${mode.toLowerCase()}.svg`;
  image.src=file;
  image.alt=`${mode}: within-seed correlations at fixed 10k lag versus training steps; green SVM, orange historical sparse, purple earlier sparse`;
  document.getElementById('lag-download').href=file;
  if(!data)return;
  const table=document.createElement('table');
  const head=table.createTHead().insertRow();
  const series=data.series.filter(s=>s.mode===mode);
  const steps=[...new Set(series.flatMap(s=>s.points.map(p=>p.physical_step)))].sort((a,b)=>a-b);
  ['Cohort',...steps.map(step=>`${step/1000}k`)].forEach(label=>{
   const cell=document.createElement('th');cell.textContent=label;head.append(cell);
  });
  const body=table.createTBody();
  for(const item of series){
   const row=body.insertRow();row.insertCell().textContent=item.cohort;
   for(const step of steps){
    const cell=row.insertCell();
    const point=item.points.find(p=>p.physical_step===step);
    const value=point?.correlation;
    cell.textContent=`${value==null?'Unavailable':value.toFixed(3)} (${point?.n_seeds??0} seeds)`;
    cell.title=item.seeds.map(s=>{const p=s.points.find(p=>p.physical_step===step);return `Seed ${s.seed}: ${p?.correlation==null?'Unavailable':p.correlation.toFixed(3)} (${p?.n_pairs??0} pairs)`;}).join('; ');
   }
  }
  document.getElementById('lag-values').replaceChildren(table);
  status.textContent=`Computed ${new Date(data.created_at).toLocaleString()} · fixed ${data.lag_steps/1000}k-step lag · cumulative within-seed correlations; light seed lines, dark arithmetic mean. Saved snapshot.`;
 }
 select.addEventListener('change',draw);
 fetch('lag-correlation.json').then(response=>{
  if(!response.ok)throw Error(`HTTP ${response.status}`);
  return response.json();
 }).then(result=>{data=result;draw();}).catch(error=>{
  status.textContent=`Could not load correlation values: ${error.message}`;
 });
})();
