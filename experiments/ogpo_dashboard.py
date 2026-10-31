"""Live, read-only OGPO results; BC counters are separate from physical RL steps."""
from __future__ import annotations

import csv
import html
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def rows(path):
    try:
        with path.open() as stream:
            return list(csv.DictReader(stream))
    except OSError:
        return []


def number(value):
    try:
        n = float(value)
        return n if math.isfinite(n) else None
    except (TypeError, ValueError):
        return None


def wilson(p, n=25):
    z = 1.959963984540054
    d = 1 + z*z/n
    center = (p + z*z/(2*n))/d
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/d
    return max(0, center-half), min(1, center+half)


def phase_offset(console):
    """The actual final BC eval step, not the configured maximum of 500k."""
    if 'Online RL Phase' not in console:
        return None
    before = console.rsplit('Online RL Phase', 1)[0]
    steps = re.findall(r'\[EVAL ODE\] step=\s*(\d+)', before)
    return int(steps[-1]) if steps else None


def points(records, key='success', offset=0, minimum=0):
    result = {}
    for row in records:
        step, value = number(row.get('step')), number(row.get(key))
        if step is not None and value is not None and step-offset >= minimum:
            result[step-offset] = value  # Duplicate evals at a boundary aren't extra episodes.
    return sorted(result.items())



def duration(seconds):
    minutes = max(1, math.ceil(seconds / 60))
    return f"{minutes // 60}h {minutes % 60:02d}m" if minutes >= 60 else f"{minutes}m"


def progress_timing(console, phase):
    matches = re.findall(re.escape(phase) + r':[^\r\n]*?\|\s*(\d+)/(\d+)\s*\[([\d:]+)<', console)
    if not matches:
        return None
    completed, total, elapsed = matches[-1]
    seconds = 0
    for part in elapsed.split(':'):
        seconds = seconds * 60 + int(part)
    return int(completed), int(total), seconds


def remaining_time(console, state, features, cfg):
    if state in ('Completed', 'Stopped', 'BC checkpoint selected') or state.startswith('Failed'):
        return ''
    if state == 'Online RL':
        timing = progress_timing(console, 'Online RL')
        if timing and timing[0] >= 1000 and timing[2] > 0:
            completed, total, elapsed = timing
            if completed >= total:
                return 'Final evaluation / checkpoint in progress'
            remaining = (total-completed) * elapsed / completed
            return ('Estimated time remaining: ~' + duration(remaining)
                    + ' · observed RL pace, including evaluations so far; evaluation time varies')
        return 'Estimated time remaining: calculating from RL progress…'
    if state == 'Behavior cloning':
        timing = progress_timing(console, 'BC Training')
        interval = cfg.get('intervals', {}).get('eval_interval_bc', 0)
        if interval and (not timing or timing[0] <= interval):
            return 'BC completion ETA: waiting for the first full evaluation cycle; rollout evaluations dominate runtime'
        if timing and timing[0] >= 100 and timing[2] > 0:
            completed, total, elapsed = timing
            remaining = (total-completed)*elapsed/completed
            return ('BC budget remaining: ~' + duration(remaining)
                    + ' at observed pace, including completed evaluations · total experiment ETA pending RL timing')
        return 'BC timing in progress · total experiment ETA pending RL timing'
    completed = features.get('completed', 0)
    total = features.get('identity', {}).get('shape', [0])[0]
    observed = completed - features.get('session_start', 0)
    elapsed = features.get('seconds_this_session', 0)
    if total > completed and observed > 0 and elapsed > 0:
        return ('Feature extraction remaining: ~' + duration((total-completed)*elapsed/observed)
                + ' · BC and RL follow; total experiment ETA pending RL timing')
    return 'Estimated time remaining: preparing timing data…'


def native_remaining_time(console, status, spec, evaluations):
    """Estimate post-warm-up collection/update pace plus scheduled evaluations."""
    state = status.get('status', 'queued')
    if state in ('completed', 'complete'):
        return 'Completed · no time remaining'
    if state in ('failed', 'stopped', 'queued'):
        return state.capitalize() + ' · completion ETA unavailable'
    samples = [(int(step), float(seconds)) for step, seconds in re.findall(
        r'RL step=(\d+) updates=\d+ elapsed=([\d.]+)s', console)
        if int(step) >= spec.get('warmup_steps', 20000)]
    if len(samples) < 2 or samples[-1][0] - samples[0][0] < 1000:
        return 'Estimated time remaining: collecting post-warm-up timing data…'
    first, last = samples[0], samples[-1]
    eval_seconds = sum(float(row.get('seconds', 0)) for rows_ in evaluations.values()
                       for row in rows_ if first[0] < float(row['step']) <= last[0])
    pace = max(0, last[1] - first[1] - eval_seconds) / (last[0] - first[0])
    progress = status.get('steps', last[0])
    budget = spec['steps']
    interval = spec.get('eval_every', 10000)
    remaining = max(0, budget - progress) * pace
    for rows_ in evaluations.values():
        if not rows_:
            continue
        recent = rows_[-3:]
        seconds = sum(float(row.get('seconds', 0)) for row in recent) / len(recent)
        last_eval = float(rows_[-1]['step'])
        remaining += max(0, math.ceil((budget - last_eval) / interval)) * seconds
    return ('Estimated time remaining: ~' + duration(remaining)
            + ' · observed post-warm-up collection + updates and estimated remaining evaluations; approximate')


def runtime_text(record, state, console, restored_bc=False):
    started, finished = number(record.get('started')), number(record.get('finished'))
    if started is not None and finished is not None:
        label = 'Total runtime: ' + duration(finished-started)
    elif started is not None and record.get('status') == 'running' and state != 'Stopped':
        label = 'Elapsed runtime: ' + duration(datetime.now(timezone.utc).timestamp()-started)
    else:
        timings = [(name, progress_timing(console, phase)) for name,phase in
                   [('BC', 'BC Training'), ('RL', 'Online RL')]]
        values = [name + ': ' + duration(timing[2]) for name,timing in timings if timing and timing[2] > 0]
        return ('Recorded phase runtime — ' + ' · '.join(values) +
                ' (setup excluded; total runtime unavailable)') if values else 'Runtime: not started / timing unavailable'
    return label + ' · includes setup and evaluations' + ('; earlier BC training excluded' if restored_bc else '')


def load(data_root):
    base = data_root/'ogpo_libero'
    result = []
    for spec_path in base.glob('libero_*_task*.json'):
        spec = read_json(spec_path)
        if not spec.get('env_name'):
            continue
        root = Path(spec['output_dir'])
        if spec.get('native_pi05_lora'):
            status = read_json(root/'status.json')
            state = status.get('status', 'queued')
            record = dict(started=status.get('started'), finished=status.get('finished'), status='running' if state in ('running','evaluating','initializing') else state)
            diagnostics=[]
            try:
                for line in (root/'diagnostics.jsonl').read_text().splitlines():
                    try: row=json.loads(line)
                    except ValueError: continue
                    diagnostics.append({'step':row['step'],'actor/pg_loss':row.get('actor_loss'),'actor/ratio':row.get('ratio'),'actor/ratio_clipped_total':row.get('clipped_fraction'),'actor/adv_std':row.get('advantage_std'),'critic/td_loss':row.get('td_loss')})
            except OSError: pass
            evaluation = {'ODE':rows(root/'eval_ode.csv'),'SDE':rows(root/'eval_sde.csv')}
            try:
                console = (root/'console.log').read_text(errors='replace')
            except OSError:
                console = ''
            eta = native_remaining_time(console, status, spec, evaluation)
            result.append(dict(name=spec_path.stem, spec=spec, config={'training': {'online_steps':spec['steps'], 'bc_pi_steps':0}},
                root=root, offset=0, progress=status.get('steps',0), selection={}, budget=spec['steps'],
                state=state.capitalize(), gpu=None, eta=eta, runtime=runtime_text(record,state.capitalize(),'',False),
                evaluation=evaluation,
                diagnostics=diagnostics,bc_progress=None,bc=[],n=spec.get('eval_episodes',25)))
            continue
        try:
            console = (root/'console.log').read_text(errors='replace')
        except OSError:
            console = ''
        flag_files = list(root.glob('logs/**/flags.json'))
        cfg = read_json(flag_files[0]) if flag_files else {}
        if not cfg:
            cfg = {'training': {'online_steps': spec.get('overrides', {}).get('training.online_steps', 200000),
                                'bc_only': spec.get('overrides', {}).get('training.bc_only', False),
                                'bc_pi_steps': spec.get('overrides', {}).get('training.bc_pi_steps', 500000)}}
        log_dir = flag_files[0].parent if flag_files else root/'not_started'
        offset = phase_offset(console)
        progress = re.findall(r'Online RL:.*?(\d+)/(\d+)', console)
        features = read_json(root/'features.json')
        state = 'Preparing dataset / encoder'
        if features:
            state = f"Extracting features: {features.get('completed',0):,}/{features.get('identity',{}).get('shape',[0])[0]:,}"
        if spec.get('bc_source') and 'Restored complete BC checkpoint' in console:
            state = 'Evaluating restored BC baseline'
        if 'BC Training Phase' in console:
            state = 'Behavior cloning'
        if offset is not None:
            state = 'Online RL'
        if 'Online RL training complete' in console:
            state = 'Completed'
        # Prefer live queue status if available; an old traceback from a fixed
        # startup attempt must not label a later successful attempt as failed.
        gpu = None
        runtime_record = {}
        for status_path in base.glob('queue_gpu*.json'):
            status = read_json(status_path)
            for task in status.get('tasks',[]):
                if Path(task.get('spec','')).name == spec_path.name:
                    gpu = status.get('gpu')
                    runtime_record = task
                    if task.get('status') in ('queued', 'waiting_for_gpus', 'blocked'):
                        state = {'queued': 'Queued — waiting for preceding scheduled runs' if len(status.get('tasks', [])) > 1 else ('Queued — after BC and sparse RL' if spec.get('bc_selection_spec') and spec.get('reward') == 'robometer' and not spec.get('allow_reward_colocation') else 'Queued — after BC checkpoint selection' if spec.get('bc_selection_spec') else 'Queued — after sparse runs'), 'waiting_for_gpus': 'Waiting for GPUs', 'blocked': 'Blocked — dependency failed'}[task['status']]
                    if task.get('status') == 'failed':
                        state = 'Failed — inspect console.log'
        if (root/'stopped.json').exists():
            state = 'Stopped'
            gpu = None
        selection = read_json(log_dir/'bc_selected/selection.json')
        if selection and spec.get('overrides', {}).get('agent.bc_select_stop', True):
            state = 'BC checkpoint selected'
        if 'BC training complete. Skipping online RL phase.' in console:
            state = 'Completed'
        evaluation = {name: rows(log_dir/file) for name,file in [('ODE','eval.csv'),('SDE','eval_sde.csv')]}
        result.append(dict(name=spec_path.stem, spec=spec, config=cfg, root=root,
            offset=offset, progress=int(progress[-1][0]) if progress else 0, selection=selection,
            budget=cfg.get('training',{}).get('online_steps',200000), state=state, gpu=gpu,
            runtime=runtime_text(runtime_record, state, console, bool(spec.get('bc_source'))),
            eta=remaining_time(console, state, features, cfg),
            evaluation=evaluation, diagnostics=rows(log_dir/'online_agent.csv'),
            bc_progress=progress_timing(console, 'BC Training'),
            bc=rows(log_dir/'offline_agent.csv'), n=cfg.get('eval',{}).get('eval_episodes',25)))
    # A BC sweep used to initialize an RL experiment is part of that same
    # experiment's plot, rather than a second dashboard card.
    merged = set()
    for task in result:
        selection_spec = task['spec'].get('bc_selection_spec')
        if selection_spec:
            parent = next((item for item in result if item['name'] == Path(selection_spec).stem), None)
            if parent is not None:
                task['bc_run'] = parent
                task['bc_history'] = parent['evaluation']
                task['bc_progress'] = parent['bc_progress']
                merged.add(parent['name'])
        source = task['spec'].get('bc_source')
        if not source:
            continue
        provenance = read_json(Path(source['checkpoint']).parent/'selection.json')
        original = provenance.get('source_checkpoint')
        for parent in result:
            if parent is not task and original and Path(original).is_relative_to(parent['root']):
                task['bc_history'] = parent['evaluation']
                task['bc_progress'] = parent['bc_progress']
                merged.add(parent['name'])
    return sorted([task for task in result if task['name'] not in merged],
                  key=lambda t:t['root'].stat().st_mtime,reverse=True)


def chart(series, title, xlabel='Physical RL steps per environment', probability=False, n=25, baselines=(), intervals=None, seed_series=None):
    esc = html.escape
    all_points = [p for _,_,ps in series for p in ps]
    if not all_points:
        return '<div class="empty">No completed measurements yet.</div>'
    xmax = max(1, max(p[0] for p in all_points))
    values = [p[1] for p in all_points]
    lo, hi = (0,1) if probability else (min(values), max(values))
    if lo == hi:
        lo,hi = lo-max(1,abs(lo)*.1),hi+max(1,abs(hi)*.1)
    elif not probability:
        pad=(hi-lo)*.08;lo-=pad;hi+=pad
    x = lambda v: 48+v/xmax*490
    y = lambda v: 190-(v-lo)/(hi-lo)*150
    out=[f'<svg viewBox="0 0 560 235" role="img" aria-label="{esc(title)}"><title>{esc(title)}; linear axes</title>']
    for i in range(5):
        v=lo+(hi-lo)*i/4
        label=f'{v*100:.0f}%' if probability else f'{v:.3g}'
        out.append(f'<line x1="48" x2="538" y1="{y(v):.1f}" y2="{y(v):.1f}" class="grid"/><text x="43" y="{y(v)+4:.1f}" text-anchor="end">{label}</text>')
        v=xmax*i/4
        out.append(f'<text x="{x(v):.1f}" y="207" text-anchor="middle">{v/1000:g}k</text>')
    for baseline in baselines:
        name,color,mean = baseline[:3]
        out.append(f'<g data-eval-mode="{"SDE" if "SDE" in name else "ODE"}">')
        lower,upper = baseline[3:5] if len(baseline) >= 5 else wilson(mean,n)
        ci_label = '95% CI' if len(baseline) >= 5 else '95% Wilson CI'
        out.append(f'<rect x="48" y="{y(upper):.1f}" width="490" height="{y(lower)-y(upper):.1f}" fill="{color}" opacity=".06"/><line x1="48" x2="538" y1="{y(mean):.1f}" y2="{y(mean):.1f}" stroke="{color}" stroke-dasharray="2 5"><title>{esc(name)} {mean:.0%}; {ci_label} {lower:.0%}–{upper:.0%}</title></line>')
        out.append('</g>')
    for index,(name,color,ps) in enumerate(series):
        out.append(f'<g data-eval-mode="{"SDE" if "SDE" in name else "ODE"}">')
        marker = '◇' if name.startswith('BC ') else '●'
        out.append(f'<text x="{48+(index%2)*245}" y="{14+(index//2)*14}" style="fill:{color}">{marker} {esc(name)}</text>')
        if probability and ps:
            bounds = (intervals.get(name, []) if intervals is not None else [(a,*wilson(b,n)) for a,b in ps])
            upper=[(x(a),y(hi)) for a,lo,hi in bounds]
            lower=[(x(a),y(lo)) for a,lo,hi in reversed(bounds)]
            pts=' '.join(f'{a:.1f},{b:.1f}' for a,b in upper+lower)
            out.append(f'<polygon points="{pts}" fill="{color}" opacity=".12"/>')
        individual = [(seed, values) for seed, values in (seed_series or {}).get(name, []) if values]
        if len(individual) > 1:
            for seed, values in individual:
                seed_points=' '.join(f'{x(a):.1f},{y(b):.1f}' for a,b in values)
                seed_dash=' stroke-dasharray="5 4"' if 'SDE' in name else ''
                out.append(f'<g class="individual-seed" data-seed="{esc(str(seed))}" opacity=".25"><title>{esc(name)} · seed {esc(str(seed))}</title>')
                out.append(f'<polyline points="{seed_points}" fill="none" stroke="{color}" stroke-width="1.2"{seed_dash}/>')
                for a,b in values:
                    out.append(f'<circle cx="{x(a):.1f}" cy="{y(b):.1f}" r="2" fill="{color}"><title>{esc(name)} · seed {esc(str(seed))}: {a:g} steps, {b:.1%}</title></circle>')
                out.append('</g>')
        pts=' '.join(f'{x(a):.1f},{y(b):.1f}' for a,b in ps)
        dash = ' stroke-dasharray="5 4"' if name.startswith('BC ') or 'SDE' in name else ''
        out.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2"{dash}/>')
        for a,b in ps:
            label=f'{b:.1%}' if probability else f'{b:.4g}'
            tooltip = f'<title>{esc(name)}: {a:g} count, {label}</title>'
            if name.startswith('BC '):
                cx,cy=x(a),y(b)
                diamond=f'{cx:.1f},{cy-4:.1f} {cx+4:.1f},{cy:.1f} {cx:.1f},{cy+4:.1f} {cx-4:.1f},{cy:.1f}'
                out.append(f'<polygon class="bc-marker" points="{diamond}" fill="white" stroke="{color}" stroke-width="2">{tooltip}</polygon>')
            else:
                out.append(f'<circle class="rl-marker" cx="{x(a):.1f}" cy="{y(b):.1f}" r="3" fill="{color}">{tooltip}</circle>')
        out.append('</g>')
    out.append(f'<text x="293" y="229" text-anchor="middle">{esc(xlabel)} · linear scale</text></svg>')
    return ''.join(out)


CSS = '''
:root{--ink:#17313b;--muted:#5d6e73;--line:#dce5e6;--bg:#f3f7f7}*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}a{color:#087f8c}.layout{display:grid;grid-template-columns:260px minmax(0,1fr);max-width:1650px;margin:auto}.sidebar{position:sticky;top:0;height:100vh;overflow:auto;padding:28px 18px;background:#eaf1f1;border-right:1px solid var(--line)}main{min-width:0;padding:22px 18px 40px}.home{font-size:19px;font-weight:800;text-decoration:none;color:var(--ink)}h1{font-size:32px;line-height:1.12;margin:8px 0}h2{font-size:19px;margin:0}h3{font-size:13px;margin:8px 0}.eyebrow{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#577176;font-weight:700}.meta,small{color:var(--muted);font-size:12px}.side-link{display:block;padding:12px 4px;border-top:1px solid #cbdadb;text-decoration:none}.side-link small{display:block}.cards{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}.card{background:white;border:1px solid var(--line);border-radius:14px;padding:16px;min-width:0}.facts{display:flex;gap:12px;flex-wrap:wrap;font-size:12px;margin:8px 0}.eta{font-size:12px;color:#46656b;margin:5px 0 8px}.badge{color:#087f8c;font-weight:700}.empty{height:210px;display:grid;place-items:center;color:var(--muted);font-size:13px}svg{width:100%;display:block}svg text{font:10px system-ui;fill:#5d6e73}.grid{stroke:#e0e8e8}details{margin:12px 0}summary{cursor:pointer;font-size:13px;font-weight:600}.diagnostics{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}.diagnostic{min-width:0}.diagnostic svg text{font-size:11px}.diagnostic .empty{height:80px}progress{width:100%;height:7px;accent-color:#087f8c}select{width:100%;padding:8px;border:1px solid #b8ccce;border-radius:8px;margin:15px 0}table{border-collapse:collapse;font-size:12px;width:100%}td,th{padding:5px;text-align:left;border-bottom:1px solid var(--line)}.detail{margin-top:16px}.note{margin:14px 0;padding:10px 14px;border-left:3px solid #087f8c;background:#eaf1f1}@media(max-width:1050px){.cards{grid-template-columns:1fr}}@media(max-width:750px){.layout{display:block}.sidebar{position:static;height:auto}.diagnostics{grid-template-columns:repeat(2,minmax(0,1fr))}}
'''


def experiment_series(task):
    """Same numeric coordinate, separate curves/markers for distinct units."""
    source=task['spec'].get('bc_source')
    offset=0 if source else task['offset']
    series=[];baselines=[]
    for label,color in [('ODE','#087f8c'),('SDE','#d36a35')]:
        records=task['evaluation'][label]
        if task['spec'].get('native_pi05_lora'):
            bc_points=[]
        elif source:
            history=task.get('bc_history', {}).get(label)
            bc_points=points(history) if history else [(source['bc_updates'],source[label.lower()+'_success'])]
        else:
            bc_points=[p for p in points(records) if offset is None or p[0]<=offset]
        if bc_points:
            series.append((f'BC {label}',color,bc_points))
        if offset is not None:
            baseline=[v for step,v in points(records) if step==offset]
            rl_points=points(records,offset=offset,minimum=1)
            if baseline:
                baselines.append((f'BC baseline {label}','#111827' if label=='ODE' else '#6b7280',baseline[-1]))
                rl_points=[(0,baseline[-1])]+rl_points
            if rl_points:
                series.append((f'RL {label}',color,rl_points))
    return series,baselines


def latency_section(data_root):
    root=data_root/'ogpo_libero/latency'
    pi05,ogpo=read_json(root/'pi05.json'),read_json(root/'ogpo.json')
    if not pi05 or not ogpo:
        return '<section class="card detail"><h2>Inference latency</h2><p class="meta">Benchmark in progress on GPU 0; task 29 is stopped. Both ODE samplers use 10 integration steps.</p></section>'
    def cell(result,key):
        return f'{result[key]["median_ms"]:.2f} ms <small>(p95 {result[key]["p95_ms"]:.2f})</small>'
    return f'''<section class="card detail"><h2>Inference latency · RTX 4090 · batch size 1</h2><table><tr><th>Policy</th><th>ODE steps</th><th>Head only, cached conditioning</th><th>Image → action chunk</th><th>Predicted / executed actions</th></tr><tr><td>Original pi05_libero action expert</td><td>10</td><td>{cell(pi05,'head')}</td><td>{cell(pi05,'complete')}</td><td>{pi05['predicted_actions']} / {pi05['executed_actions_in_prior_dsrl']} in earlier DSRL runs</td></tr><tr><td>OGPO BC-trained MLP head</td><td>10</td><td>{cell(ogpo,'head')}</td><td>{cell(ogpo,'complete')}</td><td>4 / 4</td></tr></table><p class="meta">50 warm, synchronized repetitions on GPU 0, on the same real task-29 image pair and proprioception. Excludes compilation, simulator, RPC, and training updates. Full-path encoder implementations differ: OpenPI prefix/KV cache versus OGPO's frozen PaliGemma pooled features. Encoder-only medians: π₀.₅ {pi05['prefix']['median_ms']:.2f} ms; OGPO {ogpo['prefix']['median_ms']:.2f} ms. These are measured estimates, not guarantees under training load.</p><p class="meta">An ODE step is one velocity-network evaluation and integration update; it is not one robot action. Both generate an entire action chunk using 10 Euler steps. π₀.₅ integrates noise→action from t=1→0; OGPO uses t=0→1. Head size and computation differ despite equal ODE step counts. The measured OGPO actor has {ogpo['actor_parameters']:,} parameters. Different chunk lengths mean these timings are not directly comparable rollout throughput; training also includes replay encoding, critic/actor updates and simulator time.</p><p class="meta">Actual OGPO stochastic rollout sampler uses best-of-{ogpo['sde_best_of_n']} candidates plus critic ranking: {ogpo['sde_bon_head']['median_ms']:.2f} ms head/ranking only, {ogpo['sde_bon_complete']['median_ms']:.2f} ms image→chosen chunk. This is separate from the ODE comparison. Source: OGPO_public/scripts/benchmark_heads.py; raw measurements in {html.escape(str(root))}.</p></section>'''


def gradient_explainer():
    """Shared, compact explanation of algorithm gradients and this runner's choices."""
    return r'''<section class="card" id="sampling-gradients" style="margin:14px 0;font-size:13px">
<h2>DPPO, DSRL and OGPO: where do gradients flow?</h2>
<p><b>A sampling step is a denoising/flow-network call, not a physical robot step.</b>
Training the network at every sampling step does not mean differentiating through the whole sampled trajectory.</p>
<div style="overflow-x:auto"><table><thead><tr><th>Method</th><th>What is trained?</th><th>Actor learning signal</th><th>Backpropagation through sampled steps?</th></tr></thead><tbody>
<tr><td><b>DPPO</b></td><td>Denoising policy; implementations may train only selected denoising steps.</td><td>PPO likelihood ratios for denoising transitions, weighted by on-policy return/value advantages.</td><td>Through the network evaluating each transition likelihood. Recorded noisy states are fixed inputs; no end-to-end derivative through the sampled chain.</td></tr>
<tr><td><b>DSRL · our latent-steering runs</b></td><td>Small latent/noise actor and critics. π₀/π₀.₅ stays frozen.</td><td>SAC: critic gradient with respect to the actor's latent action, plus entropy.</td><td>No gradient through the diffusion/flow sampler. The learned critic evaluates latent actions directly.</td></tr>
<tr><td><b>OGPO · our native π₀.₅ LoRA runs</b></td><td>Action-head LoRA adapters and critics; base weights and visual-language encoder stay frozen.</td><td>PPO-style likelihood objective, with detached advantages from off-policy critics scoring fresh candidate chunks.</td><td>Through transition-likelihood networks, not through sampled states or critic scores. Nine stochastic transitions contribute; the tenth, deterministic step is excluded.</td></tr>
</tbody></table></div>
<details><summary>Visual guide: sampling arrows versus gradient arrows</summary>
<p>Read <b>→</b> as forward sampling and <b>⇢</b> as a gradient path. “Fixed” means the sampled values are treated as data when updating parameters.</p>
<div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:10px">
<div style="border:1px solid #dce5e6;padding:10px;border-radius:8px"><b>DPPO</b><p>noise → denoising steps → action → environment</p><p>fixed (xₖ, xₖ₊₁) → log pθ(xₖ₊₁ | xₖ)<br>PPO loss ⇢ likelihood network ⇢ θ</p><small>Environment-return advantages weight the updates. No gradient arrow connects sampled xₖ₊₁ back to xₖ.</small></div>
<div style="border:1px solid #dce5e6;padding:10px;border-radius:8px"><b>DSRL</b><p>latent actor → noise → frozen sampler → action</p><p>SAC loss ⇢ Q(state, latent) ⇢ latent actor</p><small>The actor gradient uses the learned latent critic; it bypasses the frozen sampler entirely.</small></div>
<div style="border:1px solid #dce5e6;padding:10px;border-radius:8px"><b>OGPO</b><p>replay state → candidate sampling → Q scores → fixed advantages</p><p>fixed (xₖ, xₖ₊₁) → log pθ(xₖ₊₁ | xₖ)<br>clipped loss ⇢ likelihood network ⇢ LoRA</p><small>Q supplies weights, not ∂Q/∂action for the actor. Critics are trained separately.</small></div></div>
<p>A pathwise alternative would differentiate action → xₖ → xₖ₋₁ all the way back through sampling. That is not the actor gradient used in these three cases. DPPO is a reference comparison, not an experiment running here.</p></details>
<details><summary>What “gradient recomputation” means in our OGPO code</summary>
<ol><li>Sample 32 candidate chains with the EMA policy and keep their intermediate noisy states. Do not retain a differentiable sampling graph.</li>
<li>Evaluate current and reference transition log-probabilities to form the clipped objective. Compute each candidate's coefficient c = ∂L/∂ℓ, where ℓ is its normalized sum of transition log-probabilities.</li>
<li>For one stochastic step at a time, <b>run the likelihood network forward again</b> on the same fixed states, backpropagate c × log pθ, and add its parameter gradients to an accumulator. Repeat for all nine stochastic steps.</li>
<li>Add the success-example flow-matching regularizer gradient when available, then make <b>one Adam update</b>.</li></ol>
<p><code>∇θ L = Σₖ c · ∇θ log pθ(xₖ₊₁ | xₖ)</code>, with the implementation's dimension/time normalization included.</p>
<p>We recompute <b>forward activations to obtain gradients</b>; we do not resample the chains or apply nine optimizer updates. Parameters stay fixed throughout accumulation. This gives the same gradient for this objective (up to floating-point order) while retaining only one step's backward activations at a time. The tradeoff is extra forward computation. This sequential accumulation is our implementation choice, not a requirement of OGPO.</p></details>
<details><summary>Why this OGPO run costs more than our DSRL runs</summary>
<p>LoRA saves parameter/optimizer memory, but still requires backpropagation through the action transformer. Each actor update samples 32 chains and recomputes likelihoods; critic targets sample up to 8 chains for each of 16 replay states, currently processed sequentially. CPU feature-cache transfers and encoder recomputation on cache misses also add work.</p>
<p>Our OGPO runner executes 4 actions/chunk versus 10 for the compared DSRL runs, and evaluates 25 ODE plus 25 SDE episodes per checkpoint versus 25 DSRL episodes. These are implementation/protocol differences, not a controlled algorithm-speed benchmark. Component-level runtime percentages have not been profiled. Adam states remain on the GPU.</p></details>
<p class="meta">Sources: <a href="https://diffusion-ppo.github.io/">DPPO paper/project</a> · <a href="https://github.com/irom-princeton/dppo/blob/main/model/diffusion/diffusion_ppo.py">DPPO likelihood implementation</a> · <a href="https://github.com/ajwagen/dsrl">official DSRL repository</a> · <a href="https://simchowitzlabpublic.github.io/ogpo-site/">Off-Policy Generative Policy Optimization</a>. Local implementation: <code>OGPO_public/ogpo/pi05_lora.py: logprob_step, actor_update</code> and <code>OGPO_public/scripts/pi05_ogpo.py</code>. Our LoRA adaptation is distinct from the paper's full-finetuning setup and from our earlier separate BC-head experiments.</p>
</section>'''


def comparison_groups(tasks):
    groups = {}
    for task in tasks:
        spec = task['spec']
        source = spec.get('bc_source', {})
        key = (spec['suite'], spec['task_id'], spec.get('comparison_id', spec['seed']), task['budget'],
               spec.get('bc_selection_spec') or source.get('checkpoint'), task['n'], spec.get('native_pi05_lora',False))
        groups.setdefault(key, []).append(task)
    return list(groups.values())


def shared_bc(group, mode=None):
    """Return the shared producer's curves and live status once per reward pair."""
    parent = next((task.get('bc_run') for task in group if task.get('bc_run')), None)
    if parent is None:
        return [], ''
    budget = parent['config'].get('training', {}).get('bc_pi_steps', 0)
    progress = parent['bc_progress'][0] if parent.get('bc_progress') else 0
    measured = [x for rows_ in parent['evaluation'].values() for x, _ in points(rows_)]
    progress = max([progress, *measured])
    if parent['state'] == 'Completed':
        progress = budget
    series = [('BC ' + sampler, color, points(parent['evaluation'][sampler]))
              for sampler, color in [('ODE', '#6b7280'), ('SDE', '#a16207')]
              if mode is None or mode == sampler]
    selection = parent.get('selection', {})
    selected = (f' Selected checkpoint: {selection["bc_updates"]:,} updates, '
                f'{selection["ode_success"]:.0%} ODE success.' if selection else '')
    status = (f'<p class="meta shared-bc-progress"><b>Shared BC</b> · {html.escape(parent["state"])} · '
              f'{progress:,.0f}/{budget:,} optimizer updates<br>{html.escape(parent["runtime"])}<br>'
              f'{html.escape(parent["eta"])}{selected}</p>'
              f'<progress aria-label="Shared BC optimizer updates" value="{progress}" max="{max(1,budget)}"></progress>'
              '<p class="meta">◇ Gray/brown diamonds: shared BC ODE/SDE evaluations, x = optimizer updates. '
              '● Colored circles: RL evaluations, x = physical steps/env, starting at zero. '
              'The phases share numeric coordinates, not units; curves are never joined. '
              'BC SDE uses best-of-8 with untrained critics; checkpoint selection uses ODE.</p>')
    return series, status


def aggregate_reward_curves(group, mode=None):
    """Average independent RL seeds at matching physical-step evaluations."""
    from statistics import mean, stdev
    from scipy.stats import t as student_t
    curves=[]
    for reward, label, color in [('sparse_minus_one','Sparse','#087f8c'), ('robometer','Robometer','#d36a35')]:
        runs=[run for run in group if run['spec'].get('reward') == reward]
        for sampler in ([mode] if mode else ['ODE','SDE']):
            by_step={}
            individual=[]
            for run in runs:
                offset=0 if run['spec'].get('bc_source') else run['offset']
                if offset is None: continue
                values = points(run['evaluation'][sampler],offset=offset)
                if values: individual.append((run['spec']['seed'], values))
                for step,rate in values:
                    by_step.setdefault(step, {})[run['spec']['seed']] = (rate, run['n'])
            ps, bounds, counts=[],[],[]
            for step, observations in sorted(by_step.items()):
                rates=[v[0] for v in observations.values()];value=mean(rates);n=len(rates)
                if n > 1:
                    margin=student_t.ppf(.975,n-1)*stdev(rates)/math.sqrt(n)
                    lo,hi=max(0,value-margin),min(1,value+margin)
                else:
                    lo,hi=wilson(value,next(iter(observations.values()))[1])
                ps.append((step,value));bounds.append((step,lo,hi));counts.append(n)
            curves.append(dict(name=label+' · '+sampler,color=color,points=ps,bounds=bounds,counts=counts,seed_series=individual))
    return curves


def sde_controls():
    return ('<label class="meta" style="display:block;margin:10px 0">'
            '<input type="checkbox" class="show-sde-evals" checked> '
            '<b>Show SDE evaluations</b> · includes BC/RL curves, confidence bands and baselines</label>')


def sde_visibility_script():
    return r'''<style>body.hide-sde [data-eval-mode="SDE"]{display:none!important}</style>
<script>
(()=>{
 const key='libero-show-sde-v1';
 const controls=[...document.querySelectorAll('.show-sde-evals')];
 function apply(show){
  document.body.classList.toggle('hide-sde',!show);
  controls.forEach(input=>input.checked=show);
  document.querySelectorAll('.ogpo-choice').forEach(select=>{
   const option=[...select.options].find(o=>o.value==='SDE');
   if(option)option.disabled=!show;
   if(!show&&select.value==='SDE'){
    select.value='ODE';
    if(typeof updateAlgorithmComparison==='function')updateAlgorithmComparison(select,false);
   }
  });
 }
 let show=true;
 try{show=localStorage.getItem(key)!=='false';}catch(error){}
 apply(show);
 controls.forEach(input=>input.addEventListener('change',()=>{
  apply(input.checked);
  try{localStorage.setItem(key,String(input.checked));}catch(error){}
 }));
})();
</script>'''


def curve_style_note():
    return ('<div class="meta curve-style-note" style="border:1px solid #dce5e6;border-radius:8px;padding:8px;margin:8px 0">'
            '<b>How to read the lines</b><br>'
            '<b>Faint thin lines and small dots:</b> individual RL seeds (hover for seed and success rate). Bold lines show the mean.<br>'
            '<b>━━ Solid colored lines + circles:</b> RL ODE evaluations.<br>'
            '<b>– – Dashed colored lines + circles:</b> RL SDE evaluations (best-of-8 for the MLP runs; single sample for native LoRA).<br>'
            '<b>··· Dotted horizontal gray lines:</b> fixed pre-RL checkpoint baselines, not learning curves. Each comes from a fresh step-zero evaluation.<br>'
            '<b>◇ Gray/brown diamonds with dashed connectors:</b> shared BC evaluations, with optimizer updates on the x-axis.<br>'
            'Teal = sparse reward; orange = Robometer on reward-comparison plots. '
            'ODE still samples initial noise; SDE additionally injects noise during flow sampling. '
            'Shading = 95% intervals: t intervals across available RL seeds when n ≥ 2; Wilson episode intervals when n = 1 and for BC. '
            'RL means use available seeds at each checkpoint; counts can vary while runs finish. Seeds share one BC checkpoint, so bands exclude BC-training variability.</div>')


def comparison_card(group, details):
    esc = html.escape
    first = group[0]
    spec = first['spec']
    title = f"LIBERO-{spec['suite'].split('_')[1]} · task {spec['task_id']:02d}"
    title += f" · {first['budget']/1000:g}k physical steps"
    native = spec.get('native_pi05_lora',False)
    bc_series, bc_status = shared_bc(group)
    series, baselines, status, summary = list(bc_series), [], [bc_status] if bc_status else [], []
    intervals = {name: [(x,*wilson(y,first['n'])) for x,y in ps] for name,_,ps in bc_series}
    seed_series = {}
    for curve in aggregate_reward_curves(group):
        name,color,ps=curve['name'],curve['color'],curve['points']
        series.append((name,color,ps));intervals[name]=curve['bounds'];seed_series[name]=curve['seed_series']
        if ps:
            lo,hi=curve['bounds'][-1][1:]
            summary.append(f'<tr data-eval-mode="{"SDE" if "SDE" in name else "ODE"}"><td>{name}</td><td>{ps[-1][0]:,.0f}</td><td>{ps[-1][1]:.0%}</td><td>{lo:.0%}–{hi:.0%}</td><td>{curve["counts"][-1]}</td></tr>')
            if ps[0][0] == 0:
                lo,hi=curve['bounds'][0][1:]
                baselines.append(('Initial '+name, '#444444' if 'ODE' in name else '#888888',ps[0][1],lo,hi))
    for task in sorted(group,key=lambda t:(t['spec']['reward']=='robometer',t['spec']['seed'])):
        label,color=('Robometer','#d36a35') if task['spec']['reward']=='robometer' else ('Sparse','#087f8c')
        status.append(f'<p class="meta" id="{task["name"]}"><b style="color:{color}">{label} · seed {task["spec"]["seed"]}</b> · {esc(task["state"])} · {task["progress"]:,}/{task["budget"]:,} physical steps<br>{esc(task["runtime"])}<br>{esc(task["eta"])}</p>')
    if native: title += ' · native π₀.₅ LoRA'
    xlabel = 'Count: ◇ BC updates / ● RL physical steps per env' if bc_status else 'Physical RL steps per environment'
    plot = chart(series, title, xlabel=xlabel, probability=True, n=first['n'], baselines=baselines, intervals=intervals, seed_series=seed_series)
    source = spec.get('bc_source', {})
    bc = (f'Both reward variants start from BC-{source["bc_updates"]:,}; BC selection: ODE {source["ode_success"]:.0%}, SDE best-of-8 {source["sde_success"]:.0%}. RL starts at physical step zero; BC uses its own optimizer-update coordinates.' if source else '')
    if spec.get('bc_selection_spec'):
        bc += ' BC: 5,000 updates with evaluation every 500; first ODE checkpoint in [25%, 50%], or the first evaluation if above 50%. Original OGPO MLP path: actor/critic batch 256, 32 actor candidates/state, UTD 1 per physical step; tapered noise from the upstream PaliGemma image preset. Frozen pi05_libero backbone; 50 task demonstrations.'
        reward_spec = next((t['spec'] for t in group if t['spec'].get('reward') == 'robometer'), spec)
        precision = spec.get('encoder_dtype', 'float32')
        bc += f' Encoder precision: {precision}; MLP/optimizer precision: float32.'
        if reward_spec.get('allow_reward_colocation'):
            sanity = read_json(Path(reward_spec['colocation_sanity'])) if reward_spec.get('colocation_sanity') else {}
            bc += (' Co-location full-batch sanity passed; Robometer and policy share GPU 1, while sparse RL uses GPU 0.' if sanity.get('status') == 'passed' else ' Co-location validation pending; RL waits for a passing full-batch sanity check.')
        else:
            bc += ' Co-location failed its memory test; Robometer RL uses policy GPU 0 and reward GPU 1 after sparse RL.'
    if spec.get('save_replay_buffer') is False:
        bc += ' Replay snapshots disabled: replay remains in RAM only. Rolling model/optimizer checkpoints, evaluations and logs are retained.'
    if native: bc = 'Original pi05_libero action expert with rank-32 LoRA; no new BC policy or task demonstrations. Frozen backbone and pretrained weights. ODE and SDE both use EMA adapters; no critic ranking. Smaller native-head update batches; see provenance.json for the exact protocol.'
    sampling_note = 'Dashed: single-sample SDE, no critic ranking.' if native else 'Dashed: SDE best-of-8 (EMA actor + critic ranking).'
    date = datetime.fromtimestamp(first['root'].stat().st_mtime,timezone.utc).strftime('%Y-%m-%d')
    return f"""<article class="card"><h2>{title}</h2><small>{date} · seeds {', '.join(map(str, sorted({t['spec']['seed'] for t in group})))} · sparse vs Robometer</small>{''.join(status)}<p class="meta">{esc(spec['prompt'])}</p>{plot}{curve_style_note()}<p class="meta">Teal: sparse reward. Orange: Robometer. Solid: ODE. {sampling_note} Dotted: initial checkpoint evaluations with 95% CI.</p><p class="meta">{bc}</p><table><tr><th>Reward / sampler</th><th>Physical steps</th><th>Success</th><th>95% CI</th><th>RL seeds</th></tr>{''.join(summary)}</table>{''.join(details[t['name']] for t in group)}</article>"""


def render(data_root):
    tasks=load(data_root);esc=html.escape
    nav=[];cards=[];cards_by_name={};details_by_name={}
    for t in tasks:
        name=t['name'];spec=t['spec'];source=spec.get('bc_source')
        offset=0 if source else t['offset']
        title=f"LIBERO-{spec['suite'].split('_')[1]} · task {spec['task_id']:02d}"
        date=datetime.fromtimestamp((data_root/'ogpo_libero'/f'{name}.json').stat().st_mtime,timezone.utc).strftime('%Y-%m-%d')
        subtitle=('native π₀.₅ LoRA · no BC' if spec.get('native_pi05_lora') else f"from BC-{source['bc_updates']:,}" if source else 'BC → OGPO')
        bc_budget=t['config'].get('training',{}).get('bc_pi_steps',500000)
        bc_current=t['bc_progress'][0] if t['bc_progress'] else 0
        logged=points(t['bc'],'bc_loss')
        if logged: bc_current=max(bc_current,int(logged[-1][0]))
        if t['offset'] is not None and not source: bc_current=t['offset']
        count_text=f'BC: {bc_current:,} updates · RL: {t["progress"]:,}/{t["budget"]:,} steps'
        if source: count_text=f'RL starts from BC-{source["bc_updates"]:,} · RL: {t["progress"]:,}/{t["budget"]:,} steps'
        nav.append(f'<a class="side-link" href="#{name}">{title}<small>{subtitle} · {esc(t["state"])}<br>{count_text}</small></a>')
        series,baselines=experiment_series(t)
        plot=chart(series,title,'Count: ◇ BC updates / ● RL physical steps per env',probability=True,n=t['n'],baselines=baselines)
        summary=[]
        for label,color,ps in series:
            lower,upper=wilson(ps[-1][1],t['n'])
            summary.append(f'<tr data-eval-mode="{"SDE" if "SDE" in label else "ODE"}"><td>{label}</td><td>{ps[-1][0]:,.0f}</td><td>{ps[-1][1]:.0%}</td><td>{lower:.0%}–{upper:.0%}</td></tr>')
        eta=f'<p class="eta">{esc(t["eta"])}</p>' if t['eta'] else ''
        if t['state']=='Behavior cloning':
            eta+=f'<p class="meta">BC: {bc_current:,}/{bc_budget:,} optimizer updates. Diamonds below show completed BC evaluations.</p>'
        if source:
            eta+=f'<p class="meta">Selected BC-{source["bc_updates"]:,}: {source["ode_success"]:.0%} ODE / {source["sde_success"]:.0%} SDE. RL baseline is a fresh evaluation at zero physical steps.</p>'
        if t['config'].get('training',{}).get('bc_only'):
            interval=t['config'].get('intervals',{}).get('eval_interval_bc',spec.get('overrides',{}).get('intervals.eval_interval_bc',20000))
            eta+=f'<p class="meta">BC-only: {bc_budget:,} updates, evaluated every {interval:,}; subsequent RL is scheduled separately when a qualifying checkpoint exists.</p>'
        metrics=[]
        groups=[('Actor updates',[('actor/pg_loss','Policy objective'),('actor/approx_kl','Approximate KL'),('actor/ratio','Importance ratio'),('actor/ratio_clipped_total','Clipped fraction'),('actor/adv_std','Advantage std')]),('Exploration and critic',[('actor/entropy','Entropy'),('actor/info_sigma_mean','Noise sigma'),('critic/q_mean','Q mean'),('critic/next_q_mean','Next Q mean'),('critic/td_loss','TD loss')])]
        for group,keys in groups:
            figs=[]
            for key,label in keys:
                ps=points(t['diagnostics'],key,offset or 0) if offset is not None else []
                figs.append(f'<div class="diagnostic"><h3>{label}</h3>{chart([(label,"#087f8c",ps)],label)}</div>')
            metrics.append(f'<h3>{group}</h3><div class="diagnostics">{"".join(figs)}</div>')
        coverage='LIBERO-10 is in the documented pi05_libero training suites.' if spec['suite']=='libero_10' else 'LIBERO-90 is outside the documented pi05_libero task suites; broader pretraining overlap is unknown.'
        reward_label = 'Robometer reward' if spec.get('reward') == 'robometer' else 'sparse reward'
        reward_note = ('Robometer-4B progress minus one; success-buffer labels use predicted success ≥0.65. Evaluation uses simulator success. Robometer LIBERO-10 is in distribution; LIBERO-90 is held out.' if spec.get('reward') == 'robometer' else 'Robometer is not used.')
        training_note = ('Native pretrained action head; no task BC demonstrations used.' if spec.get('native_pi05_lora') else 'The new actor uses 50 demonstrations of this task; this is not zero-shot evaluation.')
        details=f'<details id="{name}-details"><summary>OGPO diagnostics and provenance</summary>{"".join(metrics)}<p class="meta">{coverage} {training_note} {reward_note} Source: {esc(str(t["root"]))}.</p></details>'
        details = details.replace('OGPO diagnostics and provenance</summary>', reward_label + ' diagnostics and provenance</summary>')
        details_by_name[name] = details
        progress=t['progress'] if offset is not None else bc_current
        total=t['budget'] if offset is not None else bc_budget
        cards.append(f'''<article class="card" id="{name}"><h2>{title}</h2><small>{date} · seed {spec['seed']} · {subtitle} · {reward_label}</small><p class="meta"><b>{esc(t["runtime"])}</b></p>{eta}<p class="meta">{esc(spec['prompt'])}</p><div class="facts"><span class="badge">{esc(t['state'])}</span><span>{'GPU '+str(t['gpu']) if t['gpu'] is not None else ''}</span><span>{count_text}</span></div><progress value="{progress}" max="{max(1,total)}"></progress>{plot}<p class="meta">◇ Dashed: BC updates. ● Solid: RL physical steps/env. Teal: ODE; orange: stochastic sampling (best-of-8 in BC and RL rollout evaluations). The units share numbers, not equivalent computation or data. BC and RL are separate curves. Dotted lines: BC baseline with 95% CI.</p><table><tr><th>Phase / sampler</th><th>Count</th><th>Success</th><th>95% CI</th></tr>{''.join(summary) or '<tr><td colspan="4">Waiting for the first evaluation.</td></tr>'}</table>{details}</article>''')
        cards_by_name[name] = cards[-1]
    cards = []
    for group in comparison_groups(tasks):
        rewards = {t['spec'].get('reward') for t in group}
        if len(group) >= 2 and rewards == {'sparse_minus_one', 'robometer'}:
            cards.append(comparison_card(group, details_by_name))
        else:
            cards.extend(cards_by_name[t['name']] for t in group)
    updated=datetime.now(timezone.utc).isoformat(timespec='seconds')
    page=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60"><title>OGPO · LIBERO results</title><style>{CSS}</style></head><body><div class="layout"><aside class="sidebar"><a class="home" href="/">← LIBERO results</a><p class="meta">OGPO experiment progress</p><select aria-label="Choose experiment" onchange="location.hash=this.value"><option value="top">All experiments</option>{''.join(f'<option value="{t["name"]}">{t["name"]}</option>' for t in tasks)}</select>{''.join(nav)}<p class="meta">Refreshes every 60 seconds.</p></aside><main id="top"><div class="eyebrow">Reward-modeling experiments</div><h1>OGPO · LIBERO fine-tuning</h1><p class="meta">Updated {updated} · π₀.₅ experiments: separate BC heads or native action-expert LoRA, labeled per experiment</p><div class="facts"><b>One performance plot per experiment</b><span>1 training env per run</span><span>4-action chunks</span><span>25 episodes per evaluation mode</span></div><p class="note">◇ BC optimizer updates and ● RL physical steps share a numeric x-axis. They are different units and are never joined into a single trajectory. Shading: <b>95% intervals</b> across RL seeds when available, otherwise Wilson episode intervals; see each plot’s legend. No statistical-significance claim.</p>{gradient_explainer()}{sde_controls()}<div class="cards">{''.join(cards)}</div>{latency_section(data_root)}<p class="meta">Training rewards: sparse = simulator success − 1; Robometer = predicted progress − 1. Both are in [−1, 0]. GPU placement and encoder precision are run-specific; see each experiment’s protocol; success terminates, 600 steps truncates. RL warm-up: 20k physical steps, followed by evaluations every 10k steps. BC budgets and evaluation intervals are run-specific. Evaluation states 25–49 are disjoint from online training states 0–24; overlap with demonstration states is not established.</p></main></div>{sde_visibility_script()}</body></html>'''
    return page.encode()
