#!/usr/bin/env python3
"""Build the per-task, out-of-the-box LIBERO checkpoint comparison (stdlib only)."""
import argparse
import csv
import html
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/experiment_atlas'
SUITES = [('libero_spatial', 'LIBERO-Spatial', 10), ('libero_object', 'LIBERO-Object', 10),
          ('libero_goal', 'LIBERO-Goal', 10), ('libero_10', 'LIBERO-10', 10), ('libero_90', 'LIBERO-90', 90)]


def build(snapshot):
    base = ROOT / snapshot
    evidence = json.loads((base / 'frozen_evaluations.json').read_text())
    captured = json.loads((base / 'index.json').read_text())['exported_at']
    paired = defaultdict(dict)
    for record in evidence:
        path, config = record['path'], record['config']
        if path.startswith('pi05_libero_all_tasks_20261019/'):
            model = 'pi05'
        elif (path.startswith('pi0_libero_families_20261014/')
              or (path.startswith('pi0_task_probe_20261010/') and config.get('suite') == 'libero_90')
              or path.startswith('pi0_libero90_task55_20261010/')):
            model = 'pi0'
        else:
            continue
        key = (config['suite'], config['task_id'])
        if model in paired[key]:
            raise ValueError(f'Duplicate baseline: {key}, {model}')
        paired[key][model] = record
    for suite, _, count in SUITES:
        for task_id in range(count):
            pair = paired[(suite, task_id)]
            if set(pair) != {'pi0', 'pi05'}:
                raise ValueError(f'Missing baseline: {suite}, {task_id}')
            a, b = pair['pi0']['config'], pair['pi05']['config']
            if a['task'] != b['task']:
                raise ValueError(f'Task names disagree: {suite}, {task_id}')
            for model, horizon, chunk, checkpoint in [('pi0', 400, 20, 'pi0_libero'), ('pi05', 600 if suite == 'libero_10' else 400, 10, 'pi05_libero')]:
                record = pair[model]
                config, summary = record['config'], record['summary']
                assert config['horizon'] == horizon and config['action_exec_len'] == chunk
                assert Path(config['checkpoint']).name == checkpoint
                assert config['seed'] == 100 and config['eval_state_ids'] == list(range(25, 50))
                assert summary['episodes'] == 25
                assert abs(summary['success_rate'] - summary['successes'] / summary['episodes']) < 1e-9
                assert (base / 'data' / record['path']).is_file()
    esc = html.escape

    def source(record):
        return '/' + (base / 'data' / record['path']).relative_to(ROOT).as_posix()

    def rate(record):
        s = record['summary']
        return (f'<a class="rate" href="{esc(source(record))}" title="Open original evaluation summary">'
                f'{100*s["success_rate"]:.1f}%</a><small>{s["successes"]} / {s["episodes"]} episodes</small>')

    summaries, tables, csv_rows = [], [], []
    for suite, title, count in SUITES:
        rows = [paired[(suite, i)] for i in range(count)]
        means = {m: sum(p[m]['summary']['success_rate'] for p in rows) / count for m in ['pi0', 'pi05']}
        coverage = 'Held out' if suite == 'libero_90' else 'Included'
        coverage_class = 'held-out' if suite == 'libero_90' else 'included'
        coverage_badge = f'<span class="coverage {coverage_class}">{coverage}</span>'
        summaries.append(f'<tr><th scope="row"><a href="#{suite}">{title}</a></th><td>{count}</td>'
                         f'<td>{100*means["pi0"]:.1f}%</td><td>{100*means["pi05"]:.1f}%</td><td>400 / {rows[0]["pi05"]["config"]["horizon"]}</td>'
                         f'<td>{coverage_badge}</td><td>{coverage_badge}</td></tr>')
        cells = []
        for task_id, pair in enumerate(rows):
            a, b = pair['pi0'], pair['pi05']
            task = a['config']['task']
            cells.append(f'<tr data-task="{esc(task.lower())}" data-id="{task_id}"><td>{task_id}</td>'
                         f'<th scope="row" class="task-name">{esc(task.replace("_", " "))}'
                         f'<small>{esc(task)}</small></th><td>{rate(a)}</td><td>{rate(b)}</td></tr>')
            csv_rows.append(dict(suite=suite, task_id=task_id, task=task,
                pi0_suite_training_coverage=coverage, pi05_suite_training_coverage=coverage,
                robometer_4b_suite_training_coverage=coverage,
                pi0_success_rate=a['summary']['success_rate'], pi0_successes=a['summary']['successes'],
                pi0_episodes=25, pi0_horizon=400, pi0_source=source(a),
                pi05_success_rate=b['summary']['success_rate'], pi05_successes=b['summary']['successes'],
                pi05_episodes=25, pi05_horizon=b['config']['horizon'], pi05_source=source(b)))
        tables.append(f'<section class="task-suite" id="{suite}" data-suite="{suite}"><div class="section-head">'
                      f'<h2>{title}</h2><span class="caption">{count} tasks</span></div>'
                      f'<p class="suite-coverage">Training suite coverage: <strong>π₀ / π₀.₅</strong> {coverage_badge}'
                      f' · <strong>Robometer-4B</strong> {coverage_badge}</p><div class="table-wrap">'
                      '<table><thead><tr><th>ID</th><th>Task</th><th>π₀ LIBERO</th><th>π₀.₅ LIBERO</th></tr></thead>'
                      f'<tbody>{"".join(cells)}</tbody></table></div></section>')
    template = '''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Out-of-the-box LIBERO · π₀ vs π₀.₅</title><link rel="stylesheet" href="style.css">
<style>
main{max-width:1600px}.baseline-title{font-size:clamp(34px,4vw,58px)}.baseline-summary{max-width:100%}
.coverage{display:inline-block;padding:4px 8px;border-radius:4px;font-size:14px;white-space:nowrap}.coverage.included{background:#dcf2e9;color:#176348}.coverage.held-out{background:#fff0d8;color:#82551b}.suite-coverage{font-size:14px}.baseline-summary thead th{white-space:normal;min-width:90px}
.task-name{background:white;white-space:normal;min-width:230px;max-width:720px;font-size:16px;font-weight:500;letter-spacing:0;color:#172c40;overflow-wrap:anywhere}
.task-name small{display:block;font-size:12px;color:var(--muted);font-weight:400;margin-top:7px}.rate{font-size:20px;font-variant-numeric:tabular-nums;font-weight:600}.task-suite td:not(:first-child){min-width:140px}td small{white-space:nowrap}.task-suite th:first-child,.task-suite td:first-child{min-width:45px;width:55px}.task-suite{padding:35px 0}.task-suite tr[hidden],.task-suite[hidden]{display:none}
#task-filters{position:sticky;top:0;z-index:1;background:#f6f8fb;padding:16px 0;border-bottom:1px solid var(--line)}#task-filters .filters{margin:0}#task-count{margin:10px 0 0}.baseline-summary th{font-size:14px;letter-spacing:0}.baseline-summary tbody th{background:white}
@media(max-width:760px){#task-filters{position:static}.task-name{min-width:200px}.task-suite td:not(:first-child){min-width:115px}}
</style></head><body>
<aside><a class="brand" href="./"><span class="mark">r</span> robometer<span class="subbrand">POLICY LEARNING</span></a>
<div class="sidebar-nav"><div class="nav-group"><h2 class="nav-label">Pages</h2><nav aria-label="Pages"><a href="./">Experiment Atlas</a><a href="baselines.html" class="active" aria-current="page">Baselines</a></nav></div><div class="nav-group"><h2 class="nav-label">On this page</h2><nav aria-label="On this page"><a href="#overview">Overview</a>@@NAV@@</nav></div></div>
<div class="aside-note">Released LIBERO checkpoints.<br>All 130 tasks, side by side.</div></aside>
<main><header><span>RESEARCH / FROZEN CHECKPOINTS</span><a href="./">Back to Experiment Atlas</a></header>
<section id="overview"><div class="eyebrow">OUT-OF-THE-BOX PERFORMANCE</div><h1 class="baseline-title">LIBERO: π₀ and π₀.₅</h1>
<p class="intro">Success rates for the released <code>pi0_libero</code> and <code>pi05_libero</code> checkpoints across every LIBERO task, before policy learning.</p>
<div class="snapshot">130 tasks · 25 episodes per task and checkpoint · Saved evidence @@DATE@@</div>
<div class="protocol"><h3>Evaluation protocol</h3><p>Both checkpoints use seed 100 and reset-state IDs 25–49. π₀ executes up to <strong>20 actions per chunk</strong> with a <strong>400-step horizon</strong>; π₀.₅ executes up to <strong>10 actions per chunk</strong> with a <strong>600-step horizon on LIBERO-10 and 400 steps on the other suites</strong>. The LIBERO-10 comparison therefore uses different horizons; its difference does not isolate model quality.</p><p>Task IDs are zero-based. Success means simulator task completion. Click any success rate to inspect its original evaluation summary, episode outcomes, and recorded Wilson interval; the neighboring <code>config.json</code> records the protocol.</p></div>
<div class="protocol" id="training-coverage"><h3>Which suites were used in training?</h3>
<p><strong>LIBERO-Spatial, Object, Goal, and LIBERO-10 are included</strong> in the documented LIBERO training data for both <code>pi0_libero</code> and <code>pi05_libero</code>, and in the RBM-1M training data for the general <code>Robometer-4B</code> used in this repository’s reported reward-learning runs. <strong>LIBERO-90 is held out</strong> from these documented training splits.</p>
<p>“Included” describes suite membership, not verified overlap with these particular evaluation trajectories or reset states. “Held out” describes tasks within the same simulator; broader foundation-model pretraining coverage is not established. Robometer coverage here applies to <code>robometer/Robometer-4B@beef63bc914c5c189329d49c6d712d96d632aa34</code>; separately fine-tuned Robometer variants may use different splits. These baseline success rates evaluate the π policies; Robometer coverage is provided as context for reward-learning comparisons.</p>
<p>Sources: <a href="https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py">OpenPI checkpoint training configs</a> · <a href="https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/convert_libero_data_to_lerobot.py">OpenPI LIBERO dataset construction</a> · <a href="https://huggingface.co/datasets/physical-intelligence/libero/blob/main/meta/tasks.jsonl">Policy training task list</a> · <a href="https://arxiv.org/html/2603.02115v2#A1.SS1.SSS1.Px7">Robometer paper, Appendix A: LIBERO training split</a> · <a href="/docs/reports/reward_learning_summary_20261028/report.md">Local checkpoint provenance</a></p></div>
<h2>Suite-level success</h2><p class="caption">Unweighted mean across tasks within each suite. Each task contributes 25 episodes.</p>
<div class="table-wrap baseline-summary"><table><thead><tr><th>Suite</th><th>Tasks</th><th>π₀ LIBERO</th><th>π₀.₅ LIBERO</th><th>Horizon π₀ / π₀.₅</th><th>π₀ / π₀.₅ training suite</th><th>Robometer-4B training suite</th></tr></thead><tbody>@@SUMMARY@@</tbody></table></div>
<p><a href="baselines.csv" download>Download all task results (CSV)</a> · <a href="@@EVIDENCE@@">Full frozen-evaluation catalog</a></p></section>
<div id="task-filters"><div class="filters"><label>Find a task<input id="task-search" type="search" placeholder="Task name or task ID"></label><label>Suite<select id="task-suite"><option value="">All suites</option>@@OPTIONS@@</select></label></div><p id="task-count" class="caption" aria-live="polite">Showing all 130 tasks</p></div>
<div id="task-tables">@@TABLES@@</div><p id="empty" hidden>No tasks match these filters.</p>
<footer>Source exports: π₀ family sweep (October 14), π₀ LIBERO-90 probe plus task-55 baseline (October 10), and π₀.₅ full sweep (October 19), 2026.<br><a href="./">Experiment Atlas</a> · <a href="README.md">How to extend this page</a></footer></main>
<script>
const search=document.getElementById('task-search'),suite=document.getElementById('task-suite');
function filter(){const query=search.value.trim().toLowerCase();let total=0;
 document.querySelectorAll('.task-suite').forEach(section=>{let count=0;
  section.querySelectorAll('tbody tr').forEach(row=>{const show=(!suite.value||suite.value===section.dataset.suite)&&(!query||row.dataset.task.includes(query)||row.dataset.id===query);row.hidden=!show;if(show)count++;});
  section.hidden=!count;section.querySelector('.section-head .caption').textContent=count+' tasks';total+=count;
 });document.getElementById('task-count').textContent='Showing '+total+' of 130 tasks';document.getElementById('empty').hidden=total>0;
}search.addEventListener('input',filter);suite.addEventListener('change',filter);document.body.dataset.status='ready';
</script></body></html>
'''
    substitutions = {'NAV': ''.join(f'<a href="#{suite}">{title}</a>' for suite, title, _ in SUITES),
                     'DATE': esc(captured[:10]), 'SUMMARY': ''.join(summaries),
                     'EVIDENCE': '/' + (base / 'frozen_evaluations.json').relative_to(ROOT).as_posix(),
                     'OPTIONS': ''.join(f'<option value="{suite}">{title}</option>' for suite, title, _ in SUITES),
                     'TABLES': ''.join(tables)}
    for key, value in substitutions.items():
        template = template.replace('@@' + key + '@@', value)
    (OUT / 'baselines.html').write_text(template)
    with (OUT / 'baselines.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f'Built baseline subpage and CSV: {len(csv_rows)} tasks, {2*len(csv_rows)} source evaluations')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', default='results/libero_20261031')
    build(parser.parse_args().snapshot)
