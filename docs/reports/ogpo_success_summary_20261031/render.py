"""Render the fixed OGPO success snapshot; requires matplotlib, never reads live runs."""
import csv
import json
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parent
snapshot = json.loads((ROOT / 'evidence.json').read_text())
runs = snapshot['runs']
colors = {'sparse_minus_one': '#087f8c', 'robometer': '#d36a35'}
labels = {'sparse_minus_one': 'Sparse', 'robometer': 'Robometer'}

def group(task, budget, native=False):
    return sorted([r for r in runs if r['spec']['task_id'] == task
                   and r['budget'] == budget
                   and bool(r['spec'].get('native_pi05_lora')) == native
                   and ('_50k' in r['name'] or '_100k' in r['name'])],
                  key=lambda r: (r['spec']['reward'], r['spec']['seed']))

def series(r, mode):
    return dict(r['evaluation'][mode])

def cohort_mean(rs, mode):
    maps = [series(r, mode) for r in rs]
    common = sorted(set.intersection(*(set(m) for m in maps)))
    return [(s, mean(m[s] for m in maps)) for s in common]

def percent(p):
    return f'{100*p:.1f}%'

groups = [
    ('LIBERO-10 task 8 · MLP · 50k', group(8, 50000)),
    ('LIBERO-90 task 29 · MLP · 50k', group(29, 50000)),
    ('LIBERO-90 task 2 · MLP · 50k', group(2, 50000)),
    ('LIBERO-90 task 2 · MLP · 100k', group(2, 100000)),
    ('LIBERO-90 task 29 · native π₀.₅ LoRA', group(29, 50000, True)),
]
with (ROOT / 'evaluation_points.csv').open('w') as stream:
    fields = ['run', 'reward', 'seed', 'mode', 'state', 'physical_steps_total',
              'physical_steps_per_env', 'num_envs', 'full_chunk_equivalents',
              'max_actions_per_chunk', 'successes', 'episodes', 'success_rate']
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator='\n')
    writer.writeheader()
    for r in runs:
        assert r['num_envs'] == 1
        if not r['spec'].get('native_pi05_lora') and any(r['evaluation'].values()):
            assert r['config']['agent']['horizon_length'] == 4
        for mode, values in r['evaluation'].items():
            for step, rate in values:
                assert 0 <= rate <= 1 and step >= 0
                wins = round(rate * r['episodes'])
                assert abs(wins / r['episodes'] - rate) < 1e-6
                writer.writerow(dict(run=r['name'], reward=r['spec']['reward'],
                    seed=r['spec']['seed'], mode=mode, state=r['state'],
                    physical_steps_total=int(step), physical_steps_per_env=int(step),
                    num_envs=1, full_chunk_equivalents=step/4, max_actions_per_chunk=4,
                    successes=wins, episodes=r['episodes'], success_rate=rate))

text = ['# OGPO success rates and training step counts', '',
        f"**Snapshot · {snapshot['captured_at']}** · Task IDs are zero-based.", '',
        'Success is simulator task completion, including for Robometer-trained policies. '
        'ODE and SDE are separate evaluation samplers; MLP SDE uses best-of-8, while native LoRA uses a single sample. '
        'Each checkpoint uses 25 episodes per seed and sampler. Multiple seeds share a BC checkpoint.', '',
        '[Numerical evidence](evidence.json) · [All evaluation points](evaluation_points.csv) · '
        '[DSRL and Robometer report](../reward_learning_summary_20261028/report.md)', '',
        '## X axis and environment counts', '',
        '| Runs | Training envs per run | Actions executed per chunk | X axis | Full-chunk equivalent |',
        '|---|---:|---:|---|---|',
        '| MLP OGPO, 50k budget | 1 | Up to 4 | 50,000 physical steps/env = 50,000 total | 12,500 |',
        '| MLP OGPO, 100k budget | 1 | Up to 4 | 100,000 physical steps/env = 100,000 total | 25,000 |',
        '| Native π₀.₅ LoRA, 50k budget | 1 | Up to 4 | 50,000 physical steps/env = 50,000 total | 12,500 |', '',
        '**Full-chunk equivalents = physical steps ÷ 4**, not measured chunk counts. '
        'Early termination can shorten chunks, so the actual number of policy calls can be higher. '
        'The upper chart axis uses this equivalent. Seeds are independent runs, not parallel environments within one run. '
        'Evaluation rollouts and BC optimizer updates are excluded from the RL x axis. '
        'For historical MLP runs, the recorded BC offset is subtracted from the CSV step counter.', '',
        'MLP replay records individual physical transitions; native LoRA replay records variable-duration chunks. '
        'Neither a replay-entry count nor an optimizer-update count is interchangeable with the physical-step axis. '
        'See [MLP collection](../../../OGPO/ogpo/runners/online_rl_runner.py) and '
        '[native LoRA collection](../../../OGPO/scripts/pi05_ogpo.py).', '',
        '## Success comparisons', '',
        'Solid curves show fixed-cohort means; faint curves show the individual seeds in that cohort. '
        'The 100k comparison uses only seeds with both reward arms completed. '
        'Native LoRA curves end at each run’s latest complete evaluation. No significance tests are implied.', '']

for idx, (title, rs) in enumerate(groups):
    assert rs, title
    native = bool(rs[0]['spec'].get('native_pi05_lora'))
    if idx == 3:
        complete = [{r['spec']['seed'] for r in rs if r['spec']['reward'] == rew
                     and all(r['budget'] in series(r, m) for m in ('ODE','SDE'))}
                    for rew in colors]
        matched = set.intersection(*complete)
        selected = [r for r in rs if r['spec']['seed'] in matched]
    else:
        selected = rs
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    text += [f'### {title}', '',
             '| Reward | Seeds | Latest steps/env | ODE start → latest | SDE start → latest |',
             '|---|---|---:|---:|---:|']
    for reward, color in colors.items():
        arm = [r for r in selected if r['spec']['reward'] == reward]
        assert arm
        curves = {m: cohort_mean(arm, m) for m in ('ODE','SDE')}
        ends = {p[-1][0] for p in curves.values()}
        assert len(ends) == 1
        end = ends.pop()
        text.append(f"| {labels[reward]} | {', '.join(str(r['spec']['seed']) for r in arm)} | {end:,.0f} | "
                    + ' | '.join(f'{percent(curves[m][0][1])} → **{percent(curves[m][-1][1])}**' for m in ('ODE','SDE')) + ' |')
        for ax, mode in zip(axes, ('ODE','SDE')):
            for r in arm:
                ps = r['evaluation'][mode]
                ax.plot(*zip(*ps), color=color, alpha=.25, linewidth=1)
            ax.plot(*zip(*curves[mode]), color=color, marker='o', markersize=4,
                    label=f'{labels[reward]} (n={len(arm)})', linewidth=2)
    for ax, mode in zip(axes, ('ODE','SDE')):
        ax.set(title=mode, xlabel='Physical training steps per environment')
        # Curves are stored as fractions; format the y axis as percentages.
        ax.set_ylim(-.02, 1.02)
        ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x*100:.0f}%'))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k'))
        ax.grid(alpha=.18)
        ax.legend(loc='lower right', fontsize=8)
        top = ax.secondary_xaxis('top', functions=(lambda x:x/4, lambda x:x*4))
        top.set_xlabel('Full-chunk equivalents (steps ÷ 4)', fontsize=9)
        top.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x/1000:g}k'))
    axes[0].set_ylabel('Evaluation success')
    fig.suptitle(title + ' · 1 training env/run', fontsize=12)
    fig.tight_layout()
    name = f'curve_{idx:02d}.png'
    fig.savefig(ROOT/name, dpi=170)
    plt.close(fig)
    if idx == 3:
        text += ['', 'This is a partial study: only seeds ' + ', '.join(map(str, sorted(matched)))
                 + ' have both arms evaluated at 100k. Other completed or ongoing seeds are listed below; queued runs contribute no observations.']
    if native:
        text += ['', 'Sparse completed 50k; Robometer stopped with its latest complete evaluation at 30,005 steps. '
                 'These latest values have unequal training budgets. This is the native LoRA adaptation, separate from the MLP-head experiments.']
    text += ['', '<details>', f'<summary>Show {title} success curves</summary>', '',
             f'![{title}: ODE and SDE success versus physical steps]({name})', '', '</details>', '']

text += ['## Individual latest evaluations', '',
         'This table retains partial runs and the two earlier stopped MLP pilots. '
         'Rows report the latest fully logged evaluation, not the current collection counter. '
         'The CSV also preserves every intermediate checkpoint and success count.', '',
         '| Run | State | ODE latest success @ steps/env | SDE latest success @ steps/env |',
         '|---|---|---:|---:|']
for r in sorted(runs, key=lambda r:r['name']):
    vals=[]
    for m in ('ODE','SDE'):
        p=r['evaluation'][m]
        vals.append(f'{percent(p[-1][1])} @ {p[-1][0]:,.0f}' if p else '—')
    text.append(f"| `{r['name']}` | {r['state']} | {' | '.join(vals)} |")
text += ['', '## Snapshot provenance', '',
         'The snapshot contains evaluations and run configurations from the local OGPO result directories, '
         'including records newer than the earlier numerical export. Reads occurred sequentially while some runs were active. '
         'Source paths and CSV hashes are retained in `evidence.json`; the captured values and CSV make the report portable. '
         'BC-only searches, smoke tests, and memory benchmarks are outside this RL success summary.', '',
         'Regenerate tables and plots from the fixed evidence with `python render.py` in this directory '
         '(requires Matplotlib). This does not read live runs or launch training.', '']
(ROOT/'report.md').write_text('\n'.join(text))
print('Rendered report, five figures, and evaluation CSV')
