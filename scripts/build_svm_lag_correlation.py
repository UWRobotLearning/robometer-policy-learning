#!/usr/bin/env python3
"""Plot cumulative within-seed success correlations at a fixed 10k-step lag."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

PAGE = Path(__file__).resolve().parents[1] / 'docs/experiment_atlas'
INTERVAL = 10_000
COLORS = {'SVM':'#087e70','Historical sparse':'#b85b30','Earlier sparse controls':'#6953a4'}


def checkpoint_correlations(rows):
    """Each coefficient uses only one checkpoint pair, matched across seeds."""
    rates = {}
    for row in rows:
        step, success = float(row['step']), float(row['success'])
        if not np.isfinite(step) or not np.isfinite(success) or not 0 <= success <= 1:
            raise ValueError('Invalid evaluation point')
        if step != int(step) or step < 0 or int(step) % INTERVAL:
            continue
        key = (row['seed'],int(step))
        if key in rates:
            raise ValueError(f'Duplicate seed/checkpoint: {key}')
        rates[key] = success
    seeds = sorted({seed for seed,_ in rates})
    individual = []
    for seed in seeds:
        pairs, points = [], []
        for step in range(INTERVAL,max(t for s,t in rates if s == seed)+1,INTERVAL):
            if (seed,step-INTERVAL) not in rates or (seed,step) not in rates:
                points.append(dict(physical_step=step, correlation=None, n_pairs=len(pairs),
                                   reason='Missing exact-lag checkpoint pair'))
                continue
            pairs.append(dict(previous_step=step-INTERVAL,physical_step=step,
                              previous=rates[seed,step-INTERVAL],current=rates[seed,step]))
            a = np.asarray([p['previous'] for p in pairs])
            b = np.asarray([p['current'] for p in pairs])
            correlation, reason = None, None
            if len(pairs) < 2:
                reason = 'Fewer than two checkpoint pairs'
            elif np.ptp(a) == 0 or np.ptp(b) == 0:
                reason = 'Constant success rates in one paired vector'
            else:
                correlation = float(np.corrcoef(a,b)[0,1])
            points.append(dict(physical_step=step,correlation=correlation,
                               n_pairs=len(pairs),reason=reason))
        individual.append(dict(seed=seed,points=points,pairs=pairs))
    mean = []
    for step in range(INTERVAL,max((t for _,t in rates),default=0)+1,INTERVAL):
        values = [(s['seed'],p['correlation']) for s in individual for p in s['points']
                  if p['physical_step']==step and p['correlation'] is not None]
        mean.append(dict(physical_step=step,correlation=float(np.mean([v for _,v in values])) if values else None,
                         n_seeds=len(values),seeds=[seed for seed,_ in values]))
    return dict(seeds=individual,points=mean)


def build(page):
    run = json.loads((page/'svm-run.json').read_text())
    cohort = json.loads((page/'svm-replicates.json').read_text())
    svm = cohort['evaluations'] if cohort.get('campaign_id') == run.get('campaign_id') else run['evaluations']
    controls = json.loads((page/'svm-inactive-controls.json').read_text())
    datasets = [('SVM',svm),('Historical sparse',run['baseline']),
                ('Earlier sparse controls',controls['evaluations'])]
    output = dict(created_at=datetime.now(timezone.utc).isoformat(),campaign_id=run.get('campaign_id'),
                  interval_steps=INTERVAL,lag_steps=INTERVAL,estimator="cumulative_within_seed_pearson", averaging="arithmetic mean of defined seed coefficients",series=[])
    for name, rows in datasets:
        for mode in ['ODE','SDE']:
            output['series'].append(dict(cohort=name,mode=mode,
                **checkpoint_correlations([r for r in rows if r['mode']==mode])))
    plt.rcParams.update({'font.size':11,'axes.titlesize':12,'axes.labelsize':11,
                         'svg.fonttype':'none'})
    for mode in ['ODE','SDE']:
        fig, axes = plt.subplots(1,3,figsize=(13,3.4),sharey=True,layout='constrained')
        for ax,(name,_) in zip(axes,datasets):
            color=COLORS[name]
            series=next(s for s in output['series'] if s['cohort']==name and s['mode']==mode)
            for seed in series['seeds']:
                ax.plot([p['physical_step'] for p in seed['points']],
                        [np.nan if p['correlation'] is None else p['correlation'] for p in seed['points']],
                        color=color,alpha=.27,linewidth=1.2,marker='.',markersize=3,label=f"Seed {seed['seed']}")
            points=series['points']
            ax.plot([p['physical_step'] for p in points],
                    [np.nan if p['correlation'] is None else p['correlation'] for p in points],
                    color=color,marker='o',markersize=4,linewidth=2.7,label='Seed mean')
            ax.set(title=name,xlim=(0,102000),ylim=(-1.05,1.05),
                   yticks=[-1,-.5,0,.5,1],xlabel='Physical training steps (later checkpoint)')
            ax.set_xticks(range(0,100001,20000),['0','20k','40k','60k','80k','100k'])
            ax.grid(alpha=.2)
            ax.axhline(0,color='#667788',linewidth=.7)
            ax.spines[['top','right']].set_visible(False)
            ax.legend(fontsize=9,loc='best',ncol=2)
        axes[0].set_ylabel('Within-seed Pearson correlation')
        fig.suptitle(f'{mode} · Fixed lag: 10k steps · Cumulative consecutive checkpoint pairs',fontsize=12)
        target = page/f'lag-correlation-{mode.lower()}.svg'
        fig.savefig(target)
        plt.close(fig)
    (page/'lag-correlation.json').write_text(json.dumps(output,indent=2,allow_nan=False)+'\n')
    print(f'Created ODE/SDE correlation plots and {len(output["series"])} cohort/sampler series in {page}')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--page',type=Path,default=PAGE)
    build(parser.parse_args().page)
