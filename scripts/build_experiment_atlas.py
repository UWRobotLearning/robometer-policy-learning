#!/usr/bin/env python3
"""Build the portable experiment catalog from checked-in evidence (stdlib only)."""
import argparse
import csv
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs/experiment_atlas/data.json'


def read(path, default=None):
    return json.loads(path.read_text()) if path.exists() else default


def url(path):
    return '/' + path.relative_to(ROOT).as_posix()


def build(snapshot):
    base = ROOT / snapshot
    index = read(base / 'index.json')
    runs, points = [], defaultdict(list)
    with (base / 'evaluation_points.csv').open() as stream:
        for row in csv.DictReader(stream):
            points[row['run']].append(dict(x=float(row['physical_steps_per_env']), y=float(row['success_rate']),
                episodes=int(row['eval_episodes']), successes=int(row['successes']), mode='recorded'))
    for item in read(base / 'runs.json'):
        config_path = base / 'data' / item['config']
        config = read(config_path, {})
        status_path = config_path.with_name('status.json')
        status = read(status_path, {})
        parts = item['run'].split('/')
        # Preserve the exported experiment identity, never pool by task name.
        group = '/'.join(parts[:2]) if parts[0] == 'experiments' else parts[0]
        runs.append(dict(id='sac:' + item['run'], name=item['run'], group=group,
            family='SAC' if item['base_model'] == 'scratch' else 'DSRL', suite=item['suite'],
            task=item['task'], task_id=item['task_id'], reward=item['reward'], seed=item['seed'],
            model=item['base_model'], envs=item['num_envs'], horizon=config.get('horizon'),
            state=status.get('status', 'unknown'), config=url(config_path),
            status=url(status_path) if status_path.exists() else None,
            source=url(base / 'evaluation_points.csv'), captured=index['exported_at'],
            points=sorted(points[item['run']], key=lambda p: p['x'])))
    report = ROOT / 'docs/reports/ogpo_success_summary_20261031'
    evidence = read(report / 'evidence.json', {'runs': []})
    ogpo_points = defaultdict(list)
    with (report / 'evaluation_points.csv').open() as stream:
        for row in csv.DictReader(stream):
            ogpo_points[row['run']].append(dict(x=float(row['physical_steps_per_env']), y=float(row['success_rate']),
                episodes=int(row['episodes']), successes=int(row['successes']), mode=row['mode']))
    for item in evidence['runs']:
        spec = item['spec']
        config = item.get('config') or {}
        group = config.get('experiment', {}).get('run_group') or re.sub(r'_(robometer|sparse).*', '', item['name'])
        runs.append(dict(id='ogpo:' + item['name'], name=item['name'], group='OGPO / ' + group,
            family='OGPO LoRA' if 'native_lora' in item['name'] else 'OGPO MLP', suite=spec.get('suite', ''),
            task=spec.get('prompt', ''), task_id=spec.get('task_id'), reward=spec.get('reward', 'unspecified'),
            seed=spec.get('seed'), model='pi0.5', envs=item.get('num_envs', 1), horizon=spec.get('horizon'),
            state=item['state'], config=url(report / 'evidence.json'), status=None,
            source=url(report / 'evaluation_points.csv'), captured=evidence['captured_at'],
            points=sorted(ogpo_points[item['name']], key=lambda p: (p['mode'], p['x']))))
    # Subdivide historical containers that hold multiple tasks or collection protocols.
    for run in runs:
        run['group'] += (f" / {run['suite']} task {run['task_id']}"
                         f" / {run['model']} / {run['envs']} env / H{run['horizon']}")
    frozen = []
    for item in read(base / 'frozen_evaluations.json'):
        config, summary = item.get('config') or {}, item.get('summary') or {}
        frozen.append(dict(path=item['path'], source=url(base / 'data' / item['path']),
            suite=config.get('suite', ''), task=config.get('task', ''), task_id=config.get('task_id'),
            model=config.get('checkpoint_config', config.get('base_model', config.get('checkpoint', 'unspecified'))),
            seed=config.get('seed'), horizon=config.get('horizon'), rate=summary.get('success_rate'),
            successes=summary.get('successes'), episodes=summary.get('episodes'), ci=summary.get('wilson_ci95')))
    # Full file inventory covers BC searches, smoke runs, diagnostics, and audits beyond the normalized tables.
    inventory = [dict(path=f['path'], bytes=f['bytes'], source=url(base / 'data' / f['path'])) for f in index['files']]
    plans = [dict(path=p.relative_to(ROOT).as_posix(), source=url(p), spec=read(p))
        for p in sorted((ROOT / 'experiments/libero/specs').rglob('*.json'))]
    data = dict(built_at=datetime.now(timezone.utc).isoformat(), snapshot=snapshot,
        captured=index['exported_at'], ogpo_captured=evidence.get('captured_at'), warnings=index.get('warnings', []),
        runs=runs, frozen=frozen, inventory=inventory, plans=plans,
        notes=read(ROOT / 'docs/experiment_atlas/notes.json', []))
    OUT.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False, separators=(',', ':')) + '\n')
    print(f'Built {OUT.relative_to(ROOT)}: {len(runs)} runs, {sum(len(r["points"]) for r in runs)} points, '
          f'{len(frozen)} evaluations, {len(inventory)} source files, {len(plans)} spec files')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--snapshot', default='results/libero_20261031', help='Repository-relative results snapshot')
    build(parser.parse_args().snapshot)
