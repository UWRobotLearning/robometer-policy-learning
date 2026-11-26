#!/usr/bin/env python3
"""Four additional task-2 SVM RL seeds, sequential per GPU, with live seed statistics."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time

from run_success_visitation import ROOT, SPEC_PATH, SPEC_DIR, baseline, progress, write_json, logging_data

DATA_PATH = ROOT / 'docs/experiment_atlas/svm-replicates.json'
# Two-sided 95% Student-t critical values, df=1..4 (this experiment has at most 5 seeds).
T95 = {1: 12.7062047364, 2: 4.3026527297, 3: 3.1824463053, 4: 2.7764451052}


def points(directory, seed):
    import csv
    flags = list(Path(directory).glob('logs/**/flags.json'))
    result = []
    if len(flags) != 1:
        return result
    for mode, name in [('ODE', 'eval.csv'), ('SDE', 'eval_sde.csv')]:
        path = flags[0].with_name(name)
        if not path.exists():
            continue
        with path.open() as stream:
            for row in csv.DictReader(stream):
                try:
                    step, rate = float(row['step']), float(row['success'])
                    if step >= 0 and 0 <= rate <= 1:
                        result.append(dict(seed=seed, mode=mode, step=step, success=rate))
                except (KeyError, ValueError, TypeError):
                    pass
    return result


def summarize(rows):
    """One observation per training seed/checkpoint; episodes are not extra seeds."""
    groups = defaultdict(dict)
    for row in rows:
        groups[(row['mode'], row['step'])][row['seed']] = row['success']
    result = []
    for (mode, step), samples in sorted(groups.items()):
        values = list(samples.values())
        n, mean = len(values), statistics.mean(values)
        half = T95[n - 1] * statistics.stdev(values) / n**.5 if n > 1 else None
        result.append(dict(mode=mode, step=step, mean=mean, n=n, seeds=sorted(samples),
            low=mean-half if half is not None else None, high=mean+half if half is not None else None))
    return result


def publish(jobs, source):
    svm = points(source['output_dir'], 100)
    runs = [dict(seed=100, gpu=0, status='completed', physical_steps=100000)]
    for job in jobs:
        svm.extend(points(job['output_dir'], job['seed']))
        runs.append({k: v for k, v in job.items() if k not in ('process', 'log', 'config')})
    telemetry = [logging_data(source['output_dir'],100)]+[logging_data(job['output_dir'],job['seed']) for job in jobs]
    sparse = baseline()
    write_json(DATA_PATH, dict(campaign_id=source['output_dir'],updated_at=datetime.now(timezone.utc).isoformat(), runs=runs,
        evaluations=svm, training_episodes=[row for t in telemetry for row in t['training_episodes']],
        eval_artifacts=[row for t in telemetry for row in t['eval_artifacts']], summaries={'SVM': summarize(svm), 'Historical sparse': summarize(sparse)},
        interval='95% Student-t across available independent RL training seeds at each checkpoint; no interval for n=1.',
        cohort_note='All five SVM seeds share the selected 30-demo BC checkpoint. Historical sparse uses a different 50-demo BC start. Available seed counts can vary with checkpoint and run progress.'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--launch', action='store_true')
    parser.add_argument('--python', default='/home/rohan/Robotics/Real-World-RL/OGPO_public/.venv/bin/python')
    args = parser.parse_args()
    source = json.loads(SPEC_PATH.read_text())
    expected = source['bc_source']['sha256']
    checkpoint = Path(source['overrides']['checkpoint.restore_bc_path'])
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == expected, 'Selected BC checkpoint changed'
    assert source['overrides']['checkpoint.reseed_restored_rng'] is True
    jobs = []
    for seed, gpu in [(101, 0), (102, 1), (103, 0), (104, 1)]:
        cfg = json.loads(json.dumps(source))
        out = Path(source['output_dir']).with_name(Path(source['output_dir']).name.replace('seed100', f'seed{seed}'))
        if list(out.glob('logs/**/flags.json')) or (out/'launch.json').exists():
            raise SystemExit(f'Refusing to overwrite existing seed {seed}: {out}')
        cfg.update(seed=seed, policy_gpu=gpu, output_dir=str(out), comparison_id='task02_svm_bc30demos_selected_100k')
        cfg['overrides']['experiment.seed'] = seed
        cfg['overrides']['experiment.run_group'] = 'task02_svm_bc30demos_100k'
        spec = SPEC_DIR / f'libero90_task02_svm_100k_seed{seed}.json'
        write_json(spec, cfg)
        jobs.append(dict(seed=seed, gpu=gpu, status='queued', physical_steps=0,
                         output_dir=str(out), spec=str(spec), config=cfg))
    publish(jobs, source)
    if not args.launch:
        print('Prepared seeds101–104. GPU0:101 then103; GPU1:102 then104. No BC retraining.')
        return 0
    try:
        while any(job['status'] in ('queued', 'running') for job in jobs):
            for job in jobs:
                if job['status'] == 'running':
                    code = job['process'].poll()
                    job['physical_steps'] = progress(job['output_dir'], 'Online RL', 100000)
                    if code is not None:
                        job.update(status='completed' if code == 0 else 'failed', returncode=code,
                                   finished=datetime.now(timezone.utc).isoformat())
                        job['log'].close()
                        print(f"Seed{job['seed']} {job['status']} (exit {code})", flush=True)
            busy = {job['gpu'] for job in jobs if job['status'] == 'running'}
            for gpu in (0, 1):
                if gpu in busy:
                    continue
                job = next((j for j in jobs if j['gpu'] == gpu and j['status'] == 'queued'), None)
                if job is None:
                    continue
                out = Path(job['output_dir']); out.mkdir(parents=True, exist_ok=True)
                command = [args.python, '-u', str(ROOT/'OGPO/scripts/libero.py'), 'run', '--spec', job['spec'], '--gpu', str(gpu)]
                job['log'] = (out/'console.log').open('a')
                job['process'] = subprocess.Popen(command, cwd=ROOT/'OGPO', stdout=job['log'], stderr=subprocess.STDOUT)
                job.update(status='running', pid=job['process'].pid, started=datetime.now(timezone.utc).isoformat())
                write_json(out/'launch.json', dict(pid=job['pid'], command=command, started=time.time(), spec=job['config']))
                print(f"Seed{job['seed']} started on GPU{gpu}, PID {job['pid']}", flush=True)
            publish(jobs, source)
            if any(job['status'] in ('queued', 'running') for job in jobs):
                time.sleep(15)
    except KeyboardInterrupt:
        for job in jobs:
            if job['status'] == 'running':
                job['process'].terminate(); job['process'].wait(); job['log'].close()
                job['status'] = 'stopped'
            elif job['status'] == 'queued':
                job['status'] = 'cancelled'
        publish(jobs, source)
        return 130
    return int(any(job['status'] != 'completed' for job in jobs))


if __name__ == '__main__':
    sys.exit(main())
