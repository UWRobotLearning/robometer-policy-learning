"""Evaluate the frozen official pi05_libero checkpoint on every LIBERO task."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.libero.policy import Pi0Client, evaluate
from experiments.reward_audit.config import HERE, REPO, load_suite, materialize, task_map, write_json
from experiments.reward_audit.envs import EnvPool, prepare_libero_paths
from experiments.reward_audit.gpu_guard import assert_idle
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath

DATA_ROOT = Path('/data/rohan/Reward-Modeling')
ROOT = DATA_ROOT / 'pi05_libero_all_tasks_20261019'
CHECKPOINT = DATA_ROOT / 'pi0_checkpoints/pi05_libero'
PI0_PYTHON = DATA_ROOT / '.venv-pi0/bin/python'
SUITES = ('libero_spatial', 'libero_object', 'libero_goal', 'libero_10', 'libero_90')
EVAL_STATES = tuple(range(25, 50))
SEED = 100
ACTION_EXEC_LEN = 10


def tasks():
    names = task_map()
    return [(suite, index, names[suite][index]) for index in range(90)
            for suite in SUITES if index < len(names[suite])]


def task_dir(root, suite, index, name):
    return root / suite / f'{index:02d}_{name}'


def completed(folder):
    try:
        result = json.loads((folder / 'summary.json').read_text())
        return result['episodes'] == len(EVAL_STATES)
    except (OSError, KeyError, ValueError):
        return False


def run_task(policy, base, root, suite, index, name, gpu):
    folder = task_dir(root, suite, index, name)
    if completed(folder):
        print('SKIP', suite, index, flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    horizon = 600 if suite == 'libero_10' else 400
    cfg = materialize(base, {'name': name, 'suite': suite, 'task': name,
                             'reward': 'sparse', 'ood_claim': 'in_distribution' if suite != 'libero_90' else 'held_out_task',
                             'horizon': horizon})
    write_json(folder / 'config.json', {'suite': suite, 'task_id': index, 'task': name,
                                       'checkpoint': str(CHECKPOINT), 'checkpoint_config': 'pi05_libero',
                                       'training_family': suite != 'libero_90', 'seed': SEED,
                                       'action_exec_len': ACTION_EXEC_LEN, 'horizon': horizon,
                                       'eval_state_ids': EVAL_STATES, 'gpu': gpu})
    write_json(folder / 'status.json', {'status': 'running', 'started': datetime.now(timezone.utc).isoformat()})
    env = None
    try:
        policy.reset_seed(SEED)
        env = EnvPool(cfg, 1, rank_offset=100000)
        result = evaluate(policy, env, cfg, folder, ACTION_EXEC_LEN)
        write_json(folder / 'summary.json', result)
        write_json(folder / 'status.json', {'status': 'complete', 'successes': result['successes'],
                                            'episodes': result['episodes']})
        print(f'DONE {suite} {index:02d}: {result["successes"]}/{result["episodes"]} {name}', flush=True)
    except BaseException as error:
        write_json(folder / 'status.json', {'status': 'failed', 'error': repr(error)})
        print(f'FAILED {suite} {index:02d}: {error!r}', flush=True)
    finally:
        if env is not None:
            env.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, required=True, choices=(0, 1))
    parser.add_argument('--lanes', type=int, default=2)
    parser.add_argument('--wait-idle', action='store_true')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    if not 0 <= args.gpu < args.lanes:
        parser.error('GPU index must identify a lane')
    root = ROOT
    lane = root / f'gpu{args.gpu}'
    lane.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < 30 * 1024**3:
        raise RuntimeError('Less than 30 GiB free on /data')
    if not (CHECKPOINT / 'params').exists() or not (CHECKPOINT / 'assets').exists():
        raise RuntimeError(f'Missing checkpoint: {CHECKPOINT}')
    selected = tasks()[args.gpu::args.lanes]
    if args.limit:
        selected = selected[:args.limit]
    write_json(root / 'manifest.json', {'checkpoint': str(CHECKPOINT), 'config': 'pi05_libero',
                                        'dataset': 'physical-intelligence/libero',
                                        'training_families': list(SUITES[:4]), 'held_out_family': 'libero_90',
                                        'all_tasks': len(tasks()), 'episodes_per_task': len(EVAL_STATES),
                                        'eval_state_ids': EVAL_STATES, 'seed': SEED,
                                        'action_exec_len': ACTION_EXEC_LEN,
                                        'horizons': {'libero_10': 600, 'other': 400}})
    os.environ['PYTHONPATH'] = source_pythonpath()
    # multiprocessing.spawn copies the parent's sys.path rather than re-reading PYTHONPATH.
    sys.path.insert(0, str(REPO / 'LIBERO'))
    prepare_libero_paths(lane / 'libero_paths', REPO)
    base = load_suite(HERE / 'configs/stove.json')
    base['defaults'].update(seed=SEED, train_init_states=25,
                            eval_episodes=len(EVAL_STATES))
    if args.wait_idle:
        idle_count = 0
        while idle_count < 3:
            try:
                assert_idle([args.gpu])
                idle_count += 1
            except RuntimeError:
                idle_count = 0
            if idle_count < 3:
                print(f'Waiting for GPU {args.gpu} to be idle ({idle_count}/3)', flush=True)
                time.sleep(60)
    with reserve_gpus([args.gpu]):
        assert_idle([args.gpu])
        policy = Pi0Client(PI0_PYTHON.absolute(), CHECKPOINT.resolve(), SEED, args.gpu,
                           lane, config_name='pi05_libero')
        if policy.metadata['action_horizon'] != ACTION_EXEC_LEN:
            policy.close()
            raise RuntimeError(f'Unexpected action horizon: {policy.metadata}')
        try:
            for suite, index, name in selected:
                run_task(policy, base, root, suite, index, name, args.gpu)
        finally:
            policy.close()
    failures = [(suite, index) for suite, index, name in selected
                if not completed(task_dir(root, suite, index, name))]
    if failures:
        raise RuntimeError(f'Incomplete tasks: {failures}')

if __name__ == '__main__':
    main()
