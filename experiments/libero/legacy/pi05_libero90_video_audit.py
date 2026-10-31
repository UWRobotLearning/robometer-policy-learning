"""Record 25 frozen π0.5 LIBERO-90 rollouts and audit selected videos with Robometer."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from huggingface_hub import hf_hub_download

from experiments.libero.legacy.libero10_600_video_audit import choose_video_states
from experiments.libero.legacy.task38_video_audit import FPS, SCORE_EVERY, VIDEO_EVERY, collect, scoring_indices
from experiments.reward_audit.config import REPO, task_map, write_json
from experiments.reward_audit.envs import prepare_libero_paths
from experiments.reward_audit.gpu_guard import assert_idle
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath
from experiments.reward_audit.metrics import prefix_frames, wilson
from experiments.reward_audit.rpc import RewardClient

DATA_ROOT = Path('/data/rohan/Reward-Modeling')
ROOT = DATA_ROOT / 'pi05_libero90_video_audit_20261020'
TASK_IDS = (2, 28)
EVAL_STATES = tuple(range(25, 50))
HORIZON = 600
SOURCE_REPO = 'yifengzhu-hf/LIBERO-datasets'
SOURCE_REVISION = 'f13aa24a3da8c43c7225569f28c562979fa0e35a'
SOURCE_CONFIG = (DATA_ROOT / 'pi0_libero90_task38_dsrl_3seeds_20261012/seed100/'
                 'robometer_seed100/config.json')
CHECKPOINT = DATA_ROOT / 'pi0_checkpoints/pi05_libero'
POLICY_PYTHON = DATA_ROOT / '.venv-pi0/bin/python'


def task_name(task_id: int) -> str:
    return task_map()['libero_90'][task_id]


def task_root(task_id: int) -> Path:
    return ROOT / f'task{task_id:02d}'


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def status(out: Path, phase: str, error: str | None = None) -> None:
    records = read_json(out / 'policy/episodes.json', [])
    demos = read_json(out / 'demos/manifest.json', {}).get('demonstrations', [])
    selected = read_json(out / 'policy/selection.json', {}).get('selected_state_ids', [])
    score_count = sum((out / 'policy/scores' / f'episode_{state}.json').exists() for state in selected)
    score_count += sum((out / 'demos/scores' / f'demo_{row["demo_id"]}.json').exists() for row in demos)
    write_json(out / 'status.json', {'status': phase, 'task_id': int(out.name[-2:]),
               'episodes_collected': len(records), 'episodes_target': 25,
               'successes': sum(bool(row['success']) for row in records),
               'demonstrations': len(demos), 'selected_videos': len(selected),
               'videos_scored': score_count, 'videos_target': 8,
               **({'error': error} if error else {})})


def config(task_id: int, gpu: int) -> dict:
    cfg = json.loads(SOURCE_CONFIG.read_text())
    cfg.update(suite='libero_90', task_id=task_id, task=task_name(task_id),
               name=f'pi05_libero90_task{task_id:02d}_video_audit',
               ood_claim='held_out_task', seed=100, train_init_states=25,
               eval_episodes=25, horizon=HORIZON, policy_gpu=gpu, reward_gpu=gpu,
               pi0_checkpoint=str(CHECKPOINT), pi0_python=str(POLICY_PYTHON),
               policy_config_name='pi05_libero')
    cfg['dsrl']['action_exec_len'] = 10
    return cfg


def source_demo(task_id: int) -> tuple[Path, str]:
    filename = f'libero_90/{task_name(task_id)}_demo.hdf5'
    path = Path(hf_hub_download(repo_id=SOURCE_REPO, repo_type='dataset',
                                revision=SOURCE_REVISION, filename=filename,
                                local_dir=ROOT / 'dataset_source'))
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    return path, digest.hexdigest()


def render_demos(source_path: Path, digest: str, out: Path) -> list[dict]:
    folder = out / 'demos'
    for name in ('videos', 'frames', 'scores'):
        (folder / name).mkdir(parents=True, exist_ok=True)
    records = []
    with h5py.File(source_path) as source:
        prompt = json.loads(source['data'].attrs['problem_info'])['language_instruction']
        for demo_id in range(4):
            demo = source['data'][f'demo_{demo_id}']
            length = len(demo['actions'])
            steps = list(range(0, length, VIDEO_EVERY))
            if steps[-1] != length - 1:
                steps.append(length - 1)
            video = folder / 'videos' / f'demo_{demo_id}.mp4'
            frames_file = folder / 'frames' / f'demo_{demo_id}.npz'
            if not video.exists() or not frames_file.exists():
                # HDF5 frames use the OpenGL camera orientation; rotate like the simulator wrapper.
                frames = np.stack([demo['obs']['agentview_rgb'][step][::-1, ::-1].copy()
                                   for step in steps])
                iio.imwrite(video, frames, fps=FPS)
                np.savez_compressed(frames_file, frames=frames,
                                    steps=np.asarray(steps, dtype=np.int32))
            events = [{'step': int(step), 'reward': float(demo['rewards'][step])}
                      for step in range(length) if demo['rewards'][step] != 0]
            records.append({'demo_id': demo_id, 'length': length,
                            'success': bool(demo['dones'][-1]),
                            'reward_events': events, 'prompt': prompt})
            print(f'Demo {demo_id}: {length} steps, success={records[-1]["success"]}', flush=True)
    write_json(folder / 'manifest.json', {'source': SOURCE_REPO, 'revision': SOURCE_REVISION,
               'source_file': source_path.name, 'sha256': digest, 'prompt': prompt,
               'fps': FPS, 'video_every_physical_steps': VIDEO_EVERY,
               'demonstrations': records})
    status(out, 'demos_ready')
    return records


def score_videos(cfg: dict, out: Path, gpu: int, records: list[dict], demos: list[dict], selection: dict) -> None:
    jobs = [('policy', f'episode_{state}', next(row['prompt'] for row in records
                                                if row['init_state_id'] == state))
            for state in selection['selected_state_ids']]
    jobs += [('demos', f'demo_{row["demo_id"]}', row['prompt']) for row in demos]
    pending = [(kind, prefix, prompt) for kind, prefix, prompt in jobs
               if not (out / kind / 'scores' / f'{prefix}.json').exists()]
    if not pending:
        return
    reward = RewardClient(cfg, out, gpu)
    try:
        for kind, prefix, prompt in pending:
            with np.load(out / kind / 'frames' / f'{prefix}.npz') as source:
                frames, steps = source['frames'], source['steps']
                indices = scoring_indices(steps)
                prefixes = [prefix_frames(frames[:index + 1], cfg['max_frames']) for index in indices]
                predictions = reward.predict(prefixes, prompt)
                series = [{'step': int(steps[index]), 'video_time': index / FPS,
                           'progress': float(prediction[0]),
                           'success_probability': float(prediction[1])}
                          for index, prediction in zip(indices, predictions, strict=True)]
            write_json(out / kind / 'scores' / f'{prefix}.json', series)
            if kind == 'policy':
                state = int(prefix.removeprefix('episode_'))
                row = next(row for row in records if row['init_state_id'] == state)
                row['final_progress'] = series[-1]['progress']
                row['final_success_probability'] = series[-1]['success_probability']
                write_json(out / 'policy/episodes.json', records)
            status(out, 'scoring')
            print(f'Scored {kind} {prefix}: progress={series[-1]["progress"]:.3f}, '
                  f'success={series[-1]["success_probability"]:.3f}', flush=True)
    finally:
        reward.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task-id', type=int, choices=TASK_IDS, required=True)
    parser.add_argument('--gpu', type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    out = task_root(args.task_id)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config(args.task_id, args.gpu)
    if not (CHECKPOINT / 'params').exists() or not (CHECKPOINT / 'assets').exists():
        raise RuntimeError(f'Missing pi05 checkpoint: {CHECKPOINT}')
    old_cfg = read_json(out / 'config.json', None)
    if old_cfg is not None and old_cfg != cfg:
        raise RuntimeError(f'Existing audit config differs: {out / "config.json"}')
    write_json(out / 'config.json', cfg)
    policy_out = out / 'policy'
    for name in ('videos', 'frames', 'scores'):
        (policy_out / name).mkdir(parents=True, exist_ok=True)
    write_json(policy_out / 'config.json', cfg)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_GL='egl',
                      MUJOCO_EGL_DEVICE_ID=str(args.gpu), EGL_DEVICE_ID=str(args.gpu),
                      XLA_PYTHON_CLIENT_PREALLOCATE='false', WANDB_MODE='disabled',
                      WANDB_DISABLED='true', PYTHONPATH=source_pythonpath(),
                      TMPDIR='/data/rohan/rbm-tmp')
    prepare_libero_paths(out / 'libero_paths', REPO)
    sys.path.insert(0, str(REPO / 'LIBERO'))
    try:
        status(out, 'downloading_demos')
        source_path, digest = source_demo(args.task_id)
        demos = render_demos(source_path, digest, out)
        with reserve_gpus([args.gpu]):
            assert_idle([args.gpu])
            status(out, 'collecting_policy')
            records = collect(cfg, policy_out, args.gpu, EVAL_STATES)
            if {row['init_state_id'] for row in records} != set(EVAL_STATES) or len(records) != 25:
                raise RuntimeError('Expected exactly one rollout for each of 25 official initial states')
            for state in EVAL_STATES:
                if not (policy_out / 'videos' / f'episode_{state}.mp4').exists():
                    raise RuntimeError(f'Missing recorded video for state {state}')
            selection = choose_video_states(records, 20261020 + args.task_id)
            write_json(policy_out / 'selection.json', selection)
            status(out, 'scoring')
            score_videos(cfg, out, args.gpu, records, demos, selection)
        successes = sum(row['success'] for row in records)
        write_json(out / 'summary.json', {'task_id': args.task_id, 'task': task_name(args.task_id),
                   'suite': 'libero_90', 'seed': cfg['seed'], 'horizon': HORIZON,
                   'policy_checkpoint': cfg['pi0_checkpoint'],
                   'policy_config_name': cfg['policy_config_name'],
                   'reward_checkpoint': cfg['reward_checkpoint'],
                   'episodes': 25, 'successes': successes,
                   'success_rate': successes / 25, 'wilson_ci95': wilson(successes, 25),
                   'selection': selection, 'demonstration_source': SOURCE_REPO,
                   'demonstration_revision': SOURCE_REVISION,
                   'demonstration_sha256': digest,
                   'video_every_physical_steps': VIDEO_EVERY,
                   'robometer_every_physical_steps': SCORE_EVERY})
        status(out, 'complete')
        print(f'Complete task {args.task_id}: {successes}/25 successes', flush=True)
    except BaseException as error:
        status(out, 'failed', repr(error))
        raise

if __name__ == '__main__':
    main()
