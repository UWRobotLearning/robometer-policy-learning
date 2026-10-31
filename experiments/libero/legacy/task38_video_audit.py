"""Record ten frozen pi0 LIBERO-90 rollouts and score them with Robometer."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from experiments.libero.policy import Pi0Client
from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.envs import EnvPool, prepare_libero_paths
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath
from experiments.reward_audit.metrics import prefix_frames
from experiments.reward_audit.rpc import RewardClient


DATA_ROOT = Path("/data/rohan/Reward-Modeling")
AUDIT_DATE = "20261014"
STATE_IDS = tuple(range(25, 35))
VIDEO_EVERY = 4
SCORE_EVERY = 20
FPS = 10


def collect(cfg: dict, out: Path, gpu: int,
            state_ids: tuple[int, ...] = STATE_IDS) -> list[dict]:
    records_path = out / "episodes.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    existing = {record["init_state_id"] for record in records}
    if set(state_ids) <= existing:
        return records
    policy = env = None
    try:
        policy = Pi0Client(Path(cfg["pi0_python"]), Path(cfg["pi0_checkpoint"]), cfg["seed"], gpu, out,
                           config_name=cfg.get("policy_config_name", "pi0_libero"))
        env = EnvPool(cfg, 1, rank_offset=100000)
        prompt = env.metadata[0]["language"]
        for state_id in state_ids:
            if state_id in existing:
                continue
            policy.reset_seed(cfg["seed"] * 100000 + state_id)
            observation = env.reset({0: state_id})[0]["obs"]
            frames = [observation["agentview_image"].copy()]
            frame_steps = [0]
            chunks = 0
            while True:
                actions = policy.infer(observation, prompt)
                chunks += 1
                finished = False
                for action in actions[:cfg["dsrl"]["action_exec_len"]]:
                    result = env.step([action])[0]
                    observation = result["obs"]
                    step = result["elapsed"]
                    finished = result["terminated"] or result["truncated"]
                    if step % VIDEO_EVERY == 0 or finished:
                        frames.append(observation["agentview_image"].copy())
                        frame_steps.append(step)
                    if finished:
                        break
                if finished:
                    break
            video_name = f"episode_{state_id}.mp4"
            iio.imwrite(out / "videos" / video_name, np.stack(frames), fps=FPS)
            np.savez_compressed(out / "frames" / f"episode_{state_id}.npz",
                                frames=np.stack(frames), steps=np.asarray(frame_steps, dtype=np.int32))
            record = {"init_state_id": state_id, "success": bool(result["terminated"]),
                      "length": int(result["elapsed"]), "chunks": chunks,
                      "video": video_name, "video_frames": len(frames), "prompt": prompt}
            records.append(record)
            write_json(records_path, records)
            print(f'State {state_id}: {"success" if record["success"] else "failure"}, '
                  f'{record["length"]} steps, {len(frames)} video frames', flush=True)
        return records
    finally:
        if env is not None:
            env.close()
        if policy is not None:
            policy.close()


def scoring_indices(steps: np.ndarray) -> list[int]:
    return [index for index, step in enumerate(steps)
            if int(step) == 0 or int(step) % SCORE_EVERY == 0 or index == len(steps) - 1]


def score(cfg: dict, out: Path, gpu: int, records: list[dict],
          all_records: list[dict] | None = None) -> None:
    all_records = records if all_records is None else all_records
    reward = RewardClient(cfg, out, gpu)
    try:
        for record in records:
            state_id = record["init_state_id"]
            path = out / "scores" / f"episode_{state_id}.json"
            if path.exists():
                continue
            with np.load(out / "frames" / f"episode_{state_id}.npz") as source:
                frames, steps = source["frames"], source["steps"]
                indices = scoring_indices(steps)
                prefixes = [prefix_frames(frames[:index + 1], cfg["max_frames"]) for index in indices]
                predictions = reward.predict(prefixes, record["prompt"])
                series = [{"step": int(steps[index]), "video_time": index / FPS,
                           "progress": float(prediction[0]), "success_probability": float(prediction[1])}
                          for index, prediction in zip(indices, predictions, strict=True)]
            write_json(path, series)
            record["final_progress"] = series[-1]["progress"]
            record["final_success_probability"] = series[-1]["success_probability"]
            write_json(out / "episodes.json", all_records)
            print(f'Scored state {state_id}: final progress {series[-1]["progress"]:.3f}, '
                  f'success estimate {series[-1]["success_probability"]:.3f}', flush=True)
    finally:
        reward.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=(38, 55), default=38)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    out = (args.out or DATA_ROOT / f"pi0_task{args.task_id}_video_audit_{AUDIT_DATE}").resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("videos", "frames", "scores"):
        (out / name).mkdir(exist_ok=True)
    source_config = (DATA_ROOT / f"pi0_libero90_task{args.task_id}_dsrl_3seeds_20261012" /
                     "seed100/robometer_seed100/config.json")
    cfg = json.loads(source_config.read_text())
    if cfg["task_id"] != args.task_id:
        raise RuntimeError(f"Unexpected task in {source_config}")
    cfg.update(policy_gpu=args.gpu, reward_gpu=args.gpu, eval_episodes=len(STATE_IDS))
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise RuntimeError(f"Existing audit config differs: {config_path}")
    write_json(config_path, cfg)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_GL="egl",
                      MUJOCO_EGL_DEVICE_ID=str(args.gpu), EGL_DEVICE_ID=str(args.gpu),
                      XLA_PYTHON_CLIENT_PREALLOCATE="false", WANDB_MODE="disabled",
                      WANDB_DISABLED="true", PYTHONPATH=source_pythonpath(),
                      TMPDIR="/data/rohan/rbm-tmp")
    prepare_libero_paths(out / "libero_paths", REPO)
    # multiprocessing.spawn copies sys.path from this process, not a PYTHONPATH
    # assignment made after interpreter startup. LIBERO's outer package is a namespace.
    sys.path.insert(0, str(REPO / "LIBERO"))
    with reserve_gpus([args.gpu]):
        write_json(out / "status.json", {"status": "collecting", "episodes_target": len(STATE_IDS)})
        records = collect(cfg, out, args.gpu)
        if len(records) != len(STATE_IDS):
            raise RuntimeError(f"Expected {len(STATE_IDS)} episodes, got {len(records)}")
        write_json(out / "status.json", {"status": "scoring", "episodes_collected": len(records)})
        score(cfg, out, args.gpu, records)
    write_json(out / "summary.json", {"task_id": args.task_id, "seed": cfg["seed"],
                                      "successes": sum(record["success"] for record in records),
                                      "episodes": len(records), "records": records,
                                      "policy_checkpoint": cfg["pi0_checkpoint"],
                                      "reward_checkpoint": cfg["reward_checkpoint"],
                                      "video_every_physical_steps": VIDEO_EVERY,
                                      "robometer_every_physical_steps": SCORE_EVERY,
                                      "fps": FPS})
    write_json(out / "status.json", {"status": "complete", "episodes": len(records)})
    print('Complete', flush=True)


if __name__ == "__main__":
    main()
