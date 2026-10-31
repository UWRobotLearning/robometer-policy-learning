"""Replay a final task-38/55 DSRL actor on held-out states and record its reward audit."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import imageio.v3 as iio
import numpy as np

from experiments.libero.policy import Pi0Client
from experiments.libero.legacy.task38_video_audit import FPS, SCORE_EVERY, STATE_IDS, VIDEO_EVERY, score
from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.envs import EnvPool, prepare_libero_paths
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath
from experiments.reward_audit.sac import Dino, Replay, actor_actions, load_actor, make_sac


DATA_ROOT = Path("/data/rohan/Reward-Modeling")


def collect(cfg: dict, out: Path, gpu: int, actor_checkpoint: Path,
            state_ids: tuple[int, ...] = STATE_IDS) -> list[dict]:
    import torch

    records_path = out / "episodes.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    existing = {record["init_state_id"] for record in records}
    if set(state_ids) <= existing:
        return records
    policy = env = None
    try:
        policy = Pi0Client(Path(cfg["pi0_python"]), Path(cfg["pi0_checkpoint"]), cfg["seed"], gpu, out,
                           config_name=cfg.get("policy_config_name", "pi0_libero"))
        device = torch.device("cuda:0")
        encoder = Dino(cfg["dino_checkpoint"], device)
        replay = Replay(1, {"dino_embedding": encoder.dimension,
                            "observation/state": 8, "language": 384},
                        cfg["seed"], action_dim=cfg["dsrl"]["noise_dim"])
        algorithm = make_sac(cfg, replay, device, action_dim=cfg["dsrl"]["noise_dim"])
        load_actor(algorithm.actor, actor_checkpoint, cfg, device)
        algorithm.actor.eval()
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
                features = encoder.encode([observation])
                steering = actor_actions(algorithm.actor, features, device, deterministic=True)[0]
                actions = policy.infer(observation, prompt, noise=steering,
                                       noise_mode=cfg["dsrl"]["noise_mode"])
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
        if "algorithm" in locals():
            del algorithm
        if "encoder" in locals():
            del encoder
        torch.cuda.empty_cache()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=(38, 55), required=True)
    parser.add_argument("--arm", choices=("sparse", "robometer"), required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    run = (DATA_ROOT / f"pi0_libero90_task{args.task_id}_dsrl_3seeds_20261012" /
           "seed100" / f"{args.arm}_seed100")
    actor_checkpoint = run / "checkpoints/final.pt"
    if not actor_checkpoint.is_file() or json.loads((run / "status.json").read_text())["status"] != "complete":
        raise RuntimeError(f"Final trained run is unavailable: {run}")
    out = (DATA_ROOT / f"pi0_task{args.task_id}_video_audit_20261014" /
           "final_policies" / f"{args.arm}_seed100")
    out.mkdir(parents=True, exist_ok=True)
    for name in ("videos", "frames", "scores"):
        (out / name).mkdir(exist_ok=True)
    cfg = json.loads((run / "config.json").read_text())
    if cfg["task_id"] != args.task_id or cfg["reward"] != args.arm or cfg["dsrl"]["noise_mode"] != "direct":
        raise RuntimeError(f"Unexpected DSRL run configuration: {run}")
    cfg.update(policy_gpu=args.gpu, reward_gpu=args.gpu, eval_episodes=len(STATE_IDS))
    write_json(out / "config.json", cfg)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_GL="egl",
                      MUJOCO_EGL_DEVICE_ID=str(args.gpu), EGL_DEVICE_ID=str(args.gpu),
                      XLA_PYTHON_CLIENT_PREALLOCATE="false", WANDB_MODE="disabled",
                      WANDB_DISABLED="true", PYTHONPATH=source_pythonpath(),
                      TMPDIR="/data/rohan/rbm-tmp")
    prepare_libero_paths(out / "libero_paths", REPO)
    sys.path.insert(0, str(REPO / "LIBERO"))
    with reserve_gpus([args.gpu]):
        write_json(out / "status.json", {"status": "collecting", "episodes_target": len(STATE_IDS)})
        records = collect(cfg, out, args.gpu, actor_checkpoint)
        write_json(out / "status.json", {"status": "scoring", "episodes_collected": len(records)})
        score(cfg, out, args.gpu, records)
    write_json(out / "summary.json", {"task_id": args.task_id, "arm": args.arm,
                                      "seed": cfg["seed"], "actor_checkpoint": str(actor_checkpoint),
                                      "checkpoint_env_steps": cfg["steps"],
                                      "successes": sum(record["success"] for record in records),
                                      "episodes": len(records), "records": records,
                                      "video_every_physical_steps": VIDEO_EVERY,
                                      "robometer_every_physical_steps": SCORE_EVERY,
                                      "fps": FPS})
    write_json(out / "status.json", {"status": "complete", "episodes": len(records)})
    print("Complete", flush=True)


if __name__ == "__main__":
    main()
