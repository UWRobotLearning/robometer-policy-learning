from __future__ import annotations

import argparse
import json
import os
import random
import signal
import traceback
from pathlib import Path

import numpy as np

from .config import REPO, write_json
from .envs import EnvPool, prepare_libero_paths
from .local_log import LocalLog
from .metrics import alignment, prefix_frames, training_reward, wilson
from .rpc import RewardClient
from .sac import Dino, Replay, actor_actions, load_actor, make_sac, save_checkpoint


def evaluate(cfg, actor, encoder, env, reward, language, log, step, device, out):
    """Evaluate disjoint fixed init states using simulator success only."""
    import torch

    # Evaluation must not perturb exploration or SAC's torch/NumPy RNG streams.
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    episodes = []
    try:
        for episode in range(cfg["eval_episodes"]):
            state_id = cfg["train_init_states"] + episode % cfg.get("eval_unique_states", cfg["eval_episodes"])
            reset = env.reset({0: state_id})[0]
            obs = reset["obs"]
            history = [obs[cfg["reward_camera"]]]
            frames, steps = [obs["agentview_image"]], 0
            while True:
                features = encoder.encode([obs])
                actions = actor_actions(actor, features, device, deterministic=True)
                result = env.step(actions)[0]
                obs = result["obs"]
                history.append(obs[cfg["reward_camera"]])
                steps += 1
                if episode < cfg["eval_videos"] and steps % 4 == 0:
                    frames.append(obs["agentview_image"])
                if result["terminated"] or result["truncated"]:
                    break
            record = {
                "phase": "eval",
                "env_steps": step,
                "episode": episode,
                "init_state_id": state_id,
                "success": bool(result["terminated"]),
                "length": steps,
            }
            if reward is not None:
                prediction = reward.predict([prefix_frames(history, cfg["max_frames"])], language)[0]
                record.update(final_progress=float(prediction[0]), final_success_probability=float(prediction[1]))
            log.episode(record)
            episodes.append(record)
            if episode < cfg["eval_videos"]:
                import imageio.v3 as iio

                video_dir = Path(out) / "videos"
                video_dir.mkdir(exist_ok=True)
                iio.imwrite(video_dir / f"step_{step}_episode_{episode}.mp4", np.stack(frames), fps=10)
        successes = sum(e["success"] for e in episodes)
        ci = wilson(successes, len(episodes))
        log.evaluation(step, successes, len(episodes), ci)
        metrics = {
            "success_rate": successes / len(episodes),
            "successes": successes,
            "episodes": len(episodes),
            "ci_low": ci[0],
            "ci_high": ci[1],
        }
        if reward is not None:
            metrics["reward_alignment"] = alignment(episodes)
        log.log(metrics, step, "eval")
        print(f"Evaluation at {step} steps: {successes}/{len(episodes)} simulator successes", flush=True)
        return metrics
    finally:
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


def run(cfg, out, reward_gpu):
    import torch

    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    prepare_libero_paths(Path(out) / "libero_paths", REPO)
    torch.set_num_threads(cfg["torch_threads"])
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    if not torch.cuda.is_available():
        raise RuntimeError("Real experiments require a GPU; CPU unit tests are separate")
    device = torch.device("cuda:0")
    log = LocalLog(out)
    train_env = eval_env = reward = None
    write_json(Path(out) / "status.json", {"status": "running"})
    try:
        if cfg["reward"] == "robometer" or cfg["audit_sparse_rewards"]:
            reward = RewardClient(cfg, out, reward_gpu)
        encoder = Dino(cfg["dino_checkpoint"], device)
        train_env = EnvPool(cfg, cfg["num_envs"])
        eval_env = EnvPool(cfg, 1, rank_offset=100000)
        language = train_env.metadata[0]["language"]
        write_json(Path(out) / "environment.json", train_env.metadata[0])
        write_json(Path(out) / "encoder.json", {"checkpoint": cfg["dino_checkpoint"], "revision": encoder.revision})
        replay = Replay(
            cfg["buffer_capacity"],
            {"dino_embedding": encoder.dimension, "observation/state": 8, "language": 384},
            cfg["seed"],
        )
        algorithm = make_sac(cfg, replay, device)
        if cfg["initial_actor"]:
            load_actor(algorithm.actor, cfg["initial_actor"], cfg, device)
        reset = train_env.reset()
        obs = [reset[i]["obs"] for i in range(cfg["num_envs"])]
        histories = [[o[cfg["reward_camera"]]] for o in obs]
        previous_progress = np.zeros(cfg["num_envs"], dtype=np.float32)
        if reward is not None:
            previous_progress = reward.predict([prefix_frames(h, cfg["max_frames"]) for h in histories], language)[:, 0]
        features = encoder.encode(obs)
        episode_return = np.zeros(cfg["num_envs"])
        episode_id = np.zeros(cfg["num_envs"], dtype=int)
        # Always evaluate before any update, including when loading a policy.
        evaluate(cfg, algorithm.actor, encoder, eval_env, reward, language, log, 0, device, out)
        env_steps, last_eval, last_save, last_log = 0, 0, 0, 0
        final_metrics = None
        while env_steps < cfg["steps"]:
            if env_steps < cfg["sac"]["learning_starts"] and not cfg["initial_actor"]:
                actions = np.random.uniform(-1, 1, (cfg["num_envs"], 7)).astype(np.float32)
            else:
                actions = actor_actions(algorithm.actor, features, device)
            results = train_env.step(actions)
            next_obs = [results[i]["obs"] for i in range(cfg["num_envs"])]
            terminated = np.array([results[i]["terminated"] for i in range(cfg["num_envs"])])
            truncated = np.array([results[i]["truncated"] for i in range(cfg["num_envs"])])
            raw_sparse = np.array([results[i]["sparse"] for i in range(cfg["num_envs"])])
            sparse = raw_sparse + cfg["sparse_offset"]
            for h, o in zip(histories, next_obs):
                h.append(o[cfg["reward_camera"]])
            predictions = (
                reward.predict([prefix_frames(h, cfg["max_frames"]) for h in histories], language) if reward else None
            )
            progress = predictions[:, 0] if predictions is not None else np.zeros(cfg["num_envs"])
            rewards = training_reward(cfg["reward"], cfg["reward_formula"], sparse, progress, previous_progress)
            next_features = encoder.encode(next_obs)
            # Capture terminal observations before resetting. Timeout transitions
            # bootstrap; true simulator successes do not. No model-driven resets.
            replay.add(features, actions, rewards, next_features, terminated, truncated)
            env_steps += cfg["num_envs"]
            previous_progress = progress.copy()
            episode_return += rewards
            for i in np.flatnonzero(terminated | truncated):
                record = {
                    "phase": "train",
                    "env_steps": env_steps,
                    "worker": int(i),
                    "episode": int(episode_id[i]),
                    "success": bool(terminated[i]),
                    "length": results[i]["elapsed"],
                    "training_return": float(episode_return[i]),
                }
                if predictions is not None:
                    record.update(
                        final_progress=float(predictions[i, 0]), final_success_probability=float(predictions[i, 1])
                    )
                log.episode(record)
                episode_id[i] += 1
                episode_return[i] = 0
            if env_steps >= cfg["sac"]["learning_starts"] and len(replay) >= cfg["sac"]["batch_size"]:
                # Preserve update/data ratio when increasing parallel environments.
                for _ in range(cfg["num_envs"] * cfg["updates_per_transition"]):
                    metrics = algorithm.train_step()
                if env_steps - last_log >= cfg["log_every"]:
                    log.log({k: float(v) for k, v in metrics.items()}, env_steps, "sac")
                    last_log = env_steps
            indices = np.flatnonzero(terminated | truncated)
            if len(indices):
                resets = train_env.reset({int(i): None for i in indices})
                reset_obs = [resets[int(i)]["obs"] for i in indices]
                reset_features = encoder.encode(reset_obs)
                for j, i in enumerate(indices):
                    next_obs[i] = reset_obs[j]
                    histories[i] = [reset_obs[j][cfg["reward_camera"]]]
                    for key in next_features:
                        next_features[key][i] = reset_features[key][j]
                previous_progress[indices] = 0
                if reward is not None:
                    reset_predictions = reward.predict(
                        [prefix_frames(histories[i], cfg["max_frames"]) for i in indices], language
                    )
                    previous_progress[indices] = reset_predictions[:, 0]
            obs, features = next_obs, next_features
            if env_steps - last_eval >= cfg["eval_every"] or env_steps == cfg["steps"]:
                final_metrics = evaluate(
                    cfg, algorithm.actor, encoder, eval_env, reward, language, log, env_steps, device, out
                )
                last_eval = env_steps
            if env_steps - last_save >= cfg["save_every"]:
                save_checkpoint(algorithm, Path(out) / "checkpoints/latest.pt", cfg, env_steps)
                last_save = env_steps
        latest = Path(out) / "checkpoints/latest.pt"
        final = Path(out) / "checkpoints/final.pt"
        if cfg.get("rolling_checkpoint_only") and latest.exists() and last_save == env_steps:
            latest.replace(final)
        else:
            save_checkpoint(algorithm, final, cfg, env_steps)
            if cfg.get("rolling_checkpoint_only"):
                latest.unlink(missing_ok=True)
        write_json(
            Path(out) / "status.json", {"status": "complete", "env_steps": env_steps, "final_metrics": final_metrics}
        )
    except BaseException:
        write_json(Path(out) / "status.json", {"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        for resource in (train_env, eval_env, reward):
            if resource is not None:
                resource.close()
        log.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--reward-gpu", type=int, required=True)
    args = parser.parse_args()
    if os.environ.get("AUDIT_LAUNCHER_GUARDED") != "1":
        raise RuntimeError("Use the launch module, which checks occupied GPUs before any CUDA imports")

    def stop(signum, frame):
        raise KeyboardInterrupt("Experiment cancelled")

    signal.signal(signal.SIGTERM, stop)
    run(json.loads(Path(args.config).read_text()), args.out, args.reward_gpu)


if __name__ == "__main__":
    main()
