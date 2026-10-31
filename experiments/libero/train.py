"""SAC training: direct robot actions from scratch or steering a frozen OpenPI policy."""

from __future__ import annotations

import argparse
import json
import os
import random
import signal
import time
import traceback
from pathlib import Path

import numpy as np

from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.envs import EnvPool, prepare_libero_paths
from experiments.reward_audit.local_log import LocalLog
from experiments.reward_audit.metrics import prefix_frames, wilson
from experiments.reward_audit.rpc import RewardClient
from experiments.reward_audit.sac import Dino, Replay, actor_actions, make_sac, save_checkpoint

from .policy import Pi0Client


def select_reward(kind, sparse_steps, progress_steps):
    """One scalar per pi0 action chunk; both reward arms stay in [0, 1]."""
    if kind == "sparse":
        return float(max(sparse_steps))
    if kind == "robometer":
        if progress_steps is None or not len(progress_steps):
            raise ValueError("Missing Robometer scores for action chunk")
        values = np.asarray(progress_steps, dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError("Invalid Robometer progress")
        return float(values.mean())
    raise ValueError(kind)


def execute_chunk(
    policy,
    env,
    observation,
    prompt,
    steering,
    length,
    history,
    reward_camera,
    max_frames,
    need_model,
    noise_mode="additive",
):
    actions = policy.infer(observation, prompt, noise=steering, noise_mode=noise_mode)
    prefixes, sparse_steps = [], []
    for action in actions[:length]:
        result = env.step([action])[0]
        observation = result["obs"]
        sparse_steps.append(result["sparse"])
        history.append(observation[reward_camera])
        if need_model:
            prefixes.append(prefix_frames(history, max_frames))
        if result["terminated"] or result["truncated"]:
            break
    return observation, result, prefixes, sparse_steps


def execute_parallel_chunks(
    policy, env, observations, prompt, steering, histories, cfg, decision, budget, need_model, timings=None
):
    """Infer chunks, then step all live physics workers concurrently; never step past done.

    OpenPI inference can be batched with dsrl.batched_inference (default False). Physics
    uses EnvPool.call (send all requests before receiving); encoding and reward RPC
    are batched by the caller. The budget counts actual physical steps across workers.
    """
    actions, lengths, results = {}, {}, {}
    remaining = budget
    for rank in range(len(observations)):
        length = min(cfg["dsrl"]["action_exec_len"], remaining)
        if length <= 0:
            break
        remaining -= length
        lengths[rank] = length
        results[rank] = {"prefixes": [], "sparse_steps": []}
    if not lengths:
        raise ValueError("Collection requires a positive budget and at least one environment")
    started = time.perf_counter()
    seeds = [10_000_000 + cfg["seed"] * 1_000_000 + decision + rank for rank in lengths]
    mode = cfg["dsrl"].get("noise_mode", "direct")
    if cfg["dsrl"].get("batched_inference", False):
        batch = policy.infer_batch(
            [observations[rank] for rank in lengths],
            prompt,
            noise=steering[list(lengths)],
            noise_mode=mode,
            seeds=seeds,
        )
        actions = dict(zip(lengths, batch, strict=True))
    else:
        for rank, seed in zip(lengths, seeds, strict=True):
            policy.reset_seed(seed)
            actions[rank] = policy.infer(observations[rank], prompt, noise=steering[rank], noise_mode=mode)
    inferred = time.perf_counter()
    active = set(actions)
    for index in range(max(lengths.values())):
        responses = env.call({rank: ("step", actions[rank][index]) for rank in sorted(active) if index < lengths[rank]})
        for rank, result in responses.items():
            record = results[rank]
            record["result"] = result
            record["sparse_steps"].append(result["sparse"])
            histories[rank].append(result["obs"][cfg["reward_camera"]])
            if need_model:
                record["prefixes"].append(prefix_frames(histories[rank], cfg["max_frames"]))
            if result["terminated"] or result["truncated"]:
                active.remove(rank)
    if timings is not None:
        timings.update(policy_seconds=inferred - started, physics_seconds=time.perf_counter() - inferred)
    return results


def evaluate(cfg, policy, actor, encoder, env, device, log, step):
    """Identical reset states and pi0 sampling seeds in both reward arms."""
    import torch

    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    numpy_rng, python_rng = np.random.get_state(), random.getstate()
    records = []
    try:
        prompt = env.metadata[0]["language"]
        for episode in range(cfg["eval_episodes"]):
            state_id = cfg["train_init_states"] + episode % cfg.get("eval_unique_states", cfg["eval_episodes"])
            policy.reset_seed(cfg["seed"] * 100000 + cfg["train_init_states"] + episode)
            observation = env.reset({0: state_id})[0]["obs"]
            history = [observation[cfg["reward_camera"]]]
            native_return = 0.0
            while True:
                noise_mode = cfg["dsrl"].get("noise_mode", "additive")
                if step == 0:
                    steering = None if noise_mode == "direct" else np.zeros(cfg["dsrl"]["noise_dim"], dtype=np.float32)
                else:
                    features = encoder.encode([observation])
                    steering = actor_actions(actor, features, device, deterministic=True)[0]
                observation, result, _, sparse_steps = execute_chunk(
                    policy,
                    env,
                    observation,
                    prompt,
                    steering,
                    cfg["dsrl"]["action_exec_len"],
                    history,
                    cfg["reward_camera"],
                    cfg["max_frames"],
                    False,
                    noise_mode,
                )
                native_return += sum(sparse_steps)
                if result["terminated"] or result["truncated"]:
                    break
            record = {
                "phase": "eval",
                "env_steps": step,
                "episode": episode,
                "init_state_id": state_id,
                "success": bool(result["terminated"]),
                "length": result["elapsed"],
                "native_return": native_return,
                "timeout": bool(result["truncated"]),
            }
            log.episode(record)
            records.append(record)
        successes = sum(e["success"] for e in records)
        low, high = wilson(successes, len(records))
        log.evaluation(step, successes, len(records), (low, high))
        metrics = {
            "successes": successes,
            "episodes": len(records),
            "success_rate": successes / len(records),
            "ci_low": low,
            "ci_high": high,
            "avg_episode_length": float(np.mean([row["length"] for row in records])),
            "avg_native_return": float(np.mean([row["native_return"] for row in records])),
        }
        log.log(metrics, step, "eval")
        print(f"Evaluation at {step} transitions: {successes}/{len(records)} simulator successes", flush=True)
        return metrics
    finally:
        torch.set_rng_state(cpu_rng)
        torch.cuda.set_rng_state_all(cuda_rng)
        np.random.set_state(numpy_rng)
        random.setstate(python_rng)


def run_dsrl(cfg, out):
    import torch

    if os.environ.get("AUDIT_LAUNCHER_GUARDED") != "1":
        raise RuntimeError("Launch through the guarded pi0 DSRL coordinator")
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    prepare_libero_paths(out / "libero_paths", REPO)
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.cuda.manual_seed_all(cfg["seed"])
    torch.set_num_threads(cfg["torch_threads"])
    device = torch.device("cuda:0")
    log = LocalLog(out)
    policy = train_env = eval_env = reward = None
    write_json(out / "status.json", {"status": "running"})
    try:
        policy = Pi0Client(
            Path(cfg["pi0_python"]),
            Path(cfg["pi0_checkpoint"]),
            cfg["seed"],
            cfg["policy_gpu"],
            out,
            config_name=cfg.get("policy_config_name", "pi0_libero"),
        )
        if cfg["reward"] == "robometer" or cfg.get("audit_sparse_rewards"):
            reward = RewardClient(cfg, out, cfg["reward_gpu"])
        encoder = Dino(cfg["dino_checkpoint"], device)
        train_env = EnvPool(cfg, cfg["num_envs"])
        eval_env = EnvPool(cfg, 1, rank_offset=100000)
        prompt = train_env.metadata[0]["language"]
        replay = Replay(
            cfg["buffer_capacity"],
            {"dino_embedding": encoder.dimension, "observation/state": 8, "language": 384},
            cfg["seed"],
            action_dim=cfg["dsrl"]["noise_dim"],
        )
        algorithm = make_sac(cfg, replay, device, action_dim=cfg["dsrl"]["noise_dim"])
        observations = [row["obs"] for row in train_env.reset().values()]
        histories = [[obs[cfg["reward_camera"]]] for obs in observations]
        episode_returns = np.zeros(cfg["num_envs"])
        native_returns = np.zeros(cfg["num_envs"])
        episode_chunks = np.zeros(cfg["num_envs"], dtype=int)
        episode_id = 0
        step, decision, last_eval, last_save, last_log = 0, 0, 0, 0, 0
        noise_mode = cfg["dsrl"].get("noise_mode", "additive")
        if noise_mode not in ("additive", "direct"):
            raise ValueError(f"Unknown noise mode: {noise_mode}")
        final_metrics = evaluate(cfg, policy, algorithm.actor, encoder, eval_env, device, log, 0)
        while step < cfg["steps"]:
            features = encoder.encode(observations)
            can_train = (
                (step >= cfg["sac"]["learning_starts"])
                if noise_mode == "direct"
                else (decision >= cfg["sac"]["learning_starts"])
            )
            if can_train:
                steering = actor_actions(algorithm.actor, features, device)
            elif noise_mode == "direct":
                steering = np.random.uniform(-1, 1, (len(observations), cfg["dsrl"]["noise_dim"])).astype(np.float32)
            else:
                steering = np.zeros((len(observations), cfg["dsrl"]["noise_dim"]), dtype=np.float32)
            collected = execute_parallel_chunks(
                policy,
                train_env,
                observations,
                prompt,
                steering,
                histories,
                cfg,
                decision,
                cfg["steps"] - step,
                reward is not None,
            )
            ranks = list(collected)
            next_observations = [collected[rank]["result"]["obs"] for rank in ranks]
            next_features = encoder.encode(next_observations)
            prefixes = [prefix for rank in ranks for prefix in collected[rank]["prefixes"]]
            predictions = reward.predict(prefixes, prompt)[:, 0] if reward is not None else None
            cursor = 0
            scalar_rewards, terminateds, truncateds = [], [], []
            step += sum(len(record["sparse_steps"]) for record in collected.values())
            decision += len(ranks)
            reset_ranks = {}
            for rank in ranks:
                record = collected[rank]
                result, sparse_steps = record["result"], record["sparse_steps"]
                progress = predictions[cursor : cursor + len(sparse_steps)] if predictions is not None else None
                cursor += len(sparse_steps)
                scalar_reward = select_reward(cfg["reward"], sparse_steps, progress)
                terminated, truncated = bool(result["terminated"]), bool(result["truncated"])
                scalar_rewards.append(scalar_reward)
                terminateds.append(terminated)
                truncateds.append(truncated)
                episode_returns[rank] += scalar_reward
                native_returns[rank] += sum(sparse_steps)
                episode_chunks[rank] += 1
                observations[rank] = result["obs"]
                if terminated or truncated:
                    episode = {
                        "phase": "train",
                        "env_steps": step,
                        "episode": episode_id,
                        "env": rank,
                        "success": terminated,
                        "length": result["elapsed"],
                        "training_return": float(episode_returns[rank]),
                        "native_return": float(native_returns[rank]),
                        "chunks": int(episode_chunks[rank]),
                        "timeout": truncated,
                    }
                    if progress is not None:
                        episode["final_progress"] = float(progress[-1])
                    log.episode(episode)
                    episode_id += 1
                    episode_returns[rank] = native_returns[rank] = episode_chunks[rank] = 0
                    reset_ranks[rank] = None
            replay.add(
                {key: value[ranks] for key, value in features.items()},
                steering[ranks],
                np.asarray(scalar_rewards, dtype=np.float32),
                next_features,
                np.asarray(terminateds),
                np.asarray(truncateds),
            )
            if can_train and (noise_mode == "direct" or len(replay) >= cfg["sac"]["batch_size"]):
                # Preserve one actor / 25 critic updates per collected replay transition.
                for update in range(len(ranks)):
                    log_due = step - last_log >= cfg["log_every"] and update == len(ranks) - 1
                    metrics = algorithm.train_step(collect_diagnostics=log_due)
                    if log_due:
                        metrics.update(replay_size=len(replay), collected_chunks=decision)
                        log.log({key: float(value) for key, value in metrics.items()}, step, "sac")
                        last_log = step
            if reset_ranks:
                for rank, result in train_env.reset(reset_ranks).items():
                    observations[rank] = result["obs"]
                    histories[rank] = [result["obs"][cfg["reward_camera"]]]
            if step - last_eval >= cfg["eval_every"] or step == cfg["steps"]:
                final_metrics = evaluate(cfg, policy, algorithm.actor, encoder, eval_env, device, log, step)
                last_eval = step
            if step - last_save >= cfg["save_every"]:
                save_checkpoint(algorithm, out / "checkpoints/latest.pt", cfg, step)
                last_save = step
        latest = out / "checkpoints/latest.pt"
        final = out / "checkpoints/final.pt"
        if cfg.get("rolling_checkpoint_only") and latest.exists() and last_save == step:
            latest.replace(final)
        else:
            save_checkpoint(algorithm, final, cfg, step)
            if cfg.get("rolling_checkpoint_only"):
                latest.unlink(missing_ok=True)
        write_json(
            out / "status.json",
            {"status": "complete", "env_steps": step, "decisions": decision, "final_metrics": final_metrics},
        )
    except BaseException:
        write_json(out / "status.json", {"status": "failed", "error": traceback.format_exc()})
        raise
    finally:
        for resource in (train_env, eval_env, reward, policy):
            if resource is not None:
                resource.close()
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("AUDIT_LAUNCHER_GUARDED") != "1":
        parser.error("Run via python -m experiments.libero run")

    def stop(signum, frame):
        raise KeyboardInterrupt("Experiment interrupted")

    signal.signal(signal.SIGTERM, stop)
    cfg = json.loads(args.config.read_text())
    if cfg["base_model"] == "scratch":
        from experiments.reward_audit.train import run as run_scratch

        run_scratch(cfg, args.out, cfg["reward_gpu"])
    else:
        run_dsrl(cfg, args.out)


if __name__ == "__main__":
    main()
