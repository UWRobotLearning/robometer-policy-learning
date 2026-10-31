"""Short, local serial-vs-batched DSRL benchmark; no long experiment is scheduled.

Run with a regular experiment JSON (one task/seed, sparse reward), --gpu and --out.
Uses the shared collector, real simulator workers, frozen policy, and SAC optimizer.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath

from .config import load, run_config


def stats(values):
    return {
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "n": len(values),
    }


def benchmark(spec, out, gpu, repeats, rounds):
    import torch

    from experiments.reward_audit.envs import EnvPool, prepare_libero_paths
    from experiments.reward_audit.sac import Dino, Replay, actor_actions, make_sac

    from .policy import Pi0Client
    from .train import execute_parallel_chunks, select_reward

    cfg = run_config(spec, spec["tasks"][0], spec["seeds"][0], "sparse", gpu)
    write_json(out / "config.json", cfg)
    prepare_libero_paths(out / "libero_paths", REPO)
    torch.set_num_threads(cfg["torch_threads"])
    device = torch.device("cuda:0")
    policy = pool = None
    samples = []
    stop_memory = threading.Event()

    def monitor():
        while not stop_memory.is_set():
            result = subprocess.run(
                [
                    "nvidia-smi",
                    f"--id={gpu}",
                    "--query-gpu=memory.used,utilization.gpu",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                memory, utilization = map(float, result.stdout.strip().split(","))
                samples.append({"time": time.time(), "memory_mib": memory, "utilization": utilization})
            stop_memory.wait(0.5)

    thread = threading.Thread(target=monitor, daemon=True)
    thread.start()
    try:
        write_json(out / "status.json", {"status": "initializing"})
        policy = Pi0Client(
            Path(cfg["pi0_python"]),
            Path(cfg["pi0_checkpoint"]),
            cfg["seed"],
            gpu,
            out,
            config_name=cfg["policy_config_name"],
        )
        encoder = Dino(cfg["dino_checkpoint"], device)
        pool = EnvPool(cfg, cfg["num_envs"])
        prompt = pool.metadata[0]["language"]
        initial = {rank: rank for rank in range(cfg["num_envs"])}
        observations = [row["obs"] for row in pool.reset(initial).values()]
        seeds = [cfg["seed"] * 100000 + i for i in range(cfg["num_envs"])]
        rng = np.random.default_rng(cfg["seed"])

        def infer(mode, noise):
            start = time.perf_counter()
            if mode == "batched":
                actions = policy.infer_batch(observations, prompt, noise=noise, seeds=seeds, noise_mode="direct")
            else:
                actions = []
                for i, obs in enumerate(observations):
                    policy.reset_seed(seeds[i])
                    actions.append(policy.infer(obs, prompt, noise=noise[i], noise_mode="direct"))
                actions = np.stack(actions)
            return actions, time.perf_counter() - start

        noise = rng.uniform(-1, 1, (cfg["num_envs"], cfg["dsrl"]["noise_dim"])).astype(np.float32)
        serial, serial_compile = infer("serial", noise)
        batched, batch_compile = infer("batched", noise)
        expected = (cfg["num_envs"], policy.metadata["action_horizon"], 7)
        assert serial.shape == batched.shape == expected
        durations = {"serial": [], "batched": []}
        differences = []
        write_json(out / "status.json", {"status": "inference_benchmark"})
        for repeat in range(repeats):
            noise = rng.uniform(-1, 1, noise.shape).astype(np.float32)
            outputs = {}
            for mode in ("serial", "batched") if repeat % 2 == 0 else ("batched", "serial"):
                outputs[mode], elapsed = infer(mode, noise)
                durations[mode].append(elapsed)
            differences.append(np.abs(outputs["serial"] - outputs["batched"]).ravel())
        diff = np.concatenate(differences)
        inference = {mode: stats(values) for mode, values in durations.items()}
        inference.update(
            speedup=inference["serial"]["mean"] / inference["batched"]["mean"],
            first_call_seconds={"serial": serial_compile, "batched": batch_compile},
            output_shape=list(expected),
            difference={
                "mean_abs": float(diff.mean()),
                "p99_abs": float(np.percentile(diff, 99)),
                "max_abs": float(diff.max()),
            },
        )
        write_json(out / "inference.json", inference)
        print("INFERENCE " + json.dumps(inference), flush=True)
        if inference["difference"]["max_abs"] > 0.02:
            raise RuntimeError("Serial/batch action difference exceeds 0.02; inspect before training")

        blocks = []
        with (out / "rounds.jsonl").open("w", buffering=1) as stream:
            for block_id, mode in enumerate(("serial", "batched", "batched", "serial")):
                random.seed(cfg["seed"])
                np.random.seed(cfg["seed"])
                torch.manual_seed(cfg["seed"])
                torch.cuda.manual_seed_all(cfg["seed"])
                local = copy.deepcopy(cfg)
                local["dsrl"]["batched_inference"] = mode == "batched"
                replay = Replay(
                    10000,
                    {"dino_embedding": encoder.dimension, "observation/state": 8, "language": 384},
                    cfg["seed"],
                    action_dim=cfg["dsrl"]["noise_dim"],
                )
                learner = make_sac(local, replay, device, action_dim=cfg["dsrl"]["noise_dim"])
                observations = [row["obs"] for row in pool.reset(initial).values()]
                histories = [[obs[cfg["reward_camera"]]] for obs in observations]
                episode_counts = [0] * cfg["num_envs"]
                step = decision = measured = training_warmups = 0
                block_rows = []
                write_json(out / "status.json", {"status": "dsrl_benchmark", "block": block_id, "mode": mode})
                while measured < rounds:
                    started = time.perf_counter()
                    features = encoder.encode(observations)
                    can_train = step >= cfg["sac"]["learning_starts"]
                    noise = (
                        actor_actions(learner.actor, features, device)
                        if can_train
                        else np.random.uniform(-1, 1, (cfg["num_envs"], cfg["dsrl"]["noise_dim"])).astype(np.float32)
                    )
                    timings = {"encoding_actor_seconds": time.perf_counter() - started}
                    collected = execute_parallel_chunks(
                        policy,
                        pool,
                        observations,
                        prompt,
                        noise,
                        histories,
                        local,
                        decision,
                        cfg["num_envs"] * cfg["dsrl"]["action_exec_len"],
                        False,
                        timings=timings,
                    )
                    encoded = time.perf_counter()
                    next_observations = [record["result"]["obs"] for record in collected.values()]
                    next_features = encoder.encode(next_observations)
                    replay.add(
                        features,
                        noise,
                        np.array([select_reward("sparse", r["sparse_steps"], None) for r in collected.values()]),
                        next_features,
                        np.array([r["result"]["terminated"] for r in collected.values()]),
                        np.array([r["result"]["truncated"] for r in collected.values()]),
                    )
                    timings["next_encoding_replay_seconds"] = time.perf_counter() - encoded
                    step_delta = sum(len(r["sparse_steps"]) for r in collected.values())
                    step += step_delta
                    decision += len(collected)
                    updated = time.perf_counter()
                    if can_train:
                        for _ in collected:
                            learner.train_step()
                        torch.cuda.synchronize()
                    timings["update_seconds"] = time.perf_counter() - updated
                    reset_start = time.perf_counter()
                    observations = next_observations
                    resets = {}
                    for rank, record in collected.items():
                        if record["result"]["terminated"] or record["result"]["truncated"]:
                            episode_counts[rank] += 1
                            resets[rank] = (rank + cfg["num_envs"] * episode_counts[rank]) % cfg["train_init_states"]
                    if resets:
                        for rank, result in pool.reset(resets).items():
                            observations[rank] = result["obs"]
                            histories[rank] = [result["obs"][cfg["reward_camera"]]]
                    timings["reset_seconds"] = time.perf_counter() - reset_start
                    timings["total_seconds"] = time.perf_counter() - started
                    # Exclude replay warmup and first two optimizer-bearing rounds in each block.
                    if can_train:
                        training_warmups += 1
                    record = {
                        "block": block_id,
                        "mode": mode,
                        "env_steps": step,
                        "physical_steps": step_delta,
                        "chunks": len(collected),
                        **timings,
                        "measured": can_train and training_warmups > 2,
                    }
                    stream.write(json.dumps(record) + "\n")
                    if record["measured"]:
                        measured += 1
                        block_rows.append(record)
                result = {
                    "mode": mode,
                    "block": block_id,
                    "training_steps_including_warmup": step,
                    "actor_updates": learner._diagnostic_actor_updates,
                    "critic_updates": learner._diagnostic_critic_updates,
                    "seconds": sum(row["total_seconds"] for row in block_rows),
                    "physical_steps": sum(row["physical_steps"] for row in block_rows),
                    "components": {
                        key: sum(row[key] for row in block_rows) for key in timings if key != "total_seconds"
                    },
                }
                result["physical_steps_per_second"] = result["physical_steps"] / result["seconds"]
                blocks.append(result)
                print("BLOCK " + json.dumps(result), flush=True)
                write_json(out / "blocks.json", blocks)
                del learner, replay
                torch.cuda.empty_cache()
        throughputs = {}
        for mode in ("serial", "batched"):
            chosen = [b for b in blocks if b["mode"] == mode]
            seconds = sum(b["seconds"] for b in chosen)
            throughputs[mode] = {
                "physical_steps_per_second": sum(b["physical_steps"] for b in chosen) / seconds,
                "seconds": seconds,
                "components_seconds": {k: sum(b["components"][k] for b in chosen) for k in chosen[0]["components"]},
            }
        summary = {
            "task": cfg["task"],
            "base_model": cfg["base_model"],
            "checkpoint": cfg["pi0_checkpoint"],
            "num_envs": cfg["num_envs"],
            "executed_actions_per_chunk": cfg["dsrl"]["action_exec_len"],
            "inference": inference,
            "training": throughputs,
            "training_speedup": throughputs["batched"]["physical_steps_per_second"]
            / throughputs["serial"]["physical_steps_per_second"],
            "sampled_peak_gpu_mib": max(row["memory_mib"] for row in samples),
            "notes": [
                "Sparse reward; no Robometer inference in benchmark.",
                "ABBA block order; same seed, fresh actor/critics/replay per block.",
                "Default warmup and actor/critic update ratio retained; optimizer warmup excluded from timing.",
                "Timing includes collection, encoding, replay, gradients and resets; "
                "excludes evaluations/checkpoint I/O.",
                "Policy microbenchmark alternates execution order on identical observations and noise.",
                "No long training scheduled; speed benchmark is not a success-rate comparison.",
            ],
        }
        write_json(out / "summary.json", summary)
        render_report(out)
        write_json(out / "status.json", {"status": "complete"})
    except BaseException as exc:
        write_json(out / "status.json", {"status": "failed", "error": repr(exc)})
        raise
    finally:
        stop_memory.set()
        thread.join(timeout=5)
        write_json(out / "gpu_samples.json", samples)
        for resource in (pool, policy):
            if resource is not None:
                resource.close()


def render_report(out):
    """Render saved benchmark measurements without loading ML libraries or a GPU."""
    import html

    summary = json.loads((out / "summary.json").read_text())
    blocks = json.loads((out / "blocks.json").read_text())
    inf, training = summary["inference"], summary["training"]
    rows = []
    for label, serial, batched, unit, speedup in (
        (
            "Policy inference, four observations",
            inf["serial"]["mean"] * 1000,
            inf["batched"]["mean"] * 1000,
            "ms",
            inf["speedup"],
        ),
        (
            "DSRL training throughput",
            training["serial"]["physical_steps_per_second"],
            training["batched"]["physical_steps_per_second"],
            "physical steps/s",
            summary["training_speedup"],
        ),
    ):
        rows.append(
            f"<tr><td>{label}</td><td>{serial:.2f} {unit}</td><td>{batched:.2f} {unit}</td><td>{speedup:.2f}x</td></tr>"
        )
    component_rows = []
    for key in training["serial"]["components_seconds"]:
        serial = training["serial"]["components_seconds"][key]
        batched = training["batched"]["components_seconds"][key]
        component_rows.append(f"<tr><td>{key}</td><td>{serial:.2f}</td><td>{batched:.2f}</td></tr>")
    block_rows = "".join(
        f"<tr><td>{b['block']}</td><td>{b['mode']}</td>"
        f"<td>{b['physical_steps_per_second']:.2f}</td>"
        f"<td>{b['actor_updates']}</td><td>{b['critic_updates']}</td></tr>"
        for b in blocks
    )
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in summary["notes"])
    page = (
        f"<!doctype html><html><head><meta charset='utf-8'><title>Policy batching benchmark</title>"
        "<style>body{font:14px system-ui;background:#f3f7f7;color:#17333d;max-width:1050px;margin:30px auto}"
        "table{border-collapse:collapse;width:100%;background:white;margin:12px 0 24px}"
        "th,td{padding:9px;text-align:left;border-bottom:1px solid #d8e2e2}h2{font-size:18px}"
        "code{overflow-wrap:anywhere}.note{color:#526d75}</style></head><body>"
        f"<h1>π₀ + DSRL batching benchmark</h1><p>{html.escape(summary['task'])}</p>"
        f"<p>{summary['num_envs']} environments · {summary['executed_actions_per_chunk']} executed actions/chunk · "
        "sparse reward · short performance test, not a learning result</p>"
        "<table><tr><th>Measurement</th><th>Serial inference</th><th>Batched inference</th><th>Speedup</th></tr>"
        + "".join(rows)
        + "</table><h2>Training time breakdown</h2>"
        "<p>Seconds summed across the measured rounds; compilation and warmup excluded.</p>"
        "<table><tr><th>Component</th><th>Serial</th><th>Batched</th></tr>" + "".join(component_rows) + "</table>"
        "<h2>Individual blocks</h2><table><tr><th>Block</th><th>Mode</th><th>Physical steps/s</th>"
        "<th>Actor updates, incl. warmup</th><th>Critic updates, incl. warmup</th></tr>" + block_rows + "</table>"
        f"<p>Action differences on identical inputs: mean absolute {inf['difference']['mean_abs']:.6g}; "
        f"99th percentile {inf['difference']['p99_abs']:.6g}; maximum {inf['difference']['max_abs']:.6g}.</p>"
        f"<p>Sampled peak GPU memory: {summary['sampled_peak_gpu_mib'] / 1024:.2f} GiB "
        "(single model worker, both compiled batch shapes resident).</p>"
        f"<p>Checkpoint: <code>{html.escape(summary['checkpoint'])}</code></p>"
        f"<ul>{notes}</ul><p class='note'>Other GPU workloads may share CPU resources. "
        "These timing samples are not independent training seeds. "
        "Numerical differences can change later trajectories.</p>"
        "<p><a href='summary.json'>Summary JSON</a> · <a href='inference.json'>Inference timings</a> · "
        "<a href='rounds.jsonl'>Per-round measurements</a> · <a href='config.json'>Configuration</a></p></body></html>"
    )
    (out / "report.html").write_text(page)
    return out / "report.html"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--rounds", type=int, default=20)
    args = parser.parse_args()
    spec = load(args.config)
    if (
        spec["base_model"] == "scratch"
        or len(spec["tasks"]) != 1
        or len(spec["seeds"]) != 1
        or spec["rewards"] != ["sparse"]
    ):
        parser.error("Benchmark requires one pretrained-policy task/seed with sparse reward")
    if args.rounds < 1 or args.repeats < 1:
        parser.error("rounds and repeats must be positive")
    runtime = spec["runtime"]
    cache = Path(runtime["cache_dir"])
    os.environ.update(
        CUDA_VISIBLE_DEVICES=str(args.gpu),
        MUJOCO_GL="egl",
        MUJOCO_EGL_DEVICE_ID=str(args.gpu),
        EGL_DEVICE_ID=str(args.gpu),
        WANDB_MODE="disabled",
        WANDB_DISABLED="true",
        XLA_PYTHON_CLIENT_PREALLOCATE="false",
        PYTHONPATH=source_pythonpath(),
        XDG_CACHE_HOME=str(cache),
        HF_HOME=str(cache / "hf"),
        TMPDIR=runtime["tmp_dir"],
    )
    sys.path[:0] = source_pythonpath().split(os.pathsep)
    with reserve_gpus([args.gpu]):
        args.out.mkdir(parents=True, exist_ok=False)
        benchmark(spec, args.out, args.gpu, args.repeats, args.rounds)


if __name__ == "__main__":
    main()
