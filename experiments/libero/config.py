"""Resolve a single experiment specification without importing ML libraries."""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path

from experiments.reward_audit.config import REPO, task_map

ROBOMETER = "robometer/Robometer-4B@beef63bc914c5c189329d49c6d712d96d632aa34"
BASE_MODELS = {"scratch": None, "pi0": "pi0_libero", "pi0.5": "pi05_libero"}
DEFAULTS = {
    "name": "libero_comparison",
    "suite": "libero_90",
    "tasks": [28],
    "base_model": "pi0.5",
    "rewards": ["sparse", "robometer"],
    "seeds": [100, 101, 102],
    "steps": 100000,
    "eval_every": 10000,
    "eval_episodes": 25,
    "horizon": 400,
    "train_init_states": 25,
    "num_envs": 1,
    "settle_steps": 10,
    "save_every": 10000,
    "log_every": 1000,
    "buffer_capacity": 100000,
    "torch_threads": 4,
    "worker_timeout": 1800,
    "eval_videos": 0,
    "dino_checkpoint": "facebook/dinov2-small",
    "actor_hidden": [512, 512],
    "critic_hidden": [768, 512, 256],
    "reward_checkpoint": ROBOMETER,
    "reward_camera": "agentview_image",
    "reward_batch_size": 4,
    "updates_per_transition": 1,
    "initial_actor": None,
    "ood_claim": "unverified",
    "rolling_checkpoint_only": True,
    "audit_sparse_rewards": False,
    "sac": {},
    "dsrl": {},
    "runtime": {},
}
SAC_DEFAULTS = {
    "learning_starts": 5000,
    "gamma": 0.99,
    "tau": 0.005,
    "batch_size": 128,
    "num_critics": 5,
    "n_critics_to_sample": 2,
    "pooled_critic_features": True,
    "target_update_interval": 1,
    "num_critic_updates_per_actor_update": 1,
    "num_updates_per_train_step": 1,
    "actor_optimizer_lr": 1e-5,
    "critic_optimizer_lr": 1e-5,
    "ent_coef_lr": 0.0003,
    "target_entropy": 0,
    "train_actor_with_entropy": True,
    "train_critic_with_entropy": False,
}


def absolute(path: str | Path) -> str:
    """Keep venv interpreter symlinks: resolving them loses site-packages."""
    path = Path(os.path.expandvars(str(path))).expanduser()
    return str(path.absolute() if path.is_absolute() else (REPO / path).absolute())


def resolve(spec: dict) -> dict:
    unknown = set(spec) - set(DEFAULTS) - {"checkpoint", "max_frames", "reward_formula", "sparse_offset"}
    if unknown:
        raise ValueError(f"Unknown experiment settings: {sorted(unknown)}")
    cfg = copy.deepcopy(DEFAULTS)
    cfg.update(copy.deepcopy(spec))
    if cfg["base_model"] not in BASE_MODELS:
        raise ValueError("base_model must be scratch, pi0, or pi0.5")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", cfg["name"]):
        raise ValueError("name must be a simple directory name")
    names = task_map()
    if cfg["suite"] not in names:
        raise ValueError(f"Unknown suite: {cfg['suite']}")
    tasks = cfg["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a nonempty list of zero-based IDs or exact task names")
    ids = []
    for task in tasks:
        if isinstance(task, str) and task in names[cfg["suite"]]:
            task = names[cfg["suite"]].index(task)
        if type(task) is not int or not 0 <= task < len(names[cfg["suite"]]):
            raise ValueError(f"Unknown task in {cfg['suite']}: {task}")
        ids.append(task)
    cfg["tasks"] = ids
    for key, allowed in (("seeds", None), ("tasks", None), ("rewards", {"sparse", "robometer"})):
        values = cfg[key]
        if not isinstance(values, list) or not values or len(set(values)) != len(values):
            raise ValueError(f"{key} must be a nonempty list without duplicates")
        if allowed and set(values) - allowed:
            raise ValueError(f"{key} must contain only {sorted(allowed)}")
    if any(type(seed) is not int or seed < 0 for seed in cfg["seeds"]):
        raise ValueError("seeds must be nonnegative integers")
    for key in (
        "steps",
        "eval_every",
        "eval_episodes",
        "horizon",
        "train_init_states",
        "num_envs",
        "save_every",
        "log_every",
        "buffer_capacity",
        "torch_threads",
        "worker_timeout",
        "reward_batch_size",
        "updates_per_transition",
    ):
        if type(cfg[key]) is not int or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    # Official LIBERO tasks supply 50 reset states. Repeated evaluation uses the
    # same held-out states, without requesting nonexistent state IDs.
    cfg["eval_unique_states"] = min(cfg["eval_episodes"], 25)
    if cfg["train_init_states"] + cfg["eval_unique_states"] > 50:
        raise ValueError("Training and evaluation reset states must fit disjointly within 50 states")
    if cfg["steps"] % cfg["num_envs"]:
        raise ValueError("steps must be divisible by num_envs")
    pretrained = cfg["base_model"] != "scratch"
    runtime = cfg["runtime"]
    allowed_runtime = {
        "data_root",
        "gpus",
        "reward_gpu",
        "policy_python",
        "openpi_python",
        "reward_python",
        "cache_dir",
        "tmp_dir",
        "min_free_gib",
    }
    if set(runtime) - allowed_runtime:
        raise ValueError(f"Unknown runtime settings: {sorted(set(runtime) - allowed_runtime)}")
    data = Path(absolute(runtime.get("data_root", os.environ.get("REWARD_MODELING_DATA_ROOT", REPO / "outputs"))))
    cfg["runtime"] = {
        "data_root": str(data),
        "gpus": [0, 1],
        "reward_gpu": None,
        "policy_python": absolute(".venv-policy/bin/python"),
        "openpi_python": str(data / ".venv-pi0/bin/python"),
        "reward_python": absolute(".venv-reward/bin/python"),
        "cache_dir": str(data / "cache"),
        "tmp_dir": str(data / "tmp"),
        "min_free_gib": 30,
        **runtime,
    }
    runtime = cfg["runtime"]
    for key in ("data_root", "policy_python", "openpi_python", "reward_python", "cache_dir", "tmp_dir"):
        runtime[key] = absolute(runtime[key])
    gpus = runtime["gpus"]
    if (
        not isinstance(gpus, list)
        or not gpus
        or len(gpus) != len(set(gpus))
        or any(type(gpu) is not int or gpu < 0 for gpu in gpus)
    ):
        raise ValueError("runtime.gpus must list distinct nonnegative GPU indices")
    if runtime["reward_gpu"] is not None:
        if type(runtime["reward_gpu"]) is not int or runtime["reward_gpu"] < 0 or runtime["reward_gpu"] in gpus:
            raise ValueError("A dedicated reward_gpu must be a separate nonnegative GPU index outside runtime.gpus")
    if not isinstance(runtime["min_free_gib"], (int, float)) or runtime["min_free_gib"] < 0:
        raise ValueError("min_free_gib must be nonnegative")
    sac = copy.deepcopy(SAC_DEFAULTS)
    if pretrained:
        sac.update(
            learning_starts=500,
            gamma=0.999**20,
            compute_chunked_gamma=False,
            num_critic_updates_per_actor_update=25,
            actor_optimizer_lr=1e-4,
            critic_optimizer_lr=3e-4,
            target_entropy=-16,
            clip_grad_norm=10.0,
        )
    sac.update(cfg["sac"])
    cfg["sac"] = sac
    for key in (
        "batch_size",
        "num_critics",
        "n_critics_to_sample",
        "target_update_interval",
        "num_critic_updates_per_actor_update",
        "num_updates_per_train_step",
    ):
        if type(sac[key]) is not int or sac[key] < 1:
            raise ValueError(f"sac.{key} must be a positive integer")
    if type(sac["learning_starts"]) is not int or sac["learning_starts"] < 0:
        raise ValueError("sac.learning_starts must be a nonnegative integer")
    if not 0 <= sac["gamma"] <= 1 or not 0 <= sac["tau"] <= 1:
        raise ValueError("sac.gamma and sac.tau must lie in [0, 1]")
    if sac["n_critics_to_sample"] > sac["num_critics"]:
        raise ValueError("Cannot sample more critics than are available")
    cfg["max_frames"] = cfg.get("max_frames", 8 if pretrained else 4)
    cfg["sparse_offset"] = cfg.get("sparse_offset", 0.0 if pretrained else -1.0)
    cfg["reward_formula"] = cfg.get("reward_formula", "progress_chunk_mean" if pretrained else "progress_minus_one")
    if type(cfg["max_frames"]) is not int or cfg["max_frames"] < 1:
        raise ValueError("max_frames must be a positive integer")
    if cfg["reward_camera"] not in {"agentview_image", "robot0_eye_in_hand_image"}:
        raise ValueError("Unknown reward camera")
    if pretrained:
        if cfg["initial_actor"] or cfg["updates_per_transition"] != 1 or cfg["eval_videos"]:
            raise ValueError("DSRL does not support initial_actor, updates_per_transition overrides, or eval_videos")
        cfg["checkpoint"] = absolute(cfg.get("checkpoint", data / "pi0_checkpoints" / BASE_MODELS[cfg["base_model"]]))
        dsrl = {
            "method": "SAC direct pi0 noise (original DSRL semantics)",
            "noise_dim": 32,
            "action_exec_len": 10 if cfg["base_model"] == "pi0.5" else 20,
            "noise_mode": "direct",
            **cfg["dsrl"],
        }
        if (
            dsrl["noise_mode"] not in {"direct", "additive"}
            or type(dsrl.get("batched_inference", False)) is not bool
            or type(dsrl["action_exec_len"]) is not int
            or not 1 <= dsrl["action_exec_len"] <= 50
            or dsrl["noise_dim"] != 32
        ):
            raise ValueError("Invalid DSRL noise mode, chunk length (1-50), or noise dimension (32)")
        if cfg["reward_formula"] != "progress_chunk_mean" or cfg["sparse_offset"] != 0:
            raise ValueError("DSRL uses mean progress per chunk and native 0/1 sparse chunk reward")
        cfg["dsrl"] = dsrl
    elif cfg["dsrl"] or cfg.get("checkpoint"):
        raise ValueError(
            "scratch does not use an OpenPI checkpoint or DSRL settings; use initial_actor for a SAC warm start"
        )
    elif cfg["reward_formula"] not in {"progress", "progress_minus_one", "sparse_plus_progress", "delta"}:
        raise ValueError("Unknown SAC reward formula")
    if not cfg["rolling_checkpoint_only"]:
        raise ValueError("This runner requires rolling checkpoints to bound storage")
    cfg["initial_actor"] = absolute(cfg["initial_actor"]) if cfg["initial_actor"] else None
    return cfg


def load(path: str | Path) -> dict:
    return resolve(json.loads(Path(path).read_text()))


def task_root(spec: dict, out: Path, task_id: int) -> Path:
    prefix = {"scratch": "sac", "pi0": "pi0", "pi0.5": "pi05"}[spec["base_model"]]
    suite = spec["suite"].replace("_", "")
    return out / f"{prefix}_{suite}_task{task_id:02d}_{spec['name']}"


def run_config(spec: dict, task_id: int, seed: int, reward: str, gpu: int) -> dict:
    cfg = {
        key: copy.deepcopy(value)
        for key, value in spec.items()
        if key not in {"tasks", "seeds", "rewards", "runtime", "checkpoint"}
    }
    cfg.update(
        task_id=task_id,
        task=task_map()[spec["suite"]][task_id],
        seed=seed,
        reward=reward,
        name=f"{spec['name']}_{reward}_seed{seed}",
        comparison=f"{spec['name']}_task{task_id:02d}",
        policy_gpu=gpu,
        reward_gpu=spec["runtime"]["reward_gpu"] if spec["runtime"]["reward_gpu"] is not None else gpu,
        reward_python=spec["runtime"]["reward_python"],
    )
    if spec["base_model"] != "scratch":
        cfg.update(
            pi0_checkpoint=spec["checkpoint"],
            pi0_python=spec["runtime"]["openpi_python"],
            policy_config_name=BASE_MODELS[spec["base_model"]],
        )
        if reward == "sparse":
            cfg["reward_formula"] = "sparse_chunk_terminal"
    return cfg


def plan(spec: dict, out: Path) -> list[dict]:
    return [
        {
            "path": str(task_root(spec, out, task) / f"seed{seed}" / f"{reward}_seed{seed}"),
            "config": run_config(spec, task, seed, reward, spec["runtime"]["gpus"][0]),
        }
        for task in spec["tasks"]
        for seed in spec["seeds"]
        for reward in spec["rewards"]
    ]
