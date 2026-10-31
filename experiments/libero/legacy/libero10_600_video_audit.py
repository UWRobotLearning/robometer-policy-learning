"""Evaluate 25 LIBERO-10 states at 600 steps, then sample four audit videos."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

from experiments.libero.legacy.final_policy_video_audit import collect as collect_final
from experiments.libero.legacy.libero10_final_video_audit import RUN_ROOTS
from experiments.libero.legacy.libero10_video_audit import ROOT as ORIGINAL_ROOT, task_name
from experiments.libero.legacy.task38_video_audit import (
    FPS, SCORE_EVERY, VIDEO_EVERY, collect as collect_frozen, score,
)
from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.envs import prepare_libero_paths
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath


ROOT = ORIGINAL_ROOT.parent / "pi0_libero10_video_audit_600_20261018"
HORIZON = 600
EVAL_STATES = tuple(range(25, 50))


def choose_video_states(records: list[dict], selection_seed: int) -> dict:
    """Sample two per outcome where possible, filling shortages from the other class."""
    by_state = {int(record["init_state_id"]): record for record in records}
    if set(by_state) != set(EVAL_STATES) or len(records) != len(EVAL_STATES):
        raise ValueError("Need one completed evaluation for each of the 25 held-out states")
    successes = sorted(state for state, row in by_state.items() if row["success"])
    failures = sorted(state for state, row in by_state.items() if not row["success"])
    rng = random.Random(selection_seed)
    selected_successes = rng.sample(successes, min(2, len(successes)))
    selected_failures = rng.sample(failures, min(2, len(failures)))
    if len(selected_successes) < 2:
        remaining = [state for state in failures if state not in selected_failures]
        selected_failures.extend(rng.sample(remaining, 2 - len(selected_successes)))
    if len(selected_failures) < 2:
        remaining = [state for state in successes if state not in selected_successes]
        selected_successes.extend(rng.sample(remaining, 2 - len(selected_failures)))
    selected = selected_successes + selected_failures
    if len(selected) != 4 or len(set(selected)) != 4:
        raise RuntimeError("Could not select four distinct videos")
    return {"selection_seed": selection_seed, "episodes_evaluated": 25,
            "available_successes": len(successes), "available_failures": len(failures),
            "selected_successes": selected_successes,
            "selected_failures": selected_failures,
            "selected_state_ids": selected}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=tuple(RUN_ROOTS), required=True)
    parser.add_argument("--kind", choices=("policy", "sparse", "robometer"), required=True)
    parser.add_argument("--gpu", type=int, choices=(0, 1), required=True)
    args = parser.parse_args()
    source_arm = "sparse" if args.kind == "policy" else args.kind
    run = (ROOT.parent / RUN_ROOTS[args.task_id] / "seed100" /
           f"{source_arm}_seed100")
    if json.loads((run / "status.json").read_text()).get("status") != "complete":
        raise RuntimeError(f"Training run is incomplete: {run}")
    cfg = json.loads((run / "config.json").read_text())
    if (cfg.get("suite") != "libero_10" or cfg.get("task_id") != args.task_id or
            cfg.get("task") != task_name(args.task_id) or
            cfg.get("dsrl", {}).get("noise_mode") != "direct"):
        raise RuntimeError(f"Unexpected source configuration: {run}")
    states = EVAL_STATES
    out = ROOT / f"task{args.task_id:02d}" / args.kind
    for name in ("videos", "frames", "scores"):
        (out / name).mkdir(parents=True, exist_ok=True)
    cfg.update(horizon=HORIZON, policy_gpu=args.gpu, reward_gpu=args.gpu,
               eval_episodes=len(states))
    config_path = out / "config.json"
    if config_path.exists():
        old_cfg = json.loads(config_path.read_text())
        if {key: value for key, value in old_cfg.items() if key != "eval_episodes"} != {
                key: value for key, value in cfg.items() if key != "eval_episodes"}:
            raise RuntimeError(f"Existing audit config differs beyond evaluation count: {config_path}")
    write_json(config_path, cfg)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_GL="egl",
                      MUJOCO_EGL_DEVICE_ID=str(args.gpu), EGL_DEVICE_ID=str(args.gpu),
                      XLA_PYTHON_CLIENT_PREALLOCATE="false", WANDB_MODE="disabled",
                      WANDB_DISABLED="true", PYTHONPATH=source_pythonpath(),
                      TMPDIR="/data/rohan/rbm-tmp")
    prepare_libero_paths(out / "libero_paths", REPO)
    sys.path.insert(0, str(REPO / "LIBERO"))
    try:
        with reserve_gpus([args.gpu]):
            write_json(out / "status.json", {"status": "collecting", "episodes_target": 25,
                                              "horizon": HORIZON})
            if args.kind == "policy":
                records = collect_frozen(cfg, out, args.gpu, states)
                actor_checkpoint = cfg["pi0_checkpoint"]
            else:
                actor_checkpoint = run / "checkpoints/final.pt"
                if not actor_checkpoint.is_file():
                    raise RuntimeError(f"Missing final actor checkpoint: {actor_checkpoint}")
                records = collect_final(cfg, out, args.gpu, actor_checkpoint, states)
            if set(int(row["init_state_id"]) for row in records) != set(states):
                raise RuntimeError(f"Expected states {states}, got {records}")
            seed = 20261018 + args.task_id * 10 + ("policy", "sparse", "robometer").index(args.kind)
            selection = choose_video_states(records, seed)
            selection_path = out / "selection.json"
            if selection_path.exists() and json.loads(selection_path.read_text()) != selection:
                raise RuntimeError(f"Existing video selection differs: {selection_path}")
            write_json(selection_path, selection)
            selected = [next(row for row in records if row["init_state_id"] == state)
                        for state in selection["selected_state_ids"]]
            write_json(out / "status.json", {"status": "scoring", "episodes_collected": 25,
                                              "horizon": HORIZON})
            score(cfg, out, args.gpu, selected, all_records=records)
        write_json(out / "summary.json", {
            "task_id": args.task_id, "kind": args.kind, "seed": cfg["seed"],
            "policy_checkpoint": cfg["pi0_checkpoint"],
            "actor_checkpoint": str(actor_checkpoint),
            "checkpoint_env_steps": 0 if args.kind == "policy" else cfg["steps"],
            "horizon": HORIZON, "init_state_ids": states,
            "successes": sum(row["success"] for row in records),
            "episodes": len(records), "records": records,
            "video_selection": selection,
            "video_every_physical_steps": VIDEO_EVERY,
            "robometer_every_physical_steps": SCORE_EVERY, "fps": FPS,
        })
        for record in records:
            if record["init_state_id"] in selection["selected_state_ids"]:
                continue
            ident = record["init_state_id"]
            for path in (out / "videos" / f"episode_{ident}.mp4",
                         out / "frames" / f"episode_{ident}.npz",
                         out / "scores" / f"episode_{ident}.json"):
                path.unlink(missing_ok=True)
        write_json(out / "status.json", {"status": "complete", "episodes": 25,
                                          "horizon": HORIZON})
        print(f"Complete: task {args.task_id}, {args.kind}, 25 episodes at 600 steps", flush=True)
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "horizon": HORIZON,
                                          "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
