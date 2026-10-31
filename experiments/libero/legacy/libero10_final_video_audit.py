"""Replay the final task 06/08 DSRL actors on four matched held-out states."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from experiments.libero.legacy.final_policy_video_audit import collect
from experiments.libero.legacy.libero10_video_audit import ROOT, task_name
from experiments.libero.legacy.task38_video_audit import FPS, SCORE_EVERY, VIDEO_EVERY, score
from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.envs import prepare_libero_paths
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath


RUN_ROOTS = {
    6: "pi0_libero10_task06_dsrl_5seeds_20261015",
    8: "pi0_libero10_task08_dsrl_8seeds_20261016",
}


def selected_states(task_id: int) -> tuple[int, ...]:
    """Use two successes and two failures from the frozen-policy video audit."""
    task_root = ROOT / f"task{task_id:02d}"
    selection_path = task_root / "final_policies" / "selected_states.json"
    if selection_path.exists():
        states = tuple(json.loads(selection_path.read_text())["init_state_ids"])
        if len(states) != 4 or len(set(states)) != 4:
            raise RuntimeError(f"Invalid saved state selection: {selection_path}")
        return states
    records = json.loads((task_root / "policy/episodes.json").read_text())
    successes = [int(row["init_state_id"]) for row in records if row["success"]]
    failures = [int(row["init_state_id"]) for row in records if not row["success"]]
    if len(successes) < 2 or len(failures) < 2:
        raise RuntimeError(f"Need two frozen π₀ successes and failures for task {task_id}")
    states = tuple(successes[:2] + failures[:2])
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(selection_path, {"init_state_ids": states,
                                "selection": "first two successful and first two failed frozen pi0 videos"})
    return states


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True, choices=tuple(RUN_ROOTS))
    parser.add_argument("--arm", choices=("sparse", "robometer"), required=True)
    parser.add_argument("--gpu", type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    run = (ROOT.parent / RUN_ROOTS[args.task_id] / "seed100" /
           f"{args.arm}_seed100")
    actor_checkpoint = run / "checkpoints/final.pt"
    if not actor_checkpoint.is_file() or json.loads((run / "status.json").read_text())["status"] != "complete":
        raise RuntimeError(f"Final trained actor unavailable: {run}")
    cfg = json.loads((run / "config.json").read_text())
    if (cfg["suite"] != "libero_10" or cfg["task_id"] != args.task_id or
            cfg["task"] != task_name(args.task_id) or cfg["reward"] != args.arm or
            cfg["dsrl"]["noise_mode"] != "direct"):
        raise RuntimeError(f"Unexpected DSRL run configuration: {run}")
    states = selected_states(args.task_id)
    out = ROOT / f"task{args.task_id:02d}" / "final_policies" / f"{args.arm}_seed100"
    for name in ("videos", "frames", "scores"):
        (out / name).mkdir(parents=True, exist_ok=True)
    cfg.update(policy_gpu=args.gpu, reward_gpu=args.gpu, eval_episodes=len(states))
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
    sys.path.insert(0, str(REPO / "LIBERO"))
    try:
        with reserve_gpus([args.gpu]):
            write_json(out / "status.json", {"status": "collecting", "episodes_target": len(states)})
            records = collect(cfg, out, args.gpu, actor_checkpoint, states)
            if set(int(row["init_state_id"]) for row in records) != set(states):
                raise RuntimeError(f"Expected rollouts on states {states}, got {records}")
            write_json(out / "status.json", {"status": "scoring", "episodes_collected": len(records)})
            score(cfg, out, args.gpu, records)
        write_json(out / "summary.json", {
            "task_id": args.task_id, "arm": args.arm, "seed": cfg["seed"],
            "actor_checkpoint": str(actor_checkpoint), "checkpoint_env_steps": cfg["steps"],
            "init_state_ids": states, "successes": sum(row["success"] for row in records),
            "episodes": len(records), "records": records,
            "video_every_physical_steps": VIDEO_EVERY,
            "robometer_every_physical_steps": SCORE_EVERY, "fps": FPS,
        })
        write_json(out / "status.json", {"status": "complete", "episodes": len(records)})
        print(f"Complete: task {args.task_id}, {args.arm}, states {states}", flush=True)
    except BaseException as error:
        write_json(out / "status.json", {"status": "failed", "error": repr(error)})
        raise


if __name__ == "__main__":
    main()
