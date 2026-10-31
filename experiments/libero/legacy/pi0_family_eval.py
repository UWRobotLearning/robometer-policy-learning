"""Evaluate frozen pi0_libero on one GPU's share of the four LIBERO training suites."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from experiments.libero.policy import Pi0Client, evaluate
from experiments.reward_audit.config import HERE, REPO, load_suite, materialize, task_map, write_json
from experiments.reward_audit.envs import EnvPool, prepare_libero_paths
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath


SUITES = ("libero_spatial", "libero_object", "libero_goal", "libero_10")
DATA_ROOT = Path("/data/rohan/Reward-Modeling")
ROOT = DATA_ROOT / "pi0_libero_families_20261014"
CHECKPOINT = DATA_ROOT / "pi0_checkpoints/pi0_libero"
PI0_PYTHON = DATA_ROOT / ".venv-pi0/bin/python"
SEED = 100
HORIZON = 400
EVAL_STATES = tuple(range(25, 50))
ACTION_EXEC_LEN = 20


def assignments(gpu: int) -> list[tuple[str, int, str]]:
    all_tasks = [(suite, index, name) for suite in SUITES
                 for index, name in enumerate(task_map()[suite])]
    return all_tasks[gpu::2]


def task_dir(root: Path, suite: str, index: int, name: str) -> Path:
    return root / suite / f"{index:02d}_{name}"


def completed(folder: Path) -> bool:
    summary_path = folder / "summary.json"
    status_path = folder / "status.json"
    if not summary_path.exists() or not status_path.exists():
        return False
    summary = json.loads(summary_path.read_text())
    status = json.loads(status_path.read_text())
    return (status.get("status") == "complete" and summary.get("episodes") == len(EVAL_STATES))


def evaluate_task(policy: Pi0Client, root: Path, suite_config: dict,
                  suite: str, index: int, name: str, gpu: int) -> None:
    folder = task_dir(root, suite, index, name)
    if completed(folder):
        print(f"SKIP {suite} {index:02d}: complete", flush=True)
        return
    folder.mkdir(parents=True, exist_ok=True)
    cfg = materialize(suite_config, {"name": name, "suite": suite, "task": name,
                                     "reward": "sparse", "ood_claim": "in_distribution"})
    write_json(folder / "config.json", {"suite": suite, "task_id": index, "task": name,
                                        "checkpoint": str(CHECKPOINT), "seed": SEED,
                                        "action_exec_len": ACTION_EXEC_LEN, "horizon": HORIZON,
                                        "eval_state_ids": EVAL_STATES, "gpu": gpu})
    write_json(folder / "status.json", {"status": "running"})
    env = None
    try:
        policy.reset_seed(SEED)
        env = EnvPool(cfg, 1, rank_offset=100000)
        result = evaluate(policy, env, cfg, folder, ACTION_EXEC_LEN)
        write_json(folder / "summary.json", result)
        write_json(folder / "status.json", {"status": "complete",
                                            "successes": result["successes"],
                                            "episodes": result["episodes"]})
        print(f"DONE {suite} {index:02d}: {result['successes']}/{result['episodes']} {name}", flush=True)
    except BaseException as error:
        write_json(folder / "status.json", {"status": "failed", "error": repr(error)})
        print(f"FAILED {suite} {index:02d}: {error!r}", flush=True)
    finally:
        if env is not None:
            env.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpu", type=int, required=True, choices=(0, 1))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    root = args.root.resolve()
    lane = root / f"gpu{args.gpu}"
    lane.mkdir(parents=True, exist_ok=True)
    os.environ["PYTHONPATH"] = source_pythonpath()
    prepare_libero_paths(lane / "libero_paths", REPO)
    base = load_suite(HERE / "configs/stove.json")
    base["defaults"].update(seed=SEED, horizon=HORIZON, train_init_states=25,
                            eval_episodes=len(EVAL_STATES))
    with reserve_gpus([args.gpu]):
        policy = Pi0Client(PI0_PYTHON.absolute(), CHECKPOINT.resolve(), SEED, args.gpu, lane)
        try:
            for suite, index, name in assignments(args.gpu):
                evaluate_task(policy, root, base, suite, index, name, args.gpu)
        finally:
            policy.close()
    incomplete = [(suite, index) for suite, index, name in assignments(args.gpu)
                  if not completed(task_dir(root, suite, index, name))]
    if incomplete:
        raise RuntimeError(f"GPU {args.gpu} has incomplete tasks: {incomplete}")


if __name__ == "__main__":
    main()
