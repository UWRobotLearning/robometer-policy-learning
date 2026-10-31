"""One CLI: plan or run a JSON experiment, then analyze any results folder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load, plan


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("plan", "run"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        command.add_argument("--out", type=Path, help="Default: runtime.data_root/experiments/name")
        if name == "run":
            command.add_argument("--resume", action="store_true", help="Skip complete runs; allow adding new seeds")
            command.add_argument(
                "--wait-for-gpus", action="store_true", help="Wait for idle GPUs without stopping other jobs"
            )
            command.add_argument(
                "--after-run",
                type=Path,
                action="append",
                default=[],
                help="Queue until this existing run completes successfully; repeat for multiple prerequisites",
            )
    analyze = commands.add_parser("analyze")
    analyze.add_argument("results", type=Path, help="New experiment, task folder, or legacy SAC/DSRL results")
    analyze.add_argument("--out", type=Path, help="Default: results/analysis")
    video = commands.add_parser(
        "video-audit", help="Compare pretrained and final policies on matched evaluation episodes"
    )
    video.add_argument("results", type=Path, help="Completed results directory or a JSON config for frozen evaluation")
    video.add_argument("--out", type=Path, required=True)
    video.add_argument("--seed", type=int, default=100)
    video.add_argument("--task-id", type=int)
    video.add_argument("--episodes", type=int, default=25)
    video.add_argument(
        "--horizon", type=int, help="Override and record the evaluation horizon; use 600 for LIBERO-10 video audits"
    )
    video.add_argument("--successes", type=int, default=2)
    video.add_argument("--failures", type=int, default=2)
    video.add_argument("--selection-seed", type=int, default=0)
    video.add_argument("--demos", type=int, default=0, help="Number of expert demonstration videos to include")
    video.add_argument("--demo-file", type=Path, help="Local task HDF5; default downloads the pinned LIBERO dataset")
    video.add_argument("--gpus", type=int, nargs="+", default=[0, 1])
    video.add_argument("--training-coverage", default="", help="Verified checkpoint training coverage to display")
    video.add_argument("--sampling", choices=("deterministic", "stochastic"), default="deterministic")
    args = parser.parse_args()
    if args.command == "video-audit":
        from .videos import run_audit

        print(
            run_audit(
                args.results,
                args.out,
                seed=args.seed,
                task_id=args.task_id,
                episodes=args.episodes,
                successes=args.successes,
                failures=args.failures,
                selection_seed=args.selection_seed,
                gpus=tuple(args.gpus),
                training_coverage=args.training_coverage,
                sampling=args.sampling,
                horizon=args.horizon,
                demos=args.demos,
                demo_file=args.demo_file,
            )
        )
        return
    if args.command == "analyze":
        from .analysis import generate

        print(generate(args.results, args.out))
        return
    spec = load(args.config)
    out = args.out.absolute() if args.out else Path(spec["runtime"]["data_root"]) / "experiments" / spec["name"]
    if args.command == "plan":
        print(json.dumps({"experiment": spec, "out": str(out), "runs": plan(spec, out)}, indent=2))
    else:
        from .runner import execute

        execute(spec, out, resume=args.resume, wait_for_gpus=args.wait_for_gpus, after_runs=args.after_run)


if __name__ == "__main__":
    main()
