"""Run and monitor the two-GPU frozen pi0 LIBERO-family evaluation sweep."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

from experiments.libero.legacy.pi0_family_eval import (
    ACTION_EXEC_LEN, CHECKPOINT, EVAL_STATES, HORIZON, PI0_PYTHON, ROOT, SEED,
    SUITES, assignments, completed, task_dir,
)
from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.launch import source_pythonpath


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def status() -> dict:
    tasks = [(suite, index, name) for gpu in (0, 1) for suite, index, name in assignments(gpu)]
    return {"total_tasks": len(tasks),
            "completed_tasks": sum(completed(task_dir(ROOT, suite, index, name))
                                   for suite, index, name in tasks),
            "updated": now()}


def launch(gpu: int) -> tuple[subprocess.Popen, object]:
    lane = ROOT / f"gpu{gpu}"
    lane.mkdir(parents=True, exist_ok=True)
    log = (lane / "console.log").open("ab")
    env = {**os.environ, "PYTHONPATH": source_pythonpath(),
           "MUJOCO_GL": "egl", "MUJOCO_EGL_DEVICE_ID": str(gpu),
           "EGL_DEVICE_ID": str(gpu), "TMPDIR": "/data/rohan/rbm-tmp",
           "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
           "WANDB_MODE": "disabled", "WANDB_DISABLED": "true"}
    process = subprocess.Popen([
        sys.executable, "-u", "-m", "experiments.libero.legacy.pi0_family_eval",
        "--gpu", str(gpu), "--root", str(ROOT)],
        cwd=REPO, env=env, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    return process, log


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / "scheduler.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("A LIBERO-family sweep scheduler is already running") from None
        if not (CHECKPOINT / "params").exists() or not (CHECKPOINT / "assets").exists():
            raise RuntimeError(f"Missing pi0 checkpoint: {CHECKPOINT}")
        if not PI0_PYTHON.exists():
            raise RuntimeError(f"Missing pi0 interpreter: {PI0_PYTHON}")
        if shutil.disk_usage(ROOT).free < 30 * 1024**3:
            raise RuntimeError("Less than 30 GiB free on /data")
        manifest_path = ROOT / "manifest.json"
        if manifest_path.exists():
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("suites") != list(SUITES) or manifest.get("seed") != SEED:
                raise RuntimeError(f"Unexpected sweep manifest: {manifest_path}")
        else:
            write_json(manifest_path, {"suites": SUITES, "tasks_per_suite": 10,
                                       "seed": SEED, "eval_states": EVAL_STATES,
                                       "horizon": HORIZON, "action_exec_len": ACTION_EXEC_LEN,
                                       "checkpoint": str(CHECKPOINT), "created": now()})
        active = {}
        retries = {0: 0, 1: 0}
        try:
            for gpu in (0, 1):
                active[gpu] = launch(gpu)
                print(f"Started GPU {gpu} lane: PID {active[gpu][0].pid}", flush=True)
            while active:
                write_json(ROOT / "orchestrator.json", {"status": "running", "pid": os.getpid(),
                            "active": {str(gpu): process.pid for gpu, (process, _) in active.items()},
                            **status()})
                time.sleep(30)
                for gpu, (process, log) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    log.close()
                    del active[gpu]
                    remaining = [(suite, index) for suite, index, name in assignments(gpu)
                                 if not completed(task_dir(ROOT, suite, index, name))]
                    if code == 0 and not remaining:
                        print(f"GPU {gpu} lane complete", flush=True)
                    elif retries[gpu] < 2:
                        retries[gpu] += 1
                        print(f"Retrying GPU {gpu} lane; exit {code}, remaining {remaining}", flush=True)
                        time.sleep(10)
                        active[gpu] = launch(gpu)
                    else:
                        raise RuntimeError(f"GPU {gpu} lane failed after retries: {remaining}")
            write_json(ROOT / "orchestrator.json", {"status": "complete", "pid": os.getpid(), **status()})
        except BaseException as error:
            write_json(ROOT / "orchestrator.json", {"status": "failed", "pid": os.getpid(),
                        "error": repr(error), "active": {str(gpu): process.pid
                                                         for gpu, (process, _) in active.items()},
                        **status()})
            raise


if __name__ == "__main__":
    main()
