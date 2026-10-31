"""One idle-GPU scheduler for every task, seed, policy, and reward variant."""

from __future__ import annotations

import fcntl
import json
import os
import shutil
import signal
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from experiments.reward_audit.config import REPO, task_map, write_json
from experiments.reward_audit.launch import python_check, reserve_gpus, source_pythonpath, stop_run

from .config import BASE_MODELS, plan, task_root


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def complete(job: dict) -> bool:
    folder = Path(job["path"])
    try:
        status = json.loads((folder / "status.json").read_text())
    except (OSError, ValueError):
        return False
    return (
        status.get("status") == "complete"
        and status.get("env_steps") == job["config"]["steps"]
        and (folder / "checkpoints/final.pt").is_file()
    )


def check_dependencies(spec: dict) -> None:
    runtime = spec["runtime"]
    checks = [(runtime["policy_python"], ["torch", "numpy", "gymnasium", "transformers", "libero", "robosuite"])]
    if spec["base_model"] != "scratch":
        checkpoint = Path(spec["checkpoint"])
        if not all((checkpoint / name).is_dir() for name in ("params", "assets")):
            raise RuntimeError(f"Incomplete OpenPI checkpoint: {checkpoint}")
        checks.append((runtime["openpi_python"], ["jax", "openpi"]))
    if "robometer" in spec["rewards"] or spec["audit_sparse_rewards"]:
        checks.append((runtime["reward_python"], ["torch", "robometer", "transformers", "peft", "unsloth"]))
    for interpreter, modules in checks:
        check = python_check(interpreter, modules)
        if check.get("error") or check.get("missing"):
            raise RuntimeError(f"Environment incomplete: {check}")


def initialize(spec: dict, out: Path, resume: bool) -> list[dict]:
    path = out / "experiment.json"
    if path.exists():
        if not resume:
            raise ValueError(f"Experiment already exists; use --resume to skip completed runs: {out}")
        previous = json.loads(path.read_text())
        changed = [key for key in spec if key not in {"runtime", "seeds"} and previous.get(key) != spec[key]]
        if changed or not set(previous["seeds"]) <= set(spec["seeds"]):
            raise ValueError(f"Resume requires the same experiment settings and a superset of seeds: {changed}")
    elif any(p.name != ".runner.lock" for p in out.iterdir()):
        raise ValueError(f"Refusing to overwrite a non-experiment directory: {out}")
    jobs = plan(spec, out)
    for job in jobs:
        if Path(job["path"]).exists() and not complete(job):
            raise ValueError(f"Incomplete run requires inspection; mid-run resume is not supported: {job['path']}")
    write_json(path, spec)
    for task_id in spec["tasks"]:
        root = task_root(spec, out, task_id)
        manifest_path = root / "manifest.json"
        created = json.loads(manifest_path.read_text()).get("created") if manifest_path.exists() else now()
        manifest = {
            "schema_version": 1,
            "suite": spec["suite"],
            "task_id": task_id,
            "task": task_map()[spec["suite"]][task_id],
            "base_model": spec["base_model"],
            "seeds": spec["seeds"],
            "rewards": spec["rewards"],
            "steps_per_run": spec["steps"],
            "num_envs": spec["num_envs"],
            "action_exec_len": spec["dsrl"].get("action_exec_len", 1),
            "algorithm": "SAC from scratch" if spec["base_model"] == "scratch" else "direct DSRL",
            "reward_checkpoint": spec["reward_checkpoint"],
            "horizon": spec["horizon"],
            "eval_episodes": spec["eval_episodes"],
            "eval_every": spec["eval_every"],
            "created": created,
            "planned_runs": [
                {"directory": str(Path(job["path"]).relative_to(root)), "config": job["config"]}
                for job in jobs
                if job["config"]["task_id"] == task_id
            ],
        }
        if spec["base_model"] != "scratch":
            manifest.update(pi0_checkpoint=spec["checkpoint"], policy_config_name=BASE_MODELS[spec["base_model"]])
        write_json(manifest_path, manifest)
    return jobs


def launch(job: dict, gpu: int, spec: dict):
    folder = Path(job["path"])
    cfg = dict(job["config"], policy_gpu=gpu)
    runtime = spec["runtime"]
    cfg["reward_gpu"] = runtime["reward_gpu"] if runtime["reward_gpu"] is not None else gpu
    folder.mkdir(parents=True, exist_ok=False)
    write_json(folder / "config.json", cfg)
    cache = Path(runtime["cache_dir"])
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": str(gpu),
        "MUJOCO_GL": "egl",
        "MUJOCO_EGL_DEVICE_ID": str(gpu),
        "EGL_DEVICE_ID": str(gpu),
        "AUDIT_LAUNCHER_GUARDED": "1",
        "PYTHONPATH": source_pythonpath(),
        "WANDB_MODE": "disabled",
        "WANDB_DISABLED": "true",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        "XDG_CACHE_HOME": str(cache),
        "HF_HOME": str(cache / "hf"),
        "TORCHINDUCTOR_CACHE_DIR": str(cache / "torchinductor"),
        "TRITON_CACHE_DIR": str(cache / "triton"),
        "TMPDIR": runtime["tmp_dir"],
    }
    with (folder / "console.log").open("w") as log:
        return subprocess.Popen(
            [
                runtime["policy_python"],
                "-u",
                "-m",
                "experiments.libero.train",
                "--config",
                str(folder / "config.json"),
                "--out",
                str(folder),
            ],
            cwd=REPO,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def prerequisites_ready(paths: list[Path]) -> bool:
    ready = True
    for path in paths:
        state = json.loads((path / "status.json").read_text())
        if state["status"] in {"failed", "canceled"}:
            raise RuntimeError(f"Prerequisite did not complete successfully: {path}")
        cfg = json.loads((path / "config.json").read_text())
        ready &= complete({"path": str(path), "config": cfg})
    return ready


def execute(
    spec: dict, out: Path, *, resume: bool = False, wait_for_gpus: bool = False, after_runs: list[Path] | None = None
) -> None:
    out = out.absolute()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        check_dependencies(spec)
        jobs = initialize(spec, out, resume)
        runtime = spec["runtime"]
        Path(runtime["tmp_dir"]).mkdir(parents=True, exist_ok=True)
        Path(runtime["cache_dir"]).mkdir(parents=True, exist_ok=True)
        pending = [job for job in jobs if not complete(job)]
        active = {}

        def save(state: str, **extra):
            for task_id in spec["tasks"]:
                selected = {
                    str(gpu): {
                        "pid": record["process"].pid,
                        "seed": record["job"]["config"]["seed"],
                        "kind": record["job"]["config"]["reward"],
                        "run": record["job"]["path"],
                    }
                    for gpu, record in active.items()
                    if record["job"]["config"]["task_id"] == task_id
                }
                queued = [
                    {"seed": job["config"]["seed"], "kind": job["config"]["reward"]}
                    for job in pending
                    if job["config"]["task_id"] == task_id
                ]
                task_state = state
                if state == "running":
                    task_state = "running" if selected else "queued" if queued else "complete"
                write_json(
                    task_root(spec, out, task_id) / "orchestrator.json",
                    {
                        "status": task_state,
                        "pid": os.getpid(),
                        "updated": now(),
                        "active": selected,
                        "pending": queued,
                        **extra,
                    },
                )

        def interrupted(signum, frame):
            raise KeyboardInterrupt

        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            while not prerequisites_ready(after_runs or []):
                save("queued", waiting_for_runs=[str(path) for path in after_runs])
                time.sleep(10)
            while pending or active:
                for gpu, record in list(active.items()):
                    code = record["process"].poll()
                    if code is None:
                        continue
                    record["reservation"].__exit__(None, None, None)
                    del active[gpu]
                    if code or not complete(record["job"]):
                        raise RuntimeError(f"Run failed (exit {code}); inspect {record['job']['path']}/console.log")
                    print(f"Completed {record['job']['path']}", flush=True)
                blocked = []
                for gpu in runtime["gpus"]:
                    if gpu in active or not pending:
                        continue
                    # Tasks are stages: all seeds of the first finish before the next task.
                    job = pending[0]
                    if active and any(
                        r["job"]["config"]["task_id"] != job["config"]["task_id"] for r in active.values()
                    ):
                        continue
                    if shutil.disk_usage(out).free < runtime["min_free_gib"] * 1024**3:
                        raise RuntimeError(f"Need {runtime['min_free_gib']} GiB free before starting another run")
                    indices = [gpu]
                    if job["config"]["reward"] == "robometer" or spec["audit_sparse_rewards"]:
                        if runtime["reward_gpu"] is not None:
                            indices.append(runtime["reward_gpu"])
                    reservation = reserve_gpus(indices)
                    try:
                        reservation.__enter__()
                    except RuntimeError as error:
                        blocked.append(str(error))
                        continue
                    try:
                        process = launch(job, gpu, spec)
                    except BaseException:
                        reservation.__exit__(None, None, None)
                        raise
                    pending.pop(0)
                    active[gpu] = {"job": job, "process": process, "reservation": reservation}
                    print(f"Started {job['path']} on GPU {gpu}: PID {process.pid}", flush=True)
                if pending and not active and not wait_for_gpus:
                    raise RuntimeError(f"No idle GPU available; use --wait-for-gpus to queue: {blocked}")
                save("running" if active else "waiting_for_gpus" if pending else "complete", gpu_wait_reasons=blocked)
                if pending or active:
                    time.sleep(10)
            save("complete")
        except BaseException as error:
            save("canceled" if isinstance(error, KeyboardInterrupt) else "failed", error=str(error))
            raise
        finally:
            for record in active.values():
                try:
                    stop_run(record["process"])
                finally:
                    record["reservation"].__exit__(None, None, None)
            signal.signal(signal.SIGTERM, previous_handler)
        from .analysis import generate

        generate(out)
