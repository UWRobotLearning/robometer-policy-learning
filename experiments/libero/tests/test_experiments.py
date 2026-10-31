"""CPU checks of model profiles, scheduling, resume, and paired analysis."""

import builtins
import copy
import json
import sys
from contextlib import contextmanager
from pathlib import Path

import pytest
from experiments.libero import analysis, runner
from experiments.libero.config import ROBOMETER, plan, resolve, run_config, task_root
from experiments.reward_audit.config import write_json


@pytest.mark.parametrize(
    "base,starts,critic_updates,chunk",
    [
        ("scratch", 5000, 1, None),
        ("pi0", 500, 25, 20),
        ("pi0.5", 500, 25, 10),
    ],
)
def test_model_profiles_and_reward_pairs_are_independent_of_prior_runs(tmp_path, base, starts, critic_updates, chunk):
    spec = resolve({"base_model": base, "runtime": {"data_root": str(tmp_path)}})
    sparse = run_config(spec, 28, 100, "sparse", 0)
    model = run_config(spec, 28, 100, "robometer", 0)
    assert sparse["sac"]["learning_starts"] == starts
    assert sparse["sac"]["num_critic_updates_per_actor_update"] == critic_updates
    assert sparse["sac"] == model["sac"]
    assert sparse["reward_checkpoint"] == model["reward_checkpoint"] == ROBOMETER
    assert len(plan(spec, tmp_path)) == 6
    if chunk:
        assert model["dsrl"]["action_exec_len"] == chunk
        assert model["dsrl"]["noise_mode"] == "direct"
        assert model["pi0_checkpoint"].startswith(str(tmp_path))
        assert model["reward_formula"] == "progress_chunk_mean"
        assert sparse["reward_formula"] == "sparse_chunk_terminal"
    else:
        assert "pi0_checkpoint" not in model
        assert sparse["sparse_offset"] == -1


@pytest.mark.parametrize(
    "change",
    [
        {"tasks": [999]},
        {"seeds": [100, 100]},
        {"rewards": ["oracle"]},
        {"base_model": "pi05"},
        {"base_model": "pi0", "num_envs": 0},
        {"steps": 0},
        {"unexpected": True},
        {"runtime": {"gpus": [1], "reward_gpu": 1}},
    ],
)
def test_invalid_plans_are_rejected_before_ml_imports(change):
    with pytest.raises(ValueError):
        resolve(change)


def test_plan_cli_never_imports_ml_or_creates_run_directories(tmp_path, monkeypatch, capsys):
    from experiments.libero.__main__ import main

    spec = tmp_path / "experiment.json"
    spec.write_text(json.dumps({"runtime": {"data_root": str(tmp_path)}}))
    real_import = builtins.__import__

    def no_ml(name, *args, **kwargs):
        if name.split(".")[0] in {"torch", "jax", "openpi", "libero", "transformers"}:
            raise AssertionError(f"Plan imported ML: {name}")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_ml)
    monkeypatch.setattr(sys, "argv", ["libero", "plan", str(spec)])
    main()
    planned = json.loads(capsys.readouterr().out)
    assert len(planned["runs"]) == 6
    assert not Path(planned["out"]).exists()


def finish(job):
    folder = Path(job["path"])
    folder.mkdir(parents=True)
    write_json(folder / "config.json", job["config"])
    write_json(folder / "status.json", {"status": "complete", "env_steps": job["config"]["steps"]})
    (folder / "checkpoints").mkdir()
    (folder / "checkpoints/final.pt").write_bytes(b"test checkpoint")


def test_resume_skips_complete_runs_and_allows_only_additional_seeds(tmp_path):
    spec = resolve({"base_model": "scratch", "seeds": [100], "runtime": {"data_root": str(tmp_path)}})
    jobs = runner.initialize(spec, tmp_path, False)
    for job in jobs:
        finish(job)
    expanded = copy.deepcopy(spec)
    expanded["seeds"].append(101)
    planned = runner.initialize(expanded, tmp_path, True)
    assert [job["config"]["seed"] for job in planned if not runner.complete(job)] == [101, 101]
    invalid = copy.deepcopy(expanded)
    invalid["steps"] += 100
    with pytest.raises(ValueError, match="same experiment settings"):
        runner.initialize(invalid, tmp_path, True)


def test_incomplete_runs_are_never_silently_overwritten(tmp_path):
    spec = resolve({"base_model": "scratch", "runtime": {"data_root": str(tmp_path)}})
    jobs = runner.initialize(spec, tmp_path, False)
    Path(jobs[0]["path"]).mkdir(parents=True)
    with pytest.raises(ValueError, match="Incomplete run"):
        runner.initialize(spec, tmp_path, True)


def test_scheduler_uses_idle_gpu_and_finishes_all_tasks_before_advancing(tmp_path, monkeypatch):
    spec = resolve(
        {
            "base_model": "scratch",
            "tasks": [2, 28],
            "seeds": [100],
            "runtime": {"data_root": str(tmp_path), "min_free_gib": 0},
        }
    )
    started = []

    @contextmanager
    def reserve(indices):
        if 0 in indices:
            raise RuntimeError("GPU 0 belongs to another job")
        yield

    class Process:
        pid = 123

        def poll(self):
            return 0

    def launch(job, gpu, spec):
        started.append((job["config"]["task_id"], gpu))
        finish(job)
        return Process()

    monkeypatch.setattr(runner, "check_dependencies", lambda _: None)
    monkeypatch.setattr(runner, "reserve_gpus", reserve)
    monkeypatch.setattr(runner, "launch", launch)
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    monkeypatch.setattr(analysis, "generate", lambda _: None)
    out = tmp_path / "experiment"
    runner.execute(spec, out)
    assert started == [(2, 1), (2, 1), (28, 1), (28, 1)]
    assert all(runner.complete(job) for job in plan(spec, out))
    for task in spec["tasks"]:
        assert json.loads((task_root(spec, out, task) / "orchestrator.json").read_text())["status"] == "complete"


def test_scheduler_stops_its_active_jobs_and_leaves_queue_unlaunched_on_failure(tmp_path, monkeypatch):
    spec = resolve({"base_model": "scratch", "runtime": {"data_root": str(tmp_path), "min_free_gib": 0}})
    launched, stopped = [], []

    @contextmanager
    def reserve(indices):
        yield

    class Process:
        def __init__(self, pid):
            self.pid = pid

        def poll(self):
            return 1 if self.pid == 1 else None

    def launch(job, gpu, spec):
        launched.append(job)
        return Process(len(launched))

    monkeypatch.setattr(runner, "check_dependencies", lambda _: None)
    monkeypatch.setattr(runner, "reserve_gpus", reserve)
    monkeypatch.setattr(runner, "launch", launch)
    monkeypatch.setattr(runner, "stop_run", lambda process: stopped.append(process.pid))
    monkeypatch.setattr(runner.time, "sleep", lambda _: None)
    with pytest.raises(RuntimeError, match="Run failed"):
        runner.execute(spec, tmp_path / "experiment")
    assert len(launched) == 2
    assert stopped == [2]


def test_analysis_reads_nested_results_and_excludes_incomplete_final_pairs(tmp_path):
    spec = resolve({"base_model": "pi0.5", "steps": 20, "eval_every": 10, "runtime": {"data_root": str(tmp_path)}})
    jobs = runner.initialize(spec, tmp_path, False)
    for job in jobs:
        finish(job)
        cfg = job["config"]
        folder = Path(job["path"])
        incomplete = cfg["seed"] == 102 and cfg["reward"] == "robometer"
        text = "env_steps,successes,episodes,success_rate\n0,10,25,0.4\n10,15,25,0.6\n"
        if not incomplete:
            text += "20,20,25,0.8\n"
        else:
            write_json(folder / "status.json", {"status": "running"})
        (folder / "evaluation.csv").write_text(text)
    runs = analysis.load_runs(tmp_path)
    stats = analysis.summarize_group(runs)
    assert stats["baseline"]["mean"] == 0.4
    assert stats["baseline"]["n"] == 3
    assert stats["completed_paired_seeds"] == [100, 101]
    assert stats["curves"]["robometer"][-1]["n"] == 2
    assert stats["paired_final_sparse_minus_robometer"]["n"] == 2
    runs[-1]["config"]["horizon"] += 1
    with pytest.raises(ValueError, match="Unmatched experiment settings"):
        analysis.summarize_group(runs)


def test_analysis_aligns_physical_chunk_overshoot_without_pairing_missing_evals():
    run = {
        "path": "test",
        "config": {"steps": 100, "eval_every": 30},
        "curve": [{"env_steps": step} for step in (0, 32, 63, 100)],
    }
    assert list(analysis.checkpoints(run)) == [0, 30, 60, 100]


def test_parallel_dsrl_preserves_chunk_and_update_settings(tmp_path):
    spec = resolve({"base_model": "pi0.5", "num_envs": 4, "steps": 200000, "tasks": [29], "seeds": [100]})
    jobs = plan(spec, tmp_path)
    assert len(jobs) == 2
    for job in jobs:
        cfg = job["config"]
        assert cfg["num_envs"] == 4 and cfg["steps"] == 200000
        assert cfg["dsrl"]["action_exec_len"] == 10
        assert cfg["sac"]["num_updates_per_train_step"] == 1
        assert cfg["sac"]["num_critic_updates_per_actor_update"] == 25
