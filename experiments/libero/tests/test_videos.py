"""Outcome selection and simulator labels in the reusable recorder."""

import json
from pathlib import Path

import numpy as np
import pytest
from experiments.libero.videos import build_jobs, collect, collect_demos, render_report, select_episodes, worker


def test_selection_is_seeded_and_never_invents_missing_outcomes():
    records = [{"episode": i, "success": i < 10} for i in range(25)]
    chosen = select_episodes(records, 2, 2, 12)
    assert len(chosen) == 4
    assert sum(row["success"] for row in chosen) == 2
    assert chosen == select_episodes(list(reversed(records)), 2, 2, 12)
    assert len(select_episodes(records[:3], 2, 2, 12)) == 2
    assert select_episodes([], 2, 2, 12) == []


def test_internal_workers_require_the_gpu_guarded_launcher(monkeypatch):
    monkeypatch.delenv("AUDIT_VIDEO_LAUNCHER_GUARDED", raising=False)
    with pytest.raises(RuntimeError, match="video-audit"):
        worker({}, {}, 0)


def test_frozen_audit_accepts_one_task_config_without_training_results(tmp_path):
    path = tmp_path / "experiment.json"
    path.write_text(json.dumps({"base_model": "pi0.5", "tasks": [29], "runtime": {"data_root": str(tmp_path)}}))
    jobs = build_jobs(path, tmp_path / "audit", 100, 10)
    assert len(jobs) == 1 and jobs[0]["variant"] == "pretrained"
    assert jobs[0]["actor_checkpoint"] is None
    assert jobs[0]["config"]["task_id"] == 29
    assert jobs[0]["config"]["eval_episodes"] == 10


def test_demo_extraction_preserves_labels_and_rejects_wrong_task_prompt(tmp_path, monkeypatch):
    import h5py
    import imageio.v3 as iio
    from experiments.reward_audit.config import task_map

    source = tmp_path / "demos.hdf5"
    prompt = "put the black bowl in the top drawer of the cabinet"
    with h5py.File(source, "w") as f:
        data = f.create_group("data")
        data.attrs["problem_info"] = json.dumps({"language_instruction": prompt})
        for index in range(4):
            demo = data.create_group(f"demo_{index}")
            demo.create_dataset("actions", data=np.zeros((3, 7)))
            demo.create_dataset("rewards", data=[0, 0, 1])
            demo.create_dataset("dones", data=[0, 0, 1])
            demo.create_dataset("obs/agentview_rgb", data=np.zeros((3, 4, 4, 3), dtype=np.uint8))
    folder = tmp_path / "rendered"
    for name in ("videos", "frames"):
        (folder / name).mkdir(parents=True)
    monkeypatch.setattr(iio, "imwrite", lambda path, frames, fps: Path(path).write_bytes(b"video"))
    cfg = {"suite": "libero_90", "task": task_map()["libero_90"][29], "reward_camera": "agentview_image"}
    job = {"config": cfg, "demo_file": str(source), "episodes": 4}
    records = collect_demos(job, folder)
    assert len(records) == 4 and all(row["success"] for row in records)
    with np.load(folder / "frames/episode_000.npz") as frames:
        assert frames["rewards"].tolist() == [0, 0, 1]
    with h5py.File(source, "r+") as f:
        f["data"].attrs["problem_info"] = json.dumps({"language_instruction": "close the top drawer of the cabinet"})
    with pytest.raises(ValueError, match="instruction does not match"):
        collect_demos(job, folder)


def test_report_uses_all_episodes_for_success_rate_not_the_balanced_subset(tmp_path):
    cfg = {
        "suite": "libero_goal",
        "task_id": 7,
        "task": "turn_on_the_stove",
        "horizon": 400,
        "reward_checkpoint": "robometer/Robometer-4B@pinned",
    }
    manifest = {
        "seed": 100,
        "episodes": 25,
        "successes": 2,
        "failures": 2,
        "selection_seed": 0,
        "jobs": [{"variant": "sparse", "config": cfg, "actor_checkpoint": "/actor.pt"}],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    folder = tmp_path / "sparse"
    folder.mkdir()
    records = [
        {"episode": i, "init_state_id": 25 + i, "success": i < 16, "length": 400, "video": f"episode_{i:03d}.mp4"}
        for i in range(25)
    ]
    (folder / "episodes.json").write_text(json.dumps(records))
    (folder / "selected.json").write_text(json.dumps(select_episodes(records, 2, 2, 0)))
    page = render_report(tmp_path).read_text()
    assert "16/25 simulator successes" in page
    assert page.count("<video ") == 4
    assert "95% Wilson CI" in page


def test_recorder_uses_matched_states_and_stops_on_native_terminal_or_timeout(tmp_path, monkeypatch):
    import imageio.v3 as iio
    from experiments.libero import policy
    from experiments.reward_audit import envs

    seeds = []

    class Policy:
        def __init__(self, *args, **kwargs):
            pass

        def reset_seed(self, seed):
            seeds.append(seed)

        def infer(self, observation, prompt, noise, noise_mode):
            assert noise is None
            return np.zeros((10, 7))

        def close(self):
            pass

    def obs():
        return {"agentview_image": np.zeros((4, 4, 3), dtype=np.uint8)}

    class Env:
        def __init__(self, cfg, count, rank_offset):
            assert rank_offset == 100000
            self.metadata = [{"language": "close the drawer"}]

        def reset(self, states):
            self.state = states[0]
            self.elapsed = 0
            return {0: {"obs": obs()}}

        def step(self, actions):
            self.elapsed += 1
            success = self.state == 26 and self.elapsed == 3
            return {
                0: {
                    "obs": obs(),
                    "elapsed": self.elapsed,
                    "sparse": float(success),
                    "terminated": success,
                    "truncated": self.elapsed == 4,
                }
            }

        def close(self):
            pass

    monkeypatch.setattr(policy, "Pi0Client", Policy)
    monkeypatch.setattr(envs, "EnvPool", Env)
    monkeypatch.setattr(envs, "prepare_libero_paths", lambda *args: None)
    monkeypatch.setattr(iio, "imwrite", lambda path, frames, fps: Path(path).write_bytes(b"video"))
    for directory in ("videos", "frames"):
        (tmp_path / directory).mkdir()
    cfg = {
        "seed": 100,
        "torch_threads": 1,
        "pi0_python": "python",
        "pi0_checkpoint": "/model",
        "reward_camera": "agentview_image",
        "train_init_states": 25,
        "eval_episodes": 25,
        "dsrl": {"action_exec_len": 10},
    }
    records = collect(
        {"base_model": "pi0.5", "actor_checkpoint": None, "config": cfg, "variant": "pretrained", "episodes": 2},
        tmp_path,
        0,
    )
    assert [row["init_state_id"] for row in records] == [25, 26]
    assert [row["success"] for row in records] == [False, True]
    assert [row["length"] for row in records] == [4, 3]
    assert seeds == [10000025, 10000026]
    with np.load(tmp_path / "frames/episode_000.npz") as data:
        assert data["rewards"].sum() == 0
    with np.load(tmp_path / "frames/episode_001.npz") as data:
        assert data["rewards"].tolist() == [0.0, 0.0, 0.0, 1.0]
