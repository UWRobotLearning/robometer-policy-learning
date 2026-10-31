"""DSRL logging must preserve optimization and historical metric meanings."""

import copy
import json

import numpy as np
import pytest
from experiments.libero import diagnostics


def test_legacy_bootstrap_is_not_relabelled_as_bellman_target(tmp_path, monkeypatch):
    run = {
        "base_model": "pi0.5",
        "path": str(tmp_path),
        "config": {"reward": "robometer", "seed": 100},
        "curve": [{"env_steps": 0, "success_rate": 0.5}],
    }
    monkeypatch.setattr(diagnostics, "load_runs", lambda _: [run])
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps(
            {
                "prefix": "sac",
                "env_steps": 1000,
                "metrics": {"target_q_mean": 7, "actor_log_pis_mean": -2, "ent_coef": 0.1},
            }
        )
        + '\n{"incomplete":'
    )
    (tmp_path / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"phase": "eval", "env_steps": 0, "length": 200, "success": True},
                {"phase": "eval", "env_steps": 10000, "length": 300, "success": False},
                {"phase": "train", "env_steps": 1000, "length": 400, "success": False, "training_return": 20},
            ]
        )
    )
    series = diagnostics.read_diagnostics(tmp_path)["series"]
    identity = ("robometer", 100)
    assert series["bootstrap_q_mean"][identity] == [(1000.0, 7.0)]
    assert "bellman_target_mean" not in series
    assert series["entropy"][identity] == [(1000.0, 2.0)]
    assert series["eval_length"][identity] == [(0.0, 200.0)]  # No partial eval summaries.
    assert series["train_return_failure"][identity] == [(1000.0, 20.0)]
    assert "train_native_return" not in series
    assert "Not recorded" in diagnostics.render(tmp_path)


def test_vanilla_sac_excluded_from_dsrl_display(tmp_path, monkeypatch):
    monkeypatch.setattr(diagnostics, "load_runs", lambda _: [{"base_model": "scratch"}])
    assert diagnostics.read_diagnostics(tmp_path)["runs"] == []
    assert "steer a frozen pretrained policy" in diagnostics.render(tmp_path)


def test_dsrl_logging_preserves_updates_rng_and_records_terminal_targets():
    import torch
    from experiments.reward_audit.sac import Replay, make_sac

    torch.set_num_threads(1)
    replay = Replay(16, {"state": 3}, 100, action_dim=2)
    for _ in range(8):
        obs = {"state": np.ones((1, 3), dtype=np.float32)}
        replay.add(obs, np.zeros((1, 2), dtype=np.float32), np.array([0.75]), obs, np.array([True]), np.array([False]))
    cfg = {
        "actor_hidden": [8],
        "critic_hidden": [8],
        "sac": {
            "learning_starts": 0,
            "batch_size": 4,
            "num_critics": 2,
            "n_critics_to_sample": 2,
            "num_critic_updates_per_actor_update": 2,
            "num_updates_per_train_step": 1,
            "target_entropy": -1,
            "train_actor_with_entropy": True,
            "train_critic_with_entropy": False,
        },
    }
    original = make_sac(cfg, replay, torch.device("cpu"), action_dim=2)
    logged = copy.deepcopy(original)
    # Logging must not sample additional actions or change gradients.
    torch.manual_seed(91)
    np.random.seed(91)
    plain = original.train_step(collect_diagnostics=False)
    rng_after = torch.get_rng_state()
    torch.manual_seed(91)
    np.random.seed(91)
    metrics = logged.train_step(collect_diagnostics=True)
    assert torch.equal(rng_after, torch.get_rng_state())
    for module in ("actor", "critic", "critic_target"):
        for key, value in getattr(original, module).state_dict().items():
            torch.testing.assert_close(value, getattr(logged, module).state_dict()[key], rtol=0, atol=0)
    assert metrics["bellman_target_mean"] == pytest.approx(0.75)
    assert metrics["entropy"] == pytest.approx(-metrics["actor_log_pis_mean"])
    assert metrics["temperature"] == pytest.approx(metrics["ent_coef"])
    assert metrics["actor_updates"] == 1 and metrics["critic_updates"] == 2
    assert metrics["latent_std_norm"] > 0 and metrics["latent_std"] > 0
    assert all(np.isfinite(v) for v in metrics.values())
    assert "latent_std" not in plain


@pytest.mark.parametrize("env_count", [1, 4])
def test_diagnostics_scale_only_x_coordinates_before_binning(tmp_path, monkeypatch, env_count):
    run = {
        "base_model": "pi0",
        "path": str(tmp_path),
        "config": {"reward": "sparse", "seed": 100, "num_envs": env_count},
        "curve": [{"env_steps": 40000, "success_rate": 0.8}],
    }
    monkeypatch.setattr(diagnostics, "load_runs", lambda _: [run])
    (tmp_path / "metrics.jsonl").write_text(
        json.dumps(
            {
                "prefix": "sac",
                "env_steps": 40000,
                "metrics": {"temperature": 0.2},
            }
        )
    )
    (tmp_path / "episodes.jsonl").write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {"phase": "train", "env_steps": 40000, "length": 600, "success": False, "training_return": 7},
                {"phase": "eval", "env_steps": 40000, "length": 500},
                {"phase": "eval", "env_steps": 80000, "length": 100},  # Incomplete eval remains excluded.
            ]
        )
    )
    series = diagnostics.read_diagnostics(tmp_path)["series"]
    identity = ("sparse", 100)
    step = 40000 / env_count
    assert series["temperature"][identity] == [(step, 0.2)]
    assert series["train_length"][identity] == [(step, 600)]
    assert series["train_return"][identity] == [(step, 7)]
    assert series["eval_length"][identity] == [(step, 500)]
    assert series["eval_success"][identity] == [(step, 0.8)]
    page = diagnostics.render(tmp_path)
    assert "physical steps per environment (average)" in page
    assert f"{step / 1000:g}k physical steps/env" in page
