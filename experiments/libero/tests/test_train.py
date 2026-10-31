"""Check reward attribution before collecting expensive pi0 rollouts."""

import pytest
from experiments.libero.train import select_reward


def test_oracle_and_model_rewards_use_distinct_sources_on_the_same_chunk():
    sparse = [0.0, 0.0, 1.0]
    predicted = [0.1, 0.2, 0.3]
    assert select_reward("sparse", sparse, predicted) == 1.0
    assert select_reward("robometer", sparse, predicted) == pytest.approx(0.2)
    assert select_reward("robometer", sparse, [0.0, 0.0]) == 0.0


def test_model_reward_fails_on_missing_or_invalid_predictions():
    assert select_reward("sparse", [0.0, 0.0], None) == 0.0
    for values in (None, [], [float("nan")], [-0.01], [1.01]):
        with pytest.raises(ValueError):
            select_reward("robometer", [1.0], values)


@pytest.mark.parametrize("batched", [False, True])
@pytest.mark.parametrize("budget", [1, 17, 40])
def test_parallel_collection_respects_budget_done_and_reward_histories(budget, batched):
    import numpy as np
    from experiments.libero.train import execute_parallel_chunks

    class Policy:
        def reset_seed(self, seed):
            pass

        def infer(self, obs, prompt, **kwargs):
            return np.zeros((10, 7))

        def infer_batch(self, observations, prompt, noise, seeds, **kwargs):
            assert len(observations) == len(noise) == len(seeds)
            return np.zeros((len(observations), 10, 7))

    class Pool:
        def __init__(self):
            self.steps = [0] * 4
            self.calls = []

        def call(self, commands):
            self.calls.append(set(commands))
            response = {}
            for rank in commands:
                assert self.steps[rank] < (2 if rank == 0 else 10)
                self.steps[rank] += 1
                done = rank == 0 and self.steps[rank] == 2
                response[rank] = {
                    "obs": {"image": np.full((2, 2, 3), rank, dtype=np.uint8)},
                    "sparse": float(done),
                    "terminated": done,
                    "truncated": False,
                }
            return response

    cfg = {
        "dsrl": {"action_exec_len": 10, "batched_inference": batched},
        "seed": 100,
        "reward_camera": "image",
        "max_frames": 8,
    }
    env = Pool()
    observations = [{"image": np.full((2, 2, 3), rank, dtype=np.uint8)} for rank in range(4)]
    histories = [[obs["image"]] for obs in observations]
    records = execute_parallel_chunks(
        Policy(), env, observations, "task", np.zeros((4, 32)), histories, cfg, 0, budget, True
    )
    assert sum(len(r["sparse_steps"]) for r in records.values()) <= budget
    assert len(records[0]["sparse_steps"]) == min(2, budget)
    for rank, record in records.items():
        assert len(record["prefixes"]) == len(record["sparse_steps"])
        assert all(np.all(frame == rank) for prefix in record["prefixes"] for frame in prefix)
    if budget == 40:
        assert env.calls[0] == {0, 1, 2, 3}
        assert env.calls[2] == {1, 2, 3}
        assert sum(env.steps) == 32


def test_queue_requires_successful_complete_checkpoint(tmp_path):
    import json

    from experiments.libero.runner import prerequisites_ready

    (tmp_path / "config.json").write_text(json.dumps({"steps": 200}))
    (tmp_path / "status.json").write_text(json.dumps({"status": "running"}))
    assert not prerequisites_ready([tmp_path])
    (tmp_path / "status.json").write_text(json.dumps({"status": "complete", "env_steps": 200}))
    assert not prerequisites_ready([tmp_path])
    (tmp_path / "checkpoints").mkdir()
    (tmp_path / "checkpoints/final.pt").touch()
    assert prerequisites_ready([tmp_path])
    (tmp_path / "status.json").write_text(json.dumps({"status": "failed"}))
    with pytest.raises(RuntimeError):
        prerequisites_ready([tmp_path])
