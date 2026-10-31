"""Run with the OpenPI interpreter on CPU; no checkpoint is needed."""

import importlib.util
import unittest

import numpy as np


@unittest.skipUnless(
    importlib.util.find_spec("jax") and importlib.util.find_spec("openpi"), "OpenPI environment required"
)
class BatchWorkerTests(unittest.TestCase):
    def setUp(self):
        from experiments.libero.policy_worker import infer_batch

        self.infer = infer_batch

        class Policy:
            action_horizon = 9
            action_dim = 32

            def __init__(self):
                self._sample_kwargs = {}

            @staticmethod
            def _input_transform(obs):
                return {
                    "state": np.pad(obs["observation/state"], (0, 24)),
                    "image": {"base": obs["observation/image"]},
                    "image_mask": {"base": np.True_},
                }

            @staticmethod
            def _sample_actions(obs, noise):
                return noise + obs.state[:, None, :]

            @staticmethod
            def _output_transform(output):
                # Dataset output transforms expect [time, action], not [batch, time, action].
                return {"actions": output["actions"][:, :7]}

        self.policy = Policy()
        self.requests = [
            {
                "images": {
                    key: np.zeros((224, 224, 3), np.uint8).tobytes()
                    for key in ("agentview_image", "robot0_eye_in_hand_image")
                },
                "observation": {"state": [rank] * 8},
                "prompt": "task",
                "sampling_seed": 100 + rank,
                "noise_mode": "direct",
                "noise": [0.01 * rank] * 32,
            }
            for rank in range(4)
        ]

    def test_matches_single_rows_and_permutation_for_all_noise_modes(self):
        for mode in ("direct", "additive", "frozen"):
            requests = [
                dict(
                    row,
                    noise_mode="direct" if mode == "frozen" else mode,
                    noise=None if mode == "frozen" else row["noise"],
                )
                for row in self.requests
            ]
            batch = self.infer(self.policy, requests)
            single = np.stack([self.infer(self.policy, [row])[0] for row in requests])
            self.assertEqual(batch.shape, (4, 9, 7))
            np.testing.assert_array_equal(batch, single)
            np.testing.assert_array_equal(self.infer(self.policy, requests[::-1]), batch[::-1])

    def test_direct_noise_values_and_partial_batch(self):
        actions = self.infer(self.policy, self.requests[:2])
        np.testing.assert_allclose(actions[0], 0)
        np.testing.assert_allclose(actions[1], 1.01)
        self.assertEqual(actions.shape, (2, 9, 7))

    def test_bad_noise_is_rejected(self):
        for noise in ([float("nan")] * 32, [0] * 7):
            with self.assertRaises(ValueError):
                self.infer(self.policy, [dict(self.requests[0], noise=noise)])


if __name__ == "__main__":
    unittest.main()
