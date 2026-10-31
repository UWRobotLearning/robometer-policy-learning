"""OpenPI client shared by pi0, pi0.5, evaluations, and video audits."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
import time
from multiprocessing.connection import Client
from pathlib import Path

import numpy as np

from experiments.reward_audit.config import REPO
from experiments.reward_audit.launch import source_pythonpath
from experiments.reward_audit.metrics import wilson


class Pi0Client:
    def __init__(self, interpreter, checkpoint, seed, gpu, directory, config_name="pi0_libero"):
        self.process = None
        self.connection = None
        self.temp = tempfile.TemporaryDirectory(prefix="pi0-audit-")
        self.socket = str(Path(self.temp.name) / "policy.sock")
        if len(self.socket.encode()) >= 107:
            raise ValueError("Pi0 RPC socket path too long; choose a shorter TMPDIR")
        env = {
            **os.environ,
            "CUDA_VISIBLE_DEVICES": str(gpu),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "WANDB_MODE": "disabled",
            "WANDB_DISABLED": "true",
            "AUDIT_PI0_RPC_KEY": os.urandom(32).hex(),
            "PYTHONPATH": source_pythonpath(),
        }
        self.log = (directory / "policy_worker.log").open("w")
        try:
            self.process = subprocess.Popen(
                [
                    str(interpreter),
                    "-u",
                    "-m",
                    "experiments.libero.policy_worker",
                    "--socket",
                    self.socket,
                    "--checkpoint",
                    str(checkpoint),
                    "--seed",
                    str(seed),
                    "--config",
                    config_name,
                ],
                env=env,
                cwd=REPO,
                stdout=self.log,
                stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + 1800
            while not Path(self.socket).exists():
                if self.process.poll() is not None:
                    raise RuntimeError(f"Pi0 worker failed; see {directory / 'policy_worker.log'}")
                if time.monotonic() > deadline:
                    raise TimeoutError("Pi0 worker did not initialize")
                time.sleep(0.2)
            self.connection = Client(self.socket, family="AF_UNIX", authkey=bytes.fromhex(env["AUDIT_PI0_RPC_KEY"]))
            self.metadata = self.connection.recv()
        except BaseException:
            self.close()
            raise

    def infer(self, observation, prompt, noise=None, noise_mode="additive"):
        if not self.connection:
            raise RuntimeError("Pi0 worker is closed")
        images = {
            k: np.ascontiguousarray(observation[k], dtype=np.uint8).tobytes()
            for k in ("agentview_image", "robot0_eye_in_hand_image")
        }
        self.connection.send(
            {
                "observation": {"state": observation["state"].tolist()},
                "images": images,
                "prompt": prompt,
                "noise": None if noise is None else np.asarray(noise, dtype=np.float32).tolist(),
                "noise_mode": noise_mode,
            }
        )
        if not self.connection.poll(600):
            raise TimeoutError("Pi0 inference timed out")
        actions = np.asarray(self.connection.recv(), dtype=np.float32)
        expected = (self.metadata["action_horizon"], 7)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Invalid Pi0 response: {actions.shape}")
        return actions

    def reset_seed(self, seed):
        self.connection.send({"seed": int(seed)})
        if not self.connection.poll(30) or self.connection.recv() is not True:
            raise RuntimeError("Pi0 worker did not reset its sample RNG")

    def infer_batch(self, observations, prompt, noise=None, noise_mode="direct", seeds=None):
        """One model forward for independent environments, with explicit per-row RNG seeds."""
        count = len(observations)
        if not count or seeds is None or len(seeds) != count:
            raise ValueError("Batched inference requires one seed per nonempty observation row")
        if noise is not None and len(noise) != count:
            raise ValueError("Batched noise must have one entry per observation")
        if not self.connection:
            raise RuntimeError("Pi0 worker is closed")
        requests = []
        for index, observation in enumerate(observations):
            requests.append(
                {
                    "observation": {"state": observation["state"].tolist()},
                    "images": {
                        key: np.ascontiguousarray(observation[key], dtype=np.uint8).tobytes()
                        for key in ("agentview_image", "robot0_eye_in_hand_image")
                    },
                    "prompt": prompt,
                    "sampling_seed": int(seeds[index]),
                    "noise_mode": noise_mode,
                    "noise": None if noise is None else np.asarray(noise[index], dtype=np.float32).tolist(),
                }
            )
        self.connection.send({"batch": requests})
        if not self.connection.poll(600):
            raise TimeoutError("Batched Pi0 inference timed out")
        actions = np.asarray(self.connection.recv(), dtype=np.float32)
        expected = (count, self.metadata["action_horizon"], 7)
        if actions.shape != expected or not np.isfinite(actions).all():
            raise ValueError(f"Invalid batched Pi0 response: {actions.shape}, expected {expected}")
        return actions

    def close(self):
        if self.connection:
            try:
                self.connection.send(None)
            except (EOFError, OSError):
                pass
            self.connection.close()
            self.connection = None
        if self.process:
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait(timeout=30)
            self.process = None
        self.log.close()
        self.temp.cleanup()


def evaluate(policy, env, cfg, directory, action_exec_len):
    language = env.metadata[0]["language"]
    records = []
    with (directory / "episodes.jsonl").open("w", buffering=1) as stream:
        for index in range(cfg["train_init_states"], cfg["train_init_states"] + cfg["eval_episodes"]):
            observation = env.reset({0: index})[0]["obs"]
            actions_generated = 0
            while True:
                actions = policy.infer(observation, language)
                actions_generated += 1
                finished = False
                for action in actions[:action_exec_len]:
                    result = env.step([action])[0]
                    observation = result["obs"]
                    if result["terminated"] or result["truncated"]:
                        finished = True
                        break
                if finished:
                    record = {
                        "init_state_id": index,
                        "success": bool(result["terminated"]),
                        "length": result["elapsed"],
                        "chunks": actions_generated,
                    }
                    records.append(record)
                    stream.write(json.dumps(record) + "\n")
                    print(
                        f"State {index}: {'success' if record['success'] else 'failure'} "
                        f"after {record['length']} steps",
                        flush=True,
                    )
                    break
    successes = sum(record["success"] for record in records)
    return {
        "successes": successes,
        "episodes": len(records),
        "success_rate": successes / len(records),
        "wilson_ci95": wilson(successes, len(records)),
        "records": records,
    }
