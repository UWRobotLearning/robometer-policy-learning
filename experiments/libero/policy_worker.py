"""Isolated Python 3.11/JAX pi0 inference over a private Unix socket."""

from __future__ import annotations

import argparse
import os
from multiprocessing.connection import Listener

import jax
import jax.numpy as jnp
import numpy as np
from openpi.models import model as model_module
from openpi.policies import policy_config
from openpi.training import config


def infer_batch(policy, requests):
    """Stack fully transformed samples; unbatch before dataset-specific output transforms.

    In particular LiberoOutputs slices a single sample's [time, action] array.
    Calling it on [batch, time, action] would silently slice the time dimension.
    """
    inputs, noises = [], []
    for request in requests:
        images = request["images"]
        observation = {
            "observation/image": np.frombuffer(images["agentview_image"], dtype=np.uint8).reshape(224, 224, 3),
            "observation/wrist_image": np.frombuffer(images["robot0_eye_in_hand_image"], dtype=np.uint8).reshape(
                224, 224, 3
            ),
            "observation/state": np.asarray(request["observation"]["state"], dtype=np.float32),
            "prompt": request["prompt"],
        }
        inputs.append(policy._input_transform(observation))
        key = jax.random.key(request["sampling_seed"])
        _, sample_key = jax.random.split(key)
        expected = (1, policy.action_horizon, policy.action_dim)
        steering = request.get("noise")
        mode = request.get("noise_mode", "direct")
        if mode not in ("direct", "additive"):
            raise ValueError(f"Unknown noise mode: {mode}")
        if steering is None:
            noise = jax.random.normal(sample_key, expected)
        else:
            steering = np.asarray(steering, dtype=np.float32)
            if steering.ndim == 1:
                steering = np.broadcast_to(steering, expected[1:])
            if steering.ndim == 2:
                steering = steering[None, ...]
            if steering.shape != expected or not np.isfinite(steering).all():
                raise ValueError(f"Invalid batched row noise: {steering.shape}, expected {expected}")
            noise = jnp.asarray(steering)
            if mode == "additive":
                noise = noise + jax.random.normal(sample_key, expected)
        noises.append(noise[0])
    batch = jax.tree.map(lambda *rows: jnp.stack(rows), *inputs)
    actions = np.asarray(
        policy._sample_actions(
            model_module.Observation.from_dict(batch), noise=jnp.stack(noises), **policy._sample_kwargs
        )
    )  # Host transfer synchronizes JAX before returning timings to the client.
    states = np.asarray(batch["state"])
    output = np.stack(
        [policy._output_transform({"state": states[i], "actions": actions[i]})["actions"] for i in range(len(requests))]
    ).astype(np.float32)
    expected = (len(requests), policy.action_horizon, 7)
    if output.shape != expected or not np.isfinite(output).all():
        raise ValueError(f"Invalid batched policy output: {output.shape}, expected {expected}")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--config", default="pi0_libero")
    args = parser.parse_args()
    policy = policy_config.create_trained_policy(config.get_config(args.config), args.checkpoint)
    policy._rng = jax.random.key(args.seed)
    print(f"pi0 ready: horizon={policy.action_horizon}, dim={policy.action_dim}", flush=True)
    with Listener(args.socket, family="AF_UNIX", authkey=bytes.fromhex(os.environ["AUDIT_PI0_RPC_KEY"])) as listener:
        with listener.accept() as connection:
            connection.send({"action_horizon": policy.action_horizon, "action_dim": policy.action_dim})
            while True:
                request = connection.recv()
                if request is None:
                    break
                if "batch" in request:
                    connection.send(infer_batch(policy, request["batch"]).tolist())
                    continue
                if "seed" in request:
                    policy._rng = jax.random.key(int(request["seed"]))
                    connection.send(True)
                    continue
                observation = request["observation"]
                images = request["images"]
                inputs = {
                    "observation/image": np.frombuffer(images["agentview_image"], dtype=np.uint8).reshape(224, 224, 3),
                    "observation/wrist_image": np.frombuffer(
                        images["robot0_eye_in_hand_image"], dtype=np.uint8
                    ).reshape(224, 224, 3),
                    "observation/state": np.asarray(observation["state"], dtype=np.float32),
                    "prompt": request["prompt"],
                }
                steering = request.get("noise")
                noise_mode = request.get("noise_mode", "additive")
                if noise_mode not in ("additive", "direct"):
                    raise ValueError(f"Unknown noise mode: {noise_mode}")
                noise = None
                if steering is not None:
                    steering = np.asarray(steering, dtype=np.float32)
                    if steering.ndim == 1:
                        steering = np.broadcast_to(steering, (policy.action_horizon, policy.action_dim))
                    if steering.ndim == 2:
                        steering = steering[None, ...]
                    expected = (1, policy.action_horizon, policy.action_dim)
                    if steering.shape != expected or not np.isfinite(steering).all():
                        raise ValueError(f"Steering shape {steering.shape}, expected {expected}")
                    if noise_mode == "direct":
                        # Original DSRL: the SAC action is the complete pi0 noise tensor.
                        noise = jnp.asarray(steering)
                    else:
                        # Residual steering used by the earlier task-55 comparison.
                        policy._rng, sample_key = jax.random.split(policy._rng)
                        noise = jax.random.normal(sample_key, expected) + jnp.asarray(steering)
                result = policy.infer(inputs, noise=noise)
                actions = np.asarray(result["actions"], dtype=np.float32)
                if actions.shape != (policy.action_horizon, 7) or not np.isfinite(actions).all():
                    raise ValueError(f"Invalid pi0 action chunk: {actions.shape}")
                connection.send(actions.tolist())


if __name__ == "__main__":
    main()
