"""LIBERO workers: parallel physics, explicit reset, no implicit autoreset."""

from __future__ import annotations

import multiprocessing as mp
import traceback
from pathlib import Path

import numpy as np
from PIL import Image


def prepare_libero_paths(directory, repo):
    # Avoid LIBERO's interactive first-import prompt and global ~/.libero edits.
    import os

    import yaml

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    root = Path(repo) / "LIBERO/libero/libero"
    paths = {
        "benchmark_root": str(root),
        "bddl_files": str(root / "bddl_files"),
        "init_states": str(root / "init_files"),
        "assets": str(root / "assets"),
        "datasets": str(root.parent / "datasets"),
    }
    (directory / "config.yaml").write_text(yaml.safe_dump(paths))
    os.environ["LIBERO_CONFIG_PATH"] = str(directory.resolve())


def observation(raw):
    # Same 180-degree orientation and 8D state as the repo's LIBERO PI0 wrapper.
    # This doesn't use OpenPI or load a VLA.
    quat = np.array(raw["robot0_eef_quat"], dtype=float, copy=True)
    w = np.clip(quat[3], -1.0, 1.0)
    denominator = np.sqrt(max(0.0, 1 - w * w))
    axis_angle = np.zeros(3) if denominator < 1e-8 else quat[:3] * (2 * np.arccos(w) / denominator)
    state = np.concatenate([raw["robot0_eef_pos"], axis_angle, raw["robot0_gripper_qpos"]]).astype(np.float32)
    images = {}
    for key in ("agentview_image", "robot0_eye_in_hand_image"):
        image = np.ascontiguousarray(raw[key][::-1, ::-1])
        images[key] = np.array(Image.fromarray(image).resize((224, 224), Image.Resampling.BILINEAR))
    return {"state": state, **images}


def trajectory_telemetry(env, raw):
    """Project world-space EEF onto the exact rotated/resized observation image."""
    from robosuite.utils.camera_utils import get_camera_transform_matrix
    from robosuite import macros
    height,width = raw['agentview_image'].shape[:2]
    transform = get_camera_transform_matrix(env.env.sim,'agentview',height,width)
    position = np.asarray(raw['robot0_eef_pos'],dtype=float)
    point = transform @ np.r_[position,1.]
    if point[2] <= 0:
        raise RuntimeError('End effector lies behind the agent-view camera')
    u,v = point[:2]/point[2]
    # Projection uses OpenCV top-left coordinates. Raw OpenGL RGB has its Y
    # axis flipped; observation() then rotates that raw RGB by 180 degrees.
    u = width-1-u
    if macros.IMAGE_CONVENTION != 'opengl':
        v = height-1-v
    pixel = [(u+.5)*224/width-.5,(v+.5)*224/height-.5]
    return dict(eef_pixel=pixel,eef_position=position.tolist())


def _worker(connection, cfg, rank):
    env = None
    try:
        from libero.libero import benchmark
        from libero.libero.envs import OffScreenRenderEnv

        suite = benchmark.get_benchmark_dict()[cfg["suite"]]()
        states = np.asarray(suite.get_task_init_states(cfg["task_id"]))
        if len(states) < cfg["train_init_states"] + cfg.get("eval_unique_states", cfg["eval_episodes"]):
            raise ValueError("Not enough disjoint official initial states for train and eval")
        task = suite.get_task(cfg["task_id"])
        env = OffScreenRenderEnv(
            bddl_file_name=suite.get_task_bddl_file_path(cfg["task_id"]), camera_heights=256, camera_widths=256
        )
        env.seed(cfg["seed"] + rank)
        rng = np.random.default_rng(cfg["seed"] + rank)
        connection.send(
            {"ok": True, "result": {"task": task.name, "language": task.language, "num_init_states": len(states)}}
        )
        elapsed = 0
        while True:
            command, payload = connection.recv()
            if command == "close":
                break
            if command == "reset":
                index = int(rng.integers(cfg["train_init_states"])) if payload is None else int(payload)
                env.reset()
                raw = env.set_init_state(states[index])
                # Settle the object physics; these setup actions aren't learning steps.
                for _ in range(cfg["settle_steps"]):
                    raw, _, _, _ = env.step(np.array([0, 0, 0, 0, 0, 0, -1], dtype=np.float32))
                elapsed = 0
                if env.check_success():
                    raise RuntimeError(f"Task already succeeds after reset (state {index}); cannot evaluate learning")
                result = {"obs": observation(raw), "init_state_id": index}
            elif command == "step":
                raw, reward, done, info = env.step(np.asarray(payload, dtype=np.float32))
                elapsed += 1
                success = bool(env.check_success())
                if bool(done) != success or float(reward) != float(success):
                    raise RuntimeError("Unexpected LIBERO reward/done contract; refusing incorrect labels")
                result = {
                    "obs": observation(raw),
                    "sparse": float(success),
                    "terminated": success,
                    "truncated": elapsed >= cfg["horizon"] and not success,
                    "elapsed": elapsed,
                }
            else:
                raise ValueError(command)
            if cfg.get('capture_trajectory', False):
                result['telemetry'] = trajectory_telemetry(env, raw)
            connection.send({"ok": True, "result": result})
    except (EOFError, BrokenPipeError):
        pass
    except Exception:
        try:
            connection.send({"ok": False, "error": traceback.format_exc()})
        except (EOFError, BrokenPipeError):
            pass
    finally:
        if env is not None:
            env.close()
        connection.close()


class EnvPool:
    def __init__(self, cfg, count, rank_offset=0):
        self.processes, self.connections = [], []
        self.timeout = cfg["worker_timeout"]
        context = mp.get_context("spawn")
        try:
            for rank in range(count):
                parent, child = context.Pipe()
                process = context.Process(target=_worker, args=(child, cfg, rank + rank_offset), daemon=True)
                process.start()
                child.close()
                self.processes.append(process)
                self.connections.append(parent)
            self.metadata = [self._receive(c) for c in self.connections]
        except BaseException:
            self.close()
            raise

    def _receive(self, connection):
        if not connection.poll(self.timeout):
            raise TimeoutError("LIBERO worker timed out")
        response = connection.recv()
        if not response["ok"]:
            raise RuntimeError(response["error"])
        return response["result"]

    def call(self, commands):
        """commands: {worker index: (command, payload)}; send all before waiting."""
        for index, command in commands.items():
            self.connections[index].send(command)
        return {index: self._receive(self.connections[index]) for index in commands}

    def reset(self, indices=None):
        indices = {i: None for i in range(len(self.connections))} if indices is None else indices
        return self.call({i: ("reset", initial) for i, initial in indices.items()})

    def step(self, actions):
        return self.call({i: ("step", a) for i, a in enumerate(actions)})

    def close(self):
        for c in self.connections:
            try:
                c.send(("close", None))
            except (OSError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=3)
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
        for c in self.connections:
            c.close()
