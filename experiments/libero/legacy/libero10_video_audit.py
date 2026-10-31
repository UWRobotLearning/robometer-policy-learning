"""Record four LIBERO-10 demos and four pi0 successes/failures, then score every video."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from huggingface_hub import hf_hub_download

from experiments.libero.policy import Pi0Client
from experiments.libero.legacy.task38_video_audit import FPS, SCORE_EVERY, VIDEO_EVERY, scoring_indices
from experiments.reward_audit.config import REPO, task_map, write_json
from experiments.reward_audit.envs import EnvPool, prepare_libero_paths
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath
from experiments.reward_audit.metrics import prefix_frames
from experiments.reward_audit.rpc import RewardClient


DATA_ROOT = Path("/data/rohan/Reward-Modeling")
ROOT = DATA_ROOT / "pi0_libero10_video_audit_20261015"
SOURCE_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
SOURCE_REPO = "yifengzhu-hf/LIBERO-datasets"
SOURCE_CONFIG = (DATA_ROOT / "pi0_libero90_task38_dsrl_3seeds_20261012/seed100/"
                 "robometer_seed100/config.json")
BASELINE_ROOT = DATA_ROOT / "pi0_libero_families_20261014/libero_10"
TASK_IDS = (6, 8)
TARGET_PER_OUTCOME = 4
EVAL_STATES = tuple(range(25, 50))


def task_root(task_id: int) -> Path:
    return ROOT / f"task{task_id:02d}"


def task_name(task_id: int) -> str:
    return task_map()["libero_10"][task_id]


def load_records(path: Path) -> list[dict]:
    return json.loads(path.read_text()) if path.exists() else []


def set_status(out: Path, phase: str) -> None:
    policy = load_records(out / "policy/episodes.json")
    demo_manifest = json.loads((out / "demos/manifest.json").read_text()) if (out / "demos/manifest.json").exists() else {}
    scores = list((out / "policy/scores").glob("state_*.json"))
    scores += list((out / "demos/scores").glob("demo_*.json"))
    write_json(out / "status.json", {"status": phase,
                "demos": len(demo_manifest.get("demonstrations", [])),
                "policy_successes": sum(bool(record["success"]) for record in policy),
                "policy_failures": sum(not record["success"] for record in policy),
                "videos_scored": len(scores), "videos_target": 12})


def config(task_id: int, gpu: int) -> dict:
    cfg = json.loads(SOURCE_CONFIG.read_text())
    cfg.update(suite="libero_10", task_id=task_id, task=task_name(task_id),
               name=f"libero10_task{task_id}_video_audit", reward="robometer",
               ood_claim="in_distribution", seed=100, train_init_states=25,
               eval_episodes=25, horizon=400, policy_gpu=gpu, reward_gpu=gpu)
    return cfg


def demo_source(task_id: int) -> tuple[Path, str]:
    filename = f"libero_10/{task_name(task_id)}_demo.hdf5"
    path = Path(hf_hub_download(repo_id=SOURCE_REPO, repo_type="dataset",
                                revision=SOURCE_REVISION, filename=filename,
                                local_dir=ROOT / "dataset_source"))
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return path, digest.hexdigest()


def render_demos(task_id: int, out: Path, source_path: Path, digest: str) -> list[dict]:
    demos = out / "demos"
    for name in ("videos", "frames", "scores"):
        (demos / name).mkdir(parents=True, exist_ok=True)
    records = []
    with h5py.File(source_path) as source:
        language = json.loads(source["data"].attrs["problem_info"])["language_instruction"]
        for demo_id in range(4):
            demo = source["data"][f"demo_{demo_id}"]
            length = len(demo["actions"])
            indices = list(range(0, length, VIDEO_EVERY))
            if indices[-1] != length - 1:
                indices.append(length - 1)
            video = demos / "videos" / f"demo_{demo_id}.mp4"
            frames_path = demos / "frames" / f"demo_{demo_id}.npz"
            if not video.exists() or not frames_path.exists():
                frames = np.stack([demo["obs"]["agentview_rgb"][index][::-1, ::-1].copy()
                                   for index in indices])
                iio.imwrite(video, frames, fps=FPS)
                np.savez_compressed(frames_path, frames=frames,
                                    steps=np.asarray(indices, dtype=np.int32))
            events = [{"step": int(index), "reward": float(demo["rewards"][index])}
                      for index in range(length) if demo["rewards"][index] != 0]
            records.append({"demo_id": demo_id, "length": length,
                            "success": bool(demo["dones"][-1]),
                            "reward_events": events, "prompt": language})
            print(f"Demo {demo_id}: {length} steps, success={records[-1]['success']}", flush=True)
    write_json(demos / "manifest.json", {"source": SOURCE_REPO, "revision": SOURCE_REVISION,
               "source_file": source_path.name, "sha256": digest, "prompt": language,
               "fps": FPS, "video_every_physical_steps": VIDEO_EVERY,
               "demonstrations": records})
    set_status(out, "demos_ready")
    return records


def candidate_states(task_id: int) -> list[int]:
    baseline = BASELINE_ROOT / f"{task_id:02d}_{task_name(task_id)}" / "summary.json"
    rows = json.loads(baseline.read_text())["records"]
    successes = [int(row["init_state_id"]) for row in rows if row["success"]]
    failures = [int(row["init_state_id"]) for row in rows if not row["success"]]
    prioritized = []
    for index in range(TARGET_PER_OUTCOME):
        prioritized.extend((successes[index], failures[index]))
    return prioritized + [state for state in EVAL_STATES if state not in prioritized] + list(range(25))


def collect_policy(cfg: dict, out: Path, gpu: int) -> list[dict]:
    policy_out = out / "policy"
    for name in ("videos", "frames", "scores"):
        (policy_out / name).mkdir(parents=True, exist_ok=True)
    records_path = policy_out / "episodes.json"
    attempts_path = policy_out / "attempts.json"
    selected = load_records(records_path)
    attempts = load_records(attempts_path)
    tried = {int(row["init_state_id"]) for row in attempts}
    count = lambda success: sum(bool(row["success"]) == success for row in selected)
    if count(True) >= TARGET_PER_OUTCOME and count(False) >= TARGET_PER_OUTCOME:
        return selected
    policy = env = None
    try:
        policy = Pi0Client(Path(cfg["pi0_python"]), Path(cfg["pi0_checkpoint"]), cfg["seed"], gpu, out)
        env = EnvPool(cfg, 1, rank_offset=100000)
        prompt = env.metadata[0]["language"]
        for state_id in candidate_states(cfg["task_id"]):
            if state_id in tried:
                continue
            policy.reset_seed(cfg["seed"] * 100000 + state_id)
            observation = env.reset({0: state_id})[0]["obs"]
            frames = [observation["agentview_image"].copy()]
            frame_steps = [0]
            chunks = 0
            while True:
                actions = policy.infer(observation, prompt)
                chunks += 1
                finished = False
                for action in actions[:cfg["dsrl"]["action_exec_len"]]:
                    result = env.step([action])[0]
                    observation = result["obs"]
                    step = result["elapsed"]
                    finished = result["terminated"] or result["truncated"]
                    if step % VIDEO_EVERY == 0 or finished:
                        frames.append(observation["agentview_image"].copy())
                        frame_steps.append(step)
                    if finished:
                        break
                if finished:
                    break
            success = bool(result["terminated"])
            keep = count(success) < TARGET_PER_OUTCOME
            record = {"init_state_id": state_id, "success": success,
                      "length": int(result["elapsed"]), "chunks": chunks,
                      "prompt": prompt, "held_out_initial_state": state_id in EVAL_STATES}
            if keep:
                video = f"state_{state_id}.mp4"
                iio.imwrite(policy_out / "videos" / video, np.stack(frames), fps=FPS)
                np.savez_compressed(policy_out / "frames" / f"state_{state_id}.npz",
                                    frames=np.stack(frames),
                                    steps=np.asarray(frame_steps, dtype=np.int32))
                record["video"] = video
                selected.append(record)
                write_json(records_path, selected)
            attempts.append({**record, "selected": keep})
            write_json(attempts_path, attempts)
            set_status(out, "collecting_policy")
            print(f"State {state_id}: {'success' if success else 'failure'}, "
                  f"selected={keep}, totals={count(True)}/{count(False)}", flush=True)
            if count(True) >= TARGET_PER_OUTCOME and count(False) >= TARGET_PER_OUTCOME:
                break
    finally:
        if env is not None:
            env.close()
        if policy is not None:
            policy.close()
    if count(True) < TARGET_PER_OUTCOME or count(False) < TARGET_PER_OUTCOME:
        raise RuntimeError(f"Could not collect 4 successes and 4 failures: {count(True)}/{count(False)}")
    return selected


def score_videos(cfg: dict, out: Path, gpu: int, policy_records: list[dict], demos: list[dict]) -> None:
    jobs = [("policy", int(row["init_state_id"]), row["prompt"]) for row in policy_records]
    jobs += [("demos", int(row["demo_id"]), row["prompt"]) for row in demos]
    pending = []
    for kind, ident, prompt in jobs:
        prefix = "state" if kind == "policy" else "demo"
        path = out / kind / "scores" / f"{prefix}_{ident}.json"
        if not path.exists():
            pending.append((kind, ident, prompt, prefix, path))
    if not pending:
        return
    reward = RewardClient(cfg, out, gpu)
    try:
        for kind, ident, prompt, prefix, path in pending:
            with np.load(out / kind / "frames" / f"{prefix}_{ident}.npz") as source:
                frames, steps = source["frames"], source["steps"]
                indices = scoring_indices(steps)
                prefixes = [prefix_frames(frames[:index + 1], cfg["max_frames"]) for index in indices]
                predictions = reward.predict(prefixes, prompt)
                series = [{"step": int(steps[index]), "video_time": index / FPS,
                           "progress": float(prediction[0]),
                           "success_probability": float(prediction[1])}
                          for index, prediction in zip(indices, predictions, strict=True)]
            write_json(path, series)
            set_status(out, "scoring")
            print(f"Scored {kind} {ident}: progress={series[-1]['progress']:.3f}, "
                  f"success={series[-1]['success_probability']:.3f}", flush=True)
    finally:
        reward.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, required=True, choices=TASK_IDS)
    parser.add_argument("--gpu", type=int, required=True, choices=(0, 1))
    args = parser.parse_args()
    out = task_root(args.task_id)
    out.mkdir(parents=True, exist_ok=True)
    cfg = config(args.task_id, args.gpu)
    config_path = out / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != cfg:
        raise RuntimeError(f"Existing audit config differs: {config_path}")
    write_json(config_path, cfg)
    os.environ.update(CUDA_VISIBLE_DEVICES=str(args.gpu), MUJOCO_GL="egl",
                      MUJOCO_EGL_DEVICE_ID=str(args.gpu), EGL_DEVICE_ID=str(args.gpu),
                      XLA_PYTHON_CLIENT_PREALLOCATE="false", WANDB_MODE="disabled",
                      WANDB_DISABLED="true", PYTHONPATH=source_pythonpath(),
                      TMPDIR="/data/rohan/rbm-tmp")
    prepare_libero_paths(out / "libero_paths", REPO)
    sys.path.insert(0, str(REPO / "LIBERO"))
    try:
        set_status(out, "downloading_demos")
        source_path, digest = demo_source(args.task_id)
        demos = render_demos(args.task_id, out, source_path, digest)
        with reserve_gpus([args.gpu]):
            set_status(out, "collecting_policy")
            policy_records = collect_policy(cfg, out, args.gpu)
            set_status(out, "scoring")
            score_videos(cfg, out, args.gpu, policy_records, demos)
        set_status(out, "complete")
        print("Complete", flush=True)
    except BaseException as error:
        set_status(out, "failed")
        write_json(out / "error.json", {"error": repr(error)})
        raise


if __name__ == "__main__":
    main()
