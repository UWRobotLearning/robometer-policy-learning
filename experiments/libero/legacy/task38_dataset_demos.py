"""Render two LIBERO-90 expert demonstrations and audit Robometer on them."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np

from experiments.reward_audit.config import write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath
from experiments.reward_audit.metrics import prefix_frames
from experiments.reward_audit.rpc import RewardClient


DATA_ROOT = Path("/data/rohan/Reward-Modeling")
SOURCES = {
    38: ("KITCHEN_SCENE8_put_the_right_moka_pot_on_the_stove_demo.hdf5",
         "d2b6a54c21113cc128513ce1221e8ae28432133dde54785a17ce916fb898ab77"),
    55: ("LIVING_ROOM_SCENE3_pick_up_the_alphabet_soup_and_put_it_in_the_tray_demo.hdf5",
         "d1bdd4ff5d08194649b73a94a53dca8d0bd14b1278cd85a75a9b8e6df5af88e9"),
}
REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"
VIDEO_EVERY = 4
SCORE_EVERY = 20
FPS = 10


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task-id", type=int, choices=tuple(SOURCES), default=38)
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()
    audit = DATA_ROOT / f"pi0_task{args.task_id}_video_audit_20261014"
    source_name, source_sha256 = SOURCES[args.task_id]
    source_path = audit / "dataset_source" / "libero_90" / source_name
    digest = hashlib.sha256()
    with source_path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != source_sha256:
        raise RuntimeError("Dataset SHA256 differs from the pinned source")
    out = audit / "dataset_demos"
    (out / "videos").mkdir(parents=True, exist_ok=True)
    (out / "scores").mkdir(exist_ok=True)
    cfg = json.loads((audit / "config.json").read_text())
    write_json(out / "config.json", cfg)
    records = []
    with h5py.File(source_path) as source:
        prompt = json.loads(source["data"].attrs["problem_info"])["language_instruction"]
        for demo_id in (0, 1):
            demo = source["data"][f"demo_{demo_id}"]
            length = len(demo["actions"])
            indices = list(range(0, length, VIDEO_EVERY))
            if indices[-1] != length - 1:
                indices.append(length - 1)
            # LIBERO's HDF5 stores OpenGL camera frames rotated 180 degrees.
            frames = np.stack([demo["obs"]["agentview_rgb"][i][::-1, ::-1].copy()
                               for i in indices])
            video = out / "videos" / f"demo_{demo_id}.mp4"
            if not video.exists():
                iio.imwrite(video, frames, fps=FPS)
            reward_events = [{"step": int(i), "reward": float(demo["rewards"][i])}
                             for i in range(length) if demo["rewards"][i] != 0]
            records.append({"demo_id": demo_id, "length": length,
                            "video_frames": len(frames), "success": bool(demo["dones"][-1]),
                            "reward_events": reward_events,
                            "frames": frames, "steps": np.asarray(indices)})
            print(f"Rendered demo {demo_id}: {length} steps, {len(frames)} frames", flush=True)
    os.environ.update(PYTHONPATH=source_pythonpath(), WANDB_MODE="disabled", WANDB_DISABLED="true")
    pending = [record for record in records
               if not (out / "scores" / f"demo_{record['demo_id']}.json").exists()]
    if pending:
        with reserve_gpus([args.gpu]):
            reward = RewardClient(cfg, out, args.gpu)
            try:
                for record in pending:
                    demo_id = record["demo_id"]
                    score_path = out / "scores" / f"demo_{demo_id}.json"
                    frames, steps = record["frames"], record["steps"]
                    selected = [i for i, step in enumerate(steps)
                                if step == 0 or step % SCORE_EVERY == 0 or i == len(steps) - 1]
                    prefixes = [prefix_frames(frames[:i + 1], cfg["max_frames"]) for i in selected]
                    predictions = reward.predict(prefixes, prompt)
                    series = [{"step": int(steps[i]), "video_time": i / FPS,
                               "progress": float(prediction[0]),
                               "success_probability": float(prediction[1])}
                              for i, prediction in zip(selected, predictions, strict=True)]
                    write_json(score_path, series)
                    print(f"Scored demo {demo_id}: final progress {series[-1]['progress']:.3f}, "
                          f"success estimate {series[-1]['success_probability']:.3f}", flush=True)
            finally:
                reward.close()
    write_json(out / "manifest.json", {
        "source": "yifengzhu-hf/LIBERO-datasets", "revision": REVISION,
        "source_file": source_name, "sha256": source_sha256, "prompt": prompt,
        "fps": FPS, "video_every_physical_steps": VIDEO_EVERY,
        "robometer_every_physical_steps": SCORE_EVERY,
        "demonstrations": [{k: v for k, v in record.items() if k not in ("frames", "steps")}
                           for record in records],
    })


if __name__ == "__main__":
    main()
