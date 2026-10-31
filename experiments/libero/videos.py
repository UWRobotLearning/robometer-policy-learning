"""Record matched evaluations and select outcome-balanced videos for any task."""

from __future__ import annotations

import argparse
import copy
import fcntl
import hashlib
import html
import json
import os
import random
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from experiments.reward_audit.config import REPO, write_json
from experiments.reward_audit.launch import reserve_gpus, source_pythonpath, stop_run
from experiments.reward_audit.metrics import wilson

from .analysis import load_runs, matched_settings
from .config import load, run_config

DEMO_REPO = "yifengzhu-hf/LIBERO-datasets"
DEMO_REVISION = "f13aa24a3da8c43c7225569f28c562979fa0e35a"


def select_episodes(records: list[dict], successes: int, failures: int, seed: int) -> list[dict]:
    """Sample after the full evaluation, without searching for more favorable episodes."""
    rng = random.Random(seed)
    chosen = []
    for outcome, count in ((True, successes), (False, failures)):
        candidates = sorted((row for row in records if row["success"] == outcome), key=lambda row: row["episode"])
        chosen.extend(rng.sample(candidates, min(count, len(candidates))))
    return chosen


def build_jobs(results: Path, out: Path, seed: int, episodes: int, task_id: int | None = None) -> list[dict]:
    if results.is_file():
        spec = load(results)
        if task_id is None:
            if len(spec["tasks"]) != 1:
                raise ValueError("Use --task-id when the configuration contains several tasks")
            task_id = spec["tasks"][0]
        if task_id not in spec["tasks"] or spec["base_model"] == "scratch":
            raise ValueError("Frozen video audits require a configured task and an OpenPI base model")
        cfg = run_config(spec, task_id, seed, "robometer", spec["runtime"]["gpus"][0])
        cfg.update(eval_episodes=episodes, eval_unique_states=min(25, episodes))
        return [
            {
                "variant": "pretrained",
                "config": cfg,
                "actor_checkpoint": None,
                "path": str(out / "pretrained"),
                "base_model": spec["base_model"],
                "episodes": episodes,
            }
        ]
    runs = [
        run
        for run in load_runs(results)
        if run["config"]["seed"] == seed and (task_id is None or run["config"]["task_id"] == task_id)
    ]
    if not runs or len({(run["config"]["suite"], run["config"]["task"]) for run in runs}) != 1:
        raise ValueError("Choose one task with --task-id and an available --seed")
    matched_settings(runs)
    if len({run["config"]["reward"] for run in runs}) != len(runs):
        raise ValueError("Choose one comparison; duplicate reward variants would overwrite the same output")
    if any(not run["complete"] for run in runs):
        raise ValueError("Video comparisons require completed runs at the configured budget")
    jobs = []
    if runs[0]["base_model"] != "scratch":
        jobs.append({"variant": "pretrained", "config": copy.deepcopy(runs[0]["config"]), "actor_checkpoint": None})
    for run in sorted(runs, key=lambda run: run["config"]["reward"] != "sparse"):
        checkpoint = Path(run["path"]) / "checkpoints/final.pt"
        if not checkpoint.is_file():
            raise ValueError(f"Missing actor checkpoint: {checkpoint}")
        jobs.append(
            {
                "variant": run["config"]["reward"],
                "config": copy.deepcopy(run["config"]),
                "actor_checkpoint": str(checkpoint),
                "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
            }
        )
    for job in jobs:
        job.update(path=str(out / job["variant"]), base_model=runs[0]["base_model"], episodes=episodes)
    return jobs


def read_records(folder: Path) -> list[dict]:
    path = folder / "episodes.json"
    return json.loads(path.read_text()) if path.exists() else []


def trace_svg(points: list[dict], length: int) -> str:
    if not points:
        return "<p>Reward predictions pending.</p>"
    colors = {"progress": "#087f8c", "success_probability": "#d36a35", "ground_truth_reward": "#111827"}
    parts = [
        '<svg class="score-plot" viewBox="0 0 440 145" role="img" '
        'aria-label="Robometer and ground truth reward versus physical steps">'
    ]
    for value in (0, 0.5, 1):
        yy = 116 - 90 * value
        parts.append(
            f'<line x1="30" x2="425" y1="{yy}" y2="{yy}" stroke="#ddd"/>'
            f'<text x="4" y="{yy + 4}" font-size="11">{value:g}</text>'
        )
    for key, color in colors.items():
        coords = " ".join(f"{30 + 395 * row['step'] / max(1, length):.1f},{116 - 90 * row[key]:.1f}" for row in points)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
    parts.append(
        '<line class="playhead" x1="30" x2="30" y1="22" y2="116"/>'
        f'<text x="30" y="139" font-size="11">0</text><text x="350" y="139" font-size="11">{length} steps</text></svg>'
    )
    return "".join(parts)


def render_report(out: Path) -> Path:
    manifest = json.loads((out / "manifest.json").read_text())
    sections = []
    all_records = {}
    for job in sorted(manifest["jobs"], key=lambda job: job.get("kind") != "demonstrations"):
        variant = job["variant"]
        is_demo = job.get("kind") == "demonstrations"
        folder = out / variant
        records = read_records(folder)
        all_records[variant] = records
        status_path = folder / "status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"status": "queued"}
        selected_path = folder / "selected.json"
        selected = json.loads(selected_path.read_text()) if selected_path.exists() else []
        cards = []
        for row in selected:
            score_path = folder / "scores" / f"episode_{row['episode']:03d}.json"
            scores = json.loads(score_path.read_text()) if score_path.exists() else []
            outcome = "Success" if row["success"] else "Failure"
            title = f"Demonstration {row['episode']}" if is_demo else f"{outcome} · state {row['init_state_id']}"
            cards.append(
                f'<article class="episode" data-length="{row["length"]}" '
                f'data-series="{html.escape(json.dumps(scores), quote=True)}">'
                f'<div class="episode-head"><h3>{title}</h3><span>{row["length"]} steps</span></div>'
                f'<video controls playsinline preload="metadata" src="{variant}/videos/{row["video"]}"></video>'
                '<div class="plot-pane"><div class="legend"><span class="progress">Robometer progress</span>'
                '<span class="success">Predicted success</span>'
                '<span class="ground-truth">Ground truth reward</span></div>'
                f'{trace_svg(scores, row["length"])}<div class="live-values" aria-live="off"></div></div></article>'
            )
        successes = sum(row["success"] for row in records)
        low, high = wilson(successes, len(records))
        interval = f" · 95% Wilson CI {low:.1%} to {high:.1%}" if records and not is_demo else ""
        shortfall = (
            f"<p>Available outcomes: {successes} successes, {len(records) - successes} failures; "
            "the displayed subset uses all available videos when an outcome has fewer than requested.</p>"
            if records
            and not is_demo
            and (successes < manifest["successes"] or len(records) - successes < manifest["failures"])
            else ""
        )
        checkpoint = (
            (job.get("demo_file") or f"{DEMO_REPO}@{DEMO_REVISION}")
            if is_demo
            else (job["actor_checkpoint"] or job["config"].get("pi0_checkpoint", ""))
        )
        target = job.get("episodes", manifest["episodes"])
        outcome_label = "dataset success labels" if is_demo else "simulator successes"
        source_label = "Demonstration source" if is_demo else "Checkpoint"
        sections.append(
            f'<section class="task" id="{variant}"><h2>{variant.capitalize()}</h2>'
            f"<p>{html.escape(status['status'])} · {len(records)}/{target} episodes · "
            f"{successes}/{len(records)} {outcome_label}{interval}</p>"
            f'<p class="meta">{source_label}: {html.escape(str(checkpoint))}</p>{shortfall}'
            f'<div class="cards">{"".join(cards)}</div></section>'
        )
    table = []
    variants = [job["variant"] for job in manifest["jobs"] if job.get("kind") != "demonstrations"]
    for episode in range(manifest["episodes"]):
        cells = []
        state = None
        for variant in variants:
            row = next((row for row in all_records[variant] if row["episode"] == episode), None)
            if row:
                state = row["init_state_id"]
                cells.append(
                    f'<td><a href="{variant}/videos/{row["video"]}">{"success" if row["success"] else "failure"} '
                    f"({row['length']} steps)</a></td>"
                )
            else:
                cells.append("<td>pending</td>")
        table.append(f"<tr><td>{episode}</td><td>{state if state is not None else 'pending'}</td>{''.join(cells)}</tr>")
    cfg = manifest["jobs"][0]["config"]
    prompt = next(
        (row["prompt"] for variant in variants for row in all_records[variant] if row.get("prompt")), "pending"
    )
    policy_note = (
        "Frozen pretrained policy with native Gaussian sampling; no steering actor."
        if all(not job["actor_checkpoint"] for job in manifest["jobs"])
        else f"Trained latent policy sampling: {html.escape(manifest.get('sampling', 'deterministic'))}."
    )
    protocol_note = (
        "Policy rollouts use official held-out reset states; demonstrations are separate expert trajectories. "
        if any(job.get("kind") == "demonstrations" for job in manifest["jobs"])
        else "The same reset states and policy sampling seeds are used across variants. "
    )
    diagnostic_link = (
        '<p><a href="diagnostics.html">Training diagnostics and DSRL implementation audit</a></p>'
        if (out / "diagnostics.html").exists()
        else ""
    )
    page = (
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        '<meta http-equiv="refresh" content="60"><title>Policy video comparison</title><style>'
        + Path(__file__).with_name("video_report.css").read_text()
        + "</style></head><body><main>"
        + '<a href="/">← Main experiment dashboard</a>'
        + '<div class="eyebrow">Video and reward audit</div>'
        + f"<h1>{html.escape(cfg['suite'])} task {cfg['task_id']:02d}: "
        f"{html.escape(cfg['task'].replace('_', ' '))}</h1>"
        f"<p>{manifest['episodes']} evaluation episodes per policy, seed {manifest['seed']}, "
        f"{cfg['horizon']}-step horizon. {protocol_note}"
        f"After the full evaluation, {manifest['successes']} successes and {manifest['failures']} failures "
        f"are selected randomly (selection seed {manifest['selection_seed']}). "
        "This subset is not a success-rate estimate.</p>"
        f"<p><b>Policy instruction:</b> {html.escape(prompt)}</p><p>{policy_note}</p>"
        f"<p>Policy: <code>{html.escape(str(cfg.get('pi0_checkpoint', 'SAC from scratch')))}</code><br>"
        f"Reward: <code>{html.escape(cfg['reward_checkpoint'])}</code></p>"
        f"<p>{html.escape(manifest.get('training_coverage', 'Exact checkpoint training coverage is unverified.'))}</p>"
        '<p>Reward traces: <span style="color:#087f8c">Robometer progress</span>, '
        '<span style="color:#d36a35">Robometer success probability</span>, black native simulator reward. '
        "Predictions use only the observed prefix; trace inference is sampled every 20 steps. "
        "The dotted line follows video playback and seeking; values show the last scored prefix.</p>"
        + diagnostic_link
        + "".join(sections)
        + "<h2>Matched episodes: all rollouts</h2><table><thead><tr><th>Episode</th><th>State</th>"
        + "".join(f"<th>{variant}</th>" for variant in variants)
        + "</tr></thead><tbody>"
        + "".join(table)
        + "</tbody></table></main><script>"
        + Path(__file__).with_name("video_report.js").read_text()
        + "</script></body></html>"
    )
    path = out / "index.html"
    temporary = out / "index.html.tmp"
    temporary.write_text(page)
    temporary.replace(path)
    return path


def collect(job: dict, out: Path, gpu: int) -> list[dict]:
    import imageio.v3 as iio
    import torch

    from experiments.reward_audit.envs import EnvPool, prepare_libero_paths
    from experiments.reward_audit.sac import Dino, Replay, actor_actions, load_actor, make_sac

    from .policy import Pi0Client

    cfg = job["config"]
    random.seed(cfg["seed"])
    np.random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    torch.set_num_threads(cfg["torch_threads"])
    prepare_libero_paths(out / "libero_paths", REPO)
    records = read_records(out)
    done = {row["episode"] for row in records}
    policy = env = encoder = algorithm = None
    try:
        if job["base_model"] != "scratch":
            policy = Pi0Client(
                Path(cfg["pi0_python"]),
                Path(cfg["pi0_checkpoint"]),
                cfg["seed"],
                gpu,
                out,
                config_name=cfg.get("policy_config_name", "pi0_libero"),
            )
        if job["actor_checkpoint"]:
            device = torch.device("cuda:0")
            encoder = Dino(cfg["dino_checkpoint"], device)
            dimensions = {"dino_embedding": encoder.dimension, "observation/state": 8, "language": 384}
            action_dim = cfg["dsrl"]["noise_dim"] if policy else 7
            replay = Replay(1, dimensions, cfg["seed"], action_dim=action_dim)
            algorithm = make_sac(cfg, replay, device, action_dim=action_dim)
            load_actor(algorithm.actor, job["actor_checkpoint"], cfg, device)
            algorithm.actor.eval()
        env = EnvPool(cfg, 1, rank_offset=100000)
        prompt = env.metadata[0]["language"]
        for episode in range(job["episodes"]):
            if episode in done:
                continue
            if job.get("sampling") == "stochastic":
                torch.manual_seed(cfg["seed"] * 100000 + cfg["train_init_states"] + episode)
            state_id = cfg["train_init_states"] + episode % cfg.get("eval_unique_states", cfg["eval_episodes"])
            if policy:
                policy.reset_seed(cfg["seed"] * 100000 + cfg["train_init_states"] + episode)
            observation = env.reset({0: state_id})[0]["obs"]
            frames, reward_frames, rewards = (
                [observation["agentview_image"]],
                [observation[cfg["reward_camera"]]],
                [0.0],
            )
            latent_means = []
            while True:
                steering = None
                if algorithm is not None:
                    features = encoder.encode([observation])
                    steering = actor_actions(
                        algorithm.actor,
                        features,
                        device,
                        deterministic=job.get("sampling", "deterministic") == "deterministic",
                    )[0]
                    latent_means.append(steering.tolist())
                if policy:
                    actions = policy.infer(
                        observation, prompt, noise=steering, noise_mode=cfg["dsrl"].get("noise_mode", "direct")
                    )[: cfg["dsrl"]["action_exec_len"]]
                else:
                    actions = steering[None, :]
                for action in actions:
                    result = env.step([action])[0]
                    observation = result["obs"]
                    frames.append(observation["agentview_image"])
                    reward_frames.append(observation[cfg["reward_camera"]])
                    rewards.append(result["sparse"])
                    if result["terminated"] or result["truncated"]:
                        break
                if result["terminated"] or result["truncated"]:
                    break
            filename = f"episode_{episode:03d}.mp4"
            frame_indices = list(range(0, len(frames), 2))
            if frame_indices[-1] != len(frames) - 1:
                frame_indices.append(len(frames) - 1)
            iio.imwrite(out / "videos" / filename, np.stack([frames[i] for i in frame_indices]), fps=10)
            np.savez_compressed(
                out / "frames" / f"episode_{episode:03d}.npz",
                frames=np.stack(reward_frames),
                rewards=np.array(rewards),
                latent_actions=np.asarray(latent_means, dtype=np.float32),
            )
            record = {
                "episode": episode,
                "init_state_id": state_id,
                "success": bool(result["terminated"]),
                "length": int(result["elapsed"]),
                "video": filename,
                "prompt": prompt,
            }
            records.append(record)
            write_json(out / "episodes.json", records)
            write_json(
                out / "status.json", {"status": "collecting", "episodes": len(records), "target": job["episodes"]}
            )
            print(
                f"{job['variant']} episode {episode}: success={record['success']}, steps={record['length']}", flush=True
            )
        return records
    finally:
        if env is not None:
            env.close()
        if policy is not None:
            policy.close()
        del algorithm, encoder
        torch.cuda.empty_cache()


def score_selected(cfg: dict, folder: Path, selected: list[dict], gpu: int) -> None:
    from experiments.reward_audit.metrics import prefix_frames
    from experiments.reward_audit.rpc import RewardClient

    client = None
    try:
        for record in selected:
            path = folder / "scores" / f"episode_{record['episode']:03d}.json"
            if path.exists():
                continue
            if client is None:
                client = RewardClient(cfg, folder, gpu)
            with np.load(folder / "frames" / f"episode_{record['episode']:03d}.npz") as data:
                frames, rewards = data["frames"], data["rewards"]
                indices = sorted({0, *range(20, len(frames), 20), len(frames) - 1})
                values = []
                for start in range(0, len(indices), cfg["reward_batch_size"]):
                    batch = indices[start : start + cfg["reward_batch_size"]]
                    prefixes = [prefix_frames(frames[: index + 1], cfg["max_frames"]) for index in batch]
                    predictions = client.predict(prefixes, record["prompt"])
                    values.extend(
                        {
                            "step": index,
                            "progress": float(prediction[0]),
                            "success_probability": float(prediction[1]),
                            "ground_truth_reward": float(rewards[index]),
                        }
                        for index, prediction in zip(batch, predictions, strict=True)
                    )
            write_json(path, values)
    finally:
        if client is not None:
            client.close()


def collect_demos(job: dict, folder: Path) -> list[dict]:
    """Extract task-matched expert videos from the pinned public LIBERO dataset."""
    import h5py
    import imageio.v3 as iio
    from PIL import Image

    cfg = job["config"]
    if job.get("demo_file"):
        path = Path(job["demo_file"])
    else:
        from huggingface_hub import hf_hub_download

        path = Path(
            hf_hub_download(
                repo_id=DEMO_REPO,
                repo_type="dataset",
                revision=DEMO_REVISION,
                filename=f"{cfg['suite']}/{cfg['task']}_demo.hdf5",
            )
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    bddl = REPO / "LIBERO/libero/libero/bddl_files" / cfg["suite"] / f"{cfg['task']}.bddl"
    expected = re.search(r"\(:language\s+(.*?)\)", bddl.read_text(), re.DOTALL)
    if expected is None:
        raise ValueError(f"Missing language instruction: {bddl}")
    records = read_records(folder)
    with h5py.File(path, "r") as source:
        prompt = json.loads(source["data"].attrs["problem_info"])["language_instruction"]
        if " ".join(prompt.split()).casefold() != " ".join(expected.group(1).split()).casefold():
            raise ValueError("Demonstration instruction does not match the configured LIBERO task")
        for episode in range(job["episodes"]):
            if any(row["episode"] == episode for row in records):
                continue
            demo = source["data"][f"demo_{episode}"]
            rgb_key = "agentview_rgb" if cfg["reward_camera"] == "agentview_image" else "eye_in_hand_rgb"

            def images(key, demo=demo):
                return np.stack(
                    [
                        np.asarray(Image.fromarray(frame[::-1, ::-1]).resize((224, 224), Image.Resampling.BILINEAR))
                        for frame in demo["obs"][key]
                    ]
                )

            frames = images("agentview_rgb")
            reward_frames = frames if rgb_key == "agentview_rgb" else images(rgb_key)
            rewards = np.asarray(demo["rewards"], dtype=np.float32).reshape(-1)
            if len(rewards) != len(frames):
                raise ValueError("Demonstration observation and reward lengths differ")
            indices = sorted({*range(0, len(frames), 2), len(frames) - 1})
            video = f"episode_{episode:03d}.mp4"
            iio.imwrite(folder / "videos" / video, frames[indices], fps=10)
            np.savez_compressed(folder / "frames" / f"episode_{episode:03d}.npz", frames=reward_frames, rewards=rewards)
            records.append(
                {
                    "episode": episode,
                    "init_state_id": None,
                    "success": bool(np.asarray(demo["dones"]).reshape(-1)[-1]),
                    "length": len(frames) - 1,
                    "dataset_actions": len(demo["actions"]),
                    "video": video,
                    "prompt": prompt,
                }
            )
            write_json(folder / "episodes.json", records)
            write_json(
                folder / "status.json",
                {"status": "extracting demonstrations", "episodes": len(records), "target": job["episodes"]},
            )
    write_json(
        folder / "source.json",
        {
            "repo": DEMO_REPO if not job.get("demo_file") else None,
            "revision": DEMO_REVISION if not job.get("demo_file") else None,
            "file": str(path),
            "sha256": digest.hexdigest(),
            "prompt": prompt,
        },
    )
    return records


def worker(job: dict, manifest: dict, gpu: int) -> None:
    import traceback

    if os.environ.get("AUDIT_VIDEO_LAUNCHER_GUARDED") != "1":
        raise RuntimeError("Launch video recording through python -m experiments.libero video-audit")
    folder = Path(job["path"])
    for name in ("videos", "frames", "scores"):
        (folder / name).mkdir(parents=True, exist_ok=True)
    write_json(folder / "config.json", job["config"])
    try:
        is_demo = job.get("kind") == "demonstrations"
        records = collect_demos(job, folder) if is_demo else collect(job, folder, gpu)
        selected = (
            records
            if is_demo
            else select_episodes(records, manifest["successes"], manifest["failures"], manifest["selection_seed"])
        )
        write_json(folder / "selected.json", selected)
        write_json(folder / "status.json", {"status": "scoring", "episodes": len(records)})
        score_selected(job["config"], folder, selected, gpu)
        retained = {f"episode_{row['episode']:03d}.npz" for row in selected}
        for path in (folder / "frames").glob("*.npz"):
            if path.name not in retained:
                path.unlink()
        write_json(
            folder / "status.json",
            {
                "status": "complete",
                "episodes": len(records),
                "successes": sum(row["success"] for row in records),
                "videos_selected": len(selected),
            },
        )
    except BaseException:
        write_json(folder / "status.json", {"status": "failed", "error": traceback.format_exc()})
        raise


def run_audit(
    results: Path,
    out: Path,
    *,
    seed: int = 100,
    episodes: int = 25,
    successes: int = 2,
    failures: int = 2,
    selection_seed: int = 0,
    gpus: tuple[int, ...] = (0, 1),
    task_id: int | None = None,
    training_coverage: str = "",
    sampling: str = "deterministic",
    horizon: int | None = None,
    demos: int = 0,
    demo_file: Path | None = None,
) -> Path:
    if min(episodes, successes, failures) < 1 or not gpus or len(gpus) != len(set(gpus)):
        raise ValueError("Use positive episode/outcome counts and distinct GPUs")
    out = out.absolute()
    jobs = build_jobs(results.absolute(), out, seed, episodes, task_id)
    if demos < 0:
        raise ValueError("demos must be nonnegative")
    if demos:
        jobs.append(
            {
                "variant": "demos",
                "kind": "demonstrations",
                "config": copy.deepcopy(jobs[0]["config"]),
                "actor_checkpoint": None,
                "path": str(out / "demos"),
                "base_model": jobs[0]["base_model"],
                "episodes": demos,
                "demo_file": str(demo_file.absolute()) if demo_file else None,
            }
        )
    if horizon is not None:
        if horizon < 1:
            raise ValueError("horizon must be positive")
        for job in jobs:
            job["config"]["horizon"] = horizon
    if sampling not in ("deterministic", "stochastic"):
        raise ValueError("sampling must be deterministic or stochastic")
    for job in jobs:
        job["sampling"] = sampling
    manifest = {
        "results": str(results.absolute()),
        "seed": seed,
        "episodes": episodes,
        "successes": successes,
        "failures": failures,
        "selection_seed": selection_seed,
        "jobs": jobs,
        "training_coverage": training_coverage or "Exact checkpoint training coverage is unverified.",
        "sampling": sampling,
    }
    out.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(out).free < 5 * 1024**3:
        raise RuntimeError("Video audits require at least 5 GiB free for temporary frames")
    with (out / ".audit.lock").open("a") as audit_lock:
        try:
            fcntl.flock(audit_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(f"Video audit is already running: {out}") from None
        path = out / "manifest.json"
        if path.exists() and json.loads(path.read_text()) != manifest:
            raise ValueError(f"Refusing to mix different audit protocols in {out}")
        write_json(path, manifest)
        pending = []
        for job in jobs:
            status = Path(job["path"]) / "status.json"
            if not status.exists() or json.loads(status.read_text()).get("status") != "complete":
                pending.append(job)
        active = {}
        worker_environment = dict(os.environ)
        runtime_sources = (
            (results,) if results.is_file() else (results / "experiment.json", results.parent / "experiment.json")
        )
        for config_path in runtime_sources:
            if not config_path.exists():
                continue
            runtime = json.loads(config_path.read_text()).get("runtime", {})
            if runtime.get("cache_dir"):
                cache = Path(runtime["cache_dir"])
                worker_environment.update(XDG_CACHE_HOME=str(cache), HF_HOME=str(cache / "hf"))
            if runtime.get("tmp_dir"):
                Path(runtime["tmp_dir"]).mkdir(parents=True, exist_ok=True)
                worker_environment["TMPDIR"] = runtime["tmp_dir"]
            break

        def interrupted(signum, frame):
            raise KeyboardInterrupt("Video audit interrupted")

        previous_handler = signal.signal(signal.SIGTERM, interrupted)
        try:
            while pending or active:
                for gpu, (process, reservation) in list(active.items()):
                    code = process.poll()
                    if code is None:
                        continue
                    reservation.__exit__(None, None, None)
                    del active[gpu]
                    if code:
                        raise RuntimeError(f"Video worker failed with exit {code}; see {out}")
                reasons = []
                for gpu in gpus:
                    if gpu in active or not pending:
                        continue
                    reservation = reserve_gpus([gpu])
                    try:
                        reservation.__enter__()
                    except RuntimeError as error:
                        reasons.append(str(error))
                        continue
                    job = pending.pop(0)
                    folder = Path(job["path"])
                    folder.mkdir(parents=True, exist_ok=True)
                    env = {
                        **worker_environment,
                        "CUDA_VISIBLE_DEVICES": str(gpu),
                        "MUJOCO_GL": "egl",
                        "MUJOCO_EGL_DEVICE_ID": str(gpu),
                        "EGL_DEVICE_ID": str(gpu),
                        "WANDB_MODE": "disabled",
                        "WANDB_DISABLED": "true",
                        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
                        "PYTHONPATH": source_pythonpath(),
                        "AUDIT_VIDEO_LAUNCHER_GUARDED": "1",
                    }
                    try:
                        with (folder / "console.log").open("a") as log:
                            process = subprocess.Popen(
                                [
                                    sys.executable,
                                    "-u",
                                    "-m",
                                    "experiments.libero.videos",
                                    "--manifest",
                                    str(path),
                                    "--variant",
                                    job["variant"],
                                    "--gpu",
                                    str(gpu),
                                ],
                                cwd=REPO,
                                env=env,
                                stdin=subprocess.DEVNULL,
                                stdout=log,
                                stderr=subprocess.STDOUT,
                                start_new_session=True,
                            )
                    except BaseException:
                        reservation.__exit__(None, None, None)
                        raise
                    active[gpu] = (process, reservation)
                if pending and not active:
                    raise RuntimeError(f"No idle GPUs available: {reasons}")
                render_report(out)
                write_json(
                    out / "status.json",
                    {
                        "status": "running",
                        "active": {str(gpu): process.pid for gpu, (process, _) in active.items()},
                        "pending": [job["variant"] for job in pending],
                    },
                )
                if pending or active:
                    time.sleep(10)
            write_json(out / "status.json", {"status": "complete"})
        except BaseException as error:
            write_json(out / "status.json", {"status": "failed", "error": str(error)})
            raise
        finally:
            signal.signal(signal.SIGTERM, previous_handler)
            for process, reservation in active.values():
                stop_run(process)
                reservation.__exit__(None, None, None)
            render_report(out)
        return out / "index.html"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Internal isolated video worker; use python -m experiments.libero video-audit"
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--gpu", type=int, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    job = next(job for job in manifest["jobs"] if job["variant"] == args.variant)
    worker(job, manifest, args.gpu)


if __name__ == "__main__":
    main()
