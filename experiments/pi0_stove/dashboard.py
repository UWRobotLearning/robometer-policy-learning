"""Serve one live, read-only dashboard for LIBERO-90 pi0 fine-tuning runs."""

from __future__ import annotations

import argparse
import csv
import html
import json
import math
import re
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t
from experiments import official_dsrl
from experiments.libero.analysis import paired_sign_flip_p

from experiments.libero.legacy.task38_video_report import BASE as VIDEO_BASE, render as render_task38_video
from experiments.libero.legacy.pi0_eval_report import render as render_pi0_eval
from experiments.libero.legacy.pi05_libero_report import render as render_pi05_libero
from experiments.libero.legacy.pi05_libero_eval import ROOT as PI05_ROOT, tasks as pi05_tasks, task_dir as pi05_task_dir, completed as pi05_completed
from experiments.libero.legacy.pi05_libero90_video_report import render as render_pi05_video_audit
from experiments.libero.legacy.pi05_libero90_video_audit import ROOT as PI05_VIDEO_ROOT
from experiments.libero.legacy.pi0_family_report import (
    RBM_DATASETS, render as render_pi0_families, training_coverage,
)
from experiments.libero.legacy.pi0_family_eval import SUITES as FAMILY_SUITES, task_dir as family_task_dir, completed as family_completed
from experiments.reward_audit.config import task_map
from experiments.libero.legacy.libero10_video_report import (
    audit_training_coverage, render as render_libero10_audit,
)


KINDS = (("sparse", "Sparse simulator reward", "#087f8c"),
         ("robometer", "Robometer reward", "#d36a35"))
TASK_DIR = re.compile(r"pi0(?:5)?_(libero90|libero10)_task(\d+)(?:_|$)")
SAC_TASK28_ROOT = 'sac_task28_general_robometer_20261021_runs'


def family_label(suite: str) -> str:
    return 'LIBERO-' + suite.removeprefix('libero_').capitalize()


def experiment_date(root: Path) -> float:
    """Use creation metadata, then dated directory names, then source-file time."""
    manifest = root / 'manifest.json'
    try:
        metadata = json.loads(manifest.read_text())
        for key in ('created', 'created_at', 'started_at'):
            if metadata.get(key):
                try:
                    value = datetime.fromisoformat(str(metadata[key]).replace('Z', '+00:00'))
                    return value.replace(tzinfo=value.tzinfo or timezone.utc).timestamp()
                except ValueError:
                    continue
    except (OSError, ValueError):
        pass
    match = re.search(r'(20\d{6})(?:_|$)', root.name)
    if match:
        try:
            return datetime.strptime(match.group(1), '%Y%m%d').replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            pass
    for path in (manifest, root / 'planned_suite.json', root):
        try:
            return path.stat().st_mtime
        except OSError:
            continue
    return 0.0


def experiment_date_label(root: Path) -> str:
    stamp = experiment_date(root)
    if not stamp:
        return '<span class="experiment-date">Start date unavailable</span>'
    date = datetime.fromtimestamp(stamp, timezone.utc)
    return (f'<time class="experiment-date" datetime="{date.isoformat()}">'
            f'Started {date:%Y-%m-%d} · UTC</time>')


def newest_first(tasks: list[dict]) -> list[dict]:
    return sorted(tasks, key=lambda task: (experiment_date(task['root']), str(task['root'])), reverse=True)


def task_roots(data_root: Path) -> list[Path]:
    """Discover new task folders on every request, without restarting the server."""
    roots = []
    manifests = [*data_root.glob("pi0*_libero*_task*/manifest.json"),
                 *data_root.glob("experiments/*/*/manifest.json")]
    for manifest in manifests:
        metadata = json.loads(manifest.read_text())
        match = TASK_DIR.match(manifest.parent.name)
        if match or metadata.get("schema_version") == 1:
            suite = metadata.get('suite') or match.group(1)
            task_id = metadata.get('task_id')
            roots.append((suite, int(task_id if task_id is not None else match.group(2)), manifest.parent))
    return sorted((root for _, _, root in roots), key=lambda root: (experiment_date(root), str(root)), reverse=True)


def run_dir(root: Path, manifest: dict, kind: str, seed: int) -> Path:
    if seed == 100 and "seed100_run" in manifest:
        return Path(manifest["seed100_run"]) / f"{kind}_seed100"
    return root / f"seed{seed}" / f"{kind}_seed{seed}"


def read_evaluations(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            try:
                rows.append({"steps": int(float(row["env_steps"])),
                             "successes": int(row["successes"]),
                             "episodes": int(row["episodes"]),
                             "rate": float(row["success_rate"])})
            except (KeyError, TypeError, ValueError):
                continue  # A training process may still be writing the last line.
    return rows


def read_eval_state_outcomes(path: Path) -> dict[int, dict[int, bool]]:
    """Read complete 25-state evaluations, ignoring a partially written evaluation."""
    if not path.exists():
        return {}
    by_step: dict[int, dict[int, bool]] = {}
    with path.open() as stream:
        for line in stream:
            try:
                record = json.loads(line)
                if record.get("phase") != "eval":
                    continue
                step, state = int(record["env_steps"]), int(record["init_state_id"])
                success = record["success"]
                if state not in range(25, 50) or not isinstance(success, bool):
                    continue
                by_step.setdefault(step, {})[state] = success
            except (ValueError, TypeError, KeyError):
                continue
    return {step: states for step, states in by_step.items() if len(states) == 25}


def baseline_location(task: dict, data_root: Path) -> tuple[Path, str]:
    name = task_map()[task["suite"]][task["id"]]
    if task["suite"] == "libero_90":
        return (data_root / "pi0_task_probe_20261010" / f'{task["id"]:02d}_{name}',
                "/pi0-libero90-eval/")
    return (data_root / "pi0_libero_families_20261014" / task["suite"] /
            f'{task["id"]:02d}_{name}', "/pi0-libero-families-eval/#libero_10")


def frozen_pi0_baseline(task: dict, data_root: Path) -> dict | None:
    """Use the separately recorded 25-state evaluation of this exact checkpoint."""
    suite = task["suite"]
    if suite not in ("libero_90", "libero_10") or len(task["policy_checkpoints"]) != 1:
        return None
    names = task_map().get(suite, [])
    task_id = task["id"]
    if task_id >= len(names) or task["name"] != names[task_id].replace("_", " "):
        return None
    folder, url = baseline_location(task, data_root)
    try:
        status = json.loads((folder / "status.json").read_text())
        config = json.loads((folder / "config.json").read_text())
        summary = json.loads((folder / "summary.json").read_text())
    except (OSError, ValueError):
        return None
    if (status.get("status") != "complete" or config.get("suite") != suite or
            config.get("task_id") != task_id or
            config.get("checkpoint") not in task["policy_checkpoints"] or
            config.get("eval_state_ids") != list(range(25, 50)) or
            config.get("horizon") != 400 or summary.get("episodes") != 25):
        return None
    try:
        successes = int(summary["successes"])
        rate = float(summary["success_rate"])
    except (KeyError, TypeError, ValueError):
        return None
    if not (0 <= successes <= 25 and 0 <= rate <= 1 and abs(rate - successes / 25) < 1e-9):
        return None
    interval = summary.get("wilson_ci95")
    if not (isinstance(interval, list) and len(interval) == 2):
        return None
    try:
        ci_low, ci_high = (float(value) for value in interval)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(ci_low) and math.isfinite(ci_high) and
            0 <= ci_low <= rate <= ci_high <= 1):
        return None
    return {"rate": rate, "successes": successes, "episodes": 25,
            "ci_low": ci_low, "ci_high": ci_high, "url": url}


def frozen_pi05_run_baseline(task: dict) -> dict | None:
    """Step-zero evaluations of the exact pi0.5 checkpoint used for training."""
    if not task['root'].name.startswith('pi05_'):
        return None
    policy = task.get('policy_checkpoints', set())
    if len(policy) != 1 or not next(iter(policy)).rstrip('/').endswith('/pi05_libero'):
        return None
    if task.get('eval_episodes') == 25:
        seed_rates = []
        successes = episodes = 0
        for seed in task['seeds']:
            rows = []
            for kind in ('sparse', 'robometer'):
                evaluation = next((row for row in task['runs'][kind][seed]['evaluations']
                                   if row['steps'] == 0 and row['episodes'] == 25), None)
                if evaluation is not None:
                    rows.append(evaluation)
            if len(rows) == 2:
                seed_successes = sum(row['successes'] for row in rows)
                seed_episodes = sum(row['episodes'] for row in rows)
                seed_rates.append(seed_successes / seed_episodes)
                successes += seed_successes
                episodes += seed_episodes
        if len(seed_rates) < 2:
            return None
        rate = successes / episodes
        margin = t.ppf(.975, len(seed_rates) - 1) * stdev(seed_rates) / len(seed_rates) ** .5
        return {'rate': rate, 'successes': successes, 'episodes': episodes,
                'ci_low': max(0, mean(seed_rates) - margin),
                'ci_high': min(1, mean(seed_rates) + margin),
                'unique_states': 25, 'seeds': len(seed_rates),
                'ci_method': 'seed-clustered t',
                'url': f'#{task["anchor"]}', 'policy_label': 'π₀.₅'}
    seed = task['seeds'][0]
    run = task['runs']['sparse'][seed]
    evaluation = next((row for row in run['evaluations'] if row['steps'] == 0), None)
    if evaluation is None or evaluation['episodes'] != 50:
        return None
    states: dict[int, list[bool]] = {}
    episodes = task['root'] / f'seed{seed}' / f'sparse_seed{seed}' / 'episodes.jsonl'
    try:
        with episodes.open() as stream:
            for line in stream:
                row = json.loads(line)
                if row.get('phase') == 'eval' and row.get('env_steps') == 0:
                    states.setdefault(int(row['init_state_id']), []).append(row['success'])
    except (OSError, ValueError, KeyError):
        return None
    if set(states) != set(range(25, 50)) or any(len(values) != 2 for values in states.values()):
        return None
    state_rates = [mean(states[state]) for state in sorted(states)]
    successes = evaluation['successes']
    if successes != sum(sum(values) for values in states.values()):
        return None
    rate = mean(state_rates)
    margin = t.ppf(.975, 24) * stdev(state_rates) / 25 ** .5
    low, high = max(0, rate - margin), min(1, rate + margin)
    return {'rate': successes / 50, 'successes': successes, 'episodes': 50,
            'ci_low': low, 'ci_high': high, 'unique_states': 25,
            'ci_method': 'state-clustered t',
            'url': f'#{task["anchor"]}', 'policy_label': 'π₀.₅'}


def load_task(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text())
    planned_config = next((run.get('config', {}) for run in manifest.get('planned_runs', [])), {})
    task_id = int(manifest.get("task_id") if manifest.get("task_id") is not None
                  else TASK_DIR.match(root.name).group(2))
    suite = manifest.get("suite") or ("libero_10" if TASK_DIR.match(root.name).group(1) == "libero10"
                                       else "libero_90")
    seeds = [int(seed) for seed in manifest["seeds"]]
    canceled = set()
    stop_file = root / "stop_after_seed101.json"
    if stop_file.exists():
        canceled = set(json.loads(stop_file.read_text()).get("canceled_seeds", []))
    max_steps = int(manifest.get("steps_per_run", 100000))
    orchestrator_path = root / "orchestrator.json"
    orchestrator = json.loads(orchestrator_path.read_text()) if orchestrator_path.exists() else {}
    runs = {}
    policy_checkpoints = {manifest["pi0_checkpoint"]} if manifest.get("pi0_checkpoint") else set()
    reward_checkpoints = {manifest["reward_checkpoint"]} if manifest.get("reward_checkpoint") else set()
    for kind, _, _ in KINDS:
        runs[kind] = {}
        for seed in seeds:
            folder = run_dir(root, manifest, kind, seed)
            config_path = folder / "config.json"
            if config_path.exists():
                config = json.loads(config_path.read_text())
                if config.get("pi0_checkpoint"):
                    policy_checkpoints.add(config["pi0_checkpoint"])
                if kind == "robometer" and config.get("reward_checkpoint"):
                    reward_checkpoints.add(config["reward_checkpoint"])
            evaluations = read_evaluations(folder / "evaluation.csv")
            status_path = folder / "status.json"
            status = json.loads(status_path.read_text()).get("status", "queued") if status_path.exists() else "queued"
            if manifest.get('rewards') and kind not in manifest['rewards']:
                status = 'not selected'
            if manifest.get('seeds_by_reward') and seed not in manifest['seeds_by_reward'].get(kind, []):
                status = 'not selected'
            if seed in canceled and status == "queued":
                status = "canceled"
            runs[kind][seed] = {"status": status, "evaluations": evaluations}
    task = {"id": task_id, "suite": suite, "name": manifest["task"].replace("_", " "),
            "anchor": "task-" + root.name,
            "algorithm": manifest.get("algorithm", "SAC residual steering"),
            "root": root, "seeds": seeds, "canceled": canceled,
            "base_model": manifest.get('base_model'), "selected_rewards": manifest.get('rewards', ['sparse', 'robometer']),
            "orchestration_status": orchestrator.get("status"),
            "max_steps": max_steps, "runs": runs,
            "num_envs": manifest.get('num_envs', planned_config.get('num_envs')),
            "action_exec_len": manifest.get('action_exec_len', planned_config.get('dsrl', {}).get('action_exec_len', 1)),
            "eval_episodes": manifest.get("eval_episodes"),
            "eval_episodes_by_seed": manifest.get("eval_episodes_by_seed"),
            "eval_every": manifest.get("eval_every"),
            "horizon": manifest.get("horizon"),
            'analysis_note': manifest.get('analysis_note'),
            'analysis_url': manifest.get('analysis_url'),
            "policy_checkpoints": policy_checkpoints,
            "reward_checkpoints": reward_checkpoints}
    if manifest.get('attached_workers'):
        states = [run['status'] for kind in task['selected_rewards'] for run in runs[kind].values()
                  if run['status'] != 'not selected']
        task['orchestration_status'] = ('running' if 'running' in states else
                                        'complete' if all(state == 'complete' for state in states) else
                                        'queued' if 'queued' in states else 'stopped')
    task["baseline"] = (frozen_pi05_run_baseline(task) if root.name.startswith('pi05_')
                        else frozen_pi0_baseline(task, root.parent))
    if manifest.get('schema_version') == 1 and manifest.get('base_model') != 'scratch':
        from experiments.libero.analysis import baseline, load_runs

        frozen = baseline(load_runs(root))
        if frozen and frozen['t_ci95']:
            task['baseline'] = {
                'rate': frozen['mean'], 'successes': round(frozen['mean'] * frozen['episodes']),
                'episodes': int(frozen['episodes']), 'ci_low': max(0, frozen['t_ci95'][0]),
                'ci_high': min(1, frozen['t_ci95'][1]),
                'ci_method': 'seed-level t' if len(frozen['seed_rates']) > 1 else 'reset-state t',
                'url': f'#{task["anchor"]}',
                'policy_label': 'π₀.₅' if manifest['base_model'] == 'pi0.5' else 'π₀',
            }
    if task["baseline"] and not root.name.startswith('pi05_'):
        baseline_folder, _ = baseline_location(task, root.parent)
        try:
            records = json.loads((baseline_folder / "summary.json").read_text())["records"]
            states = {int(record["init_state_id"]): record["success"] for record in records}
            if (set(states) == set(range(25, 50)) and len(records) == 25 and
                    all(isinstance(success, bool) for success in states.values()) and
                    sum(states.values()) == task["baseline"]["successes"]):
                task["baseline_states"] = states
                for kind, _, _ in KINDS:
                    for seed in seeds:
                        folder = run_dir(root, manifest, kind, seed)
                        task["runs"][kind][seed]["state_evaluations"] = read_eval_state_outcomes(
                            folder / "episodes.jsonl")
        except (OSError, ValueError, KeyError, TypeError):
            pass
    return task


def checkpoint_training_note(task: dict, data_root: Path | None = None) -> str:
    """Describe exact-task coverage only for checkpoints with verified training mixes."""
    policy = task.get("policy_checkpoints", set())
    reward = task.get("reward_checkpoints", set())
    if (Path(task.get('root', '')).name.startswith('pi05_') and task.get('suite') == 'libero_90' and
            len(policy) == 1 and next(iter(policy)).rstrip('/').endswith('/pi05_libero') and
            len(reward) == 1 and next(iter(reward)).startswith('robometer/Robometer-4B@')):
        return ('<p class="training-note"><b>This task in checkpoint training data:</b> '
                'π₀.₅ policy (pi05_libero): <b>No · LIBERO-90 held out</b> · '
                'Robometer reward model (Robometer-4B): <b>No · LIBERO-90 held out</b>. '
                'Exact trajectory overlap is unknown.</p>')
    if task.get("suite") == "libero_10" and data_root is not None:
        provenance = training_coverage(data_root)
        policy_known = (len(policy) == 1 and provenance["policy_known"] and
                        next(iter(policy)) == provenance["policy_checkpoint"])
        reward_known = (len(reward) == 1 and provenance["reward_known"] and
                        next(iter(reward)) == provenance["reward_checkpoint"] and
                        RBM_DATASETS["libero_10"] in provenance["reward_datasets"])
        return ('<p class="training-note"><b>Checkpoint training coverage:</b> '
                f'π₀ policy: <b>{"Yes · LIBERO-10 suite" if policy_known else "Unverified"}</b> · '
                f'Robometer reward: <b>{"Yes · LIBERO-10 suite" if reward_known else "Unverified"}</b>. '
                'Exact training-trajectory overlap is unknown.</p>')
    if (len(policy) == 1 and len(reward) == 1 and
            next(iter(policy)).rstrip("/").endswith("/pi0_libero") and
            next(iter(reward)).startswith("robometer/Robometer-4B@")):
        return ('<p class="training-note"><b>This task in checkpoint training data:</b> '
                'π₀ policy (pi0_libero): <b>No</b> · '
                'Robometer reward model (Robometer-4B): <b>No</b>. '
                'Both checkpoints trained on other LIBERO suites; neither included LIBERO-90.</p>')
    return ('<p class="training-note"><b>This task in checkpoint training data:</b> '
            'Unverified for the configured checkpoints.</p>')


def aggregate(curves: list[list[dict]]) -> list[dict]:
    points = []
    for index in range(max((len(curve) for curve in curves), default=0)):
        rows = [curve[index] for curve in curves if len(curve) > index]
        rates = [row["rate"] for row in rows]
        margin = t.ppf(.975, len(rates) - 1) * stdev(rates) / len(rates) ** .5 if len(rates) > 1 else None
        center = mean(rates)
        points.append({"steps": sum(row["steps"] for row in rows) / len(rows),
                       "rate": center, "n": len(rows),
                       "ci_low": max(0, center - margin) if margin is not None else None,
                       "ci_high": min(1, center + margin) if margin is not None else None})
    return points


def paired_comparison(task: dict) -> dict | None:
    """Compare final success rates only for seeds completed under both rewards."""
    differences = []
    for seed in task["seeds"]:
        sparse = task["runs"]["sparse"][seed]
        robometer = task["runs"]["robometer"][seed]
        if any(run["status"] != "complete" or not run["evaluations"] or
               run["evaluations"][-1]["steps"] < task["max_steps"]
               for run in (sparse, robometer)):
            continue
        differences.append(sparse["evaluations"][-1]["rate"] - robometer["evaluations"][-1]["rate"])
    n = len(differences)
    if n < 2:
        return {"n": n, "mean": None, "ci_low": None, "ci_high": None, "p": None}
    effect = mean(differences)
    margin = t.ppf(.975, n - 1) * stdev(differences) / n ** .5
    p = paired_sign_flip_p(differences)
    return {"n": n, "mean": effect, "ci_low": effect - margin,
            "ci_high": effect + margin, "p": p}




def checkpoint_comparisons(task: dict) -> list[dict]:
    """Pair reward arms by training seed and evaluation index at each checkpoint."""
    runs = task["runs"]
    maximum = max((len(runs[kind][seed]["evaluations"])
                   for kind, _, _ in KINDS for seed in task["seeds"]), default=0)
    comparisons = []
    for index in range(maximum):
        differences = []
        steps = []
        for seed in task["seeds"]:
            sparse = runs["sparse"][seed]["evaluations"]
            robometer = runs["robometer"][seed]["evaluations"]
            if len(sparse) <= index or len(robometer) <= index:
                continue
            differences.append(sparse[index]["rate"] - robometer[index]["rate"])
            steps.extend((sparse[index]["steps"], robometer[index]["steps"]))
        if len(differences) >= 2:
            comparisons.append({"steps": mean(steps), "n": len(differences),
                                "difference": mean(differences),
                                "p": paired_sign_flip_p(differences)})
    return comparisons


def integer_sign_flip_p(differences: list[int]) -> float:
    """Exact two-sided sign-flip p-value without enumerating 2**25 state assignments."""
    counts = {0: 1}
    for difference in differences:
        next_counts = {}
        for total, count in counts.items():
            for signed in (total - difference, total + difference):
                next_counts[signed] = next_counts.get(signed, 0) + count
        counts = next_counts
    observed = abs(sum(differences))
    return sum(count for total, count in counts.items() if abs(total) >= observed) / 2 ** len(differences)


def baseline_comparisons(task: dict, kind: str) -> list[dict]:
    """Compare one reward arm with π₀ using both seed and matched-state variation."""
    baseline = task.get("baseline")
    baseline_states = task.get("baseline_states")
    if not baseline or not baseline_states or set(baseline_states) != set(range(25, 50)):
        return []
    runs = task["runs"][kind]
    maximum = max((len(runs[seed]["evaluations"]) for seed in task["seeds"]), default=0)
    comparisons = []
    for index in range(maximum):
        selected = []
        steps = []
        rates = []
        for seed in task["seeds"]:
            run = runs[seed]
            if len(run["evaluations"]) <= index:
                continue
            evaluation = run["evaluations"][index]
            states = run.get("state_evaluations", {}).get(evaluation["steps"])
            if states is None or set(states) != set(baseline_states):
                continue
            selected.append(states)
            steps.append(evaluation["steps"])
            rates.append(sum(states.values()) / len(states))
        if len(selected) < 2:
            continue
        n = len(selected)
        seed_p = paired_sign_flip_p([rate - baseline["rate"] for rate in rates])
        state_differences = [sum(states[state] for states in selected) - n * baseline_states[state]
                             for state in sorted(baseline_states)]
        state_p = integer_sign_flip_p(state_differences)
        comparisons.append({"steps": mean(steps), "n": n, "states": len(baseline_states),
                            "difference": mean(rates) - baseline["rate"],
                            "seed_p": seed_p, "state_p": state_p,
                            "p": max(seed_p, state_p)})
    return comparisons


def sparse_baseline_comparisons(task: dict) -> list[dict]:
    """Retain the existing sparse-arm comparison entry point."""
    return baseline_comparisons(task, "sparse")


def confidence_polygons(points: list[dict], x, y, color: str) -> list[str]:
    """Do not bridge checkpoints with fewer than two available seeds."""
    polygons = []
    segment = []
    for point in [*points, None]:
        if point is not None and point["ci_low"] is not None:
            segment.append(point)
            continue
        if len(segment) > 1:
            upper = [(x(row["steps"]), y(row["ci_high"])) for row in segment]
            lower = [(x(row["steps"]), y(row["ci_low"])) for row in reversed(segment)]
            coordinates = " ".join(f"{xx:.1f},{yy:.1f}" for xx, yy in upper + lower)
            polygons.append(f'<polygon class="confidence-band" points="{coordinates}" '
                            f'fill="{color}" opacity=".15"/>')
        segment = []
    return polygons


def plot(task: dict) -> str:
    width, height = 640, 300
    left, right, top, bottom = 58, 20, 64, 48
    inner_w, inner_h = width - left - right, height - top - bottom
    env_count = task.get('num_envs') or 1
    x = lambda steps: left + inner_w * min(steps / task["max_steps"], 1)
    y = lambda rate: top + inner_h * (1 - rate)
    lines = [f'<svg viewBox="0 0 {width} {height}" role="img" '
             f'aria-label="Success rate versus environment transitions for task {task["id"]}">']
    for fraction in (0, .25, .5, .75, 1):
        yy = y(fraction)
        lines.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>')
        lines.append(f'<text x="{left-12}" y="{yy+4:.1f}" text-anchor="end" class="tick">{fraction:.0%}</text>')
        xx = x(fraction * task["max_steps"])
        lines.append(f'<line x1="{xx:.1f}" y1="{top}" x2="{xx:.1f}" y2="{height-bottom}" class="grid"/>')
        lines.append(f'<text x="{xx:.1f}" y="{height-bottom+22}" text-anchor="middle" class="tick">'
                     f'{int(fraction * task["max_steps"] / env_count / 1000)}k</text>')
    baseline = task.get("baseline")
    if baseline:
        yy = y(baseline["rate"])
        policy_label = baseline.get('policy_label', 'π₀')
        sample = (f'{baseline["episodes"]} episodes on {baseline["unique_states"]} held-out states'
                  if baseline.get('unique_states') else f'{baseline["episodes"]} held-out states')
        method = baseline.get('ci_method', 'Wilson')
        title = (f'Frozen {policy_label} checkpoint: {baseline["successes"]}/{baseline["episodes"]} '
                 f'({baseline["rate"]:.0%}) simulator successes on {sample}; '
                 f'95% {method} CI {baseline["ci_low"]:.0%}–{baseline["ci_high"]:.0%}')
        band_top = y(baseline["ci_high"])
        band_height = y(baseline["ci_low"]) - band_top
        lines.append(f'<rect class="baseline-ci" x="{left}" y="{band_top:.1f}" '
                     f'width="{inner_w}" height="{band_height:.1f}" fill="#475569" opacity=".11">'
                     f'<title>{html.escape(title)}</title></rect>')
        lines.append(f'<line class="baseline-line" x1="{left}" y1="{yy:.1f}" '
                     f'x2="{width-right}" y2="{yy:.1f}" stroke="#475569" '
                     f'stroke-width="2.3" stroke-dasharray="2 7" stroke-linecap="round">'
                     f'<title>{html.escape(title)}</title></line>')
        label_y = yy + 17 if yy < top + 18 else yy - 7
        lines.append(f'<text class="baseline-label" x="{width-right-7}" y="{label_y:.1f}" '
                     f'text-anchor="end">{policy_label} {baseline["rate"]:.0%}</text>')
    for kind, _, color in KINDS:
        curves = [task["runs"][kind][seed]["evaluations"] for seed in task["seeds"]]
        for curve in curves:
            if len(curve) > 1:
                points = " ".join(f'{x(row["steps"]):.1f},{y(row["rate"]):.1f}' for row in curve)
                lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                             'stroke-width="1.4" opacity=".24"/>')
        mean = aggregate(curves)
        if mean:
            lines.extend(confidence_polygons(mean, x, y, color))
            points = " ".join(f'{x(row["steps"]):.1f},{y(row["rate"]):.1f}' for row in mean)
            lines.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                         'stroke-width="3.2" stroke-linejoin="round"/>')
            last = mean[-1]
            lines.append(f'<circle cx="{x(last["steps"]):.1f}" cy="{y(last["rate"]):.1f}" '
                         f'r="4.5" fill="{color}"/>')
    for kind, label, star_class, star_y in (
            ("sparse", "sparse", "sparse-baseline-star", 20),
            ("robometer", "Robometer", "robometer-baseline-star", 39)):
        for comparison in baseline_comparisons(task, kind):
            if comparison["p"] < .05:
                title = (f'{comparison["steps"] / env_count:,.0f} physical steps/env: {label} − frozen π₀ '
                         f'{comparison["difference"] * 100:+.1f} percentage points; '
                         f'{comparison["n"]} training seeds, {comparison["states"]} matched reset states; '
                         f'exact seed sign-flip p={comparison["seed_p"]:.4f}, '
                         f'exact state sign-flip p={comparison["state_p"]:.4f}; '
                         'both must be below 0.05')
                lines.append(f'<text class="significance-star {star_class}" '
                             f'x="{x(comparison["steps"]):.1f}" y="{star_y}" text-anchor="middle" '
                             f'aria-label="{html.escape(title)}">'
                             f'<title>{html.escape(title)}</title>★</text>')
    for comparison in checkpoint_comparisons(task):
        if comparison["p"] < .05:
            title = (f'{comparison["steps"] / env_count:,.0f} physical steps/env: sparse − Robometer '
                     f'{comparison["difference"] * 100:+.1f} percentage points; '
                     f'n={comparison["n"]} paired seeds; exact two-sided p={comparison["p"]:.4f}')
            lines.append(f'<text class="significance-star reward-comparison-star" '
                         f'x="{x(comparison["steps"]):.1f}" y="58" '
                         f'text-anchor="middle" aria-label="{html.escape(title)}">'
                         f'<title>{html.escape(title)}</title>★</text>')
    axis_label = 'Physical training steps per environment (average)' if env_count > 1 else 'Physical training steps'
    lines.append(f'<text x="{width/2}" y="{height-5}" text-anchor="middle" class="axis">'
                 f'{axis_label}</text></svg>')
    return '<div class="axis-scale-note">Y-axis: linear · fixed 0–100%</div>' + "".join(lines)


def video_audit_url(task: dict, data_root: Path) -> str | None:
    """Match task and policy checkpoint; prefer comparisons from this experiment."""
    checkpoints = task.get('policy_checkpoints', set())
    if len(checkpoints) != 1:
        return None
    checkpoint = next(iter(checkpoints)).rstrip('/')
    candidates = []
    for path in sorted((data_root / 'video_audits').glob('*/manifest.json')):
        try:
            audit = json.loads(path.read_text())
            jobs = audit['jobs']
            cfg = jobs[0]['config']
            if (cfg['suite'] != task['suite'] or cfg['task_id'] != task['id'] or
                    str(cfg.get('pi0_checkpoint', '')).rstrip('/') != checkpoint):
                continue
            actors = [Path(job['actor_checkpoint']) for job in jobs if job.get('actor_checkpoint')]
            matching = bool(actors) and all(actor.is_relative_to(task['root']) for actor in actors)
            # A different fine-tuning experiment is not evidence for this curve.
            if actors and not matching:
                continue
            candidates.append(((matching, audit.get('sampling') != 'stochastic'), path.parent.name))
        except (OSError, ValueError, KeyError, IndexError, TypeError):
            continue
    if candidates:
        return '/video-audits/' + max(candidates)[1] + '/'
    if checkpoint.endswith('/pi05_libero') and task['suite'] == 'libero_90' and task['id'] in (2, 28):
        if (data_root / PI05_VIDEO_ROOT.name / f'task{task["id"]:02d}').is_dir():
            return f'/pi05-libero90-video-audit/#task-{task["id"]:02d}'
    if checkpoint.endswith('/pi0_libero'):
        if task['suite'] == 'libero_10' and task['id'] in (6, 8):
            if (data_root / 'pi0_libero10_video_audit_20261015' / f'task{task["id"]:02d}').is_dir():
                return f'/pi0-libero10-video-audit/#task-{task["id"]:02d}'
        if task['suite'] == 'libero_90' and task['id'] in (38, 55):
            return VIDEO_BASE if task['id'] == 38 else '/task55-pi0-baseline/'
    return None


def task_card(task: dict, data_root: Path | None = None) -> str:
    rows = []
    summary = []
    for kind, label, color in KINDS:
        completed = []
        for seed in task["seeds"]:
            run = task["runs"][kind][seed]
            if run['status'] == 'not selected':
                continue
            latest = run["evaluations"][-1] if run["evaluations"] else None
            if run["status"] == "complete" and latest:
                completed.append(latest["rate"])
            result = (f'{latest["successes"]}/{latest["episodes"]} ({latest["rate"]:.0%})'
                      if latest else "—")
            steps = f'{latest["steps"]:,}' if latest else "—"
            rows.append(f'<tr><td>{seed}</td><td>{html.escape(label)}</td>'
                        f'<td>{html.escape(run["status"])}</td><td>{steps}</td><td>{result}</td></tr>')
        if completed:
            summary.append(f'<span style="--series:{color}"><b>{html.escape(label)}:</b> '
                           f'{sum(completed)/len(completed):.1%} final mean '
                           f'({len(completed)} completed seed{"s" if len(completed) != 1 else ""})</span>')
    title = f'Task {task["id"]}: {task["name"].capitalize()}'
    audit_url = video_audit_url(task, data_root)
    title_markup = (f'<a class="audit-title" href="{html.escape(audit_url, quote=True)}" '
                    f'title="View video audit">{html.escape(title)} ↗</a>'
                    if audit_url else html.escape(title))
    baseline = task.get("baseline")
    policy_label = baseline.get('policy_label', 'π₀') if baseline else {
        'scratch': 'SAC', 'pi0.5': 'π₀.₅'}.get(task.get('base_model'), 'π₀')
    baseline_legend = (f'<a class="baseline-legend" href="{baseline["url"]}">'
                       f'Frozen {policy_label} checkpoint: {baseline["successes"]}/{baseline["episodes"]} '
                       f'({baseline["rate"]:.0%}; 95% {baseline.get("ci_method", "Wilson")} CI '
                       f'{baseline["ci_low"]:.0%}–{baseline["ci_high"]:.0%})</a>' if baseline else
                       '<span>No pretrained checkpoint (SAC from scratch).</span>' if task.get('base_model') == 'scratch' else
                       '<span>Frozen checkpoint baseline unavailable.</span>')
    star_legend = ('· <span class="sparse-baseline-star">★</span> green: sparse vs frozen π₀ '
                   '· <span class="robometer-baseline-star">★</span> orange: Robometer vs frozen π₀ '
                   '· <span class="reward-comparison-star">★</span> black: sparse vs Robometer '
                   '(pointwise p &lt; 0.05; hover for tests and sample sizes)')
    comparison = paired_comparison(task)
    if comparison["p"] is None:
        significance = (f'<div class="inference">Final paired comparison: {comparison["n"]} completed '
                        'seed pairs; at least 2 are needed for a confidence interval and test.</div>')
    else:
        verdict = ("Statistically significant at α = 0.05" if comparison["p"] < .05
                   else "Not statistically significant at α = 0.05")
        significance = (f'<div class="inference"><b>{verdict}.</b> Sparse − Robometer final success: '
                        f'{comparison["mean"] * 100:+.1f} percentage points '
                        f'(95% paired t CI {comparison["ci_low"] * 100:+.1f} to '
                        f'{comparison["ci_high"] * 100:+.1f} pp; '
                        f'two-sided exact paired sign-flip p = {comparison["p"]:.4f}; '
                        f'n = {comparison["n"]} completed pairs).</div>')
    suite_label = family_label(task.get('suite', 'libero_90'))
    state = task.get("orchestration_status")
    state_note = ('Queued: waiting for all task 06 seed pairs to finish' if state == "waiting_for_task06"
                  else state.capitalize() if state else "")
    state_markup = f'<p class="meta">{html.escape(state_note)}</p>' if state_note else ""
    analysis_note = html.escape(task.get('analysis_note') or '')
    analysis_url = task.get('analysis_url') or ''
    analysis_markup = (f'<p class="meta">{analysis_note} '
                       + (f'<a href="{html.escape(analysis_url, quote=True)}">Diagnostic analysis and videos</a>'
                          if analysis_url.startswith('/') else '') + '</p>' if analysis_note else '')
    by_seed = task.get('eval_episodes_by_seed')
    episode_note = (', '.join(f'seed {seed}: {count}' for seed, count in by_seed.items())
                    if by_seed else str(task.get('eval_episodes') or ''))
    baseline_protocol = (
        f'The frozen baseline pools {baseline["episodes"]} step-0 rollouts across '
        f'{baseline["seeds"]} seeds on {baseline["unique_states"]} reset states; '
        f'its 95% interval is clustered by seed.'
        if baseline and baseline.get('seeds') else
        'The frozen baseline uses 50 step-0 rollouts and a 95% interval clustered by reset state.'
        if baseline and baseline.get('unique_states') else '')
    protocol = (f'<p class="meta">Evaluation episodes per checkpoint ({html.escape(episode_note)}), '
                f'at step 0 and every {task["eval_every"]:,} total physical training steps; '
                f'{task["horizon"]}-step episode horizon. '
                f'{baseline_protocol}</p>'
                if episode_note and task.get('eval_every') and task.get('horizon') else '')
    parallel_note = ''
    if task.get('num_envs', 0) and task['num_envs'] > 1:
        count, chunk = task['num_envs'], task['action_exec_len']
        parallel_note = (f'<p class="meta"><b>{count} parallel training environments</b> · '
                         f'{chunk} executed actions/chunk.<br>'
                         f'Budget: {task["max_steps"] / count:,.0f} physical steps/env on average '
                         f'({task["max_steps"]:,} total). '
                         f'One full collection round = {chunk} x-axis steps = {count} replay transitions. '
                         f'X-axis = total physical steps / {count}; chunks can end early.</p>')
    return (f'<section class="card experiment-plot" id="{html.escape(task["anchor"])}"><div class="card-head">'
            f'<div><div class="eyebrow">{suite_label} · {policy_label} + {html.escape(task["algorithm"])}</div>'
            f'<h2>{title_markup}</h2>{experiment_date_label(task["root"])}</div>'
            f'<a href="#top" class="back">Back to top ↑</a></div>'
            f'<a class="diagnostics-link" href="#{html.escape(task["anchor"])}">Open experiment diagnostics →</a>'
            f'{state_markup}{parallel_note}<div class="summary">{"".join(summary)}</div>{plot(task)}'
            '<div class="compact-legend"><span style="color:#087f8c">━ Sparse</span> · '
            '<span style="color:#d36a35">━ Robometer</span> · Dotted: frozen checkpoint</div>'
            '<details><summary>Protocol, statistics and run details</summary>'
            f'{analysis_markup}{protocol}'
            f'{checkpoint_training_note(task, data_root)}'
            '<div class="legend"><span style="--series:#087f8c">Sparse simulator reward</span>'
            '<span style="--series:#d36a35">Robometer reward</span>'
            f'{baseline_legend}'
            '<small>Bold: mean · Shading: 95% seed-level t CI · Faint: individual seeds '
            f'{star_legend}</small></div>'
            f'{significance}'
            '<details><summary>Run details</summary><div class="table-wrap"><table>'
            '<thead><tr><th>Seed</th><th>Reward</th><th>Status</th>'
            '<th>Latest transitions</th><th>Latest success</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></details></details></section>')


def sac_task28(data_root: Path) -> dict | None:
    root = data_root / SAC_TASK28_ROOT
    planned = root / 'planned_suite.json'
    if not planned.exists():
        return None
    plan = json.loads(planned.read_text())['runs']
    seeds = sorted({int(row['seed']) for row in plan})
    runs = {kind: {} for kind, _, _ in KINDS}
    for row in plan:
        kind, seed = row['reward'], int(row['seed'])
        folder = root / row['name']
        status_file = folder / 'status.json'
        status = (json.loads(status_file.read_text()).get('status', 'queued')
                  if status_file.exists() else 'queued')
        runs[kind][seed] = {'status': status, 'evaluations': read_evaluations(folder / 'evaluation.csv')}
    return {'id': 28, 'name': task_map()['libero_90'][28].replace('_', ' '),
            'anchor': 'sac-task28-general', 'suite': 'libero_90', 'algorithm': 'SAC from scratch',
            'root': root, 'seeds': seeds, 'runs': runs, 'max_steps': 100000,
            'baseline': None}


def sac_task28_card(task: dict) -> str:
    rows = []
    for seed in task['seeds']:
        for kind, label, _ in KINDS:
            run = task['runs'][kind][seed]
            latest = run['evaluations'][-1] if run['evaluations'] else None
            outcome = f'{latest["successes"]}/{latest["episodes"]} ({latest["rate"]:.0%})' if latest else '—'
            steps = f'{latest["steps"]:,}' if latest else '—'
            rows.append(f'<tr><td>{seed}</td><td>{html.escape(label)}</td>'
                        f'<td>{html.escape(run["status"])}</td><td>{steps}</td><td>{outcome}</td></tr>')
    comparison = paired_comparison(task)
    completed, total = paired_seed_progress(task)
    return (f'<section class="card experiment-plot" id="{task["anchor"]}">'
            '<div class="card-head"><div><div class="eyebrow">LIBERO-90 · SAC from scratch</div>'
            '<h2>Task 28: Close the top drawer of the cabinet</h2>'
            f'{experiment_date_label(task["root"])}</div>'
            '<a href="#top" class="back">Back to top ↑</a></div>'
            '<p class="meta">Three paired seeds · 100,000 training transitions · '
            '25 held-out episodes at step 0 and every 10,000 transitions · '
            f'{completed}/{total} paired seeds complete. '
            'DINOv2-small external-camera features; simulator success determines evaluation.</p>'
            '<p class="training-note"><b>Reward checkpoint:</b> '
            '<code>robometer/Robometer-4B@beef63bc</code>, the general-purpose RBM-1M checkpoint. '
            'This is a separate comparison from the paper’s LIBERO-specific Figure 5 ablation. '
            'The sparse arm never loads a reward model.</p>'
            f'{plot(task)}'
            '<div class="legend"><span style="--series:#087f8c">Sparse simulator reward</span>'
            '<span style="--series:#d36a35">Robometer reward</span>'
            '<small>Bold: seed mean · Shading: 95% seed-level t CI · Faint: individual seeds. '
            'The exact paired test cannot reach p &lt; 0.05 with three seeds.</small></div>'
            f'<p class="meta">Final paired comparison: {comparison["n"]} of {total} seed pairs complete.</p>'
            '<details><summary>Run details</summary><div class="table-wrap"><table>'
            '<thead><tr><th>Seed</th><th>Reward</th><th>Status</th>'
            '<th>Latest transitions</th><th>Latest success</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div></details></section>')


def experiment_tracker(data_root: Path, tasks: list[dict]) -> str:
    """Read current run files on every page request."""
    rows = []
    family_root = data_root / "pi0_libero_families_20261014"
    family_names = task_map()
    family_done = sum(family_completed(family_task_dir(family_root, suite, index, name))
                      for suite in FAMILY_SUITES for index, name in enumerate(family_names[suite]))
    family_target = sum(len(family_names[suite]) for suite in FAMILY_SUITES)
    rows.append('<tr><td><a href="/pi0-libero-families-eval/">Four-family π₀ baseline</a></td>'
                f'<td>{family_done}/{family_target} tasks evaluated</td><td>π₀: in distribution · '
                'Robometer: in distribution at suite level</td></tr>')
    pi05_done = sum(pi05_completed(pi05_task_dir(data_root / PI05_ROOT.name, suite, index, name))
                    for suite, index, name in pi05_tasks())
    rows.append('<tr><td><a href="/pi05-libero-eval/">All-family π₀.₅ evaluation</a></td>'
                f'<td>{pi05_done}/130 tasks evaluated · 25 episodes each</td>'
                '<td>π₀.₅: Spatial, Object, Goal, LIBERO-10 in distribution; '
                'LIBERO-90 out of distribution at suite level</td></tr>')
    for task in tasks:
        completed, _ = paired_seed_progress(task)
        suite_label = family_label(task.get('suite', 'libero_90'))
        if task.get('base_model') is not None:
            coverage_label = 'See checkpoint training note in the plot; custom checkpoint coverage is unverified'
        elif task.get("suite") == "libero_10":
            provenance = training_coverage(data_root)
            policy_known = (task["policy_checkpoints"] == {provenance["policy_checkpoint"]} and
                            provenance["policy_known"])
            reward_known = (task["reward_checkpoints"] == {provenance["reward_checkpoint"]} and
                            provenance["reward_known"] and
                            RBM_DATASETS["libero_10"] in provenance["reward_datasets"])
            coverage_label = (f'π₀: {"Yes · LIBERO-10 suite" if policy_known else "Unverified"} · '
                              f'Robometer: {"Yes · LIBERO-10 suite" if reward_known else "Unverified"}')
        else:
            policy_label = 'π₀.₅' if task['root'].name.startswith('pi05_') else 'π₀'
            coverage_label = f'{policy_label}: out of distribution · Robometer: out of distribution at suite level'
        rows.append(f'<tr><td><a href="#{html.escape(task["anchor"])}">'
                    f'{suite_label} task {task["id"]:02d} · {html.escape(task["algorithm"])}</a></td>'
                    f'<td>{"Waiting for task 06 · " if task.get("orchestration_status") == "waiting_for_task06" else ""}'
                    f'{completed}/{len(task["seeds"])} seed groups complete at '
                    f'{task["max_steps"]:,} transitions</td><td>{coverage_label}</td></tr>')
    for task_id in (6, 8):
        path = data_root / "pi0_libero10_video_audit_20261015" / f"task{task_id:02d}" / "status.json"
        try:
            status = json.loads(path.read_text())
        except (OSError, ValueError):
            status = {}
        phase = html.escape(str(status.get("status", "queued")))
        coverage = audit_training_coverage(data_root, task_id)
        policy_label = "Yes · LIBERO-10 suite" if coverage["policy"] else "Unverified"
        reward_label = "Yes · LIBERO-10 suite" if coverage["reward"] else "Unverified"
        rows.append(f'<tr><td><a href="/pi0-libero10-video-audit/#task-{task_id:02d}">'
                    f'LIBERO-10 task {task_id:02d} · video audit</a></td>'
                    f'<td>{phase}: {status.get("demos", 0)}/4 demos, '
                    f'{status.get("policy_successes", 0)}/4 successes, '
                    f'{status.get("policy_failures", 0)}/4 failures, '
                    f'{status.get("videos_scored", 0)}/12 scored</td>'
                    f'<td>π₀: {policy_label} · Robometer: {reward_label}</td></tr>')
    return ('<section class="card" id="experiment-progress"><h2>Experiment progress</h2>'
            '<p class="meta">Checkpoint labels refer to the released <code>pi0_libero</code> and '
            '<code>pi05_libero</code> policies and the pinned <code>Robometer-4B</code> reward model. Distribution labels are '
            'based on suite membership in their training mixes; exact trajectory overlap is unknown. '
            'RL rows track paired sparse-reward and Robometer-reward runs.</p>'
            '<div class="table-wrap"><table><thead><tr><th>Experiment</th><th>Live progress</th>'
            '<th>Initial checkpoint training coverage</th></tr></thead><tbody>'
            + "".join(rows) + '</tbody></table></div></section>')


def paired_seed_progress(task: dict) -> tuple[int, int]:
    finished = sum(
        all(task["runs"][kind][seed]["status"] == "complete" and
            task["runs"][kind][seed]["evaluations"] and
            task["runs"][kind][seed]["evaluations"][-1]["steps"] >= task["max_steps"]
            for kind in task.get('selected_rewards', ['sparse', 'robometer'])
            if task["runs"][kind][seed]["status"] != 'not selected')
        for seed in task["seeds"]
    )
    return finished, len(task["seeds"])


def sidebar(data_root: Path, tasks: list[dict]) -> str:
    """Compact, live progress and navigation for the three experiment groups."""
    names = task_map()
    family_root = data_root / "pi0_libero_families_20261014"
    family_done = sum(family_completed(family_task_dir(family_root, suite, index, name))
                      for suite in FAMILY_SUITES for index, name in enumerate(names[suite]))
    pi05_root = data_root / PI05_ROOT.name
    pi05_done = sum(pi05_completed(pi05_task_dir(pi05_root, suite, index, name))
                    for suite, index, name in pi05_tasks())
    pi0_90_root = data_root / "pi0_task_probe_20261010"
    pi0_90_done = sum(
        (path.parent / "status.json").exists() and
        json.loads((path.parent / "status.json").read_text()).get("status") == "complete"
        for path in pi0_90_root.glob("[0-8][0-9]_*/summary.json")
    )
    frozen = (
        ('π₀.₅ · all LIBERO', '/pi05-libero-eval/', f'{pi05_done}/130'),
        ('π₀ · LIBERO-90', '/pi0-libero90-eval/', f'{pi0_90_done}/90'),
        ('π₀ · four families', '/pi0-libero-families-eval/', f'{family_done}/40'),
    )
    frozen_roots = {'/pi05-libero-eval/': pi05_root, '/pi0-libero90-eval/': pi0_90_root,
                    '/pi0-libero-families-eval/': family_root}
    frozen = sorted(frozen, key=lambda row: experiment_date(frozen_roots[row[1]]), reverse=True)
    frozen_links = ''.join(
        f'<a class="side-link" href="{url}"><span>{label}</span><b>{progress}</b></a>'
        for label, url, progress in frozen)
    sac = sac_task28(data_root)
    fine_links = []
    options = ['<option value="all">All fine-tuning plots</option>']
    official = [{'root': root, 'official': True} for root in official_dsrl.roots(data_root)]
    for task in newest_first(tasks + ([sac] if sac else []) + official):
        if task.get('official'):
            root = task['root']
            metadata = json.loads((root / 'manifest.json').read_text())
            status = json.loads((root / 'status.json').read_text()) if (root / 'status.json').exists() else {}
            label = f"Robomimic {metadata['task'].capitalize()} · {metadata['algorithm']}"
            anchor = html.escape('official-' + root.name, quote=True)
            progress = status.get('env_steps', 0) / metadata['physical_steps']
            fine_links.append(f'<a class="side-link plot-link" href="#{anchor}">'
                              f'<span>{html.escape(label)}</span><b>{progress:.0%}</b></a>')
            options.append(f'<option value="{anchor}">{html.escape(label)}</option>')
            continue
        suite = family_label(task.get('suite', 'libero_90'))
        label = f'{suite} task {task["id"]:02d} · {task["algorithm"]}'
        anchor = html.escape(task['anchor'], quote=True)
        finished, total = paired_seed_progress(task)
        fine_links.append(f'<a class="side-link plot-link" href="#{anchor}">'
                          f'<span>{html.escape(label)}</span><b>{finished}/{total}</b></a>')
        options.append(f'<option value="{anchor}">{html.escape(label)}</option>')
    videos = []
    for task_id in (6, 8):
        path = data_root / 'pi0_libero10_video_audit_20261015' / f'task{task_id:02d}' / 'status.json'
        try:
            status = json.loads(path.read_text())
        except (OSError, ValueError):
            status = {}
        count = int(status.get('videos_scored', 0))
        target = int(status.get('videos_target', 12))
        videos.append(f'<a class="side-link" href="/pi0-libero10-video-audit/#task-{task_id:02d}">'
                      f'<span>LIBERO-10 task {task_id:02d}</span><b>{count}/{target}</b></a>')
    pi05_scored = 0
    pi05_episodes = 0
    for task_id in (2, 28):
        folder = data_root / PI05_VIDEO_ROOT.name / f'task{task_id:02d}'
        path = folder / 'status.json'
        try:
            pi05_scored += int(json.loads(path.read_text()).get('videos_scored', 0))
        except (OSError, ValueError):
            pass
        try:
            pi05_episodes += len(json.loads((folder / 'policy/episodes.json').read_text()))
        except (OSError, ValueError):
            pass
    videos.append('<a class="side-link" href="/pi05-libero90-video-audit/">'
                  '<span>π₀.₅ · LIBERO-90 tasks 02 &amp; 28'
                  f'<small>{pi05_scored}/16 videos scored</small></span><b>{pi05_episodes}/50 eval</b></a>')
    videos.extend((
        f'<a class="side-link" href="{VIDEO_BASE}"><span>Task 38 · π₀ videos</span><b>View</b></a>',
        '<a class="side-link" href="/task55-pi0-baseline/"><span>Task 55 · π₀ videos</span><b>View</b></a>',
    ))
    for manifest_path in sorted((data_root / 'video_audits').glob('*/manifest.json')):
        audit = json.loads(manifest_path.read_text())
        cfg = audit['jobs'][0]['config']
        if all(not job.get('actor_checkpoint') for job in audit['jobs']):
            audit_label = ('demonstrations + frozen policy' if any(job.get('kind') == 'demonstrations'
                                                                  for job in audit['jobs']) else 'frozen policy')
        else:
            audit_label = audit.get('sampling', 'deterministic') + ' policy comparison'
        finished = sum((manifest_path.parent / job['variant'] / 'status.json').exists() and
                       json.loads((manifest_path.parent / job['variant'] / 'status.json').read_text()).get('status') == 'complete'
                       for job in audit['jobs'])
        videos.append(f'<a class="side-link" href="/video-audits/{html.escape(manifest_path.parent.name)}/">'
                      f'<span>{family_label(cfg["suite"])} task {cfg["task_id"]:02d} · '
                      f'{html.escape(audit_label)}</span>'
                      f'<b>{finished}/{len(audit["jobs"])}</b></a>')
    video_roots = {
        '/pi0-libero10-video-audit/#task-06': data_root / 'pi0_libero10_video_audit_20261015',
        '/pi0-libero10-video-audit/#task-08': data_root / 'pi0_libero10_video_audit_20261015',
        '/pi05-libero90-video-audit/': data_root / PI05_VIDEO_ROOT.name,
        VIDEO_BASE: data_root / 'pi0_task38_video_audit_20261014',
        '/task55-pi0-baseline/': data_root / 'pi0_task55_video_audit_20261014',
    }
    for path in (data_root / 'video_audits').glob('*/manifest.json'):
        video_roots[f'/video-audits/{path.parent.name}/'] = path.parent
    videos.sort(key=lambda markup: experiment_date(video_roots[re.search(r'href="([^"]+)"', markup).group(1)]),
                reverse=True)
    fine_markup = ''.join(fine_links) or '<p class="side-empty">No fine-tuning plots yet.</p>'
    return (f'<aside class="sidebar" aria-label="Experiment progress and navigation">'
            '<a class="sidebar-home" href="#top">LIBERO results</a>'
            '<p class="side-caption">Live experiment progress</p>'
            '<nav aria-label="Experiment groups">'
            '<a class="side-link" href="/ogpo/"><span>OGPO · LIBERO fine-tuning<small>π₀.₅ features · live results</small></span><b>↗</b></a>'
            '<section class="side-group"><h2>Frozen Eval</h2>' + frozen_links + '</section>'
            '<section class="side-group"><h2>Fine-tuning</h2>'
            '<label class="side-label" for="plot-picker">Experiment plots</label>'
            '<select id="plot-picker" aria-label="Choose fine-tuning plot">' + ''.join(options) + '</select>'
            + fine_markup + '</section>'
            '<section class="side-group"><h2>Video Audit</h2>' + ''.join(videos) + '</section>'
            '</nav><p class="side-caption">Counts refresh with the page every 60 seconds.</p></aside>')


def algorithm_comparison(group, tasks, anchor):
    from experiments import ogpo_dashboard as ogpo
    spec = group[0]['spec']
    native = spec.get('native_pi05_lora',False)
    actor_description = 'original π₀.₅ head + LoRA, no BC' if native else 'π₀.₅ features, separate BC head'
    sampler_description = 'SDE is single-sample EMA inference, no critic ranking.' if native else 'SDE means best-of-8 with EMA actor and critic ranking.'
    candidates = [t for t in tasks if t['suite'] == spec['suite'] and t['id'] == spec['task_id'] and 'DSRL' in t['algorithm']]
    candidates.sort(key=lambda t: ((t.get('num_envs') or 1) != 1, t['max_steps']))
    if not candidates:
        return ''
    selection_key = '|'.join(sorted(t['name'] for t in group))
    options, panels = [], []
    for index, task in enumerate(candidates):
        count = task.get('num_envs') or 1
        model = task.get('base_model') or ('pi0.5' if task['root'].name.startswith('pi05') else 'pi0')
        label = f"{model} · {count} env(s) · {task['max_steps']/count:,.0f} steps/env · {len(task['seeds'])} planned seeds"
        options.append(f'<option value="{html.escape(task["anchor"], quote=True)}">{label}</option>')
        for mode in ('ODE', 'SDE'):
            series, intervals, notes = [], {}, []
            seed_series = {}
            for kind, label, color in KINDS:
                name = 'DSRL ' + ('sparse' if kind == 'sparse' else 'Robometer')
                curves = [r['evaluations'] for r in task['runs'][kind].values() if r['evaluations']]
                values = aggregate(curves)
                series.append((name, color, [(v['steps']/count,v['rate']) for v in values]))
                intervals[name] = [(v['steps']/count,v['ci_low'],v['ci_high']) for v in values if v['ci_low'] is not None]
                seed_series[name] = [(seed, [(v['steps']/count, v['rate']) for v in run['evaluations']]) for seed, run in task['runs'][kind].items() if run['evaluations']]
                notes.append(f"DSRL {kind}: {len(curves)} seed(s) with evaluations")
            for curve in ogpo.aggregate_reward_curves(group, mode):
                name = 'OGPO ' + curve['name']
                color = '#a855f7' if curve['name'].startswith('Robometer') else '#2563eb'
                series.append((name,color,curve['points']))
                intervals[name] = curve['bounds']
                seed_series[name] = curve['seed_series']
            bc_series, bc_status = ogpo.shared_bc(group, mode)
            for name, color, ps in bc_series:
                series.append((name, color, ps))
                parent = next(t['bc_run'] for t in group if t.get('bc_run'))
                intervals[name] = [(x, *ogpo.wilson(y, parent['n'])) for x, y in ps]
            xlabel = 'Count: ◇ BC updates / ● RL physical steps per env' if bc_status else 'Physical RL steps per environment'
            plot = bc_status + ogpo.chart(series,'DSRL vs OGPO',xlabel=xlabel,probability=True,intervals=intervals,seed_series=seed_series)
            hidden = '' if index==0 and mode=='ODE' else ' hidden'
            panels.append(f'<div data-comparison="{html.escape(task["anchor"], quote=True)}-{mode}"{hidden}>{plot}<p class="meta">{model} DSRL: {count} training env(s), up to {task["action_exec_len"]} actions/chunk. {"; ".join(notes)}. OGPO: {actor_description}, 1 env, 4 actions/chunk, {len({t["spec"]["seed"] for t in group})} planned RL seeds per reward.</p><a href="#{task["anchor"]}">Original DSRL experiment and baseline →</a></div>')
    return f'''<div class="algorithm-comparison" data-selection-key="{html.escape(selection_key, quote=True)}"><h3>DSRL and OGPO · same physical-step axis</h3><label>DSRL run <select class="dsrl-choice" onchange="updateAlgorithmComparison(this)">{''.join(options)}</select></label> <label>OGPO evaluation <select class="ogpo-choice" onchange="updateAlgorithmComparison(this)"><option>ODE</option><option>SDE</option></select></label>{''.join(panels)}<p class="meta">Teal/orange: DSRL sparse/Robometer. Blue/purple: OGPO sparse/Robometer. {sampler_description} DSRL bands: 95% t intervals across seeds (omitted for one seed). OGPO bands: 95% t intervals across available RL seeds when n ≥ 2; Wilson episode intervals for one seed. Seeds share one BC checkpoint. Methods have different starting policies and BC data; this is descriptive, not a controlled algorithm comparison. Each curve includes its own step-zero baseline; no shared baseline is assumed.</p></div>'''


def render(data_root: Path) -> bytes:
    tasks = []
    errors = []
    for root in task_roots(data_root):
        try:
            tasks.append(load_task(root))
        except (OSError, ValueError, KeyError) as error:
            errors.append(f'{root.name}: {error}')
    from experiments import ogpo_dashboard
    ogpo = []
    for group in ogpo_dashboard.comparison_groups(ogpo_dashboard.load(data_root)):
        if len(group) < 2 or {t['spec'].get('reward') for t in group} != {'sparse_minus_one', 'robometer'}:
            continue
        first = group[0]
        anchor = 'ogpo-' + first['name']
        label = f"OGPO · {family_label(first['spec']['suite'])} task {first['spec']['task_id']:02d}"
        if first['spec'].get('native_pi05_lora'): label += ' · native π₀.₅ LoRA'
        label += f" · {first['budget']/1000:g}k steps · {len({t['spec']['seed'] for t in group})} seeds"
        combined = algorithm_comparison(group, tasks, anchor)
        markup = ('<article class="card"><h2>' + label.replace('OGPO · ', '') + ' · DSRL vs OGPO</h2>' + combined + '</article>') if combined else ogpo_dashboard.comparison_card(group, {t['name']: '' for t in group})
        markup = markup.replace('<article class="card">', f'<article class="card experiment-plot ogpo-plot" id="{anchor}"><div class="eyebrow">DSRL / OGPO comparison</div>', 1)
        runtime = ''.join(f'<p class="meta">OGPO {t["spec"]["reward"]}: {html.escape(t["state"])} · {html.escape(t["runtime"])}<br><b>{html.escape(t["eta"] or ("Completed · no time remaining" if t["state"] == "Completed" else t["state"] + " · completion ETA unavailable"))}</b></p>' for t in group)
        if combined: runtime += ogpo_dashboard.curve_style_note()
        runtime += '<p class="meta">' + html.escape(first['spec'].get('note', '')) + '</p>'
        markup = markup.replace('</article>', runtime + f'<a href="/ogpo/#{first["name"]}">OGPO diagnostics and full experiment details →</a></article>')
        ogpo.append(dict(root=first['root'], ogpo=True, anchor=anchor, label=label, markup=markup))
    sidebar_markup = sidebar(data_root, tasks)
    extra_options = ''.join(f'<option value="{t["anchor"]}">{t["label"]}</option>' for t in ogpo)
    sidebar_markup = sidebar_markup.replace('</select>', extra_options+'</select>', 1)
    sac = sac_task28(data_root)
    official = [{'root': root, 'official': True} for root in official_dsrl.roots(data_root)]
    ordered = newest_first(tasks + ([sac] if sac else []) + official + ogpo)
    cards = "".join(task["markup"] if task.get("ogpo") else official_dsrl.card(task['root']) if task.get('official') else
                    sac_task28_card(task) if task is sac else task_card(task, data_root) for task in ordered)
    if not ordered:
        cards = '<section class="card"><p>No task runs found yet. New runs appear here automatically.</p></section>'
    error_text = (f'<p class="error">Could not read: {html.escape("; ".join(errors))}</p>'
                  if errors else "")
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60"><title>LIBERO reward-learning results</title>
<style>
:root{{--ink:#17313b;--muted:#5d6e73;--line:#dce5e6;--paper:#fff;--bg:#f3f7f7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
.dashboard-layout{{display:grid;grid-template-columns:280px minmax(0,1fr);max-width:1650px;margin:auto}}
main{{min-width:0;max-width:none;width:100%;margin:0 auto;align-self:start;padding:22px 18px 40px}}
.sidebar{{position:sticky;top:0;height:100vh;overflow-y:auto;padding:28px 18px;background:#eaf1f1;border-right:1px solid var(--line)}}
.sidebar-home{{display:block;font-size:19px;font-weight:800;text-decoration:none;color:var(--ink);margin-bottom:2px}}
.side-caption{{font-size:12px;color:var(--muted);margin:0 0 18px}}
.side-group{{border-top:1px solid #cbdadb;padding:17px 0 19px}}.side-group h2{{font-size:12px;letter-spacing:.09em;text-transform:uppercase;color:#46656b;margin:0 0 10px}}
.side-link{{display:flex;align-items:flex-start;justify-content:space-between;gap:8px;padding:8px 9px;border-radius:8px;color:var(--ink);text-decoration:none;font-size:13px;line-height:1.35}}
.side-link:hover,.side-link:focus-visible{{background:#dce9e9}}.side-link span{{min-width:0}}.side-link b{{color:#087f8c;white-space:nowrap;font-size:12px}}
.side-link small{{display:block;color:var(--muted);font-size:11px;margin-top:2px}}
.side-label{{display:block;color:var(--muted);font-size:12px;margin:8px 0 5px}}
#plot-picker{{width:100%;padding:9px 30px 9px 9px;border:1px solid #b8ccce;border-radius:8px;background:white;color:var(--ink);font:inherit;font-size:13px;margin-bottom:8px}}
.side-empty{{font-size:13px;color:var(--muted)}}.detail-progress{{margin-top:28px}}.detail-progress>summary{{padding:12px 0}}
@media(max-width:850px){{.dashboard-layout{{display:block}}.sidebar{{position:static;height:auto;border-right:0;border-bottom:1px solid var(--line);padding:17px 20px}}.sidebar nav{{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px}}.side-group{{padding-bottom:0}}main{{padding:24px 18px 60px}}}}
h1{{font-size:clamp(26px,3vw,34px);line-height:1.12;margin:8px 0}}h2{{font-size:22px;line-height:1.25;margin:4px 0 0}}
p{{margin:8px 0}}a{{color:#087f8c}}.hero{{margin-bottom:16px}}.eyebrow{{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#577176;font-weight:700}}
.meta{{color:var(--muted);font-size:14px}}.task-nav{{display:flex;gap:10px;flex-wrap:wrap;margin-top:20px}}
.task-nav a{{padding:8px 13px;background:white;border:1px solid var(--line);border-radius:9px;text-decoration:none;font-weight:650}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:16px;padding:25px;margin:22px 0;box-shadow:0 5px 22px #17313b09}}
.card-head{{display:flex;justify-content:space-between;gap:16px;align-items:start}}.back{{font-size:13px;white-space:nowrap}}
.summary{{display:flex;gap:18px;flex-wrap:wrap;margin:20px 0 10px}}.summary span{{border-left:4px solid var(--series);padding:4px 0 4px 10px}}
svg{{display:block;width:100%;height:auto;margin:8px auto 4px}}.grid{{stroke:#e7eeee;stroke-width:1}}.tick,.axis{{fill:#64797e;font-size:12px}}
.axis{{font-size:13px}}.legend{{display:flex;align-items:center;gap:18px;flex-wrap:wrap;margin:6px 0 18px;font-size:14px}}
.training-note{{font-size:13px;color:var(--muted);margin:4px 0 10px}}
.significance-star{{font-size:19px;font-weight:700}}
.significance-star.sparse-baseline-star{{fill:#16803c}}
.significance-star.robometer-baseline-star{{fill:#d36a35}}
.significance-star.reward-comparison-star{{fill:#111827}}
.legend .sparse-baseline-star{{color:#16803c;font-weight:700}}
.legend .robometer-baseline-star{{color:#d36a35;font-weight:700}}
.legend .reward-comparison-star{{color:#111827;font-weight:700}}
.baseline-label{{fill:#475569;font-size:13px;font-weight:700;paint-order:stroke;stroke:white;stroke-width:4px}}
.legend .baseline-legend:before{{content:"";display:inline-block;width:18px;border-top:3px dotted #475569;vertical-align:middle;margin-right:7px}}
.legend span[style*="--series"]:before{{content:"";display:inline-block;width:18px;height:4px;background:var(--series);vertical-align:middle;margin-right:7px;border-radius:3px}}
.legend small{{color:var(--muted)}}details{{border-top:1px solid var(--line);padding-top:12px}}summary{{cursor:pointer;font-weight:650}}
.inference{{background:#f3f7f7;border-radius:9px;padding:11px 14px;margin:10px 0 20px;font-size:14px}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;margin-top:12px;font-size:14px}}th,td{{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line)}}
th{{color:#536c72}}.error{{color:#9b3a2b}}
.plot-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:10px;align-items:start}}
.experiment-plot{{min-width:0;margin:0;padding:12px;border-radius:12px;box-shadow:none}}
.experiment-plot[hidden]{{display:none}}.experiment-plot .card-head h2{{font-size:14px}}
.experiment-plot .back{{display:none}}.experiment-plot .eyebrow{{font-size:10px}}
.experiment-plot .summary{{gap:4px;margin:6px 0 0;font-size:11px}}
.experiment-plot .summary span{{padding:2px 0 2px 6px}}
.experiment-plot .meta,.experiment-plot .legend,.experiment-plot .inference{{font-size:12px}}
.experiment-plot .meta{{margin:5px 0}}.experiment-plot details{{margin-top:10px;font-size:12px}}
.axis-scale-note{{font-size:10px;color:var(--muted);margin-top:6px}}
.compact-legend{{font-size:12px;color:var(--muted)}}.audit-title{{text-decoration:none}}
.audit-title:hover,.audit-title:focus-visible{{text-decoration:underline}}
.experiment-date{{display:block;font-size:11px;font-weight:400;color:var(--muted);margin-top:4px}}
#fine-tuning-plots>h2{{margin-bottom:16px}}
.diagnostics-link{{display:inline-block;font-size:11px;margin-top:5px}}
#diagnostics-panel{{margin:14px 0 0;background:white;border:1px solid var(--line);border-radius:12px;padding:12px}}
#diagnostics-panel[hidden]{{display:none}}#diagnostics-panel h2{{font-size:18px}}
#diagnostics-panel .meta{{font-size:11px;margin:4px 0}}
.diagnostic-group h3{{font-size:13px;margin:12px 0 5px}}
.diagnostic-grid{{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:6px}}
.diagnostic-chart{{min-width:0;border:1px solid var(--line);border-radius:6px;padding:6px}}
.diagnostic-chart h4{{font-size:11px;margin:0;font-weight:600;cursor:help}}
.diagnostic-chart svg{{width:100%;margin:2px 0 0}}.diagnostic-chart text{{font-size:10px;fill:#61757a}}
.ogpo-plot svg{{width:100%;display:block}}.ogpo-plot svg text{{font:10px system-ui;fill:#5d6e73}}.ogpo-plot .grid{{stroke:#e0e8e8}}
.diagnostic-empty{{font-size:11px;color:var(--muted);min-height:75px;display:grid;align-items:center}}
@media(max-width:1350px){{.plot-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}
@media(max-width:1050px){{.diagnostic-grid{{grid-template-columns:repeat(3,minmax(0,1fr))}}}}
@media(max-width:620px){{.plot-grid{{grid-template-columns:1fr}}.diagnostic-grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}}


</style></head><body><div class="dashboard-layout">{sidebar_markup}<main id="top"><div class="hero"><div class="eyebrow">Reward-modeling experiments</div>
<h1>LIBERO fine-tuning results</h1><p>Held-out simulator success versus physical training steps per environment. Click a linked plot title to open its video audit.</p>
{ogpo_dashboard.gradient_explainer()}
<section class="card" aria-label="DSRL step counters"><h2>DSRL x-axis: physical steps vs replay transitions</h2>
<p><b>The x-axis counts physical training steps per environment, averaged across parallel environments.</b> It does not count action chunks, replay samples, or optimizer updates. Evaluation rollouts and reset/settling steps are excluded.</p>
<p class="meta"><b>Physical step:</b> one robot action executed by one simulator environment.<br>
<b>Replay transition:</b> one DSRL noise/steering decision and its executed action chunk in one environment, stored as one experience record with a chunk reward and the resulting observation.<br>
<b>Collection round:</b> each of N environments executes up to K actions from one chunk. A full round adds N × K total physical steps, K x-axis steps, and N replay transitions.</p>
<p><b>At x-axis value X: total physical steps = N × X; collected replay transitions ≈ N × X / K.</b> K is the number of actions actually executed per chunk, not the model’s predicted horizon or flow-matching integration steps. Check each experiment’s chunk length and environment count.</p>
<table><thead><tr><th>Example</th><th>X-axis</th><th>Total physical steps</th><th>Collected replay transitions (approximately)</th></tr></thead><tbody>
<tr><td>1 env, 10 actions/chunk</td><td>100k</td><td>100k</td><td>10k</td></tr>
<tr><td>4 envs, 8 actions/chunk</td><td>200k</td><td>800k</td><td>100k total (25k/env)</td></tr>
<tr><td>4 envs, 10 actions/chunk</td><td>200k</td><td>800k</td><td>80k total (20k/env)</td></tr>
</tbody></table><p class="meta">Chunks can stop early on success, timeout, or the training budget, so the exact replay count can exceed the full-chunk estimate. Collected replay transitions are cumulative; current replay-buffer size can be smaller after eviction. Sampling the same transition repeatedly for gradient updates does not advance the x-axis. With one environment, per-environment and total physical steps are identical. SAC-from-scratch stores one transition per physical step; OGPO comparisons below use physical training steps in one environment, with one replay transition per physical step; their detailed diagnostics remain on the OGPO dashboard.</p></section>
<p class="meta"><a href="/ogpo/">OGPO experiments · live dashboard ↗</a></p>
<p class="meta"><a href="/reports/reward_learning_summary_20261028/">Concise experiment report · 2026-10-28 ↗</a></p>
<details class="method-notes"><summary>How to read the plots and significance markers</summary><p class="meta">Colored bands are 95% t intervals across available independent RL seeds at each evaluation checkpoint (shown for n ≥ 2; clipped to 0–100% on the plot). The gray band around each dotted frozen-π₀ baseline is a 95% Wilson interval over its 25 evaluation episodes; it does not measure checkpoint or training-seed variation. On every task plot, green stars compare sparse reward with frozen π₀, orange stars compare Robometer reward with frozen π₀, and black stars compare sparse with Robometer reward. The black-star test is a two-sided exact paired sign-flip test across matched training seeds. Baseline stars require p &lt; 0.05 in both an exact seed-level sign-flip test and an exact matched-state sign-flip test of seed-mean outcomes. Hover for sample sizes, effect, and p-values. These pointwise tests are unadjusted for testing multiple checkpoints. Final comparisons use only matching seed pairs completed at the target transition count. The paired t interval summarizes final effect size and may disagree with the exact test for small samples. With five paired seeds, the smallest possible two-sided exact p-value is 0.0625, so a five-seed comparison cannot receive a significance star at α = 0.05.</p></details>
<p class="meta">Refreshes every 60 seconds · Latest read: {updated} UTC · All curves are read from local evaluation files</p>
{error_text}</div><section id="fine-tuning-plots" aria-label="Fine-tuning plots"><h2>Fine-tuning plots · newest first</h2>{ogpo_dashboard.sde_controls()}<div class="plot-grid">{cards}</div></section>
<section id="diagnostics-panel" hidden aria-live="polite"></section>
<details class="detail-progress"><summary>Detailed experiment progress and training coverage</summary>{experiment_tracker(data_root, tasks)}</details>
</main></div><script>
function updateAlgorithmComparison(select, persist=true){{
 const card=select.closest('.algorithm-comparison');
 const dsrl=card.querySelector('.dsrl-choice').value;
 const mode=card.querySelector('.ogpo-choice').value;
 const key=dsrl+'-'+mode;
 for(const panel of card.querySelectorAll('[data-comparison]'))panel.hidden=panel.dataset.comparison!==key;
 if(persist){{
  try{{localStorage.setItem('libero-comparison-v1:'+card.dataset.selectionKey,JSON.stringify({{dsrl,mode}}));}}catch(error){{}}
 }}
}}
for(const card of document.querySelectorAll('.algorithm-comparison')){{
 try{{
  const saved=JSON.parse(localStorage.getItem('libero-comparison-v1:'+card.dataset.selectionKey)||'null');
  if(saved){{
   for(const [selector,value] of [['.dsrl-choice',saved.dsrl],['.ogpo-choice',saved.mode]]){{
    const select=card.querySelector(selector);
    if([...select.options].some(option=>option.value===value))select.value=value;
   }}
  }}
 }}catch(error){{}}
 updateAlgorithmComparison(card.querySelector('.dsrl-choice'),false);
}}
const picker=document.getElementById('plot-picker');
const plots=[...document.querySelectorAll('.experiment-plot')];
const diagnostics=document.getElementById('diagnostics-panel');
let diagnosticsRequest=0;
async function loadDiagnostics(id){{
  const request=++diagnosticsRequest;
  diagnostics.hidden=id==='all';
  if(id==='all')return;
  if(id.startsWith('ogpo-')){{diagnostics.hidden=true;return;}}
  diagnostics.innerHTML='<p class="meta">Loading DSRL diagnostics…</p>';
  try{{
    const response=await fetch('/diagnostics/'+encodeURIComponent(id),{{cache:'no-store'}});
    if(!response.ok)throw new Error('Could not load diagnostics');
    const content=await response.text();
    if(request===diagnosticsRequest)diagnostics.innerHTML=content;
  }}catch(error){{if(request===diagnosticsRequest)diagnostics.textContent=error.message;}}
}}
function showPlot(id,scroll){{
  const chosen=plots.some(plot=>plot.id===id)?id:'all';
  picker.value=chosen;
  loadDiagnostics(chosen);
  for(const plot of plots)plot.hidden=chosen!=='all'&&plot.id!==chosen;
  if(scroll&&chosen!=='all')document.getElementById(chosen).scrollIntoView({{behavior:'smooth',block:'start'}});
}}
picker.addEventListener('change',()=>{{
  const id=picker.value;
  if(id==='all'){{history.replaceState(null,'',location.pathname);showPlot('all',false);}}
  else{{history.replaceState(null,'','#'+id);showPlot(id,true);}}
}});
window.addEventListener('hashchange',()=>showPlot(decodeURIComponent(location.hash.slice(1)),true));
showPlot(decodeURIComponent(location.hash.slice(1)),false);
</script>{ogpo_dashboard.sde_visibility_script()}</body></html>'''
    return page.encode("utf-8")


def serve(data_root: Path, host: str, port: int) -> None:
    class Handler(BaseHTTPRequestHandler):
        def serve_video(self, path: Path) -> None:
            if not path.is_file():
                self.send_error(404)
                return
            size = path.stat().st_size
            start, end = 0, size - 1
            requested = self.headers.get("Range")
            if requested:
                match = re.fullmatch(r"bytes=(\d+)-(\d*)", requested)
                if not match:
                    self.send_error(416)
                    return
                start = int(match.group(1))
                end = min(int(match.group(2)), size - 1) if match.group(2) else size - 1
                if start >= size or start > end:
                    self.send_error(416)
                    return
            self.send_response(206 if requested else 200)
            self.send_header("Content-Type", "video/mp4")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            if requested:
                self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
            self.end_headers()
            with path.open("rb") as video:
                video.seek(start)
                remaining = end - start + 1
                while remaining:
                    block = video.read(min(1024 * 1024, remaining))
                    if not block:
                        break
                    self.wfile.write(block)
                    remaining -= len(block)

        def do_GET(self) -> None:
            if self.path.split('?')[0].rstrip('/') == '/ogpo':
                from experiments.ogpo_dashboard import render as render_ogpo

                body = render_ogpo(data_root)
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith('/reports/'):
                from urllib.parse import unquote, urlsplit

                base = (data_root / 'reports').resolve()
                relative = unquote(urlsplit(self.path).path.removeprefix('/reports/'))
                path = (base / relative / 'index.html').resolve() if relative.endswith('/') else (base / relative).resolve()
                types = {'.html': 'text/html; charset=utf-8', '.md': 'text/markdown; charset=utf-8',
                         '.json': 'application/json', '.svg': 'image/svg+xml',
                         '.png': 'image/png', '.zip': 'application/zip'}
                if not path.is_relative_to(base) or not path.is_file() or path.suffix not in types:
                    self.send_error(404)
                    return
                body = path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', types[path.suffix])
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith('/official-dsrl/'):
                from urllib.parse import unquote, urlsplit

                name = unquote(urlsplit(self.path).path.removeprefix('/official-dsrl/')).strip('/')
                root = {root.name: root for root in official_dsrl.roots(data_root)}.get(name)
                if root is None:
                    self.send_error(404)
                    return
                body = official_dsrl.render_page(root)
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith('/diagnostics/'):
                from urllib.parse import unquote, urlsplit
                from experiments.libero.diagnostics import render as render_diagnostics

                anchor = unquote(urlsplit(self.path).path.removeprefix('/diagnostics/'))
                roots = {'task-' + root.name: root for root in task_roots(data_root)}
                roots['sac-task28-general'] = data_root / SAC_TASK28_ROOT
                roots.update({'official-' + root.name: root for root in official_dsrl.roots(data_root)})
                root = roots.get(anchor)
                if root is None or not root.is_dir():
                    self.send_error(404)
                    return
                try:
                    renderer = official_dsrl.render_diagnostics if anchor.startswith('official-') else render_diagnostics
                    body = renderer(root).encode('utf-8')
                except (OSError, ValueError, KeyError) as error:
                    self.send_error(500, str(error))
                    return
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path.startswith('/video-audits/'):
                from urllib.parse import unquote, urlsplit

                base = (data_root / 'video_audits').resolve()
                relative = unquote(urlsplit(self.path).path.removeprefix('/video-audits/'))
                path = (base / relative / 'index.html').resolve() if relative.endswith('/') else (base / relative).resolve()
                if not path.is_relative_to(base) or not path.is_file() or path.suffix not in ('.html', '.json', '.mp4'):
                    self.send_error(404)
                    return
                if path.suffix == '.mp4':
                    self.serve_video(path)
                    return
                body = path.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'text/html; charset=utf-8' if path.suffix == '.html' else 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Cache-Control', 'no-store')
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path in ("/pi05-libero90-video-audit", "/pi05-libero90-video-audit/"):
                body = render_pi05_video_audit(data_root)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            match = re.fullmatch(
                r"/pi05-libero90-video-audit/videos/(02|28)/(demos|policy)/(demo|episode)_(\d+)\.mp4",
                self.path)
            if match:
                task_id, kind, prefix, ident = match.groups()
                if ((kind, prefix) not in (("demos", "demo"), ("policy", "episode")) or
                        int(ident) not in (range(4) if kind == "demos" else range(25, 50))):
                    self.send_error(404)
                    return
                self.serve_video(data_root / PI05_VIDEO_ROOT.name / f"task{task_id}" /
                                 kind / "videos" / f"{prefix}_{ident}.mp4")
                return
            if self.path in ("/pi0-libero10-video-audit", "/pi0-libero10-video-audit/"):
                body = render_libero10_audit(data_root)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            match = re.fullmatch(
                r"/pi0-libero10-video-audit/videos/(06|08)/(demos|policy|sparse|robometer|policy600|sparse600|robometer600)/(demo|state|episode)_(\d+)\.mp4",
                self.path)
            if match:
                task_id, kind, prefix, ident = match.groups()
                if ((kind, prefix) not in (("demos", "demo"), ("policy", "state"),
                                           ("sparse", "episode"), ("robometer", "episode"),
                                           ("policy600", "episode"), ("sparse600", "episode"),
                                           ("robometer600", "episode")) or
                        int(ident) not in (range(4) if kind == "demos" else range(50))):
                    self.send_error(404)
                    return
                if kind.endswith("600"):
                    folder = (data_root / "pi0_libero10_video_audit_600_20261018" /
                              f"task{task_id}" / kind.removesuffix("600"))
                else:
                    root = data_root / "pi0_libero10_video_audit_20261015" / f"task{task_id}"
                    folder = root / "final_policies" / f"{kind}_seed100" if kind in ("sparse", "robometer") else root / kind
                self.serve_video(folder / "videos" / f"{prefix}_{ident}.mp4")
                return
            if self.path in ("/pi0-libero-families-eval", "/pi0-libero-families-eval/"):
                body = render_pi0_families(data_root)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path in ("/pi05-libero-eval", "/pi05-libero-eval/"):
                body = render_pi05_libero(data_root)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            if self.path in ("/pi0-libero90-eval", "/pi0-libero90-eval/"):
                body = render_pi0_eval(data_root)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            match = re.fullmatch(r"/task(38|55)-pi0-baseline/?", self.path)
            if match:
                task_id = int(match.group(1))
                body = render_task38_video(data_root / f"pi0_task{task_id}_video_audit_20261014",
                                           task_id=task_id)
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.end_headers()
                self.wfile.write(body)
                return
            match = re.fullmatch(r"/task(38|55)-pi0-baseline/videos/episode_(\d+)\.mp4", self.path)
            if match:
                task_id, state_id = int(match.group(1)), int(match.group(2))
                if state_id not in range(25, 35):
                    self.send_error(404)
                    return
                self.serve_video(data_root / f"pi0_task{task_id}_video_audit_20261014" /
                                 "videos" / f"episode_{state_id}.mp4")
                return
            match = re.fullmatch(r"/task(38|55)-pi0-baseline/videos/demo_([01])\.mp4", self.path)
            if match:
                task_id = int(match.group(1))
                self.serve_video(data_root / f"pi0_task{task_id}_video_audit_20261014" /
                                 "dataset_demos" / "videos" / f"demo_{match.group(2)}.mp4")
                return
            match = re.fullmatch(r"/task(38|55)-pi0-baseline/videos/(sparse|robometer)/episode_(\d+)\.mp4",
                                 self.path)
            if match:
                task_id, arm, state_id = int(match.group(1)), match.group(2), int(match.group(3))
                if state_id not in range(25, 35):
                    self.send_error(404)
                    return
                self.serve_video(data_root / f"pi0_task{task_id}_video_audit_20261014" /
                                 "final_policies" / f"{arm}_seed100" / "videos" / f"episode_{state_id}.mp4")
                return
            if self.path not in ("/", "/index.html"):
                self.send_error(404)
                return
            body = render(data_root)
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

    ThreadingHTTPServer((host, port), Handler).serve_forever()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/rohan/Reward-Modeling"))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8768)
    args = parser.parse_args()
    serve(args.data_root.resolve(), args.host, args.port)


if __name__ == "__main__":
    main()
