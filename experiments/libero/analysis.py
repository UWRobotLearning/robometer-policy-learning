"""Reusable local curves and paired-seed statistics for SAC and DSRL."""

from __future__ import annotations

import csv
import html
import json
import math
import statistics
from itertools import product
from pathlib import Path
from statistics import mean

from experiments.reward_audit.config import write_json

LABELS = {"sparse": "Sparse simulator reward", "robometer": "Robometer reward"}
COLORS = {"sparse": "#087f8c", "robometer": "#d36a35"}


def read_jsonl(path):
    """Ignore a trailing partial record while a run is writing its logs."""
    if not path.exists():
        return
    with path.open() as stream:
        for line in stream:
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def training_reward_windows(episodes, measured_steps, window=5000):
    """On-policy returns from episodes ending in each transition window.

    Evaluation episodes and unfinished training episodes never contribute.
    Reward density counts terminal successes per collected transition; for
    native LIBERO 0/1 rewards, there is one reward on successful termination.
    """
    windows = {}
    for end in range(window, int(measured_steps) + window, window):
        end = min(end, int(measured_steps))
        start = (end - 1) // window * window
        windows[start] = {
            "start": start,
            "env_steps": end,
            "partial": end - start < window,
            "episodes": 0,
            "successes": 0,
            "returns": [],
        }
    for record in episodes:
        if record.get("phase") != "train" or not 0 < record["env_steps"] <= measured_steps:
            continue
        start = (int(record["env_steps"]) - 1) // window * window
        point = windows[start]
        point["episodes"] += 1
        point["successes"] += int(record["success"])
        point["returns"].append(float(record["training_return"]))
    result = []
    for point in windows.values():
        returns = point.pop("returns")
        point["mean_episode_return"] = statistics.mean(returns) if returns else None
        point["success_rewards_per_1000_transitions"] = (
            1000 * point["successes"] / (point["env_steps"] - point["start"])
        )
        result.append(point)
    return result


def summarize(values):
    """The independent unit is a training seed, never an evaluation episode."""
    values = [float(v) for v in values]
    if not values:
        return {"n": 0, "mean": None, "sd": None, "t_ci95": None, "values": []}
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else None
    interval = None
    if sd is not None:
        from scipy.stats import t

        half_width = float(t.ppf(0.975, len(values) - 1)) * sd / len(values) ** 0.5
        interval = [mean - half_width, mean + half_width]
    return {"n": len(values), "mean": mean, "sd": sd, "t_ci95": interval, "values": values}


def running_auc(curve):
    """Trapezoidal success AUC through each checkpoint, normalized by elapsed steps."""
    if not curve:
        return []
    if curve[0]["env_steps"] != 0:
        raise ValueError("Running AUC requires an evaluation at zero training transitions")
    area, result = 0.0, []
    for i, point in enumerate(curve):
        step = point["env_steps"]
        if i:
            previous = curve[i - 1]
            width = step - previous["env_steps"]
            if width <= 0:
                raise ValueError("AUC checkpoints must be strictly increasing")
            area += width * (previous["success_rate"] + point["success_rate"]) / 2
        result.append({"env_steps": step, "auc": area, "normalized_auc": area / step if step else None})
    return result


def paired_sign_flip_p(differences: list[float]) -> float:
    """Two-sided exact paired randomization p-value for a mean difference."""
    observed = abs(mean(differences))
    permutations = (
        abs(mean(sign * difference for sign, difference in zip(signs, differences, strict=False)))
        for signs in product((-1, 1), repeat=len(differences))
    )
    return sum(value >= observed - 1e-12 for value in permutations) / 2 ** len(differences)


def read_curve(path: Path) -> list[dict]:
    points = []
    if not path.exists():
        return points
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                point = {key: float(row[key]) for key in ("env_steps", "successes", "episodes", "success_rate")}
            except (KeyError, ValueError, TypeError):
                continue  # A running writer may have flushed only part of a row.
            if not all(math.isfinite(value) for value in point.values()):
                continue
            if (
                point["episodes"] <= 0
                or not 0 <= point["successes"] <= point["episodes"]
                or abs(point["success_rate"] - point["successes"] / point["episodes"]) > 1e-8
            ):
                raise ValueError(f"Invalid evaluation counts: {path}")
            if points and point["env_steps"] <= points[-1]["env_steps"]:
                raise ValueError(f"Duplicate or out-of-order evaluations: {path}")
            points.append(point)
    return points


def load_runs(directory: Path) -> list[dict]:
    """Read new task manifests, legacy flat SAC, and legacy nested DSRL runs."""
    directory = Path(directory).absolute()
    configs = {}
    manifests = [directory / "manifest.json", *directory.glob("*/manifest.json")]
    for path in manifests:
        if not path.exists():
            continue
        manifest = json.loads(path.read_text())
        root = path.parent
        for planned in manifest.get("planned_runs", []):
            configs[root / planned["directory"]] = planned["config"]
        if manifest.get("seed100_run"):
            for cfg_path in Path(manifest["seed100_run"]).glob("*/config.json"):
                configs[cfg_path.parent] = json.loads(cfg_path.read_text())
        for cfg_path in root.glob("seed*/*/config.json"):
            if cfg_path.parent.name in {
                f"{reward}_seed{seed}" for seed in manifest.get("seeds", []) for reward in ("sparse", "robometer")
            }:
                configs[cfg_path.parent] = json.loads(cfg_path.read_text())
    planned = directory / "planned_suite.json"
    if planned.exists():
        for cfg in json.loads(planned.read_text())["runs"]:
            configs[directory / cfg["name"]] = cfg
    for cfg_path in directory.glob("*/config.json"):
        configs[cfg_path.parent] = json.loads(cfg_path.read_text())
    if (directory / "config.json").exists() and (directory / "evaluation.csv").exists():
        configs[directory] = json.loads((directory / "config.json").read_text())
    runs = []
    for folder, cfg in sorted(configs.items()):
        if cfg.get("reward") not in LABELS:
            continue
        curve = read_curve(folder / "evaluation.csv")
        status_path = folder / "status.json"
        status = json.loads(status_path.read_text()).get("status", "queued") if status_path.exists() else "queued"
        policy = cfg.get("policy_config_name", "pi0_libero" if cfg.get("pi0_checkpoint") else "scratch")
        base_model = cfg.get("base_model", {"pi0_libero": "pi0", "pi05_libero": "pi0.5"}.get(policy, policy))
        runs.append(
            {
                "path": str(folder),
                "config": cfg,
                "base_model": base_model,
                "status": status,
                "curve": curve,
                "complete": status == "complete" and bool(curve) and curve[-1]["env_steps"] == cfg["steps"],
            }
        )
    if not runs:
        raise ValueError(f"No training runs or planned manifests found in {directory}")
    return runs


def matched_settings(runs: list[dict]) -> None:
    """Never silently pool different budgets, policies, rewards, or reset protocols."""
    keys = (
        "suite",
        "task",
        "num_envs",
        "steps",
        "horizon",
        "settle_steps",
        "train_init_states",
        "eval_episodes",
        "eval_every",
        "initial_actor",
        "sac",
        "actor_hidden",
        "critic_hidden",
        "dino_checkpoint",
        "updates_per_transition",
        "sparse_offset",
        "buffer_capacity",
        "reward_checkpoint",
        "reward_camera",
        "max_frames",
        "pi0_checkpoint",
        "policy_config_name",
        "dsrl",
    )
    reference = runs[0]["config"]
    differences = [
        f"{run['path']}: {key}" for run in runs for key in keys if run["config"].get(key) != reference.get(key)
    ]
    if differences:
        raise ValueError(f"Unmatched experiment settings: {differences}")
    for kind in LABELS:
        formulas = {run["config"].get("reward_formula") for run in runs if run["config"]["reward"] == kind}
        if len(formulas) > 1:
            raise ValueError(f"Different reward formulas within the {kind} arm")


def checkpoints(run: dict) -> dict[int, dict]:
    """Use scheduled transition bins; DSRL can cross a boundary inside a chunk."""
    cfg = run["config"]
    points = {}
    for row in run["curve"]:
        step = int(row["env_steps"])
        scheduled = cfg["steps"] if step == cfg["steps"] else step // cfg["eval_every"] * cfg["eval_every"]
        if scheduled in points:
            raise ValueError(f"Two evaluations map to checkpoint {scheduled}: {run['path']}")
        points[scheduled] = row
    return points


def comparison(values: list[float]) -> dict:
    return {**summarize(values), "p": paired_sign_flip_p(values) if values else None}


def baseline(runs: list[dict]) -> dict | None:
    if runs[0]["base_model"] == "scratch":
        return None
    by_seed = {}
    for run in runs:
        if run["curve"] and run["curve"][0]["env_steps"] == 0:
            by_seed.setdefault(run["config"]["seed"], []).append(run)
    if not by_seed:
        return None
    seed_rates = {seed: mean(run["curve"][0]["success_rate"] for run in selected) for seed, selected in by_seed.items()}
    stats = summarize(seed_rates.values())
    episodes = sum(run["curve"][0]["episodes"] for selected in by_seed.values() for run in selected)
    stats.update(
        seed_rates=seed_rates,
        episodes=episodes,
        method="95% Student t interval across seed means; reward arms averaged within each seed",
    )
    if len(seed_rates) == 1:
        # One seed cannot estimate seed variation. Use one rollout set rather
        # than treating two arms on the same reset states as independent samples.
        from experiments.reward_audit.metrics import wilson

        selected = next(iter(by_seed.values()))[0]
        row = selected["curve"][0]
        states = {}
        for record in read_jsonl(Path(selected["path"]) / "episodes.jsonl"):
            if record.get("phase") == "eval" and record.get("env_steps") == 0:
                states.setdefault(record["init_state_id"], []).append(float(record["success"]))
        if states and sum(len(values) for values in states.values()) == row["episodes"]:
            stats = {
                **summarize([mean(values) for values in states.values()]),
                "seed_rates": seed_rates,
                "episodes": row["episodes"],
                "method": "95% Student t interval across reset-state means, one sampling seed",
            }
        else:
            stats.update(
                mean=row["success_rate"],
                episodes=row["episodes"],
                t_ci95=list(wilson(row["successes"], row["episodes"])),
                method="95% Wilson interval over one rollout set; seed uncertainty unavailable",
            )
    return stats


def summarize_group(runs: list[dict]) -> dict:
    matched_settings(runs)
    grids = {}
    for run in runs:
        key = (run["config"]["seed"], run["config"]["reward"])
        if key in grids:
            raise ValueError(f"Duplicate seed/reward: {key}")
        grids[key] = checkpoints(run)
    curves, auc_curves, finals, auc_finals = {}, {}, {}, {}
    for kind in LABELS:
        selected = [run for run in runs if run["config"]["reward"] == kind]
        points = sorted({step for run in selected for step in grids[(run["config"]["seed"], kind)]})
        curves[kind] = []
        auc_grids = []
        for run in selected:
            auc = running_auc(run["curve"])
            auc_run = dict(run, curve=auc)
            auc_grids.append(checkpoints(auc_run))
        auc_curves[kind] = []
        for step in points:
            rows = [
                grids[(run["config"]["seed"], kind)][step]
                for run in selected
                if step in grids[(run["config"]["seed"], kind)]
            ]
            curves[kind].append(
                {
                    "env_steps": step,
                    "mean_actual_steps": mean(row["env_steps"] for row in rows),
                    **summarize(row["success_rate"] for row in rows),
                }
            )
            values = [
                grid[step]["normalized_auc"]
                for grid in auc_grids
                if step in grid and grid[step]["normalized_auc"] is not None
            ]
            if values:
                auc_curves[kind].append({"env_steps": step, **summarize(values)})
        completed = [run for run in selected if run["complete"]]
        finals[kind] = summarize(run["curve"][-1]["success_rate"] for run in completed)
        auc_finals[kind] = summarize(running_auc(run["curve"])[-1]["normalized_auc"] for run in completed)
    pretrained = baseline(runs)
    comparisons = []
    for step in sorted({key for grid in grids.values() for key in grid}):
        differences = [
            grids[(seed, "sparse")][step]["success_rate"] - grids[(seed, "robometer")][step]["success_rate"]
            for seed in {key[0] for key in grids}
            if step in grids.get((seed, "sparse"), {}) and step in grids.get((seed, "robometer"), {})
        ]
        entry = {"env_steps": step, "sparse_minus_robometer": comparison(differences)}
        if pretrained:
            for kind in LABELS:
                entry[f"{kind}_minus_pretrained"] = comparison(
                    [
                        grid[step]["success_rate"] - pretrained["seed_rates"][seed]
                        for (seed, reward), grid in grids.items()
                        if reward == kind and step in grid and seed in pretrained["seed_rates"]
                    ]
                )
        comparisons.append(entry)
    completed = {(run["config"]["seed"], run["config"]["reward"]): run for run in runs if run["complete"]}
    paired = [
        (completed[(seed, "sparse")], completed[(seed, "robometer")])
        for seed in sorted({key[0] for key in completed})
        if (seed, "sparse") in completed and (seed, "robometer") in completed
    ]
    return {
        "suite": runs[0]["config"]["suite"],
        "task": runs[0]["config"]["task"],
        "base_model": runs[0]["base_model"],
        "curves": curves,
        "normalized_auc_curves": auc_curves,
        "final": finals,
        "normalized_auc": auc_finals,
        "baseline": pretrained,
        "checkpoint_comparisons": comparisons,
        "paired_final_sparse_minus_robometer": comparison(
            [a["curve"][-1]["success_rate"] - b["curve"][-1]["success_rate"] for a, b in paired]
        ),
        "paired_auc_sparse_minus_robometer": comparison(
            [
                running_auc(a["curve"])[-1]["normalized_auc"] - running_auc(b["curve"])[-1]["normalized_auc"]
                for a, b in paired
            ]
        ),
        "completed_paired_seeds": [a["config"]["seed"] for a, _ in paired],
        "uncertainty": (
            "95% Student t intervals over training seeds, conditional on this held-out reset set; missing runs excluded"
        ),
        "tests": "Two-sided exact paired seed sign-flip tests; checkpoint p-values are pointwise and uncorrected",
        "alignment": "Scheduled transition bins; mean_actual_steps records chunk-boundary overshoot",
    }


def draw_group(out: Path, name: str, runs: list[dict], stats: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
    for run in runs:
        kind = run["config"]["reward"]
        axes[0].plot(
            [row["env_steps"] for row in run["curve"]],
            [row["success_rate"] for row in run["curve"]],
            color=COLORS[kind],
            alpha=0.2,
            lw=1,
        )
        episodes = list(read_jsonl(Path(run["path"]) / "episodes.jsonl"))
        measured = max(
            [row["env_steps"] for row in run["curve"]]
            + [row["env_steps"] for row in episodes if row.get("phase") == "train"]
            + [0]
        )
        windows = training_reward_windows(episodes, measured, run["config"]["eval_every"])
        axes[2].plot(
            [row["env_steps"] for row in windows],
            [row["mean_episode_return"] for row in windows],
            color=COLORS[kind],
            alpha=0.4,
            label=f"{kind} seed {run['config']['seed']}",
        )
    for axis, key in zip(axes[:2], ("curves", "normalized_auc_curves"), strict=False):
        for kind, points in stats[key].items():
            if not points:
                continue
            xs = [point["env_steps"] for point in points]
            axis.plot(xs, [point["mean"] for point in points], color=COLORS[kind], lw=2.5, label=LABELS[kind])
            lo = [max(0, point["t_ci95"][0]) if point["t_ci95"] is not None else float("nan") for point in points]
            hi = [min(1, point["t_ci95"][1]) if point["t_ci95"] is not None else float("nan") for point in points]
            axis.fill_between(xs, lo, hi, color=COLORS[kind], alpha=0.15)
    frozen = stats["baseline"]
    if frozen:
        axes[0].axhline(frozen["mean"], color="black", linestyle=":", label=f"Frozen checkpoint ({frozen['mean']:.1%})")
        if frozen["t_ci95"] is not None:
            axes[0].axhspan(max(0, frozen["t_ci95"][0]), min(1, frozen["t_ci95"][1]), color="black", alpha=0.08)
    for key, color, height in (
        ("sparse_minus_pretrained", "green", 1.035),
        ("robometer_minus_pretrained", "#d36a35", 1.075),
        ("sparse_minus_robometer", "black", 1.115),
    ):
        points = [
            entry["env_steps"]
            for entry in stats["checkpoint_comparisons"]
            if entry.get(key, {}).get("p") is not None and entry[key]["p"] < 0.05
        ]
        axes[0].scatter(points, [height] * len(points), marker="*", color=color, s=65)
    axes[0].set(ylabel="Simulator success rate", ylim=(-0.02, 1.15), title="Mean and 95% confidence interval")
    axes[1].set(ylabel="Normalized running success AUC", ylim=(-0.02, 1.02))
    env_count = runs[0]["config"].get("num_envs", 1)
    axis_label = "Physical training steps per environment (average)" if env_count > 1 else "Physical training steps"
    axes[2].set(ylabel="Mean training episode return", xlabel=axis_label)
    from matplotlib.ticker import FuncFormatter

    axes[2].xaxis.set_major_formatter(FuncFormatter(lambda step, _: f"{step / env_count:,.0f}"))
    for axis in axes:
        axis.grid(alpha=0.2)
        if axis.get_legend_handles_labels()[0]:
            axis.legend(fontsize=8)
        axis.set_xlim(0, runs[0]["config"]["steps"])
    cfg = runs[0]["config"]
    figure.suptitle(
        f"{stats['suite']}: {stats['task'].replace('_', ' ')}\nBase policy: {stats['base_model']} · "
        f"{cfg.get('num_envs', 1)} training environments · "
        f"{cfg.get('dsrl', {}).get('action_exec_len', 1)} actions/chunk"
    )
    figure.tight_layout()
    for extension in ("png", "pdf"):
        figure.savefig(out / f"{name}.{extension}", dpi=170)
    plt.close(figure)


def generate(directory: Path, out: Path | None = None) -> Path:
    directory = Path(directory).absolute()
    out = Path(out).absolute() if out else directory / "analysis"
    runs = load_runs(directory)
    groups = {}
    for run in runs:
        cfg = run["config"]
        key = (cfg["suite"], cfg["task"], run["base_model"], cfg.get("comparison", "comparison"))
        groups.setdefault(key, []).append(run)
    statistics_by_group = {}
    # Validate all groups before creating files, so incompatible data stays obvious.
    for key, selected in groups.items():
        statistics_by_group[key] = summarize_group(selected)
    out.mkdir(parents=True, exist_ok=True)
    sections = []
    payload = []
    for index, (key, selected) in enumerate(groups.items()):
        stats = statistics_by_group[key]
        name = f"experiment_{index:02d}"
        draw_group(out, name, selected, stats)
        payload.append(stats)
        final = stats["paired_final_sparse_minus_robometer"]
        p = final["p"]
        conclusion = (
            ("Statistically significant" if p < 0.05 else "Not statistically significant")
            if p is not None
            else "Insufficient completed pairs"
        )
        detail = f"; exact p={p:.4f}, n={final['n']} paired seeds" if p is not None else ""
        interval = final["t_ci95"]
        effect = (
            (
                f"Sparse - Robometer: {final['mean'] * 100:+.1f} percentage points"
                + (f" (95% CI {interval[0] * 100:+.1f} to {interval[1] * 100:+.1f})" if interval else "")
            )
            if final["mean"] is not None
            else ""
        )
        frozen = stats["baseline"]
        baseline_note = ""
        if frozen:
            ci = frozen["t_ci95"]
            baseline_note = (
                f"<p>Frozen checkpoint: {frozen['mean']:.1%}"
                + (f" (95% CI {max(0, ci[0]):.1%} to {min(1, ci[1]):.1%})" if ci else "")
                + f". {html.escape(frozen['method'])}.</p>"
            )
        sections.append(
            f"<section><h2>{html.escape(stats['suite'] + ': ' + stats['task'].replace('_', ' '))}</h2>"
            f"<p>Base policy: {html.escape(stats['base_model'])}. {conclusion}{detail}.</p>"
            f'<p>{effect}</p>{baseline_note}<img src="{name}.png" alt="Success, AUC, and training reward curves">'
            f'<p><a href="{name}.pdf">Download PDF</a></p></section>'
        )
    write_json(out / "statistics.json", {"experiments": payload})
    with (out / "curves.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(["suite", "task", "base_model", "reward", "seed", "status", "env_steps", "success_rate"])
        for run in runs:
            cfg = run["config"]
            for row in run["curve"]:
                writer.writerow(
                    [
                        cfg["suite"],
                        cfg["task"],
                        run["base_model"],
                        cfg["reward"],
                        cfg["seed"],
                        run["status"],
                        row["env_steps"],
                        row["success_rate"],
                    ]
                )
    page = (
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        "<title>LIBERO experiment analysis</title><style>"
        "body{font:16px system-ui;max-width:1100px;margin:2rem auto;padding:1rem}"
        "img{width:100%}section{margin-bottom:3rem}</style></head><body><h1>LIBERO experiment analysis</h1>"
        "<p>Bold curves: means; shaded bands: 95% Student t intervals across available training seeds. "
        "Dotted black: frozen pretrained checkpoint. Final tests include only completed paired seeds. "
        "Stars: green sparse vs pretrained, orange Robometer vs pretrained, black sparse vs Robometer "
        '(pointwise p &lt; 0.05; uncorrected).</p><p><a href="statistics.json">Statistics JSON</a> · '
        '<a href="curves.csv">Curves CSV</a></p>' + "".join(sections) + "</body></html>"
    )
    (out / "index.html").write_text(page)
    return out / "index.html"
