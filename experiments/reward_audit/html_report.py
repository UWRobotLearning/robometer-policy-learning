"""Local SAC comparison with matched-seed statistics and an offline HTML page."""

from __future__ import annotations

import argparse
import csv
import html
import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from experiments.libero.analysis import read_jsonl, running_auc, summarize, training_reward_windows

from .config import write_json

# Settings that must stay fixed across rewards and training seeds.
MATCH_FIELDS = (
    "suite",
    "task",
    "task_id",
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
    "reward_formula",
    "max_frames",
    "audit_sparse_rewards",
)
LABELS = {"sparse": "Simulator sparse reward", "robometer": "Robometer reward"}
COLORS = {"sparse": "#087f8c", "robometer": "#d36a35"}


def read_runs(directory):
    planned = directory / "planned_suite.json"
    configs = {}
    if planned.exists():
        configs.update({cfg["name"]: cfg for cfg in json.loads(planned.read_text())["runs"]})
    for path in directory.glob("*/config.json"):
        configs[path.parent.name] = json.loads(path.read_text())
    runs = []
    for name, cfg in configs.items():
        folder = directory / name
        curve = []
        path = folder / "evaluation.csv"
        if path.exists():
            with path.open() as stream:
                for row in csv.DictReader(stream):
                    # A concurrent writer may not yet have flushed its final row.
                    try:
                        point = {key: float(value) for key, value in row.items()}
                    except (ValueError, TypeError):
                        continue
                    if all(np.isfinite(v) for v in point.values()):
                        curve.append(point)
        path = folder / "status.json"
        status = json.loads(path.read_text()) if path.exists() else {"status": "pending"}
        path = folder / "decoder_checks.json"
        decoder = json.loads(path.read_text()) if path.exists() else None
        path = folder / "reward_provenance.json"
        provenance = json.loads(path.read_text()) if path.exists() else {}
        measured_steps = curve[-1]["env_steps"] if curve else 0
        path = folder / "metrics.jsonl"
        for record in read_jsonl(path):
            measured_steps = max(measured_steps, record.get("env_steps") or 0)
        training = [r for r in read_jsonl(folder / "episodes.jsonl") if r.get("phase") == "train"]
        measured_steps = max([measured_steps] + [r["env_steps"] for r in training])
        complete = status["status"] == "complete" and bool(curve) and curve[-1]["env_steps"] == cfg["steps"]
        runs.append(
            {
                "name": name,
                "label": LABELS[cfg["reward"]],
                "color": COLORS[cfg["reward"]],
                "config": cfg,
                "status": status,
                "complete": complete,
                "curve": curve,
                "final": curve[-1] if curve else None,
                "peak": max(curve, key=lambda p: p["success_rate"]) if curve else None,
                "decoder": decoder,
                "decoder_mode": provenance.get("progress_decoder", "not_recorded"),
                "training_steps": measured_steps,
                "training_reward_windows": training_reward_windows(training, measured_steps, cfg["eval_every"])
                if cfg["reward"] == "sparse" else [],
            }
        )
    runs.sort(key=lambda r: (r["config"]["seed"], r["config"]["reward"] != "sparse"))
    return runs


def aggregate(runs):
    reference = runs[0]["config"]
    mismatches = [f"{r['name']}: {key}" for r in runs for key in MATCH_FIELDS if r["config"][key] != reference[key]]
    if mismatches:
        raise ValueError(f"Unmatched experiment settings: {mismatches}")
    expected_steps = list(range(0, reference["steps"] + 1, reference["eval_every"]))
    if expected_steps[-1] != reference["steps"]:
        expected_steps.append(reference["steps"])
    for run in runs:
        if run["complete"] and [p["env_steps"] for p in run["curve"]] != expected_steps:
            raise ValueError(f"Missing or duplicated evaluation checkpoints: {run['name']}")
    by_seed = {}
    for run in runs:
        cfg = run["config"]
        pair = by_seed.setdefault(cfg["seed"], {})
        if cfg["reward"] in pair:
            raise ValueError(f"Duplicate {cfg['reward']} run for seed {cfg['seed']}")
        pair[cfg["reward"]] = run
    eligible = {
        seed: pair
        for seed, pair in by_seed.items()
        if set(pair) == {"sparse", "robometer"} and all(run["complete"] for run in pair.values())
    }
    decoder_versions = {
        run["name"]: run.get("decoder_mode", "not_recorded") for run in runs if run["config"]["reward"] == "robometer"
    }
    for pair in eligible.values():
        run = pair["robometer"]
        if run.get("decoder_mode", "").startswith("upstream_unchanged"):
            counts = run.get("decoder") or {}
            if any(counts.get(key, 0) for key in ("branch_hits", "mismatches", "invalid_outputs")):
                raise ValueError(f"Cannot pool an anomalous pre-fix run: {run['name']}")
    final, auc, curves = {}, {}, {}
    for kind in LABELS:
        selected = [pair[kind] for pair in eligible.values()]
        final[kind] = summarize([run["final"]["success_rate"] for run in selected])
        auc[kind] = summarize(
            [
                float(
                    np.trapz([p["success_rate"] for p in run["curve"]], [p["env_steps"] for p in run["curve"]])
                    / run["config"]["steps"]
                )
                for run in selected
            ]
        )
        points = [{p["env_steps"]: p for p in run["curve"]} for run in selected]
        common = sorted(set.intersection(*(set(p) for p in points))) if points else []
        curves[kind] = [
            {
                "env_steps": step,
                "env_iterations": step / reference["num_envs"],
                **summarize([p[step]["success_rate"] for p in points]),
            }
            for step in common
        ]
    paired = []
    for seed, pair in eligible.items():
        left, right = pair["robometer"], pair["sparse"]
        lx = [p["env_steps"] for p in left["curve"]]
        rx = [p["env_steps"] for p in right["curve"]]
        if lx != rx:
            raise ValueError(f"Evaluation grids differ for seed {seed}")
        difference = [a["success_rate"] - b["success_rate"] for a, b in zip(left["curve"], right["curve"], strict=True)]
        paired.append(
            {
                "seed": seed,
                "final_difference": difference[-1],
                "normalized_auc_difference": float(np.trapz(difference, lx) / reference["steps"]),
                "initial_difference": difference[0],
            }
        )
    return {
        "expected_seeds": sorted(by_seed),
        "completed_paired_seeds": sorted(eligible),
        "all_complete": len(eligible) == len(by_seed),
        "final": final,
        "normalized_auc": auc,
        "paired_final_difference": summarize([p["final_difference"] for p in paired]),
        "paired_auc_difference": summarize([p["normalized_auc_difference"] for p in paired]),
        "per_seed_differences": paired,
        "decoder_versions": decoder_versions,
        "curves": curves,
        "uncertainty_unit": "Training seed; sample SD and two-sided Student t 95% interval for the mean",
        "difference_direction": "Robometer minus simulator",
        "limitations": [
            f"Only {len(by_seed)} training seeds are planned; intervals with few seeds are imprecise.",
            "The same 25 held-out reset states are evaluated per seed; they are not 75 independent training runs.",
            "Aggregates use only seeds with both arms complete at the full budget; "
            "pending runs are never counted as zero.",
            "Curve bands show sample SD across training seeds, not a confidence interval.",
            "Student t intervals are approximate and unbounded; zero SD in a tiny sample does not prove certainty.",
            "LIBERO-Goal is in the declared reward-model training mix; this is not an OOD demonstration.",
            "Policies train from scratch with frozen Robometer; online reward-model fine-tuning is not tested.",
            "Seed 100 used the original decoder with no recorded bypasses or mismatches. "
            "Corrected runs always softmax; decoder versions are recorded separately.",
        ],
    }


def plot_seed_curves(directory, runs, stats):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.8))
    count = runs[0]["config"]["num_envs"]
    for run in runs:
        if run["curve"]:
            ax.plot(
                [p["env_steps"] / count for p in run["curve"]],
                [p["success_rate"] for p in run["curve"]],
                color=run["color"],
                alpha=0.25,
                lw=1,
            )
    for kind, points in stats["curves"].items():
        if not points:
            continue
        x = np.array([p["env_iterations"] for p in points])
        mean = np.array([p["mean"] for p in points])
        sd = np.array([p["sd"] or 0 for p in points])
        ax.plot(x, mean, color=COLORS[kind], lw=2.3, label=f"{LABELS[kind]} (n={points[0]['n']} paired seeds)")
        if points[0]["n"] > 1:
            ax.fill_between(x, np.maximum(0, mean - sd), np.minimum(1, mean + sd), color=COLORS[kind], alpha=0.12)
    ax.set(
        xlabel=f"Environment iterations ({count} transitions each)",
        ylabel="Simulator success rate",
        ylim=(-0.02, 1.02),
        xlim=(0, runs[0]["config"]["steps"] / count),
        title=f"{runs[0]['config']['task'].replace('_', ' ')} — mean ± sample SD across seeds",
    )
    secondary = ax.secondary_xaxis("top", functions=(lambda x: x * count, lambda x: x / count))
    secondary.set_xlabel("Aggregate training transitions")
    ax.grid(alpha=0.2)
    if any(stats["curves"].values()):
        ax.legend(fontsize=9)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"success_rate_vs_iterations.{ext}", dpi=180)
    plt.close(fig)


def plot_sparse_curves(directory, runs):
    """Show all sparse runs without waiting for their paired Robometer runs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sparse = [r for r in runs if r["config"]["reward"] == "sparse"]
    completed = [r for r in sparse if r["complete"]]
    grids = [{p["env_steps"]: p for p in r["curve"]} for r in completed]
    common = sorted(set.intersection(*(set(g) for g in grids))) if grids else []
    mean_curve = [{"env_steps": step, **summarize([g[step]["success_rate"] for g in grids])} for step in common]
    stats = {"completed_seeds": [r["config"]["seed"] for r in completed],
             "final": summarize([r["final"]["success_rate"] for r in completed]), "curve": mean_curve}
    write_json(directory / "sparse_reward_statistics.json", stats)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10.5), sharex=True)
    palette = ("#087f8c", "#7759ad", "#bb7b13")
    for i, run in enumerate(sparse):
        color = palette[i % len(palette)]
        label = f"Seed {run['config']['seed']} ({'complete' if run['complete'] else run['status']['status']})"
        axes[0].plot([p["env_steps"] for p in run["curve"]], [p["success_rate"] for p in run["curve"]],
                     color=color, label=label, marker="o", markersize=3, linewidth=1.5)
        points = run["training_reward_windows"]
        for ax, key in zip(axes[1:], ("mean_episode_return", "success_rewards_per_1000_transitions")):
            ax.plot([p["env_steps"] for p in points], [np.nan if p[key] is None else p[key] for p in points],
                    color=color, label=label, marker="o", markersize=3, linewidth=1.5)
            for p in points:
                if p["partial"] and p[key] is not None:
                    ax.scatter(p["env_steps"], p[key], s=65, facecolors="white", edgecolors=color, zorder=5)
    if mean_curve:
        x = np.array([p["env_steps"] for p in mean_curve])
        mean = np.array([p["mean"] for p in mean_curve])
        sd = np.array([p["sd"] or 0 for p in mean_curve])
        axes[0].plot(x, mean, color="#172d35", lw=2.4, label=f"Mean of {len(completed)} completed sparse seeds")
        if len(completed) > 1:
            axes[0].fill_between(x, np.maximum(0, mean - sd), np.minimum(1, mean + sd), color="#172d35", alpha=.1)
    axes[0].set(title="Held-out simulator success (25 evaluation episodes per checkpoint)", ylabel="Success rate", ylim=(-.03, 1.03))
    axes[1].set(title="Collected training reward: mean return of episodes ending in each window", ylabel="Mean episode return")
    if all(r["config"]["sparse_offset"] == 0 for r in sparse):
        axes[1].set_ylim(-.03, 1.03)
    axes[2].set(title="Frequency of successful terminal rewards during collection", ylabel="Success rewards / 1,000 transitions",
                xlabel="Aggregate training transitions")
    for ax in axes:
        ax.set_xlim(0, runs[0]["config"]["steps"])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="upper left")
        ax.ticklabel_format(axis="x", style="plain")
    count = runs[0]["config"]["num_envs"]
    top = axes[0].secondary_xaxis("top", functions=(lambda x: x / count, lambda x: x * count))
    top.set_xlabel(f"Environment iterations ({count} transitions each)")
    fig.suptitle(f"{runs[0]['config']['task'].replace('_', ' ').capitalize()} — SAC with simulator sparse reward", fontsize=15)
    fig.tight_layout(rect=(0, .035, 1, .98))
    fig.text(.07, .015, f"Training: {runs[0]['config']['eval_every']:,}-transition windows; open circles mark partial windows. "
             "Unfinished episodes are excluded. No smoothing.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"sparse_reward_curves.{ext}", dpi=150)
    plt.close(fig)
    return stats


def plot_sparse_auc(directory, runs):
    """Running area and average success, with uncertainty across training seeds."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [r for r in runs if r["config"]["reward"] == "sparse"]
    per_seed = [{"seed": r["config"]["seed"], "complete": r["complete"],
                 "status": r["status"]["status"], "curve": running_auc(r["curve"])} for r in selected]
    grids = [{p["env_steps"]: p for p in run["curve"]} for run in per_seed if run["complete"]]
    common = sorted(set.intersection(*(set(g) for g in grids))) if grids else []
    mean_curve = [{"env_steps": step, "auc": summarize([g[step]["auc"] for g in grids]),
                   "normalized_auc": summarize([g[step]["normalized_auc"] for g in grids]) if step else summarize([])}
                  for step in common]
    stats = {"integration": "Trapezoidal integration of held-out success versus training transitions",
             "normalization": "Area from zero through the current checkpoint divided by that checkpoint's transitions",
             "zero_checkpoint": "Raw area is zero; normalized area is undefined and omitted",
             "completed_seeds": [r["seed"] for r in per_seed if r["complete"]],
             "per_seed": per_seed, "mean_curve": mean_curve}
    write_json(directory / "sparse_auc_statistics.json", stats)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.3), sharex=True)
    palette = ("#087f8c", "#7759ad", "#bb7b13")
    for i, run in enumerate(per_seed):
        for ax, key, scale in zip(axes, ("normalized_auc", "auc"), (100, 1)):
            points = [p for p in run["curve"] if p[key] is not None]
            ax.plot([p["env_steps"] for p in points], [p[key] * scale for p in points],
                    color=palette[i % len(palette)], marker="o", markersize=3, lw=1.5,
                    label=f"Seed {run['seed']} ({run['status']})")
    for ax, key, scale in zip(axes, ("normalized_auc", "auc"), (100, 1)):
        points = [p for p in mean_curve if p[key]["mean"] is not None]
        if points:
            x = np.array([p["env_steps"] for p in points])
            mean = np.array([p[key]["mean"] * scale for p in points])
            sd = np.array([(p[key]["sd"] or 0) * scale for p in points])
            ax.plot(x, mean, color="#172d35", lw=2.5, label=f"Mean of {len(grids)} completed sparse seeds")
            if len(grids) > 1:
                ax.fill_between(x, np.maximum(0, mean - sd), np.minimum(100, mean + sd) if key == "normalized_auc" else mean + sd,
                                color="#172d35", alpha=.12)
        ax.set_xlim(0, runs[0]["config"]["steps"])
        ax.grid(alpha=.2)
        ax.legend(fontsize=8, loc="upper left")
        ax.ticklabel_format(axis="x", style="plain")
    upper_values = [p["normalized_auc"] * 100 for r in per_seed for p in r["curve"] if p["normalized_auc"] is not None]
    upper_values += [(p["normalized_auc"]["mean"] + (p["normalized_auc"]["sd"] or 0)) * 100
                     for p in mean_curve if p["normalized_auc"]["mean"] is not None]
    upper = min(100, max(10, np.ceil(max(upper_values, default=0) / 10) * 10))
    axes[0].set(title="Normalized running AUC: average success over training so far", ylabel="Normalized success AUC (%)", ylim=(-.3, upper))
    axes[1].set(title="Cumulative area under the success curve", ylabel="Area (success fraction × transitions)",
                xlabel="Aggregate training transitions", ylim=(0, None))
    count = runs[0]["config"]["num_envs"]
    top = axes[0].secondary_xaxis("top", functions=(lambda x: x / count, lambda x: x * count))
    top.set_xlabel(f"Environment iterations ({count} transitions each)")
    fig.suptitle("SAC with simulator sparse reward — running success-curve AUC", fontsize=15)
    fig.tight_layout(rect=(0, .035, 1, .98))
    fig.text(.07, .015, "Trapezoidal integration of recorded evaluation checkpoints. Bands: ±1 sample SD across completed training seeds.", fontsize=9)
    for ext in ("png", "pdf"):
        fig.savefig(directory / f"sparse_auc_curves.{ext}", dpi=150)
    plt.close(fig)
    return stats


def generate_html(directory: Path) -> Path:
    directory = Path(directory).resolve()
    runs = read_runs(directory)
    if not runs:
        raise ValueError("No measured runs or planned suite found")
    stats = aggregate(runs)
    fix_path = directory / "decoder_fix.json"
    payload = {
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "runs": runs,
        "stats": stats,
        "decoder_fix": json.loads(fix_path.read_text()) if fix_path.exists() else None,
        "sparse_stats": plot_sparse_curves(directory, runs),
        "sparse_auc": plot_sparse_auc(directory, runs),
    }
    write_json(directory / "seed_statistics.json", stats)
    plot_seed_curves(directory, runs, stats)
    encoded = json.dumps(payload, allow_nan=False).replace("<", "\\u003c")
    template = Path(__file__).with_name("results_template.html").read_text()
    page = template.replace("__REPORT_DATA__", encoded).replace("__GENERATED__", html.escape(payload["generated"]))
    path = directory / "index.html"
    # Coordinators and the HTTP preview can refresh independently.
    with tempfile.NamedTemporaryFile(mode="w", dir=directory, suffix=".html.tmp", delete=False) as stream:
        stream.write(page)
        temp = Path(stream.name)
    temp.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    print(generate_html(args.directory))


if __name__ == "__main__":
    main()
