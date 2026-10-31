"""Compact DSRL diagnostics from local logs; no model loading or GPU work."""

from __future__ import annotations

import html
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .analysis import COLORS, LABELS, load_runs, read_jsonl

# These describe the noise-space policy pi(w|s), not pi0's robot-action outputs.
GROUPS = (
    (
        "Latent-noise policy · Gaussian parameters before tanh",
        (
            ("latent_mean_norm", "Mean-vector norm", "Mean L2 norm of the Gaussian location vector."),
            ("latent_std_norm", "Std-vector norm", "Mean L2 norm of the Gaussian standard-deviation vector."),
            ("latent_mean", "Mean parameter", "Average Gaussian location across states and noise coordinates."),
            ("latent_std", "Std parameter", "Average exp(log_std), before tanh; not the std of robot actions."),
            ("latent_std_max", "Largest std parameter", "Largest Gaussian std within each logged actor batch."),
        ),
    ),
    (
        "Latent exploration and entropy tuning",
        (
            (
                "entropy",
                "Latent-policy entropy",
                "Monte Carlo -log pi(w|s), after the action transform; may be negative.",
            ),
            ("target_entropy", "Target entropy", "Configured target for the latent-noise policy."),
            ("temperature", "Temperature (alpha)", "Weight of the latent-policy entropy term."),
            ("temperature_loss", "Temperature loss", "Implementation's temperature optimization objective."),
            (
                "actor_entropy_term",
                "Actor entropy term",
                "Mean alpha * log pi(w|s); the entropy component of the objective.",
            ),
        ),
    ),
    (
        "Noise-space critic · Q(s, w)",
        (
            ("critic_q_mean", "Replay Q(s, w)", "Critic value of replayed steering noise."),
            ("bootstrap_q_mean", "Next-state bootstrap", "Reduced target-critic value before reward and discount."),
            ("bellman_target_mean", "Bellman target", "Actual reward + discount * nonterminal mask * bootstrap."),
            ("critic_loss", "Critic TD loss", "Mean squared Bellman residual used for critic optimization."),
            (
                "critic_ensemble_std",
                "Critic disagreement",
                "Std across critics for the same state/noise, averaged over batch.",
            ),
        ),
    ),
    (
        "Steering actor objective and data collection",
        (
            ("actor_loss", "Latent actor loss", "The objective minimized by the learned steering actor."),
            ("actor_chosen_q_mean", "Q(s, actor noise)", "Critic value of noise sampled from the steering actor."),
            ("actor_updates", "Actor optimizer steps", "Cumulative completed steering-actor optimizer steps."),
            ("critic_updates", "Critic optimizer steps", "Cumulative completed noise-critic optimizer steps."),
            (
                "collected_chunks",
                "Collected action chunks",
                "Each executed pi0 action chunk creates one replay transition.",
            ),
        ),
    ),
    (
        "Training rollouts",
        (
            ("train_success", "Simulator success", "Mean terminal success within each 1k-step collection bin."),
            ("train_length", "Episode length", "Physical simulator steps in completed training episodes."),
            (
                "train_return",
                "Training reward return",
                "Sum of the rewards assigned to chunks, in each arm's reward units.",
            ),
            (
                "train_native_return",
                "Native simulator return",
                "Sum of simulator rewards, independent of the learned reward.",
            ),
            ("replay_size", "Replay transitions", "Current replay size measured in action chunks, not physical steps."),
        ),
    ),
    (
        "Held-out evaluation and reward alignment",
        (
            (
                "eval_success",
                "Evaluation success",
                "Held-out simulator success rate; same measurement as the main plot.",
            ),
            ("eval_length", "Evaluation length", "Mean physical steps per evaluation episode."),
            (
                "eval_native_return",
                "Evaluation native return",
                "Mean simulator return; no Robometer calls added to evaluation.",
            ),
            (
                "train_return_success",
                "Return · successful episodes",
                "Training reward return restricted to simulator successes.",
            ),
            (
                "train_return_failure",
                "Return · failed episodes",
                "Training reward return restricted to simulator failures.",
            ),
        ),
    ),
)
ALIASES = {
    "critic_q_mean": "q_values_mean",
    # The historical name target_q_mean contains the bootstrap, not a Bellman target.
    "bootstrap_q_mean": "target_q_mean",
    "temperature": "ent_coef",
    "temperature_loss": "ent_coef_loss",
}


def read_diagnostics(root: Path) -> dict:
    """Preserve seed identity; leave unavailable historical quantities missing."""
    series = defaultdict(lambda: defaultdict(list))
    runs = [run for run in load_runs(root) if run["base_model"] != "scratch"]
    keys = {key for _, group in GROUPS for key, _, _ in group}
    env_counts = {
        (run["config"]["reward"], run["config"]["seed"]): int(run["config"].get("num_envs", 1)) for run in runs
    }

    def add(key, identity, step, value):
        if isinstance(value, (int, float)) and math.isfinite(value) and isinstance(step, (int, float)):
            series[key][identity].append((float(step) / env_counts[identity], float(value)))

    for run in runs:
        cfg = run["config"]
        identity = (cfg["reward"], cfg["seed"])
        folder = Path(run["path"])
        for row in read_jsonl(folder / "metrics.jsonl"):
            if row.get("prefix") != "sac":
                continue
            metrics = row.get("metrics", {})
            step = row.get("env_steps")
            for key in keys:
                value = metrics.get(key, metrics.get(ALIASES.get(key)))
                if key == "entropy" and value is None and "actor_log_pis_mean" in metrics:
                    value = -metrics["actor_log_pis_mean"]
                add(key, identity, step, value)
        completed_evals = {row["env_steps"] for row in run["curve"]}
        for row in read_jsonl(folder / "episodes.jsonl"):
            step = row.get("env_steps")
            phase = row.get("phase")
            if phase not in ("train", "eval") or (phase == "eval" and step not in completed_evals):
                continue
            add(f"{phase}_length", identity, step, row.get("length"))
            add(f"{phase}_native_return", identity, step, row.get("native_return"))
            if phase == "train":
                add("train_success", identity, step, row.get("success"))
                add("train_return", identity, step, row.get("training_return"))
                outcome = "success" if row.get("success") else "failure"
                add(f"train_return_{outcome}", identity, step, row.get("training_return"))
        # Only complete evaluation checkpoints contribute success rates.
        for row in run["curve"]:
            add("eval_success", identity, row["env_steps"], row["success_rate"])
    return {"series": dict(series), "runs": runs}


def bucket(points: list[tuple[float, float]]) -> dict[int, tuple[float, float]]:
    grouped = defaultdict(list)
    for step, value in points:
        grouped[int(step // 1000)].append((step, value))
    return {key: (mean(x for x, _ in rows), mean(y for _, y in rows)) for key, rows in grouped.items()}


def chart(series: dict, title: str) -> str:
    curves = {identity: bucket(points) for identity, points in series.items() if points}
    points = [point for curve in curves.values() for point in curve.values()]
    if not points:
        return '<p class="diagnostic-empty">Not recorded in these runs yet.</p>'
    xmax = max(1000, max(x for x, _ in points))
    low, high = min(y for _, y in points), max(y for _, y in points)
    padding = max((high - low) * 0.08, abs(high) * 0.02, 0.01)
    low, high = low - padding, high + padding

    def x(step):
        return 42 + 188 * step / xmax

    def y(value):
        return 108 - 88 * (value - low) / (high - low)

    parts = [
        f'<svg viewBox="0 0 242 135" role="img" '
        f'aria-label="{html.escape(title)} versus physical steps per environment (average)">'
    ]
    parts.append('<text x="42" y="11">Y: linear · auto range</text>')
    for value in (low, (low + high) / 2, high):
        yy = y(value)
        parts.append(
            f'<line x1="42" x2="230" y1="{yy:.2f}" y2="{yy:.2f}" stroke="#e5eded"/>'
            f'<text x="38" y="{yy + 3:.2f}" text-anchor="end">{value:.3g}</text>'
        )
    for reward, color in COLORS.items():
        averages = defaultdict(list)
        for (kind, seed), curve in curves.items():
            if kind != reward:
                continue
            coords = " ".join(f"{x(xx):.2f},{y(yy):.2f}" for _, (xx, yy) in sorted(curve.items()))
            parts.append(
                f'<polyline points="{coords}" fill="none" stroke="{color}" opacity=".3" stroke-width="1">'
                f"<title>{LABELS[kind]}, seed {seed}</title></polyline>"
            )
            for key, point in curve.items():
                averages[key].append(point)
        values = [(mean(xx for xx, _ in rows), mean(yy for _, yy in rows)) for _, rows in sorted(averages.items())]
        if values:
            coords = " ".join(f"{x(xx):.2f},{y(yy):.2f}" for xx, yy in values)
            parts.append(f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2"/>')
            for xx, yy in values:
                parts.append(
                    f'<circle cx="{x(xx):.2f}" cy="{y(yy):.2f}" r="1.6" fill="{color}">'
                    f"<title>{LABELS[reward]}: {yy:.5g} at {xx:,.0f} physical steps/env</title></circle>"
                )
    parts.append(
        '<text x="42" y="128">0</text><text x="230" y="128" text-anchor="end">'
        f"{xmax / 1000:g}k physical steps/env</text></svg>"
    )
    return "".join(parts)


def render(root: Path) -> str:
    data = read_diagnostics(root)
    if not data["runs"]:
        return '<p class="meta">DSRL diagnostics apply to experiments that steer a frozen pretrained policy.</p>'
    sections = []
    for title, group in GROUPS:
        cards = []
        for key, label, explanation in group:
            cards.append(
                f'<article class="diagnostic-chart"><h4 title="{html.escape(explanation)}">'
                f"{html.escape(label)}</h4>{chart(data['series'].get(key, {}), label)}</article>"
            )
        sections.append(
            f'<section class="diagnostic-group"><h3>{title}</h3>'
            f'<div class="diagnostic-grid">{"".join(cards)}</div></section>'
        )
    return (
        '<h2>DSRL steering diagnostics</h2><p class="meta">These describe the learned noise-space actor and critic; '
        'the pretrained robot policy stays frozen. <span style="color:#087f8c">Sparse</span> / '
        '<span style="color:#d36a35">Robometer</span>. Faint: seeds; bold: seed mean. '
        "X-axis: physical steps per environment (average), matching the main curve. "
        "Logged total steps are divided by the training environment count before averaging within "
        "1,000-step/env bins. Metric values are unchanged. Hover over titles for definitions.</p>"
        '<p class="meta">Historical logs supply the available metrics; missing fields are shown explicitly. '
        "The Bellman target is distinct from the next-state bootstrap. Reward returns have different meanings "
        "across reward variants. New latent-distribution and target metrics require a run "
        "launched with the new logging.</p>" + "".join(sections)
    )
