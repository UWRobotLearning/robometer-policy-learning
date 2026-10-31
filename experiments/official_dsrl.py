"""Read-only dashboard adapter for runs of the separate official DSRL checkout."""

from __future__ import annotations

import csv
import html
import json
from collections import defaultdict
from pathlib import Path

from experiments.libero.analysis import read_jsonl
from experiments.libero.diagnostics import chart
from experiments.reward_audit.metrics import wilson

GROUPS = (
    (
        "Latent-noise policy",
        [
            ("latent_mean_norm", "Mean-vector norm"),
            ("latent_std_norm", "Std-vector norm"),
            ("latent_mean", "Mean parameter"),
            ("latent_std", "Std parameter"),
            ("latent_std_max", "Largest std"),
        ],
    ),
    (
        "Entropy and latent actor",
        [
            ("entropy", "Latent entropy"),
            ("ent_coef", "Temperature"),
            ("ent_coef_loss", "Temperature loss"),
            ("actor_loss", "Actor loss"),
            ("actor_noise_q", "Actor noise Q"),
        ],
    ),
    (
        "Action-space critic Q(s, a)",
        [
            ("action_q", "Replay action Q"),
            ("bellman_target", "Bellman target"),
            ("bootstrap_soft_q", "Soft bootstrap Q"),
            ("critic_loss", "Action critic loss"),
            ("action_q_disagreement", "Action critic disagreement"),
        ],
    ),
    (
        "Distilled noise critic Q(s, w)",
        [
            ("noise_q", "Noise Q prediction"),
            ("distillation_target", "Action-Q teacher target"),
            ("noise_critic_loss", "Noise distillation loss"),
            ("n_updates", "Actor/critic updates"),
            ("replay_transitions", "Replay chunks"),
        ],
    ),
    (
        "Rollouts and evaluations",
        [
            ("rollout_success", "Training success"),
            ("rollout_return", "Training return"),
            ("rollout_length", "Episode length"),
            ("eval_success", "Official evaluation success"),
            ("eval_native_success", "Native simulator success"),
        ],
    ),
)


def roots(data_root: Path) -> list[Path]:
    paths = [path.parent for path in (data_root / "official_dsrl_runs").glob("*/manifest.json")]
    return sorted(paths, key=lambda p: json.loads((p / "manifest.json").read_text())["created"], reverse=True)


def evaluations(root: Path) -> list[dict]:
    path = root / "evaluation.csv"
    if not path.exists():
        return []
    rows = []
    with path.open() as stream:
        for row in csv.DictReader(stream):
            try:
                rows.append({k: v if k == "policy" else float(v) for k, v in row.items()})
            except (ValueError, TypeError):
                pass
    return rows


def success_plot(root: Path) -> str:
    rows = evaluations(root)
    metadata = json.loads((root / "manifest.json").read_text())

    def x(step):
        return 48 + 570 * step / metadata["physical_steps"]

    def y(rate):
        return 185 - 160 * rate

    parts = ['<svg viewBox="0 0 640 220" role="img" aria-label="Success rate on a linear axis">']
    for rate in (0, 0.5, 1):
        parts.append(
            f'<line x1="48" x2="618" y1="{y(rate)}" y2="{y(rate)}" stroke="#dce5e6"/>'
            f'<text x="40" y="{y(rate) + 4}" text-anchor="end">{rate:.0%}</text>'
        )
    frozen = next((row for row in rows if row["policy"] == "frozen"), None)
    if frozen:
        lo, hi = wilson(int(frozen["successes"]), int(frozen["episodes"]))
        parts.append(
            f'<rect x="48" y="{y(hi)}" width="570" height="{y(lo) - y(hi)}" fill="#475569" opacity=".12"/>'
            f'<line x1="48" x2="618" y1="{y(frozen["success_rate"])}" y2="{y(frozen["success_rate"])}" '
            'stroke="#111827" stroke-width="2" stroke-dasharray="2 5"/>'
        )
    trained = [row for row in rows if row["policy"] == "dsrl_na"]
    if trained:
        coords = " ".join(f"{x(row['env_steps']):.2f},{y(row['success_rate']):.2f}" for row in trained)
        parts.append(f'<polyline points="{coords}" fill="none" stroke="#087f8c" stroke-width="2.5"/>')
        for row in trained:
            lo, hi = wilson(int(row["successes"]), int(row["episodes"]))
            parts.append(
                f'<line x1="{x(row["env_steps"])}" x2="{x(row["env_steps"])}" '
                f'y1="{y(lo)}" y2="{y(hi)}" stroke="#087f8c" opacity=".4"/>'
            )
            parts.append(
                f'<circle cx="{x(row["env_steps"])}" cy="{y(row["success_rate"])}" r="3" fill="#087f8c">'
                f"<title>{row['successes']:g}/{row['episodes']:g} at "
                f"{row['env_steps'] / metadata.get('n_envs', 1):,.0f} physical steps/env</title></circle>"
            )
    parts.append(
        '<text x="48" y="210">0</text><text x="618" y="210" text-anchor="end">'
        f"{metadata['physical_steps'] / metadata.get('n_envs', 1):,.0f} physical steps/env</text></svg>"
    )
    return "".join(parts)


def card(root: Path) -> str:
    metadata = json.loads((root / "manifest.json").read_text())
    status = json.loads((root / "status.json").read_text()) if (root / "status.json").exists() else {"status": "queued"}
    label = f"Robomimic {metadata['task'].capitalize()} · {metadata['algorithm']}"
    return (
        f'<section class="card experiment-plot" id="official-{html.escape(root.name)}">'
        '<div class="eyebrow">Official DSRL repository · diffusion MLP</div>'
        f'<h2><a href="/official-dsrl/{html.escape(root.name)}/">{label} ↗</a></h2>'
        f'<time class="experiment-date">Started {metadata["created"][:10]} · UTC</time>'
        f'<p class="meta">{html.escape(status["status"])} · {status.get("env_steps", 0):,} / '
        f"{metadata['physical_steps']:,} total steps · seed {metadata['seed']}</p>"
        f'<a class="diagnostics-link" href="#official-{html.escape(root.name)}">Open experiment diagnostics →</a>'
        '<div class="axis-scale-note">Y-axis: linear · fixed 0-100%</div>'
        f'{success_plot(root)}<p class="meta">Teal: DSRL-NA · Dotted black: frozen policy '
        "(95% Wilson episode CIs). Native sparse reward, shifted by -1. "
        "Step-zero teal is the initialized steering actor, before learning. "
        "One seed; intervals measure evaluation uncertainty, not seed variability. "
        "X-axis: average physical steps per training environment.</p></section>"
    )


def render_diagnostics(root: Path) -> str:
    metadata = json.loads((root / "manifest.json").read_text())
    series = defaultdict(list)
    for row in read_jsonl(root / "metrics.jsonl"):
        for key, value in row["metrics"].items():
            series[key].append((row["env_steps"], value))
    for row in read_jsonl(root / "episodes.jsonl"):
        for key, source in [("rollout_success", "success"), ("rollout_return", "return"), ("rollout_length", "length")]:
            series[key].append((row["env_steps"], float(row[source])))
    for row in evaluations(root):
        if row["policy"] == "dsrl_na":
            series["eval_success"].append((row["env_steps"], row["success_rate"]))
            series["eval_native_success"].append((row["env_steps"], row["native_success_rate"]))
    sections = []
    for title, group in GROUPS:
        plots = "".join(
            f'<article class="diagnostic-chart"><h4>{label}</h4>'
            + chart(
                {
                    ("sparse", metadata["seed"]): [
                        (step / metadata.get("n_envs", 1), value) for step, value in series[key]
                    ]
                },
                label,
            )
            + "</article>"
            for key, label in group
        )
        sections.append(
            f'<section class="diagnostic-group"><h3>{title}</h3><div class="diagnostic-grid">{plots}</div></section>'
        )
    return (
        '<h2>Official DSRL-NA diagnostics</h2><p class="meta">This algorithm trains an action-space critic and '
        "distills it into a noise-space critic. X-axis: physical steps per training environment (average); "
        "1,000-step/env bins. "
        "Native success detects any successful state; the upstream evaluation criterion requires a fully "
        "successful four-step chunk. All axes are linear.</p>" + "".join(sections)
    )


def render_page(root: Path) -> bytes:
    metadata = json.loads((root / "manifest.json").read_text())
    style = (Path(__file__).parent / "libero/video_report.css").read_text()
    style += """.diagnostic-grid{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:8px}
.diagnostic-chart{border:1px solid #dce5e6;border-radius:6px;padding:8px;min-width:0}
.diagnostic-chart h4{font-size:12px;margin:0}.diagnostic-chart svg{width:100%}
.diagnostic-chart text{font-size:10px;fill:#61757a}
.diagnostic-empty{font-size:12px;color:#61757a;min-height:80px}.experiment-plot{max-width:600px}
.experiment-plot svg{width:100%}.experiment-plot text{font-size:12px;fill:#61757a}
.experiment-date,.axis-scale-note{font-size:11px;color:#61757a}
@media(max-width:900px){.diagnostic-grid{grid-template-columns:repeat(2,minmax(0,1fr))}}"""
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in metadata["implementation_notes"])
    return (
        f'<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">'
        f'<meta http-equiv="refresh" content="60"><title>Official DSRL · {metadata["task"]}</title>'
        f"<style>{style}</style>"
        '</head><body><main><a href="/">← Main experiment dashboard</a>'
        f"{card(root)}{render_diagnostics(root)}"
        "<details><summary>How to read these diagnostics</summary>"
        "<p>The pretrained diffusion policy stays frozen. The latent actor changes the noise supplied to it. "
        "Mean/std describe the Gaussian before tanh; entropy describes the bounded noise distribution. "
        "Temperature controls the entropy bonus. Low entropy or std can indicate reduced exploration.</p>"
        "<p>Action Q is trained against the Bellman target. The noise critic learns to match action Q "
        "after passing noise through the frozen diffusion policy; its distillation loss measures that mismatch. "
        "Large disagreement or unstable targets can help identify critic problems. These are entropy-regularized "
        "Q values, so they can be positive despite nonpositive environment rewards.</p>"
        "<p>Judge learning by held-out success alongside these diagnostics; a small critic or actor loss "
        "alone does not establish good behavior.</p></details>"
        "<details><summary>Protocol and provenance</summary>"
        f"<p>{metadata['eval_episodes']} episodes per evaluation; "
        f"{metadata['eval_every_physical']:,} total physical steps "
        f"between evaluations after {metadata['warmup_physical_steps']:,} warmup steps.</p>"
        f"<p>Frozen checkpoint: <code>{html.escape(metadata['checkpoint'])}</code></p>"
        f"<p>Official repository commit: <code>{metadata['commit']}</code></p><ul>{notes}</ul></details>"
        "</main></body></html>"
    ).encode()
