"""Live, local video/score report for two LIBERO-10 checkpoint audits."""

from __future__ import annotations

import html
import json
from pathlib import Path

from experiments.libero.legacy.task38_video_report import score_plot
from experiments.libero.legacy.pi0_family_report import RBM_DATASETS, training_coverage


BASE = "/pi0-libero10-video-audit/"
NAMES = {
    6: "Put the white mug on the plate and the chocolate pudding to its right",
    8: "Put both moka pots on the stove",
}
BASELINE = {6: (13, 25), 8: (10, 25)}


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def selection_note(selection: dict) -> str:
    if not selection:
        return "Random examples selected after all 25 evaluations."
    available_successes = int(selection["available_successes"])
    available_failures = int(selection["available_failures"])
    note = (f'{len(selection["selected_successes"])} successes and '
            f'{len(selection["selected_failures"])} failures shown; '
            f'random selection seed {selection["selection_seed"]}')
    if min(available_successes, available_failures) < 2:
        note += (f'. Only {available_successes} successes and {available_failures} failures '
                 'occurred in 25 evaluations, so a 2+2 display was impossible')
    return note + "."


def evaluation_note(records: dict[int, dict], phase: str, scored: int, selection: dict) -> str:
    if len(records) < 25:
        phase = "queued for full evaluation" if phase == "complete" else phase
        return f'{len(records)}/25 evaluations recorded · {html.escape(phase)}.'
    successes = sum(bool(row["success"]) for row in records.values())
    return (f'{successes}/25 simulator successes · {scored}/4 selected videos scored · '
            f'{html.escape(phase)}. {html.escape(selection_note(selection))}')


def audit_training_coverage(data_root: Path, task_id: int) -> dict:
    """Match each audit's actual checkpoint to independently saved training provenance."""
    config = read_json(data_root / "pi0_libero10_video_audit_20261015" /
                       f"task{task_id:02d}" / "config.json", {})
    provenance = training_coverage(data_root)
    policy = (provenance["policy_known"] and
              config.get("pi0_checkpoint") == provenance["policy_checkpoint"])
    reward = (provenance["reward_known"] and
              config.get("reward_checkpoint") == provenance["reward_checkpoint"] and
              RBM_DATASETS["libero_10"] in provenance["reward_datasets"])
    return {"policy": bool(policy), "reward": bool(reward),
            "policy_checkpoint": config.get("pi0_checkpoint", ""),
            "reward_checkpoint": config.get("reward_checkpoint", "")}


def video_card(root: Path, task_id: int, kind: str, record: dict,
               source_folder: Path | None = None) -> str:
    is_demo = kind == "demos"
    ident = int(record["demo_id"] if is_demo else record["init_state_id"])
    is_final = kind in ("sparse", "robometer")
    is_600 = source_folder is not None
    prefix = "demo" if is_demo else "episode" if is_600 or is_final else "state"
    folder = source_folder or (root / "final_policies" / f"{kind}_seed100" if is_final else root / kind)
    length = int(record["length"])
    series = read_json(folder / "scores" / f"{prefix}_{ident}.json", [])
    events = record.get("reward_events", []) if is_demo else (
        [{"step": length, "reward": 1}] if record["success"] else [])
    label = (f"Expert demo {ident}" if is_demo else
             f"{'Frozen π₀' if kind == 'policy' else 'Sparse DSRL' if kind == 'sparse' else 'Robometer DSRL'} "
             f"{'success' if record['success'] else 'failure'} · state {ident}")
    video_kind = f"{kind}600" if is_600 else kind
    video = f"{BASE}videos/{task_id:02d}/{video_kind}/{prefix}_{ident}.mp4"
    chart = score_plot(series, length, events) if series else "<p>Robometer scoring pending.</p>"
    latest = (f"Final progress {series[-1]['progress']:.1%} · "
              f"predicted success {series[-1]['success_probability']:.1%}"
              if series else "Robometer pending")
    series_data = html.escape(json.dumps(series, separators=(",", ":")), quote=True)
    reward_step = int(events[0]["step"]) if events else -1
    return (f'<article class="episode" data-series="{series_data}" data-length="{length}" '
            f'data-reward-step="{reward_step}"><div class="episode-head"><h3>{html.escape(label)}</h3>'
            f'<span>{length} steps</span></div><video controls preload="metadata" playsinline '
            f'src="{video}"></video><div class="plot-pane"><div class="legend">'
            '<span class="progress">Progress</span><span class="success">Predicted success</span>'
            '<span class="ground-truth">Simulator reward</span></div>'
            f'{chart}<div class="live-values">{html.escape(latest)}</div></div></article>')


def task_section(data_root: Path, task_id: int) -> str:
    root = data_root / "pi0_libero10_video_audit_20261015" / f"task{task_id:02d}"
    root_600 = data_root / "pi0_libero10_video_audit_600_20261018" / f"task{task_id:02d}"
    status = read_json(root / "status.json", {})
    demos = read_json(root / "demos/manifest.json", {}).get("demonstrations", [])
    policy = read_json(root / "policy/episodes.json", [])
    attempts = read_json(root / "policy/attempts.json", [])
    scores = int(status.get("videos_scored", 0))
    successes = sum(bool(row["success"]) for row in policy)
    failures = len(policy) - successes
    baseline_s, baseline_n = BASELINE[task_id]
    coverage = audit_training_coverage(data_root, task_id)
    policy_label = "Yes — LIBERO-10 suite" if coverage["policy"] else "Unverified"
    reward_label = "Yes — LIBERO-10 suite" if coverage["reward"] else "Unverified"
    demo_cards = "".join(video_card(root, task_id, "demos", row) for row in demos)
    frozen_folder = root_600 / "policy"
    frozen_records = {int(row["init_state_id"]): row
                      for row in read_json(frozen_folder / "episodes.json", [])}
    frozen_selection = read_json(frozen_folder / "selection.json", {})
    frozen_states = frozen_selection.get("selected_state_ids", [])
    frozen_cards = "".join(video_card(root, task_id, "policy", frozen_records[state], frozen_folder)
                           for state in frozen_states if state in frozen_records)
    frozen_phase = read_json(frozen_folder / "status.json", {}).get("status", "queued")
    frozen_scored = sum((frozen_folder / "scores" / f"episode_{state}.json").exists()
                        for state in frozen_states)
    final_sections = []
    for arm, heading in (("sparse", "π₀ + DSRL fine-tuned with sparse reward"),
                         ("robometer", "π₀ + DSRL fine-tuned with Robometer reward")):
        directory = root_600 / arm
        records = {int(row["init_state_id"]): row
                   for row in read_json(directory / "episodes.json", [])}
        selection = read_json(directory / "selection.json", {})
        selected_states = selection.get("selected_state_ids", [])
        scored = sum((directory / "scores" / f"episode_{state}.json").exists()
                     for state in selected_states)
        phase = read_json(directory / "status.json", {}).get("status", "queued")
        cards = "".join(video_card(root, task_id, arm, records[state], directory)
                        for state in selected_states if state in records)
        final_sections.append(f'<h3>{html.escape(heading)}</h3>'
                              f'<p class="meta">Final actor · seed 100 · 600-step horizon · '
                              f'{evaluation_note(records, phase, scored, selection)}</p>'
                              f'<div class="cards">{cards or "<p>Rollouts pending.</p>"}</div>')
    terminal_rows = []
    for label, kind, records, key, prefix in (
        ("Expert demos", "demos", demos, "demo_id", "demo"),
        ("π₀ successes", "policy", [row for row in policy if row["success"]], "init_state_id", "state"),
        ("π₀ failures", "policy", [row for row in policy if not row["success"]], "init_state_id", "state"),
    ):
        values = []
        for row in records:
            series = read_json(root / kind / "scores" / f"{prefix}_{int(row[key])}.json", [])
            if series:
                values.append(series[-1]["success_probability"])
        value = f"{sum(values) / len(values):.1%}" if values else "Pending"
        terminal_rows.append(f"<tr><td>{label}</td><td>{len(values)}/{len(records)}</td>"
                             f"<td>{value}</td></tr>")
    terminal_table = ('<div class="table-wrap"><table><thead><tr><th>Outcome group</th>'
                      '<th>Videos scored</th><th>Mean terminal Robometer success estimate</th>'
                      '</tr></thead><tbody>' + "".join(terminal_rows) + '</tbody></table></div>')
    return (f'<section class="task" id="task-{task_id:02d}"><div class="eyebrow">'
            f'LIBERO-10 · task {task_id:02d} · checkpoint training coverage</div>'
            f'<h2>{html.escape(NAMES[task_id])}</h2>'
            '<div class="coverage"><div><b>In π₀ training data?</b> '
            f'{html.escape(policy_label)}<br><code>{html.escape(coverage["policy_checkpoint"] or "unknown")}</code>'
            '</div><div><b>In Robometer training data?</b> '
            f'{html.escape(reward_label)}<br><code>{html.escape(coverage["reward_checkpoint"] or "unknown")}</code>'
            f'<br><code>{RBM_DATASETS["libero_10"]}</code></div></div>'
            f'<p class="meta">Official 25-state frozen π₀ baseline at 400 steps: {baseline_s}/{baseline_n} '
            f'({baseline_s/baseline_n:.0%}) success. Original 400-step selection audit: '
            f'{html.escape(str(status.get("status", "queued")))} · {len(demos)}/4 demos · '
            f'{successes}/4 selected successes · {failures}/4 selected failures · '
            f'{scores}/12 original audit videos scored · {len(attempts)} π₀ rollout attempts. '
            f'The 600-step audits below evaluate all 25 held-out reset states per policy.</p>'
            f'<details><summary>Original 400-step selection audit scores</summary>{terminal_table}</details>'
            '<h3>Expert demonstrations</h3><div class="cards">'
            f'{demo_cards or "<p>Preparing expert demos.</p>"}</div>'
            '<h3>Frozen π₀ checkpoint</h3>'
            f'<p class="meta">600-step horizon · '
            f'{evaluation_note(frozen_records, frozen_phase, frozen_scored, frozen_selection)}</p><div class="cards">'
            f'{frozen_cards or "<p>Collecting frozen-policy rollouts.</p>"}</div>'
            f'{"".join(final_sections)}</section>')


def render(data_root: Path) -> bytes:
    sections = "".join(task_section(data_root, task_id) for task_id in NAMES)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="refresh" content="60"><title>LIBERO-10 π₀ video and Robometer audit</title>
<style>
:root{{--ink:#17313b;--muted:#61757a;--line:#dce5e6;--bg:#f3f7f7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1450px;margin:auto;padding:25px 20px 70px}}a{{color:#087f8c}}
h1{{font-size:clamp(28px,4vw,42px);margin:6px 0}}h2{{font-size:25px;margin:5px 0}}
h3{{font-size:17px;margin:18px 0 8px}}p{{margin:9px 0}}.meta{{color:var(--muted);font-size:14px}}
.eyebrow{{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#577176;font-weight:700}}
.task{{background:white;border:1px solid var(--line);border-radius:16px;padding:24px;margin:25px 0}}
.coverage{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:13px 0}}
.coverage>div{{border:1px solid var(--line);border-radius:10px;background:#f8fbfb;padding:10px;font-size:14px}}
.coverage code{{font-size:12px;overflow-wrap:anywhere}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}}
th,td{{padding:6px 9px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted)}}
.cards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
.episode{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px;display:grid;grid-template-columns:minmax(150px,35%) minmax(0,1fr);gap:8px 12px;align-items:start}}
.episode-head{{grid-column:1/-1;display:flex;justify-content:space-between;gap:10px;align-items:baseline}}
.episode-head h3{{margin:0;font-size:16px}}.episode-head span{{font-size:12px;color:var(--muted)}}
video{{display:block;width:100%;max-width:235px;aspect-ratio:1;background:#102024;border-radius:7px}}
.plot-pane{{min-width:0}}.score-plot{{width:100%;height:auto}}.grid{{stroke:#e5eded}}
.tick,.axis{{fill:#718187;font-size:12px}}.legend{{display:flex;gap:5px 10px;flex-wrap:wrap;font-size:11px}}
.legend span:before{{content:"";display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}}
.progress:before{{background:#087f8c}}.success:before{{background:#d36a35}}
.ground-truth:before{{background:#7754a7}}.live-values{{font-size:12px;color:var(--muted)}}
@media(max-width:1000px){{.cards{{grid-template-columns:1fr}}}}
@media(max-width:520px){{.episode,.coverage{{grid-template-columns:1fr}}video{{max-width:235px}}}}
</style></head><body><main><a href="/">← Main experiment dashboard</a> ·
<a href="/pi0-libero-families-eval/">Four-family π₀ baseline</a>
<div class="eyebrow">Policy comparison · video and reward audit</div><h1>LIBERO-10 tasks 06 and 08</h1>
<p>Each policy is evaluated on all 25 held-out LIBERO reset states with a 600-step horizon. Afterward, four display videos are randomly selected per policy: two successes and two failures when both outcomes occur often enough. Each video has Robometer progress and success predictions beside the ground-truth sparse reward; scrub to inspect them.</p>
<p class="meta">The selected videos illustrate outcomes; the 25-episode counts are the evaluation results. A policy with fewer than two successes or failures cannot supply a balanced 2+2 display, and its section shows the shortfall. Both tasks belong to the LIBERO-10 suite included in the released <code>pi0_libero</code> policy training config and the <code>Robometer-4B</code> checkpoint's <code>rbm-1m-id</code> training mix. Suite-level membership does not prove whether these exact demonstrations or initial states were seen in training. Four expert demos per task come from the pinned LIBERO dataset mirror (revision <code>f13aa24a</code>) and retain their original lengths. Robometer scores are computed afterward for the selected policy videos and all demos. One video frame represents four physical steps; reward scores are sampled every 20 steps and at termination. The sparse simulator reward is one at success and zero otherwise. This page refreshes every 60 seconds.</p>
{sections}</main><script>
for (const card of document.querySelectorAll('.episode')) {{
  const series = JSON.parse(card.dataset.series || '[]');
  if (!series.length) continue;
  const video = card.querySelector('video');
  const cursor = card.querySelector('.playhead');
  const label = card.querySelector('.live-values');
  const length = Number(card.dataset.length);
  const rewardStep = Number(card.dataset.rewardStep);
  function update() {{
    const elapsed = video.currentTime;
    const step = Math.min(length, elapsed * 40);
    const x = 38 + (620 - 38 - 12) * step / Math.max(length, 1);
    cursor.setAttribute('x1', x.toFixed(1)); cursor.setAttribute('x2', x.toFixed(1));
    let nearest = series[0];
    for (const item of series) {{ if (Math.abs(item.video_time-elapsed) < Math.abs(nearest.video_time-elapsed)) nearest=item; }}
    const reward = rewardStep >= 0 && step >= rewardStep ? 1 : 0;
    label.textContent = `Step ${{nearest.step}} · Progress ${{(nearest.progress*100).toFixed(1)}}% · Predicted success ${{(nearest.success_probability*100).toFixed(1)}}% · Simulator reward ${{reward}}`;
  }}
  video.addEventListener('timeupdate', update); video.addEventListener('seeked', update); update();
}}
</script></body></html>'''
    return page.encode("utf-8")
