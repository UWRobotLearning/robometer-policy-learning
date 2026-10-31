"""Render frozen-pi0 LIBERO-90 video and Robometer audit subpages."""

from __future__ import annotations

import html
import json
from pathlib import Path


BASE = "/task38-pi0-baseline/"
TASK_NAMES = {38: "Put the right moka pot on the stove",
              55: "Pick up the alphabet soup and put it in the tray"}


def score_plot(series: list[dict], length: int, reward_events: list[dict]) -> str:
    width, height, left, right, top, bottom = 620, 215, 38, 12, 12, 32
    x = lambda step: left + (width - left - right) * step / max(length, 1)
    y = lambda value: top + (height - top - bottom) * (1 - value)
    parts = [f'<svg class="score-plot" viewBox="0 0 {width} {height}" role="img" '
             'aria-label="Robometer estimates and sparse simulator reward versus rollout step">']
    for value in (0, .5, 1):
        yy = y(value)
        parts.append(f'<line x1="{left}" y1="{yy:.1f}" x2="{width-right}" y2="{yy:.1f}" class="grid"/>')
        parts.append(f'<text x="{left-7}" y="{yy+4:.1f}" text-anchor="end" class="tick">{value:.0%}</text>')
    for step in (0, length // 2, length):
        xx = x(step)
        parts.append(f'<text x="{xx:.1f}" y="{height-8}" text-anchor="middle" class="tick">{step}</text>')
    for key, color in (("progress", "#087f8c"), ("success_probability", "#d36a35")):
        points = " ".join(f'{x(row["step"]):.1f},{y(row[key]):.1f}' for row in series)
        parts.append(f'<polyline points="{points}" fill="none" stroke="{color}" '
                     'stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"/>')
    parts.append(f'<line x1="{left}" y1="{y(0):.1f}" x2="{x(length):.1f}" '
                 f'y2="{y(0):.1f}" stroke="#7754a7" stroke-width="2.5"/>')
    for event in reward_events:
        xx, yy = x(event["step"]), y(event["reward"])
        parts.append(f'<line x1="{xx:.1f}" y1="{y(0):.1f}" x2="{xx:.1f}" '
                     f'y2="{yy:.1f}" stroke="#7754a7" stroke-width="2.5"/>')
        parts.append(f'<circle cx="{xx:.1f}" cy="{yy:.1f}" r="4" fill="#7754a7"/>')
    parts.append(f'<line class="playhead" x1="{left}" y1="{top}" x2="{left}" '
                 f'y2="{height-bottom}" stroke="#26343a" stroke-width="1.5" stroke-dasharray="4 3"/>')
    parts.append(f'<text x="{width/2}" y="{height-1}" text-anchor="middle" class="axis">Physical steps</text></svg>')
    return "".join(parts)


def render(out: Path, task_id: int = 38) -> bytes:
    if task_id not in TASK_NAMES:
        raise ValueError(f"Unsupported video audit task: {task_id}")
    base = f"/task{task_id}-pi0-baseline/"
    task_name = TASK_NAMES[task_id]
    records_path = out / "episodes.json"
    records = json.loads(records_path.read_text()) if records_path.exists() else []
    status_path = out / "status.json"
    status = json.loads(status_path.read_text()).get("status", "queued") if status_path.exists() else "queued"
    selected = [*([r for r in records if r["success"]][:2]),
                *([r for r in records if not r["success"]][:2])]
    selected.extend(record for record in records if record not in selected)
    selected = selected[:4]
    cards = []
    for record in selected:
        state = int(record["init_state_id"])
        score_path = out / "scores" / f"episode_{state}.json"
        series = json.loads(score_path.read_text()) if score_path.exists() else []
        label = "Success" if record["success"] else "Failure"
        video_url = f"{base}videos/episode_{state}.mp4"
        reward_events = ([{"step": int(record["length"]), "reward": 1}]
                         if record["success"] else [])
        chart = score_plot(series, int(record["length"]), reward_events) if series else '<p>Robometer scoring pending.</p>'
        current = ('<span class="live-values">Progress — · Success estimate —</span>' if series else "")
        data = html.escape(json.dumps(series, separators=(",", ":")), quote=True)
        reward_step = reward_events[-1]["step"] if reward_events else -1
        cards.append(f'<article class="episode" data-series="{data}" data-length="{record["length"]}" '
                     f'data-reward-step="{reward_step}">'
                     f'<div class="episode-head"><h2>{label} · state {state}</h2>'
                     f'<span>{record["length"]} steps</span></div>'
                     f'<video controls preload="metadata" playsinline src="{video_url}"></video>'
                     f'<div class="plot-pane"><div class="chart-legend"><span class="progress">Progress</span>'
                     f'<span class="success">Predicted success</span>'
                     f'<span class="ground-truth">Simulator reward</span>{current}</div>{chart}</div></article>')
    demo_manifest_path = out / "dataset_demos" / "manifest.json"
    demo_manifest = json.loads(demo_manifest_path.read_text()) if demo_manifest_path.exists() else {}
    demo_cards = []
    for record in demo_manifest.get("demonstrations", []):
        demo_id = int(record["demo_id"])
        if demo_id not in (0, 1):
            continue
        score_path = out / "dataset_demos" / "scores" / f"demo_{demo_id}.json"
        series = json.loads(score_path.read_text()) if score_path.exists() else []
        length = int(record["length"])
        reward_events = record.get("reward_events", [])
        chart = score_plot(series, length, reward_events) if series else '<p>Robometer scoring pending.</p>'
        current = '<span class="live-values">Progress — · Success estimate —</span>' if series else ""
        data = html.escape(json.dumps(series, separators=(",", ":")), quote=True)
        reward_step = reward_events[-1]["step"] if reward_events else -1
        demo_cards.append(f'<article class="episode" data-series="{data}" data-length="{length}" '
                          f'data-reward-step="{reward_step}">'
                          f'<div class="episode-head"><h2>Expert demonstration {demo_id}</h2>'
                          f'<span>{length} steps · simulator success</span></div>'
                          f'<video controls preload="metadata" playsinline '
                          f'src="{base}videos/demo_{demo_id}.mp4"></video>'
                          f'<div class="plot-pane"><div class="chart-legend"><span class="progress">Progress</span>'
                          f'<span class="success">Predicted success</span>'
                          f'<span class="ground-truth">Simulator reward</span>{current}</div>{chart}</div></article>')
    final_dirs = {arm: out / "final_policies" / f"{arm}_seed100"
                  for arm in ("sparse", "robometer")}
    final_records = {}
    for arm, directory in final_dirs.items():
        path = directory / "episodes.json"
        final_records[arm] = {int(record["init_state_id"]): record
                              for record in json.loads(path.read_text())} if path.exists() else {}
    paired_states = sorted(set(final_records["sparse"]) & set(final_records["robometer"]))
    differing = [state for state in paired_states
                 if final_records["sparse"][state]["success"] != final_records["robometer"][state]["success"]]
    both_success = [state for state in paired_states
                    if final_records["sparse"][state]["success"] and final_records["robometer"][state]["success"]]
    display_states = []
    for state in differing[:1] + both_success[:1] + differing[1:] + paired_states:
        if state not in display_states:
            display_states.append(state)
        if len(display_states) == 2:
            break
    final_cards = []
    for state in display_states:
        for arm in ("sparse", "robometer"):
            record = final_records[arm][state]
            directory = final_dirs[arm]
            score_path = directory / "scores" / f"episode_{state}.json"
            series = json.loads(score_path.read_text()) if score_path.exists() else []
            length = int(record["length"])
            reward_events = ([{"step": length, "reward": 1}] if record["success"] else [])
            chart = score_plot(series, length, reward_events) if series else '<p>Robometer scoring pending.</p>'
            current = '<span class="live-values">Progress — · Success estimate —</span>' if series else ""
            data = html.escape(json.dumps(series, separators=(",", ":")), quote=True)
            reward_step = length if record["success"] else -1
            reward_name = "Sparse reward" if arm == "sparse" else "Robometer reward"
            outcome = "Success" if record["success"] else "Failure"
            final_cards.append(f'<article class="episode" data-series="{data}" data-length="{length}" '
                               f'data-reward-step="{reward_step}">'
                               f'<div class="episode-head"><h2>{reward_name} · state {state}</h2>'
                               f'<span>{outcome} · {length} steps</span></div>'
                               f'<video controls preload="metadata" playsinline '
                               f'src="{base}videos/{arm}/episode_{state}.mp4"></video>'
                               f'<div class="plot-pane"><div class="chart-legend">'
                               f'<span class="progress">Progress</span>'
                               f'<span class="success">Predicted success</span>'
                               f'<span class="ground-truth">Simulator reward</span>{current}</div>'
                               f'{chart}</div></article>')
    final_rows = []
    for state in paired_states:
        cells = []
        for arm in ("sparse", "robometer"):
            record = final_records[arm][state]
            outcome = "Success" if record["success"] else "Failure"
            cells.append(f'<td>{outcome}</td><td><a href="{base}videos/{arm}/episode_{state}.mp4">Video</a></td>')
        final_rows.append(f'<tr><td>{state}</td>{"".join(cells)}</tr>')
    final_counts = ", ".join(f'{"sparse" if arm == "sparse" else "Robometer"}: '
                             f'{sum(r["success"] for r in final_records[arm].values())}/'
                             f'{len(final_records[arm])}' for arm in ("sparse", "robometer"))
    table_rows = []
    for record in records:
        state = int(record["init_state_id"])
        score_path = out / "scores" / f"episode_{state}.json"
        series = json.loads(score_path.read_text()) if score_path.exists() else []
        progress = f'{series[-1]["progress"]:.1%}' if series else "—"
        probability = f'{series[-1]["success_probability"]:.1%}' if series else "—"
        table_rows.append(f'<tr><td>{state}</td><td>{"Success" if record["success"] else "Failure"}</td>'
                          f'<td>{record["length"]}</td><td>{progress}</td><td>{probability}</td>'
                          f'<td><a href="{base}videos/episode_{state}.mp4">Video</a></td></tr>')
    successes = sum(record["success"] for record in records)
    failed_estimates = [record["final_success_probability"] for record in records
                        if not record["success"] and "final_success_probability" in record]
    failure_signal = (f'<p><b>Reward-model check:</b> The {len(failed_estimates)} simulator failures '
                      f'received final success estimates from {min(failed_estimates):.1%} to '
                      f'{max(failed_estimates):.1%}.</p>' if failed_estimates else "")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Task {task_id} · π₀ checkpoint video audit</title>
<style>
:root{{--ink:#17313b;--muted:#61757a;--line:#dce5e6;--bg:#f3f7f7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1450px;margin:auto;padding:20px 18px 55px}}a{{color:#087f8c}}h1{{font-size:clamp(26px,4vw,42px);margin:4px 0 10px}}
.eyebrow{{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#577176;font-weight:700}}
.meta{{font-size:14px;color:var(--muted)}}.cards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px;margin:14px 0 24px}}
.episode,.table-card{{background:white;border:1px solid var(--line);border-radius:14px;padding:12px;box-shadow:0 5px 22px #17313b09}}
.episode{{display:grid;grid-template-columns:minmax(145px,34%) minmax(0,1fr);gap:6px 12px;align-items:start}}
.episode-head{{display:flex;grid-column:1/-1;justify-content:space-between;align-items:baseline;gap:12px}}h2{{font-size:16px;margin:0 0 4px}}
.episode-head span{{font-size:12px;color:var(--muted)}}video{{display:block;width:100%;max-width:230px;aspect-ratio:1;background:#102024;border-radius:7px}}
.plot-pane{{min-width:0}}
.score-plot{{width:100%;height:auto}}.grid{{stroke:#e5eded}}.tick,.axis{{fill:#718187;font-size:12px}}.axis{{font-size:13px}}
.chart-legend{{display:flex;gap:5px 10px;flex-wrap:wrap;font-size:11px;margin:0 0 2px}}
.chart-legend .progress:before,.chart-legend .success:before,.chart-legend .ground-truth:before{{content:"";display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}}
.chart-legend .progress:before{{background:#087f8c}}.chart-legend .success:before{{background:#d36a35}}.chart-legend .ground-truth:before{{background:#7754a7}}
.live-values{{color:var(--muted)}}.table-card{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:8px;text-align:left;border-bottom:1px solid var(--line)}}th{{color:var(--muted)}}
@media(max-width:1000px){{.cards{{grid-template-columns:1fr}}}}
@media(max-width:520px){{.episode{{grid-template-columns:1fr}}.episode video{{max-width:230px}}}}
</style></head><body><main><a href="/">← All task curves</a>
<div class="eyebrow">LIBERO-90 · task {task_id} · checkpoint video audit</div>
<h1>{html.escape(task_name)}</h1>
<p class="meta"><b>In checkpoint task-training data?</b> π₀ <code>pi0_libero</code>: <b>No — LIBERO-90 suite held out</b> · Robometer <code>Robometer-4B</code>: <b>No — LIBERO-90 suite held out</b>. These are suite-level labels; similar objects or instructions may appear in other training suites.</p>
<h2>Final trained policies · seed 100</h2>
<p class="meta">The two SAC/DSRL actors were trained for 100,000 physical transitions, one with sparse simulator reward and one with Robometer progress reward. Both steer the same frozen π₀ checkpoint. Videos use the same held-out initial states and deterministic actor actions. Robometer's plotted estimates were computed afterward for both arms; purple is the actual sparse simulator reward. Recording outcomes so far: {html.escape(final_counts)}. Paired states with differing outcomes are prioritized below.</p>
<div class="cards">{"".join(final_cards) or "<p>Final-policy videos are being prepared.</p>"}</div>
<section class="table-card"><h2>All paired final-policy videos</h2><table><thead><tr><th>Initial state</th><th>Sparse-trained outcome</th><th>Recording</th><th>Robometer-trained outcome</th><th>Recording</th></tr></thead><tbody>{"".join(final_rows)}</tbody></table></section>
<h2>Initial π₀ checkpoint</h2>
<p>Ten held-out initial states (25–34), seed 100. Frozen <code>pi0_libero</code> policy, with no RL fine-tuning: {successes}/{len(records)} simulator successes. Status: {html.escape(status)}.</p>
{failure_signal}
<p class="meta">Up to four rollouts are shown: the first two successes and failures where available, then other states to fill remaining slots. All ten videos are linked in the table. Robometer predictions were computed afterward from the observed video prefix using the pinned <code>Robometer-4B</code> checkpoint; they did not control the policy. Purple shows the simulator's sparse reward: zero until a success step, where it is one. For these π₀ rollouts, this trace is reconstructed from the saved terminal outcome under the reward contract checked during collection. Video records one frame every four physical steps and plays at 10 fps. Predictions are sampled every 20 physical steps and at termination. Scrub a video to follow the estimates and reward.</p>
<div class="cards">{"".join(cards) or "<p>Rollouts are being collected.</p>"}</div>
<h2>Dataset demonstrations</h2>
<p class="meta">These are two expert trajectories (<code>demo_0</code> and <code>demo_1</code>) from the LIBERO-90 task dataset, separate from the frozen π₀ evaluations above. Both end with simulator success. Their purple reward markers come directly from the dataset's <code>rewards</code> arrays. Robometer estimates are computed afterward from the observed frames. The source is the <a href="https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_90">pinned LIBERO-90 dataset mirror</a>.</p>
<div class="cards">{"".join(demo_cards) or "<p>Dataset demonstration videos are being prepared.</p>"}</div>
<section class="table-card"><h2>All ten evaluations</h2><table><thead><tr><th>Initial state</th><th>Simulator outcome</th><th>Steps</th><th>Final progress</th><th>Final success estimate</th><th>Recording</th></tr></thead>
<tbody>{"".join(table_rows)}</tbody></table></section></main>
<script>
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
    cursor.setAttribute('x1', x.toFixed(1));
    cursor.setAttribute('x2', x.toFixed(1));
    let nearest = series[0];
    for (const item of series) {{ if (Math.abs(item.video_time - elapsed) < Math.abs(nearest.video_time - elapsed)) nearest = item; }}
    const reward = rewardStep >= 0 && step >= rewardStep ? 1 : 0;
    label.textContent = `Step ${{nearest.step}} · Progress ${{(nearest.progress*100).toFixed(1)}}% · Success estimate ${{(nearest.success_probability*100).toFixed(1)}}% · Simulator reward ${{reward}}`;
  }}
  video.addEventListener('timeupdate', update);
  video.addEventListener('seeked', update);
  update();
}}
</script></body></html>'''
    return page.encode("utf-8")
