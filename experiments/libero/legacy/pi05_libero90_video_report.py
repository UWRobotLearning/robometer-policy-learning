"""Live video and Robometer report for frozen π0.5 on LIBERO-90 tasks 02 and 28."""
from __future__ import annotations

import html
import json
from pathlib import Path

from experiments.libero.legacy.pi05_libero90_video_audit import ROOT, TASK_IDS, task_name
from experiments.libero.legacy.task38_video_report import score_plot
from experiments.reward_audit.metrics import wilson

BASE = '/pi05-libero90-video-audit/'


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return default


def card(task_id: int, kind: str, row: dict, folder: Path) -> str:
    is_demo = kind == 'demos'
    ident = int(row['demo_id'] if is_demo else row['init_state_id'])
    prefix = f'demo_{ident}' if is_demo else f'episode_{ident}'
    score = read_json(folder / 'scores' / f'{prefix}.json', [])
    length = int(row['length'])
    events = row.get('reward_events', []) if is_demo else (
        [{'step': length, 'reward': 1}] if row['success'] else [])
    label = (f'Demonstration {ident}' if is_demo else
             f'π₀.₅ {"success" if row["success"] else "failure"} · state {ident}')
    video = f'{BASE}videos/{task_id:02d}/{kind}/{prefix}.mp4'
    chart = score_plot(score, length, events) if score else '<p>Robometer scoring pending.</p>'
    endpoint = (f'Final progress {score[-1]["progress"]:.1%} · '
                f'predicted success {score[-1]["success_probability"]:.1%}'
                if score else 'Robometer pending')
    reward_step = int(events[0]['step']) if events else -1
    series = html.escape(json.dumps(score, separators=(',', ':')), quote=True)
    return (f'<article class="episode" data-series="{series}" data-length="{length}" '
            f'data-reward-step="{reward_step}"><div class="episode-head"><h3>{html.escape(label)}</h3>'
            f'<span>{length} steps</span></div><video controls preload="metadata" playsinline '
            f'src="{video}"></video><div class="plot-pane"><div class="legend">'
            '<span class="progress">Robometer progress / reward</span><span class="success">Predicted success</span>'
            '<span class="ground-truth">Simulator reward</span></div>'
            f'{chart}<div class="live-values">{html.escape(endpoint)}</div></div></article>')


def task_section(data_root: Path, task_id: int) -> str:
    root = data_root / ROOT.name / f'task{task_id:02d}'
    status = read_json(root / 'status.json', {})
    cfg = read_json(root / 'config.json', {})
    demos = read_json(root / 'demos/manifest.json', {}).get('demonstrations', [])
    records = read_json(root / 'policy/episodes.json', [])
    selection = read_json(root / 'policy/selection.json', {})
    by_state = {int(row['init_state_id']): row for row in records}
    selected = [by_state[state] for state in selection.get('selected_state_ids', []) if state in by_state]
    success_count = sum(bool(row['success']) for row in records)
    lo, hi = wilson(success_count, len(records))
    ci = f' · 95% Wilson {lo:.0%}–{hi:.0%}' if lo is not None else ''
    phase = html.escape(str(status.get('status', 'queued')))
    shortfall = ''
    if selection and (selection['available_successes'] < 2 or selection['available_failures'] < 2):
        shortfall = (' Fewer than two episodes occurred for one outcome in the 25 evaluations; '
                     'the remaining displayed videos come from the other outcome.')
    selected_note = (f'{len(selection.get("selected_successes", []))} successes and '
                     f'{len(selection.get("selected_failures", []))} failures randomly selected '
                     f'(seed {selection.get("selection_seed")}).{shortfall}'
                     if selection else 'Four display videos will be selected after all 25 evaluations finish.')
    demo_cards = ''.join(card(task_id, 'demos', row, root / 'demos') for row in demos)
    policy_cards = ''.join(card(task_id, 'policy', row, root / 'policy') for row in selected)
    terminal_rows = []
    for kind, rows, key, prefix in (('Demonstration', demos, 'demo_id', 'demo'),
                                   ('π₀.₅ evaluation', selected, 'init_state_id', 'episode')):
        for row in rows:
            ident = int(row[key])
            folder = 'demos' if kind == 'Demonstration' else 'policy'
            series = read_json(root / folder / 'scores' / f'{prefix}_{ident}.json', [])
            terminal_rows.append(
                f'<tr><td>{kind} {ident}</td><td>{"Success" if row["success"] else "Failure"}</td>'
                f'<td>{series[-1]["progress"]:.1%}</td>'
                f'<td>{series[-1]["success_probability"]:.1%}</td></tr>'
                if series else
                f'<tr><td>{kind} {ident}</td><td>{"Success" if row["success"] else "Failure"}</td>'
                '<td>Pending</td><td>Pending</td></tr>'
            )
    all_rows = ''.join(
        f'<tr><td>{state}</td><td>{"Success" if row["success"] else "Failure"}</td>'
        f'<td>{row["length"]}</td><td><a href="{BASE}videos/{task_id:02d}/policy/episode_{state}.mp4">'
        'Video</a></td><td>'
        + ('Displayed and scored' if state in selection.get('selected_state_ids', []) else 'Recorded')
        + '</td></tr>'
        for state, row in sorted(by_state.items())
    )
    return (f'<section class="task" id="task-{task_id:02d}"><div class="eyebrow">LIBERO-90 task {task_id:02d}</div>'
            f'<h2>{html.escape(task_name(task_id).replace("_", " "))}</h2>'
            f'<p class="meta">Phase: <b>{phase}</b> · {len(records)}/25 π₀.₅ evaluations · '
            f'{success_count} successes{ci} · {len(demos)}/4 demos · '
            f'{status.get("videos_scored", 0)}/8 selected videos scored.</p>'
            f'<p class="meta">Frozen policy checkpoint: <code>{html.escape(cfg.get("pi0_checkpoint", "pending"))}</code> '
            f'(<code>pi05_libero</code>). Robometer checkpoint: '
            f'<code>{html.escape(cfg.get("reward_checkpoint", "pending"))}</code>. '
            'Both tasks are in LIBERO-90, a suite held out from these checkpoints’ declared LIBERO fine-tuning mixes; '
            'broader backbone pretraining is not audited.</p>'
            '<h3>Four expert demonstrations</h3><div class="cards">'
            f'{demo_cards or "<p>Preparing demonstrations.</p>"}</div>'
            '<h3>Frozen π₀.₅ · selected evaluation episodes</h3>'
            f'<p class="meta">{html.escape(selected_note)}</p><div class="cards">'
            f'{policy_cards or "<p>Collecting 25 evaluation episodes.</p>"}</div>'
            '<details><summary>Terminal Robometer predictions for displayed videos</summary>'
            '<div class="table-wrap"><table><thead><tr><th>Video</th><th>Simulator outcome</th>'
            '<th>Progress / reward</th><th>Predicted success</th></tr></thead><tbody>'
            f'{"".join(terminal_rows)}</tbody></table></div></details>'
            '<details><summary>All 25 recorded π₀.₅ videos</summary>'
            '<div class="table-wrap"><table><thead><tr><th>State</th><th>Simulator outcome</th>'
            '<th>Steps</th><th>Video</th><th>Audit selection</th></tr></thead><tbody>'
            f'{all_rows}</tbody></table></div></details></section>')


def render(data_root: Path) -> bytes:
    sections = ''.join(task_section(data_root, task_id) for task_id in TASK_IDS)
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="60">
<title>LIBERO-90 π₀.₅ video audit</title><style>
:root{{--ink:#17313b;--muted:#61757a;--line:#dce5e6;--bg:#f3f7f7}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.5 system-ui,sans-serif}}
main{{max-width:1450px;margin:auto;padding:28px 20px 70px}}a{{color:#087f8c}}
h1{{font-size:clamp(28px,4vw,42px);margin:6px 0}}h2{{font-size:25px;margin:5px 0}}h3{{font-size:17px;margin:18px 0 8px}}
p{{margin:9px 0}}.meta{{color:var(--muted);font-size:14px}}.eyebrow{{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#577176;font-weight:700}}
.task{{background:white;border:1px solid var(--line);border-radius:16px;padding:24px;margin:25px 0}}
.cards{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:12px}}
.episode{{background:#fff;border:1px solid var(--line);border-radius:12px;padding:12px;display:grid;grid-template-columns:minmax(150px,35%) minmax(0,1fr);gap:8px 12px;align-items:start}}
.episode-head{{grid-column:1/-1;display:flex;justify-content:space-between;gap:10px;align-items:baseline}}
.episode-head h3{{margin:0;font-size:16px}}.episode-head span{{font-size:12px;color:var(--muted)}}
video{{display:block;width:100%;max-width:235px;aspect-ratio:1;background:#102024;border-radius:7px}}
.plot-pane{{min-width:0}}.score-plot{{width:100%;height:auto}}.grid{{stroke:#e5eded}}.tick,.axis{{fill:#718187;font-size:12px}}
.legend{{display:flex;gap:5px 10px;flex-wrap:wrap;font-size:11px}}.legend span:before{{content:"";display:inline-block;width:14px;height:3px;vertical-align:middle;margin-right:5px}}
.progress:before{{background:#087f8c}}.success:before{{background:#d36a35}}.ground-truth:before{{background:#7754a7}}
.live-values{{font-size:12px;color:var(--muted)}}details{{margin-top:20px}}summary{{cursor:pointer;font-weight:650}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px;margin:12px 0}}th,td{{padding:7px 9px;border-bottom:1px solid var(--line);text-align:left}}th{{color:var(--muted)}}
@media(max-width:1000px){{.cards{{grid-template-columns:1fr}}}}@media(max-width:520px){{.episode{{grid-template-columns:1fr}}}}
</style></head><body><main><a href="/">← Main experiment dashboard</a>
<div class="eyebrow">Frozen evaluation · video and reward audit</div><h1>π₀.₅ on LIBERO-90 tasks 02 and 28</h1>
<p>Each task has four expert demonstrations and 25 recorded π₀.₅ episodes on official reset states 25–49, with a 600-step horizon. After all 25 episodes finish, two successes and two failures are randomly chosen when available. The displayed videos show Robometer progress and success predictions alongside simulator reward.</p>
<p class="meta">The 25 videos are linked in each task’s table; Robometer is scored on the four selected policy videos and four demonstrations. Its progress prediction is the learned reward signal shown here; its success prediction is a separate head. Videos sample every four physical steps, and Robometer is queried every 20 steps and at the endpoint. Scrub a video to inspect predictions. Demonstrations come from the pinned <a href="https://huggingface.co/datasets/yifengzhu-hf/LIBERO-datasets/tree/f13aa24a3da8c43c7225569f28c562979fa0e35a/libero_90">LIBERO-90 HDF5 dataset</a>. This page refreshes every 60 seconds.</p>
{sections}</main><script>
for(const card of document.querySelectorAll('.episode')){{
 const series=JSON.parse(card.dataset.series||'[]');if(!series.length)continue;
 const video=card.querySelector('video'),cursor=card.querySelector('.playhead'),label=card.querySelector('.live-values');
 const length=Number(card.dataset.length),rewardStep=Number(card.dataset.rewardStep);
 function update(){{const elapsed=video.currentTime,step=Math.min(length,elapsed*40);
  const x=38+(620-38-12)*step/Math.max(length,1);cursor.setAttribute('x1',x.toFixed(1));cursor.setAttribute('x2',x.toFixed(1));
  let nearest=series[0];for(const item of series){{if(Math.abs(item.video_time-elapsed)<Math.abs(nearest.video_time-elapsed))nearest=item;}}
  const reward=rewardStep>=0&&step>=rewardStep?1:0;
  label.textContent=`Step ${{nearest.step}} · Progress ${{(nearest.progress*100).toFixed(1)}}% · Predicted success ${{(nearest.success_probability*100).toFixed(1)}}% · Simulator reward ${{reward}}`;
 }}video.addEventListener('timeupdate',update);video.addEventListener('seeked',update);update();
}}
</script></body></html>'''
    return page.encode('utf-8')
