"""Live read-only report of all 130 π0.5 LIBERO evaluations."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.libero.legacy.pi05_libero_eval import ROOT, SUITES, task_dir, completed
from experiments.reward_audit.config import task_map
from experiments.reward_audit.metrics import wilson

LABELS = {'libero_spatial': 'Spatial', 'libero_object': 'Object',
          'libero_goal': 'Goal', 'libero_10': 'LIBERO-10', 'libero_90': 'LIBERO-90'}


def task_result(folder):
    records = []
    file = folder / 'episodes.jsonl'
    if file.exists():
        for line in file.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except ValueError:
                pass
    if completed(folder):
        state = 'complete'
    else:
        try:
            state = json.loads((folder / 'status.json').read_text()).get('status', 'queued')
        except (OSError, ValueError):
            state = 'queued'
    return state, sum(bool(record.get('success')) for record in records), len(records)


def render(data_root: Path) -> bytes:
    root = data_root / ROOT.name
    names = task_map()
    sections = []
    nav = []
    total_complete = total_episodes = total_successes = 0
    for suite in SUITES:
        rows = []
        complete = episodes = successes = 0
        ind = suite != 'libero_90'
        for index, name in enumerate(names[suite]):
            state, wins, count = task_result(task_dir(root, suite, index, name))
            if state == 'complete':
                complete += 1
            successes += wins
            episodes += count
            lo, hi = wilson(wins, count)
            interval = f'{lo:.0%}–{hi:.0%}' if lo is not None else '—'
            rate = f'{wins/count:.0%}' if count else '—'
            progress = f'{wins}/{count}' if count else '—'
            rows.append(f'<tr><td>{index:02d}</td><td>{html.escape(name.replace("_", " "))}</td>'
                        f'<td>{html.escape(state)}</td><td>{progress}</td>'
                        f'<td>{rate}</td><td>{interval}</td></tr>')
        total_complete += complete
        total_episodes += episodes
        total_successes += successes
        lo, hi = wilson(successes, episodes)
        rate = f'{successes/episodes:.1%}' if episodes else '—'
        ci = f'{lo:.1%}–{hi:.1%}' if lo is not None else '—'
        label = LABELS[suite]
        nav.append(f'<a href="#{suite}">{label} {complete}/{len(names[suite])}</a>')
        sections.append(f'<section id="{suite}"><h2>{label}</h2>'
                        f'<p><b>{"In distribution" if ind else "Out of distribution"}</b> for this fine-tuned checkpoint '
                        f'· {complete}/{len(names[suite])} tasks complete · {successes}/{episodes} episode successes '
                        f'({rate}; 95% Wilson {ci}).</p>'
                        '<div class="scroll"><table><thead><tr><th>ID</th><th>Task</th><th>Status</th>'
                        '<th>Successes</th><th>Rate</th><th>95% Wilson CI</th></tr></thead><tbody>'
                        + ''.join(rows) + '</tbody></table></div></section>')
    timestamp = datetime.now(timezone.utc).isoformat(timespec='seconds')
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="refresh" content="60"><title>π0.5 LIBERO evaluation</title><style>
body{{font:16px/1.5 system-ui,sans-serif;background:#f3f7f7;color:#17313b;margin:0}}main{{max-width:1100px;margin:auto;padding:28px}}
a{{color:#087f8c}}nav{{display:flex;gap:14px;flex-wrap:wrap}}nav a{{background:white;border:1px solid #dce5e6;border-radius:8px;padding:7px 12px;text-decoration:none}}
section{{background:white;border:1px solid #dce5e6;border-radius:15px;padding:22px;margin:22px 0}}h1{{margin-bottom:5px}}.muted{{color:#5d6e73}}.scroll{{overflow:auto}}table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:7px 9px;text-align:left;border-bottom:1px solid #e5eeee}}th{{white-space:nowrap}}
</style></head><body><main><p><a href="/">← Main results page</a></p><h1>π0.5 on all LIBERO tasks</h1>
<p>The <a href="https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md">official <code>pi05_libero</code> checkpoint</a> is evaluated on 25 official initial states per task (IDs 25–49). Policy is frozen. The checkpoint was fine-tuned on <a href="https://huggingface.co/datasets/physical-intelligence/libero/blob/main/README.md">a dataset containing Spatial, Object, Goal and LIBERO-10</a> (40 tasks); LIBERO-90 is outside that fine-tuning set. This is family-level distribution status; exact demonstration overlap is not claimed.</p>
<p>Each episode ends on simulator success or at 400 steps, except LIBERO-10 at 600 steps as requested. One 10-action chunk is executed per inference. Success is the simulator's sparse success check. Intervals are per-task Wilson binomial 95% intervals and reflect only these initial-state episodes.</p>
<p><b>Queue status:</b> “Queued” means no rollout has started for that task yet. Tasks run sequentially on each available GPU. GPU 1 was occupied by another user's run, so its pending tasks have been reassigned to GPU 0 after the current lane finishes. Completed tasks are skipped when the resumed sweep starts.</p>
<p class="muted">{total_complete}/130 tasks complete · {total_successes}/{total_episodes} successes observed · updated {timestamp} UTC · auto-refreshes every 60 seconds</p>
<nav>{''.join(nav)}</nav>{''.join(sections)}</main></body></html>'''
    return page.encode('utf-8')
