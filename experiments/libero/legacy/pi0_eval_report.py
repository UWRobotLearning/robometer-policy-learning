"""Read-only sub-page for the frozen pi0 LIBERO-90 evaluation sweep."""

from __future__ import annotations

import html
import json
from pathlib import Path


SWEEP = "pi0_task_probe_20261010"
REWARD_CONFIG = ("pi0_libero90_task38_dsrl_3seeds_20261012/seed100/"
                 "robometer_seed100/config.json")


def render(data_root: Path) -> bytes:
    sweep = data_root / SWEEP
    tasks = []
    policy_checkpoints = set()
    for summary_path in sweep.glob("[0-9][0-9]_*/summary.json"):
        folder = summary_path.parent
        config_path = folder / "config.json"
        status_path = folder / "status.json"
        if not config_path.exists() or not status_path.exists():
            continue
        config = json.loads(config_path.read_text())
        status = json.loads(status_path.read_text())
        if config.get("suite") != "libero_90" or status.get("status") != "complete":
            continue
        summary = json.loads(summary_path.read_text())
        task_id = int(folder.name[:2])
        policy_checkpoints.add(config.get("checkpoint", ""))
        tasks.append({"id": task_id, "name": config["task"].replace("_", " "),
                      "successes": int(summary["successes"]),
                      "episodes": int(summary["episodes"]),
                      "rate": float(summary["success_rate"]),
                      "ci": summary.get("wilson_ci95"),
                      "states": tuple(config.get("eval_state_ids", []))})
    tasks.sort(key=lambda task: task["id"])

    reward_config_path = data_root / REWARD_CONFIG
    reward_config = json.loads(reward_config_path.read_text()) if reward_config_path.exists() else {}
    policy_known = (len(policy_checkpoints) == 1 and
                    next(iter(policy_checkpoints)).rstrip("/").endswith("/pi0_libero"))
    reward_known = reward_config.get("reward_checkpoint", "").startswith("robometer/Robometer-4B@")
    policy_label = "No · LIBERO-90 held out" if policy_known else "Unverified checkpoint"
    reward_label = "No · LIBERO-90 held out" if reward_known else "Unverified checkpoint"

    rows = []
    for task in tasks:
        interval = (f'{task["ci"][0]:.0%}–{task["ci"][1]:.0%}'
                    if task["ci"] and len(task["ci"]) == 2 else "—")
        name = html.escape(task["name"])
        rows.append(
            f'<tr data-search="{task["id"]} {name.lower()}"><td>{task["id"]:02d}</td>'
            f'<td>{name}</td><td><span class="rate" style="--rate:{task["rate"]:.0%}">'
            f'{task["rate"]:.0%}</span> <small>{task["successes"]}/{task["episodes"]}</small></td>'
            f'<td>{interval}</td><td>{html.escape(policy_label)}</td>'
            f'<td>{html.escape(reward_label)}</td></tr>')

    total_successes = sum(task["successes"] for task in tasks)
    total_episodes = sum(task["episodes"] for task in tasks)
    overall = total_successes / total_episodes if total_episodes else 0
    zero_tasks = sum(task["successes"] == 0 for task in tasks)
    half_tasks = sum(task["rate"] >= .5 for task in tasks)
    same_states = len({task["states"] for task in tasks}) == 1
    state_note = ("Initial states 25–49 for every task" if same_states and tasks and
                  tasks[0]["states"] == tuple(range(25, 50)) else
                  "See each run’s config for evaluated initial states")

    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Frozen π₀ · LIBERO-90 evaluation</title>
<style>
:root{{--ink:#17313b;--muted:#5d6e73;--line:#dce5e6;--bg:#f3f7f7;--teal:#087f8c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1250px;margin:auto;padding:34px 20px 80px}}a{{color:var(--teal)}}
h1{{font-size:clamp(28px,4vw,42px);line-height:1.15;margin:12px 0}}h2{{font-size:20px;margin:0 0 8px}}
.eyebrow{{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#577176;font-weight:700}}
.meta{{color:var(--muted)}}.cards{{display:flex;gap:12px;flex-wrap:wrap;margin:24px 0}}
.stat,.panel{{background:white;border:1px solid var(--line);border-radius:13px;padding:18px}}
.stat{{min-width:175px;flex:1}}.stat b{{display:block;font-size:25px}}.stat span{{color:var(--muted);font-size:13px}}
.panel{{margin:18px 0}}.panel p{{margin:7px 0}}.sources{{font-size:13px;color:var(--muted)}}
input{{width:min(100%,360px);padding:10px 12px;border:1px solid #bacbce;border-radius:8px;font:inherit;margin:9px 0 17px}}
.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:1050px;font-size:13px}}
th,td{{text-align:left;padding:10px 9px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{position:sticky;top:0;background:white;color:#536c72}}td:nth-child(2){{min-width:380px}}
.rate{{display:inline-block;min-width:78px;padding:2px 6px;border-radius:4px;background:linear-gradient(90deg,#b9e6df var(--rate),#edf4f4 var(--rate))}}
small{{color:var(--muted)}}.note{{font-size:13px;color:var(--muted)}}
</style></head><body><main>
<a href="/">← Main fine-tuning dashboard</a><div class="eyebrow">Frozen policy · no RL updates</div>
<h1>π₀ evaluation across LIBERO-90</h1>
<p class="meta">Released <code>pi0_libero</code> checkpoint · seed 100 · 25 simulator episodes per task · {html.escape(state_note)}. Success uses the LIBERO simulator condition.</p>
<div class="cards"><div class="stat"><b>{len(tasks)}</b><span>completed tasks</span></div>
<div class="stat"><b>{overall:.1%}</b><span>pooled success · {total_successes}/{total_episodes} episodes</span></div>
<div class="stat"><b>{zero_tasks}</b><span>tasks with zero successes</span></div>
<div class="stat"><b>{half_tasks}</b><span>tasks at or above 50% success</span></div></div>
<section class="panel"><h2>Training-data coverage</h2>
<p><b>π₀ policy:</b> {html.escape(policy_label)}. The released <code>pi0_libero</code> checkpoint uses the Physical Intelligence LIBERO dataset, which contains Spatial, Object, Goal, and 10.</p>
<p><b>Robometer reward:</b> {html.escape(reward_label)} for the general <code>Robometer-4B</code> checkpoint used in the later reward-learning runs. Its declared RBM-1M training mix uses LIBERO 10, Object, Spatial, and Goal; LIBERO-90 is an evaluation suite.</p>
<p class="note">These are suite-level coverage labels for the configured checkpoints. Similar instructions or objects may occur in other suites, and the VLM backbones’ broad pretraining data is not audited here. This page evaluates π₀ policy success only; it does not measure Robometer on these 90 tasks.</p>
<p class="sources">Sources: <a href="https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py">π₀ training configuration</a>, <a href="https://huggingface.co/datasets/physical-intelligence/libero/blob/main/README.md">policy dataset card</a>, <a href="https://arxiv.org/html/2603.02115">Robometer paper</a>, <a href="https://github.com/robometer/robometer/blob/main/robometer/robometer/data/dataset_category.py">Robometer dataset split</a>.</p>
</section>
<section class="panel"><h2>Per-task simulator success</h2>
<p class="note">95% Wilson intervals describe uncertainty over the 25 evaluation episodes for each task. They do not account for policy training-seed variation.</p>
<label for="task-search">Find a task by ID or name</label><br><input id="task-search" type="search" placeholder="e.g. stove, drawer, 55">
<div class="table-wrap"><table><thead><tr><th>ID</th><th>LIBERO-90 task</th><th>π₀ success</th><th>95% interval</th><th>In π₀ task training?</th><th>In Robometer task training?</th></tr></thead>
<tbody id="task-rows">{"".join(rows)}</tbody></table></div></section>
<p class="note">Local result files: <code>{html.escape(str(sweep))}</code>. Values are read from each completed task’s <code>summary.json</code> when this page loads.</p>
</main><script>
const search=document.getElementById('task-search');
search.addEventListener('input',()=>{{const q=search.value.trim().toLowerCase();
for(const row of document.querySelectorAll('#task-rows tr'))row.hidden=!row.dataset.search.includes(q);}});
</script></body></html>'''
    return page.encode("utf-8")
