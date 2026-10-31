"""Live local report for frozen pi0 on LIBERO Spatial, Object, Goal, and 10."""

from __future__ import annotations

import html
import json
from datetime import datetime, timezone
from pathlib import Path

from experiments.libero.legacy.pi0_family_eval import EVAL_STATES, ROOT, SUITES, completed, task_dir
from experiments.reward_audit.config import task_map
from experiments.reward_audit.metrics import wilson


LABELS = {"libero_spatial": "Spatial", "libero_object": "Object",
          "libero_goal": "Goal", "libero_10": "LIBERO-10"}
RBM_DATASETS = {
    "libero_spatial": "abraranwar_libero_rfm_libero256_spatial",
    "libero_object": "abraranwar_libero_rfm_libero256_object",
    "libero_goal": "abraranwar_libero_rfm_libero256_goal",
    "libero_10": "abraranwar_libero_rfm_libero256_10",
}
REWARD_PROVENANCE = ("pi0_libero90_task38_dsrl_3seeds_20261012/seed100/"
                     "robometer_seed100/reward_provenance.json")


def training_coverage(data_root: Path) -> dict:
    """Use the configured local policy and pinned reward provenance, not model names alone."""
    manifest_path = data_root / ROOT.name / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    policy_checkpoint = manifest.get("checkpoint", "")
    policy_known = (policy_checkpoint.rstrip("/").endswith("/pi0_libero") and
                    (Path(policy_checkpoint) / "assets/physical-intelligence/libero/norm_stats.json").exists())
    reward_path = data_root / REWARD_PROVENANCE
    reward = json.loads(reward_path.read_text()) if reward_path.exists() else {}
    reward_checkpoint = reward.get("checkpoint", "")
    reward_known = (reward_checkpoint.startswith("robometer/Robometer-4B@") and
                    "rbm-1m-id" in reward.get("training_datasets", []))
    expanded = set(reward.get("expanded_training_datasets", []))
    return {"policy_checkpoint": policy_checkpoint, "policy_known": policy_known,
            "reward_checkpoint": reward_checkpoint, "reward_known": reward_known,
            "reward_datasets": expanded}


def task_result(folder: Path) -> dict:
    if completed(folder):
        summary = json.loads((folder / "summary.json").read_text())
        return {"status": "complete", "successes": int(summary["successes"]),
                "episodes": int(summary["episodes"]), "ci": summary.get("wilson_ci95")}
    status_path = folder / "status.json"
    state = json.loads(status_path.read_text()) if status_path.exists() else {}
    episodes_path = folder / "episodes.jsonl"
    records = []
    if episodes_path.exists():
        for line in episodes_path.read_text().splitlines():
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    condition = state.get("status", "queued")
    return {"status": condition, "successes": sum(bool(row.get("success")) for row in records),
            "episodes": len(records), "ci": None}


def render(data_root: Path) -> bytes:
    root = data_root / ROOT.name
    names = task_map()
    coverage = training_coverage(data_root)
    coverage_rows = []
    sections = []
    navigation = []
    all_completed = 0
    all_successes = 0
    all_episodes = 0
    for suite in SUITES:
        rows = []
        finished = 0
        successes = 0
        episodes = 0
        active = 0
        policy_label = "Yes · suite" if coverage["policy_known"] else "Unverified"
        reward_label = ("Yes · suite" if coverage["reward_known"] and
                        RBM_DATASETS[suite] in coverage["reward_datasets"] else "Unverified")
        for index, name in enumerate(names[suite]):
            result = task_result(task_dir(root, suite, index, name))
            state = result["status"]
            if state == "complete":
                finished += 1
                successes += result["successes"]
                episodes += result["episodes"]
            if state == "running":
                active += 1
            rate = f'{result["successes"] / result["episodes"]:.0%}' if result["episodes"] else "—"
            interval = (f'{result["ci"][0]:.0%}–{result["ci"][1]:.0%}'
                        if result["ci"] and len(result["ci"]) == 2 else "—")
            progress = f'{result["successes"]}/{result["episodes"]}' if result["episodes"] else "—"
            rows.append(f'<tr data-search="{index:02d} {html.escape(name.lower())}">'
                        f'<td>{index:02d}</td><td>{html.escape(name.replace("_", " "))}'
                        + (f' <a href="/pi0-libero10-video-audit/#task-{index:02d}">Video audit ↗</a>'
                           if suite == "libero_10" and index in (6, 8) else "") + '</td>'
                        f'<td><span class="state {html.escape(state)}">{html.escape(state)}</span></td>'
                        f'<td>{rate} <small>{progress}</small></td><td>{interval}</td>'
                        f'<td>{policy_label}</td><td>{reward_label}</td></tr>')
        all_completed += finished
        all_successes += successes
        all_episodes += episodes
        lo, hi = wilson(successes, episodes)
        pooled = f'{successes / episodes:.1%}' if episodes else "—"
        pooled_ci = f'{lo:.1%}–{hi:.1%}' if lo is not None else "—"
        label = LABELS[suite]
        coverage_rows.append(f'<tr><td>{label}</td><td><b>{policy_label}</b></td>'
                             f'<td><b>{reward_label}</b></td>'
                             f'<td><code>{RBM_DATASETS[suite]}</code></td></tr>')
        navigation.append(f'<a href="#{suite}">{label} · {finished}/10</a>')
        sections.append(f'<section class="panel" id="{suite}"><h2>{label}</h2>'
                        f'<p class="note">Training-family coverage: π₀ {policy_label}; '
                        f'Robometer {reward_label}.</p>'
                        f'<p class="meta">{finished}/10 tasks complete · {active} running · '
                        f'pooled completed-task success {pooled} ({successes}/{episodes}; 95% Wilson {pooled_ci})</p>'
                        '<div class="table-wrap"><table><thead><tr><th>ID</th><th>Task</th>'
                        '<th>Status</th><th>π₀ success</th><th>95% interval</th>'
                        '<th>In π₀ training?</th><th>In Robometer training?</th></tr></thead>'
                        f'<tbody>{"".join(rows)}</tbody></table></div></section>')
    overall = f'{all_successes / all_episodes:.1%}' if all_episodes else "—"
    focus_rows = []
    for task_id in (6, 8):
        task_name = names["libero_10"][task_id].replace("_", " ")
        result = task_result(task_dir(root, "libero_10", task_id,
                                      names["libero_10"][task_id]))
        rate = (f'{result["successes"]}/{result["episodes"]} '
                f'({result["successes"] / result["episodes"]:.0%})'
                if result["episodes"] else "Pending")
        policy = "Yes — LIBERO-10 suite" if coverage["policy_known"] else "Unverified"
        reward = ("Yes — LIBERO-10 suite" if coverage["reward_known"] and
                  RBM_DATASETS["libero_10"] in coverage["reward_datasets"] else "Unverified")
        focus_rows.append(f'<tr><td>{task_id:02d}</td><td>{html.escape(task_name)}</td>'
                          f'<td>{rate}</td><td>{policy}</td><td>{reward}</td>'
                          f'<td><a href="/pi0-libero10-video-audit/#task-{task_id:02d}">Videos and scores</a></td></tr>')
    state_path = root / "orchestrator.json"
    state = json.loads(state_path.read_text()) if state_path.exists() else {"status": "queued"}
    updated = datetime.now(timezone.utc).isoformat(timespec="seconds")
    page = f'''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><meta http-equiv="refresh" content="60">
<title>Frozen π₀ · LIBERO family evaluation</title>
<style>
:root{{--ink:#17313b;--muted:#5d6e73;--line:#dce5e6;--bg:#f3f7f7;--teal:#087f8c}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 system-ui,-apple-system,sans-serif}}
main{{max-width:1200px;margin:auto;padding:34px 20px 80px}}a{{color:var(--teal)}}
h1{{font-size:clamp(28px,4vw,42px);line-height:1.15;margin:12px 0}}h2{{font-size:22px;margin:0 0 8px}}
.eyebrow{{font-size:12px;letter-spacing:.1em;text-transform:uppercase;color:#577176;font-weight:700}}
.meta,small{{color:var(--muted)}}.cards,.nav{{display:flex;gap:12px;flex-wrap:wrap;margin:18px 0}}
.stat,.panel{{background:white;border:1px solid var(--line);border-radius:13px;padding:18px}}
.stat{{min-width:175px;flex:1}}.stat b{{display:block;font-size:25px}}.stat span{{color:var(--muted);font-size:13px}}
.nav a{{padding:8px 12px;background:white;border:1px solid var(--line);border-radius:8px;text-decoration:none}}
.panel{{margin:20px 0}}.table-wrap{{overflow:auto}}table{{border-collapse:collapse;width:100%;min-width:800px;font-size:14px}}
th,td{{text-align:left;padding:10px 9px;border-bottom:1px solid var(--line)}}th{{background:white;color:#536c72}}
td:nth-child(2){{min-width:360px}}small{{padding-left:5px}}.state{{border-radius:5px;padding:2px 6px;background:#edf2f2}}
.coverage-table td:nth-child(2){{min-width:0}}.coverage-table code{{font-size:12px;overflow-wrap:anywhere}}
.state.complete{{background:#dcf3e7}}.state.running{{background:#fff0cb}}.state.failed{{background:#ffe1dc}}
.note{{font-size:13px;color:var(--muted)}}
</style></head><body><main>
<a href="/">← Main fine-tuning dashboard</a> · <a href="/pi0-libero90-eval/">LIBERO-90 baseline</a>
<div class="eyebrow">Frozen policy · no RL updates</div><h1>π₀ evaluation across LIBERO task families</h1>
<p class="meta">Released <code>pi0_libero</code> checkpoint · seed 100 · {len(EVAL_STATES)} official initial states per task (25–49) · 400-step limit · simulator success.</p>
<div class="cards"><div class="stat"><b>{all_completed}/40</b><span>completed tasks</span></div>
<div class="stat"><b>{overall}</b><span>pooled completed-task success · {all_successes}/{all_episodes}</span></div>
<div class="stat"><b>{html.escape(state.get("status", "unknown"))}</b><span>sweep status</span></div></div>
<section class="panel"><h2>Training-set coverage for the exact checkpoints</h2>
<p><b>Evaluated π₀ policy:</b> <code>{html.escape(coverage["policy_checkpoint"] or "unverified")}</code> (<code>pi0_libero</code> config). Its configured training dataset is <code>physical-intelligence/libero</code>, which contains these four families.</p>
<p><b>Robometer reward model used in the reward-learning comparisons:</b> <code>{html.escape(coverage["reward_checkpoint"] or "unverified")}</code>. Its saved provenance expands <code>rbm-1m-id</code> into the LIBERO datasets shown below. Robometer is <b>not evaluated</b> on this page.</p>
<div class="table-wrap"><table class="coverage-table"><thead><tr><th>LIBERO family</th><th>In π₀ training?</th><th>In Robometer training?</th><th>Robometer training dataset in checkpoint provenance</th></tr></thead>
<tbody>{"".join(coverage_rows)}</tbody></table></div>
<p class="note">These are suite-level coverage labels from the configured training data and saved Robometer provenance. They do not establish that the evaluated initial states or exact demonstration trajectories were seen during training. Sources: <a href="https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py">OpenPI training config</a>, <a href="https://huggingface.co/datasets/physical-intelligence/libero/blob/main/README.md">π₀ dataset card</a>, <a href="https://github.com/robometer/robometer/blob/main/robometer/data/dataset_category.py">Robometer dataset categories</a>.</p></section>
<section class="panel"><h2>LIBERO-10 tasks 06 and 08</h2>
<p class="note">Both tasks belong to LIBERO-10, a suite included in both checkpoint training mixes. The available provenance does not list training episodes by task, demonstration, or reset state.</p>
<div class="table-wrap"><table><thead><tr><th>ID</th><th>Task</th><th>π₀ baseline</th><th>In π₀ training?</th><th>In Robometer training?</th><th>Audit</th></tr></thead><tbody>{"".join(focus_rows)}</tbody></table></div></section>
<p class="note">95% Wilson intervals describe episode-level uncertainty for completed tasks; they do not measure variation across policy training seeds. Partial running-task results are shown without intervals and excluded from pooled rates. New files are read on every request; this page refreshes every 60 seconds. Latest read: {updated} UTC.</p>
<nav class="nav">{"".join(navigation)}</nav>{"".join(sections)}
<p class="note">Local results: <code>{html.escape(str(root))}</code>.</p>
</main></body></html>'''
    return page.encode("utf-8")
