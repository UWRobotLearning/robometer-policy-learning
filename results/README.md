# LIBERO numerical results

[2026-10-31 snapshot](libero_20261031/index.json) · [Run catalog](libero_20261031/runs.json) · [Evaluation points](libero_20261031/evaluation_points.csv) · [Frozen/evaluation summaries](libero_20261031/frozen_evaluations.json)

This directory contains plot inputs, not trained models or media. It includes historical,
stopped, incomplete, and smoke runs as well as completed experiments. A directory's presence
does not imply completion or inclusion in a published comparison; consult its status and
experiment manifest. Do not pool different experiments just because their task names match.

## Contents

- `evaluation_points.csv`: portable SAC/DSRL evaluation table, with run identity, reward,
  seed, task, environment count, total physical steps, physical steps per environment,
  success count, episode count, and success rate. Aliases to the same underlying run are
  deduplicated. Some old configs lack task IDs; their task names and source paths remain.
- `runs.json`: run catalog and relative paths to the saved configurations under `data/`.
- `frozen_evaluations.json`: success-rate summaries and their configurations, including
  frozen-policy and video-audit evaluations. Check the configuration before choosing a
  pretrained baseline; these are not all interchangeable baselines.
- `data/`: original directory structure, containing allowlisted JSON, JSONL, and CSV files.
  Configs/manifests document checkpoints, rewards, seeds, horizons, and hyperparameters.
  Evaluation CSVs support success curves; episode outcomes support paired-state tests and
  training-return curves; metrics/diagnostics support optimizer curves. OGPO's `eval.csv`,
  `eval_sde.csv`, and training logs retain their original schemas and units and are not
  merged into the SAC/DSRL table.
- `index.json`: export timestamp, file sizes, SHA-256 hashes, source aliases, and any warnings.

No checkpoint weights, replay contents, images, videos, generated HTML/PDF plots, raw console
logs, or decoder-debug traces are exported. Checkpoint paths/names in JSON are provenance,
not copies of those checkpoints. Absolute machine paths are retained as historical metadata;
use the relative catalog paths when reading this snapshot elsewhere.

## Local data and refreshing the export

On the experiment machine, the ignored repository symlink `local_results` points to
`/data/rohan/Reward-Modeling`. It provides access to the full live data, while this directory
contains real files that work after cloning from GitHub. A Git symlink alone would not upload
the referenced data. To recreate the local link on that machine:

```sh
ln -s /data/rohan/Reward-Modeling local_results
```

Export a new snapshot with Python's standard library; no GPU or training dependencies:

```sh
python3 scripts/export_results.py --source local_results --out results/libero_YYYYMMDD
```

The exporter refuses to overwrite an existing snapshot. It materializes source symlinks
within the data root, excludes outside targets and cycles, and skips incomplete trailing
log records. Files are read individually: active experiments can advance during export,
so consult per-file endpoints rather than assuming an atomic global checkpoint. Original
data is never changed. Check `index.json` for warnings before publishing a new export.

## Plotting and statistical conventions

For SAC/DSRL, `physical_steps_per_env = physical_steps_total / num_envs`. A replay transition
is an action chunk, not one physical step. Four π₀ environments executing 20 actions each
advance this axis by about 20, while collecting 80 physical steps and four replay entries.
Early termination can shorten a chunk. Evaluation rollouts are not training steps.

Select one experiment and its intended seeds/reward variants. Align evaluations to that
experiment's scheduled evaluation bins (chunk execution can overshoot the exact boundary),
then compute the mean across independent training seeds. Use a 95% Student-t interval across
seeds when there are at least two; do not treat episodes as independent training seeds.
Retain the number of seeds at each point, since stopped runs disappear later in training.
For reward comparisons, match seed IDs and use the recorded episode/reset-state outcomes
where required. Frozen-baseline intervals depend on the original protocol (Wilson for one
episode sample, or the documented seed/state-based interval for pooled baselines).

For a quick inspection of individual seed curves, install Matplotlib in an isolated uv
environment and run this from the repository root. This draws individual runs, not a mean
or confidence interval:

```sh
uv venv /tmp/libero-results-venv
uv pip install --python /tmp/libero-results-venv/bin/python matplotlib
```

```python
import csv
from collections import defaultdict
import matplotlib.pyplot as plt

prefix = "experiments/libero10_task08_pi0_dsrl_batched_4env_200k_20261027/"
groups = defaultdict(list)
with open("results/libero_20261031/evaluation_points.csv") as stream:
    for row in csv.DictReader(stream):
        if row["run"].startswith(prefix):
            groups[(row["reward"], row["seed"])].append(row)
for (reward, seed), rows in groups.items():
    rows.sort(key=lambda row: float(row["physical_steps_per_env"]))
    plt.plot([float(r["physical_steps_per_env"]) for r in rows],
             [float(r["success_rate"]) for r in rows], label=f"{reward}, seed {seed}")
plt.xlabel("Physical training steps per environment")
plt.ylabel("Simulator success rate")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("/tmp/libero-task08-curves.png")
```
