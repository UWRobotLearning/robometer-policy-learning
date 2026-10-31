# Historical LIBERO dashboard and video pages

Training and curve/statistics analysis now use the task-independent
[`experiments.libero`](../libero/README.md) package. Configure tasks, rewards,
seeds, and `scratch` / `pi0` / `pi0.5` in one JSON file. The task-specific training
launchers have been removed.

This directory retains the existing dashboard command. Frozen-evaluation and
historical video page adapters live in `experiments.libero.legacy` so saved
results and URLs keep working:

```bash
.venv-policy/bin/python -m experiments.pi0_stove.dashboard \
  --data-root /data/rohan/Reward-Modeling --port 8768
```

The dashboard reads logs on each refresh and discovers both historical task
folders and the new config-driven output layout. No result files or checkpoints
were removed by the refactor. Shared OpenPI inference now lives in
`experiments.libero.policy` and `experiments.libero.policy_worker`.
