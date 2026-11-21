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

The `/reward-model-methods/` subpage explains frozen reward/value heuristics,
supervised RM adaptation, Bellman and inverse-RL objectives, optional feature
distillation, and BC-initialized flow-policy updates. Its standalone source is
[`reward_model_methods.html`](reward_model_methods.html); equations use native
MathML with their editable LaTeX retained in each `annotation` element. The page
has no CDN dependencies. Restart an already-running dashboard to load new routes
or navigation changes; the guide's HTML is read from disk on every request.
