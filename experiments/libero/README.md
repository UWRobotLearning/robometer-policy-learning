# Reusable LIBERO experiments

Use one JSON specification and one CLI for every task. The training code does
not read settings from previous experiments. Existing result files are retained.

```bash
.venv-policy/bin/python -m experiments.libero plan experiments/libero/experiment.json
.venv-policy/bin/python -m experiments.libero run experiments/libero/experiment.json
```

`plan` validates and prints the fully resolved runs without importing ML
libraries, touching GPUs, or creating output directories. `run` performs the
comparison. This refactor does not launch any experiments by itself.

Edit [experiment.json](experiment.json) to select:

| Setting | Options |
| --- | --- |
| `base_model` | `scratch`, `pi0`, `pi0.5` |
| `rewards` | `["sparse"]`, `["robometer"]`, or both |
| `suite` | Any official LIBERO suite, including `libero_90` and `libero_10` |
| `tasks` | Zero-based task IDs or exact task names; tasks run in order |
| `seeds` | Any distinct nonnegative seeds; rewards share the seed plan |
| `steps`, `eval_every`, `eval_episodes`, `horizon` | Physical-transition budget and evaluation protocol |
| `checkpoint` | Optional OpenPI checkpoint directory; overrides the base model's default |
| `reward_checkpoint` | Pinned Robometer checkpoint; default is the original general Robometer-4B |
| `sac`, `dsrl` | Optional overrides of the model profile's hyperparameters |
| `runtime.gpus` | GPUs available for training, with one run per GPU |
| `runtime.reward_gpu` | Optional dedicated reward GPU outside `runtime.gpus`; otherwise colocated |

Changing tasks, seed count, rewards, or base model requires editing this file,
not adding Python files. Give a new experiment a different `name`. Several tasks
can share the same specification, e.g. `"tasks": [2, 28]`: all runs for task 2
finish before task 28 starts.

## Profiles and reward semantics

The profiles preserve the completed experiments' hyperparameters:

| Base | Training | Learning starts | Actor/critic updates | Action execution |
| --- | --- | --- | --- | --- |
| `scratch` | SAC MLP over frozen DINOv2 features | 5,000 physical transitions | One actor and critic update per transition | One robot action |
| `pi0` | SAC steering of frozen `pi0_libero` | 500 physical transitions | One actor and 25 critic updates per chunk | 20 actions |
| `pi0.5` | SAC steering of frozen `pi05_libero` | 500 physical transitions | One actor and 25 critic updates per chunk | 10 actions |

DSRL uses direct 32D noise steering by default. It trains a steering actor; the
OpenPI model weights stay frozen. `dsrl.noise_mode: "additive"` remains available
for historical residual-steering comparisons. DSRL supports parallel training physics workers through `num_envs`. The step
budget sums physical steps across workers. Each collected chunk adds one replay
transition and triggers one actor / 25 critic updates after warmup. OpenPI
inference is serialized by default. Set `dsrl.batched_inference: true` to stack
observations and independent steering noise into one shared VLM forward pass.
Input/output transforms are applied per sample, and sampling seeds are retained
per environment. DINO and reward requests are batched. Evaluation remains sequential. Scratch SAC supports `num_envs > 1` and keeps its update to
transition ratio fixed with `updates_per_transition`.

Scratch SAC defaults to simulator reward minus one or Robometer progress minus
one. DSRL uses the maximum native sparse reward across the action chunk, or mean
Robometer progress across the chunk, without an oracle success bonus. Both
terminate on simulator success and evaluate success using the simulator.
Robometer and DINO stay frozen. `sac` overrides are applied after the profile;
changing `dsrl.action_exec_len` does not implicitly change `sac.gamma`.

Every run evaluates at step zero, before learning, and then at `eval_every`.
DSRL evaluations can occur a few physical steps after a boundary because an
action chunk must finish first. The default reset split is training states 0–24
and evaluation states 25–49. Larger evaluation episode counts repeat the same
held-out states. An initial SAC actor can be supplied through `initial_actor`
when `base_model` is `scratch`.

## Environments, GPUs, and storage

Use the existing uv environments: `.venv-policy` for SAC/LIBERO, `.venv-reward`
for Robometer, and the Python 3.11 OpenPI environment specified by
`runtime.openpi_python`. OpenPI runs in its own process; Robometer has a separate
worker. This keeps JAX and reward-model dependencies isolated.

Relative paths are interpreted from the repository root. Interpreter symlinks
are preserved so virtual-environment packages remain available. The default
OpenPI paths are `runtime.data_root/pi0_checkpoints/{pi0_libero,pi05_libero}` and
`runtime.data_root/.venv-pi0/bin/python`; override them if needed. The shared
cache and temporary directory default to `runtime.data_root/cache` and `tmp`.
`runtime.cache_dir` can point at an existing Hugging Face cache root to reuse
previous downloads.

The scheduler checks idle GPUs and never stops unrelated processes. If no GPU
is idle, it exits; add `--wait-for-gpus` to wait. For a reward model that needs a
separate device, use e.g. `"gpus": [0], "reward_gpu": 1`. W&B is disabled in all
workers. CSV, JSONL, plots, and checkpoints are local.

Only one rolling `checkpoints/latest.pt` is kept during training; completion
replaces it with `final.pt`. The frozen VLM checkpoints and replay buffer are not
copied into run checkpoints. The scheduler checks free disk space before each
launch (`runtime.min_free_gib`, default 30).

Default output:

```text
<data_root>/experiments/<name>/
  experiment.json                      # fully resolved settings
  <base>_<suite>_task<id>_<name>/
    manifest.json                      # planned runs and dashboard metadata
    orchestrator.json
    seed100/sparse_seed100/
      config.json
      evaluation.csv
      episodes.jsonl
      metrics.jsonl
      status.json
      checkpoints/latest.pt            # final.pt when complete
    seed100/robometer_seed100/...
  analysis/...
```

Pass `--out` to choose another experiment root. To add seeds, append them to the
same specification and rerun with `--resume`. Completed runs are skipped; settings
must otherwise match. Use `run --after-run /path/to/existing/run --wait-for-gpus` to queue a new
experiment until an existing run finishes successfully. Repeat `--after-run`
for multiple prerequisites; failures block the queued experiment.

This is job-level resume: interrupted training runs are
refused rather than silently overwritten or treated as resumed learning.

## Reusable analysis

```bash
.venv-policy/bin/python -m experiments.libero analyze /path/to/experiment
.venv-policy/bin/python -m experiments.libero analyze /path/to/task --out /path/to/report
```

The same command reads new experiment roots, individual task roots, historical
nested DSRL folders, and historical flat SAC folders. It writes PNG/PDF plots,
`statistics.json`, `curves.csv`, and a local `index.html`. `run` invokes it once
all runs complete; it can also be invoked while training is in progress.

Plots contain success rates, normalized running success AUC, and collected
training episode returns. Success curves include individual seeds, seed means,
and 95% Student t confidence intervals. The frozen pretrained baseline uses
step-zero evaluations, averaging reward arms within each seed before estimating
uncertainty. With one seed, the baseline interval instead describes reset-state
variation and is labeled accordingly. Scratch models have no pretrained baseline.

Final comparisons use completed paired seeds at the same budget. Pointwise
stars use exact two-sided paired sign-flip tests: green sparse vs pretrained,
orange Robometer vs pretrained, black sparse vs Robometer. Tests condition on
the chosen held-out reset set; no multiple-comparison correction is applied.
With three paired seeds the smallest two-sided exact p-value is 0.25, so these
tests cannot establish significance at 0.05. Missing runs are excluded, never
filled with zero. Incompatible protocols are rejected instead of pooled.

The existing dashboard command and URL remain available:

```bash
.venv-policy/bin/python -m experiments.pi0_stove.dashboard \
  --data-root /data/rohan/Reward-Modeling --port 8768
```

The dashboard discovers new manifests under `<data_root>/experiments/*/*` and
continues displaying historical results and video audits. Its old package name
is retained only for the dashboard command. Historical page adapters are
isolated in `experiments.libero.legacy`; new training and analysis do not
import them.

## DSRL diagnostics

Select an experiment in the dashboard dropdown, or click **Open experiment
diagnostics**, to load its diagnostic plots. The overview keeps the main
success curves compact; the selected experiment shows five small charts per
row on wide screens, grouped by latent policy, exploration, critic, actor,
training rollouts, and evaluation.

DSRL enables `collect_diagnostics` in the shared SAC optimizer only at the
configured `log_every` interval (normally 1,000 physical steps). These metrics
describe the learned noise-space steering actor and critic, not the frozen
OpenPI model's robot-action distribution. Scratch SAC leaves the new option
disabled. Distribution statistics use Gaussian parameters before tanh;
entropy estimates use the transformed latent-policy log probability.

`metrics.jsonl` includes latent mean/std and norms, entropy and temperature,
critic predictions, actual Bellman targets, critic disagreement, actor
objective, replay size, and cumulative actor/critic updates and collected
chunks. `episodes.jsonl` includes simulator success, physical length, chunk
reward return, native simulator return, timeout, and chunk count. Evaluation
also summarizes native return and episode length. Checkpoint storage is
unaffected.

The historical `target_q_mean` field retains its old meaning: next-state
bootstrap value. New `bellman_target_mean` includes reward, discount, and the
terminal mask. The diagnostic reader preserves this distinction, derives
entropy from historical log probabilities when available, and leaves missing
quantities blank. It averages within 1,000-step bins separately for each seed
before computing the mean across seeds; these are diagnostic traces without
confidence bands. Partial evaluation batches are excluded.

The logger is implemented in `train.py` and
`robometer_policy_learning/algorithms/sac/modeling_sac.py`; the reusable local
reader and chart renderer live in `diagnostics.py`. Already running processes
use the code they loaded at startup; new logging takes effect on subsequent
launches. No restart or mid-run checkpoint continuation is implied.

## Implementation

- `config.py`: validated task/seed/reward plans and explicit SAC/DSRL profiles.
- `runner.py`: shared GPU scheduling, worker launch, storage checks, and job resume.
- `train.py`: DSRL loop and dispatch to the existing scratch SAC loop.
- `policy.py`, `policy_worker.py`: shared OpenPI RPC client and JAX worker.
- `analysis.py`: result readers, statistics, plots, and report generation.
- `download.py`: one OpenPI checkpoint downloader for both policy variants.
- `legacy/`: historical dashboard page adapters; not used by the new runner.

Low-level LIBERO workers, replay/SAC implementation, reward RPC, and GPU guards
remain shared in `experiments.reward_audit`. The task-specific fine-tuning
launchers were removed, rather than retained as additional orchestration layers.

## Reusable policy video audits

```bash
.venv-policy/bin/python -m experiments.libero video-audit /path/to/task-results \
  --out /data/rohan/Reward-Modeling/video_audits/comparison-name \
  --seed 100 --episodes 25 --successes 2 --failures 2 --gpus 0 1
```

The command reads the saved training configurations and final actor checkpoints,
so it works for any supported task and base model, with either or both reward
variants. It also evaluates the frozen base checkpoint for OpenPI policies.
Use `--task-id` when the results contain several tasks. Choose the training seed
in advance; it does not select the best seed. The task, horizon, state split,
action execution length, encoder, and policy configuration come from the run.
Use `--horizon 600` for LIBERO-10 video audits; a horizon override is recorded
as a separate evaluation protocol and does not change training results.

Each policy is evaluated on 25 matched episodes by default. After collecting all
episodes, two successes and two failures are selected randomly, using
`--selection-seed` (default 0). When an outcome has fewer examples, the report
shows the available examples and reports the shortfall. The full episode table
links to all recordings, allowing comparisons from the same initial state.

The report includes Robometer progress and success-probability traces, plus the
native simulator reward, for selected clips. Reward inference uses observed
prefixes and the run's reward checkpoint, camera, and frame limit. Trace inference
is sampled every 20 physical steps. All episode MP4s are retained; raw frames are
pruned to the selected examples after scoring to bound storage. No training is
performed. All workers use the existing GPU reservations and W&B-disabled setup.
The audit requires at least 5 GiB free for temporary frame files.

`--sampling deterministic` reproduces the latent-mean evaluation convention of
the existing curves. `--sampling stochastic` evaluates samples from the learned
latent policy, as in the official DSRL evaluator. The convention is saved and
displayed in the report. These are separate evaluation protocols; new audit
measurements do not overwrite training curves. A scratch policy has no frozen
OpenPI baseline. Restarting the same audit skips completed variants and resumes
recorded episodes, while an output lock prevents concurrent writers.

Set `HF_HOME`, `XDG_CACHE_HOME`, and `TMPDIR` to the existing shared cache and
temporary directories when working with older runs that do not store runtime
paths. Pass verified checkpoint coverage through `--training-coverage`; otherwise
the report labels it unverified. The checkpoint identities and actor file hashes
are recorded regardless.

Audits saved under `<data_root>/video_audits/<name>` appear automatically in the
main dashboard's Video Audit sidebar and are served at `/video-audits/<name>/`.

For a frozen model without any fine-tuning results, pass the ordinary experiment
JSON instead of a results directory. The task and checkpoint are taken from that
file; no training is started:

```bash
.venv-policy/bin/python -m experiments.libero video-audit experiment.json \
  --out /data/rohan/Reward-Modeling/video_audits/frozen-task-comparison \
  --episodes 10 --demos 4 --successes 2 --failures 2 --gpus 0 1
```

`--demos` extracts the first requested demonstrations from the pinned public
LIBERO dataset and includes their Robometer traces and stored native rewards.
Use `--demo-file` for an existing local task HDF5. The demonstration instruction
is checked against the configured task's BDDL instruction before rendering.
Demonstrations are displayed separately from held-out policy evaluations and do
not contribute to the policy success rate. If 10 evaluations do not contain two
examples of each outcome, the report shows the available examples and the
shortfall; it does not add evaluations beyond the requested budget.

## Policy batching benchmark

```bash
.venv-policy/bin/python -m experiments.libero.benchmark experiment.json \
  --out /data/rohan/Reward-Modeling/benchmarks/my-test --gpu 0 --repeats 20 --rounds 20
```

Use one pretrained-policy task/seed with sparse reward and `num_envs: 4`.
This short benchmark compares serial and batched policy inference on identical
observations/noise, then measures real simulator collection and DSRL updates in
serial-batched-batched-serial blocks. It retains the configured warmup/update ratio,
excludes compilation/warmup from steady-state timing, and records per-round times,
action differences, and sampled GPU memory locally. Evaluation/checkpoint time
and Robometer inference are excluded. It does not schedule a long training run.
Outputs include `summary.json`, `rounds.jsonl`, and `report.html`.
