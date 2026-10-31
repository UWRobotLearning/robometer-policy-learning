> New experiments use [experiments.libero](../libero/README.md): one configuration for scratch SAC, pi0, or pi0.5 and sparse or Robometer rewards. This directory provides shared low-level workers and the historical SAC reports.

# LIBERO reward audit

Compare a **frozen pretrained Robometer** against LIBERO's sparse simulator reward using the repository's native SAC learner, actor, and critic. This is the baseline study for deciding whether online reward-model adaptation is needed. It fine-tunes the policy; it does not update Robometer online.

**October 6 run:** a real GPU-1 LIBERO/SAC smoke test completed, including updates, evaluation, and checkpoints. The full `sanity_sparse` and `ood_sparse` runs were started sequentially on GPU 1. Robometer inference is queued until both GPUs are genuinely idle; GPU 0's existing job is untouched. Outputs are under `outputs/reward_audit_20261006/`, linked to `/data/rohan/Reward-Modeling/reward_audit_20261006/`. The short smoke test is a plumbing check, not performance evidence.

**October 7 results:** both sparse baselines completed. Drawer closing reached 25/25 held-out reset-state successes at 100,000 steps (3.89 hours); the butter-in-drawer task remained at 0/25 at 200,000 steps (5.54 hours), with zero successes during training as well. These runs trained SAC from scratch. The original Robometer arm failed before learning on a float32/bfloat16 mismatch. Its inference path now uses checkpoint-precision autocast and restores the final hidden-state tuple entry that Unsloth's Qwen3 wrapper omits, using its actual `last_hidden_state`. Neither change modifies the reward model's weights. The OOD result is inconclusive about reward harm because the sparse baseline did not learn. Curves and raw evaluations are in `outputs/reward_audit_20261006/full/`.

The repaired Robometer worker passed real GPU inference probes with one, two, and eight frames, including four-trajectory batches and mixed-length requests for eight trajectories. All 25 CPU tests passed. A fresh Robometer retry queue is in `outputs/reward_audit_20261007/full/`, with links to the completed sparse baselines for matched reports. Original failed logs are preserved. Probe outputs are saved in `outputs/reward_audit_20261007/inference_probe2/results.json`. Valid inference does not itself establish policy performance or reward accuracy.

**Additional sparse task:** `configs/microwave.json` selects LIBERO-90 task 33, `KITCHEN_SCENE6_close_the_microwave`, for 100,000 aggregate transitions with eight parallel environments, seed 100, and the same SAC settings. This is the other task in the paper's Appendix E.1. The new sparse-only queue is in `outputs/reward_audit_20261007/microwave/`; it waits for GPU 1 and its launcher reservation to be free. It leaves the active drawer Robometer run and following butter Robometer run in progress. Sparse-only queues check only their policy GPU, including during model-loading gaps where an existing launcher holds a reservation but has not allocated CUDA memory yet.

**LIBERO-Goal stove comparison:** `configs/stove.json` selects `turn_on_the_stove`, task 7, for two matched SAC runs with 100,000 total transitions, eight environments, seed 100, and 25 evaluations every 5,000 transitions. Ground truth is the native sparse simulator reward 0/1, with no offset. The goal is `(Turnon flat_stove_1)`, satisfied when its control joint reaches 0.5 radians. The model-only reward is Robometer progress in [0,1]. This suite is in the checkpoint's declared training mix, so no held-out task claim is made. Outputs are in `outputs/reward_audit_20261007/stove_parallel/`.

The stove pair uses both GPUs concurrently: ground-truth SAC on GPU 1, and SAC plus the Robometer worker on GPU 0 with `--allow-shared-reward-gpu`. Both selected GPUs must be initially idle. A short shared-GPU smoke run validates model inference, rendering, and learner updates before the full Robometer arm starts. The default launcher still requires separate policy/reward GPUs unless this option is explicitly supplied. Planning estimates are 3–4 hours for the sparse run and 5–8 hours for the model run; concurrent wall time depends on shared-GPU throughput and whether the unchanged decoder completes the run.

The initial runs kept the upstream progress decoder condition **unchanged**, as requested. Diagnostic hooks capture its input logits and compare decoded scores with an independent NumPy softmax expectation. The reference is never substituted for training rewards. `decoder_checks.json` summarizes counts; `decoder_checks.jsonl` records sampled checks and every anomaly, and `decoder_anomaly_*.npz` preserves raw frames/logits for replay. Existing invalid-score checks still stop a run after saving diagnostics. The first diagnostics smoke exposed a squeezed single-frame shape in the observer, which is handled by the updated observer; its failed logs are preserved under `smoke/`, and the retry is under `smoke2/`.

**Confirmed decoder defect and correction:** stove Robometer seed 101 stopped around 22,400 transitions. Its saved ten-bin logits sum exactly to one but include negative numbers: the original heuristic bypassed softmax and returned progress −4.9778647. Replaying the actual logits through unconditional softmax yields 0.10182635. The user subsequently authorized fixing both Robometer copies. `convert_bins_to_continuous` now always softmaxes prediction logits, with a stable NumPy implementation as well. Explicit normalized target distributions use a separate weighted expectation; integer target bins map to floating-point centers. The success head retains sigmoid. New worker provenance identifies `always_softmax_v1`; diagnostic `legacy_condition_hits` count where the removed heuristic would have triggered, not actual bypasses. Reference differences and invalid-output guards remain enabled.

The corrected follow-up under `stove_three_seeds/finish_robometer_seeds.py` waits for the existing sparse seed 102 coordinator to finish naturally, preserves failed Robometer seed 101 under `failed_runs/stove_robometer_seed101_before_decoder_fix/`, and starts Robometer seeds 101 and 102 from scratch with the fixed decoder. It keeps the original sparse baselines and labels decoder versions in the HTML. Seed 100 is retained as pre-fix with zero recorded bypasses or discrepancies; an anomalous completed pre-fix run is rejected from pooled statistics. `decoder_fix.json` links the correction and archived failure. A failed partial run is not an additional seed measurement and is never counted as zero success.

Plots now additionally save `success_rate_vs_iterations.png`, `.pdf`, and `iteration_curves.json`. The main x-axis counts vector-environment iterations (total transitions divided by environment count), and the upper axis counts total physical transitions. For the stove runs, 100,000 transitions means 12,500 iterations. Report refreshes are local only. All 31 CPU tests, including decoder-observer and axis-conversion regressions, passed.

**Completed stove results:** both arms finished 100,000 training transitions. Simulator-reward SAC finished at 8/25 successes (32%) after 2h 53m; Robometer-reward SAC finished at 2/25 (8%) after 5h. The Robometer peak was 10/25 (40%) at 40,000 transitions, which is a peak checkpoint, not an early stop. The original decoder remained unchanged; the latest persisted checks cover 100,698 samples with no skip-condition hits, reference discrepancies, or invalid outputs. These are single-seed results on an in-distribution suite.

Open `outputs/reward_audit_20261007/stove_parallel/index.html` for the local interactive report. It embeds all curve data and works directly from a file without a server or network access. It includes transitions/iterations/runtime axes, checkpoint inspection, confidence bands, final versus peak results, decoder diagnostics, and relative links to raw CSVs and final evaluation videos. Regenerate it after the numerical report with:

```bash
.venv-policy/bin/python -m experiments.reward_audit.html_report \
  outputs/reward_audit_20261007/stove_parallel
```

The HTML generator supports multiple matched training seeds. It includes pending runs from `planned_suite.json`, uses only fully completed seed pairs in aggregate statistics, and validates matching experiment settings and evaluation grids. `seed_statistics.json` records final means, sample SD, approximate Student t intervals across training seeds, normalized learning-curve AUC, paired Robometer-minus-simulator differences, and all aggregate curve points. The PNG/PDF and interactive mean-curve bands show sample SD across seeds; individual seed views retain reset-state Wilson intervals. Missing seeds are never treated as zero success.

**Two additional stove seeds:** the October 8 follow-up uses seeds **101 and 102** for both reward arms and includes the completed seed 100 pair without retraining it. Outputs and the automatically refreshed HTML are under `outputs/reward_audit_20261008/stove_three_seeds/`. All training settings remain those in `configs/stove.json`. Both GPU lanes wait for the wine-bottle Robometer run to complete before starting; GPU 1 runs sparse SAC seeds sequentially, and GPU 0 runs Robometer SAC seeds sequentially. The single coordinator starts training workers directly, without spawning queue or launcher subprocesses. Existing GPU occupancy/reservation checks remain. `queue.py` is retained for compatibility with earlier runs and is not used by this workflow.

```bash
HF_HOME=/data/rohan/Reward-Modeling/huggingface \
.venv-policy/bin/python -u -m experiments.reward_audit.parallel \
  --suite experiments/reward_audit/configs/stove.json \
  --seeds 101 102 \
  --include-existing outputs/reward_audit_20261007/stove_parallel \
  --after-run outputs/reward_audit_20261008/wine_rack_parallel/wine_robometer \
  --out outputs/reward_audit_20261008/stove_three_seeds \
  --oracle-gpu 1 --model-gpu 0
```

This follow-up's `experiment_status.json` records the dependency, coordinator PID, and each new run's seed, reward, status, PID, and GPU. It refreshes the HTML every five minutes during training and after all runs finish. Three training seeds remain a small sample: the same 25 held-out reset states are evaluated per seed, and the 75 episodes per arm must not be interpreted as 75 independent training runs. With only one completed pair, sample SD and seed-level confidence intervals are unavailable. Existing seed 100 outputs are linked and never modified.

**Wine bottle / rack comparison:** `configs/wine_rack.json` selects LIBERO-Goal task **9** (zero-based, the tenth task), `put_the_wine_bottle_on_the_rack`. Its goal is `(On wine_bottle_1 wine_rack_1_top_region)`. Both arms use the stove comparison's matched SAC settings: 100,000 aggregate transitions, eight environments, seed 100, raw simulator reward 0/1 versus frozen Robometer progress, and 25 held-out evaluations every 5,000 transitions. This is in the declared reward-model training mix. The similarly named LIBERO-90 task has ID 27 and is a different task definition.

The October 8 wine pair is queued under `outputs/reward_audit_20261008/wine_rack_parallel/`: `wine_sparse` on GPU 1 and `wine_robometer` (policy plus reward worker) on GPU 0. At queue creation both GPUs were occupied by unrelated jobs; no experiment used either GPU. Each arm independently waits for three idle observations and passes the launcher's final idle/lock checks. Existing processes are never stopped. The supervisor refreshes combined success curves every five minutes and at completion. Per-arm queue state is under `.queues/<run>/reward_queue.json`; `parallel_status.json` records the supervisor and both queue PIDs. The unchanged decoder and existing diagnostics apply to the Robometer arm.

The wine experiment above used the earlier queue-based coordinator. To launch a fresh pair with the simplified direct coordinator:

```bash
HF_HOME=/data/rohan/Reward-Modeling/huggingface \
.venv-policy/bin/python -u -m experiments.reward_audit.parallel \
  --suite experiments/reward_audit/configs/wine_rack.json \
  --seeds 100 \
  --out outputs/reward_audit_wine_new \
  --oracle-gpu 1 --model-gpu 0
```

This command refuses an existing output directory. The coordinator writes each new run directly into that directory and generates the combined report. All outputs remain local, with W&B disabled.

Measured sparse runtimes, including evaluation, were 3.89 hours for 100,000 drawer transitions and 5.54 hours for 200,000 butter transitions. Initial planning estimates are 3–4 hours for the new sparse microwave run, 6–10 hours for the 100,000-transition drawer Robometer run, and 8–12 hours for the 200,000-transition butter Robometer run. Robometer estimates combine measured sparse runtime with warmed inference timings; they are not measured full-run durations, and queue waiting time is additional. Current curves and episode counts should be used to refine estimates once training checkpoints are available.

## Experiment choices

The [Robometer paper, Appendix E.1](https://arxiv.org/html/2603.02115v1#A5.SS1) uses SAC from scratch on LIBERO-90 task 28, close the top drawer, and task 33, close the microwave. DSRL is used in its real-robot experiments. We use SAC throughout, as requested.

| Arm | Task | Training reward | Physical steps / environments |
| --- | --- | --- | --- |
| `sanity_sparse` | 28: close the top drawer | Simulator success − 1 | 100,000 / 1 |
| `sanity_robometer` | 28: close the top drawer | Robometer progress − 1 | 100,000 / 1 |
| `sanity_hybrid` | 28: close the top drawer | Simulator success − 1 + Robometer progress | 100,000 / 1 |
| `ood_sparse` | 4: put front butter in drawer and close it | Simulator success − 1 | 200,000 / 8 |
| `ood_robometer` | Same task 4 | Robometer progress − 1 | 200,000 / 8 |
| `ood_robometer_wrist` | Same task 4 | Same model reward, wrist camera input | 200,000 / 8 |

Task names are checked against the pinned LIBERO task map. Task 4 requires correct object selection, placement **inside** a drawer, and closure. A model that rewards grasping or drawer closure without correct placement could reinforce incomplete execution. This is a prespecified candidate, not a measured Robometer failure. No outcome is guaranteed.

The wrist arm is an optional reward-observation stress test: SAC still observes the same external camera and proprioception, and simulator dynamics/goals remain the same. Only Robometer gets the wrist view. This distinguishes an explicit input shift from natural task generalization; wrist views may also work well. Compare it with both `ood_robometer` and `ood_sparse`.

The sanity protocol uses Appendix E.1's batch size 128, gamma .99, tau .005, five critics, two sampled target critics, pooled critic features, one critic update per actor update, actor/critic LR 1e−5, entropy LR 3e−4, target entropy 0, 5,000 warmup steps, and 25 evaluations every 5,000 steps. DINOv2-small encodes the external image. The actor/critic architectures and SAC target reduction are inherited from this repository. Every additional parallel transition gets one learner update, preserving the update/data ratio.

**Reproduction limit:** the released [general Robometer-4B](https://huggingface.co/robometer/Robometer-4B) is trained on RBM-1M. The paper's Figure 5 uses a LIBERO-only H2 ablation checkpoint. Our default is a **paper-protocol sanity check with public weights**, not an exact Figure 5 reproduction. The model is pinned to `beef63bc914c5c189329d49c6d712d96d632aa34`; its `config.yaml` lists `rbm-1m-id`, whose training alias excludes LIBERO-90 in the pinned Robometer repository. Supply the matching H2 checkpoint with `--reward-checkpoint` if available. The hybrid arm follows the dense-plus-sparse formulation in the policy repository's LIBERO example; its advantage cannot be attributed to model-only rewards.

Do not use [jesbu1/robometer-4b-fft-libero](https://huggingface.co/jesbu1/robometer-4b-fft-libero) to claim LIBERO-90 task OOD: its model card explicitly includes LIBERO-90 training data. The harness rejects that known checkpoint for held-out-task claims and expands training dataset aliases from the loaded model to catch leakage.

## Environments with uv

From `robometer-policy-learning/`:

```bash
export PATH="$HOME/.local/bin:$PATH"
bash experiments/reward_audit/setup.sh
```

The script uses **`uv venv`** to create `.venv-policy` and `.venv-reward`, both with Python 3.10. It installs the CUDA 12.8 PyTorch build and the simulator/inference dependencies, and installs LIBERO and Robometer from pinned submodules over HTTPS. It does not download checkpoints, initialize CUDA, or launch an experiment. Both dependency sets have been resolved with `uv pip compile`; the setup uses their version-pinned `.lock` files. TorchAO is pinned to the [release compatible with PyTorch 2.8](https://github.com/pytorch/ao/issues/2919). Freeze files are saved locally. The parent project mixes Python 3.11/OpenPI requirements with Robometer's Python 3.10 requirements; this SAC harness runs the parent source directly and isolates Robometer in a second process, avoiding that conflicting dependency graph.

The inference requirements include W&B because upstream Robometer imports it. This harness never initializes its logger; `WANDB_MODE=disabled` and `WANDB_DISABLED=true` are set in both processes. There is no W&B run or upload, including offline W&B logging. Dataset/model downloads are the only intended remote access.

The CPU validation environment already created here is `.venv-audit-test`, also created with `uv venv`. It contains a CPU-only PyTorch build. It cannot run the real experiment and is not a substitute for the two runtime environments.

## Review and launch

These commands do not initialize CUDA:

```bash
.venv-policy/bin/python -m experiments.reward_audit.launch --dry-run \
  --runs sanity_sparse sanity_robometer ood_sparse ood_robometer

.venv-policy/bin/python -m experiments.reward_audit.launch --check \
  --policy-gpu 0 --reward-gpu 1
```

`--check` exits with code 2 if GPUs are occupied or dependencies/storage are missing. It writes `outputs/reward_audit/preflight.json`. The launch path makes the same check before any CUDA import and immediately before each run. An occupied GPU has a compute process, more than 512 MiB allocated, or more than 5% utilization. There is no bypass or action on other users' processes. SAC/DINO/rendering and Robometer use separate GPUs by default; `--allow-shared-reward-gpu` permits colocation within one experiment on an initially idle GPU after checking memory with a smoke run. EGL uses the physical policy GPU index, including when the two GPU assignments are swapped.

An explicit queue can wait for completed prerequisite runs and three idle observations, 30 seconds apart. It then invokes the same guarded launcher, which rechecks the GPUs and acquires launcher locks. It never uses an estimated release time as permission to launch. The October 6 queue uses:

```bash
HF_HOME=/data/rohan/Reward-Modeling/huggingface \
.venv-policy/bin/python -u -m experiments.reward_audit.queue \
  --out outputs/reward_audit_20261006/full \
  --after sanity_sparse ood_sparse \
  --runs sanity_robometer ood_robometer \
  --policy-gpu 1 --reward-gpu 0
```

`full/reward_queue.json` records its PID and state; `reward_queue.log`, `sparse_driver.log`, and `reward_driver.log` contain launcher output. The queue stops if a prerequisite fails. It refreshes local PNG/PDF curves every five minutes while waiting or running. Environments were created with `uv venv`; the reward environment and Hugging Face cache are stored on `/data` to avoid filling the root filesystem.

The additional microwave baseline waits for its required GPU to become idle without depending on either Robometer run succeeding:

```bash
HF_HOME=/data/rohan/Reward-Modeling/huggingface \
.venv-policy/bin/python -u -m experiments.reward_audit.queue \
  --suite experiments/reward_audit/configs/microwave.json \
  --runs microwave_sparse \
  --out outputs/reward_audit_20261007/microwave \
  --policy-gpu 1 --reward-gpu 0
```

**After the assigned GPUs are idle**, run the four core arms:

```bash
.venv-policy/bin/python -m experiments.reward_audit.launch \
  --runs sanity_sparse sanity_robometer ood_sparse ood_robometer \
  --policy-gpu 0 --reward-gpu 1 \
  --out outputs/reward_audit_seed100
```

To include the hybrid sanity and wrist-view stress test, omit `--runs`. Runs execute sequentially; each model arm uses both GPUs. A launcher lock prevents two copies of this harness from colliding. The guard cannot reserve devices against unrelated applications that start later; it checks immediately before launching. Existing run folders are never overwritten.

A failed partial run keeps its local logs/checkpoints. Use a new output directory for another run. Exact resume, including replay and simulator state, is not implemented.

## Policy fine-tuning

Defaults train from scratch, following the LIBERO paper experiment. To fine-tune an existing matching SAC policy, use the same checkpoint in both reward arms:

```bash
.venv-policy/bin/python -m experiments.reward_audit.launch \
  --runs ood_sparse ood_robometer \
  --initial-actor /absolute/path/to/task4/checkpoints/final.pt \
  --out outputs/reward_audit_warmstart
```

This loads only actor weights and starts fresh replay and optimizer state. It accepts harness checkpoints or a compatible native MLP actor state dict; it does not accept a VLA checkpoint. Task/encoder/architecture metadata is checked when present. Both arms evaluate the initial policy at step zero. A pretrained SAC policy already near 100% success leaves little headroom; choose a partly successful checkpoint when assessing improvement or degradation.

If sparse SAC never improves on task 4, the experiment is inconclusive about reward harm. A task-matched policy warm-start or more exploration is then necessary. Changing both arms together preserves the comparison. Task 33 is another paper-backed sanity option: replace the task name in a copy of the suite JSON with `KITCHEN_SCENE6_close_the_microwave` and pass `--suite`.

## Reward and evaluation contract

- Raw simulator reward is 0/1. The configured −1 offset makes the sparse learning reward −1/0, matching the repository's LIBERO wrapper. Set `sparse_offset=0` in a copied suite for raw 0/1, consistently across paired arms.
- Model-only reward is `progress−1`, with no simulator reward or success bonus added. `progress`, `delta`, and explicit `sparse_plus_progress` are supported controlled variants. Formula choices are recorded in the resolved config.
- Simulator success terminates every arm; timeout is shared. Robometer's success probability is diagnostic only. Thus this tests reward quality with oracle termination, not fully autonomous model-based termination.
- At each physical step, Robometer sees at most eight evenly sampled frames from that environment's observed prefix, including the current endpoint. No future frames or other episodes enter its context. Equal-frame-count microbatches prevent padded head outputs from being mistaken for endpoints. Malformed, missing, or nonfinite predictions abort the run.
- SAC/DINO receives the external camera and the repository's 8D end-effector/gripper state convention. It receives no object-state features, task progress, or success indicators. The repository's unused constant language embedding is retained for native network compatibility.
- Official initial states 0–24 are used for training; states 25–49 are used once per evaluation checkpoint. Evaluation runs in a separate environment, uses deterministic actor actions, and restores learner RNG state. These are reset-state uncertainty estimates for **one training seed**.
- Replays store frozen DINO features and proprioception, not image histories. True terminal observations enter replay before reset. Timeout transitions bootstrap; successful terminal transitions do not.

## Local outputs and interpretation

Each run folder contains `config.json`, source/model/encoder provenance, `console.log`, `metrics.jsonl`, `episodes.jsonl`, `evaluation.csv`, `status.json`, one evaluation video per checkpoint, and `checkpoints/latest.pt` / `final.pt`. Training episode records include true success, return, final progress, and predicted success probability. Evaluation logs additionally summarize progress AUROC, success Brier score, and false-positive/false-negative rates. AUROC is undefined when there are no successes or no failures; it is stored as null.

After the suite, or while inspecting completed/partial runs:

```bash
.venv-policy/bin/python -m experiments.reward_audit.report outputs/reward_audit_seed100
```

This saves `performance.png`, `performance.pdf`, `summary.json`, and `REPORT.md` locally. No curves are fabricated when no measurements exist. It compares final simulator success and normalized success-curve AUC at equal physical-step budgets, checks matching initial policies/configuration, and gives paired reset-state bootstrap intervals. A negative final difference with a learning sparse baseline is labeled underperformance **in this seed**. It never forces a negative result and does not treat two failed learners as evidence of a bad reward model. Multiple training seeds would be needed to establish reliability beyond this sanity check.

Robometer's evaluation metrics are evidence about reward calibration; policy success is evidence about downstream utility. High learned returns and high false-positive probabilities on simulator failures strengthen a reward-error explanation. The wrist arm supports conclusions about that explicit reward-camera shift only. The natural task is already held out for this checkpoint; it is not a new synthetic task, and nominally OOD tasks can still generalize well.

For the multi-seed HTML report and sparse-only reward curves:

```bash
.venv-policy/bin/python -m experiments.reward_audit.html_report outputs/reward_audit_20261008/stove_three_seeds
.venv-policy/bin/python -m experiments.reward_audit.serve_report outputs/reward_audit_20261008/stove_three_seeds --port 8765
```

The server refreshes the report from local logs whenever its HTML page is requested. It binds to all interfaces by default; use `--bind 127.0.0.1` for localhost only. Sparse-only plots show held-out success, collected training episode returns, and terminal success rewards per 1,000 transitions. Training episodes are grouped by completion time in 5,000-transition windows; unfinished episodes are excluded, and incomplete windows are marked. This summary includes completed sparse seeds even when their Robometer partner is pending. Matched comparison statistics continue to require both arms to complete.

## CPU validation

```bash
CUDA_VISIBLE_DEVICES='' .venv-audit-test/bin/python -m pytest experiments/reward_audit/tests -q
```

Tests use synthetic fixtures, including the real native SAC update, an occupied-GPU queue check, and the Qwen3 hidden-state compatibility regression. They do not instantiate LIBERO, load a VLM, or generate experimental performance measurements. Separately, the GPU-1 smoke test and completed sparse baselines validated LIBERO rendering and SAC updates; the October 7 GPU-0 probes validated repaired Robometer checkpoint inference.
