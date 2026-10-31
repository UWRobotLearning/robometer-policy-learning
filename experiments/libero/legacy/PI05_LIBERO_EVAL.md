# Frozen π0.5 LIBERO evaluation

The checkpoint is `gs://openpi-assets/checkpoints/pi05_libero/` (OpenPI config `pi05_libero`). The downloader puts its 16 files under `/data/rohan/Reward-Modeling/pi0_checkpoints/pi05_libero`, checking each remote file size before renaming it into place. Download with:

```bash
.venv-policy/bin/python -m experiments.libero.download --name pi05_libero \
  --out /data/rohan/Reward-Modeling/pi0_checkpoints/pi05_libero
```

The sweep covers all 130 task IDs in LIBERO Spatial, Object, Goal, 10, and 90. It uses official initial-state IDs 25–49, one frozen policy, and 25 episodes per task. Success is the LIBERO simulator success predicate. The environment ends on success, otherwise at 400 steps; LIBERO-10 uses 600 steps. The π0.5 model outputs ten actions per inference, and the evaluator executes all ten. The policy worker runs in the existing uv Python 3.11 venv; the simulator uses the existing uv Python 3.10 venv. WandB is disabled.

A lane evaluates alternating tasks and resumes by skipping tasks with 25 completed episodes. Before starting a lane, check GPU occupancy. The `--wait-idle` option rechecks every minute and requires three consecutive idle checks before acquiring a GPU. To run the two lanes:

```bash
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=0 EGL_DEVICE_ID=0 TMPDIR=/data/rohan/rbm-tmp XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DISABLED=true .venv-policy/bin/python -u -m experiments.libero.legacy.pi05_libero_eval --gpu 0
MUJOCO_GL=egl MUJOCO_EGL_DEVICE_ID=1 EGL_DEVICE_ID=1 TMPDIR=/data/rohan/rbm-tmp XLA_PYTHON_CLIENT_PREALLOCATE=false WANDB_DISABLED=true .venv-policy/bin/python -u -m experiments.libero.legacy.pi05_libero_eval --gpu 1 --wait-idle
```

Per-task `episodes.jsonl`, `summary.json`, `config.json`, and `status.json` live under `/data/rohan/Reward-Modeling/pi05_libero_all_tasks_20261019/`. The live report is `/pi05-libero-eval/` on the existing port-8768 dashboard. It labels Spatial, Object, Goal, and LIBERO-10 in distribution at the family level and LIBERO-90 out of distribution: the [OpenPI checkpoint](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md) uses a [40-task dataset containing the former four suites](https://huggingface.co/datasets/physical-intelligence/libero/blob/main/README.md). This label does not establish exact trajectory overlap or absence of related tasks from base pretraining.
