# November OGPO experiment configurations

These are the actual task specs from the experiment machine, preserved on 2026-11-21.
They contain absolute paths and must be adapted before launching on another machine.
Do not launch every JSON indiscriminately: BC specs train a new head; RL specs reuse
that campaign's selected checkpoint. Checkpoint weights and datasets are not in Git.

- LIBERO-10 task08: 50 demos, fresh BC5000, final checkpoint explicitly selected
  (ODE4%, SDE0%); three RL seeds100–102 per reward,100k physical steps each.
- LIBERO-GOAL task07: 25 demos, BC5000/evaluation500; first checkpoint with
  0<ODE success<50%, otherwise the lowest positive success rate. Zero never
  qualifies. Selected BC500 measured ODE12%, SDE16%; five RL seeds100–104 per
  reward,100k physical steps each. The two arms share the BC weights; the RL RNG
  is reseeded independently. Historical superseded task07 starts are omitted.

All runs use frozen BF16 pi05_libero features and the original OGPO small flow MLP,
25 evaluation episodes per mode, one environment and four-action chunks. Robometer
runs use robometer/Robometer-4B, with explicitly enabled GPU colocation. Replay
buffers are not saved. A saved RL checkpoint alone cannot resume the exact prior
training state without its replay; use the BC checkpoint for fresh RL seeds.

## On the new machine

```sh
git clone --branch rss27 --recurse-submodules \
  git@github.com:UWRobotLearning/robometer-policy-learning.git
cd robometer-policy-learning
```

Follow `docs/REPOSITORIES.md` and `OGPO/scripts/LIBERO.md` to build separate uv
virtual environments for OGPO/JAX, simulator/OpenPI, and the Robometer worker.

Transfer or provision:

1. The selected BC files in `checkpoint_manifest.json`, preserving their hashes.
   Copy their `bc_selected/selection.json` and the BC run's spec too. Task08's
   final checkpoint was explicitly copied to `bc_selected` after training.
2. pi05_libero encoder params, the PaliGemma tokenizer, and Robometer-4B weights.
3. The task demonstration HDF5 files. The original dataset source is
   `yifengzhu-hf/LIBERO-datasets`, revision
   `f13aa24a3da8c43c7225569f28c562979fa0e35a`.
4. Existing run directories/evaluation logs if historical plots should appear on
   the dashboard. Feature caches are optional; they can be rebuilt.

Copy the desired specs to the new data root. Update `policy_repo`, `sim_python`,
`output_dir`, `demo_file`, `encoder_checkpoint`, `feature_cache`,
`bc_selection_spec`, `bc_source.checkpoint`, and
`overrides["checkpoint.restore_bc_path"]`. Set
`overrides["agent.paligemma_tokenizer_path"]` explicitly for the new machine.
The launcher also defaults to the old tokenizer path, so do not omit this override.
Update/remove the old-machine sanity-result paths (`required_sanity`,
`colocation_sanity`) only after checking memory for the new hardware. Choose
`policy_gpu` and `reward_gpu` for that machine. Absolute paths in feature-cache
metadata may require rebuilding the cache.

For a fresh additional RL seed, use a new output directory and seed, retain the
campaign `comparison_id`, and run from `OGPO/`:

```sh
.venv/bin/python scripts/libero.py start --spec /new/data/run.json --gpu 0 \
  --queue-name queue_gpu0_new_run
```

The queue resolves `bc_selection_spec` to its saved selected checkpoint. Do not
retrain BC if the intent is to compare additional RL seeds from the same initial
weights. The local dashboard reads specs/run data under `<data-root>/ogpo_libero`.
