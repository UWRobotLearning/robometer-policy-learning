# Experiment Atlas

A local, buildless HTML research record for Robometer policy learning. Uses only Python's standard library to prepare and serve data; the browser needs no external scripts, fonts, or CDN.

From the repository root:

```sh
python3 scripts/build_experiment_atlas.py
python3 scripts/serve_experiment_atlas.py --port 8768
```

Open http://localhost:8768/. The server binds to `0.0.0.0` by default; use `--host 127.0.0.1` for loopback only. The server serves documentation, numerical results, and experiment specs. It does not launch training.

## What is included

- All 138 runs in the October 31 SAC/DSRL run catalog, including failed and incomplete runs.
- All 24 OGPO RL runs in the newer October 31 report evidence, including queued runs. ODE/SDE evaluations remain separate.
- The baseline subpage compares both released checkpoints across all 130 tasks. The generated data retains all 266 frozen/audit records for provenance; the main page focuses on training experiments.
- The complete 2,310-file numerical export inventory, including BC searches, reward audits, optimizer logs, and historical pilots outside the normalized tables.
- All 13 saved report figures and links to both full reports.
- November experiment specifications, explicitly presented as planned work.

These counts describe the initial checked-in evidence, not live experiment processes. Run grouping preserves exported experiment directories; it does not imply matched protocols or that all runs belong in a statistical comparison. Curves show individual runs without averaging across seeds or interpolating missing checkpoints. The table shows runs in the selected experiment and respects the selected sampler.

## Append experiments

Export another snapshot following `results/README.md`, then rebuild:

```sh
python3 scripts/build_experiment_atlas.py --snapshot results/libero_YYYYMMDD
```

The snapshot must have the same `index.json`, `runs.json`, `evaluation_points.csv`, and `frozen_evaluations.json` schema as the existing exporter. The builder switches the numerical catalog to that snapshot; use a cumulative export to retain older runs. The OGPO adapter currently reads the October 31 report's CSV and evidence separately, because those evaluations use a different schema and capture time. To ingest a newer OGPO report, update the `report` path in the builder and preserve its schema and BC-offset conventions. New result formats should get explicit adapters rather than guessed step units.

Refresh the browser after rebuilding. The server disables browser caching so updates are visible without restarting it.

## Add understanding or branch an analysis

Edit `notes.json` and rebuild. Each note has `title`, `kind` (for example `Observation` or `Hypothesis to test`), `body`, and a repository-relative URL in `source`. Keep evidence and interpretation distinct.

For a deeper analysis, add another HTML page under `docs/`, then link it from a note's `source`. The shared server will serve that page too. Direct experiment links use `#experiment=<URL-encoded group name>`; selecting a group updates the URL for sharing.

## Files

- `index.html`, `style.css`, `app.js`: page presentation and interaction.
- `notes.json`: durable, editable research notes.
- `data.json`: generated portable page data; rebuild instead of editing directly.
- `scripts/build_experiment_atlas.py`: source adapters and catalog generation.
- `scripts/serve_experiment_atlas.py`: local HTTP server and root redirect.

All linked archive files retain their original relative repository locations. No checkpoint weights or raw videos are available in this export.

## Out-of-the-box LIBERO comparison

Open [π₀ and π₀.₅ baselines](baselines.html) for all 130 tasks, suite means, original source links, and a CSV download. Regenerate with:

```sh
python3 scripts/build_libero_baselines.py
```

The builder selects the full π₀.₅ sweep, the π₀ family sweep, and the π₀ LIBERO-90 probe plus its separate task-55 baseline. It excludes retries and training evaluations, validates task pairing and protocol, and preserves the LIBERO-10 horizon distinction (400 steps for π₀ versus 600 for π₀.₅; both use 400 on other suites). `--snapshot` accepts a replacement export containing those source groups. `baselines.html` and `baselines.csv` are generated files.

The baseline page also annotates documented suite training coverage for both π checkpoints and the general Robometer-4B checkpoint used in the local reward-learning report. The suite summary, each task suite, and CSV carry these annotations. Source links and the distinction between suite membership and exact evaluation-trajectory overlap are included on the page.

## Success visitation matching

Open [the SVM subpage](success-visitation.html) for the loss, discriminator inputs, comparison to the sibling MPAIL implementation, and task-2 sparse baseline curves.

`python3 scripts/run_success_visitation.py` prepares the one-seed 100k spec and checks inputs. Supply `--demo-file` if needed, then add `--launch` to train fresh BC on 30 demonstrations for 5,000 updates, select the first ODE checkpoint in [25%,50%] (otherwise the smallest positive), and run SVM RL for 100k physical steps. Each BC evaluation uses 25 episodes per sampler every 500 updates; early stopping is disabled. Historical sparse curves use a different 50-demo BC-2000 initialization, which is labeled on the page. The launch monitor updates `svm-run.json` every 15 seconds; the page refreshes every 30 seconds. This monitor and the HTTP server must stay running during training. The classifier uses the paper's 0/1 outcome, while the historical sparse baseline used −1/0; the subpage labels this comparison caveat.

The SVM subpage renders LaTeX using locally vendored KaTeX 0.16.22 (MIT), including its fonts. It does not need an external CDN or a network connection for math rendering.

The SVM page uses one compact chart with a BC/RL phase selector. BC plots ODE and SDE together against optimizer training steps; RL retains physical steps per environment. Detailed evaluation values are collapsed beneath the chart.

## Additional SVM RL seeds and uncertainty

`python3 scripts/run_svm_replicates.py --launch` runs fresh seeds101–104 from the same selected BC checkpoint: GPU0 runs101 then103, GPU1 runs102 then104. It refuses existing outputs, records per-run launch metadata, and does not retrain BC. `svm-replicates.json` refreshes during the queue and includes the original seed100.

The default RL plot shows reward-wise seed means and two-sided95% Student-t intervals at each logged checkpoint, separately for ODE/SDE. Seed rates get equal weight. Intervals require at least two independent training seeds; episodes do not count as seeds. The displayed band is clipped to[0,1], while raw bounds and contributing seed IDs remain in the table/data. The available-seed cohort can vary during training and where historical runs stopped. All SVM seeds share one BC initialization; historical sparse uses different BC weights.

## November 23 SVM audit correction

The five completed seeds100–104 did not activate the discriminator: the production
EpisodeMonitor wrapper hid `cfg` from the runner. All five lack `svm.csv` and
`svm_discriminator.msgpack`. Their curves are retained and labeled as unshaped
0/1 sparse-reward controls. See `svm-audit.json`; the original CSVs and run specs
are unchanged. They are not evidence of SVM performance.

The corrected runner reads `env.unwrapped.cfg`, initializes the discriminator
explicitly, updates it every 1,000 physical steps (including warm-up),
and recomputes sampled replay rewards with current parameters. Future specs use
`svm.update_every: 1000` and `svm.gradient_steps: 32`, retaining the original
paper Appendix D, Table 5 schedule. A corrected run must use a new output directory and new
experiment identity, never append to these historical controls.

## SVM rollout logging

New SVM runs write `training_episodes.csv` under their log directory. Each row is
one **completed** training episode: physical end step, integer completed-episode
count, window size `min(20, completed episodes)`, window successes, and success
rate. The training view and optional overlay on the RL evaluation plot use a step
curve, holding the previous rate during an
ongoing episode. Historical runs without these logs are not reconstructed.

Every 10k physical steps, including during the 20k RL warm-up, each ODE/SDE
25-episode evaluation saves an `eval_artifacts/step_XXXXXX/<mode>/manifest.json`,
a single `reward_overlay.png`, and `trajectories.npz`. Measured world-space EEF
positions are projected with the agent-view camera calibration and the exact
RGB rotation/resize used by the policy. All 25 EEF paths are overlaid on the
first episode's initial image; differing reset layouts are documented. Colors
show the learned process reward (lambda times clipped logit), excluding the
sparse outcome. The NPZ also preserves total shaped rewards and executed actions.
The discriminator is read only during evaluation, with no replay insertion.

The configured recording seed saves four MP4s **from the 25 scored episodes**
per sampler; no extra rollouts are added. Recording defaults to literal seed0 as requested. The historical campaign used
seeds100–104; `--record-seed 100` explicitly selects its first run instead. Other
seeds still save trajectory overlays. `eval_logging` in the run spec records the choice.

The server's `/experiment-artifacts/` route serves only PNG, MP4, NPZ and JSON
files below `../experiment_results/**/eval_artifacts/`. Checkpoints and datasets
are excluded. Run the launch monitor to publish new media and training rows to
the atlas; reload/restart the HTTP server once after this server-code update.

## Corrected two-seed campaign

`python3 scripts/run_svm_pair.py --launch` starts fresh seeds0 and1 on GPUs0 and1
from the verified selected30-demo BC checkpoint. Each has100k physical steps,
paper Table5 discriminator settings,25 episodes per sampler every10k steps,
rolling20-episode training statistics, trajectory overlays, and seed0 videos.
It never includes old inactive runs in the SVM cohort. The monitor verifies the
activation marker once online collection begins and requires discriminator
updates before accepting a completed run. See `svm_verified_20261123_seeds01`
under the external `experiment_results/ogpo_libero` root for status and logs.

## Stable evaluation media viewer

Trajectory rewards use one overlay image with seed/sampler selectors and an
available-checkpoint slider. The chosen physical step stays selected when new
evaluation manifests arrive. Recordings have independent seed/checkpoint controls.
The live30-second poll keeps the same video DOM elements and source URLs; it does
not reset playback, volume, speed, or open episodes. Players are replaced only
when the user selects different recording URLs. The overlay slider never changes
the video selection. Reload the browser once after deploying this JavaScript edit.

## Successful / unsuccessful trajectory filter

The single overlay viewer includes All / Successful only / Unsuccessful only.
Filtering uses the simulator outcome of the entire evaluation episode, not the
sign of its learned reward. All variants use the same background, projection,
and fixed reward color scale. Empty subsets are labeled explicitly. The filter
and checkpoint stay selected through live polling and do not change video players.

New evaluation logging saves `background.png` and the three overlay PNGs before
publishing its manifest. For runs already in flight when this feature was added,
`scripts/backfill_svm_overlay_filters.py --watch` renders subsets from the saved
NPZ rewards/paths and manifest outcomes. It does not rerun policies or reward
models, and exits after final artifacts are covered. Where the original RGB was
not saved, it regenerates the known official reset state with the original seed
and settle actions, checking against initial recorded frames where available;
background provenance is recorded in the manifest. Raw trajectories, outcomes,
rewards, and videos remain unchanged.

### Optional inactive-SVM sparse controls

The SVM plot has a checkbox for the earlier task-2 seeds 100–104, labeled as sparse controls with SVM inactive. `svm-inactive-controls.json` contains their original ODE/SDE evaluation CSV points and separate Student-t summaries, exported using `points` and `summarize` from `scripts/run_svm_replicates.py`. These share the selected 30-demo BC start; the older historical sparse baseline uses 50 demonstrations. Controls load only when requested and never enter corrected SVM statistics. The checkbox applies to RL mean/individual views, preserves its state during polling, and is disabled for BC/training views.

### Fixed-lag checkpoint correlations

Run `python3 scripts/build_svm_lag_correlation.py` to regenerate the saved ODE/SDE
SVG figures and `lag-correlation.json` after evaluation data changes. These are
snapshot plots, separate from live training/media refresh. The sampler selector
changes only the correlation figure and table.

At each checkpoint T, correlate success rates across matched seeds at T−10k and
T. Each coefficient uses exactly two checkpoints, with no trailing or cumulative
window. The x-axis is T. The first possible SVM point is 10k; sparse logs skip
10k, so their first possible point is 30k. Require at least two matched seeds and
nonzero variance at both checkpoints; otherwise leave the coefficient undefined.
JSON records all paired seed rates and counts. Green is SVM, orange historical
sparse, and purple earlier inactive-SVM sparse controls. Individual-seed
correlations are undefined from a single pair of scalar success rates. These
curves describe across-seed association, not within-seed autocorrelation; few-seed
estimates are exploratory.
