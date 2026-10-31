# Central experiment workspace

Use `robometer-policy-learning` as the parent repository. Algorithm and reward-model
implementations retain separate Git histories and environments, with exact commits pinned
as submodules. Experiment outputs remain outside the code checkouts.

| Directory | Repository | Pinned experiment version |
|---|---|---|
| `OGPO/` | [UWRobotLearning/OGPO_public](https://github.com/UWRobotLearning/OGPO_public) | `b113b2c70b314956f2b3fdb51a10966b5fcb7b11` on `rss27` |
| `robometer/` | [UWRobotLearning/robometer](https://github.com/UWRobotLearning/robometer) | `3ad5da10038717e028aa6ab146f40bfc74bfecd5` on `fixes/reward-decoder` |
| `LIBERO/` | `nakamotoo/LIBERO` | Existing parent-repository pin, unchanged |
| `dsrl_openpi/` | `jesbu1/dsrl_openpi` | Existing parent-repository pin, unchanged |

The OGPO fork preserves upstream history from `simchowitzlabpublic/OGPO_public` and the
local LIBERO integration previously held in the sibling `OGPO_public` checkout. It includes
the simulator/reward bridge, sparse and Robometer reward variants, separate native π₀.₅
LoRA experiments, local logging, rolling checkpoint logic, and four test modules. It is
an experimental integration, not an assertion of exact reproduction of upstream OGPO.
The Robometer pin includes the existing decoder/checkpoint-loading fix.

## Clone and initialize

```sh
git clone --branch rss27 --recurse-submodules \
  git@github.com:UWRobotLearning/robometer-policy-learning.git
cd robometer-policy-learning
git submodule status
```

For an existing checkout of this branch:

```sh
git submodule sync --recursive
git submodule update --init --recursive
```

Git checks out the recorded commits, not the latest upstream branch tips. Do not use
`git submodule update --remote` to reproduce a run. When updating OGPO, push its commit to
the UW fork first, then commit the updated `OGPO` pointer in the parent repository.
For all subsequent Robometer development, edit algorithms inside this parent checkout:
`OGPO/` for OGPO and `dsrl_openpi/` for DSRL. Use `rss27` in both the parent
and the UW OGPO fork. The OGPO submodule tracks `rss27` in `.gitmodules`, while
the parent still pins an exact commit for reproducibility. After initializing a clone,
attach the OGPO checkout to its development branch:

```sh
git -C OGPO switch rss27
# Edit, test, and commit inside OGPO.
git -C OGPO push origin rss27
git add OGPO
git commit -m "Pin tested OGPO revision"
git push origin rss27
```

Fresh clones use `origin`; in the existing sibling OGPO checkout and parent checkout,
the UW remote is named `uw` instead. Both UW repositories keep only `main` and
`rss27` on the remote. The prior integration and results branch histories are retained
in `rss27`. Push algorithm commits before pushing a parent commit that references them.

## Environments and existing jobs

Keep OGPO's JAX environment separate from the parent simulator and Robometer environments.
For a new OGPO MLP environment, use uv from inside the submodule:

```sh
cd OGPO
uv venv .venv
uv sync --extra cuda12
```

Native π₀.₅ LoRA uses a separate `.venv-pi05` with OpenPI-compatible dependencies; do not
sync it against the MLP environment. See [OGPO's experiment notes](../OGPO/scripts/LIBERO.md)
for the distinct methods, environments, reward conventions, and run commands. Task specs
must identify the parent checkout with `policy_repo`, its simulator interpreter with
`sim_python`, and the appropriate checkpoint/tokenizer/data paths. Existing specs and some
launcher defaults reference this experiment machine's `/data/rohan/Reward-Modeling` paths;
cloning code does not provision the datasets, pretrained models, or environments.

The original sibling `../OGPO_public` checkout and its environments are deliberately left
in place and checked out on `rss27`. Existing runs/queues may still use them. All new
algorithm development belongs in the parent submodules after setting up their environments;
no running process is migrated or restarted by this repository reorganization.

## Results

Full run data remains under `/data/rohan/Reward-Modeling`, including `ogpo_libero/`.
The local `local_results` symlink points there; it is not a portable data archive.
[The numerical results on rss27](../results/README.md)
contain the exported SAC/DSRL data and OGPO numeric logs without weights or media.
OGPO ODE/SDE evaluations and step conventions remain distinct in that export.

Validation of the pinned OGPO integration: all 10 existing unit tests passed using
`CUDA_VISIBLE_DEVICES='' JAX_PLATFORMS=cpu`, with no training or GPU smoke run launched.
