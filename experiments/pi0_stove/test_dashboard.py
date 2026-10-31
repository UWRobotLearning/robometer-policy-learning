"""Statistical checks for the live multi-task results page."""

from experiments.pi0_stove.dashboard import (
    aggregate, baseline_comparisons, checkpoint_comparisons, checkpoint_training_note, frozen_pi0_baseline,
    frozen_pi05_run_baseline, load_task,
    paired_comparison, plot, sparse_baseline_comparisons, task_roots,
)
import json

from experiments.reward_audit.config import task_map


def test_curve_confidence_band_needs_multiple_seeds_and_stays_in_range():
    curves = [[{"steps": 0, "rate": 0.0}, {"steps": 100, "rate": 0.6}],
              [{"steps": 0, "rate": 1.0}]]
    points = aggregate(curves)
    assert points[0]["n"] == 2
    assert points[0]["ci_low"] == 0
    assert points[0]["ci_high"] == 1
    assert points[1]["n"] == 1
    assert points[1]["ci_low"] is None
    assert points[1]["ci_high"] is None


def test_final_test_excludes_incomplete_pairs_and_uses_exact_p_value():
    def run(rate, status, steps):
        return {"status": status, "evaluations": [{"rate": rate, "steps": steps}]}

    task = {"seeds": [100, 101, 102, 103], "max_steps": 100000,
            "runs": {"sparse": {seed: run(.8, "complete", 100000)
                                for seed in [100, 101, 102, 103]},
                     "robometer": {100: run(.2, "complete", 100000),
                                   101: run(.3, "complete", 100000),
                                   102: run(.1, "running", 50000),
                                   103: run(.1, "complete", 99000)}}}
    result = paired_comparison(task)
    assert result["n"] == 2
    assert abs(result["mean"] - .55) < 1e-10
    assert result["p"] == .5


def test_five_favorable_pairs_are_not_significant_at_five_percent():
    task = {"seeds": list(range(5)), "max_steps": 100,
            "runs": {kind: {} for kind in ("sparse", "robometer")}}
    for seed in task["seeds"]:
        for kind, rate in (("sparse", .8), ("robometer", .2)):
            task["runs"][kind][seed] = {"status": "complete",
                                         "evaluations": [{"steps": 100, "rate": rate}]}
    result = paired_comparison(task)
    assert result["p"] == .0625


def test_checkpoint_stars_require_significant_matched_pairs():
    task = {"id": 38, "seeds": list(range(6)), "max_steps": 100,
            "runs": {kind: {} for kind in ("sparse", "robometer")}}
    for seed in task["seeds"]:
        for kind, rate in (("sparse", .8), ("robometer", .2)):
            evaluations = [{"steps": 0, "rate": rate}]
            if not (kind == "robometer" and seed == 5):
                evaluations.append({"steps": 50, "rate": rate})
            task["runs"][kind][seed] = {"status": "running", "evaluations": evaluations}
    comparisons = checkpoint_comparisons(task)
    assert [(item["n"], item["p"]) for item in comparisons] == [(6, .03125), (5, .0625)]
    assert plot(task).count('class="significance-star reward-comparison-star"') == 1


def test_plot_displays_all_three_colored_stars_at_one_evaluation():
    seeds = list(range(100, 108))
    states = {state: state < 35 for state in range(25, 50)}
    task = {"id": 8, "suite": "libero_10", "algorithm": "direct DSRL",
            "seeds": seeds, "max_steps": 100000,
            "baseline": {"rate": .4, "successes": 10, "episodes": 25,
                         "ci_low": .23, "ci_high": .59},
            "baseline_states": states, "runs": {"sparse": {}, "robometer": {}}}
    for seed in seeds:
        task["runs"]["sparse"][seed] = {
            "evaluations": [{"steps": 5000, "rate": 1.0}],
            "state_evaluations": {5000: {state: True for state in states}}}
        task["runs"]["robometer"][seed] = {
            "evaluations": [{"steps": 5000, "rate": 0.0}],
            "state_evaluations": {5000: {state: False for state in states}}}
    for kind in ("sparse", "robometer"):
        comparisons = baseline_comparisons(task, kind)
        assert len(comparisons) == 1
        assert comparisons[0]["seed_p"] == .0078125
        assert comparisons[0]["state_p"] < .05
    svg = plot(task)
    assert svg.count('class="significance-star sparse-baseline-star"') == 1
    assert svg.count('class="significance-star robometer-baseline-star"') == 1
    assert svg.count('class="significance-star reward-comparison-star"') == 1


def test_task08_baseline_star_requires_state_level_significance():
    seeds = list(range(100, 108))
    states = {state: False for state in range(25, 50)}
    task = {"id": 8, "suite": "libero_10", "algorithm": "direct DSRL",
            "seeds": seeds, "max_steps": 100000,
            "baseline": {"rate": 0.0, "successes": 0, "episodes": 25,
                         "ci_low": 0.0, "ci_high": .13},
            "baseline_states": states, "runs": {"sparse": {}, "robometer": {}}}
    for seed in seeds:
        successes = {state: state == 25 for state in states}
        task["runs"]["sparse"][seed] = {
            "evaluations": [{"steps": 5000, "rate": .04}],
            "state_evaluations": {5000: successes}}
        task["runs"]["robometer"][seed] = {"evaluations": []}
    comparison = sparse_baseline_comparisons(task)[0]
    assert comparison["seed_p"] < .05
    assert comparison["state_p"] == 1.0
    assert 'class="significance-star sparse-baseline-star"' not in plot(task)


def test_training_note_uses_verified_checkpoint_identity():
    task = {"policy_checkpoints": {"/data/pi0_checkpoints/pi0_libero"},
            "reward_checkpoints": {"robometer/Robometer-4B@pinned"}}
    note = checkpoint_training_note(task)
    assert "π₀ policy (pi0_libero): <b>No</b>" in note
    assert "Robometer reward model (Robometer-4B): <b>No</b>" in note
    task["reward_checkpoints"] = {"jesbu1/robometer-4b-fft-libero"}
    assert "Unverified" in checkpoint_training_note(task)


def test_libero10_dsrl_manifest_appears_as_its_own_suite(tmp_path):
    root = tmp_path / "pi0_libero10_task06_dsrl_5seeds_20261015"
    root.mkdir()
    (root / "manifest.json").write_text(json.dumps({
        "suite": "libero_10", "task_id": 6, "task": "LIVING_ROOM_SCENE6_example",
        "seeds": [100, 101, 102, 103, 104], "algorithm": "direct DSRL",
        "steps_per_run": 100000,
    }))
    assert task_roots(tmp_path) == [root]
    task = load_task(root)
    assert task["suite"] == "libero_10" and task["id"] == 6
    assert len(task["runs"]["sparse"]) == 5


def test_plot_uses_only_matching_frozen_checkpoint_baseline(tmp_path):
    name = task_map()["libero_10"][6]
    folder = tmp_path / "pi0_libero_families_20261014" / "libero_10" / f"06_{name}"
    folder.mkdir(parents=True)
    (folder / "status.json").write_text(json.dumps({"status": "complete"}))
    config = {"suite": "libero_10", "task_id": 6, "checkpoint": "/models/pi0_libero",
              "eval_state_ids": list(range(25, 50)), "horizon": 400}
    (folder / "config.json").write_text(json.dumps(config))
    (folder / "summary.json").write_text(json.dumps({
        "successes": 13, "episodes": 25, "success_rate": .52,
        "wilson_ci95": [.335, .7],
    }))
    task = {"id": 6, "suite": "libero_10", "name": name.replace("_", " "),
            "policy_checkpoints": {"/models/pi0_libero"}, "max_steps": 100000,
            "seeds": [], "runs": {"sparse": {}, "robometer": {}}}
    task["baseline"] = frozen_pi0_baseline(task, tmp_path)
    assert task["baseline"] == {"rate": .52, "successes": 13, "episodes": 25,
                                "ci_low": .335, "ci_high": .7,
                                "url": "/pi0-libero-families-eval/#libero_10"}
    assert 'class="baseline-line"' in plot(task)
    assert 'class="baseline-ci"' in plot(task)
    assert 'π₀ 52%' in plot(task)
    config["checkpoint"] = "/models/other"
    (folder / "config.json").write_text(json.dumps(config))
    assert frozen_pi0_baseline(task, tmp_path) is None


def test_pi05_25_episode_step_zero_baseline_pools_both_arms_by_seed(tmp_path):
    root = tmp_path / 'pi05_libero90_task28_dsrl_3seeds_20261022'
    rates = {100: (14, 15), 101: (13, 14), 102: (16, 17)}
    task = {'id': 28, 'root': root, 'anchor': 'task-pi05-task28',
            'max_steps': 100000, 'eval_episodes': 25,
            'policy_checkpoints': {'/data/pi0_checkpoints/pi05_libero'},
            'seeds': list(rates), 'runs': {'sparse': {}, 'robometer': {}}}
    for seed, (sparse, robometer) in rates.items():
        for kind, successes in (('sparse', sparse), ('robometer', robometer)):
            task['runs'][kind][seed] = {'evaluations': [
                {'steps': 0, 'episodes': 25, 'successes': successes,
                 'rate': successes / 25}]}
    baseline = frozen_pi05_run_baseline(task)
    assert baseline['successes'] == 89
    assert baseline['episodes'] == 150
    assert baseline['rate'] == 89 / 150
    assert baseline['seeds'] == 3
    assert baseline['ci_low'] < baseline['rate'] < baseline['ci_high']
    task['baseline'] = baseline
    assert 'class="baseline-line"' in plot(task)
    assert 'class="baseline-ci"' in plot(task)


def test_dashboard_discovers_config_driven_scratch_experiments(tmp_path):
    from experiments.libero.config import resolve
    from experiments.libero.runner import initialize

    out = tmp_path / 'experiments' / 'new_experiment'
    out.mkdir(parents=True)
    spec = resolve({'base_model': 'scratch', 'suite': 'libero_goal', 'tasks': [7], 'rewards': ['sparse'],
                    'runtime': {'data_root': str(tmp_path)}})
    initialize(spec, out, False)
    roots = task_roots(tmp_path)
    assert len(roots) == 1
    task = load_task(roots[0])
    assert task['base_model'] == 'scratch'
    assert task['suite'] == 'libero_goal'
    assert task['baseline'] is None
    assert task['runs']['robometer'][100]['status'] == 'not selected'


def test_audit_links_match_task_checkpoint_and_training_experiment(tmp_path):
    from experiments.pi0_stove.dashboard import video_audit_url

    task = {'id': 29, 'suite': 'libero_90', 'root': tmp_path / 'training',
            'policy_checkpoints': {'/models/pi05_libero'}}
    def audit(name, checkpoint='/models/pi05_libero', actor=None):
        folder = tmp_path / 'video_audits' / name
        folder.mkdir(parents=True)
        (folder / 'manifest.json').write_text(json.dumps({'jobs': [{
            'config': {'task_id': 29, 'suite': 'libero_90', 'pi0_checkpoint': checkpoint},
            'actor_checkpoint': actor}]}))

    assert video_audit_url(task, tmp_path) is None
    audit('wrong-model', '/models/pi0_libero')
    audit('other-training', actor=str(tmp_path / 'other/checkpoints/final.pt'))
    assert video_audit_url(task, tmp_path) is None
    audit('frozen')
    assert video_audit_url(task, tmp_path) == '/video-audits/frozen/'
    audit('matching-training', actor=str(task['root'] / 'seed100/checkpoints/final.pt'))
    assert video_audit_url(task, tmp_path) == '/video-audits/matching-training/'
