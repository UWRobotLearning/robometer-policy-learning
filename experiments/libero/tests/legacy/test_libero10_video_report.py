"""Check matched-state LIBERO-10 video comparison rendering."""

import json

from experiments.libero.legacy.libero10_video_report import task_section
from experiments.libero.legacy.libero10_600_video_audit import choose_video_states


def test_video_page_renders_four_per_group_with_ground_truth_and_predictions(tmp_path):
    root = tmp_path / "pi0_libero10_video_audit_20261015" / "task06"
    root_600 = tmp_path / "pi0_libero10_video_audit_600_20261018" / "task06"
    states = (26, 25, 29, 35)
    demos = [{"demo_id": index, "length": 100, "success": True,
              "reward_events": [{"step": 99, "reward": 1}]} for index in range(4)]
    policy = [{"init_state_id": state, "length": 100, "success": index < 2}
              for index, state in enumerate(states)]
    def save(path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    save(root / "demos/manifest.json", {"demonstrations": demos})
    save(root / "policy/episodes.json", policy)
    save(root / "final_policies/selected_states.json", {"init_state_ids": states})
    series = [{"step": 0, "video_time": 0, "progress": .1,
               "success_probability": .2},
              {"step": 100, "video_time": 2.5, "progress": .9,
               "success_probability": .8}]
    for index in range(4):
        save(root / "demos/scores" / f"demo_{index}.json", series)
    for arm in ("policy", "sparse", "robometer"):
        directory = root_600 / arm
        save(directory / "episodes.json", policy)
        save(directory / "selection.json", {
            "selected_state_ids": states, "selected_successes": states[:2],
            "selected_failures": states[2:], "available_successes": 2,
            "available_failures": 2, "selection_seed": 123,
        })
        for row in policy:
            save(directory / "scores" / f'episode_{row["init_state_id"]}.json', series)
    page = task_section(tmp_path, 6)
    assert page.count('class="episode"') == 16
    assert page.count('class="score-plot"') == 16
    assert page.count('class="ground-truth"') == 16
    assert page.count("Sparse DSRL") == 4
    assert page.count("Robometer DSRL") == 4
    assert page.count("Frozen π₀ success") == 2
    assert page.count("Frozen π₀ failure") == 2
    assert "/videos/06/policy600/episode_26.mp4" in page
    assert "/videos/06/sparse600/episode_26.mp4" in page
    assert "/videos/06/robometer600/episode_26.mp4" in page
    assert "600-step horizon" in page


def test_random_video_selection_uses_all_25_outcomes_and_reports_shortage():
    records = [{"init_state_id": state, "success": state < 35}
               for state in range(25, 50)]
    first = choose_video_states(records, 42)
    assert first == choose_video_states(list(reversed(records)), 42)
    assert len(first["selected_successes"]) == 2
    assert len(first["selected_failures"]) == 2
    assert len(set(first["selected_state_ids"])) == 4
    assert first["episodes_evaluated"] == 25

    rare = [{"init_state_id": state, "success": state == 25}
            for state in range(25, 50)]
    selected = choose_video_states(rare, 42)
    assert selected["available_successes"] == 1
    assert len(selected["selected_successes"]) == 1
    assert len(selected["selected_failures"]) == 3
