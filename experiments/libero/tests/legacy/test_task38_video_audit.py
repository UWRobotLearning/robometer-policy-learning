"""Checks for rollout sampling and the linked video report."""

import json

import numpy as np

from experiments.libero.legacy.task38_video_audit import scoring_indices
from experiments.libero.legacy.task38_video_report import render


def test_score_samples_include_start_regular_steps_and_terminal():
    assert scoring_indices(np.array([0, 4, 8, 20, 24, 27])) == [0, 3, 5]


def test_report_selects_two_of_each_outcome_and_lists_all_ten(tmp_path):
    records = [{"init_state_id": 25 + index, "success": index % 2 == 0,
                "length": 100, "video": f"episode_{25 + index}.mp4"}
               for index in range(10)]
    (tmp_path / "episodes.json").write_text(json.dumps(records))
    (tmp_path / "scores").mkdir()
    (tmp_path / "scores" / "episode_25.json").write_text(json.dumps([
        {"step": 0, "video_time": 0, "progress": .2, "success_probability": .1},
        {"step": 100, "video_time": 2.5, "progress": .8, "success_probability": .7},
    ]))
    page = render(tmp_path).decode()
    assert page.count('class="episode"') == 4
    assert page.count('<video ') == 4
    assert page.count('>Video</a>') == 10
    assert 'Progress' in page and 'Predicted success' in page
    assert page.count('Simulator reward</span>') == 4
    assert 'data-reward-step="25"' not in page
    assert 'data-reward-step="100"' in page
    assert 'data-reward-step="-1"' in page
    assert '5/10 simulator successes' in page


def test_report_displays_two_dataset_demonstrations_with_scores(tmp_path):
    demos = tmp_path / "dataset_demos"
    (demos / "scores").mkdir(parents=True)
    (demos / "manifest.json").write_text(json.dumps({"demonstrations": [
        {"demo_id": 0, "length": 192, "success": True,
         "reward_events": [{"step": 191, "reward": 1}]},
        {"demo_id": 1, "length": 207, "success": True,
         "reward_events": [{"step": 206, "reward": 1}]},
    ]}))
    for demo_id in (0, 1):
        (demos / "scores" / f"demo_{demo_id}.json").write_text(json.dumps([
            {"step": 0, "video_time": 0, "progress": .1,
             "success_probability": .2},
        ]))
    page = render(tmp_path).decode()
    assert page.count('class="episode"') == 2
    assert 'Expert demonstration 0' in page
    assert 'Expert demonstration 1' in page
    assert '/task38-pi0-baseline/videos/demo_0.mp4' in page
    assert '/task38-pi0-baseline/videos/demo_1.mp4' in page
    assert 'data-reward-step="191"' in page
    assert 'data-reward-step="206"' in page
    assert page.count('Simulator reward</span>') == 2


def test_task55_report_uses_its_own_title_and_video_routes(tmp_path):
    (tmp_path / "episodes.json").write_text(json.dumps([
        {"init_state_id": 25 + index, "success": True, "length": 110}
        for index in range(10)
    ]))
    page = render(tmp_path, task_id=55).decode()
    assert 'Pick up the alphabet soup and put it in the tray' in page
    assert page.count('class="episode"') == 4
    assert '/task55-pi0-baseline/videos/episode_25.mp4' in page
    assert '/task38-pi0-baseline/videos/' not in page


def test_final_trained_policies_show_paired_videos_with_compact_plots(tmp_path):
    for arm, outcomes in (("sparse", (True, True)), ("robometer", (False, True))):
        folder = tmp_path / "final_policies" / f"{arm}_seed100"
        (folder / "scores").mkdir(parents=True)
        (folder / "episodes.json").write_text(json.dumps([
            {"init_state_id": 25 + index, "success": success, "length": 100}
            for index, success in enumerate(outcomes)
        ]))
        for state in (25, 26):
            (folder / "scores" / f"episode_{state}.json").write_text(json.dumps([
                {"step": 100, "video_time": 2.5, "progress": .5,
                 "success_probability": .5},
            ]))
    page = render(tmp_path, task_id=55).decode()
    assert page.count('class="episode"') == 4
    assert page.count('class="plot-pane"') == 4
    assert 'videos/sparse/episode_25.mp4' in page
    assert 'videos/robometer/episode_25.mp4' in page
    assert 'Sparse reward · state 25' in page
    assert 'Robometer reward · state 25' in page
