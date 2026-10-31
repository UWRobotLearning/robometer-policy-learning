"""Checks for the four-suite frozen-policy sweep and live result accounting."""

import json

from experiments.libero.legacy.pi0_family_eval import ROOT, SUITES, assignments
from experiments.libero.legacy.pi0_family_report import (
    RBM_DATASETS, REWARD_PROVENANCE, task_result, training_coverage,
)
from experiments.libero.legacy.libero10_video_report import audit_training_coverage, render as render_video_audit


def test_gpu_assignments_cover_each_task_once():
    first, second = assignments(0), assignments(1)
    assert len(first) == len(second) == 20
    assert len(set(first + second)) == 40
    assert {suite for suite, _, _ in first + second} == set(SUITES)


def test_partial_task_is_visible_but_not_complete(tmp_path):
    folder = tmp_path / "task"
    folder.mkdir()
    (folder / "status.json").write_text(json.dumps({"status": "running"}))
    (folder / "episodes.jsonl").write_text('{"success": true}\n{"success": false}\n')
    result = task_result(folder)
    assert result == {"status": "running", "successes": 1, "episodes": 2, "ci": None}


def test_training_labels_require_the_configured_checkpoints_and_datasets(tmp_path):
    checkpoint = tmp_path / "pi0_libero"
    asset = checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    asset.parent.mkdir(parents=True)
    asset.write_text("{}")
    manifest = tmp_path / ROOT.name / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"checkpoint": str(checkpoint)}))
    provenance = tmp_path / REWARD_PROVENANCE
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({
        "checkpoint": "robometer/Robometer-4B@revision",
        "training_datasets": ["rbm-1m-id"],
        "expanded_training_datasets": list(RBM_DATASETS.values())[:-1],
    }))
    coverage = training_coverage(tmp_path)
    assert coverage["policy_known"] and coverage["reward_known"]
    assert RBM_DATASETS["libero_spatial"] in coverage["reward_datasets"]
    assert RBM_DATASETS["libero_10"] not in coverage["reward_datasets"]


def test_libero10_audit_labels_require_matching_checkpoint_and_dataset(tmp_path):
    checkpoint = tmp_path / "pi0_libero"
    asset = checkpoint / "assets/physical-intelligence/libero/norm_stats.json"
    asset.parent.mkdir(parents=True)
    asset.write_text("{}")
    manifest = tmp_path / ROOT.name / "manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(json.dumps({"checkpoint": str(checkpoint)}))
    provenance = tmp_path / REWARD_PROVENANCE
    provenance.parent.mkdir(parents=True)
    provenance.write_text(json.dumps({
        "checkpoint": "robometer/Robometer-4B@revision",
        "training_datasets": ["rbm-1m-id"],
        "expanded_training_datasets": [RBM_DATASETS["libero_10"]],
    }))
    audit = tmp_path / "pi0_libero10_video_audit_20261015" / "task06"
    audit.mkdir(parents=True)
    config = {"pi0_checkpoint": str(checkpoint),
              "reward_checkpoint": "robometer/Robometer-4B@revision"}
    (audit / "config.json").write_text(json.dumps(config))
    assert audit_training_coverage(tmp_path, 6)["policy"]
    assert audit_training_coverage(tmp_path, 6)["reward"]
    page = render_video_audit(tmp_path).decode()
    assert "Yes — LIBERO-10 suite" in page
    config["reward_checkpoint"] = "robometer/Robometer-4B@different"
    (audit / "config.json").write_text(json.dumps(config))
    assert not audit_training_coverage(tmp_path, 6)["reward"]
