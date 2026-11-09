import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from step4b_config import STEP4B_CONFIG, step4b_config_for_json
from step4b_single_person_pilot import compute_mean_pose, evaluate_constant_pose


def make_person(offset_x=0.0):
    joints = np.arange(14, dtype=np.float64)
    return np.column_stack(
        (
            joints * 0.02 + offset_x,
            np.sin(joints * 0.3) * 0.1,
            (joints % 4) * 0.03,
        )
    )


def test_step4b_config_is_fixed_and_json_serializable():
    config = step4b_config_for_json()

    assert config["seed"] == 0
    assert config["subset"]["train_group_count"] == 8
    assert config["subset"]["development_group_count"] == 4
    assert config["subset"]["train_max_frames_per_group"] == 256
    assert config["subset"]["development_max_frames_per_group"] == 128
    assert config["model"]["q"] == 1
    assert config["model"]["keypoints"] == 14
    assert config["model"]["train_from_scratch"] is True
    assert config["model"]["checkpoint_format"] == "state_dict"
    assert config["training"]["epochs"] == 10
    assert config["training"]["train_batch_size"] == 16
    assert config["overfit"]["steps"] == 200
    assert config["metrics"]["stored_unit_to_mm"] == 1000.0
    assert config["metrics"]["root_index"] == 0
    assert config["metrics"]["pck_scale_joint_indices"] == [6, 4]
    assert config["label_access"]["official_test_used"] is False
    assert config == json.loads(json.dumps(STEP4B_CONFIG))


def test_constant_pose_evaluation_uses_q1_and_reports_set_metrics():
    config = step4b_config_for_json()
    gt_person = make_person()
    constant_pose = gt_person + np.array([0.1, 0.0, 0.0])

    result = evaluate_constant_pose([gt_person[None, :, :]], constant_pose, config, presence_score=1.0)
    overall = result["overall"]

    assert overall["matched_people"] == 1
    assert overall["matched_mpjpe_mm"] == pytest.approx(100.0)
    assert overall["matched_pampjpe_mm"] == pytest.approx(0.0, abs=1e-9)
    assert overall["root_error_mm"] == pytest.approx(100.0)
    assert overall["exact_count_accuracy_percent"] == pytest.approx(100.0)
    assert overall["missed_people"] == 0
    assert overall["duplicate_people"] == 0
    assert overall["spurious_predictions"] == 0
    assert overall["stored_unit_to_mm"] == 1000.0


def test_compute_mean_pose_uses_only_supplied_labelled_samples(tmp_path):
    poses = [make_person(offset_x=0.0), make_person(offset_x=0.2), make_person(offset_x=0.4)]
    paths = []
    for index, pose in enumerate(poses):
        path = tmp_path / f"pose_{index}.npy"
        np.save(path, pose[None, :, :].astype(np.float32))
        paths.append(path)
    dataset = SimpleNamespace(
        samples=[{"keypoint_path": path} for path in paths],
        read_pose=lambda path: torch.from_numpy(np.load(path)),
    )

    mean_pose, source_count = compute_mean_pose(dataset)

    assert source_count == 3
    np.testing.assert_allclose(mean_pose, np.mean(np.stack(poses, axis=0), axis=0), rtol=1e-6, atol=1e-7)
