import cv2
import numpy as np
import pytest

from step4b_d1_diagnostic import (
    D1_METRIC_INTERVAL_STEPS,
    D1_STEP_MULTIPLIER,
    EXPECTED_ORIGINAL_BATCH_IDS,
    EXPECTED_ORIGINAL_STEPS,
    analyze_prediction_batch,
    plot_prediction_comparisons,
)


def make_batch(offset=0.0):
    sample_ids = ["S11_06_10", "S11_06_11", "S11_06_13", "S11_06_14"]
    target = np.zeros((4, 14, 3), dtype=np.float64)
    prediction = np.zeros((4, 14, 3), dtype=np.float64)
    for sample_index, sample_id in enumerate(sample_ids):
        for joint in range(14):
            target[sample_index, joint] = np.array(
                [sample_index + offset, joint * 0.01, joint * 0.001 + sample_index * 0.1]
            )
            prediction[sample_index, joint] = target[sample_index, joint] + np.array(
                [0.1, -0.02, 0.0]
            )
    return sample_ids, target, prediction


def test_d1_budget_and_fixed_tiny_batch_are_frozen():
    assert D1_STEP_MULTIPLIER == 4
    assert EXPECTED_ORIGINAL_STEPS == 200
    assert EXPECTED_ORIGINAL_STEPS * D1_STEP_MULTIPLIER == 800
    assert D1_METRIC_INTERVAL_STEPS == 25
    assert 200 % D1_METRIC_INTERVAL_STEPS == 0
    assert 800 % D1_METRIC_INTERVAL_STEPS == 0
    assert EXPECTED_ORIGINAL_BATCH_IDS == [
        "S11_06_10",
        "S11_06_11",
        "S11_06_13",
        "S11_06_14",
        "S11_06_16",
        "S11_06_17",
        "S11_06_19",
        "S11_06_20",
    ]


def test_prediction_analysis_reports_axis_joint_sample_and_mean_pose_evidence():
    sample_ids, target, prediction = make_batch()
    train_mean_pose = np.mean(target + np.array([1.0, 0.0, 0.0]), axis=0)

    summary, arrays = analyze_prediction_batch(prediction, target, sample_ids, train_mean_pose)

    expected_distance = float(np.linalg.norm(np.array([0.1, -0.02, 0.0])))
    assert summary["mpjpe_stored_units"] == pytest.approx(expected_distance)
    assert arrays["joint_errors"].shape == (4, 14)
    assert summary["per_joint_mpjpe_stored_units"] == pytest.approx([expected_distance] * 14)
    assert [sample["sample_id"] for sample in summary["per_sample"]] == sample_ids
    assert [sample["mpjpe_stored_units"] for sample in summary["per_sample"]] == pytest.approx(
        [expected_distance] * 4
    )
    assert summary["per_coordinate_index_mae_stored_units"] == pytest.approx([0.1, 0.02, 0.0])
    assert summary["per_coordinate_index_rmse_stored_units"] == pytest.approx([0.1, 0.02, 0.0])
    assert summary["prediction_coordinate_ranges_and_variance"]["overall"]["min"] < 0.0
    assert summary["target_coordinate_ranges_and_variance"]["overall"]["variance"] > 0.0

    comparison = summary["mean_pose_and_collapsed_output_comparison"]
    assert comparison["fixed_batch_mean_pose_mpjpe_baseline_stored_units"] > 0.0
    assert comparison["model_to_fixed_batch_mean_pose_mpjpe_stored_units"] > 0.0
    assert comparison["train_dataset_mean_pose_mpjpe_baseline_stored_units"] > 0.0
    assert len(comparison["prediction_to_target_variance_ratio_per_coordinate_index"]) == 3
    assert comparison["prediction_mean_distance_to_own_mean_mpjpe_stored_units"] > 0.0


def test_prediction_analysis_accepts_q1_model_shapes():
    sample_ids, target, prediction = make_batch()
    summary, arrays = analyze_prediction_batch(
        prediction[:, None, :, :], target[:, None, :, :], sample_ids
    )

    assert summary["mpjpe_stored_units"] == pytest.approx(0.1 + 0.0, abs=0.2)
    assert arrays["prediction"].shape == (4, 14, 3)
    assert arrays["target"].shape == (4, 14, 3)


def test_prediction_plots_are_written_for_the_fixed_sample_set(tmp_path):
    sample_ids, target, prediction = make_batch()
    paths = plot_prediction_comparisons(
        sample_ids,
        target,
        {0: prediction, 200: prediction * 0.5, 800: prediction * 0.25},
        tmp_path,
    )

    assert len(paths) == len(sample_ids)
    for path in paths:
        image = cv2.imread(str(path))
        assert image is not None
        assert image.shape[0] > 500
        assert image.shape[1] > 1000
