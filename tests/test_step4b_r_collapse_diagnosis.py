import json

import numpy as np
import pytest

from diagnose_step4b_r_collapse import (
    _assert_outputs_outside_baseline,
    activation_statistics,
    check_premises,
    evenly_spaced_indices,
    group_centroid_statistics,
    pattern_matches,
    prediction_spread,
    render_markdown,
    round_matches,
    single_point_summary,
    target_decomposition,
)


def test_constant_predictions_have_zero_spread_and_single_point():
    point = np.array([2.797, 1.784, 3.278])
    predictions = np.broadcast_to(point, (10, 1, 14, 3)).copy()

    spread = prediction_spread(predictions)
    summary = single_point_summary(predictions)

    assert spread["std_across_samples_max_over_joints_and_axes"] == pytest.approx(0.0, abs=1e-12)
    assert spread["mean_distance_of_joint_from_pose_centroid"] == pytest.approx(0.0, abs=1e-12)
    assert summary["is_single_point"] is True
    assert summary["common_point"] == pytest.approx(point.tolist())


def test_varied_predictions_are_not_a_single_point():
    rng = np.random.default_rng(0)
    predictions = rng.normal(size=(20, 14, 3))

    spread = prediction_spread(predictions)

    assert spread["std_across_samples_max_over_joints_and_axes"] > 0.5
    assert single_point_summary(predictions)["is_single_point"] is False


def test_prediction_spread_rejects_bad_shape():
    with pytest.raises(ValueError):
        prediction_spread(np.zeros((4, 14, 2)))


def test_activation_statistics_known_values():
    # (samples=2, joints=2, channels=1): values chosen so population stds are exact.
    activation = np.array([[[0.0], [4.0]], [[2.0], [4.0]]])

    stats = activation_statistics(activation)

    # axis 0 std: joint0 -> 1.0, joint1 -> 0.0 => mean 0.5
    assert stats["std_across_samples"] == pytest.approx(0.5)
    # axis 1 std: sample0 -> 2.0, sample1 -> 1.0 => mean 1.5
    assert stats["std_across_joints"] == pytest.approx(1.5)
    assert stats["fraction_exact_zeros"] == pytest.approx(0.25)
    assert stats["all_finite"] is True


def test_activation_statistics_pooled_feature_has_no_joint_std():
    stats = activation_statistics(np.arange(12, dtype=np.float64).reshape(4, 3))

    assert stats["std_across_joints"] is None
    assert stats["std_across_samples"] > 0


def test_activation_statistics_detects_dead_activation():
    dead = np.tile(np.array([1.5, -0.25, 0.0]), (8, 14, 1))

    stats = activation_statistics(dead)

    assert stats["std_across_samples"] == 0.0
    assert stats["std_across_joints"] == 0.0
    assert stats["fraction_exact_zeros"] == pytest.approx(1 / 3)


def test_activation_statistics_squeezes_single_person_axis_and_flags_nonfinite():
    pose = np.zeros((3, 1, 14, 3))
    pose[0, 0, 0, 0] = np.nan

    stats = activation_statistics(pose)

    assert stats["all_finite"] is False
    assert activation_statistics(np.ones((3, 1, 14, 3)))["std_across_joints"] == 0.0


def test_target_decomposition_translation_only_variation():
    # Every sample is the same skeleton shifted by a different translation.
    rng = np.random.default_rng(1)
    skeleton = rng.normal(size=(14, 3))
    shifts = rng.normal(size=(30, 1, 3))
    train = skeleton[None] + shifts[:20]
    development = skeleton[None] + shifts[20:]

    result = target_decomposition(train, development)

    # Skeleton shape never varies, so centroid-relative error is zero while the
    # absolute error equals the centroid distance to the train mean centroid.
    for split in ("train", "development"):
        assert result[split]["centroid_relative_mean_skeleton_mpjpe"] == pytest.approx(0.0, abs=1e-12)
        assert result[split]["absolute_mean_pose_mpjpe"] == pytest.approx(
            result[split]["centroid_distance_to_train_mean_centroid"], abs=1e-9
        )
    assert result["train"]["sample_count"] == 20
    assert result["development"]["sample_count"] == 10


def test_target_decomposition_uses_train_statistics_only():
    train = np.zeros((5, 14, 3))
    development = np.ones((4, 14, 3))

    result = target_decomposition(train, development)

    assert result["train"]["absolute_mean_pose_mpjpe"] == pytest.approx(0.0)
    assert result["development"]["absolute_mean_pose_mpjpe"] == pytest.approx(np.sqrt(3.0))
    assert result["development"]["centroid_relative_mean_skeleton_mpjpe"] == pytest.approx(0.0)
    assert result["train_mean_centroid"] == [0.0, 0.0, 0.0]


def test_group_centroid_statistics_groups_by_candidate_prefix():
    ids = ["S11_01_10", "S11_01_12", "S11_02_10", "S11_02_11"]
    poses = np.zeros((4, 14, 3))
    poses[1] += np.array([1.0, 0.0, 0.0])
    poses[3] += np.array([0.0, 2.0, 0.0])

    stats = group_centroid_statistics(ids, poses)

    assert sorted(stats) == ["S11_01", "S11_02"]
    assert stats["S11_01"]["sample_count"] == 2
    assert stats["S11_01"]["centroid_mean"] == pytest.approx([0.5, 0.0, 0.0])
    assert stats["S11_01"]["centroid_std_per_axis"] == pytest.approx([0.5, 0.0, 0.0])
    assert stats["S11_02"]["centroid_std_norm"] == pytest.approx(1.0)


def test_group_centroid_statistics_validates_inputs():
    with pytest.raises(ValueError):
        group_centroid_statistics(["S11_01_10"], np.zeros((2, 14, 3)))
    with pytest.raises(ValueError):
        group_centroid_statistics(["not-an-id"], np.zeros((1, 14, 3)))


def test_evenly_spaced_indices_are_deterministic_unique_and_bounded():
    assert evenly_spaced_indices(100, 5) == [0, 25, 50, 74, 99]
    assert evenly_spaced_indices(100, 5) == evenly_spaced_indices(100, 5)
    assert evenly_spaced_indices(3, 16) == [0, 1, 2]
    assert evenly_spaced_indices(100, 16)[0] == 0 and evenly_spaced_indices(100, 16)[-1] == 99
    with pytest.raises(ValueError):
        evenly_spaced_indices(0, 4)


def test_rounding_and_pattern_helpers():
    assert round_matches(0.98984, 0.9898, 4) is True
    assert round_matches(0.9903, 0.9898, 4) is False
    assert round_matches(0.9684, 0.968, 3) is True
    assert round_matches(0.9690, 0.968, 3) is False
    assert pattern_matches(0.5, 0.18) is True
    assert pattern_matches(0.01, 0.18) is False
    assert pattern_matches(None, 0.18) is False
    assert pattern_matches(0.0, 4e-7) is False


def _synthetic_report(point=(2.797, 1.784, 3.278), decomposition_shift=0.0):
    constant = np.broadcast_to(np.array(point), (8, 1, 14, 3)).copy()
    activation_names = [
        "pooled_encoder_feature",
        "gconv3",
        "transformer_encoder1",
        "transformer_encoder3",
    ]
    expected = {
        "pooled_encoder_feature": (0.18, None),
        "gconv3": (0.47, 3.28),
        "transformer_encoder1": (4e-4, 1.5e-2),
        "transformer_encoder3": (4e-7, 3e-6),
    }
    combined = {
        name: {"std_across_samples": expected[name][0], "std_across_joints": expected[name][1]}
        for name in activation_names
    }
    decomposition = {
        "train": {
            "absolute_mean_pose_mpjpe": 0.9898 + decomposition_shift,
            "centroid_distance_to_train_mean_centroid": 0.968,
            "centroid_relative_mean_skeleton_mpjpe": 0.155,
        },
        "development": {
            "absolute_mean_pose_mpjpe": 0.8014,
            "centroid_distance_to_train_mean_centroid": 0.787,
            "centroid_relative_mean_skeleton_mpjpe": 0.128,
        },
    }
    return {
        "predictions": {
            "sample_count": 8,
            "spread": prediction_spread(constant),
            "single_point": single_point_summary(constant),
            "ground_truth_spread": prediction_spread(constant),
            "stored_mean_pose_prediction_matches_recomputed_train_mean": True,
        },
        "activations": {
            "definition": "synthetic",
            "train_sample_ids": ["S11_01_10"],
            "development_sample_ids": ["S11_21_10"],
            "combined": combined,
        },
        "target_decomposition": {**decomposition, "train_mean_centroid": [0.0, 0.0, 0.0]},
        "group_centroids": {
            "train": {f"S11_{i:02d}": {} for i in range(8)},
            "development": {f"S11_{i:02d}": {} for i in range(20, 24)},
        },
    }


def test_check_premises_passes_when_measurements_match_the_plan():
    checks = check_premises(_synthetic_report())

    assert checks and all(check["passed"] for check in checks)


def test_check_premises_flags_numbers_that_do_not_reproduce():
    checks = check_premises(_synthetic_report(decomposition_shift=0.01))

    failed = [check["name"] for check in checks if not check["passed"]]
    assert failed == ["absolute_mean_pose_mpjpe (train)"]


def test_check_premises_flags_a_non_constant_prediction_set():
    report = _synthetic_report()
    rng = np.random.default_rng(2)
    varied = rng.normal(size=(8, 1, 14, 3))
    report["predictions"]["spread"] = prediction_spread(varied)
    report["predictions"]["single_point"] = single_point_summary(varied)

    failed = {check["name"] for check in check_premises(report) if not check["passed"]}

    assert "all stored development predictions are one common 3D point" in failed
    assert "prediction std across the 512 samples is ~0" in failed


def test_render_markdown_contains_verdict_and_tables():
    report = _synthetic_report()
    # Fill the fields render_markdown needs beyond the premise inputs.
    for name in ("gconv1", "gconv2", "transformer_encoder2", "final_pose"):
        report["activations"]["combined"][name] = {"std_across_samples": 0.1, "std_across_joints": 0.2}
    for name, stats in report["activations"]["combined"].items():
        stats.setdefault("shape", [4, 14, 256])
        stats.setdefault("fraction_exact_zeros", 0.0)
    report["group_centroids"] = {
        split: {
            "S11_01": {
                "sample_count": 4,
                "centroid_mean": [0.0, 0.0, 0.0],
                "centroid_std_per_axis": [0.0, 0.0, 0.0],
                "centroid_std_norm": 0.0,
            }
        }
        for split in ("train", "development")
    }
    report["premise_checks"] = check_premises(_synthetic_report())
    report["all_premises_reproduced"] = True

    text = render_markdown(report)

    assert "# Step-4B-R Collapse Diagnosis" in text
    assert "REPRODUCED" in text
    assert "| gconv3 |" in text
    json.dumps(report["premise_checks"])  # must be JSON serialisable


def test_outputs_inside_baseline_artifacts_are_refused(tmp_path):
    root = tmp_path / "step4b_r"
    (root / "baseline").mkdir(parents=True)

    with pytest.raises(ValueError):
        _assert_outputs_outside_baseline(root, [root / "baseline" / "diagnosis.json"])
    _assert_outputs_outside_baseline(root, [tmp_path / "step4b_r_diagnosis" / "diagnosis.json"])
