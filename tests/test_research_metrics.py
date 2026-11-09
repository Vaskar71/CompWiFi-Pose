import numpy as np
import pytest

from research_metrics import (
    ResearchMetricAccumulator,
    evaluate_oracle_count_diagnostic,
    evaluate_set_metrics,
)


def make_person(offset_x=0.0, offset_y=0.0):
    joints = np.arange(14, dtype=np.float64)
    return np.column_stack(
        (
            joints * 0.1 + offset_x,
            np.sin(joints * 0.7) * 0.2 + offset_y,
            (joints % 5) * 0.04,
        )
    )


def pad_predictions(*people, total_slots=3):
    padded = np.zeros((total_slots, 14, 3), dtype=np.float64)
    for index, person in enumerate(people):
        padded[index] = person
    return padded


def test_perfect_prediction_has_zero_error_and_full_person_detection():
    gt_person = make_person()
    result = evaluate_set_metrics(
        [gt_person[None, :, :]],
        [pad_predictions(gt_person)],
        [[1.0, 0.0, 0.0]],
    )
    overall = result["overall"]

    assert overall["matched_mpjpe_mm"] == pytest.approx(0.0)
    assert overall["matched_pampjpe_mm"] == pytest.approx(0.0, abs=1e-10)
    assert overall["root_error_mm"] == pytest.approx(0.0)
    assert overall["pck_percent"] == pytest.approx(100.0)
    assert overall["exact_count_accuracy_percent"] == pytest.approx(100.0)
    assert overall["missed_people"] == 0
    assert overall["duplicate_people"] == 0
    assert overall["spurious_predictions"] == 0
    assert overall["person_precision"] == pytest.approx(1.0)
    assert overall["person_recall"] == pytest.approx(1.0)
    assert overall["person_f1"] == pytest.approx(1.0)
    assert result["per_person_count"]["1"]["samples"] == 1


def test_hungarian_matching_recovers_person_permutation():
    gt_people = np.stack((make_person(offset_x=0.0), make_person(offset_x=10.0)))
    predictions = np.stack((gt_people[1], gt_people[0]))

    result = evaluate_set_metrics(
        [gt_people], [predictions], [[1.0, 1.0]]
    )

    match_pairs = {
        (match["gt_index"], match["prediction_index"])
        for match in result["samples"][0]["matches"]
    }
    assert match_pairs == {(0, 1), (1, 0)}
    assert result["overall"]["matched_mpjpe_mm"] == pytest.approx(0.0)
    assert result["overall"]["matched_people"] == 2


def test_missing_person_is_counted_as_false_negative():
    gt_people = np.stack((make_person(offset_x=0.0), make_person(offset_x=10.0)))

    result = evaluate_set_metrics(
        [gt_people],
        [pad_predictions(gt_people[0])],
        [[1.0, 0.0, 0.0]],
    )
    overall = result["overall"]

    assert result["samples"][0]["predicted_count"] == 1
    assert overall["exact_count_accuracy_percent"] == pytest.approx(0.0)
    assert overall["matched_people"] == 1
    assert overall["missed_people"] == 1
    assert overall["duplicate_people"] == 0
    assert overall["spurious_predictions"] == 0
    assert overall["person_precision"] == pytest.approx(1.0)
    assert overall["person_recall"] == pytest.approx(0.5)
    assert overall["person_f1"] == pytest.approx(2.0 / 3.0)
    assert overall["matched_mpjpe_mm"] == pytest.approx(0.0)


def test_extra_far_prediction_is_spurious_not_a_duplicate():
    gt_person = make_person()
    predictions = pad_predictions(
        gt_person,
        make_person(offset_x=10.0),
        total_slots=3,
    )

    result = evaluate_set_metrics(
        [gt_person[None, :, :]],
        [predictions],
        [[1.0, 1.0, 0.0]],
    )
    overall = result["overall"]

    assert result["samples"][0]["predicted_count"] == 2
    assert overall["exact_count_accuracy_percent"] == pytest.approx(0.0)
    assert overall["matched_people"] == 1
    assert overall["missed_people"] == 0
    assert overall["duplicate_people"] == 0
    assert overall["spurious_predictions"] == 1
    assert overall["person_precision"] == pytest.approx(0.5)
    assert overall["person_recall"] == pytest.approx(1.0)
    assert overall["person_f1"] == pytest.approx(2.0 / 3.0)


def test_duplicate_prediction_near_an_already_matched_person_is_counted():
    gt_people = np.stack((make_person(offset_x=0.0), make_person(offset_x=10.0)))
    predictions = np.stack(
        (
            make_person(offset_x=0.001),
            make_person(offset_x=0.002),
            np.zeros((14, 3)),
        )
    )

    result = evaluate_set_metrics(
        [gt_people],
        [predictions],
        [[1.0, 1.0, 0.0]],
    )
    overall = result["overall"]

    assert result["samples"][0]["predicted_count"] == 2
    assert overall["matched_people"] == 1
    assert overall["missed_people"] == 1
    assert overall["duplicate_people"] == 1
    assert overall["spurious_predictions"] == 0
    assert overall["person_precision"] == pytest.approx(0.5)
    assert overall["person_recall"] == pytest.approx(0.5)
    assert overall["person_f1"] == pytest.approx(0.5)


def test_millimetre_conversion_is_applied_exactly_once():
    gt_person = make_person()
    translated_prediction = gt_person + np.array([1.0, 0.0, 0.0])

    result = evaluate_set_metrics(
        [gt_person[None, :, :]],
        [pad_predictions(translated_prediction)],
        [[1.0, 0.0, 0.0]],
        stored_unit_to_mm=1000.0,
        match_threshold_mm=2000.0,
    )
    overall = result["overall"]

    assert overall["stored_unit_to_mm"] == 1000.0
    assert overall["matched_mpjpe_mm"] == pytest.approx(1000.0)
    assert overall["root_error_mm"] == pytest.approx(1000.0)
    assert overall["matched_pampjpe_mm"] == pytest.approx(0.0, abs=1e-9)

    one_scale_result = evaluate_set_metrics(
        [gt_person[None, :, :]],
        [pad_predictions(translated_prediction)],
        [[1.0, 0.0, 0.0]],
        stored_unit_to_mm=1.0,
    )
    assert one_scale_result["overall"]["matched_mpjpe_mm"] == pytest.approx(1.0)
    assert one_scale_result["overall"]["root_error_mm"] == pytest.approx(1.0)


def test_scaled_prediction_has_pa_mpjpe_but_nonzero_mpjpe():
    gt_person = make_person()
    scaled_prediction = gt_person * 2.0 + np.array([0.5, -0.25, 1.0])

    result = evaluate_set_metrics(
        [gt_person[None, :, :]],
        [pad_predictions(scaled_prediction)],
        [[1.0, 0.0, 0.0]],
        match_threshold_mm=20000.0,
    )

    assert result["overall"]["matched_pampjpe_mm"] == pytest.approx(0.0, abs=1e-9)
    assert result["overall"]["matched_mpjpe_mm"] > 1.0


def test_pck_near_zero_scale_is_guarded_and_excluded_from_denominator():
    gt_person = make_person()
    gt_person[6] = gt_person[4]

    result = evaluate_set_metrics(
        [gt_person[None, :, :]],
        [pad_predictions(gt_person.copy())],
        [[1.0, 0.0, 0.0]],
    )
    overall = result["overall"]

    assert overall["matched_mpjpe_mm"] == pytest.approx(0.0)
    assert overall["pck_percent"] is None
    assert overall["pck_denominator_joints"] == 0
    assert overall["pck_guarded_people"] == 1
    assert overall["matched_pampjpe_mm"] == pytest.approx(0.0, abs=1e-10)


def test_global_aggregation_weights_unequal_batches_by_joint_count():
    gt_person = make_person()
    translated_prediction = gt_person + np.array([1.0, 0.0, 0.0])
    ground_truths = [
        gt_person[None, :, :],
        gt_person[None, :, :],
        gt_person[None, :, :],
    ]
    predictions = [
        pad_predictions(gt_person),
        pad_predictions(gt_person),
        pad_predictions(translated_prediction),
    ]
    scores = [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]

    accumulator = ResearchMetricAccumulator(match_threshold_mm=2000.0)
    accumulator.update(ground_truths[:2], predictions[:2], scores[:2])
    accumulator.update(ground_truths[2:], predictions[2:], scores[2:])
    accumulated = accumulator.compute()["overall"]

    direct = evaluate_set_metrics(
        ground_truths,
        predictions,
        scores,
        match_threshold_mm=2000.0,
    )["overall"]

    assert accumulated["matched_mpjpe_mm"] == pytest.approx(1000.0 / 3.0)
    assert accumulated["matched_mpjpe_denominator_joints"] == 3 * 14
    assert direct["matched_mpjpe_mm"] == pytest.approx(accumulated["matched_mpjpe_mm"])
    assert direct["matched_mpjpe_denominator_joints"] == accumulated[
        "matched_mpjpe_denominator_joints"
    ]
    assert direct["exact_count_accuracy_percent"] == accumulated[
        "exact_count_accuracy_percent"
    ]


def test_oracle_count_is_a_separately_named_diagnostic():
    gt_people = np.stack((make_person(offset_x=0.0), make_person(offset_x=10.0)))
    predictions = np.stack(
        (
            make_person(offset_x=100.0),
            gt_people[0],
            gt_people[1],
        )
    )
    scores = [0.3, 0.9, 0.8]

    inferred = evaluate_set_metrics(
        [gt_people], [predictions], [scores], presence_threshold=0.85
    )
    oracle = evaluate_oracle_count_diagnostic(
        [gt_people], [predictions], [scores], presence_threshold=0.85
    )

    assert inferred["overall"]["evaluation_name"] == "inferred-count"
    assert inferred["overall"]["oracle_count"] is False
    assert inferred["samples"][0]["predicted_count"] == 1
    assert inferred["overall"]["missed_people"] == 1

    assert oracle["overall"]["evaluation_name"] == "oracle-count-diagnostic"
    assert oracle["overall"]["oracle_count"] is True
    assert oracle["diagnostic_only"] is True
    assert oracle["headline_comparison_allowed"] is False
    assert oracle["samples"][0]["predicted_count"] == 2
    assert oracle["overall"]["exact_count_accuracy_percent"] == pytest.approx(100.0)
    assert oracle["overall"]["missed_people"] == 0


def test_invalid_shapes_and_nonfinite_values_are_rejected():
    gt_person = make_person()[None, :, :]
    with pytest.raises(ValueError, match="same number of samples"):
        evaluate_set_metrics([gt_person, gt_person], [pad_predictions(gt_person[0])])
    with pytest.raises(ValueError, match="presence score array"):
        evaluate_set_metrics(
            [gt_person],
            [pad_predictions(gt_person[0])],
            [[1.0, 0.0]],
        )
    bad_gt = gt_person.copy()
    bad_gt[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        evaluate_set_metrics([bad_gt], [pad_predictions(gt_person[0])], [[1.0, 0.0, 0.0]])
