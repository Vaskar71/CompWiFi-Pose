from dataclasses import dataclass, replace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass(frozen=True)
class ResearchMetricConfig:
    stored_unit_to_mm: float = 1000.0
    presence_threshold: float = 0.5
    match_threshold_mm: float = 500.0
    pck_threshold: float = 0.5
    pck_scale_joint_indices: Tuple[int, int] = (6, 4)
    pck_guard_epsilon_stored: float = 1e-6
    root_index: int = 0
    pa_degenerate_epsilon_stored: float = 1e-12
    evaluation_name: str = "inferred-count"

    def validate(self):
        positive_finite = {
            "stored_unit_to_mm": self.stored_unit_to_mm,
            "match_threshold_mm": self.match_threshold_mm,
            "pa_degenerate_epsilon_stored": self.pa_degenerate_epsilon_stored,
        }
        for name, value in positive_finite.items():
            if not np.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be positive and finite")
        for name in ("presence_threshold", "pck_threshold", "pck_guard_epsilon_stored"):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be nonnegative and finite")
        if len(self.pck_scale_joint_indices) != 2:
            raise ValueError("pck_scale_joint_indices must contain exactly two indices")
        if len(set(self.pck_scale_joint_indices)) != 2:
            raise ValueError("pck_scale_joint_indices must be distinct")
        if any(index < 0 for index in self.pck_scale_joint_indices):
            raise ValueError("pck_scale_joint_indices must be nonnegative")
        if self.root_index < 0:
            raise ValueError("root_index must be nonnegative")
        if not self.evaluation_name:
            raise ValueError("evaluation_name must be nonempty")

    def as_dict(self):
        return {
            "stored_unit_to_mm": self.stored_unit_to_mm,
            "presence_threshold": self.presence_threshold,
            "match_threshold_mm": self.match_threshold_mm,
            "pck_threshold": self.pck_threshold,
            "pck_scale_joint_indices": list(self.pck_scale_joint_indices),
            "pck_guard_epsilon_stored": self.pck_guard_epsilon_stored,
            "root_index": self.root_index,
            "pa_degenerate_epsilon_stored": self.pa_degenerate_epsilon_stored,
            "evaluation_name": self.evaluation_name,
        }


@dataclass
class _MetricState:
    samples: int = 0
    exact_count_correct: int = 0
    matched_people: int = 0
    matched_joints: int = 0
    mpjpe_sum_mm: float = 0.0
    pa_people: int = 0
    pampjpe_sum_mm: float = 0.0
    root_pairs: int = 0
    root_error_sum_mm: float = 0.0
    pck_correct_joints: int = 0
    pck_denominator_joints: int = 0
    pck_guarded_people: int = 0
    pampjpe_undefined_people: int = 0
    missed_people: int = 0
    duplicate_people: int = 0
    spurious_predictions: int = 0

    def add(self, other):
        self.samples += other.samples
        self.exact_count_correct += other.exact_count_correct
        self.matched_people += other.matched_people
        self.matched_joints += other.matched_joints
        self.mpjpe_sum_mm += other.mpjpe_sum_mm
        self.pa_people += other.pa_people
        self.pampjpe_sum_mm += other.pampjpe_sum_mm
        self.root_pairs += other.root_pairs
        self.root_error_sum_mm += other.root_error_sum_mm
        self.pck_correct_joints += other.pck_correct_joints
        self.pck_denominator_joints += other.pck_denominator_joints
        self.pck_guarded_people += other.pck_guarded_people
        self.pampjpe_undefined_people += other.pampjpe_undefined_people
        self.missed_people += other.missed_people
        self.duplicate_people += other.duplicate_people
        self.spurious_predictions += other.spurious_predictions


def _resolve_config(config=None, overrides=None):
    if config is None:
        config = ResearchMetricConfig()
    if not isinstance(config, ResearchMetricConfig):
        raise TypeError("config must be a ResearchMetricConfig")
    if overrides:
        config = replace(config, **overrides)
    config.validate()
    return config


def _coordinate_samples(values, name):
    if isinstance(values, np.ndarray) and values.ndim == 4:
        samples = [np.asarray(sample, dtype=np.float64) for sample in values]
    else:
        samples = [np.asarray(sample, dtype=np.float64) for sample in values]
    for sample in samples:
        if sample.ndim != 3 or sample.shape[2] != 3:
            raise ValueError(f"{name} samples must have shape (people, joints, 3)")
        if not np.isfinite(sample).all():
            raise ValueError(f"{name} samples must be finite")
    return samples


def _presence_samples(presence_scores, prediction_samples):
    if presence_scores is None:
        return [np.ones(sample.shape[0], dtype=np.float64) for sample in prediction_samples]
    if isinstance(presence_scores, np.ndarray) and presence_scores.ndim == 2:
        score_samples = [np.asarray(sample, dtype=np.float64) for sample in presence_scores]
    else:
        score_samples = [np.asarray(sample, dtype=np.float64) for sample in presence_scores]
    if len(score_samples) != len(prediction_samples):
        raise ValueError("presence_scores must match the number of prediction samples")
    for scores, prediction in zip(score_samples, prediction_samples):
        if scores.ndim != 1 or scores.shape[0] != prediction.shape[0]:
            raise ValueError("each presence score array must match its prediction count")
        if not np.isfinite(scores).all():
            raise ValueError("presence scores must be finite")
    return score_samples


def _prepare_samples(ground_truths, predictions, presence_scores):
    gt_samples = _coordinate_samples(ground_truths, "ground_truths")
    pred_samples = _coordinate_samples(predictions, "predictions")
    score_samples = _presence_samples(presence_scores, pred_samples)
    if len(gt_samples) != len(pred_samples):
        raise ValueError("ground_truths and predictions must contain the same number of samples")
    for gt, pred in zip(gt_samples, pred_samples):
        if gt.shape[1] != pred.shape[1]:
            raise ValueError("ground truth and prediction joint counts must match")
    return gt_samples, pred_samples, score_samples


def _similarity_error_mm(prediction, ground_truth, config):
    gt_centered = ground_truth - ground_truth.mean(axis=0)
    pred_centered = prediction - prediction.mean(axis=0)
    gt_norm = np.sqrt(np.sum(gt_centered**2))
    pred_norm = np.sqrt(np.sum(pred_centered**2))
    if (
        gt_norm <= config.pa_degenerate_epsilon_stored
        or pred_norm <= config.pa_degenerate_epsilon_stored
    ):
        return None
    gt_unit = gt_centered / gt_norm
    pred_unit = pred_centered / pred_norm
    matrix = gt_unit.T @ pred_unit
    u, singular_values, vt = np.linalg.svd(matrix, full_matrices=False)
    v = vt.T
    rotation = v @ u.T
    determinant = np.linalg.det(rotation)
    correction = np.sign(determinant)
    if correction == 0:
        correction = 1.0
    v[:, -1] *= correction
    singular_values[-1] *= correction
    rotation = v @ u.T
    aligned = gt_norm * singular_values.sum() * (pred_unit @ rotation) + ground_truth.mean(axis=0)
    return float(np.linalg.norm(aligned - ground_truth, axis=-1).mean() * config.stored_unit_to_mm)


def _accumulate_sample(gt, pred, scores, config):
    joint_count = gt.shape[1]
    if max(config.pck_scale_joint_indices) >= joint_count:
        raise ValueError("pck_scale_joint_indices exceed the joint count")
    if config.root_index >= joint_count:
        raise ValueError("root_index exceeds the joint count")

    active_predictions = np.flatnonzero(scores >= config.presence_threshold)
    gt_count = gt.shape[0]
    predicted_count = active_predictions.shape[0]
    cost = None
    matches = []
    if gt_count and predicted_count:
        active_coordinates = pred[active_predictions]
        cost = np.linalg.norm(
            (gt[:, None, :, :] - active_coordinates[None, :, :, :])
            * config.stored_unit_to_mm,
            axis=-1,
        ).mean(axis=2)
        gt_indices, active_indices = linear_sum_assignment(cost)
        for gt_index, active_index in zip(gt_indices, active_indices):
            match_cost = float(cost[gt_index, active_index])
            if match_cost <= config.match_threshold_mm:
                matches.append(
                    {
                        "gt_index": int(gt_index),
                        "prediction_index": int(active_predictions[active_index]),
                        "active_prediction_index": int(active_index),
                        "cost_mm": match_cost,
                    }
                )

    matched_gt = {match["gt_index"] for match in matches}
    matched_predictions = {match["prediction_index"] for match in matches}
    missed_people = gt_count - len(matches)

    duplicate_people = 0
    spurious_predictions = 0
    if cost is not None:
        for prediction_index in active_predictions:
            if int(prediction_index) in matched_predictions:
                continue
            active_index = int(np.flatnonzero(active_predictions == prediction_index)[0])
            nearest_gt = int(np.argmin(cost[:, active_index]))
            if (
                nearest_gt in matched_gt
                and float(cost[nearest_gt, active_index]) <= config.match_threshold_mm
            ):
                duplicate_people += 1
            else:
                spurious_predictions += 1
    else:
        spurious_predictions = predicted_count

    mpjpe_values = []
    pa_values = []
    root_values = []
    pck_correct = 0
    pck_denominator = 0
    pck_guarded = 0
    pampjpe_undefined = 0
    scale_joint_a, scale_joint_b = config.pck_scale_joint_indices

    for match in matches:
        gt_person = gt[match["gt_index"]]
        pred_person = pred[match["prediction_index"]]
        joint_errors_mm = (
            np.linalg.norm(pred_person - gt_person, axis=-1) * config.stored_unit_to_mm
        )
        mpjpe_values.append(joint_errors_mm)
        root_error_mm = float(
            np.linalg.norm(pred_person[config.root_index] - gt_person[config.root_index])
            * config.stored_unit_to_mm
        )
        root_values.append(root_error_mm)

        pck_scale_stored = float(
            np.linalg.norm(gt_person[scale_joint_a] - gt_person[scale_joint_b])
        )
        if pck_scale_stored < config.pck_guard_epsilon_stored:
            pck_guarded += 1
        else:
            threshold_mm = (
                config.pck_threshold * pck_scale_stored * config.stored_unit_to_mm
            )
            pck_correct += int(np.count_nonzero(joint_errors_mm <= threshold_mm))
            pck_denominator += joint_count

        pa_error = _similarity_error_mm(pred_person, gt_person, config)
        if pa_error is None:
            pampjpe_undefined += 1
        else:
            pa_values.append(pa_error)

    state = _MetricState(
        samples=1,
        exact_count_correct=int(predicted_count == gt_count),
        matched_people=len(matches),
        matched_joints=sum(value.size for value in mpjpe_values),
        mpjpe_sum_mm=float(sum(value.sum() for value in mpjpe_values)),
        pa_people=len(pa_values),
        pampjpe_sum_mm=float(sum(pa_values)),
        root_pairs=len(root_values),
        root_error_sum_mm=float(sum(root_values)),
        pck_correct_joints=pck_correct,
        pck_denominator_joints=pck_denominator,
        pck_guarded_people=pck_guarded,
        pampjpe_undefined_people=pampjpe_undefined,
        missed_people=missed_people,
        duplicate_people=duplicate_people,
        spurious_predictions=spurious_predictions,
    )
    summary = {
        "gt_count": int(gt_count),
        "predicted_count": int(predicted_count),
        "count_correct": bool(predicted_count == gt_count),
        "matches": matches,
        "matched_mpjpe_mm": _mean(state.mpjpe_sum_mm, state.matched_joints),
        "matched_pampjpe_mm": _mean(state.pampjpe_sum_mm, state.pa_people),
        "root_error_mm": _mean(state.root_error_sum_mm, state.root_pairs),
        "pck_percent": _percent(pck_correct, pck_denominator),
        "pck_denominator_joints": pck_denominator,
        "pck_guarded_people": pck_guarded,
        "pampjpe_undefined_people": pampjpe_undefined,
        "missed_people": missed_people,
        "duplicate_people": duplicate_people,
        "spurious_predictions": spurious_predictions,
    }
    return state, summary


def _mean(total, count):
    if count == 0:
        return None
    return float(total / count)


def _percent(numerator, denominator):
    if denominator == 0:
        return None
    return float(100.0 * numerator / denominator)


def _ratio(numerator, denominator):
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _finalize_state(state, config, oracle_count=False):
    true_positives = state.matched_people
    false_positives = state.duplicate_people + state.spurious_predictions
    false_negatives = state.missed_people
    precision = _ratio(true_positives, true_positives + false_positives)
    recall = _ratio(true_positives, true_positives + false_negatives)
    if precision is None or recall is None or precision + recall == 0:
        f1 = None
    else:
        f1 = float(2 * precision * recall / (precision + recall))
    return {
        "evaluation_name": config.evaluation_name,
        "oracle_count": bool(oracle_count),
        "samples": state.samples,
        "stored_unit_to_mm": config.stored_unit_to_mm,
        "presence_threshold": config.presence_threshold,
        "matching_rule": (
            "Rectangular Hungarian one-to-one assignment using mean per-joint "
            "Euclidean distance in millimetres; assignments above "
            f"{config.match_threshold_mm} mm are invalid."
        ),
        "match_threshold_mm": config.match_threshold_mm,
        "matched_people": true_positives,
        "missed_people": false_negatives,
        "duplicate_people": state.duplicate_people,
        "spurious_predictions": state.spurious_predictions,
        "person_precision": precision,
        "person_recall": recall,
        "person_f1": f1,
        "person_precision_denominator": true_positives + false_positives,
        "person_recall_denominator": true_positives + false_negatives,
        "exact_count_accuracy_percent": _percent(state.exact_count_correct, state.samples),
        "exact_count_correct_samples": state.exact_count_correct,
        "matched_mpjpe_mm": _mean(state.mpjpe_sum_mm, state.matched_joints),
        "matched_mpjpe_denominator_joints": state.matched_joints,
        "matched_pampjpe_mm": _mean(state.pampjpe_sum_mm, state.pa_people),
        "matched_pampjpe_denominator_people": state.pa_people,
        "pampjpe_undefined_people": state.pampjpe_undefined_people,
        "root_error_mm": _mean(state.root_error_sum_mm, state.root_pairs),
        "root_error_denominator_people": state.root_pairs,
        "root_index": config.root_index,
        "pck_percent": _percent(state.pck_correct_joints, state.pck_denominator_joints),
        "pck_correct_joints": state.pck_correct_joints,
        "pck_denominator_joints": state.pck_denominator_joints,
        "pck_threshold": config.pck_threshold,
        "pck_scale_joint_indices": list(config.pck_scale_joint_indices),
        "pck_scale_definition": (
            "Euclidean distance between the two configured ground-truth joints "
            "in stored coordinate units."
        ),
        "pck_near_zero_guard_stored_units": config.pck_guard_epsilon_stored,
        "pck_guarded_people": state.pck_guarded_people,
    }


def evaluate_set_metrics(
    ground_truths,
    predictions,
    presence_scores=None,
    *,
    config=None,
    **config_overrides,
):
    config = _resolve_config(config, config_overrides)
    gt_samples, pred_samples, score_samples = _prepare_samples(
        ground_truths, predictions, presence_scores
    )
    global_state = _MetricState()
    per_count_states: Dict[int, _MetricState] = {}
    sample_summaries = []
    for gt, pred, scores in zip(gt_samples, pred_samples, score_samples):
        state, summary = _accumulate_sample(gt, pred, scores, config)
        global_state.add(state)
        count_state = per_count_states.setdefault(gt.shape[0], _MetricState())
        count_state.add(state)
        sample_summaries.append(summary)

    return {
        "overall": _finalize_state(global_state, config),
        "per_person_count": {
            str(person_count): _finalize_state(state, config)
            for person_count, state in sorted(per_count_states.items())
        },
        "samples": sample_summaries,
        "config": config.as_dict(),
    }


class ResearchMetricAccumulator:
    def __init__(self, config=None, **config_overrides):
        self.config = _resolve_config(config, config_overrides)
        self._global_state = _MetricState()
        self._per_count_states: Dict[int, _MetricState] = {}

    def update(self, ground_truths, predictions, presence_scores=None):
        gt_samples, pred_samples, score_samples = _prepare_samples(
            ground_truths, predictions, presence_scores
        )
        sample_summaries = []
        for gt, pred, scores in zip(gt_samples, pred_samples, score_samples):
            state, summary = _accumulate_sample(gt, pred, scores, self.config)
            self._global_state.add(state)
            count_state = self._per_count_states.setdefault(gt.shape[0], _MetricState())
            count_state.add(state)
            sample_summaries.append(summary)
        return sample_summaries

    def compute(self):
        return {
            "overall": _finalize_state(self._global_state, self.config),
            "per_person_count": {
                str(person_count): _finalize_state(state, self.config)
                for person_count, state in sorted(self._per_count_states.items())
            },
            "config": self.config.as_dict(),
        }


def evaluate_oracle_count_diagnostic(
    ground_truths,
    predictions,
    presence_scores=None,
    *,
    config=None,
    **config_overrides,
):
    base_config = _resolve_config(config, config_overrides)
    diagnostic_config = replace(
        base_config, evaluation_name="oracle-count-diagnostic"
    )
    gt_samples, pred_samples, score_samples = _prepare_samples(
        ground_truths, predictions, presence_scores
    )
    oracle_predictions = []
    oracle_scores = []
    for gt, pred, scores in zip(gt_samples, pred_samples, score_samples):
        target_count = gt.shape[0]
        if scores is None:
            order = np.arange(min(target_count, pred.shape[0]))
        else:
            order = np.argsort(-scores, kind="stable")[: min(target_count, pred.shape[0])]
        oracle_predictions.append(pred[order])
        oracle_scores.append(np.ones(order.shape[0], dtype=np.float64))
    result = evaluate_set_metrics(
        gt_samples,
        oracle_predictions,
        oracle_scores,
        config=diagnostic_config,
    )
    result["diagnostic_only"] = True
    result["headline_comparison_allowed"] = False
    result["overall"]["oracle_count"] = True
    for metrics in result["per_person_count"].values():
        metrics["oracle_count"] = True
    return result
