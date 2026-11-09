import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from feeder.piw3d_single_person import SinglePersonPiW3DDataset, make_single_person_dataloader
from step4b_config import validate_step4b_config
from step4b_single_person_pilot import (
    _resolve_device,
    _seed_everything,
    build_model,
    compute_mean_pose,
)
from step4b_subset import is_single_person_id


D1_VERSION = "compwifi-step4b-d1-v1"
D1_STEP_MULTIPLIER = 4
D1_METRIC_INTERVAL_STEPS = 25
EXPECTED_ORIGINAL_STEPS = 200
EXPECTED_ORIGINAL_BATCH_IDS = [
    "S11_06_10",
    "S11_06_11",
    "S11_06_13",
    "S11_06_14",
    "S11_06_16",
    "S11_06_17",
    "S11_06_19",
    "S11_06_20",
]
PLOT_SAMPLE_IDS = list(EXPECTED_ORIGINAL_BATCH_IDS)
EXPECTED_CSI_SHAPE = (3, 180, 20)
EXPECTED_KEYPOINT_SHAPE = (1, 14, 3)
DEFAULT_SOURCE_ARTIFACT_ROOT = Path("data/_artifacts/step4b_pilot")
DEFAULT_D1_ROOT = Path("data/_artifacts/step4b_d1")


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _resolve_source_paths(source_artifact_root, freeze_config, pilot_root):
    source_artifact_root = Path(source_artifact_root)
    freeze_config = (
        Path(freeze_config)
        if freeze_config
        else source_artifact_root / "freeze_v2" / "step4b_freeze_config.json"
    )
    pilot_root = Path(pilot_root) if pilot_root else source_artifact_root / "pilot_data_v2"
    return source_artifact_root, freeze_config, pilot_root


def _source_file_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def verify_frozen_inputs(source_artifact_root=DEFAULT_SOURCE_ARTIFACT_ROOT, freeze_config=None, pilot_root=None):
    source_artifact_root, freeze_config, pilot_root = _resolve_source_paths(
        source_artifact_root, freeze_config, pilot_root
    )
    if freeze_config.resolve().parent.name != "freeze_v2":
        raise ValueError("Step-4B-D1 requires the corrected freeze_v2 configuration")
    if pilot_root.name != "pilot_data_v2":
        raise ValueError("Step-4B-D1 requires the corrected pilot_data_v2 artifact")

    freeze = _load_json(freeze_config)
    config = freeze["config"]
    validate_step4b_config(config)
    if config["seed"] != 0:
        raise ValueError("Frozen seed changed")
    if config["overfit"]["steps"] != EXPECTED_ORIGINAL_STEPS:
        raise ValueError("Frozen original optimizer-step budget changed")
    if config["overfit"]["batch_size"] != len(EXPECTED_ORIGINAL_BATCH_IDS):
        raise ValueError("Frozen tiny-batch size changed")
    if config["training"]["optimizer"] != "SGD":
        raise ValueError("Frozen optimizer changed")
    if config["training"]["learning_rate"] != 0.01:
        raise ValueError("Frozen learning rate changed")
    if config["training"]["momentum"] != 0.9:
        raise ValueError("Frozen momentum changed")
    if config["label_access"]["official_test_used"] is not False:
        raise ValueError("Frozen configuration must not use official test data")
    if config["label_access"]["multi_person_labels_used"] is not False:
        raise ValueError("Frozen configuration must not use multi-person labels")

    validation_path = source_artifact_root / "preprocessing_validation_v2.json"
    validation = _load_json(validation_path)
    if validation["official_test_used"] is not False:
        raise ValueError("Pilot validation must not use official test data")
    if validation["source_dataset_modified"] is not False:
        raise ValueError("Pilot validation must not modify the source dataset")
    if validation["counts"] != {
        "failed_ids": 0,
        "selected_development_ids": 512,
        "selected_train_ids": 2048,
        "total_selected_ids": 2560,
        "validated_ids": 2560,
    }:
        raise ValueError("Corrected pilot preprocessing validation counts changed")
    if validation["failures"]:
        raise ValueError("Corrected pilot preprocessing validation contains failures")

    overfit_path = source_artifact_root / "overfit_result.json"
    overfit = _load_json(overfit_path)
    if overfit["status"] != "FAIL":
        raise ValueError("Original Step-4B overfit artifact is not the recorded failure")
    if overfit["steps"] != EXPECTED_ORIGINAL_STEPS:
        raise ValueError("Original Step-4B overfit step budget changed")
    if overfit["batch_sample_ids"] != EXPECTED_ORIGINAL_BATCH_IDS:
        raise ValueError("Original Step-4B tiny-batch sample IDs changed")
    if overfit["final_mpjpe_stored_units"] <= config["overfit"]["criterion"]["max_final_mpjpe_stored_units"]:
        raise ValueError("Original Step-4B result no longer records the threshold failure")

    decision_path = source_artifact_root / "decision.json"
    decision = _load_json(decision_path)
    if decision["status"] != "BLOCKED":
        raise ValueError("Original Step-4B decision must remain BLOCKED")

    selected_train_ids = list(freeze["selected_ids"]["internal_train"])
    selected_development_ids = list(freeze["selected_ids"]["internal_development"])
    if selected_train_ids[: len(EXPECTED_ORIGINAL_BATCH_IDS)] != EXPECTED_ORIGINAL_BATCH_IDS:
        raise ValueError("Frozen internal-train ordering does not reproduce the tiny batch")
    if len(selected_train_ids) != 2048 or len(selected_development_ids) != 512:
        raise ValueError("Frozen internal subset counts changed")
    if set(selected_train_ids) & set(selected_development_ids):
        raise ValueError("Frozen internal train/development IDs overlap")
    if not all(is_single_person_id(sample_id) for sample_id in selected_train_ids):
        raise ValueError("Frozen internal-train subset contains a non-single-person ID")
    if not all(freeze["split_assignments"].get(sample_id) == "internal_train" for sample_id in selected_train_ids):
        raise ValueError("Frozen internal-train split assignment changed")
    if not all(
        freeze["split_assignments"].get(sample_id) == "internal_development"
        for sample_id in selected_development_ids
    ):
        raise ValueError("Frozen internal-development split assignment changed")

    selected_train_path = pilot_root / "train_data" / "selected_train_ids.txt"
    persisted_train_ids = selected_train_path.read_text(encoding="utf-8-sig").splitlines()
    if persisted_train_ids != selected_train_ids:
        raise ValueError("Pilot selected_train_ids.txt does not exactly match freeze_v2")

    pairing_records = []
    pilot_train_root = pilot_root / "train_data"
    for sample_id in EXPECTED_ORIGINAL_BATCH_IDS:
        csi_path = pilot_train_root / "csi_ap" / f"{sample_id}.npy"
        keypoint_path = pilot_train_root / "keypoint" / f"{sample_id}.npy"
        if csi_path.stem != sample_id or keypoint_path.stem != sample_id:
            raise ValueError(f"CSI/keypoint filename pairing changed for {sample_id}")
        if not csi_path.is_file():
            raise FileNotFoundError(f"Missing pilot CSI for frozen tiny batch: {csi_path}")
        if not keypoint_path.is_file():
            raise FileNotFoundError(f"Missing pilot keypoint for frozen tiny batch: {keypoint_path}")
        csi = np.load(csi_path, allow_pickle=False)
        keypoint = np.load(keypoint_path, allow_pickle=False)
        if csi.shape != EXPECTED_CSI_SHAPE:
            raise ValueError(f"Unexpected CSI shape for {sample_id}: {csi.shape}")
        if keypoint.shape != EXPECTED_KEYPOINT_SHAPE:
            raise ValueError(f"Unexpected keypoint shape for {sample_id}: {keypoint.shape}")
        if not np.isfinite(csi).all() or not np.isfinite(keypoint).all():
            raise ValueError(f"Non-finite frozen tiny-batch arrays for {sample_id}")
        pairing_records.append(
            {
                "sample_id": sample_id,
                "csi_path": str(csi_path),
                "keypoint_path": str(keypoint_path),
                "csi_shape": list(csi.shape),
                "keypoint_shape": list(keypoint.shape),
                "paired_by_exact_sample_id": True,
            }
        )

    return {
        "diagnostic_version": D1_VERSION,
        "source_algorithm_version": config["algorithm_version"],
        "source_seed": config["seed"],
        "source_batch_size": config["overfit"]["batch_size"],
        "source_original_optimizer_steps": config["overfit"]["steps"],
        "source_optimizer": config["training"]["optimizer"],
        "source_learning_rate": config["training"]["learning_rate"],
        "source_momentum": config["training"]["momentum"],
        "source_loss": "mean Euclidean distance over people, joints, and coordinate triples",
        "source_coordinate_convention": (
            "stored coordinates unchanged; coordinate axes are reported by zero-based index only; "
            "no axis reordering or official axis-name interpretation"
        ),
        "source_model": config["model"],
        "source_parameter_counts": overfit["model_parameters"],
        "source_initial_loss": overfit["initial_loss"],
        "source_final_loss": overfit["final_loss"],
        "source_initial_mpjpe_stored_units": overfit["initial_mpjpe_stored_units"],
        "source_final_mpjpe_stored_units": overfit["final_mpjpe_stored_units"],
        "source_final_to_initial_mpjpe_ratio": overfit["final_to_initial_mpjpe_ratio"],
        "source_absolute_mpjpe_criterion_stored_units": config["overfit"]["criterion"][
            "max_final_mpjpe_stored_units"
        ],
        "source_overfit_status": overfit["status"],
        "source_overfit_failures": overfit["failures"],
        "source_step4b_decision": decision,
        "input_manifests": freeze["input_manifests"],
        "subset_counts": freeze["counts"],
        "label_access": config["label_access"],
        "preprocessing_validation_counts": validation["counts"],
        "selected_internal_train_ids_count": len(selected_train_ids),
        "selected_internal_development_ids_count": len(selected_development_ids),
        "tiny_batch_sample_ids": list(EXPECTED_ORIGINAL_BATCH_IDS),
        "tiny_batch_exact_pairing": pairing_records,
        "official_test_used": False,
        "development_samples_loaded": False,
    }


def write_diagnostic_freeze(source_artifact_root, d1_root, freeze_config, pilot_root, verification):
    source_artifact_root = Path(source_artifact_root)
    d1_root = Path(d1_root)
    d1_root.mkdir(parents=True, exist_ok=True)
    freeze_path = d1_root / "step4b_d1_freeze.json"
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen D1 diagnostic: {freeze_path}")
    original_steps = int(verification["source_original_optimizer_steps"])
    diagnostic_steps = original_steps * D1_STEP_MULTIPLIER
    record = {
        "diagnostic_version": D1_VERSION,
        "diagnostic_type": "bounded under-training diagnostic after failed frozen Step-4B tiny-overfit gate",
        "source_records": {
            "step4b_freeze_v2": _source_file_record(freeze_config),
            "preprocessing_validation_v2": _source_file_record(
                source_artifact_root / "preprocessing_validation_v2.json"
            ),
            "overfit_result": _source_file_record(source_artifact_root / "overfit_result.json"),
            "step4b_decision": _source_file_record(source_artifact_root / "decision.json"),
            "selected_train_ids": _source_file_record(
                Path(pilot_root) / "train_data" / "selected_train_ids.txt"
            ),
        },
        "frozen_source_summary": verification,
        "diagnostic_freeze": {
            "step_multiplier": D1_STEP_MULTIPLIER,
            "original_optimizer_steps": original_steps,
            "diagnostic_optimizer_steps": diagnostic_steps,
            "metric_interval_steps": D1_METRIC_INTERVAL_STEPS,
            "original_absolute_final_mpjpe_criterion_stored_units": verification[
                "source_absolute_mpjpe_criterion_stored_units"
            ],
            "only_optimizer_step_budget_changed": True,
            "unchanged_fields": [
                "corrected freeze_v2 internal-train sample IDs",
                "ordered fixed batch of 8 samples",
                "seed",
                "model family and dimensions",
                "randomly initialized from-scratch weights",
                "CSI preprocessing and loader normalization",
                "pose loss",
                "SGD optimizer",
                "learning rate",
                "momentum",
                "batch size",
                "device selection mode",
                "coordinate order",
                "absolute final-MPJPE criterion",
            ],
            "plot_sample_ids": PLOT_SAMPLE_IDS,
            "development_samples_loaded": False,
            "official_test_used": False,
            "coordinate_order": verification["source_coordinate_convention"],
        },
    }
    _write_json(freeze_path, record)
    return freeze_path, record


def _coordinate_summary(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "overall": {
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "variance": float(np.var(values)),
        },
        "per_coordinate_index": {
            str(axis): {
                "min": float(np.min(values[:, axis])),
                "max": float(np.max(values[:, axis])),
                "variance": float(np.var(values[:, axis])),
            }
            for axis in range(values.shape[-1])
        },
    }


def _mean_pose_comparison(prediction, target, train_mean_pose):
    fixed_batch_mean_pose = np.mean(target, axis=0)
    per_sample_distance_to_fixed_mean = np.linalg.norm(target - fixed_batch_mean_pose[None, :, :], axis=-1)
    model_distance_to_fixed_mean = np.linalg.norm(prediction - fixed_batch_mean_pose[None, :, :], axis=-1)
    model_distance_to_train_mean = np.linalg.norm(prediction - train_mean_pose[None, :, :], axis=-1)
    train_mean_distance_to_target = np.linalg.norm(target - train_mean_pose[None, :, :], axis=-1)
    target_variance_per_axis = np.var(target, axis=0).mean(axis=0)
    prediction_variance_per_axis = np.var(prediction, axis=0).mean(axis=0)
    guard = 1e-12
    return {
        "fixed_batch_mean_pose": fixed_batch_mean_pose,
        "fixed_batch_mean_pose_mpjpe_baseline_stored_units": float(
            np.mean(np.mean(per_sample_distance_to_fixed_mean, axis=-1))
        ),
        "model_to_fixed_batch_mean_pose_mpjpe_stored_units": float(
            np.mean(np.mean(model_distance_to_fixed_mean, axis=-1))
        ),
        "train_dataset_mean_pose": train_mean_pose,
        "train_dataset_mean_pose_mpjpe_baseline_stored_units": float(
            np.mean(np.mean(train_mean_distance_to_target, axis=-1))
        ),
        "model_to_train_dataset_mean_pose_mpjpe_stored_units": float(
            np.mean(np.mean(model_distance_to_train_mean, axis=-1))
        ),
        "target_cross_sample_variance_per_coordinate_index": [
            float(value) for value in target_variance_per_axis
        ],
        "prediction_cross_sample_variance_per_coordinate_index": [
            float(value) for value in prediction_variance_per_axis
        ],
        "prediction_to_target_variance_ratio_per_coordinate_index": [
            float(prediction_variance_per_axis[axis] / max(target_variance_per_axis[axis], guard))
            for axis in range(target.shape[-1])
        ],
        "target_mean_distance_to_own_mean_mpjpe_stored_units": float(
            np.mean(np.linalg.norm(target - np.mean(target, axis=0, keepdims=True), axis=-1))
        ),
        "prediction_mean_distance_to_own_mean_mpjpe_stored_units": float(
            np.mean(np.linalg.norm(prediction - np.mean(prediction, axis=0, keepdims=True), axis=-1))
        ),
    }


def analyze_prediction_batch(prediction, target, sample_ids, train_mean_pose=None):
    prediction = np.asarray(prediction, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if prediction.ndim == 4 and prediction.shape[1] == 1:
        prediction = prediction[:, 0, :, :]
    if target.ndim == 4 and target.shape[1] == 1:
        target = target[:, 0, :, :]
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError(f"Prediction shape {prediction.shape} is incompatible with target shape {target.shape}")

    difference = prediction - target
    joint_errors = np.linalg.norm(difference, axis=-1)
    per_sample_mpjpe = np.mean(joint_errors, axis=-1)
    per_joint_mpjpe = np.mean(joint_errors, axis=0)
    per_axis_mae = np.mean(np.abs(difference), axis=(0, 1))
    per_axis_rmse = np.sqrt(np.mean(np.square(difference), axis=(0, 1)))
    per_axis_max_abs = np.max(np.abs(difference), axis=(0, 1))

    summary = {
        "mpjpe_stored_units": float(np.mean(joint_errors)),
        "per_sample": [
            {
                "sample_id": sample_ids[index],
                "mpjpe_stored_units": float(per_sample_mpjpe[index]),
                "per_coordinate_index_absolute_error": [
                    float(value) for value in np.mean(np.abs(difference[index]), axis=0)
                ],
            }
            for index in range(len(sample_ids))
        ],
        "per_joint_mpjpe_stored_units": [float(value) for value in per_joint_mpjpe],
        "per_coordinate_index_mae_stored_units": [float(value) for value in per_axis_mae],
        "per_coordinate_index_rmse_stored_units": [float(value) for value in per_axis_rmse],
        "per_coordinate_index_max_abs_error_stored_units": [float(value) for value in per_axis_max_abs],
        "prediction_coordinate_ranges_and_variance": _coordinate_summary(prediction),
        "target_coordinate_ranges_and_variance": _coordinate_summary(target),
        "coordinate_order": (
            "stored coordinate order unchanged; coordinate indices 0, 1, and 2 are not interpreted "
            "as officially verified axis names"
        ),
    }
    arrays = {
        "prediction": prediction,
        "target": target,
        "difference": difference,
        "joint_errors": joint_errors,
        "per_sample_mpjpe": per_sample_mpjpe,
        "per_joint_mpjpe": per_joint_mpjpe,
    }
    if train_mean_pose is not None:
        comparison = _mean_pose_comparison(prediction, target, np.asarray(train_mean_pose, dtype=np.float64))
        summary["mean_pose_and_collapsed_output_comparison"] = {
            key: value for key, value in comparison.items() if not isinstance(value, np.ndarray)
        }
        arrays["fixed_batch_mean_pose"] = comparison["fixed_batch_mean_pose"]
        arrays["train_dataset_mean_pose"] = comparison["train_dataset_mean_pose"]
    return summary, arrays


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _clone_model_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore_model_state(model, state):
    with torch.no_grad():
        for name, value in model.state_dict().items():
            value.copy_(state[name])


def _gradient_statistics(model):
    finite_tensor_count = 0
    total_tensor_count = 0
    max_abs_gradient = 0.0
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        total_tensor_count += 1
        if parameter.grad is None:
            raise ValueError("Trainable parameter has no gradient")
        if bool(torch.isfinite(parameter.grad).all().item()):
            finite_tensor_count += 1
        tensor_max = float(parameter.grad.detach().abs().max().item())
        if np.isfinite(tensor_max):
            max_abs_gradient = max(max_abs_gradient, tensor_max)
    return {
        "gradient_tensors": total_tensor_count,
        "finite_gradient_tensors": finite_tensor_count,
        "all_gradients_finite": finite_tensor_count == total_tensor_count,
        "max_abs_gradient": max_abs_gradient,
    }


def _capture_checkpoint(model, csi, pose, sample_ids, step, train_mean_pose):
    if not model.training:
        raise ValueError("D1 checkpoint evidence must use the same training-mode model state as the frozen gate")
    with torch.no_grad():
        predicted_pose, _ = model(csi)
        loss = torch.mean(torch.norm(predicted_pose - pose, dim=-1))
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite D1 checkpoint loss at step {step}")
    summary, arrays = analyze_prediction_batch(
        predicted_pose.detach().cpu().numpy(),
        pose.detach().cpu().numpy(),
        sample_ids,
        train_mean_pose=train_mean_pose,
    )
    summary["step"] = step
    summary["loss_stored_units"] = float(loss.detach().cpu())
    summary["mpjpe_stored_units"] = float(loss.detach().cpu())
    summary["sample_ids"] = list(sample_ids)
    return summary, arrays


def _draw_line_chart(image, rect, series, title, y_label):
    x, y, width, height = rect
    cv2.rectangle(image, (x, y), (x + width - 1, y + height - 1), (215, 215, 215), 1)
    cv2.putText(
        image,
        title,
        (x + 12, y + 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (25, 25, 25),
        1,
        cv2.LINE_AA,
    )
    left = x + 58
    right = x + width - 18
    top = y + 52
    bottom = y + height - 52
    all_values = np.concatenate([np.asarray(values, dtype=np.float64) for _, values, _ in series])
    minimum = float(np.min(all_values))
    maximum = float(np.max(all_values))
    span = maximum - minimum
    if span <= 1e-12:
        minimum -= 0.5
        maximum += 0.5
        span = maximum - minimum
    minimum -= span * 0.08
    maximum += span * 0.08
    span = maximum - minimum

    def point(index, value):
        point_x = int(left + (right - left) * index / 13)
        point_y = int(bottom - (bottom - top) * (value - minimum) / span)
        return point_x, point_y

    for axis_value in (minimum, (minimum + maximum) / 2, maximum):
        _, axis_y = point(0, axis_value)
        cv2.line(image, (left, axis_y), (right, axis_y), (230, 230, 230), 1)
        cv2.putText(
            image,
            f"{axis_value:.3f}",
            (x + 4, axis_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
    for joint in (0, 7, 13):
        joint_x, _ = point(joint, minimum)
        cv2.putText(
            image,
            str(joint),
            (joint_x - 5, bottom + 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (70, 70, 70),
            1,
            cv2.LINE_AA,
        )
    if minimum < 0 < maximum:
        zero_x, zero_y = point(0, 0)
        cv2.line(image, (left, zero_y), (right, zero_y), (190, 190, 190), 1)
    for label, values, color in series:
        points = [point(index, float(value)) for index, value in enumerate(values)]
        for index in range(len(points) - 1):
            cv2.line(image, points[index], points[index + 1], color, 2, cv2.LINE_AA)
        for point_xy in points:
            cv2.circle(image, point_xy, 3, color, -1, cv2.LINE_AA)
    legend_x = x + 12
    legend_y = y + height - 18
    for label, _, color in series:
        cv2.line(image, (legend_x, legend_y), (legend_x + 24, legend_y), color, 3, cv2.LINE_AA)
        cv2.putText(
            image,
            label,
            (legend_x + 30, legend_y + 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.34,
            (45, 45, 45),
            1,
            cv2.LINE_AA,
        )
        text_width = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.34, 1
        )[0][0]
        legend_x += 30 + text_width + 18
    cv2.putText(
        image,
        y_label,
        (x + 10, y + height - 34),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.34,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )


def plot_prediction_comparisons(sample_ids, target, predictions_by_step, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        0: (180, 80, 0),
        200: (0, 140, 255),
        800: (40, 160, 40),
    }
    paths = []
    for sample_index, sample_id in enumerate(sample_ids):
        image = np.full((820, 1500, 3), 255, dtype=np.uint8)
        cv2.putText(
            image,
            f"Step-4B-D1 {sample_id}: prediction vs ground truth",
            (25, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            "stored coordinates; coordinate order unchanged; axes shown by index only",
            (25, 62),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (80, 80, 80),
            1,
            cv2.LINE_AA,
        )
        for axis in range(3):
            series = [("target", target[sample_index, :, axis], (35, 35, 35))]
            for step in sorted(predictions_by_step):
                series.append(
                    (
                        f"step {step}",
                        predictions_by_step[step][sample_index, :, axis],
                        colors.get(step, (120, 120, 120)),
                    )
                )
            _draw_line_chart(
                image,
                (20 + axis * 495, 82, 470, 680),
                series,
                f"coordinate index {axis}",
                "joint index",
            )
        path = output_dir / f"{sample_id}_prediction_vs_ground_truth.png"
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not write plot: {path}")
        paths.append(path)
    return paths


def plot_mean_pose_comparison(target, predictions_by_step, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "init": (180, 80, 0),
        "step 200": (0, 140, 255),
        "step 800": (40, 160, 40),
    }
    image = np.full((820, 1500, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "Step-4B-D1 mean-pose / collapsed-output comparison",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        image,
        "black = fixed-batch ground-truth mean; colors = prediction means at recorded checkpoints",
        (25, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (80, 80, 80),
        1,
        cv2.LINE_AA,
    )
    step_by_label = {
        "init": 0,
        "step 200": 200,
        "step 800": 800,
    }
    for axis in range(3):
        series = [("target mean", np.mean(target[:, :, axis], axis=0), (35, 35, 35))]
        for label, color in colors.items():
            series.append(
                (
                    label,
                    np.mean(predictions_by_step[step_by_label[label]][:, :, axis], axis=0),
                    color,
                )
            )
        _draw_line_chart(
            image,
            (20 + axis * 495, 92, 470, 670),
            series,
            f"coordinate index {axis}",
            "joint index",
        )
    path = output_dir / "mean_pose_comparison.png"
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write plot: {path}")
    return path


def plot_per_sample_errors(sample_ids, summaries_by_step, output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((760, 1500, 3), 255, dtype=np.uint8)
    cv2.putText(
        image,
        "Step-4B-D1 per-sample MPJPE at recorded checkpoints",
        (25, 38),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    left, top = 80, 90
    width, height = 1350, 560
    values = []
    for step in sorted(summaries_by_step):
        values.extend(
            sample["mpjpe_stored_units"] for sample in summaries_by_step[step]["per_sample"]
        )
    maximum = max(float(max(values)), 1e-9) * 1.15
    colors = {0: (180, 80, 0), 200: (0, 140, 255), 800: (40, 160, 40)}
    cv2.rectangle(image, (left, top), (left + width, top + height), (220, 220, 220), 1)
    group_width = width / len(sample_ids)
    bar_width = int(group_width * 0.22)
    for sample_index, sample_id in enumerate(sample_ids):
        group_left = int(left + sample_index * group_width)
        for offset, step in enumerate(sorted(summaries_by_step)):
            value = summaries_by_step[step]["per_sample"][sample_index]["mpjpe_stored_units"]
            bar_height = int(height * min(value / maximum, 1.0))
            bar_x = group_left + int(group_width * 0.12) + offset * (bar_width + 4)
            bar_top = top + height - bar_height
            cv2.rectangle(image, (bar_x, bar_top), (bar_x + bar_width, top + height), colors[step], -1)
        cv2.putText(
            image,
            sample_id.replace("S11_06_", ""),
            (group_left + 8, top + height + 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            (60, 60, 60),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        image,
        f"0 to {maximum:.3f} stored units",
        (left, top - 12),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.42,
        (70, 70, 70),
        1,
        cv2.LINE_AA,
    )
    legend_x = left
    for step, color in colors.items():
        cv2.rectangle(image, (legend_x, top + height + 45), (legend_x + 20, top + height + 60), color, -1)
        cv2.putText(
            image,
            f"step {step}",
            (legend_x + 27, top + height + 58),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (50, 50, 50),
            1,
            cv2.LINE_AA,
        )
        legend_x += 125
    path = output_dir / "per_sample_mpjpe.png"
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write plot: {path}")
    return path


def _trajectory_summary(trajectory):
    losses = [record["pre_update_loss_stored_units"] for record in trajectory]
    every_interval = [
        {
            "step": record["step"],
            "pre_update_loss_stored_units": record["pre_update_loss_stored_units"],
            "pre_update_mpjpe_stored_units": record["pre_update_mpjpe_stored_units"],
        }
        for record in trajectory
        if record["step"] % D1_METRIC_INTERVAL_STEPS == 0
    ]
    last_100 = losses[-100:]
    previous_100 = losses[-200:-100]
    return {
        "first": losses[0],
        "last": losses[-1],
        "min": float(np.min(losses)),
        "max": float(np.max(losses)),
        "every_25_steps": every_interval,
        "mean_last_100_pre_update_loss": float(np.mean(last_100)),
        "mean_previous_100_pre_update_loss": float(np.mean(previous_100)),
        "last_100_change": float(np.mean(last_100) - np.mean(previous_100)),
    }


def run_diagnostic(source_artifact_root, d1_root, freeze_config, pilot_root, device_name):
    source_artifact_root, freeze_config, pilot_root = _resolve_source_paths(
        source_artifact_root, freeze_config, pilot_root
    )
    d1_root = Path(d1_root)
    verification = verify_frozen_inputs(source_artifact_root, freeze_config, pilot_root)
    _write_json(d1_root / "input_verification.json", verification)
    freeze_path, freeze_record = write_diagnostic_freeze(
        source_artifact_root, d1_root, freeze_config, pilot_root, verification
    )

    source_freeze = _load_json(freeze_config)
    config = source_freeze["config"]
    diagnostic_freeze = freeze_record["diagnostic_freeze"]
    original_steps = diagnostic_freeze["original_optimizer_steps"]
    diagnostic_steps = diagnostic_freeze["diagnostic_optimizer_steps"]
    if diagnostic_steps != D1_STEP_MULTIPLIER * original_steps:
        raise AssertionError("D1 diagnostic step budget is not exactly 4x the frozen budget")

    device = _resolve_device(device_name)
    _seed_everything(config["seed"])
    train_dataset = SinglePersonPiW3DDataset(
        pilot_root,
        pilot_root / "train_data" / "selected_train_ids.txt",
        expected_split="internal_train",
        split_assignments_path=freeze_config.parent / "split_assignments.json",
    )
    if train_dataset.sample_ids[: len(EXPECTED_ORIGINAL_BATCH_IDS)] != EXPECTED_ORIGINAL_BATCH_IDS:
        raise AssertionError("D1 loader did not reproduce the frozen ordered tiny batch")
    model = build_model(config, device)
    model.train()
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
    )
    ordered_loader = make_single_person_dataloader(
        train_dataset,
        is_training=False,
        generator=torch.Generator().manual_seed(config["seed"]),
        batch_size=config["overfit"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    batch = next(iter(ordered_loader))
    csi = batch["csi"].to(device)
    pose = batch["pose"].to(device)
    batch_sample_ids = list(train_dataset.sample_ids[: csi.shape[0]])

    start_time = time.perf_counter()
    train_mean_pose, train_mean_pose_count = compute_mean_pose(train_dataset)
    checkpoint_summaries = {}
    checkpoint_arrays = {}
    initial_summary, initial_arrays = _capture_checkpoint(
        model, csi, pose, batch_sample_ids, 0, train_mean_pose
    )
    checkpoint_summaries[0] = initial_summary
    checkpoint_arrays[0] = initial_arrays

    trajectory = []
    original_stop_summary = None
    original_stop_arrays = None
    for step in range(1, diagnostic_steps + 1):
        optimizer.zero_grad()
        predicted_pose, _ = model(csi)
        loss = torch.mean(torch.norm(predicted_pose - pose, dim=-1))
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite D1 loss at step {step}")
        loss.backward()
        gradient_stats = _gradient_statistics(model)
        if not gradient_stats["all_gradients_finite"]:
            raise ValueError(f"Non-finite D1 gradients at step {step}")
        optimizer.step()
        trajectory.append(
            {
                "step": step,
                "pre_update_loss_stored_units": float(loss.detach().cpu()),
                "pre_update_mpjpe_stored_units": float(loss.detach().cpu()),
                **gradient_stats,
            }
        )

        if step == original_steps:
            model_state = _clone_model_state(model)
            rng_state = _capture_rng_state()
            original_stop_summary, original_stop_arrays = _capture_checkpoint(
                model, csi, pose, batch_sample_ids, step, train_mean_pose
            )
            checkpoint_summaries[step] = original_stop_summary
            checkpoint_arrays[step] = original_stop_arrays
            _restore_model_state(model, model_state)
            _restore_rng_state(rng_state)

    final_summary, final_arrays = _capture_checkpoint(
        model, csi, pose, batch_sample_ids, diagnostic_steps, train_mean_pose
    )
    checkpoint_summaries[diagnostic_steps] = final_summary
    checkpoint_arrays[diagnostic_steps] = final_arrays
    runtime_seconds = time.perf_counter() - start_time

    for step, summary in checkpoint_summaries.items():
        _write_json(d1_root / f"checkpoint_step_{step:04d}.json", summary)
    np.savez(
        d1_root / "checkpoint_predictions.npz",
        sample_ids=np.array(batch_sample_ids),
        train_mean_pose=train_mean_pose[None, :, :],
        **{
            f"prediction_step_{step:04d}": arrays["prediction"][:, None, :, :]
            for step, arrays in checkpoint_arrays.items()
        },
        ground_truth=checkpoint_arrays[0]["target"][:, None, :, :],
    )
    _write_json(d1_root / "loss_trajectory.json", {"records": trajectory})

    initial = initial_summary["mpjpe_stored_units"]
    original_stop = original_stop_summary["mpjpe_stored_units"]
    final = final_summary["mpjpe_stored_units"]
    criterion = diagnostic_freeze["original_absolute_final_mpjpe_criterion_stored_units"]
    criterion_passed = final <= criterion
    finite_evidence = all(record["all_gradients_finite"] for record in trajectory) and all(
        np.isfinite(
            [record["pre_update_loss_stored_units"] for record in trajectory]
        )
    )
    if not finite_evidence:
        raise AssertionError("D1 completed without finite-loss/finite-gradient evidence")

    plots_dir = d1_root / "plots"
    prediction_plot_paths = plot_prediction_comparisons(
        batch_sample_ids,
        checkpoint_arrays[0]["target"],
        {step: arrays["prediction"] for step, arrays in checkpoint_arrays.items()},
        plots_dir,
    )
    mean_pose_plot_path = plot_mean_pose_comparison(
        checkpoint_arrays[0]["target"],
        {step: arrays["prediction"] for step, arrays in checkpoint_arrays.items()},
        plots_dir,
    )
    per_sample_plot_path = plot_per_sample_errors(
        batch_sample_ids, checkpoint_summaries, plots_dir
    )

    checkpoint_path = d1_root / "step4b_d1_final_state_dict.pt"
    torch.save(model.state_dict(), checkpoint_path)
    reloaded_model = build_model(config, device)
    reloaded_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    reloaded_model.eval()
    model.eval()
    with torch.no_grad():
        reloaded_prediction, _ = reloaded_model(csi)
        reloaded_loss = torch.mean(torch.norm(reloaded_prediction - pose, dim=-1))
        evaluation_prediction, _ = model(csi)
        evaluation_loss = torch.mean(torch.norm(evaluation_prediction - pose, dim=-1))
    if not torch.isfinite(reloaded_loss) or not torch.isfinite(evaluation_loss):
        raise ValueError("Non-finite checkpoint reload evaluation loss")
    if not torch.isclose(evaluation_loss, reloaded_loss, rtol=1e-6, atol=1e-8):
        raise AssertionError("Reloaded D1 checkpoint evaluation does not match the saved model")

    if criterion_passed:
        diagnostic_status = "EXTENDED_STEP_DIAGNOSTIC_PASSED"
        interpretation = (
            "original gate failed; extended-step diagnostic passed. The evidence indicates that "
            "the original optimizer-step budget was insufficient for this fixed tiny batch. This "
            "diagnostic does not claim generalization."
        )
    else:
        diagnostic_status = "EXTENDED_STEP_DIAGNOSTIC_ABOVE_CRITERION"
        interpretation = (
            "The extended-step diagnostic remained above the original absolute final-MPJPE "
            "criterion; see the recorded trend, mean-pose, and error-concentration evidence."
        )

    result = {
        "diagnostic_version": D1_VERSION,
        "diagnostic_run_status": "COMPLETE",
        "diagnostic_status": diagnostic_status,
        "interpretation": interpretation,
        "original_result_preserved": True,
        "source_step4b_decision": verification["source_step4b_decision"],
        "freeze_record": str(freeze_path),
        "device": str(device),
        "runtime_seconds": runtime_seconds,
        "seed": config["seed"],
        "batch_size": int(csi.shape[0]),
        "batch_sample_ids": batch_sample_ids,
        "original_steps": original_steps,
        "diagnostic_steps": diagnostic_steps,
        "step_multiplier": D1_STEP_MULTIPLIER,
        "metric_interval_steps": D1_METRIC_INTERVAL_STEPS,
        "optimizer": config["training"]["optimizer"],
        "learning_rate": config["training"]["learning_rate"],
        "momentum": config["training"]["momentum"],
        "loss": verification["source_loss"],
        "absolute_final_mpjpe_criterion_stored_units": criterion,
        "criterion_passed": criterion_passed,
        "all_losses_finite": True,
        "all_gradients_finite": True,
        "initial": initial_summary,
        "original_stop": original_stop_summary,
        "final": final_summary,
        "initial_mpjpe_stored_units": initial,
        "original_stop_mpjpe_stored_units": original_stop,
        "final_mpjpe_stored_units": final,
        "final_to_initial_mpjpe_ratio": final / initial if initial > 0 else None,
        "original_stop_to_initial_mpjpe_ratio": original_stop / initial if initial > 0 else None,
        "final_minus_original_stop_mpjpe_stored_units": final - original_stop,
        "trajectory_summary": _trajectory_summary(trajectory),
        "model_parameters": verification["source_parameter_counts"],
        "train_mean_pose_source_count": train_mean_pose_count,
        "coordinate_convention": verification["source_coordinate_convention"],
        "development_samples_loaded": False,
        "official_test_used": False,
        "checkpoint": {
            "path": str(checkpoint_path),
            "format": "state_dict",
            "reloaded_loss_stored_units": float(reloaded_loss.detach().cpu()),
            "reloaded_mpjpe_stored_units": float(reloaded_loss.detach().cpu()),
            "reload_matches_saved_model": True,
        },
        "plots": {
            "prediction_vs_ground_truth": [str(path) for path in prediction_plot_paths],
            "mean_pose_comparison": str(mean_pose_plot_path),
            "per_sample_mpjpe": str(per_sample_plot_path),
        },
    }
    _write_json(d1_root / "step4b_d1_result.json", result)
    _write_json(
        d1_root / "step4b_d1_decision.json",
        {
            "status": diagnostic_status,
            "reason": interpretation,
            "final_mpjpe_stored_units": final,
            "original_absolute_criterion_stored_units": criterion,
            "original_step4b_status": verification["source_step4b_decision"]["status"],
            "original_result_preserved": True,
            "result": str(d1_root / "step4b_d1_result.json"),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"d1_result={d1_root / 'step4b_d1_result.json'}")
    print(f"d1_decision={d1_root / 'step4b_d1_decision.json'}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run the frozen Step-4B-D1 4x under-training diagnostic.")
    parser.add_argument("--source-artifact-root", type=Path, default=DEFAULT_SOURCE_ARTIFACT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_D1_ROOT)
    parser.add_argument("--freeze-config", type=Path, default=None)
    parser.add_argument("--pilot-root", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_artifact_root, freeze_config, pilot_root = _resolve_source_paths(
        args.source_artifact_root, args.freeze_config, args.pilot_root
    )
    verification = verify_frozen_inputs(source_artifact_root, freeze_config, pilot_root)
    if args.verify_only:
        path = _write_json(args.artifact_root / "input_verification.json", verification)
        print(json.dumps(verification, indent=2, sort_keys=True))
        print(f"input_verification={path}")
        return 0
    return run_diagnostic(
        source_artifact_root,
        args.artifact_root,
        freeze_config,
        pilot_root,
        args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
