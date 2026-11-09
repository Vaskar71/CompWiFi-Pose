"""Read-only collapse diagnosis of the Step-4B-R single-person baseline.

Nothing under ``data/_artifacts/step4b_r`` or the pilot data is modified. The
script reads the saved predictions and final state dict, runs a CPU forward pass
in eval mode on a small fixed sample, and recomputes the target decomposition of
the pilot labels. It writes only the JSON and Markdown outputs named on the
command line (defaults below).

All distances are in stored units. No unit, axis, or joint-identity claim is made;
the 14-joint centroid is used as the translation reference.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from piw3d_manifest import parse_sample_id


DEFAULT_ARTIFACT_ROOT = Path("data/_artifacts/step4b_r")
DEFAULT_OUTPUT_JSON = Path("data/_artifacts/step4b_r_diagnosis/diagnosis.json")
DEFAULT_OUTPUT_MD = Path("docs/STEP_4B_R_COLLAPSE_DIAGNOSIS.md")
ACTIVATION_SAMPLES_PER_SPLIT = 16
ACTIVATION_POINTS = (
    "pooled_encoder_feature",
    "gconv1",
    "gconv2",
    "gconv3",
    "transformer_encoder1",
    "transformer_encoder2",
    "transformer_encoder3",
    "final_pose",
)

# Values stated in docs/NEXT_PHASE_PLAN.md section 1. ``decimals`` is the number of
# decimals the plan printed; a measured value reproduces if it rounds to the same
# printed number.
EXPECTED_SINGLE_POINT = (2.797, 1.784, 3.278)
EXPECTED_TARGET_DECOMPOSITION = {
    "absolute_mean_pose_mpjpe": {"train": (0.9898, 4), "development": (0.8014, 4)},
    "centroid_distance_to_train_mean_centroid": {"train": (0.968, 3), "development": (0.787, 3)},
    "centroid_relative_mean_skeleton_mpjpe": {"train": (0.155, 3), "development": (0.128, 3)},
}
EXPECTED_GROUP_COUNTS = {"train": 8, "development": 4}
# Plan values for activation spreads are given to ~1-2 significant figures and the
# plan does not state the exact std convention, so these are pattern checks only.
EXPECTED_ACTIVATION_PATTERN = {
    "pooled_encoder_feature": {"across_samples": 0.18},
    "gconv3": {"across_samples": 0.47, "across_joints": 3.28},
    "transformer_encoder1": {"across_samples": 4e-4, "across_joints": 1.5e-2},
    "transformer_encoder3": {"across_samples": 4e-7, "across_joints": 3e-6},
}
PATTERN_FACTOR = 3.0


# --------------------------------------------------------------------------- #
# Pure statistics helpers (numpy only; unit-tested on synthetic arrays)
# --------------------------------------------------------------------------- #
def _as_joint_array(poses):
    array = np.asarray(poses, dtype=np.float64)
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected poses shaped (N, joints, 3) or (N, 1, joints, 3), got {array.shape}")
    return array


def prediction_spread(predictions):
    """Spread of predicted poses across samples and across joints."""
    poses = _as_joint_array(predictions)
    std_across_samples = poses.std(axis=0)  # (joints, 3)
    centroids = poses.mean(axis=1)  # (N, 3)
    centred = poses - centroids[:, None, :]
    joint_extent = np.linalg.norm(centred, axis=-1)  # (N, joints)
    joint_zero = poses[:, :1, :]
    first_joint_offsets = np.linalg.norm(poses - joint_zero, axis=-1)
    return {
        "sample_count": int(poses.shape[0]),
        "std_across_samples_max_over_joints_and_axes": float(std_across_samples.max()),
        "std_across_samples_mean_over_joints_and_axes": float(std_across_samples.mean()),
        "std_across_samples_per_joint": [float(v) for v in std_across_samples.mean(axis=1)],
        "centroid_std_across_samples_per_axis": [float(v) for v in centroids.std(axis=0)],
        "mean_distance_of_joint_from_pose_centroid": float(joint_extent.mean()),
        "max_distance_between_joint0_and_any_joint": float(first_joint_offsets.max()),
        "mean_pose_point_per_joint": [[float(v) for v in joint] for joint in poses.mean(axis=0)],
    }


def single_point_summary(predictions, tolerance=1e-4):
    """Whether every joint of every sample is (numerically) one common 3D point."""
    poses = _as_joint_array(predictions)
    flat = poses.reshape(-1, 3)
    point = flat.mean(axis=0)
    max_deviation = float(np.abs(flat - point).max())
    return {
        "common_point": [float(v) for v in point],
        "max_abs_deviation_from_common_point": max_deviation,
        "is_single_point": bool(max_deviation <= tolerance),
        "tolerance": float(tolerance),
    }


def activation_statistics(activation):
    """Std across samples / joints and exact-zero fraction for one activation.

    ``across_samples`` is the population std along axis 0 averaged over every other
    axis. ``across_joints`` is the population std along axis 1 averaged over the
    remaining axes and exists only for tensors with a joint axis (rank >= 3);
    pooled (B, D) features report ``None``.
    """
    array = np.asarray(activation, dtype=np.float64)
    if array.ndim < 2:
        raise ValueError("Activation must have a batch axis and at least one feature axis")
    if array.ndim == 4 and array.shape[1] == 1:
        array = array[:, 0]
    across_samples = float(array.std(axis=0).mean())
    across_joints = float(array.std(axis=1).mean()) if array.ndim >= 3 else None
    return {
        "shape": [int(v) for v in np.asarray(activation).shape],
        "std_across_samples": across_samples,
        "std_across_joints": across_joints,
        "fraction_exact_zeros": float((array == 0.0).mean()),
        "mean_abs": float(np.abs(array).mean()),
        "all_finite": bool(np.isfinite(array).all()),
    }


def pose_centroids(poses):
    return _as_joint_array(poses).mean(axis=1)


def target_decomposition(train_poses, evaluation_poses):
    """Absolute vs translation vs centroid-relative error of the train-mean predictors.

    The mean pose, mean centroid and mean relative skeleton all come from the train
    poses only.
    """
    train = _as_joint_array(train_poses)
    evaluation = _as_joint_array(evaluation_poses)
    mean_pose = train.mean(axis=0)
    train_centroids = train.mean(axis=1)
    mean_centroid = train_centroids.mean(axis=0)
    mean_relative = (train - train_centroids[:, None, :]).mean(axis=0)

    def per_split(poses):
        centroids = poses.mean(axis=1)
        relative = poses - centroids[:, None, :]
        return {
            "sample_count": int(poses.shape[0]),
            "absolute_mean_pose_mpjpe": float(np.linalg.norm(poses - mean_pose, axis=-1).mean()),
            "centroid_distance_to_train_mean_centroid": float(
                np.linalg.norm(centroids - mean_centroid, axis=-1).mean()
            ),
            "centroid_relative_mean_skeleton_mpjpe": float(
                np.linalg.norm(relative - mean_relative, axis=-1).mean()
            ),
        }

    return {
        "train": per_split(train),
        "development": per_split(evaluation),
        "train_mean_centroid": [float(v) for v in mean_centroid],
    }


def group_centroid_statistics(sample_ids, poses):
    """Per candidate-filename-group centroid mean/std (labels of single-person samples)."""
    array = _as_joint_array(poses)
    if len(sample_ids) != array.shape[0]:
        raise ValueError("sample_ids and poses must have the same length")
    members = {}
    for index, sample_id in enumerate(sample_ids):
        members.setdefault(parse_sample_id(sample_id)["group"], []).append(index)
    centroids = array.mean(axis=1)
    statistics = {}
    for group in sorted(members):
        group_centroids = centroids[members[group]]
        per_axis_std = group_centroids.std(axis=0)
        statistics[group] = {
            "sample_count": len(members[group]),
            "centroid_mean": [float(v) for v in group_centroids.mean(axis=0)],
            "centroid_std_per_axis": [float(v) for v in per_axis_std],
            "centroid_std_norm": float(np.linalg.norm(per_axis_std)),
        }
    return statistics


def evenly_spaced_indices(total, count):
    if total <= 0 or count <= 0:
        raise ValueError("total and count must be positive")
    count = min(count, total)
    return [int(v) for v in np.unique(np.linspace(0, total - 1, count).round().astype(int))]


def round_matches(measured, printed, decimals):
    """True when ``measured`` rounds to the number the plan printed."""
    return bool(abs(measured - printed) <= 0.5 * 10 ** (-decimals) + 1e-12)


def pattern_matches(measured, expected, factor=PATTERN_FACTOR):
    if measured is None or expected is None or measured <= 0 or expected <= 0:
        return False
    ratio = measured / expected
    return bool(1.0 / factor <= ratio <= factor)


def check_premises(report):
    """Compare measurements to the plan's stated premises. Returns a list of checks."""
    checks = []
    point = report["predictions"]["single_point"]
    spread = report["predictions"]["spread"]
    checks.append(
        {
            "name": "all stored development predictions are one common 3D point",
            "kind": "exact",
            "measured": point["max_abs_deviation_from_common_point"],
            "expected": f"<= {point['tolerance']}",
            "passed": point["is_single_point"],
        }
    )
    checks.append(
        {
            "name": "that point rounds to [2.797, 1.784, 3.278]",
            "kind": "rounding",
            "measured": point["common_point"],
            "expected": list(EXPECTED_SINGLE_POINT),
            "passed": all(
                round_matches(measured, printed, 3)
                for measured, printed in zip(point["common_point"], EXPECTED_SINGLE_POINT)
            ),
        }
    )
    checks.append(
        {
            "name": "prediction std across the 512 samples is ~0",
            "kind": "exact",
            "measured": spread["std_across_samples_max_over_joints_and_axes"],
            "expected": "<= 1e-4",
            "passed": spread["std_across_samples_max_over_joints_and_axes"] <= 1e-4,
        }
    )
    for metric, per_split in EXPECTED_TARGET_DECOMPOSITION.items():
        for split, (printed, decimals) in per_split.items():
            measured = report["target_decomposition"][split][metric]
            checks.append(
                {
                    "name": f"{metric} ({split})",
                    "kind": "rounding",
                    "measured": measured,
                    "expected": printed,
                    "passed": round_matches(measured, printed, decimals),
                }
            )
    for split, expected_count in EXPECTED_GROUP_COUNTS.items():
        measured = len(report["group_centroids"][split])
        checks.append(
            {
                "name": f"candidate-group count ({split})",
                "kind": "exact",
                "measured": measured,
                "expected": expected_count,
                "passed": measured == expected_count,
            }
        )
    activations = report["activations"]["combined"]
    for point_name, expectations in EXPECTED_ACTIVATION_PATTERN.items():
        for key, expected in expectations.items():
            measured = activations[point_name][f"std_{key}"]
            checks.append(
                {
                    "name": f"activation {point_name} std {key.replace('_', ' ')}",
                    "kind": f"pattern (within {PATTERN_FACTOR:g}x; plan gives ~1-2 significant figures)",
                    "measured": measured,
                    "expected": expected,
                    "passed": pattern_matches(measured, expected),
                }
            )
    return checks


# --------------------------------------------------------------------------- #
# Artifact-reading parts (torch is imported lazily)
# --------------------------------------------------------------------------- #
def load_stored_predictions(artifact_root):
    path = Path(artifact_root) / "baseline" / "development_predictions.npz"
    with np.load(path, allow_pickle=False) as stored:
        return {name: stored[name] for name in stored.files}


def read_poses(dataset):
    poses = [dataset.read_pose(sample["keypoint_path"]).numpy().astype(np.float64)[0] for sample in dataset.samples]
    return np.stack(poses, axis=0)


def collect_activations(model, dataset, indices, torch_module):
    """Eval-mode CPU forward pass; returns activation arrays keyed by ACTIVATION_POINTS."""
    captured = {}
    handles = []
    for name in ("gconv1", "gconv2", "gconv3", "transformer_encoder1", "transformer_encoder2", "transformer_encoder3"):
        module = getattr(model, name)
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, key=name: captured.__setitem__(key, output.detach().cpu().numpy())
            )
        )
    try:
        batch = torch_module.stack([dataset[index]["csi"] for index in indices])
        was_training = model.training
        model.eval()
        with torch_module.no_grad():
            pose, pooled = model(batch)
        model.train(was_training)
    finally:
        for handle in handles:
            handle.remove()
    captured["pooled_encoder_feature"] = pooled.detach().cpu().numpy()
    captured["final_pose"] = pose.detach().cpu().numpy()
    return captured


def run_diagnosis(artifact_root, activation_samples):
    import torch

    import step4b_r_baseline as baseline
    from step4b_r_config import step4b_r_config_for_json

    baseline.load_runtime_modules()
    config = step4b_r_config_for_json()
    freeze_config = Path(config["data"]["freeze_config"])
    pilot_root = Path(config["data"]["pilot_root"])
    train_dataset, development_dataset = baseline._make_datasets(freeze_config, pilot_root)

    stored = load_stored_predictions(artifact_root)
    predictions = stored["model_predictions"]

    train_poses = read_poses(train_dataset)
    development_poses = read_poses(development_dataset)

    device = torch.device("cpu")
    model = baseline.build_step4b_r_model(config, device)
    state_path = Path(artifact_root) / "baseline" / "final_state_dict.pt"
    model.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))

    train_indices = evenly_spaced_indices(len(train_dataset), activation_samples)
    development_indices = evenly_spaced_indices(len(development_dataset), activation_samples)
    train_activations = collect_activations(model, train_dataset, train_indices, torch)
    development_activations = collect_activations(model, development_dataset, development_indices, torch)
    combined_activations = {
        key: np.concatenate([train_activations[key], development_activations[key]], axis=0)
        for key in ACTIVATION_POINTS
    }

    def activation_block(activations):
        return {key: activation_statistics(activations[key]) for key in ACTIVATION_POINTS}

    report = {
        "diagnosis_version": "compwifi-step4b-r-collapse-diagnosis-v1",
        "read_only_on_artifacts": True,
        "distance_units": "stored units (no unit conversion; 14-joint centroid used as translation reference)",
        "inputs": {
            "development_predictions": str(Path(artifact_root) / "baseline" / "development_predictions.npz"),
            "final_state_dict": str(state_path),
            "freeze_config": str(freeze_config),
            "pilot_root": str(pilot_root),
            "train_sample_count": len(train_dataset),
            "development_sample_count": len(development_dataset),
        },
        "predictions": {
            "sample_count": int(predictions.shape[0]),
            "spread": prediction_spread(predictions),
            "single_point": single_point_summary(predictions),
            "ground_truth_spread": prediction_spread(stored["ground_truths"]),
            "stored_mean_pose_prediction_matches_recomputed_train_mean": bool(
                np.allclose(stored["mean_pose"][0], train_poses.mean(axis=0), atol=1e-9)
            ),
        },
        "activations": {
            "definition": (
                "eval mode, CPU, no gradient; std is population std; 'across_samples' = std over the batch axis "
                "averaged over other axes; 'across_joints' = std over the joint axis averaged over the rest "
                "(not defined for the pooled (B, D) feature)"
            ),
            "train_sample_ids": [train_dataset.sample_ids[i] for i in train_indices],
            "development_sample_ids": [development_dataset.sample_ids[i] for i in development_indices],
            "combined": activation_block(combined_activations),
            "train_only": activation_block(train_activations),
            "development_only": activation_block(development_activations),
        },
        "target_decomposition": target_decomposition(train_poses, development_poses),
        "group_centroids": {
            "train": group_centroid_statistics(train_dataset.sample_ids, train_poses),
            "development": group_centroid_statistics(development_dataset.sample_ids, development_poses),
        },
    }
    report["premise_checks"] = check_premises(report)
    report["all_premises_reproduced"] = all(check["passed"] for check in report["premise_checks"])
    return report


# --------------------------------------------------------------------------- #
# Output
# --------------------------------------------------------------------------- #
def _fmt(value, digits=4):
    if value is None:
        return "n/a"
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v, digits) for v in value) + "]"
    if isinstance(value, bool) or isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return value
    if value != 0 and (abs(value) < 1e-3 or abs(value) >= 1e4):
        return f"{value:.3e}"
    return f"{value:.{digits}f}"


def render_markdown(report):
    verdict = "REPRODUCED" if report["all_premises_reproduced"] else "NOT REPRODUCED - STOP AND REPORT"
    lines = [
        "# Step-4B-R Collapse Diagnosis",
        "",
        "Read-only diagnosis of the Step-4B-R single-person baseline (`data/_artifacts/step4b_r/baseline`),",
        "generated by `diagnose_step4b_r_collapse.py`. All distances are in stored units; no unit,",
        "axis or joint-identity claim is made (the 14-joint centroid is the translation reference).",
        "Nothing in the baseline artifacts or pilot data was modified.",
        "",
        f"**Premise status (docs/NEXT_PHASE_PLAN.md section 1): {verdict}.**",
        "",
        "## 1. Stored development predictions",
        "",
    ]
    predictions = report["predictions"]
    point = predictions["single_point"]
    spread = predictions["spread"]
    lines += [
        f"- Samples: {predictions['sample_count']}.",
        f"- Common point of all predicted joints: {_fmt(point['common_point'], 4)}; maximum absolute deviation of any "
        f"coordinate from it: {_fmt(point['max_abs_deviation_from_common_point'])} "
        f"(single point: {point['is_single_point']}).",
        f"- Std across samples (max over joints and axes): {_fmt(spread['std_across_samples_max_over_joints_and_axes'])}; "
        f"mean over joints and axes: {_fmt(spread['std_across_samples_mean_over_joints_and_axes'])}.",
        f"- Mean distance of a predicted joint from its own pose centroid: "
        f"{_fmt(spread['mean_distance_of_joint_from_pose_centroid'])}.",
        f"- For contrast, ground-truth std across samples (mean over joints and axes): "
        f"{_fmt(predictions['ground_truth_spread']['std_across_samples_mean_over_joints_and_axes'])}; ground-truth "
        f"mean joint-to-centroid distance: "
        f"{_fmt(predictions['ground_truth_spread']['mean_distance_of_joint_from_pose_centroid'])}.",
        f"- Stored mean pose equals the recomputed train-label mean: "
        f"{predictions['stored_mean_pose_prediction_matches_recomputed_train_mean']}.",
        "",
        "## 2. Activation statistics of the reloaded `final_state_dict.pt`",
        "",
        report["activations"]["definition"] + ".",
        f"Samples: {len(report['activations']['train_sample_ids'])} train + "
        f"{len(report['activations']['development_sample_ids'])} development (evenly spaced indices).",
        "",
        "| Point | Shape | Std across samples | Std across joints | Fraction exact zeros |",
        "|---|---|---|---|---|",
    ]
    for key in ACTIVATION_POINTS:
        stats = report["activations"]["combined"][key]
        lines.append(
            f"| {key} | {tuple(stats['shape'])} | {_fmt(stats['std_across_samples'])} | "
            f"{_fmt(stats['std_across_joints'])} | {_fmt(stats['fraction_exact_zeros'])} |"
        )
    lines += [
        "",
        "Train-only / development-only values are stored in `diagnosis.json`.",
        "",
        "## 3. Target decomposition (train-label statistics only)",
        "",
        "| Quantity | Train | Development |",
        "|---|---|---|",
    ]
    decomposition = report["target_decomposition"]
    for metric in EXPECTED_TARGET_DECOMPOSITION:
        lines.append(
            f"| {metric.replace('_', ' ')} | {_fmt(decomposition['train'][metric])} | "
            f"{_fmt(decomposition['development'][metric])} |"
        )
    lines += [
        "",
        f"Train-mean centroid: {_fmt(decomposition['train_mean_centroid'])}.",
        "",
        "### Per candidate-filename-group centroid (label statistics of single-person samples)",
        "",
        "Groups are candidate filename groups only; no subject/recording/environment claim is made.",
        "",
    ]
    for split in ("train", "development"):
        lines += [
            f"**{split}** ({len(report['group_centroids'][split])} groups)",
            "",
            "| Group | Samples | Centroid mean | Centroid std per axis | Std norm |",
            "|---|---|---|---|---|",
        ]
        for group, stats in report["group_centroids"][split].items():
            lines.append(
                f"| {group} | {stats['sample_count']} | {_fmt(stats['centroid_mean'], 3)} | "
                f"{_fmt(stats['centroid_std_per_axis'], 3)} | {_fmt(stats['centroid_std_norm'], 3)} |"
            )
        lines.append("")
    lines += [
        "## 4. Premise checks against docs/NEXT_PHASE_PLAN.md section 1",
        "",
        "| Check | Kind | Measured | Expected | Result |",
        "|---|---|---|---|---|",
    ]
    for check in report["premise_checks"]:
        lines.append(
            f"| {check['name']} | {check['kind']} | {_fmt(check['measured'])} | {_fmt(check['expected'])} | "
            f"{'PASS' if check['passed'] else 'FAIL'} |"
        )
    lines += ["", f"Overall: **{verdict}**.", ""]
    return "\n".join(lines)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--artifact_root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output_json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output_md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--activation_samples", type=int, default=ACTIVATION_SAMPLES_PER_SPLIT)
    return parser.parse_args(argv)


def _assert_outputs_outside_baseline(artifact_root, outputs):
    protected = Path(artifact_root).resolve()
    for output in outputs:
        resolved = Path(output).resolve()
        if protected == resolved or protected in resolved.parents:
            raise ValueError(f"Refusing to write inside the read-only baseline artifacts: {output}")


def main(argv=None):
    args = parse_args(argv)
    _assert_outputs_outside_baseline(args.artifact_root, [args.output_json, args.output_md])
    report = run_diagnosis(args.artifact_root, args.activation_samples)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(report), encoding="utf-8", newline="\n")
    failed = [check["name"] for check in report["premise_checks"] if not check["passed"]]
    print(f"Wrote {args.output_json} and {args.output_md}")
    print("Premises reproduced:", report["all_premises_reproduced"])
    for name in failed:
        print("  FAILED:", name)
    return 0 if report["all_premises_reproduced"] else 2


if __name__ == "__main__":
    sys.exit(main())
