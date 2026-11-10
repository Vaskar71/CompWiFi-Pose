"""Step-5b Step 0: reproduce the premises of docs/STEP_5B_PLAN.md (read-only, CPU).

Reads only existing Step-5 artifacts (saved predictions, ``data_v1`` stored CSI, freeze ID
lists) and the keypoints of single-person *training-partition* recordings. Writes
``data/_artifacts/step5b_pose_signal/premises/premises.json`` and ``premises_table.md``.

"Recording" = candidate filename group (``sample_id.rsplit("_", 1)[0]``); no subject,
session or environment claim. Distances are stored units.
"""

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from piw3d_manifest import parse_sample_id


STEP5_ARTIFACTS = "data/_artifacts/step5_single_person"
OUTPUT_DIR = "data/_artifacts/step5b_pose_signal/premises"
SOURCE_KEYPOINTS = "data/person_in_wifi_3d/wifi_dataset/train_data/keypoint"
TRAIN_MANIFEST = "data/_artifacts/piw3d_candidate_group_manifests/internal_train_ids.txt"
LAGS = (1, 2, 4, 8, 16)
AMPLITUDE_ROWS = 90

# Values quoted in docs/STEP_5B_PLAN.md section 1 (rounded as printed there).
PLAN_VALUES = {
    "oracle_abs": 0.139,
    "oracle_rel": 0.095,
    "R1_seed0_abs": 0.204,
    "R1_seed0_rel": 0.102,
    "R2_seed0_abs": 0.231,
    "R2_seed0_rel": 0.104,
    "R1_seed0_share_beating_oracle": 0.16,
    "R2_seed0_share_beating_oracle": 0.10,
    "R1_seed0_median_deviation_correlation": 0.05,
    "R2_seed0_median_deviation_correlation": 0.04,
    "nearest_recording_mean_identification": 0.72,
    "autocorrelation_lag_1": 0.94,
    "autocorrelation_lag_2": 0.83,
    "autocorrelation_lag_4": 0.59,
    "autocorrelation_lag_8": 0.17,
    "autocorrelation_lag_16": -0.08,
    "within_recording_deviation": 0.092,
    "between_recording_mean_skeleton_deviation": 0.104,
}


# ----------------------------------------------------------------------------- helpers

def recording_of(sample_id):
    return sample_id.rsplit("_", 1)[0]


def relative(poses):
    """Subtract each skeleton's own 14-joint centroid. poses (..., 14, 3)."""
    poses = np.asarray(poses, dtype=np.float64)
    return poses - poses.mean(axis=-2, keepdims=True)


def mpjpe(pred, gt):
    return float(np.mean(np.linalg.norm(np.asarray(pred) - np.asarray(gt), axis=-1)))


def recording_means(sample_ids, values):
    """Mean of ``values`` (N, ...) per recording -> {recording: mean}."""
    groups = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        groups[recording_of(sample_id)].append(index)
    values = np.asarray(values, dtype=np.float64)
    return {name: values[indices].mean(axis=0) for name, indices in sorted(groups.items())}


def oracle_predictions(sample_ids, means):
    return np.stack([means[recording_of(sample_id)] for sample_id in sample_ids])


def _indices_by_recording(sample_ids):
    groups = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        groups[recording_of(sample_id)].append(index)
    return dict(sorted(groups.items()))


def share_recordings_beating_oracle(sample_ids, pred, gt, oracle):
    """Fraction of recordings where the model's abs MPJPE is below the oracle's."""
    flags = [
        mpjpe(pred[idx], gt[idx]) < mpjpe(oracle[idx], gt[idx])
        for idx in _indices_by_recording(sample_ids).values()
    ]
    return float(np.mean(flags)), len(flags)


def median_deviation_correlation(sample_ids, pred, gt):
    """Per recording: Pearson r between (pred - mean pred) and (gt - mean gt), flattened; median."""
    values = []
    for idx in _indices_by_recording(sample_ids).values():
        if len(idx) < 2:
            continue
        dp = pred[idx] - pred[idx].mean(axis=0)
        dg = gt[idx] - gt[idx].mean(axis=0)
        values.append(float(np.corrcoef(dp.ravel(), dg.ravel())[0, 1]))
    return float(np.median(values)), values


def nearest_mean_identification(train_x, train_recordings, eval_x, eval_recordings):
    """Assign each eval sample to the recording with the nearest (Euclidean) mean train block."""
    names = sorted(set(train_recordings))
    train_recordings = np.asarray(train_recordings)
    train_x = np.asarray(train_x, dtype=np.float64).reshape(len(train_recordings), -1)
    means = np.stack([train_x[train_recordings == name].mean(axis=0) for name in names])
    eval_x = np.asarray(eval_x, dtype=np.float64).reshape(len(eval_recordings), -1)
    distances = (
        np.square(eval_x).sum(axis=1)[:, None] - 2.0 * eval_x @ means.T + np.square(means).sum(axis=1)[None]
    )
    assigned = np.asarray(names)[np.argmin(distances, axis=1)]
    return float(np.mean(assigned == np.asarray(eval_recordings))), 1.0 / len(names)


def lag_autocorrelation(deviations, frames, lag):
    """Pearson r over all frame pairs whose frame indices differ by exactly ``lag``."""
    position = {int(frame): index for index, frame in enumerate(frames)}
    pairs = [(position[f], position[f + lag]) for f in position if f + lag in position]
    if len(pairs) < 2:
        return None
    a = np.asarray(deviations)[[p[0] for p in pairs]].ravel()
    b = np.asarray(deviations)[[p[1] for p in pairs]].ravel()
    return float(np.corrcoef(a, b)[0, 1])


def motion_statistics(recording_poses, lags=LAGS):
    """recording_poses: {recording: (frames sorted, poses (n,14,3))}."""
    autocorr = {lag: [] for lag in lags}
    within = []
    mean_skeletons = []
    for frames, poses in recording_poses.values():
        rel = relative(poses)
        mean_rel = rel.mean(axis=0)
        deviation = rel - mean_rel
        within.append(float(np.linalg.norm(deviation, axis=-1).mean()))
        mean_skeletons.append(mean_rel)
        for lag in lags:
            value = lag_autocorrelation(deviation, frames, lag)
            if value is not None:
                autocorr[lag].append(value)
    mean_skeletons = np.stack(mean_skeletons)
    between = float(np.linalg.norm(mean_skeletons - mean_skeletons.mean(axis=0), axis=-1).mean())
    return {
        "autocorrelation_median": {str(lag): float(np.median(v)) for lag, v in autocorr.items()},
        "autocorrelation_recordings": {str(lag): len(v) for lag, v in autocorr.items()},
        "within_recording_deviation": float(np.mean(within)),
        "between_recording_mean_skeleton_deviation": between,
        "recordings": len(recording_poses),
    }


def compare(measured, tolerance):
    rows = {}
    for key, quoted in PLAN_VALUES.items():
        value = measured.get(key)
        rows[key] = {
            "plan": quoted,
            "measured": value,
            "abs_difference": None if value is None else abs(value - quoted),
            "reproduces": bool(value is not None and abs(value - quoted) <= tolerance[key]),
        }
    return rows


def _tolerances():
    tolerance = {}
    for key, quoted in PLAN_VALUES.items():
        text = f"{abs(quoted):.3f}".rstrip("0")
        decimals = len(text.split(".")[1]) if "." in text else 0
        tolerance[key] = 0.5 * 10 ** (-max(decimals, 2)) + 1e-9
    return tolerance


# ----------------------------------------------------------------------------- io

def _read_ids(path):
    return [line.split()[0] for line in Path(path).read_text(encoding="utf-8-sig").splitlines() if line.split()]


def analyze(repo_root):
    from step5_subset import single_person_groups

    repo_root = Path(repo_root)
    step5 = repo_root / STEP5_ARTIFACTS
    measured = {}
    details = {"definitions": {}}

    # 1. recording-mean oracle on temporal_holdout
    for run in ("R1_seed0", "R2_seed0"):
        with np.load(step5 / "runs" / run / "predictions_train.npz") as tr, np.load(
            step5 / "runs" / run / "predictions_temporal_holdout.npz"
        ) as ho:
            train_ids = [str(s) for s in tr["sample_ids"]]
            train_gt = tr["ground_truths"][:, 0].astype(np.float64)
            ids = [str(s) for s in ho["sample_ids"]]
            gt = ho["ground_truths"][:, 0].astype(np.float64)
            pred = ho["predictions"][:, 0].astype(np.float64)
        oracle = oracle_predictions(ids, recording_means(train_ids, train_gt))
        measured["oracle_abs"] = mpjpe(oracle, gt)
        measured["oracle_rel"] = mpjpe(relative(oracle), relative(gt))
        measured[f"{run}_abs"] = mpjpe(pred, gt)
        measured[f"{run}_rel"] = mpjpe(relative(pred), relative(gt))
        share, count = share_recordings_beating_oracle(ids, pred, gt, oracle)
        measured[f"{run}_share_beating_oracle"] = share
        median_corr, _ = median_deviation_correlation(ids, pred, gt)
        measured[f"{run}_median_deviation_correlation"] = median_corr
        details[f"{run}_holdout_recordings"] = count
    details["definitions"]["oracle"] = "mean ground-truth pose of the recording in predictions_train.npz"
    details["definitions"]["share_beating_oracle"] = "fraction of recordings with model abs MPJPE < oracle abs MPJPE"
    details["definitions"]["deviation_correlation"] = (
        "per recording Pearson r of flattened (pred - recording mean pred) vs (gt - recording mean gt), "
        "absolute coordinates, temporal_holdout frames; median over recordings"
    )

    # 2. nearest-recording-mean identification on stored amplitude rows
    def _amplitude(split):
        ids = _read_ids(step5 / "freeze_v1" / f"{split}_ids.txt")
        blocks = np.stack(
            [
                np.load(step5 / "data_v1" / "train_data" / "csi_ap" / f"{s}.npy", allow_pickle=False)[
                    :, :AMPLITUDE_ROWS, :
                ].astype(np.float64)
                for s in ids
            ]
        )
        return ids, blocks

    train_ids, train_x = _amplitude("train")
    holdout_ids, holdout_x = _amplitude("temporal_holdout")
    accuracy, chance = nearest_mean_identification(
        train_x, [recording_of(s) for s in train_ids], holdout_x, [recording_of(s) for s in holdout_ids]
    )
    measured["nearest_recording_mean_identification"] = accuracy
    details["identification_chance"] = chance
    details["definitions"]["identification"] = (
        "mean stored csi_ap[:, :90, :] block per recording over Step-5 train frames; each temporal_holdout "
        "frame assigned to the Euclidean-nearest mean (full 3x90x20 block)"
    )

    # 3. pose motion statistics over all frames of the 51 single-person internal-train recordings
    groups = single_person_groups(_read_ids(repo_root / TRAIN_MANIFEST))
    keypoints = repo_root / SOURCE_KEYPOINTS
    recording_poses = {}
    for group, ids in groups.items():
        frames = [parse_sample_id(s)["frame"] for s in ids]
        poses = np.stack([np.load(keypoints / f"{s}.npy", allow_pickle=False)[0] for s in ids]).astype(np.float64)
        recording_poses[group] = (frames, poses)
    motion = motion_statistics(recording_poses)
    for lag in LAGS:
        measured[f"autocorrelation_lag_{lag}"] = motion["autocorrelation_median"][str(lag)]
    measured["within_recording_deviation"] = motion["within_recording_deviation"]
    measured["between_recording_mean_skeleton_deviation"] = motion["between_recording_mean_skeleton_deviation"]
    details["motion"] = motion
    details["motion_frames"] = int(sum(len(v[0]) for v in recording_poses.values()))
    details["definitions"]["autocorrelation"] = (
        "per recording, relative-pose deviation from the recording mean relative skeleton; Pearson r over all "
        "frame pairs whose frame indices differ by exactly the lag; median over recordings"
    )
    details["definitions"]["within"] = "mean over recordings of mean per-joint norm of the deviation"
    details["definitions"]["between"] = (
        "mean per-joint norm of each recording mean relative skeleton minus the unweighted mean of those skeletons"
    )

    comparison = compare(measured, _tolerances())
    return {
        "measured": measured,
        "comparison_to_plan": comparison,
        "all_reproduce": all(row["reproduces"] for row in comparison.values()),
        "not_reproduced": sorted(key for key, row in comparison.items() if not row["reproduces"]),
        "details": details,
        "distance_units": "stored units",
        "label_access": (
            "single-person training-partition labels and saved Step-5 temporal_holdout predictions only; "
            "no group_dev labels, no multi-person IDs, no official test_data"
        ),
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def premises_table(result):
    lines = ["| Quantity | Plan | Measured | Reproduces |", "|---|---|---|---|"]
    for key, row in result["comparison_to_plan"].items():
        lines.append(f"| {key} | {row['plan']} | {row['measured']:.4f} | {'yes' if row['reproduces'] else 'NO'} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Reproduce the Step-5b premises (read-only).")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    result = analyze(args.repo_root)
    out_dir = args.repo_root / OUTPUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "premises.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "premises_table.md").write_text(premises_table(result), encoding="utf-8")
    print(premises_table(result))
    print(json.dumps({"all_reproduce": result["all_reproduce"], "not_reproduced": result["not_reproduced"]}))


if __name__ == "__main__":
    main()
