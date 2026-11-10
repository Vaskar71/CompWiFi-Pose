"""Step-5 subset freeze (B1): train / temporal_holdout / group_dev single-person IDs.

Selection is deterministic (no seed). It reads only the internal candidate-group manifests,
filename single-person metadata, and, for the freeze summary, the keypoints of the selected
single-person training-partition IDs (permitted). Official ``test_data`` is never opened.
"""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from piw3d_manifest import candidate_group_hash, parse_sample_id
from step4b_subset import is_single_person_id, read_manifest_ids, select_evenly_spaced_frames
from step5_config import resolve_path, step5_config_for_json


SPLITS = ("train", "temporal_holdout", "group_dev")
ID_LIST_NAMES = {split: f"{split}_ids.txt" for split in SPLITS}
FREEZE_JSON_NAME = "step5_freeze.json"
SPLIT_ASSIGNMENTS_NAME = "split_assignments.json"


def _frame_sorted(sample_ids):
    return sorted(sample_ids, key=lambda sample_id: (parse_sample_id(sample_id)["frame"], sample_id))


def _group_sorted(sample_ids):
    return sorted(
        sample_ids,
        key=lambda sample_id: (
            parse_sample_id(sample_id)["group"],
            parse_sample_id(sample_id)["frame"],
            sample_id,
        ),
    )


def single_person_groups(sample_ids):
    grouped = defaultdict(list)
    for sample_id in sample_ids:
        if is_single_person_id(sample_id):
            grouped[parse_sample_id(sample_id)["group"]].append(sample_id)
    return {group: _frame_sorted(ids) for group, ids in sorted(grouped.items())}


def holdout_layout(n, subset_config):
    """Return (block_start, block_end, guard_low_start, guard_high_end) in sorted positions."""
    start = n // subset_config["holdout_start_divisor"]
    length = min(subset_config["holdout_length_cap"], n // subset_config["holdout_length_divisor"])
    if length <= 0:
        raise ValueError(f"Group with {n} frames is too short for a temporal-holdout block")
    guard = subset_config["guard_positions"]
    end = start + length
    guard_low = max(0, start - guard)
    guard_high = min(n, end + guard)
    if guard_low == 0 and start - guard < 0:
        raise ValueError(f"Guard band underflows for a group with {n} frames")
    return start, end, guard_low, guard_high


def split_train_group(ordered_ids, subset_config):
    n = len(ordered_ids)
    start, end, guard_low, guard_high = holdout_layout(n, subset_config)
    block = ordered_ids[start:end]
    guard_ids = ordered_ids[guard_low:start] + ordered_ids[end:guard_high]
    remaining = ordered_ids[:guard_low] + ordered_ids[guard_high:]
    train = select_evenly_spaced_frames(remaining, subset_config["train_frames_per_group"])
    holdout = select_evenly_spaced_frames(block, subset_config["temporal_holdout_frames_per_group"])
    return {
        "available_frames": n,
        "block_positions": [start, end],
        "guard_positions": [[guard_low, start], [end, guard_high]],
        "block_frame_range": [parse_sample_id(block[0])["frame"], parse_sample_id(block[-1])["frame"]],
        "guard_ids": guard_ids,
        "train": train,
        "temporal_holdout": holdout,
    }


def _sha256_text_lines(sample_ids):
    return hashlib.sha256("".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")).hexdigest()


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _centroid_summary(source_root, sample_ids):
    keypoint_dir = Path(source_root) / "train_data" / "keypoint"
    centroids = []
    for sample_id in sample_ids:
        pose = np.load(keypoint_dir / f"{sample_id}.npy", allow_pickle=False)
        if pose.shape != (1, 14, 3):
            raise ValueError(f"Unexpected single-person keypoint shape for {sample_id}: {pose.shape}")
        if not np.isfinite(pose).all():
            raise ValueError(f"Non-finite keypoints for {sample_id}")
        centroids.append(pose[0].astype(np.float64).mean(axis=0))
    centroids = np.stack(centroids)
    return {
        "count": int(centroids.shape[0]),
        "centroid_mean": centroids.mean(axis=0).tolist(),
        "centroid_std": centroids.std(axis=0).tolist(),
    }


def build_step5_selection(train_ids, development_ids, subset_config):
    if set(train_ids) & set(development_ids):
        raise AssertionError("Internal train/development manifests overlap")
    train_groups = single_person_groups(train_ids)
    development_groups = single_person_groups(development_ids)
    if set(train_groups) & set(development_groups):
        raise AssertionError("Single-person train/development candidate groups overlap")

    selected = {split: [] for split in SPLITS}
    group_records = {"train": [], "group_dev": []}
    for group, ordered_ids in train_groups.items():
        layout = split_train_group(ordered_ids, subset_config)
        selected["train"].extend(layout["train"])
        selected["temporal_holdout"].extend(layout["temporal_holdout"])
        group_records["train"].append(
            {
                "group": group,
                "candidate_group_hash": candidate_group_hash(group),
                "available_frames": layout["available_frames"],
                "block_positions": layout["block_positions"],
                "guard_positions": layout["guard_positions"],
                "block_frame_range": layout["block_frame_range"],
                "guard_id_count": len(layout["guard_ids"]),
                "guard_ids": layout["guard_ids"],
                "train_count": len(layout["train"]),
                "temporal_holdout_count": len(layout["temporal_holdout"]),
            }
        )
    for group, ordered_ids in development_groups.items():
        dev_ids = select_evenly_spaced_frames(ordered_ids, subset_config["group_dev_frames_per_group"])
        selected["group_dev"].extend(dev_ids)
        group_records["group_dev"].append(
            {
                "group": group,
                "candidate_group_hash": candidate_group_hash(group),
                "available_frames": len(ordered_ids),
                "group_dev_count": len(dev_ids),
            }
        )
    selected = {split: _group_sorted(ids) for split, ids in selected.items()}
    validate_selection(selected, train_ids, development_ids, group_records)
    return selected, group_records, train_groups, development_groups


def validate_selection(selected, train_ids, development_ids, group_records):
    train_set = set(train_ids)
    development_set = set(development_ids)
    all_ids = [sample_id for split in SPLITS for sample_id in selected[split]]
    if len(all_ids) != len(set(all_ids)):
        raise AssertionError("Step-5 splits overlap or contain duplicate IDs")
    for split in SPLITS:
        for sample_id in selected[split]:
            if not is_single_person_id(sample_id):
                raise AssertionError(f"Multi-person ID selected in {split}: {sample_id}")
    for split in ("train", "temporal_holdout"):
        outside = [sample_id for sample_id in selected[split] if sample_id not in train_set]
        if outside:
            raise AssertionError(f"{split} contains IDs outside the internal-train manifest: {outside[:5]}")
    outside = [sample_id for sample_id in selected["group_dev"] if sample_id not in development_set]
    if outside:
        raise AssertionError(f"group_dev contains IDs outside the internal-development manifest: {outside[:5]}")
    train_groups = {parse_sample_id(sample_id)["group"] for sample_id in selected["train"]}
    dev_groups = {parse_sample_id(sample_id)["group"] for sample_id in selected["group_dev"]}
    if train_groups & dev_groups:
        raise AssertionError("train and group_dev share candidate groups")
    guard_ids = {sample_id for record in group_records["train"] for sample_id in record["guard_ids"]}
    if guard_ids & set(all_ids):
        raise AssertionError("Guard-band IDs were selected into a split")
    return True


def freeze_step5_subset(repo_root=".", train_manifest=None, development_manifest=None, source_root=None, freeze_dir=None):
    config = step5_config_for_json()
    subset_config = config["subset"]
    train_manifest = Path(train_manifest) if train_manifest else resolve_path(repo_root, "train_manifest")
    development_manifest = (
        Path(development_manifest) if development_manifest else resolve_path(repo_root, "development_manifest")
    )
    source_root = Path(source_root) if source_root else resolve_path(repo_root, "source_root")
    freeze_dir = Path(freeze_dir) if freeze_dir else resolve_path(repo_root, "freeze_dir")
    if freeze_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing Step-5 freeze directory: {freeze_dir}")

    train_ids = read_manifest_ids(train_manifest)
    development_ids = read_manifest_ids(development_manifest)
    selected, group_records, train_groups, development_groups = build_step5_selection(
        train_ids, development_ids, subset_config
    )

    centroid_summary = {"train": [], "temporal_holdout": [], "group_dev": []}
    for split in SPLITS:
        by_group = defaultdict(list)
        for sample_id in selected[split]:
            by_group[parse_sample_id(sample_id)["group"]].append(sample_id)
        for group, ids in sorted(by_group.items()):
            record = _centroid_summary(source_root, ids)
            record["group"] = group
            centroid_summary[split].append(record)

    split_assignments = {sample_id: split for split in SPLITS for sample_id in selected[split]}
    freeze = {
        "algorithm_version": config["algorithm_version"],
        "subset_config": subset_config,
        "selection_rule": subset_config["selection_rule"],
        "split_labels": subset_config["split_labels"],
        "deterministic": True,
        "seed": None,
        "input_manifests": {
            "internal_train": {
                "path": str(Path(train_manifest).resolve()),
                "sha256": _sha256_file(train_manifest),
                "id_count": len(train_ids),
            },
            "internal_development": {
                "path": str(Path(development_manifest).resolve()),
                "sha256": _sha256_file(development_manifest),
                "id_count": len(development_ids),
            },
        },
        "source_root": str(source_root.resolve()),
        "counts": {
            "single_person_train_groups": len(train_groups),
            "single_person_development_groups": len(development_groups),
            **{f"{split}_ids": len(selected[split]) for split in SPLITS},
            "guard_ids_excluded": sum(record["guard_id_count"] for record in group_records["train"]),
        },
        "id_list_sha256": {split: _sha256_text_lines(selected[split]) for split in SPLITS},
        "selected_ids": selected,
        "group_records": group_records,
        "centroid_summary_stored_units": centroid_summary,
        "eligibility_checks": {
            "single_person_filename_metadata_only": True,
            "official_test_used": False,
            "multi_person_label_inspection": False,
            "keypoints_read_for": "selected single-person training-partition IDs only (freeze summary)",
        },
    }
    expected_train = subset_config["expected_train_groups"]
    expected_dev = subset_config["expected_development_groups"]
    freeze["counts"]["expected_groups_match"] = (
        len(train_groups) == expected_train and len(development_groups) == expected_dev
    )

    freeze_dir.mkdir(parents=True)
    for split in SPLITS:
        (freeze_dir / ID_LIST_NAMES[split]).write_text(
            "".join(f"{sample_id}\n" for sample_id in selected[split]), encoding="utf-8"
        )
    (freeze_dir / SPLIT_ASSIGNMENTS_NAME).write_text(
        json.dumps(split_assignments, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (freeze_dir / FREEZE_JSON_NAME).write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return freeze


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze the Step-5 single-person subset (deterministic).")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main():
    args = parse_args()
    freeze = freeze_step5_subset(repo_root=args.repo_root)
    print(json.dumps(freeze["counts"], indent=2, sort_keys=True))
    print(f"freeze={resolve_path(args.repo_root, 'freeze_dir') / FREEZE_JSON_NAME}")


if __name__ == "__main__":
    main()
