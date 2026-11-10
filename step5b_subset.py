"""Step-5b dense freeze ``freeze_v2`` (docs/STEP_5B_PLAN.md section 4).

Deterministic (no seed). Reads the internal candidate-group manifests (filename single-person
metadata selects the labelled single-person subset, as in Step 5) and the frozen Step-5
``group_dev`` ID list. Official ``test_data`` is never opened; no keypoint is read here.
"""

import argparse
import hashlib
import json
from pathlib import Path

from piw3d_manifest import parse_sample_id
from step4b_subset import is_single_person_id, read_manifest_ids, select_evenly_spaced_frames
from step5_subset import holdout_layout, single_person_groups
from step5b_config import ROLES, VIEWS, resolve_path, step5b_config_for_json


FREEZE_JSON_NAME = "step5b_freeze.json"
SPLIT_ASSIGNMENTS_NAME = "split_assignments.json"
VIEW_FILE = "{view}_ids.txt"


def _sha256_lines(sample_ids):
    return hashlib.sha256("".join(f"{s}\n" for s in sample_ids).encode("utf-8")).hexdigest()


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _sorted_ids(sample_ids):
    return sorted(sample_ids, key=lambda s: (parse_sample_id(s)["group"], parse_sample_id(s)["frame"], s))


def recording_hash(name):
    return hashlib.sha256(name.encode("utf-8")).hexdigest()


def validation_recordings(recording_names, per_scene):
    """First ``per_scene`` names per scene prefix by ascending sha256(name). Names only."""
    by_scene = {}
    for name in sorted(set(recording_names)):
        by_scene.setdefault(name.split("_", 1)[0], []).append(name)
    selected = []
    for scene in sorted(by_scene):
        ordered = sorted(by_scene[scene], key=recording_hash)
        selected.extend(ordered[:per_scene])
    return selected


def dense_layout(ordered_ids, subset_config):
    """Frame roles of one recording (ids sorted by frame): train / temporal_holdout / guard."""
    n = len(ordered_ids)
    start, end, guard_low, guard_high = holdout_layout(n, subset_config)
    return {
        "available_frames": n,
        "block_positions": [start, end],
        "guard_positions": [[guard_low, start], [end, guard_high]],
        "temporal_holdout": ordered_ids[start:end],
        "guard": ordered_ids[guard_low:start] + ordered_ids[end:guard_high],
        "train": ordered_ids[:guard_low] + ordered_ids[guard_high:],
    }


def build_step5b_views(train_manifest_ids, group_dev_ids, subset_config):
    groups = single_person_groups(train_manifest_ids)
    validation = validation_recordings(groups, subset_config["validation_recordings_per_scene"])
    validation_set = set(validation)
    sparse_n = subset_config["sparse_frames_per_recording"]
    views = {view: [] for view in VIEWS}
    guard_ids = []
    records = []
    for group, ordered in groups.items():
        layout = dense_layout(ordered, subset_config)
        sparse = select_evenly_spaced_frames(layout["train"], sparse_n)
        is_validation = group in validation_set
        views["all_train_dense"].extend(layout["train"])
        views["all_train_sparse"].extend(sparse)
        views["all_holdout"].extend(layout["temporal_holdout"])
        if is_validation:
            views["group_val"].extend(layout["train"])
        else:
            views["fit_train_dense"].extend(layout["train"])
            views["fit_train_sparse"].extend(sparse)
            views["fit_holdout"].extend(layout["temporal_holdout"])
        guard_ids.extend(layout["guard"])
        records.append(
            {
                "recording": group,
                "partition": "validation" if is_validation else "fit",
                "available_frames": layout["available_frames"],
                "block_positions": layout["block_positions"],
                "guard_positions": layout["guard_positions"],
                "train_frames": len(layout["train"]),
                "sparse_frames": len(sparse),
                "holdout_frames": len(layout["temporal_holdout"]),
                "guard_frames": len(layout["guard"]),
            }
        )
    views["group_dev"] = list(group_dev_ids)
    views = {view: _sorted_ids(ids) for view, ids in views.items()}
    assignments = {}
    for sample_id in views["all_train_dense"]:
        assignments[sample_id] = "train"
    for sample_id in views["all_holdout"]:
        assignments[sample_id] = "temporal_holdout"
    for sample_id in views["group_dev"]:
        assignments[sample_id] = "group_dev"
    validate_views(views, assignments, guard_ids, validation, train_manifest_ids)
    return views, assignments, sorted(guard_ids), validation, records


def validate_views(views, assignments, guard_ids, validation, train_manifest_ids):
    train_manifest = set(train_manifest_ids)
    sets = {view: set(ids) for view, ids in views.items()}
    for view, ids in views.items():
        if len(ids) != len(sets[view]):
            raise AssertionError(f"Duplicate IDs in view {view}")
        for sample_id in ids:
            if not is_single_person_id(sample_id):
                raise AssertionError(f"Multi-person ID in view {view}: {sample_id}")
    for view in ("fit_train_dense", "fit_train_sparse", "fit_holdout"):
        if sets[view] & sets["group_val"]:
            raise AssertionError(f"{view} overlaps group_val")
        recordings = {parse_sample_id(s)["group"] for s in sets[view]}
        if recordings & set(validation):
            raise AssertionError(f"{view} contains validation recordings")
    if {parse_sample_id(s)["group"] for s in sets["group_val"]} - set(validation):
        raise AssertionError("group_val contains non-validation recordings")
    if not sets["fit_train_sparse"] <= sets["fit_train_dense"]:
        raise AssertionError("fit_train_sparse is not a subset of fit_train_dense")
    if not sets["all_train_sparse"] <= sets["all_train_dense"]:
        raise AssertionError("all_train_sparse is not a subset of all_train_dense")
    if sets["all_train_dense"] & sets["all_holdout"]:
        raise AssertionError("train and holdout roles overlap")
    guard = set(guard_ids)
    used = set(assignments)
    if guard & used:
        raise AssertionError("Guard frames were assigned a role")
    for view in ("all_train_dense", "all_holdout", "fit_train_dense", "fit_holdout", "group_val"):
        outside = sets[view] - train_manifest
        if outside:
            raise AssertionError(f"{view} has IDs outside the internal-train manifest")
    train_recordings = {parse_sample_id(s)["group"] for s in sets["all_train_dense"]}
    if {parse_sample_id(s)["group"] for s in sets["group_dev"]} & train_recordings:
        raise AssertionError("group_dev shares recordings with train")
    if sets["fit_train_dense"] | sets["group_val"] != sets["all_train_dense"]:
        raise AssertionError("fit_train_dense + group_val != all_train_dense")
    return True


def read_group_dev_ids(step5_freeze_dir):
    step5_freeze_dir = Path(step5_freeze_dir)
    ids = [line.strip() for line in (step5_freeze_dir / "group_dev_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
    recorded = json.loads((step5_freeze_dir / "step5_freeze.json").read_text(encoding="utf-8"))["id_list_sha256"]["group_dev"]
    if _sha256_lines(ids) != recorded:
        raise AssertionError("Step-5 group_dev ID list does not match its recorded SHA-256")
    return ids, recorded


def freeze_step5b_subset(
    repo_root=".", train_manifest=None, development_manifest=None, step5_freeze_dir=None, freeze_dir=None,
    subset_config=None,
):
    """``subset_config`` overrides the frozen subset block (tests only)."""
    config = step5b_config_for_json()
    subset = subset_config if subset_config is not None else config["subset"]
    train_manifest = Path(train_manifest) if train_manifest else resolve_path(repo_root, "train_manifest")
    development_manifest = Path(development_manifest) if development_manifest else resolve_path(repo_root, "development_manifest")
    step5_freeze_dir = Path(step5_freeze_dir) if step5_freeze_dir else resolve_path(repo_root, "step5_freeze_dir")
    freeze_dir = Path(freeze_dir) if freeze_dir else resolve_path(repo_root, "freeze_dir")
    if freeze_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze directory: {freeze_dir}")

    train_ids = read_manifest_ids(train_manifest)
    development_ids = set(read_manifest_ids(development_manifest))
    group_dev_ids, group_dev_sha = read_group_dev_ids(step5_freeze_dir)
    if set(group_dev_ids) - development_ids:
        raise AssertionError("group_dev IDs outside the internal-development manifest")
    views, assignments, guard_ids, validation, records = build_step5b_views(train_ids, group_dev_ids, subset)

    counts = {view: len(ids) for view, ids in views.items()}
    counts["guard"] = len(guard_ids)
    counts["train_recordings"] = len(records)
    counts["fit_recordings"] = sum(r["partition"] == "fit" for r in records)
    counts["validation_recordings"] = len(validation)
    mismatches = {
        key: {"expected": value, "observed": counts.get(key)}
        for key, value in subset["expected_counts"].items()
        if counts.get(key) != value
    }
    validation_matches = validation == subset["expected_validation_recordings"]
    freeze = {
        "algorithm_version": config["algorithm_version"],
        "subset_config": subset,
        "deterministic": True,
        "seed": None,
        "input_manifests": {
            "internal_train": {"path": str(train_manifest.resolve()), "sha256": _sha256_file(train_manifest)},
            "internal_development": {"path": str(development_manifest.resolve()), "sha256": _sha256_file(development_manifest)},
            "step5_group_dev": {"path": str((step5_freeze_dir / "group_dev_ids.txt").resolve()), "sha256_lines": group_dev_sha},
        },
        "validation_recordings": validation,
        "validation_recordings_match_expected": validation_matches,
        "counts": counts,
        "count_mismatches": mismatches,
        "id_list_sha256": {view: _sha256_lines(ids) for view, ids in views.items()},
        "guard_ids_sha256": _sha256_lines(guard_ids),
        "recording_records": records,
        "label_access": {
            "keypoints_read": False,
            "single_person_filename_metadata_used_for": "selecting the labelled single-person subset only",
            "validation_selection_uses": "recording names and sha256 only",
            "official_test_used": False,
        },
    }
    if mismatches or not validation_matches:
        raise AssertionError(
            f"Freeze does not match the plan: mismatches={mismatches}, validation={validation}"
        )
    freeze_dir.mkdir(parents=True)
    for view in VIEWS:
        (freeze_dir / VIEW_FILE.format(view=view)).write_text("".join(f"{s}\n" for s in views[view]), encoding="utf-8")
    (freeze_dir / "guard_ids.txt").write_text("".join(f"{s}\n" for s in guard_ids), encoding="utf-8")
    (freeze_dir / SPLIT_ASSIGNMENTS_NAME).write_text(json.dumps(assignments, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (freeze_dir / FREEZE_JSON_NAME).write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return freeze


def load_view_ids(freeze_dir, view):
    freeze_dir = Path(freeze_dir)
    ids = [line.strip() for line in (freeze_dir / VIEW_FILE.format(view=view)).read_text(encoding="utf-8").splitlines() if line.strip()]
    recorded = json.loads((freeze_dir / FREEZE_JSON_NAME).read_text(encoding="utf-8"))["id_list_sha256"][view]
    if _sha256_lines(ids) != recorded:
        raise AssertionError(f"View {view} does not match its frozen SHA-256")
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze the Step-5b dense views (deterministic).")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    freeze = freeze_step5b_subset(repo_root=args.repo_root)
    print(json.dumps({"counts": freeze["counts"], "validation_recordings": freeze["validation_recordings"]}, indent=2))


if __name__ == "__main__":
    main()
