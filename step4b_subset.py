import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from piw3d_manifest import candidate_group_hash, parse_sample_id
from step4b_config import step4b_config_for_json, validate_step4b_config


DEFAULT_TRAIN_MANIFEST = Path("data/_artifacts/piw3d_candidate_group_manifests/internal_train_ids.txt")
DEFAULT_DEVELOPMENT_MANIFEST = Path(
    "data/_artifacts/piw3d_candidate_group_manifests/internal_development_ids.txt"
)
DEFAULT_OUTPUT_DIR = Path("data/_artifacts/step4b_pilot/freeze")


def read_manifest_ids(path):
    path = Path(path)
    sample_ids = [line.split()[0] for line in path.read_text(encoding="utf-8-sig").splitlines() if line.split()]
    if not sample_ids:
        raise ValueError(f"ID manifest is empty: {path}")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"ID manifest contains duplicate IDs: {path}")
    for sample_id in sample_ids:
        parse_sample_id(sample_id)
    return sample_ids


def is_single_person_id(sample_id):
    scene_prefix = sample_id.split("_", 1)[0]
    return len(scene_prefix) == 3 and scene_prefix[0] == "S" and scene_prefix[2] == "1"


def _single_person_groups(sample_ids):
    grouped = defaultdict(list)
    for sample_id in sample_ids:
        parsed = parse_sample_id(sample_id)
        if is_single_person_id(sample_id):
            grouped[parsed["group"]].append(sample_id)
    for group in grouped:
        grouped[group].sort(key=lambda sample_id: (parse_sample_id(sample_id)["frame"], sample_id))
    return dict(grouped)


def select_groups(group_names, group_count, seed):
    ordered_groups = sorted(set(group_names))
    if len(ordered_groups) < group_count:
        raise ValueError(f"Only {len(ordered_groups)} eligible groups are available; need {group_count}")
    shuffled_groups = list(ordered_groups)
    random.Random(seed).shuffle(shuffled_groups)
    return shuffled_groups[:group_count]


def select_evenly_spaced_frames(sample_ids, max_frames):
    if max_frames <= 0:
        raise ValueError("max_frames must be positive")
    ordered_ids = sorted(
        sample_ids,
        key=lambda sample_id: (parse_sample_id(sample_id)["frame"], sample_id),
    )
    if len(ordered_ids) <= max_frames:
        return ordered_ids
    selected_positions = np.linspace(0, len(ordered_ids) - 1, num=max_frames, dtype=int)
    if len(set(selected_positions.tolist())) != max_frames:
        raise AssertionError("Evenly spaced frame selection produced duplicate positions")
    return [ordered_ids[int(position)] for position in selected_positions]


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _selected_group_record(group, selected_ids, all_single_person_ids):
    return {
        "group": group,
        "candidate_group_hash": candidate_group_hash(group),
        "one_person_ids_available": len(all_single_person_ids[group]),
        "selected_frame_count": len(selected_ids),
        "selected_ids": selected_ids,
    }


def freeze_step4b_subset(
    train_manifest_path=DEFAULT_TRAIN_MANIFEST,
    development_manifest_path=DEFAULT_DEVELOPMENT_MANIFEST,
    output_dir=DEFAULT_OUTPUT_DIR,
    seed=0,
):
    config = step4b_config_for_json()
    validate_step4b_config(config)
    if seed != config["seed"]:
        raise ValueError(f"Step-4B seed must remain frozen at {config['seed']}")

    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing freeze directory: {output_dir}")

    train_ids = read_manifest_ids(train_manifest_path)
    development_ids = read_manifest_ids(development_manifest_path)
    if set(train_ids) & set(development_ids):
        raise AssertionError("Internal train/development manifests overlap")

    subset_config = config["subset"]
    train_groups = _single_person_groups(train_ids)
    development_groups = _single_person_groups(development_ids)
    selected_train_groups = select_groups(
        train_groups.keys(), subset_config["train_group_count"], seed
    )
    selected_development_groups = select_groups(
        development_groups.keys(), subset_config["development_group_count"], seed
    )

    selected_train_ids = []
    selected_development_ids = []
    train_group_records = []
    development_group_records = []
    for group in selected_train_groups:
        group_ids = select_evenly_spaced_frames(
            train_groups[group], subset_config["train_max_frames_per_group"]
        )
        selected_train_ids.extend(group_ids)
        train_group_records.append(_selected_group_record(group, group_ids, train_groups))
    for group in selected_development_groups:
        group_ids = select_evenly_spaced_frames(
            development_groups[group], subset_config["development_max_frames_per_group"]
        )
        selected_development_ids.extend(group_ids)
        development_group_records.append(
            _selected_group_record(group, group_ids, development_groups)
        )

    selected_train_ids = sorted(set(selected_train_ids), key=lambda sample_id: parse_sample_id(sample_id)["frame"])
    selected_train_ids.sort(key=lambda sample_id: (parse_sample_id(sample_id)["group"], parse_sample_id(sample_id)["frame"]))
    selected_development_ids = sorted(
        set(selected_development_ids),
        key=lambda sample_id: (parse_sample_id(sample_id)["group"], parse_sample_id(sample_id)["frame"]),
    )
    if set(selected_train_ids) & set(selected_development_ids):
        raise AssertionError("Selected train/development ID overlap must be zero")
    selected_train_groups = {parse_sample_id(sample_id)["group"] for sample_id in selected_train_ids}
    selected_development_groups = {
        parse_sample_id(sample_id)["group"] for sample_id in selected_development_ids
    }
    if selected_train_groups & selected_development_groups:
        raise AssertionError("Selected train/development group overlap must be zero")

    split_assignments = {sample_id: "internal_train" for sample_id in selected_train_ids}
    split_assignments.update(
        {sample_id: "internal_development" for sample_id in selected_development_ids}
    )
    freeze = {
        "config": config,
        "input_manifests": {
            "internal_train": {
                "path": str(Path(train_manifest_path).resolve()),
                "sha256": _sha256_file(train_manifest_path),
                "id_count": len(train_ids),
            },
            "internal_development": {
                "path": str(Path(development_manifest_path).resolve()),
                "sha256": _sha256_file(development_manifest_path),
                "id_count": len(development_ids),
            },
        },
        "split_label": "candidate-group-disjoint",
        "seed": seed,
        "selection_rule": subset_config["selection_rule"],
        "selected_ids": {
            "internal_train": selected_train_ids,
            "internal_development": selected_development_ids,
        },
        "selected_groups": {
            "internal_train": train_group_records,
            "internal_development": development_group_records,
        },
        "split_assignments": split_assignments,
        "counts": {
            "internal_train_groups_requested": subset_config["train_group_count"],
            "internal_train_groups_selected": len(selected_train_groups),
            "internal_train_ids_selected": len(selected_train_ids),
            "internal_train_frames_per_group_cap": subset_config["train_max_frames_per_group"],
            "internal_development_groups_requested": subset_config["development_group_count"],
            "internal_development_groups_selected": len(selected_development_groups),
            "internal_development_ids_selected": len(selected_development_ids),
            "internal_development_frames_per_group_cap": subset_config["development_max_frames_per_group"],
            "total_ids_selected_before_eligibility_checks": len(selected_train_ids)
            + len(selected_development_ids),
        },
        "eligibility_checks": {
            "single_person_filename_metadata_only": True,
            "official_test_used": False,
            "multi_person_label_inspection": False,
            "groups_moved_between_partitions": False,
        },
    }

    output_dir.mkdir(parents=True)
    (output_dir / "selected_train_ids.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in selected_train_ids), encoding="utf-8"
    )
    (output_dir / "selected_development_ids.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in selected_development_ids), encoding="utf-8"
    )
    (output_dir / "split_assignments.json").write_text(
        json.dumps(split_assignments, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "step4b_freeze_config.json").write_text(
        json.dumps(freeze, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return freeze


def parse_args():
    parser = argparse.ArgumentParser(description="Freeze the bounded Step-4B single-person sanity subset.")
    parser.add_argument("--train-manifest", type=Path, default=DEFAULT_TRAIN_MANIFEST)
    parser.add_argument("--development-manifest", type=Path, default=DEFAULT_DEVELOPMENT_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()
    freeze = freeze_step4b_subset(
        train_manifest_path=args.train_manifest,
        development_manifest_path=args.development_manifest,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(freeze["counts"], indent=2, sort_keys=True))
    print(f"freeze_config={Path(args.output_dir) / 'step4b_freeze_config.json'}")


if __name__ == "__main__":
    main()
