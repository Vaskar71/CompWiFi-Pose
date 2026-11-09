import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path


ALGORITHM_VERSION = "compwifi-step3-v1"
HASH_PREFIX = "compwifi-step3-v1:"
SPLIT_LABEL = "candidate-group-disjoint"
ID_PATTERN = re.compile(r"^(S\d\d)_(\d+)_(\d+)$")


def parse_sample_id(sample_id):
    match = ID_PATTERN.fullmatch(sample_id)
    if match is None:
        raise ValueError(f"Malformed Person-in-WiFi-3D training ID: {sample_id!r}")
    return {
        "id": sample_id,
        "group": f"{match.group(1)}_{match.group(2)}",
        "frame": int(match.group(3)),
    }


def candidate_group_hash(group):
    return hashlib.sha256(f"{HASH_PREFIX}{group}".encode("utf-8")).hexdigest()


def read_official_training_ids(train_list_path):
    train_list_path = Path(train_list_path)
    raw_list = train_list_path.read_bytes()
    lines = raw_list.decode("utf-8-sig").splitlines()
    sample_ids = [line.split()[0] for line in lines if line.split()]
    if not sample_ids:
        raise ValueError(f"Official training list is empty: {train_list_path}")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError("Official training list contains duplicate IDs")
    for sample_id in sample_ids:
        parse_sample_id(sample_id)
    source_hash = hashlib.sha256(raw_list).hexdigest()
    return sample_ids, source_hash


def assign_candidate_groups(groups, train_fraction=0.8):
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must be strictly between 0 and 1")
    ordered_groups = sorted(set(groups), key=lambda group: (candidate_group_hash(group), group))
    train_group_count = int(len(ordered_groups) * train_fraction)
    train_groups = set(ordered_groups[:train_group_count])
    development_groups = set(ordered_groups[train_group_count:])
    return {
        "ordered_groups": ordered_groups,
        "train_groups": train_groups,
        "development_groups": development_groups,
        "train_group_count": len(train_groups),
        "development_group_count": len(development_groups),
    }


def _id_sort_key(parsed_id):
    return (parsed_id["group"], parsed_id["frame"], parsed_id["id"])


def _write_id_manifest(path, sample_ids):
    path.write_text("".join(f"{sample_id}\n" for sample_id in sample_ids), encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _ensure_output_is_allowed(output_dir, train_list_path):
    output = Path(output_dir).resolve()
    train_split_dir = Path(train_list_path).resolve().parent
    if output == train_split_dir or train_split_dir in output.parents:
        raise ValueError(f"Refusing to write manifests inside the training split: {output}")


def _validate_assignments(sample_ids, assignments, groups_by_key):
    train_ids = [sample_id for sample_id in sample_ids if assignments[sample_id] == "internal_train"]
    development_ids = [
        sample_id for sample_id in sample_ids if assignments[sample_id] == "internal_development"
    ]
    if len(train_ids) + len(development_ids) != len(sample_ids):
        raise AssertionError("Every official-training ID must be assigned exactly once")
    if set(train_ids) & set(development_ids):
        raise AssertionError("Internal train/development ID overlap must be zero")

    train_groups = {parse_sample_id(sample_id)["group"] for sample_id in train_ids}
    development_groups = {
        parse_sample_id(sample_id)["group"] for sample_id in development_ids
    }
    if train_groups & development_groups:
        raise AssertionError("Candidate-group overlap between internal partitions must be zero")

    for sample_id in sample_ids:
        parsed = parse_sample_id(sample_id)
        next_frame_id = (parsed["group"], parsed["frame"] + 1)
        if next_frame_id in groups_by_key:
            next_id = groups_by_key[next_frame_id]
            if assignments[next_id] != assignments[sample_id]:
                raise AssertionError("Immediate-next-frame pairs must not cross partitions")

    return {
        "train_ids": train_ids,
        "development_ids": development_ids,
        "train_groups": train_groups,
        "development_groups": development_groups,
        "id_overlap": 0,
        "group_overlap": 0,
        "cross_partition_next_frame_pairs": 0,
    }


def build_candidate_group_manifests(
    train_list_path,
    output_dir,
    train_fraction=0.8,
):
    train_list_path = Path(train_list_path)
    output_dir = Path(output_dir)
    _ensure_output_is_allowed(output_dir, train_list_path)
    sample_ids, source_list_hash = read_official_training_ids(train_list_path)
    parsed_ids = [parse_sample_id(sample_id) for sample_id in sample_ids]

    grouped_ids = defaultdict(list)
    for parsed in parsed_ids:
        grouped_ids[parsed["group"]].append(parsed)
    assignment = assign_candidate_groups(grouped_ids.keys(), train_fraction=train_fraction)
    assignments = {
        parsed["id"]: (
            "internal_train"
            if parsed["group"] in assignment["train_groups"]
            else "internal_development"
        )
        for parsed in parsed_ids
    }
    groups_by_key = {
        (parsed["group"], parsed["frame"]): parsed["id"] for parsed in parsed_ids
    }
    validated = _validate_assignments(sample_ids, assignments, groups_by_key)

    train_ids = sorted(
        validated["train_ids"],
        key=lambda sample_id: _id_sort_key(parse_sample_id(sample_id)),
    )
    development_ids = sorted(
        validated["development_ids"],
        key=lambda sample_id: _id_sort_key(parse_sample_id(sample_id)),
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    train_manifest_hash = _write_id_manifest(output_dir / "internal_train_ids.txt", train_ids)
    development_manifest_hash = _write_id_manifest(
        output_dir / "internal_development_ids.txt", development_ids
    )

    summary = {
        "algorithm_version": ALGORITHM_VERSION,
        "split_label": SPLIT_LABEL,
        "train_fraction": train_fraction,
        "source_list_path": str(train_list_path.resolve()),
        "source_list_sha256": source_list_hash,
        "official_training_id_count": len(sample_ids),
        "candidate_group_count": len(grouped_ids),
        "internal_train": {
            "group_count": len(validated["train_groups"]),
            "id_count": len(train_ids),
            "manifest_sha256": train_manifest_hash,
        },
        "internal_development": {
            "group_count": len(validated["development_groups"]),
            "id_count": len(development_ids),
            "manifest_sha256": development_manifest_hash,
        },
        "person_count_stratified": False,
        "filename_person_counts_used": False,
        "official_test_list_read": False,
        "official_test_array_read": False,
        "id_overlap": validated["id_overlap"],
        "candidate_group_overlap": validated["group_overlap"],
        "cross_partition_next_frame_pairs": validated["cross_partition_next_frame_pairs"],
        "semantic_claim": (
            "Candidate filename groups are assigned only by deterministic hash. "
            "No recording-, subject-, or environment-disjointness is claimed."
        ),
        "group_assignments": {
            group: (
                "internal_train"
                if group in assignment["train_groups"]
                else "internal_development"
            )
            for group in assignment["ordered_groups"]
        },
        "group_hashes": {
            group: candidate_group_hash(group) for group in assignment["ordered_groups"]
        },
    }
    summary_path = output_dir / "manifest_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Create deterministic candidate-group-disjoint internal train/development "
            "manifests from the official Person-in-WiFi-3D training list only."
        )
    )
    parser.add_argument(
        "--train-list",
        type=Path,
        default=Path("data/person_in_wifi_3d/wifi_dataset/train_data/train_data_list.txt"),
        help="Official training list path. No test list or array is read.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/_artifacts/piw3d_candidate_group_manifests"),
        help="Ignored artifact directory for generated manifests and summary.",
    )
    parser.add_argument("--train-fraction", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = build_candidate_group_manifests(
        train_list_path=args.train_list,
        output_dir=args.output_dir,
        train_fraction=args.train_fraction,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
