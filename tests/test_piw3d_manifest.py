import json

import pytest

from piw3d_manifest import (
    SPLIT_LABEL,
    build_candidate_group_manifests,
    candidate_group_hash,
    read_official_training_ids,
)


MIXED_COUNT_IDS = [
    "S11_01_10",
    "S11_01_11",
    "S12_01_10",
    "S13_01_10",
    "S21_01_10",
    "S22_01_10",
    "S23_01_10",
]


def write_training_list(root, sample_ids):
    train_dir = root / "train_data"
    train_dir.mkdir(parents=True)
    list_path = train_dir / "train_data_list.txt"
    list_path.write_text("".join(f"{sample_id}\n" for sample_id in sample_ids), encoding="utf-8")
    return list_path


def test_manifest_assigns_whole_candidate_groups_without_count_stratification(tmp_path):
    train_list = write_training_list(tmp_path, list(reversed(MIXED_COUNT_IDS)))
    output_dir = tmp_path / "artifacts"

    summary = build_candidate_group_manifests(train_list, output_dir)

    expected_train_groups = {"S23_01", "S12_01", "S21_01", "S13_01"}
    expected_development_groups = {"S11_01", "S22_01"}
    assert {
        group
        for group, partition in summary["group_assignments"].items()
        if partition == "internal_train"
    } == expected_train_groups
    assert {
        group
        for group, partition in summary["group_assignments"].items()
        if partition == "internal_development"
    } == expected_development_groups
    assert summary["split_label"] == SPLIT_LABEL
    assert summary["split_label"] == "candidate-group-disjoint"
    assert summary["person_count_stratified"] is False
    assert summary["filename_person_counts_used"] is False
    assert summary["official_test_list_read"] is False
    assert summary["official_test_array_read"] is False
    assert summary["id_overlap"] == 0
    assert summary["candidate_group_overlap"] == 0
    assert summary["cross_partition_next_frame_pairs"] == 0
    assert summary["internal_train"]["group_count"] == 4
    assert summary["internal_development"]["group_count"] == 2

    train_ids = (output_dir / "internal_train_ids.txt").read_text(encoding="utf-8").splitlines()
    development_ids = (
        (output_dir / "internal_development_ids.txt").read_text(encoding="utf-8").splitlines()
    )
    assert set(train_ids) == {
        "S12_01_10",
        "S13_01_10",
        "S21_01_10",
        "S23_01_10",
    }
    assert set(development_ids) == {"S11_01_10", "S11_01_11", "S22_01_10"}
    assert sorted(train_ids + development_ids) == sorted(MIXED_COUNT_IDS)
    assert len(train_ids + development_ids) == len(set(train_ids + development_ids))

    persisted = json.loads(
        (output_dir / "manifest_summary.json").read_text(encoding="utf-8")
    )
    assert persisted == summary


def test_immediate_next_frame_pairs_stay_in_one_internal_partition(tmp_path):
    sample_ids = [
        "S11_01_10",
        "S11_01_11",
        "S11_01_12",
        "S12_01_10",
        "S13_01_10",
        "S21_01_10",
    ]
    train_list = write_training_list(tmp_path, sample_ids)
    output_dir = tmp_path / "artifacts"

    summary = build_candidate_group_manifests(train_list, output_dir)
    assignments = {
        sample_id: (
            "internal_train"
            if sample_id in set(
                (output_dir / "internal_train_ids.txt").read_text(encoding="utf-8").splitlines()
            )
            else "internal_development"
        )
        for sample_id in sample_ids
    }

    assert assignments["S11_01_10"] == assignments["S11_01_11"]
    assert assignments["S11_01_11"] == assignments["S11_01_12"]
    assert summary["cross_partition_next_frame_pairs"] == 0


def test_generator_does_not_read_a_synthetic_test_list(tmp_path):
    train_list = write_training_list(tmp_path, MIXED_COUNT_IDS)
    test_dir = tmp_path / "test_data"
    test_dir.mkdir()
    (test_dir / "test_data_list.txt").write_text("not_a_real_id\n", encoding="utf-8")

    summary = build_candidate_group_manifests(
        train_list, tmp_path / "artifacts", train_fraction=0.8
    )

    assert summary["official_training_id_count"] == len(MIXED_COUNT_IDS)
    assert summary["official_test_list_read"] is False


def test_rejects_duplicate_or_malformed_training_ids(tmp_path):
    duplicate_list = write_training_list(
        tmp_path / "duplicate", ["S11_01_10", "S11_01_10"]
    )
    with pytest.raises(ValueError, match="duplicate"):
        read_official_training_ids(duplicate_list)

    malformed_list = write_training_list(tmp_path / "malformed", ["S11_01_10", "bad_id"])
    with pytest.raises(ValueError, match="Malformed"):
        read_official_training_ids(malformed_list)


def test_refuses_to_write_manifests_inside_the_training_split(tmp_path):
    train_list = write_training_list(tmp_path, MIXED_COUNT_IDS)
    with pytest.raises(ValueError, match="inside the training split"):
        build_candidate_group_manifests(train_list, tmp_path / "train_data" / "manifests")


def test_candidate_group_hash_is_stable():
    assert candidate_group_hash("S11_01") == (
        "985388fc35fe97a8318f9181c320a245c31090cce783425112472bbc68059858"
    )
