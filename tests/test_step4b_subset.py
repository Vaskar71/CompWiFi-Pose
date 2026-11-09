import json

import pytest

from piw3d_manifest import parse_sample_id
from step4b_subset import (
    freeze_step4b_subset,
    is_single_person_id,
    select_evenly_spaced_frames,
)


def test_filename_single_person_rule_requires_third_character_one():
    assert is_single_person_id("S11_01_10") is True
    assert is_single_person_id("S21_01_10") is True
    assert is_single_person_id("S51_01_10") is True
    assert is_single_person_id("S12_01_10") is False
    assert is_single_person_id("S13_01_10") is False


def write_manifest(path, sample_ids):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{sample_id}\n" for sample_id in sample_ids), encoding="utf-8")
    return path


def make_manifests(tmp_path, train_group_count=10, development_group_count=5):
    train_ids = []
    development_ids = []
    for group_index in range(train_group_count):
        group = f"S11_{group_index + 1:02d}"
        train_ids.extend(f"{group}_{frame}" for frame in range(10, 310))
    for group_index in range(development_group_count):
        group = f"S11_{group_index + 21:02d}"
        development_ids.extend(f"{group}_{frame}" for frame in range(10, 160))
    return (
        write_manifest(tmp_path / "internal_train_ids.txt", train_ids),
        write_manifest(tmp_path / "internal_development_ids.txt", development_ids),
    )


def test_evenly_spaced_frame_selection_is_deterministic_and_capped():
    sample_ids = [f"S11_01_{frame}" for frame in range(10, 20)]

    selected = select_evenly_spaced_frames(sample_ids, 4)

    assert selected == ["S11_01_10", "S11_01_13", "S11_01_16", "S11_01_19"]
    assert select_evenly_spaced_frames(list(reversed(sample_ids)), 4) == selected
    assert select_evenly_spaced_frames(sample_ids, 20) == sample_ids


def test_freeze_selects_bounded_disjoint_single_person_subset(tmp_path):
    train_manifest, development_manifest = make_manifests(tmp_path)
    output_dir = tmp_path / "freeze"

    freeze = freeze_step4b_subset(train_manifest, development_manifest, output_dir, seed=0)

    counts = freeze["counts"]
    assert counts["internal_train_groups_selected"] == 8
    assert counts["internal_development_groups_selected"] == 4
    assert counts["internal_train_ids_selected"] == 8 * 256
    assert counts["internal_development_ids_selected"] == 4 * 128
    assert counts["total_ids_selected_before_eligibility_checks"] == 2560
    assert freeze["seed"] == 0
    assert freeze["eligibility_checks"]["official_test_used"] is False
    assert freeze["eligibility_checks"]["multi_person_label_inspection"] is False

    selected_train_ids = freeze["selected_ids"]["internal_train"]
    selected_development_ids = freeze["selected_ids"]["internal_development"]
    assert len(set(selected_train_ids)) == len(selected_train_ids)
    assert len(set(selected_development_ids)) == len(selected_development_ids)
    assert set(selected_train_ids).isdisjoint(selected_development_ids)
    assert all(is_single_person_id(sample_id) for sample_id in selected_train_ids)
    assert all(is_single_person_id(sample_id) for sample_id in selected_development_ids)
    train_groups = {parse_sample_id(sample_id)["group"] for sample_id in selected_train_ids}
    development_groups = {
        parse_sample_id(sample_id)["group"] for sample_id in selected_development_ids
    }
    assert len(train_groups) == 8
    assert len(development_groups) == 4
    assert train_groups.isdisjoint(development_groups)
    assert set(freeze["split_assignments"]) == set(selected_train_ids + selected_development_ids)

    persisted = json.loads(
        (output_dir / "step4b_freeze_config.json").read_text(encoding="utf-8")
    )
    assert persisted == freeze
    assert (output_dir / "selected_train_ids.txt").read_text(encoding="utf-8").splitlines() == selected_train_ids
    assert (
        output_dir / "selected_development_ids.txt"
    ).read_text(encoding="utf-8").splitlines() == selected_development_ids


def test_freeze_selection_is_reproducible_in_a_new_output_directory(tmp_path):
    train_manifest, development_manifest = make_manifests(tmp_path)
    first = freeze_step4b_subset(train_manifest, development_manifest, tmp_path / "first")
    second = freeze_step4b_subset(train_manifest, development_manifest, tmp_path / "second")
    assert first["selected_ids"] == second["selected_ids"]
    assert first["selected_groups"] == second["selected_groups"]


def test_freeze_refuses_to_overwrite_or_use_wrong_seed(tmp_path):
    train_manifest, development_manifest = make_manifests(tmp_path)
    output_dir = tmp_path / "freeze"
    freeze_step4b_subset(train_manifest, development_manifest, output_dir)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        freeze_step4b_subset(train_manifest, development_manifest, output_dir)
    with pytest.raises(ValueError, match="seed must remain frozen"):
        freeze_step4b_subset(
            train_manifest,
            development_manifest,
            tmp_path / "other_seed",
            seed=1,
        )


def test_freeze_rejects_insufficient_eligible_groups(tmp_path):
    train_manifest, development_manifest = make_manifests(
        tmp_path, train_group_count=3, development_group_count=5
    )
    with pytest.raises(ValueError, match="Only 3 eligible groups"):
        freeze_step4b_subset(train_manifest, development_manifest, tmp_path / "freeze")
