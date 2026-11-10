import json

import numpy as np
import pytest

from make_step5_data import build_step5_data
from piw3d_manifest import parse_sample_id
from step5_config import step5_config_for_json
from step5_subset import (
    SPLITS,
    build_step5_selection,
    freeze_step5_subset,
    holdout_layout,
    split_train_group,
)


SUBSET = step5_config_for_json()["subset"]


def group_ids(prefix, group, frames):
    return [f"{prefix}_{group}_{frame}" for frame in frames]


def make_manifests(tmp_path):
    train_ids = (
        group_ids("S01", "01", range(10, 10 + 2 * 120, 2))  # 120 frames, single-person
        + group_ids("S11", "02", range(5, 5 + 300))  # 300 frames, single-person
        + group_ids("S02", "03", range(10, 200))  # multi-person, must be ignored
    )
    development_ids = group_ids("S21", "04", range(10, 110)) + group_ids("S03", "05", range(10, 60))
    train_manifest = tmp_path / "internal_train_ids.txt"
    development_manifest = tmp_path / "internal_development_ids.txt"
    train_manifest.write_text("".join(f"{i}\n" for i in train_ids), encoding="utf-8")
    development_manifest.write_text("".join(f"{i}\n" for i in development_ids), encoding="utf-8")
    return train_ids, development_ids, train_manifest, development_manifest


def make_source(tmp_path, sample_ids):
    keypoint_dir = tmp_path / "source" / "train_data" / "keypoint"
    keypoint_dir.mkdir(parents=True)
    for index, sample_id in enumerate(sample_ids):
        pose = np.full((1, 14, 3), 0.01 * index, dtype=np.float64) + np.arange(14)[None, :, None] * 0.1
        np.save(keypoint_dir / f"{sample_id}.npy", pose)
    return tmp_path / "source"


def test_holdout_layout_follows_rule():
    assert holdout_layout(105, SUBSET) == (21, 31, 16, 36)
    assert holdout_layout(539, SUBSET) == (107, 156, 102, 161)


def test_split_train_group_respects_guard_and_caps():
    ordered = group_ids("S01", "01", range(10, 10 + 300))
    layout = split_train_group(ordered, SUBSET)
    start, end = layout["block_positions"]
    block = set(ordered[start:end])
    guard = set(layout["guard_ids"])
    assert len(guard) == 10
    assert len(layout["train"]) == 64
    assert len(layout["temporal_holdout"]) == 16
    assert set(layout["temporal_holdout"]) <= block
    assert not (set(layout["train"]) & (block | guard))
    assert not (guard & block)


def test_selection_is_disjoint_single_person_and_deterministic(tmp_path):
    train_ids, development_ids, _, _ = make_manifests(tmp_path)
    selected, records, train_groups, dev_groups = build_step5_selection(train_ids, development_ids, SUBSET)
    again, _, _, _ = build_step5_selection(list(reversed(train_ids)), development_ids, SUBSET)
    assert selected == again
    assert sorted(train_groups) == ["S01_01", "S11_02"]
    assert sorted(dev_groups) == ["S21_04"]
    all_ids = [i for split in SPLITS for i in selected[split]]
    assert len(all_ids) == len(set(all_ids))
    assert all(i.split("_")[0][2] == "1" for i in all_ids)
    assert set(selected["train"]) | set(selected["temporal_holdout"]) <= set(train_ids)
    assert set(selected["group_dev"]) <= set(development_ids)
    train_groups_used = {parse_sample_id(i)["group"] for i in selected["train"]}
    dev_groups_used = {parse_sample_id(i)["group"] for i in selected["group_dev"]}
    assert not train_groups_used & dev_groups_used
    guard_ids = {i for record in records["train"] for i in record["guard_ids"]}
    assert not guard_ids & set(all_ids)
    assert len(selected["group_dev"]) == 64


def test_selection_rejects_manifest_overlap(tmp_path):
    train_ids, development_ids, _, _ = make_manifests(tmp_path)
    with pytest.raises(AssertionError):
        build_step5_selection(train_ids, development_ids + [train_ids[0]], SUBSET)


def test_freeze_writes_lists_and_refuses_overwrite(tmp_path):
    train_ids, development_ids, train_manifest, development_manifest = make_manifests(tmp_path)
    source = make_source(tmp_path, train_ids + development_ids)
    freeze_dir = tmp_path / "freeze"
    freeze = freeze_step5_subset(
        train_manifest=train_manifest,
        development_manifest=development_manifest,
        source_root=source,
        freeze_dir=freeze_dir,
    )
    for split in SPLITS:
        listed = (freeze_dir / f"{split}_ids.txt").read_text(encoding="utf-8").split()
        assert listed == freeze["selected_ids"][split]
    assignments = json.loads((freeze_dir / "split_assignments.json").read_text(encoding="utf-8"))
    assert set(assignments.values()) == set(SPLITS)
    assert freeze["eligibility_checks"]["official_test_used"] is False
    assert freeze["centroid_summary_stored_units"]["train"][0]["count"] == 64
    with pytest.raises(FileExistsError):
        freeze_step5_subset(
            train_manifest=train_manifest,
            development_manifest=development_manifest,
            source_root=source,
            freeze_dir=freeze_dir,
        )


def test_build_step5_data_validates_and_writes_train_only_stats(tmp_path):
    train_ids, development_ids, train_manifest, development_manifest = make_manifests(tmp_path)
    source = make_source(tmp_path, train_ids + development_ids)
    (source / "train_data" / "csi").mkdir()
    for sample_id in train_ids + development_ids:
        (source / "train_data" / "csi" / f"{sample_id}.mat").write_bytes(b"raw")
    freeze_dir = tmp_path / "freeze"
    freeze = freeze_step5_subset(
        train_manifest=train_manifest,
        development_manifest=development_manifest,
        source_root=source,
        freeze_dir=freeze_dir,
    )
    seen = []

    def fake_process(name_list, data_root, split):
        assert split == "train_data"
        for name in name_list:
            seen.append(name)
            rng = np.random.default_rng(len(seen))
            amplitude = rng.normal(size=(3, 90, 20))
            phase = rng.normal(size=(3, 90, 20))
            base = data_root / split
            np.save(base / "csi_amplitude" / f"{name}.npy", amplitude)
            np.save(base / "csi_phase" / f"{name}.npy", phase)
            np.save(base / "csi_ap" / f"{name}.npy", np.concatenate((amplitude, phase), axis=1))

    data_root = tmp_path / "data_v1"
    validation = build_step5_data(
        source_root=source, freeze_dir=freeze_dir, data_root=data_root, process_fn=fake_process
    )
    assert validation["counts"]["failed_ids"] == 0
    assert seen == [i for split in SPLITS for i in freeze["selected_ids"][split]]
    stats = validation["train_statistics"]
    assert stats["codec_stats"]["sample_count"] == len(freeze["selected_ids"]["train"])
    assert (freeze_dir / "train_input_stats_v1.npz").is_file()
    assert (freeze_dir / "pose_codec_stats_v1.json").is_file()
    with pytest.raises(FileExistsError):
        build_step5_data(source_root=source, freeze_dir=freeze_dir, data_root=data_root, process_fn=fake_process)
