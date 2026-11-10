import copy
import hashlib
import json

import numpy as np
import pytest

from make_step5b_data import build_step5b_data, recording_input_means
from piw3d_manifest import parse_sample_id
from step5b_config import VIEWS, step5b_config_for_json
from step5b_subset import (
    build_step5b_views,
    dense_layout,
    freeze_step5b_subset,
    load_view_ids,
    read_group_dev_ids,
    validation_recordings,
)


SUBSET = step5b_config_for_json()["subset"]


def ids(group, frames):
    return [f"{group}_{f}" for f in frames]


def synthetic_manifest():
    out = []
    for scene in ("S11", "S21"):
        for rec in range(1, 6):
            out += ids(f"{scene}_{rec:02d}", range(10, 10 + 150 + rec))
    out += ids("S12_01", range(10, 200))  # multi-person, must be ignored
    return out


def group_dev():
    return ids("S51_90", range(10, 74))


def small_subset():
    subset = copy.deepcopy(SUBSET)
    subset["validation_recordings_per_scene"] = 1
    return subset


def test_validation_recordings_hash_order_names_only():
    names = ["S11_01", "S11_02", "S11_03", "S21_01", "S21_02"]
    chosen = validation_recordings(names, 1)
    expected = []
    for scene in ("S11", "S21"):
        group = [n for n in names if n.startswith(scene)]
        expected.append(min(group, key=lambda n: hashlib.sha256(n.encode("utf-8")).hexdigest()))
    assert chosen == expected
    assert validation_recordings(list(reversed(names)), 1) == chosen


def test_validation_recordings_reproduce_plan_expectation():
    # The plan's nine expected names must be the first three per scene by hash among themselves.
    expected = SUBSET["expected_validation_recordings"]
    assert validation_recordings(expected, 3) == expected


def test_dense_layout_guard_16_and_roles():
    ordered = ids("S11_01", range(10, 10 + 300))
    layout = dense_layout(ordered, SUBSET)
    start, end = layout["block_positions"]
    assert (start, end) == (60, 60 + 30)
    assert len(layout["guard"]) == 32
    assert set(layout["temporal_holdout"]) == set(ordered[start:end])
    assert not set(layout["train"]) & (set(layout["guard"]) | set(layout["temporal_holdout"]))
    assert len(layout["train"]) + len(layout["guard"]) + len(layout["temporal_holdout"]) == 300


def test_views_disjoint_subsets_and_deterministic():
    manifest = synthetic_manifest()
    views, assignments, guard, validation, records = build_step5b_views(manifest, group_dev(), small_subset())
    again = build_step5b_views(list(reversed(manifest)), group_dev(), small_subset())
    assert views == again[0] and validation == again[3]
    assert len(validation) == 2 and len(records) == 10
    sets = {v: set(x) for v, x in views.items()}
    assert not sets["fit_train_dense"] & sets["group_val"]
    assert not sets["fit_holdout"] & sets["group_val"]
    assert sets["fit_train_sparse"] <= sets["fit_train_dense"]
    assert sets["all_train_sparse"] <= sets["all_train_dense"]
    assert not sets["all_train_dense"] & sets["all_holdout"]
    assert not set(guard) & set(assignments)
    assert all(parse_sample_id(s)["group"] in validation for s in sets["group_val"])
    every = set().union(*sets.values())
    assert all(s.split("_")[0][2] == "1" for s in every)
    assert every - sets["group_dev"] <= set(manifest)
    assert set(assignments.values()) == {"train", "temporal_holdout", "group_dev"}
    assert all(len(views["fit_train_sparse"]) <= 64 * 8 for _ in [0])


def test_validate_rejects_injected_overlap():
    manifest = synthetic_manifest()
    views, assignments, guard, validation, _ = build_step5b_views(manifest, group_dev(), small_subset())
    from step5b_subset import validate_views

    broken = dict(views)
    broken["fit_train_dense"] = views["fit_train_dense"] + [views["group_val"][0]]
    with pytest.raises(AssertionError):
        validate_views(broken, assignments, guard, validation, manifest)
    broken = dict(views)
    broken["group_val"] = views["group_val"] + ["S12_01_10"]
    with pytest.raises(AssertionError):
        validate_views(broken, assignments, guard, validation, manifest)


def _write_inputs(tmp_path):
    manifest = synthetic_manifest()
    dev = group_dev()
    train_path = tmp_path / "internal_train_ids.txt"
    dev_path = tmp_path / "internal_development_ids.txt"
    train_path.write_text("".join(f"{s}\n" for s in manifest), encoding="utf-8")
    dev_path.write_text("".join(f"{s}\n" for s in dev + ids("S52_01", range(10, 40))), encoding="utf-8")
    step5 = tmp_path / "freeze_v1"
    step5.mkdir()
    (step5 / "group_dev_ids.txt").write_text("".join(f"{s}\n" for s in dev), encoding="utf-8")
    sha = hashlib.sha256("".join(f"{s}\n" for s in dev).encode("utf-8")).hexdigest()
    (step5 / "step5_freeze.json").write_text(json.dumps({"id_list_sha256": {"group_dev": sha}}), encoding="utf-8")
    subset = small_subset()
    views, _, guard, validation, records = build_step5b_views(manifest, dev, subset)
    subset["expected_validation_recordings"] = validation
    subset["expected_counts"] = {
        **{v: len(x) for v, x in views.items()},
        "guard": len(guard),
        "train_recordings": len(records),
        "fit_recordings": len(records) - len(validation),
        "validation_recordings": len(validation),
    }
    return manifest, dev, train_path, dev_path, step5, subset


def test_group_dev_sha_is_verified(tmp_path):
    _, dev, _, _, step5, _ = _write_inputs(tmp_path)
    assert read_group_dev_ids(step5)[0] == dev
    (step5 / "group_dev_ids.txt").write_text("".join(f"{s}\n" for s in dev[:-1]), encoding="utf-8")
    with pytest.raises(AssertionError):
        read_group_dev_ids(step5)


def test_freeze_writes_views_refuses_overwrite_and_checks_counts(tmp_path):
    manifest, dev, train_path, dev_path, step5, subset = _write_inputs(tmp_path)
    freeze_dir = tmp_path / "freeze_v2"
    kwargs = dict(train_manifest=train_path, development_manifest=dev_path, step5_freeze_dir=step5, freeze_dir=freeze_dir)
    freeze = freeze_step5b_subset(subset_config=subset, **kwargs)
    for view in VIEWS:
        assert load_view_ids(freeze_dir, view) == sorted(
            load_view_ids(freeze_dir, view), key=lambda s: (parse_sample_id(s)["group"], parse_sample_id(s)["frame"])
        )
    assert freeze["label_access"]["keypoints_read"] is False
    with pytest.raises(FileExistsError):
        freeze_step5b_subset(subset_config=subset, **kwargs)
    wrong = copy.deepcopy(subset)
    wrong["expected_counts"]["fit_train_dense"] += 1
    with pytest.raises(AssertionError):
        freeze_step5b_subset(subset_config=wrong, **dict(kwargs, freeze_dir=tmp_path / "other"))


def test_build_data_validates_and_writes_stats(tmp_path):
    manifest, dev, train_path, dev_path, step5, subset = _write_inputs(tmp_path)
    freeze_dir = tmp_path / "freeze_v2"
    freeze_step5b_subset(train_manifest=train_path, development_manifest=dev_path, step5_freeze_dir=step5,
                         freeze_dir=freeze_dir, subset_config=subset)
    source = tmp_path / "source" / "train_data"
    (source / "csi").mkdir(parents=True)
    (source / "keypoint").mkdir()
    for index, sample_id in enumerate(manifest + dev):
        (source / "csi" / f"{sample_id}.mat").write_bytes(b"raw")
        pose = np.arange(42, dtype=np.float64).reshape(1, 14, 3) * 0.01 + 0.001 * index
        np.save(source / "keypoint" / f"{sample_id}.npy", pose)
    seen = []

    def fake_process(name_list, data_root, split):
        for name in name_list:
            seen.append(name)
            rng = np.random.default_rng(len(seen))
            amplitude, phase = rng.normal(size=(3, 90, 20)), rng.normal(size=(3, 90, 20))
            base = data_root / split
            np.save(base / "csi_amplitude" / f"{name}.npy", amplitude)
            np.save(base / "csi_phase" / f"{name}.npy", phase)
            np.save(base / "csi_ap" / f"{name}.npy", np.concatenate((amplitude, phase), axis=1))

    data_root = tmp_path / "data_v2"
    validation = build_step5b_data(source_root=tmp_path / "source", freeze_dir=freeze_dir, data_root=data_root,
                                   process_fn=fake_process, progress_every=0)
    assert validation["counts"]["failed_ids"] == 0
    guard = set((freeze_dir / "guard_ids.txt").read_text(encoding="utf-8").split())
    assert not guard & set(seen)
    for name in ("input_stats_fit.npz", "input_stats_all.npz", "codec_stats_fit.json", "codec_stats_all.json",
                 "recording_input_means.npz"):
        assert (freeze_dir / name).is_file()
    with np.load(freeze_dir / "recording_input_means.npz") as stored:
        assert stored["means"].shape == (11, 3, 180, 1)
        assert int(np.sum(stored["sources"] == "group_dev_own_frames")) == 1
    assert json.loads((freeze_dir / "codec_stats_fit.json").read_text())["sample_count"] == subset["expected_counts"]["fit_train_dense"]
    with pytest.raises(FileExistsError):
        build_step5b_data(source_root=tmp_path / "source", freeze_dir=freeze_dir, data_root=data_root, process_fn=fake_process)


def test_recording_input_means_value(tmp_path):
    for index, value in enumerate((1.0, 3.0)):
        np.save(tmp_path / f"S11_01_{index}.npy", np.full((3, 180, 20), value))
    names, means, counts = recording_input_means(tmp_path, {"S11_01": ["S11_01_0", "S11_01_1"]})
    assert names == ["S11_01"] and counts.tolist() == [2]
    assert np.allclose(means, 2.0)
