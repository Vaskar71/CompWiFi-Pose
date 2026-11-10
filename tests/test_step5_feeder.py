import json

import numpy as np
import pytest
import torch

from feeder.piw3d_single_person import SinglePersonPiW3DDataset
from feeder.piw3d_step5 import (
    Step5SinglePersonDataset,
    compute_train_input_stats,
    load_input_stats,
    save_input_stats,
)


IDS = ["S11_01_10", "S11_01_12", "S11_01_14"]


def make_root(tmp_path):
    train = tmp_path / "train_data"
    (train / "csi_ap").mkdir(parents=True)
    (train / "keypoint").mkdir()
    rng = np.random.default_rng(0)
    for sample_id in IDS:
        np.save(train / "csi_ap" / f"{sample_id}.npy", rng.normal(loc=3.0, scale=2.0, size=(3, 180, 20)))
        np.save(train / "keypoint" / f"{sample_id}.npy", rng.normal(size=(1, 14, 3)))
    (train / "train_data_list.txt").write_text("".join(f"{i}\n" for i in IDS), encoding="utf-8")
    (train / "train_ids.txt").write_text("".join(f"{i}\n" for i in IDS), encoding="utf-8")
    (tmp_path / "split_assignments.json").write_text(json.dumps({i: "train" for i in IDS}), encoding="utf-8")
    return tmp_path


def test_input_stats_match_numpy(tmp_path):
    root = make_root(tmp_path)
    paths = [root / "train_data" / "csi_ap" / f"{i}.npy" for i in IDS]
    stats = compute_train_input_stats(paths)
    stacked = np.stack([np.load(p) for p in paths])
    np.testing.assert_allclose(stats["mean"], stacked.mean(axis=(0, 3))[..., None], rtol=1e-10)
    np.testing.assert_allclose(stats["std"], stacked.std(axis=(0, 3))[..., None], rtol=1e-8)
    assert stats["mean"].shape == (3, 180, 1)
    path = save_input_stats(tmp_path / "stats.npz", stats, "abc")
    loaded = load_input_stats(path)
    assert loaded["train_id_sha256"] == "abc"
    assert not loaded["mean"].flags.writeable
    with pytest.raises(FileExistsError):
        save_input_stats(path, stats, "abc")


def test_legacy_minmax_identical_to_parent(tmp_path):
    root = make_root(tmp_path)
    kwargs = dict(expected_split="train", split_assignments_path=root / "split_assignments.json")
    parent = SinglePersonPiW3DDataset(root, root / "train_data" / "train_ids.txt", **kwargs)
    child = Step5SinglePersonDataset(root, root / "train_data" / "train_ids.txt", input_norm="legacy_minmax", **kwargs)
    for index in range(len(IDS)):
        assert torch.equal(parent[index]["csi"], child[index]["csi"])
        assert torch.equal(parent[index]["pose"], child[index]["pose"])
        assert sorted(child[index]) == ["csi", "pose"]


def test_train_zscore_applies_frozen_stats(tmp_path):
    root = make_root(tmp_path)
    paths = [root / "train_data" / "csi_ap" / f"{i}.npy" for i in IDS]
    stats = compute_train_input_stats(paths)
    dataset = Step5SinglePersonDataset(
        root, root / "train_data" / "train_ids.txt", input_norm="train_zscore", input_stats=stats
    )
    stored = np.load(paths[0])
    expected = (stored - stats["mean"]) / np.maximum(stats["std"], 1e-6)
    np.testing.assert_allclose(dataset[0]["csi"].numpy(), expected.astype(np.float32), rtol=1e-6)
    stacked = torch.stack([dataset[i]["csi"] for i in range(len(IDS))]).double()
    assert abs(float(stacked.mean())) < 1e-5


def test_input_norm_validation(tmp_path):
    root = make_root(tmp_path)
    with pytest.raises(ValueError):
        Step5SinglePersonDataset(root, root / "train_data" / "train_ids.txt", input_norm="train_zscore")
    with pytest.raises(ValueError):
        Step5SinglePersonDataset(root, root / "train_data" / "train_ids.txt", input_norm="bogus")
