import json

import numpy as np
import pytest
import torch

from feeder.piw3d_single_person import (
    BATCH_KEYS,
    SinglePersonPiW3DDataset,
    make_single_person_dataloader,
)


TRAIN_IDS = ["S11_01_10", "S11_01_12", "S11_01_14"]
DEVELOPMENT_IDS = ["S11_21_10", "S11_21_12"]


def make_processed_csi(offset):
    csi = np.zeros((3, 180, 20), dtype=np.float32)
    csi[:, :90, :] = np.linspace(offset, offset + 1.0, 3 * 90 * 20, dtype=np.float32).reshape(
        3, 90, 20
    )
    csi[:, 90:, :] = np.linspace(
        offset + 2.0, offset + 3.0, 3 * 90 * 20, dtype=np.float32
    ).reshape(3, 90, 20)
    return csi


def make_pose(offset=0.0):
    joints = np.arange(14, dtype=np.float32).reshape(14, 1)
    return (np.concatenate((joints, joints * 0.1, joints * 0.01), axis=1) + offset)[None, :, :]


def make_root(tmp_path, train_ids=TRAIN_IDS, development_ids=DEVELOPMENT_IDS):
    train_dir = tmp_path / "train_data"
    for directory in ("csi_ap", "keypoint"):
        (train_dir / directory).mkdir(parents=True, exist_ok=True)
    selected_ids = train_ids + development_ids
    (train_dir / "train_data_list.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in selected_ids), encoding="utf-8"
    )
    (train_dir / "selected_train_ids.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in train_ids), encoding="utf-8"
    )
    (train_dir / "selected_development_ids.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in development_ids), encoding="utf-8"
    )
    assignments = {sample_id: "internal_train" for sample_id in train_ids}
    assignments.update({sample_id: "internal_development" for sample_id in development_ids})
    split_path = tmp_path / "split_assignments.json"
    split_path.write_text(json.dumps(assignments), encoding="utf-8")
    for index, sample_id in enumerate(selected_ids):
        np.save(train_dir / "csi_ap" / f"{sample_id}.npy", make_processed_csi(index))
        np.save(train_dir / "keypoint" / f"{sample_id}.npy", make_pose(index * 0.01))
    return tmp_path, split_path


def make_dataset(tmp_path, expected_split="internal_train", train_ids=TRAIN_IDS):
    root, split_path = make_root(tmp_path, train_ids=train_ids)
    id_filename = (
        "selected_train_ids.txt" if expected_split == "internal_train" else "selected_development_ids.txt"
    )
    return SinglePersonPiW3DDataset(
        root,
        root / "train_data" / id_filename,
        expected_split=expected_split,
        split_assignments_path=split_path,
    )


def test_single_person_batches_expose_only_csi_and_pose(tmp_path):
    dataset = make_dataset(tmp_path)
    loader = make_single_person_dataloader(
        dataset,
        is_training=False,
        generator=torch.Generator().manual_seed(0),
        batch_size=2,
        num_workers=0,
    )

    batches = list(loader)

    assert dataset.sample_ids == TRAIN_IDS
    assert len(dataset) == 3
    assert [tuple(batch["csi"].shape) for batch in batches] == [(2, 3, 180, 20), (1, 3, 180, 20)]
    assert [tuple(batch["pose"].shape) for batch in batches] == [(2, 1, 14, 3), (1, 1, 14, 3)]
    for batch in batches:
        assert tuple(dataset.batch_keys) == BATCH_KEYS
        assert set(batch) == {"csi", "pose"}
        assert torch.isfinite(batch["csi"]).all()
        assert torch.isfinite(batch["pose"]).all()


def test_loader_enforces_frozen_split_assignment(tmp_path):
    root, split_path = make_root(tmp_path)
    with pytest.raises(ValueError, match="expected 'internal_development'"):
        SinglePersonPiW3DDataset(
            root,
            root / "train_data" / "selected_train_ids.txt",
            expected_split="internal_development",
            split_assignments_path=split_path,
        )

    modified_path = tmp_path / "modified_split_assignments.json"
    assignments = json.loads(split_path.read_text(encoding="utf-8"))
    assignments[TRAIN_IDS[0]] = "internal_development"
    modified_path.write_text(json.dumps(assignments), encoding="utf-8")
    with pytest.raises(ValueError, match=f"Split assignment for {TRAIN_IDS[0]}"):
        SinglePersonPiW3DDataset(
            root,
            root / "train_data" / "selected_train_ids.txt",
            expected_split="internal_train",
            split_assignments_path=modified_path,
        )


def test_loader_rejects_multi_person_or_incompatible_labels(tmp_path):
    with pytest.raises(ValueError, match="not filename-marked single-person"):
        make_dataset(tmp_path, train_ids=["S12_01_10"])

    root, split_path = make_root(tmp_path / "bad_shape", train_ids=["S11_01_10"])
    np.save(root / "train_data" / "keypoint" / "S11_01_10.npy", np.zeros((2, 14, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="Unexpected pilot keypoint shape"):
        SinglePersonPiW3DDataset(
            root,
            root / "train_data" / "selected_train_ids.txt",
            expected_split="internal_train",
            split_assignments_path=split_path,
        )

    root, split_path = make_root(tmp_path / "nonfinite", train_ids=["S11_01_10"])
    bad_pose = make_pose()
    bad_pose[0, 0, 0] = np.nan
    np.save(root / "train_data" / "keypoint" / "S11_01_10.npy", bad_pose)
    with pytest.raises(ValueError, match="Non-finite pilot keypoint"):
        SinglePersonPiW3DDataset(
            root,
            root / "train_data" / "selected_train_ids.txt",
            expected_split="internal_train",
            split_assignments_path=split_path,
        )


def test_loader_rejects_unknown_ids_and_missing_pilot_files(tmp_path):
    root, split_path = make_root(tmp_path)
    unknown_path = tmp_path / "unknown_ids.txt"
    unknown_path.write_text("S11_01_10\nS99_01_10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the pilot training list"):
        SinglePersonPiW3DDataset(
            root,
            unknown_path,
            expected_split="internal_train",
            split_assignments_path=split_path,
        )

    root, split_path = make_root(tmp_path / "missing")
    (root / "train_data" / "csi_ap" / "S11_01_12.npy").unlink()
    with pytest.raises(FileNotFoundError, match="Missing pilot CSI file"):
        SinglePersonPiW3DDataset(
            root,
            root / "train_data" / "selected_train_ids.txt",
            expected_split="internal_train",
            split_assignments_path=split_path,
        )
