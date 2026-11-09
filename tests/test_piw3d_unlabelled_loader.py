import numpy as np
import pytest
import torch

from feeder.piw3d_unlabelled import (
    BATCH_KEYS,
    UnlabelledPiW3DCSIDataset,
    make_unlabelled_csi_dataloader,
)


MIXED_COUNT_IDS = ["S11_01_10", "S12_01_10", "S13_01_10"]


def make_processed_csi(offset):
    csi = np.zeros((3, 180, 20), dtype=np.float32)
    csi[:, :90, :] = np.linspace(
        offset, offset + 1.0, 3 * 90 * 20, dtype=np.float32
    ).reshape(3, 90, 20)
    csi[:, 90:, :] = np.linspace(
        offset + 2.0, offset + 3.0, 3 * 90 * 20, dtype=np.float32
    ).reshape(3, 90, 20)
    return csi


def make_unlabelled_root(tmp_path, sample_ids=MIXED_COUNT_IDS):
    train_dir = tmp_path / "train_data"
    csi_dir = train_dir / "csi_ap"
    csi_dir.mkdir(parents=True)
    (train_dir / "train_data_list.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in sample_ids), encoding="utf-8"
    )
    for index, sample_id in enumerate(sample_ids):
        np.save(csi_dir / f"{sample_id}.npy", make_processed_csi(index))
    return tmp_path


def test_unlabelled_loading_works_without_keypoint_files(tmp_path):
    data_root = make_unlabelled_root(tmp_path)
    assert not (data_root / "train_data" / "keypoint").exists()

    dataset = UnlabelledPiW3DCSIDataset(data_root)

    assert len(dataset) == len(MIXED_COUNT_IDS)
    assert dataset.skipped_count == 0
    item = dataset[0]
    assert set(item) == set(BATCH_KEYS)
    assert item["csi"].shape == (3, 180, 20)
    assert item["csi"].dtype == torch.float32
    assert torch.isfinite(item["csi"]).all()


def test_mixed_filename_count_fixture_is_not_count_filtered(tmp_path):
    data_root = make_unlabelled_root(tmp_path)

    dataset = UnlabelledPiW3DCSIDataset(data_root)
    loader = make_unlabelled_csi_dataloader(
        dataset,
        is_training=False,
        generator=torch.manual_seed(0),
        batch_size=2,
        num_workers=0,
    )

    batches = list(loader)
    assert [len(batch["csi"]) for batch in batches] == [2, 1]
    for batch in batches:
        assert set(batch) == {"csi"}
        assert batch["csi"].shape[1:] == (3, 180, 20)
        assert torch.isfinite(batch["csi"]).all()
    assert dataset.csi_paths
    assert not hasattr(dataset, "person_num")
    assert not hasattr(dataset, "poses")


def test_batches_expose_only_declared_csi_tensor(tmp_path):
    data_root = make_unlabelled_root(tmp_path)
    dataset = UnlabelledPiW3DCSIDataset(data_root)
    loader = make_unlabelled_csi_dataloader(
        dataset,
        is_training=False,
        generator=torch.manual_seed(0),
        batch_size=3,
        num_workers=0,
    )

    batch = next(iter(loader))

    assert tuple(dataset.batch_keys) == ("csi",)
    assert set(batch.keys()) == {"csi"}
    assert batch["csi"].shape == (3, 3, 180, 20)


def test_explicit_id_list_is_restricted_to_official_training_ids(tmp_path):
    data_root = make_unlabelled_root(tmp_path)
    selected_ids_path = tmp_path / "selected_ids.txt"
    selected_ids_path.write_text("\n".join(MIXED_COUNT_IDS) + "\n", encoding="utf-8")

    dataset = UnlabelledPiW3DCSIDataset(data_root, id_list_path=selected_ids_path)
    assert len(dataset) == len(MIXED_COUNT_IDS)

    selected_ids_path.write_text("S11_01_10\nS99_01_10\n", encoding="utf-8")
    with pytest.raises(ValueError, match="absent from the official training list"):
        UnlabelledPiW3DCSIDataset(data_root, id_list_path=selected_ids_path)


def test_technical_shape_and_finite_validation(tmp_path):
    data_root = make_unlabelled_root(tmp_path)
    bad_shape_path = data_root / "train_data" / "csi_ap" / "S12_01_10.npy"
    np.save(bad_shape_path, np.zeros((3, 90, 20), dtype=np.float32))
    dataset = UnlabelledPiW3DCSIDataset(data_root)
    with pytest.raises(ValueError, match="Unexpected unlabelled CSI shape"):
        dataset[1]

    data_root = make_unlabelled_root(tmp_path / "nonfinite")
    nonfinite_path = data_root / "train_data" / "csi_ap" / "S11_01_10.npy"
    csi = make_processed_csi(0.0)
    csi[0, 0, 0] = np.nan
    np.save(nonfinite_path, csi)
    dataset = UnlabelledPiW3DCSIDataset(data_root)
    with pytest.raises(ValueError, match="Non-finite unlabelled CSI"):
        dataset[0]


def test_missing_csi_is_technically_skipped_without_count_filtering(tmp_path):
    data_root = make_unlabelled_root(tmp_path)
    (data_root / "train_data" / "csi_ap" / "S13_01_10.npy").unlink()

    dataset = UnlabelledPiW3DCSIDataset(data_root)

    assert len(dataset) == 2
    assert dataset.skipped_count == 1
