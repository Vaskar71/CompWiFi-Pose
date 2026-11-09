from pathlib import Path

import h5py
import numpy as np


def write_raw_csi(path: Path, offset: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    real = np.linspace(-1.0 + offset, 1.0 + offset, 20 * 30 * 3 * 3).reshape(
        20, 30, 3, 3
    )
    imag = np.linspace(0.5 + offset, -0.5 + offset, 20 * 30 * 3 * 3).reshape(
        20, 30, 3, 3
    )
    csi = np.empty(real.shape, dtype=[("real", "<f8"), ("imag", "<f8")])
    csi["real"] = real
    csi["imag"] = imag
    with h5py.File(path, "w") as raw_file:
        raw_file.create_dataset("csi_out", data=csi)


def test_process_id_saves_finite_exactly_concatenated_channels(tmp_path):
    from data.person_in_wifi_3d.process import process_id

    train_root = tmp_path / "train_data"
    write_raw_csi(train_root / "csi" / "S11_01_10.mat")

    amplitude, phase, combined = process_id("S11_01_10", train_root)

    assert amplitude.shape == (3, 90, 20)
    assert phase.shape == (3, 90, 20)
    assert combined.shape == (3, 180, 20)
    assert np.isfinite(amplitude).all()
    assert np.isfinite(phase).all()
    assert np.isfinite(combined).all()
    np.testing.assert_array_equal(
        combined, np.concatenate((amplitude, phase), axis=1)
    )
    for directory, expected in (
        ("csi_amplitude", amplitude),
        ("csi_phase", phase),
        ("csi_ap", combined),
    ):
        stored = np.load(train_root / directory / "S11_01_10.npy", allow_pickle=False)
        np.testing.assert_array_equal(stored, expected)


def test_process_ids_is_bounded_to_explicit_training_ids(tmp_path):
    from data.person_in_wifi_3d.process import process_ids

    data_root = tmp_path / "fixture"
    selected_ids = ["S11_01_10", "S11_01_11"]
    for index, name in enumerate(selected_ids + ["S11_01_12"]):
        write_raw_csi(
            data_root / "train_data" / "csi" / f"{name}.mat", offset=index * 0.1
        )

    results = process_ids(selected_ids, data_root, split="train_data")

    assert list(results) == selected_ids
    assert (data_root / "train_data" / "csi_ap" / "S11_01_10.npy").is_file()
    assert (data_root / "train_data" / "csi_ap" / "S11_01_11.npy").is_file()
    assert not (data_root / "train_data" / "csi_ap" / "S11_01_12.npy").exists()
