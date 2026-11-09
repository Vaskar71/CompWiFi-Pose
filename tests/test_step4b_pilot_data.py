import json

import h5py
import numpy as np
import pytest

from make_step4b_pilot_data import build_step4b_pilot_data
from step4b_config import step4b_config_for_json


SELECTED_ID = "S11_01_10"


def write_raw_csi(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    real = np.linspace(-1.0, 1.0, 20 * 30 * 3 * 3).reshape(20, 30, 3, 3)
    imag = np.linspace(0.5, -0.5, 20 * 30 * 3 * 3).reshape(20, 30, 3, 3)
    csi = np.empty(real.shape, dtype=[("real", "<f8"), ("imag", "<f8")])
    csi["real"] = real
    csi["imag"] = imag
    with h5py.File(path, "w") as raw_file:
        raw_file.create_dataset("csi_out", data=csi)


def make_source_root(tmp_path):
    train_dir = tmp_path / "source" / "train_data"
    (train_dir / "csi").mkdir(parents=True)
    (train_dir / "keypoint").mkdir()
    write_raw_csi(train_dir / "csi" / f"{SELECTED_ID}.mat")
    pose = np.arange(14 * 3, dtype=np.float32).reshape(1, 14, 3) * 0.01
    np.save(train_dir / "keypoint" / f"{SELECTED_ID}.npy", pose)
    return tmp_path / "source"


def write_freeze(tmp_path, split="internal_train"):
    freeze = {
        "config": step4b_config_for_json(),
        "split_label": "candidate-group-disjoint",
        "selected_ids": {"internal_train": [SELECTED_ID], "internal_development": []},
        "selected_groups": {"internal_train": [], "internal_development": []},
        "split_assignments": {SELECTED_ID: split},
        "counts": {},
    }
    freeze_dir = tmp_path / "freeze"
    freeze_dir.mkdir()
    path = freeze_dir / "step4b_freeze_config.json"
    path.write_text(json.dumps(freeze), encoding="utf-8")
    return path


def test_pilot_data_builder_processes_and_validates_explicit_id(tmp_path):
    source_root = make_source_root(tmp_path)
    freeze_path = write_freeze(tmp_path)
    pilot_root = tmp_path / "pilot"
    validation_path = tmp_path / "validation.json"

    validation = build_step4b_pilot_data(
        source_root=source_root,
        freeze_config_path=freeze_path,
        pilot_root=pilot_root,
        validation_output=validation_path,
    )

    assert validation["counts"]["validated_ids"] == 1
    assert validation["counts"]["failed_ids"] == 0
    assert validation["official_test_used"] is False
    assert validation["source_dataset_modified"] is False
    assert (pilot_root / "train_data" / "csi" / f"{SELECTED_ID}.mat").is_file()
    assert (pilot_root / "train_data" / "keypoint" / f"{SELECTED_ID}.npy").is_file()
    assert (pilot_root / "train_data" / "csi_ap" / f"{SELECTED_ID}.npy").is_file()
    persisted = json.loads(validation_path.read_text(encoding="utf-8"))
    assert persisted == validation


def test_pilot_data_builder_reports_split_assignment_failure_with_id(tmp_path):
    source_root = make_source_root(tmp_path)
    freeze_path = write_freeze(tmp_path, split="internal_development")
    pilot_root = tmp_path / "pilot"
    validation_path = tmp_path / "validation.json"

    with pytest.raises(ValueError, match="validation failed for 1 samples"):
        build_step4b_pilot_data(
            source_root=source_root,
            freeze_config_path=freeze_path,
            pilot_root=pilot_root,
            validation_output=validation_path,
        )

    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    assert validation["failures"][0]["sample_id"] == SELECTED_ID
    assert "expected 'internal_train'" in validation["failures"][0]["issues"][0]
