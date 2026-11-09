import argparse
import contextlib
import io
import json
import shutil
from pathlib import Path

import numpy as np

from data.person_in_wifi_3d.process import process_ids
from step4b_config import validate_step4b_config
from step4b_subset import is_single_person_id, read_manifest_ids


DEFAULT_SOURCE_ROOT = Path("data/person_in_wifi_3d/wifi_dataset")
DEFAULT_FREEZE_CONFIG = Path("data/_artifacts/step4b_pilot/freeze_v2/step4b_freeze_config.json")
DEFAULT_PILOT_ROOT = Path("data/_artifacts/step4b_pilot/pilot_data_v2")
DEFAULT_VALIDATION_OUTPUT = Path("data/_artifacts/step4b_pilot/preprocessing_validation_v2.json")
EXPECTED_AMPLITUDE_SHAPE = (3, 90, 20)
EXPECTED_PHASE_SHAPE = (3, 90, 20)
EXPECTED_CSI_SHAPE = (3, 180, 20)
EXPECTED_KEYPOINT_SHAPE = (1, 14, 3)


def _load_freeze(path):
    freeze = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_step4b_config(freeze["config"])
    return freeze


def _selected_ids(freeze):
    train_ids = list(freeze["selected_ids"]["internal_train"])
    development_ids = list(freeze["selected_ids"]["internal_development"])
    if set(train_ids) & set(development_ids):
        raise AssertionError("Frozen train/development ID overlap must be zero")
    return train_ids, development_ids


def _preflight_source_files(source_root, train_ids, development_ids):
    source_train_root = Path(source_root) / "train_data"
    missing = []
    non_single_person = []
    for sample_id in train_ids + development_ids:
        if not is_single_person_id(sample_id):
            non_single_person.append(sample_id)
        for relative_path in (
            Path("csi") / f"{sample_id}.mat",
            Path("keypoint") / f"{sample_id}.npy",
        ):
            if not (source_train_root / relative_path).is_file():
                missing.append(str(source_train_root / relative_path))
    if non_single_person:
        raise ValueError("Non-single-person IDs in frozen subset: " + ", ".join(non_single_person[:10]))
    if missing:
        raise FileNotFoundError("Missing required source files: " + ", ".join(missing[:20]))


def _copy_fixture_files(source_root, pilot_root, train_ids, development_ids):
    source_train_root = Path(source_root) / "train_data"
    pilot_train_root = Path(pilot_root) / "train_data"
    for directory in ("csi", "keypoint", "csi_amplitude", "csi_phase", "csi_ap"):
        (pilot_train_root / directory).mkdir(parents=True, exist_ok=True)
    all_selected_ids = train_ids + development_ids
    (pilot_train_root / "train_data_list.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in all_selected_ids), encoding="utf-8"
    )
    (pilot_train_root / "selected_train_ids.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in train_ids), encoding="utf-8"
    )
    (pilot_train_root / "selected_development_ids.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in development_ids), encoding="utf-8"
    )
    for sample_id in all_selected_ids:
        shutil.copyfile(
            source_train_root / "csi" / f"{sample_id}.mat",
            pilot_train_root / "csi" / f"{sample_id}.mat",
        )
        shutil.copyfile(
            source_train_root / "keypoint" / f"{sample_id}.npy",
            pilot_train_root / "keypoint" / f"{sample_id}.npy",
        )


def _append_issue(failures, sample_id, issue):
    for failure in failures:
        if failure["sample_id"] == sample_id:
            failure["issues"].append(issue)
            return
    failures.append({"sample_id": sample_id, "issues": [issue]})


def _validate_processed_sample(pilot_train_root, sample_id, expected_split, split_assignments, failures):
    amplitude_path = pilot_train_root / "csi_amplitude" / f"{sample_id}.npy"
    phase_path = pilot_train_root / "csi_phase" / f"{sample_id}.npy"
    csi_path = pilot_train_root / "csi_ap" / f"{sample_id}.npy"
    keypoint_path = pilot_train_root / "keypoint" / f"{sample_id}.npy"

    if split_assignments.get(sample_id) != expected_split:
        _append_issue(
            failures,
            sample_id,
            f"split assignment is {split_assignments.get(sample_id)!r}, expected {expected_split!r}",
        )

    arrays = {}
    for name, path, expected_shape in (
        ("amplitude", amplitude_path, EXPECTED_AMPLITUDE_SHAPE),
        ("phase", phase_path, EXPECTED_PHASE_SHAPE),
        ("csi_ap", csi_path, EXPECTED_CSI_SHAPE),
        ("keypoint", keypoint_path, EXPECTED_KEYPOINT_SHAPE),
    ):
        if not path.is_file():
            _append_issue(failures, sample_id, f"missing {name}: {path}")
            continue
        try:
            arrays[name] = np.load(path, allow_pickle=False)
        except Exception as exc:
            _append_issue(failures, sample_id, f"unreadable {name}: {exc}")
            continue
        if arrays[name].shape != expected_shape:
            _append_issue(
                failures,
                sample_id,
                f"{name} shape {arrays[name].shape} != {expected_shape}",
            )
        if not np.isfinite(arrays[name]).all():
            _append_issue(failures, sample_id, f"non-finite values in {name}")

    if {"amplitude", "phase", "csi_ap"}.issubset(arrays):
        if not np.array_equal(arrays["csi_ap"], np.concatenate((arrays["amplitude"], arrays["phase"]), axis=1)):
            _append_issue(failures, sample_id, "csi_ap is not exact amplitude/phase concatenation")
    return {"sample_id": sample_id, "split": expected_split, "passed": not any(f["sample_id"] == sample_id for f in failures)}


def build_step4b_pilot_data(
    source_root=DEFAULT_SOURCE_ROOT,
    freeze_config_path=DEFAULT_FREEZE_CONFIG,
    pilot_root=DEFAULT_PILOT_ROOT,
    validation_output=DEFAULT_VALIDATION_OUTPUT,
):
    freeze = _load_freeze(freeze_config_path)
    train_ids, development_ids = _selected_ids(freeze)
    split_assignments = freeze["split_assignments"]
    source_root = Path(source_root)
    pilot_root = Path(pilot_root)
    validation_output = Path(validation_output)
    if pilot_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing pilot root: {pilot_root}")
    _preflight_source_files(source_root, train_ids, development_ids)

    pilot_root.mkdir(parents=True)
    try:
        _copy_fixture_files(source_root, pilot_root, train_ids, development_ids)
        pilot_train_root = pilot_root / "train_data"
        failures = []
        records = []
        ordered_ids = [(sample_id, "internal_train") for sample_id in train_ids] + [
            (sample_id, "internal_development") for sample_id in development_ids
        ]
        for index, (sample_id, expected_split) in enumerate(ordered_ids, start=1):
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    process_ids([sample_id], pilot_root, "train_data")
            except Exception as exc:
                _append_issue(failures, sample_id, f"process_ids failed: {exc}")
            records.append(
                _validate_processed_sample(
                    pilot_train_root,
                    sample_id,
                    expected_split,
                    split_assignments,
                    failures,
                )
            )
            if index % 100 == 0:
                print(f"validated={index}/{len(ordered_ids)} failures={len(failures)}")
    except Exception:
        raise

    validation = {
        "freeze_config": str(Path(freeze_config_path).resolve()),
        "pilot_root": str(pilot_root.resolve()),
        "source_root": str(source_root.resolve()),
        "official_test_used": False,
        "source_dataset_modified": False,
        "process_path": "data.person_in_wifi_3d.process.process_ids",
        "counts": {
            "selected_train_ids": len(train_ids),
            "selected_development_ids": len(development_ids),
            "total_selected_ids": len(ordered_ids),
            "validated_ids": len(records),
            "failed_ids": len(failures),
        },
        "expected_shapes": {
            "csi_amplitude": list(EXPECTED_AMPLITUDE_SHAPE),
            "csi_phase": list(EXPECTED_PHASE_SHAPE),
            "csi_ap": list(EXPECTED_CSI_SHAPE),
            "keypoint": list(EXPECTED_KEYPOINT_SHAPE),
        },
        "exact_amplitude_phase_concatenation_required": True,
        "records": records,
        "failures": failures,
    }
    validation_output.parent.mkdir(parents=True, exist_ok=True)
    validation_output.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise ValueError(f"Step-4B pilot preprocessing validation failed for {len(failures)} samples; see {validation_output}")
    print(json.dumps(validation["counts"], indent=2, sort_keys=True))
    print(f"validation_report={validation_output}")
    return validation


def parse_args():
    parser = argparse.ArgumentParser(description="Build and validate the bounded Step-4B pilot fixture.")
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--freeze-config", type=Path, default=DEFAULT_FREEZE_CONFIG)
    parser.add_argument("--pilot-root", type=Path, default=DEFAULT_PILOT_ROOT)
    parser.add_argument("--validation-output", type=Path, default=DEFAULT_VALIDATION_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    build_step4b_pilot_data(
        source_root=args.source_root,
        freeze_config_path=args.freeze_config,
        pilot_root=args.pilot_root,
        validation_output=args.validation_output,
    )


if __name__ == "__main__":
    main()
