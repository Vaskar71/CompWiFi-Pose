"""Step-5 preprocessing (B1): copy selected raw files, run process_ids, validate, freeze stats.

Originals under the source root are only read. Outputs go to the Step-5 data root; the
train-split input statistics and pose-codec statistics are written once to the freeze dir.
"""

import argparse
import contextlib
import hashlib
import io
import json
import shutil
import time
from pathlib import Path

import numpy as np

from data.person_in_wifi_3d.process import process_ids
from make_step4b_pilot_data import _append_issue, _validate_processed_sample
from step4b_subset import is_single_person_id
from step5_config import resolve_path, step5_config_for_json
from step5_subset import FREEZE_JSON_NAME, ID_LIST_NAMES, SPLITS


VALIDATION_NAME = "preprocessing_validation_v1.json"


def _load_freeze(freeze_dir):
    freeze = json.loads((Path(freeze_dir) / FREEZE_JSON_NAME).read_text(encoding="utf-8"))
    for split in SPLITS:
        listed = [
            line.strip()
            for line in (Path(freeze_dir) / ID_LIST_NAMES[split]).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if listed != freeze["selected_ids"][split]:
            raise AssertionError(f"Freeze ID list for {split} does not match the freeze JSON")
    return freeze


def _sha256_lines(sample_ids):
    return hashlib.sha256("".join(f"{sample_id}\n" for sample_id in sample_ids).encode("utf-8")).hexdigest()


def preflight(source_root, ordered_ids):
    source_train = Path(source_root) / "train_data"
    missing = []
    for sample_id in ordered_ids:
        if not is_single_person_id(sample_id):
            raise ValueError(f"Non-single-person ID in Step-5 freeze: {sample_id}")
        for relative in (Path("csi") / f"{sample_id}.mat", Path("keypoint") / f"{sample_id}.npy"):
            if not (source_train / relative).is_file():
                missing.append(str(source_train / relative))
    if missing:
        raise FileNotFoundError("Missing source files: " + ", ".join(missing[:20]))


def copy_selected_files(source_root, data_root, freeze):
    source_train = Path(source_root) / "train_data"
    target_train = Path(data_root) / "train_data"
    for directory in ("csi", "keypoint", "csi_amplitude", "csi_phase", "csi_ap"):
        (target_train / directory).mkdir(parents=True, exist_ok=True)
    ordered_ids = [sample_id for split in SPLITS for sample_id in freeze["selected_ids"][split]]
    (target_train / "train_data_list.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in ordered_ids), encoding="utf-8"
    )
    for split in SPLITS:
        (target_train / ID_LIST_NAMES[split]).write_text(
            "".join(f"{sample_id}\n" for sample_id in freeze["selected_ids"][split]), encoding="utf-8"
        )
    for sample_id in ordered_ids:
        shutil.copyfile(source_train / "csi" / f"{sample_id}.mat", target_train / "csi" / f"{sample_id}.mat")
        shutil.copyfile(
            source_train / "keypoint" / f"{sample_id}.npy", target_train / "keypoint" / f"{sample_id}.npy"
        )
    return ordered_ids


def write_train_statistics(data_root, freeze_dir, train_ids):
    from feeder.piw3d_step5 import compute_train_input_stats, save_input_stats
    from model.research_models import compute_codec_stats, save_codec_stats

    config = step5_config_for_json()
    train_root = Path(data_root) / "train_data"
    train_sha = _sha256_lines(train_ids)
    input_stats = compute_train_input_stats(train_root / "csi_ap" / f"{sample_id}.npy" for sample_id in train_ids)
    input_path = save_input_stats(Path(freeze_dir) / config["input_norm"]["stats_file"], input_stats, train_sha)
    poses = np.stack(
        [np.load(train_root / "keypoint" / f"{sample_id}.npy", allow_pickle=False) for sample_id in train_ids]
    )
    codec_stats = compute_codec_stats(poses)
    codec_path = save_codec_stats(Path(freeze_dir) / config["target_codec"]["stats_file"], codec_stats, train_sha)
    return {
        "input_stats_path": str(input_path.resolve()),
        "codec_stats_path": str(codec_path.resolve()),
        "train_id_sha256": train_sha,
        "input_std_min": float(np.min(input_stats["std"])),
        "input_std_max": float(np.max(input_stats["std"])),
        "codec_stats": codec_stats,
    }


def build_step5_data(repo_root=".", source_root=None, freeze_dir=None, data_root=None, process_fn=process_ids):
    source_root = Path(source_root) if source_root else resolve_path(repo_root, "source_root")
    freeze_dir = Path(freeze_dir) if freeze_dir else resolve_path(repo_root, "freeze_dir")
    data_root = Path(data_root) if data_root else resolve_path(repo_root, "data_root")
    if data_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing Step-5 data root: {data_root}")
    validation_path = freeze_dir / VALIDATION_NAME
    if validation_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing validation report: {validation_path}")
    freeze = _load_freeze(freeze_dir)
    ordered_ids = [sample_id for split in SPLITS for sample_id in freeze["selected_ids"][split]]
    preflight(source_root, ordered_ids)
    split_assignments = json.loads((freeze_dir / "split_assignments.json").read_text(encoding="utf-8"))

    started = time.time()
    data_root.mkdir(parents=True)
    copy_selected_files(source_root, data_root, freeze)
    target_train = data_root / "train_data"
    failures = []
    records = []
    for split in SPLITS:
        for sample_id in freeze["selected_ids"][split]:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    process_fn([sample_id], data_root, "train_data")
            except Exception as exc:  # recorded, then the build fails below
                _append_issue(failures, sample_id, f"process_ids failed: {exc}")
            records.append(
                _validate_processed_sample(target_train, sample_id, split, split_assignments, failures)
            )
            if len(records) % 500 == 0:
                print(f"validated={len(records)}/{len(ordered_ids)} failures={len(failures)}", flush=True)

    validation = {
        "freeze_dir": str(freeze_dir.resolve()),
        "data_root": str(data_root.resolve()),
        "source_root": str(source_root.resolve()),
        "official_test_used": False,
        "source_dataset_modified": False,
        "process_path": "data.person_in_wifi_3d.process.process_ids",
        "counts": {
            **{f"{split}_ids": len(freeze["selected_ids"][split]) for split in SPLITS},
            "validated_ids": len(records),
            "failed_ids": len(failures),
        },
        "exact_amplitude_phase_concatenation_required": True,
        "elapsed_seconds": None,
        "failures": failures,
    }
    if not failures:
        validation["train_statistics"] = write_train_statistics(
            data_root, freeze_dir, freeze["selected_ids"]["train"]
        )
    validation["elapsed_seconds"] = time.time() - started
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise ValueError(f"Step-5 preprocessing failed for {len(failures)} samples; see {validation_path}")
    return validation


def parse_args():
    parser = argparse.ArgumentParser(description="Preprocess and validate the frozen Step-5 subset.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    return parser.parse_args()


def main():
    args = parse_args()
    validation = build_step5_data(repo_root=args.repo_root)
    print(json.dumps(validation["counts"], indent=2, sort_keys=True))
    print(f"elapsed_seconds={validation['elapsed_seconds']:.1f}")


if __name__ == "__main__":
    main()
