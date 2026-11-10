"""Step-5b dense preprocessing into ``data_v2`` plus frozen statistics (plan section 4).

Copies the selected raw ``.mat``/keypoint files (originals are only read), runs
``process_ids`` per sample, validates every output (shape, finiteness, exact
amplitude/phase concatenation, role assignment) and writes, once, into ``freeze_v2``:
``input_stats_{fit,all}.npz``, ``codec_stats_{fit,all}.json`` and ``recording_input_means.npz``.
Guard frames are not preprocessed.
"""

import argparse
import contextlib
import io
import json
import time
from pathlib import Path

import numpy as np

from data.person_in_wifi_3d.process import process_ids
from make_step4b_pilot_data import _append_issue, _validate_processed_sample
from make_step5_data import copy_selected_files, preflight
from step5b_config import ROLES, VIEWS, resolve_path, step5b_config_for_json
from step5b_subset import FREEZE_JSON_NAME, SPLIT_ASSIGNMENTS_NAME, VIEW_FILE, _sha256_lines, load_view_ids


VALIDATION_NAME = "preprocessing_validation_v2.json"
RECORDING_MEANS_NAME = "recording_input_means.npz"
STAT_VIEWS = {"fit": "fit_train_dense", "all": "all_train_dense"}


def role_ids(views):
    return {"train": views["all_train_dense"], "temporal_holdout": views["all_holdout"], "group_dev": views["group_dev"]}


def recording_input_means(csi_dir, recording_frames):
    """recording_frames: {recording: [ids]} -> names, means (R,3,180,1), counts. Label-free."""
    names = sorted(recording_frames)
    means = []
    counts = []
    for name in names:
        total = np.zeros((3, 180, 1), dtype=np.float64)
        values = 0
        for sample_id in recording_frames[name]:
            stored = np.load(Path(csi_dir) / f"{sample_id}.npy", allow_pickle=False).astype(np.float64)
            if stored.shape != (3, 180, 20) or not np.isfinite(stored).all():
                raise ValueError(f"Invalid stored CSI for {sample_id}")
            total += stored.sum(axis=2, keepdims=True)
            values += stored.shape[2]
        means.append(total / values)
        counts.append(len(recording_frames[name]))
    return names, np.stack(means), np.asarray(counts)


def write_statistics(data_root, freeze_dir, views):
    from feeder.piw3d_step5 import compute_train_input_stats, save_input_stats
    from model.research_models import compute_codec_stats, save_codec_stats

    train_root = Path(data_root) / "train_data"
    freeze_dir = Path(freeze_dir)
    summary = {}
    for suffix, view in STAT_VIEWS.items():
        ids = views[view]
        sha = _sha256_lines(ids)
        stats = compute_train_input_stats(train_root / "csi_ap" / f"{s}.npy" for s in ids)
        save_input_stats(freeze_dir / f"input_stats_{suffix}.npz", stats, sha)
        poses = np.stack([np.load(train_root / "keypoint" / f"{s}.npy", allow_pickle=False) for s in ids])
        codec = compute_codec_stats(poses)
        save_codec_stats(freeze_dir / f"codec_stats_{suffix}.json", codec, sha)
        summary[suffix] = {
            "view": view,
            "id_sha256": sha,
            "input_std_min": float(np.min(stats["std"])),
            "input_std_max": float(np.max(stats["std"])),
            "codec_mu_c": codec["mu_c"],
            "codec_sigma_c": codec["sigma_c"],
            "codec_sigma_r": codec["sigma_r"],
        }

    recording_frames = {}
    for sample_id in views["all_train_dense"]:
        recording_frames.setdefault(sample_id.rsplit("_", 1)[0], []).append(sample_id)
    dev_frames = {}
    for sample_id in views["group_dev"]:
        dev_frames.setdefault(sample_id.rsplit("_", 1)[0], []).append(sample_id)
    if set(recording_frames) & set(dev_frames):
        raise AssertionError("group_dev recordings overlap training recordings")
    recording_frames.update(dev_frames)
    names, means, counts = recording_input_means(train_root / "csi_ap", recording_frames)
    path = freeze_dir / RECORDING_MEANS_NAME
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    sources = np.asarray(["group_dev_own_frames" if n in dev_frames else "train_role_frames" for n in names])
    np.savez(path, recordings=np.asarray(names), means=means, frame_counts=counts, sources=sources)
    summary["recording_input_means"] = {
        "recordings": len(names),
        "train_recordings": int(np.sum(sources == "train_role_frames")),
        "group_dev_recordings": int(np.sum(sources == "group_dev_own_frames")),
        "label_free": True,
        "transductive_for_unseen_recordings": True,
    }
    return summary


def build_step5b_data(repo_root=".", source_root=None, freeze_dir=None, data_root=None, process_fn=process_ids, progress_every=1000):
    source_root = Path(source_root) if source_root else resolve_path(repo_root, "source_root")
    freeze_dir = Path(freeze_dir) if freeze_dir else resolve_path(repo_root, "freeze_dir")
    data_root = Path(data_root) if data_root else resolve_path(repo_root, "data_root")
    if data_root.exists():
        raise FileExistsError(f"Refusing to overwrite existing data root: {data_root}")
    validation_path = freeze_dir / VALIDATION_NAME
    if validation_path.exists():
        raise FileExistsError(f"Refusing to overwrite {validation_path}")
    views = {view: load_view_ids(freeze_dir, view) for view in VIEWS}
    assignments = json.loads((freeze_dir / SPLIT_ASSIGNMENTS_NAME).read_text(encoding="utf-8"))
    roles = role_ids(views)
    ordered = [s for role in ROLES for s in roles[role]]
    if len(ordered) != len(set(ordered)) or set(ordered) != set(assignments):
        raise AssertionError("Role ID lists do not match split_assignments.json")
    preflight(source_root, ordered)

    started = time.time()
    data_root.mkdir(parents=True)
    copy_selected_files(source_root, data_root, {"selected_ids": roles})
    target_train = data_root / "train_data"
    for view in VIEWS:
        (target_train / VIEW_FILE.format(view=view)).write_text("".join(f"{s}\n" for s in views[view]), encoding="utf-8")
    failures = []
    records = []
    for role in ROLES:
        for sample_id in roles[role]:
            try:
                with contextlib.redirect_stdout(io.StringIO()):
                    process_fn([sample_id], data_root, "train_data")
            except Exception as exc:  # recorded; the build fails below
                _append_issue(failures, sample_id, f"process_ids failed: {exc}")
            records.append(_validate_processed_sample(target_train, sample_id, role, assignments, failures))
            if progress_every and len(records) % progress_every == 0:
                elapsed = time.time() - started
                print(f"validated={len(records)}/{len(ordered)} failures={len(failures)} elapsed={elapsed:.0f}s", flush=True)

    validation = {
        "freeze_dir": str(freeze_dir.resolve()),
        "data_root": str(data_root.resolve()),
        "source_root": str(source_root.resolve()),
        "official_test_used": False,
        "source_dataset_modified": False,
        "guard_frames_preprocessed": False,
        "process_path": "data.person_in_wifi_3d.process.process_ids",
        "counts": {**{f"{role}_ids": len(roles[role]) for role in ROLES}, "validated_ids": len(records), "failed_ids": len(failures)},
        "exact_amplitude_phase_concatenation_required": True,
        "failures": failures,
    }
    if not failures:
        validation["statistics"] = write_statistics(data_root, freeze_dir, views)
    validation["elapsed_seconds"] = time.time() - started
    validation_path.write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    if failures:
        raise ValueError(f"Step-5b preprocessing failed for {len(failures)} samples; see {validation_path}")
    return validation


def main(argv=None):
    parser = argparse.ArgumentParser(description="Preprocess and validate the Step-5b dense freeze.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    validation = build_step5b_data(repo_root=args.repo_root)
    print(json.dumps(validation["counts"], indent=2, sort_keys=True))
    print(f"elapsed_seconds={validation['elapsed_seconds']:.1f}")


if __name__ == "__main__":
    main()
