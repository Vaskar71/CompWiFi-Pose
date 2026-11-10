"""Step-5 single-person dataset wrapper (B2).

Subclasses ``SinglePersonPiW3DDataset`` and overrides only ``read_csi``. Batches remain
``{"csi", "pose"}``; reuse ``collate_single_person`` / ``make_single_person_dataloader``.
"""

from pathlib import Path

import numpy as np
import torch

from feeder.piw3d_single_person import SinglePersonPiW3DDataset


INPUT_NORMS = ("legacy_minmax", "train_zscore")
STATS_SHAPE = (3, 180, 1)
STD_FLOOR = 1e-6


def compute_train_input_stats(csi_paths, expected_shape=(3, 180, 20)):
    """Mean/std of shape (3,180,1) over samples and time, from stored csi_ap arrays (float64)."""
    total = np.zeros(STATS_SHAPE, dtype=np.float64)
    total_sq = np.zeros(STATS_SHAPE, dtype=np.float64)
    count = 0
    for path in csi_paths:
        stored = np.load(path, allow_pickle=False).astype(np.float64)
        if stored.shape != tuple(expected_shape):
            raise ValueError(f"Unexpected CSI shape for {path}: {stored.shape}")
        if not np.isfinite(stored).all():
            raise ValueError(f"Non-finite CSI for {path}")
        total += stored.sum(axis=2, keepdims=True)
        total_sq += np.square(stored).sum(axis=2, keepdims=True)
        count += stored.shape[2]
    if count == 0:
        raise ValueError("No CSI arrays supplied for input statistics")
    mean = total / count
    variance = np.maximum(total_sq / count - np.square(mean), 0.0)
    return {"mean": mean, "std": np.sqrt(variance), "value_count_per_channel": count}


def save_input_stats(path, stats, train_id_sha256):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite input statistics: {path}")
    np.savez(
        path,
        mean=stats["mean"],
        std=stats["std"],
        value_count_per_channel=np.asarray(stats["value_count_per_channel"]),
        train_id_sha256=np.asarray(train_id_sha256),
    )
    return path


def load_input_stats(path):
    with np.load(path, allow_pickle=False) as stored:
        mean = np.array(stored["mean"], dtype=np.float64)
        std = np.array(stored["std"], dtype=np.float64)
        train_id_sha256 = str(stored["train_id_sha256"])
    if mean.shape != STATS_SHAPE or std.shape != STATS_SHAPE:
        raise ValueError(f"Input statistics must have shape {STATS_SHAPE}")
    if not (np.isfinite(mean).all() and np.isfinite(std).all()):
        raise ValueError("Input statistics must be finite")
    mean.setflags(write=False)
    std.setflags(write=False)
    return {"mean": mean, "std": std, "train_id_sha256": train_id_sha256}


class Step5SinglePersonDataset(SinglePersonPiW3DDataset):
    def __init__(self, *args, input_norm="legacy_minmax", input_stats=None, **kwargs):
        if input_norm not in INPUT_NORMS:
            raise ValueError(f"input_norm must be one of {INPUT_NORMS}, got {input_norm!r}")
        if input_norm == "train_zscore":
            if input_stats is None:
                raise ValueError("train_zscore requires frozen train-split input statistics")
            self._zscore_mean = np.asarray(input_stats["mean"], dtype=np.float64)
            self._zscore_scale = np.maximum(np.asarray(input_stats["std"], dtype=np.float64), STD_FLOOR)
            if self._zscore_mean.shape != STATS_SHAPE or self._zscore_scale.shape != STATS_SHAPE:
                raise ValueError(f"Input statistics must have shape {STATS_SHAPE}")
        elif input_stats is not None:
            raise ValueError("legacy_minmax does not use input statistics")
        self.input_norm = input_norm
        super().__init__(*args, **kwargs)

    def read_csi(self, csi_path):
        if self.input_norm == "legacy_minmax":
            return super().read_csi(csi_path)
        stored = np.load(csi_path, allow_pickle=False)
        if stored.shape != self.expected_csi_shape:
            raise ValueError(f"Unexpected CSI shape for {csi_path}: {stored.shape}")
        if not np.isfinite(stored).all():
            raise ValueError(f"Non-finite CSI for {csi_path}")
        normalized = (stored.astype(np.float64) - self._zscore_mean) / self._zscore_scale
        return torch.from_numpy(normalized.astype(np.float32))
