"""Step-5b dataset: adds ``recording_centered`` input normalisation (plan section 5).

``Step5bDataset`` subclasses ``Step5SinglePersonDataset`` and overrides ``read_csi`` only.
The recording of each sample is resolved once in the constructor from its ID; the training
loop never sees IDs. Batches remain ``{"csi", "pose"}``.
"""

from pathlib import Path

import numpy as np
import torch

from feeder.piw3d_step5 import STATS_SHAPE, STD_FLOOR, Step5SinglePersonDataset
from piw3d_manifest import parse_sample_id


STEP5B_INPUT_NORMS = ("legacy_minmax", "train_zscore", "recording_centered")


def load_recording_means(path):
    with np.load(path, allow_pickle=False) as stored:
        names = [str(n) for n in stored["recordings"]]
        means = np.array(stored["means"], dtype=np.float64)
    if means.shape != (len(names),) + STATS_SHAPE or not np.isfinite(means).all():
        raise ValueError("Recording input means have the wrong shape or are not finite")
    return {name: means[index] for index, name in enumerate(names)}


class Step5bDataset(Step5SinglePersonDataset):
    def __init__(self, *args, input_norm="legacy_minmax", input_stats=None, recording_means=None, **kwargs):
        if input_norm not in STEP5B_INPUT_NORMS:
            raise ValueError(f"input_norm must be one of {STEP5B_INPUT_NORMS}, got {input_norm!r}")
        if input_norm == "recording_centered":
            if recording_means is None or input_stats is None:
                raise ValueError("recording_centered requires recording means and the stage's train std")
            super().__init__(*args, input_norm="train_zscore", input_stats=input_stats, **kwargs)
            self.input_norm = "recording_centered"
            self._path_mean = {}
            for sample in self.samples:
                recording = parse_sample_id(sample["sample_id"])["group"]
                if recording not in recording_means:
                    raise KeyError(f"No input mean for recording {recording}")
                self._path_mean[Path(sample["csi_path"])] = np.asarray(recording_means[recording], dtype=np.float64)
        else:
            if recording_means is not None:
                raise ValueError("recording means are used only by recording_centered")
            super().__init__(*args, input_norm=input_norm, input_stats=input_stats, **kwargs)

    def read_csi(self, csi_path):
        if self.input_norm != "recording_centered":
            return super().read_csi(csi_path)
        stored = np.load(csi_path, allow_pickle=False)
        if stored.shape != self.expected_csi_shape:
            raise ValueError(f"Unexpected CSI shape for {csi_path}: {stored.shape}")
        if not np.isfinite(stored).all():
            raise ValueError(f"Non-finite CSI for {csi_path}")
        mean = self._path_mean[Path(csi_path)]
        normalized = (stored.astype(np.float64) - mean) / self._zscore_scale
        return torch.from_numpy(normalized.astype(np.float32))


__all__ = ["STEP5B_INPUT_NORMS", "Step5bDataset", "load_recording_means", "STD_FLOOR"]
