from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from piw3d_manifest import parse_sample_id


EXPECTED_CSI_SHAPE = (3, 180, 20)
BATCH_KEYS = ("csi",)


def _read_id_list(path):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    sample_ids = [line.split()[0] for line in lines if line.split()]
    if not sample_ids:
        raise ValueError(f"ID list is empty: {path}")
    if len(set(sample_ids)) != len(sample_ids):
        raise ValueError(f"ID list contains duplicate IDs: {path}")
    for sample_id in sample_ids:
        parse_sample_id(sample_id)
    return sample_ids


def _min_max_normalize(data):
    minimum = np.min(data)
    maximum = np.max(data)
    if maximum == minimum:
        return np.zeros_like(data)
    return (data - minimum) / (maximum - minimum)


class UnlabelledPiW3DCSIDataset(Dataset):
    batch_keys = BATCH_KEYS

    def __init__(self, data_root, id_list_path=None, expected_shape=EXPECTED_CSI_SHAPE):
        self.data_root = Path(data_root)
        self.train_root = self.data_root / "train_data"
        self.csi_dir = self.train_root / "csi_ap"
        self.expected_shape = tuple(expected_shape)
        if len(self.expected_shape) != 3:
            raise ValueError("expected_shape must contain three dimensions")

        official_list_path = self.train_root / "train_data_list.txt"
        official_ids = set(_read_id_list(official_list_path))
        if id_list_path is None:
            selected_ids = list(dict.fromkeys(_read_id_list(official_list_path)))
        else:
            selected_ids = _read_id_list(id_list_path)
            unknown_ids = [sample_id for sample_id in selected_ids if sample_id not in official_ids]
            if unknown_ids:
                raise ValueError(
                    "Unlabelled IDs are absent from the official training list: "
                    + ", ".join(unknown_ids[:5])
                )

        self.csi_paths = []
        self.skipped_count = 0
        for sample_id in selected_ids:
            csi_path = self.csi_dir / f"{sample_id}.npy"
            if csi_path.is_file():
                self.csi_paths.append(csi_path)
            else:
                self.skipped_count += 1
        if not self.csi_paths:
            raise ValueError(f"No eligible unlabelled training CSI files under {self.csi_dir}")

    def __len__(self):
        return len(self.csi_paths)

    def read_frame(self, csi_path):
        stored = np.load(csi_path, allow_pickle=False)
        if stored.shape != self.expected_shape:
            raise ValueError(
                f"Unexpected unlabelled CSI shape for {csi_path}: {stored.shape}"
            )
        if not np.isfinite(stored).all():
            raise ValueError(f"Non-finite unlabelled CSI for {csi_path}")
        split_channel = self.expected_shape[1] // 2
        amplitude = _min_max_normalize(stored[:, :split_channel, :])
        phase = _min_max_normalize(stored[:, split_channel:, :])
        normalized = np.concatenate((amplitude, phase), axis=1)
        return torch.from_numpy(normalized.astype(np.float32))

    def __getitem__(self, index):
        return {"csi": self.read_frame(self.csi_paths[index])}


def collate_unlabelled_csi(batch):
    return {"csi": torch.stack([sample["csi"] for sample in batch])}


def make_unlabelled_csi_dataloader(
    dataset,
    is_training,
    generator,
    batch_size,
    num_workers=0,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_unlabelled_csi,
        shuffle=is_training,
        drop_last=is_training,
        generator=generator,
        num_workers=num_workers,
    )
