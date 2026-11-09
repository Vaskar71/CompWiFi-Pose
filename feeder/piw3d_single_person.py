import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from piw3d_manifest import parse_sample_id
from step4b_subset import is_single_person_id


EXPECTED_CSI_SHAPE = (3, 180, 20)
EXPECTED_KEYPOINT_SHAPE = (1, 14, 3)
BATCH_KEYS = ("csi", "pose")


def _read_id_list(path):
    path = Path(path)
    sample_ids = [line.split()[0] for line in path.read_text(encoding="utf-8-sig").splitlines() if line.split()]
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


class SinglePersonPiW3DDataset(Dataset):
    batch_keys = BATCH_KEYS

    def __init__(
        self,
        data_root,
        id_list_path,
        expected_split=None,
        split_assignments_path=None,
        expected_csi_shape=EXPECTED_CSI_SHAPE,
        expected_keypoint_shape=EXPECTED_KEYPOINT_SHAPE,
    ):
        self.data_root = Path(data_root)
        self.train_root = self.data_root / "train_data"
        self.csi_dir = self.train_root / "csi_ap"
        self.keypoint_dir = self.train_root / "keypoint"
        self.expected_csi_shape = tuple(expected_csi_shape)
        self.expected_keypoint_shape = tuple(expected_keypoint_shape)
        if len(self.expected_csi_shape) != 3 or len(self.expected_keypoint_shape) != 3:
            raise ValueError("Expected CSI and keypoint shapes must have three dimensions")

        official_list_path = self.train_root / "train_data_list.txt"
        official_ids = set(_read_id_list(official_list_path))
        selected_ids = _read_id_list(id_list_path)
        unknown_ids = [sample_id for sample_id in selected_ids if sample_id not in official_ids]
        if unknown_ids:
            raise ValueError(
                "Single-person IDs are absent from the pilot training list: "
                + ", ".join(unknown_ids[:5])
            )

        split_assignments = None
        if split_assignments_path is not None:
            split_assignments = json.loads(Path(split_assignments_path).read_text(encoding="utf-8"))
        self.samples = []
        for sample_id in selected_ids:
            if not is_single_person_id(sample_id):
                raise ValueError(f"Selected ID is not filename-marked single-person: {sample_id}")
            if expected_split is not None:
                assigned_split = split_assignments.get(sample_id) if split_assignments else None
                if assigned_split != expected_split:
                    raise ValueError(
                        f"Split assignment for {sample_id} is {assigned_split!r}, expected {expected_split!r}"
                    )
            csi_path = self.csi_dir / f"{sample_id}.npy"
            keypoint_path = self.keypoint_dir / f"{sample_id}.npy"
            if not csi_path.is_file():
                raise FileNotFoundError(f"Missing pilot CSI file: {csi_path}")
            if not keypoint_path.is_file():
                raise FileNotFoundError(f"Missing pilot keypoint file: {keypoint_path}")
            pose = self.read_pose(keypoint_path)
            if pose.shape != self.expected_keypoint_shape:
                raise ValueError(
                    f"Unexpected pilot keypoint shape for {sample_id}: {pose.shape}"
                )
            self.samples.append({"sample_id": sample_id, "csi_path": csi_path, "keypoint_path": keypoint_path})
        self.sample_ids = [sample["sample_id"] for sample in self.samples]

    def __len__(self):
        return len(self.samples)

    def read_csi(self, csi_path):
        stored = np.load(csi_path, allow_pickle=False)
        if stored.shape != self.expected_csi_shape:
            raise ValueError(f"Unexpected pilot CSI shape for {csi_path}: {stored.shape}")
        if not np.isfinite(stored).all():
            raise ValueError(f"Non-finite pilot CSI for {csi_path}")
        split_channel = self.expected_csi_shape[1] // 2
        amplitude = _min_max_normalize(stored[:, :split_channel, :])
        phase = _min_max_normalize(stored[:, split_channel:, :])
        normalized = np.concatenate((amplitude, phase), axis=1)
        return torch.from_numpy(normalized.astype(np.float32))

    def read_pose(self, keypoint_path):
        pose = np.load(keypoint_path, allow_pickle=False)
        if pose.shape != self.expected_keypoint_shape:
            raise ValueError(f"Unexpected pilot keypoint shape for {keypoint_path}: {pose.shape}")
        if not np.isfinite(pose).all():
            raise ValueError(f"Non-finite pilot keypoint values for {keypoint_path}")
        return torch.from_numpy(pose.astype(np.float32))

    def __getitem__(self, index):
        sample = self.samples[index]
        return {
            "csi": self.read_csi(sample["csi_path"]),
            "pose": self.read_pose(sample["keypoint_path"]),
        }


def collate_single_person(batch):
    return {
        "csi": torch.stack([sample["csi"] for sample in batch]),
        "pose": torch.stack([sample["pose"] for sample in batch]),
    }


def make_single_person_dataloader(
    dataset,
    is_training,
    generator,
    batch_size,
    num_workers=0,
):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_single_person,
        shuffle=is_training,
        drop_last=False,
        generator=generator,
        num_workers=num_workers,
    )
