import argparse
import shutil
from pathlib import Path

import numpy as np

from data.person_in_wifi_3d.process import load_file_name_list, process_ids


def successor_id(name):
    sequence, clip, frame = name.split("_")
    return f"{sequence}_{clip}_{int(frame) + 1}"


def select_consecutive_one_person_ids(train_root):
    training_ids = load_file_name_list(train_root / "train_data_list.txt")
    training_id_set = set(training_ids)
    for name in training_ids:
        next_name = successor_id(name)
        if int(name.split("_")[0][2]) == 1 and next_name in training_id_set:
            return [name, next_name]
    raise RuntimeError("No labelled one-person training ID has an immediate successor")


def build_fixture(source_root, output_root):
    source_train = source_root / "train_data"
    output_train = output_root / "train_data"
    selected_ids = select_consecutive_one_person_ids(source_train)

    for name in selected_ids:
        raw_path = source_train / "csi" / f"{name}.mat"
        keypoint_path = source_train / "keypoint" / f"{name}.npy"
        if not raw_path.is_file():
            raise FileNotFoundError(f"Missing official training CSI: {raw_path}")
        if not keypoint_path.is_file():
            raise FileNotFoundError(f"Missing official training keypoints: {keypoint_path}")
        keypoints = np.load(keypoint_path, allow_pickle=False)
        if keypoints.shape != (1, 14, 3):
            raise ValueError(f"Unexpected keypoint shape for {name}: {keypoints.shape}")
        if not np.isfinite(keypoints).all():
            raise ValueError(f"Non-finite keypoints for {name}")

    output_train.mkdir(parents=True, exist_ok=True)
    (output_train / "csi").mkdir(exist_ok=True)
    (output_train / "keypoint").mkdir(exist_ok=True)
    with (output_train / "train_data_list.txt").open("w", encoding="utf-8") as list_file:
        list_file.write("\n".join(selected_ids) + "\n")
    for name in selected_ids:
        shutil.copyfile(
            source_train / "csi" / f"{name}.mat",
            output_train / "csi" / f"{name}.mat",
        )
        shutil.copyfile(
            source_train / "keypoint" / f"{name}.npy",
            output_train / "keypoint" / f"{name}.npy",
        )

    results = process_ids(selected_ids, output_root, split="train_data")
    if list(results) != selected_ids:
        raise AssertionError("The bounded processor did not preserve the explicit ID order")

    for name in selected_ids:
        amplitude, phase, combined = results[name]
        if amplitude.shape != (3, 90, 20):
            raise AssertionError(f"Unexpected amplitude shape for {name}: {amplitude.shape}")
        if phase.shape != (3, 90, 20):
            raise AssertionError(f"Unexpected phase shape for {name}: {phase.shape}")
        if combined.shape != (3, 180, 20):
            raise AssertionError(f"Unexpected csi_ap shape for {name}: {combined.shape}")
        if not all(np.isfinite(array).all() for array in (amplitude, phase, combined)):
            raise AssertionError(f"Non-finite processed CSI for {name}")
        np.testing.assert_array_equal(
            combined, np.concatenate((amplitude, phase), axis=1)
        )
        for directory, expected in (
            ("csi_amplitude", amplitude),
            ("csi_phase", phase),
            ("csi_ap", combined),
        ):
            stored = np.load(
                output_train / directory / f"{name}.npy", allow_pickle=False
            )
            np.testing.assert_array_equal(stored, expected)
        print(
            f"Verified {name}: amplitude={amplitude.shape}, phase={phase.shape}, "
            f"csi_ap={combined.shape}, finite=True, exact_concatenation=True"
        )

    print(f"Selected official training IDs: {', '.join(selected_ids)}")
    print(f"Fixture root: {output_root}")
    return selected_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("data/person_in_wifi_3d/wifi_dataset"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/_fixtures/piw3d_two_frame"),
    )
    args = parser.parse_args()
    build_fixture(args.source_root, args.output_root)


if __name__ == "__main__":
    main()
