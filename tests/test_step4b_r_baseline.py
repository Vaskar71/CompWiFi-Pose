import json

import numpy as np
import pytest
import torch

from feeder.piw3d_single_person import SinglePersonPiW3DDataset, make_single_person_dataloader
from step4b_config import step4b_config_for_json
from step4b_r_baseline import (
    _gradient_statistics,
    _mean_per_joint_error,
    build_step4b_r_model,
    constant_pose_mpjpe,
    evaluate_train_mode_mpjpe,
    load_runtime_modules,
    make_adamw_optimizer,
    parameter_audit,
    run_baseline,
    validate_process_environment,
)
from step4b_r_config import step4b_r_config_for_json, validate_step4b_r_config


TRAIN_IDS = ["S11_01_10", "S11_01_12", "S11_01_14", "S11_01_16"]
DEVELOPMENT_IDS = ["S11_21_10", "S11_21_12"]


@pytest.fixture(scope="module", autouse=True)
def runtime_modules():
    load_runtime_modules()


def make_root(tmp_path):
    train_dir = tmp_path / "train_data"
    for directory in ("csi_ap", "keypoint"):
        (train_dir / directory).mkdir(parents=True, exist_ok=True)
    selected_ids = TRAIN_IDS + DEVELOPMENT_IDS
    (train_dir / "train_data_list.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in selected_ids), encoding="utf-8"
    )
    (train_dir / "selected_train_ids.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in TRAIN_IDS), encoding="utf-8"
    )
    (train_dir / "selected_development_ids.txt").write_text(
        "".join(f"{sample_id}\n" for sample_id in DEVELOPMENT_IDS), encoding="utf-8"
    )
    assignments = {sample_id: "internal_train" for sample_id in TRAIN_IDS}
    assignments.update({sample_id: "internal_development" for sample_id in DEVELOPMENT_IDS})
    split_path = tmp_path / "split_assignments.json"
    split_path.write_text(json.dumps(assignments), encoding="utf-8")
    for index, sample_id in enumerate(selected_ids):
        np.save(train_dir / "csi_ap" / f"{sample_id}.npy", np.zeros((3, 180, 20), dtype=np.float32))
        offset = index * 0.1
        joints = np.arange(14, dtype=np.float32).reshape(14, 1)
        pose = np.concatenate((joints, joints * 0.1, joints * 0.01), axis=1) + offset
        np.save(train_dir / "keypoint" / f"{sample_id}.npy", pose[None, :, :])
    return tmp_path, split_path


def make_dataset(tmp_path, split):
    root, split_path = make_root(tmp_path)
    filename = (
        "selected_train_ids.txt" if split == "internal_train" else "selected_development_ids.txt"
    )
    return SinglePersonPiW3DDataset(
        root,
        root / "train_data" / filename,
        expected_split=split,
        split_assignments_path=split_path,
    )


def test_step4b_r_config_is_separate_and_frozen():
    config = step4b_r_config_for_json()
    validate_step4b_r_config(config)
    source_config = step4b_config_for_json()

    for key, value in source_config["model"].items():
        assert config["model"][key] == value
    assert config["algorithm_version"] == "compwifi-step4b-r-v1"
    assert config["source_algorithm_version"] == source_config["algorithm_version"]
    assert config["model"]["train_encoder"] is True
    assert config["training"]["optimizer"] == "AdamW"
    assert config["training"]["learning_rate"] == 0.001
    assert config["training"]["weight_decay"] == 0.0
    assert config["training"]["batch_size"] == 8
    assert config["training"]["development_batch_size"] == 8
    assert config["training"]["max_epochs"] == 10
    assert config["training"]["early_stopping"] is False
    assert config["training"]["learning_rate_tuning"] is False
    assert config["training"]["checkpoint_selection"].startswith("final epoch only")
    assert config["determinism"]["PYTHONHASHSEED"] == "0"
    assert config["determinism"]["CUBLAS_WORKSPACE_CONFIG"] == ":4096:8"
    assert config["determinism"]["torch_deterministic_algorithms"] is True
    assert config["measurements"]["distance_units"] == "stored units"
    assert config["measurements"]["convert_to_millimetres"] is False
    assert config["label_access"]["development_labels_in_training_or_loss"] is False
    assert config["label_access"]["mean_pose_uses_training_labels_only"] is True

    changed = json.loads(json.dumps(config))
    changed["training"]["learning_rate"] = 0.002
    with pytest.raises(ValueError, match="does not match step4b_r_config.py"):
        validate_step4b_r_config(changed)


def test_step4b_r_requires_pre_process_deterministic_environment(monkeypatch):
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    valid = validate_process_environment()
    assert valid["valid"] is True
    assert valid["failures"] == []

    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    invalid = validate_process_environment()
    assert invalid["valid"] is False
    assert any("PYTHONHASHSEED" in failure for failure in invalid["failures"])


def test_step4b_r_refuses_to_run_without_pre_process_hash_seed(monkeypatch, tmp_path):
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    with pytest.raises(RuntimeError, match="PYTHONHASHSEED"):
        run_baseline(tmp_path, None, None, "cpu")


def test_step4b_r_adamw_optimizer_covers_trainable_encoder_and_updates_once():
    config = step4b_r_config_for_json()
    torch.manual_seed(config["seed"])
    device = torch.device("cpu")
    model = build_step4b_r_model(config, device)
    model.train()
    optimizer = make_adamw_optimizer(model, config)
    audit = parameter_audit(model, optimizer)

    assert audit["optimizer"]["contains_exact_trainable_parameter_set"] is True
    assert audit["optimizer"]["parameter_references"] == 97
    assert audit["optimizer"]["unique_parameter_references"] == 97
    assert audit["parameter_counts"]["trainable_named_parameters"] == 97
    assert audit["module_counts"]["encoder"]["trainable_named_parameters"] == 50
    assert audit["module_counts"]["pose_decoder"]["trainable_named_parameters"] == 47
    assert optimizer.param_groups[0]["lr"] == 0.001
    assert optimizer.param_groups[0]["weight_decay"] == 0.0

    csi = torch.randn(2, 3, 180, 20)
    pose = torch.randn(2, 1, 14, 3)
    initial_values = [parameter.detach().clone() for parameter in model.parameters()]
    predicted, _ = model(csi)
    assert predicted.shape == pose.shape
    loss = _mean_per_joint_error(predicted, pose)
    assert torch.isfinite(loss)
    loss.backward()
    gradient_stats = _gradient_statistics(model)
    assert gradient_stats["all_gradients_finite"] is True
    assert gradient_stats["gradient_tensors"] == 97
    optimizer.step()

    changed = [
        not torch.equal(parameter.detach(), initial)
        for parameter, initial in zip(model.parameters(), initial_values)
    ]
    assert all(changed)


def test_step4b_r_train_mode_evaluation_restores_rng(tmp_path):
    config = step4b_r_config_for_json()
    dataset = make_dataset(tmp_path, "internal_train")
    loader = make_single_person_dataloader(
        dataset,
        is_training=False,
        generator=torch.Generator().manual_seed(config["seed"]),
        batch_size=2,
        num_workers=0,
    )
    torch.manual_seed(config["seed"])
    model = build_step4b_r_model(config, torch.device("cpu"))
    model.train()

    first_value, first_count = evaluate_train_mode_mpjpe(model, loader, torch.device("cpu"))
    second_value, second_count = evaluate_train_mode_mpjpe(model, loader, torch.device("cpu"))

    assert first_count == len(dataset)
    assert second_count == len(dataset)
    assert first_value == second_value
    assert np.isfinite(first_value)


def test_step4b_r_mean_pose_uses_training_labels_only(tmp_path):
    train_dataset = make_dataset(tmp_path, "internal_train")
    development_dataset = make_dataset(tmp_path / "development_view", "internal_development")
    train_offsets = np.arange(len(TRAIN_IDS), dtype=np.float64) * 0.1
    base_joints = np.arange(14, dtype=np.float64).reshape(14, 1)
    base_pose = np.concatenate((base_joints, base_joints * 0.1, base_joints * 0.01), axis=1)
    expected_mean = base_pose + train_offsets.mean()

    poses = []
    for sample in train_dataset.samples:
        poses.append(train_dataset.read_pose(sample["keypoint_path"]).numpy()[0])
    actual_mean = np.mean(np.stack(poses, axis=0), axis=0)
    assert np.allclose(actual_mean, expected_mean)

    train_mpjpe, train_count = constant_pose_mpjpe(train_dataset, actual_mean)
    development_mpjpe, development_count = constant_pose_mpjpe(
        development_dataset, actual_mean
    )
    assert train_count == len(TRAIN_IDS)
    assert development_count == len(DEVELOPMENT_IDS)
    assert np.isclose(
        train_mpjpe,
        np.sqrt(3.0) * np.mean(np.abs(train_offsets - train_offsets.mean())),
    )
    assert development_mpjpe > train_mpjpe
    assert np.isfinite(development_mpjpe)
