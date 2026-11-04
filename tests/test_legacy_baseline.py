import os
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
PIW3D_CONFIG_DIR = REPO_ROOT / "config" / "person_in_wifi_3d"


def make_mae(image_size=(8, 4), mask_ratio=0.75):
    from model.model import MAE_ViT

    return MAE_ViT(
        image_size=image_size,
        patch_size=(2, 2),
        encoder_layer=1,
        encoder_head=4,
        decoder_layer=1,
        decoder_head=4,
        emb_dim=32,
        input_dim=3,
        mask_ratio=mask_ratio,
    )


def make_pose_decoder():
    from model.model import ViT_Pose_Decoder

    mae = make_mae(image_size=(180, 20))
    return ViT_Pose_Decoder(
        mae.encoder,
        keypoints=14,
        coor_num=3,
        token_num=900,
        dataset="person-in-wifi-3d",
        num_person=1,
    )


def test_mae_forward_backward_on_synthetic_input():
    torch.manual_seed(0)
    np.random.seed(0)
    model = make_mae()
    model.train()
    csi = torch.randn(2, 3, 8, 4)

    predicted, mask, features, cl_feature = model(csi, "train")

    assert predicted.shape == csi.shape
    assert mask.shape == csi.shape
    assert features.shape == (2, 32)
    assert cl_feature.shape == (2, 32)
    loss = (
        torch.mean((predicted - csi) ** 2 * mask) / 0.75
        + features.square().mean()
        + cl_feature.square().mean()
    )
    assert torch.isfinite(loss)

    loss.backward()

    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_pose_decoder_output_shape_and_finite():
    torch.manual_seed(0)
    model = make_pose_decoder()
    csi = torch.randn(2, 3, 180, 20)

    pose, features = model(csi)

    assert pose.shape == (2, 1, 14, 3)
    assert features.shape == (2, 32)
    assert torch.isfinite(pose).all()
    assert torch.isfinite(features).all()


def test_pose_decoder_one_optimization_step():
    torch.manual_seed(0)
    model = make_pose_decoder()
    optimizer = torch.optim.SGD(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=1e-2,
        momentum=0.9,
    )
    trainable_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    csi = torch.randn(2, 3, 180, 20)

    optimizer.zero_grad()
    pose, _ = model(csi)
    loss = torch.mean(torch.norm(pose - torch.zeros_like(pose), dim=-1))
    assert torch.isfinite(loss)
    loss.backward()
    optimizer.step()

    assert trainable_before
    assert any(
        not torch.equal(trainable_before[name], parameter.detach())
        for name, parameter in model.named_parameters()
        if name in trainable_before
    )


def test_checkpoint_save_and_load(tmp_path):
    from model.model import ViT_Pose_Decoder

    torch.manual_seed(0)
    model = make_mae(image_size=(180, 20))
    checkpoint_path = tmp_path / "pretrain_checkpoint.pt"

    torch.save(model, checkpoint_path)
    restored = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    decoder = ViT_Pose_Decoder(
        restored.encoder,
        keypoints=14,
        coor_num=3,
        token_num=900,
        dataset="person-in-wifi-3d",
        num_person=1,
    )

    assert torch.equal(
        model.encoder.patchify.weight.detach(),
        restored.encoder.patchify.weight.detach(),
    )
    with torch.no_grad():
        pose, features = decoder(torch.randn(2, 3, 180, 20))
    assert pose.shape == (2, 1, 14, 3)
    assert torch.isfinite(pose).all()
    assert torch.isfinite(features).all()


def test_constant_csi_normalization(tmp_path):
    from feeder.person_in_wifi_3d import PersonInWif3D

    csi = np.empty((3, 180, 20), dtype=np.float32)
    csi[:, :90, :] = 3.25
    phase = np.linspace(-1.0, 1.0, 90 * 20, dtype=np.float32).reshape(90, 20)
    csi[:, 90:, :] = phase[None, :, :]
    csi_path = tmp_path / "constant_amplitude.npy"
    np.save(csi_path, csi)
    dataset = object.__new__(PersonInWif3D)

    normalized = dataset.read_frame(csi_path)

    assert normalized.shape == csi.shape
    assert np.isfinite(normalized).all()
    assert np.count_nonzero(normalized[:, :90, :]) == 0
    expected_phase = (phase - phase.min()) / (phase.max() - phase.min())
    np.testing.assert_allclose(
        normalized[:, 90:, :],
        np.broadcast_to(expected_phase[None, :, :], (3, 90, 20)),
        rtol=0,
        atol=1e-7,
    )


def test_pretrained_checkpoint_path_is_consistent():
    with (PIW3D_CONFIG_DIR / "pretrain_config.yaml").open(
        encoding="utf-8"
    ) as config_file:
        pretrain_config = yaml.safe_load(config_file)
    with (PIW3D_CONFIG_DIR / "pose_config.yaml").open(encoding="utf-8") as config_file:
        pose_config = yaml.safe_load(config_file)

    expected_path = (
        Path(pretrain_config["save_path"])
        / pretrain_config["dataset_name"]
        / pretrain_config["experiment_name"]
        / f"pretrain_{pretrain_config['model_name']}.pt"
    )

    assert Path(pose_config["pretrained_model_path"]) == expected_path


def test_real_training_sample_preprocessing_and_load():
    with (PIW3D_CONFIG_DIR / "pose_config.yaml").open(encoding="utf-8") as config_file:
        configured_root = yaml.safe_load(config_file)["dataset_root"]
    data_root = Path(os.environ.get("PIW3D_DATA_ROOT", configured_root))
    if not data_root.is_absolute():
        data_root = REPO_ROOT / data_root
    train_root = data_root / "train_data"
    if not (train_root / "train_data_list.txt").is_file():
        pytest.skip("Person-in-WiFi-3D training data is not available")

    from feeder.person_in_wifi_3d import PersonInWif3D, piw3_make_dataloader

    dataset = PersonInWif3D("training", str(data_root), "one-person")
    if not dataset.data_list:
        pytest.skip("No eligible single-person training pair is available")
    item = dataset.data_list[0]
    csi_path = Path(item["csi_path"])
    keypoint_path = Path(item["keypoint_path"])
    assert csi_path.is_file(), f"Missing training csi_ap file: {csi_path}"
    assert keypoint_path.is_file(), f"Missing training keypoint file: {keypoint_path}"

    stored_csi = np.load(csi_path, allow_pickle=False)
    assert stored_csi.shape == (3, 180, 20)
    amplitude_path = train_root / "csi_amplitude" / csi_path.name
    phase_path = train_root / "csi_phase" / csi_path.name
    if amplitude_path.is_file() and phase_path.is_file():
        amplitude = np.load(amplitude_path, allow_pickle=False)
        phase = np.load(phase_path, allow_pickle=False)
        np.testing.assert_array_equal(
            stored_csi, np.concatenate((amplitude, phase), axis=1)
        )

    normalized = dataset.read_frame(csi_path)
    pose = np.load(keypoint_path, allow_pickle=False)

    assert normalized.shape == (3, 180, 20)
    assert np.isfinite(normalized).all()
    assert pose.shape == (1, 14, 3)
    assert np.isfinite(pose).all()

    loader = piw3_make_dataloader(
        dataset,
        is_training=False,
        generator=torch.manual_seed(0),
        batch_size=1,
        num_workers=0,
    )
    batch = next(iter(loader))
    assert loader.num_workers == 0
    assert batch["input_wifi-csi"].shape == (1, 3, 180, 20)
    assert batch["input_wifi-csi_next_frame"].shape == (1, 3, 180, 20)
    assert batch["output"].shape == (1, 3, 14, 3)
    assert batch["person_num"].tolist() == [1]
    assert torch.isfinite(batch["input_wifi-csi"]).all()
    assert torch.isfinite(batch["output"]).all()
