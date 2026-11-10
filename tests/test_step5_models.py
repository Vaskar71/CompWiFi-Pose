import numpy as np
import pytest
import torch

from model.research_models import (
    PoseTargetCodec,
    build_research_model,
    compute_codec_stats,
)
from step5_config import step5_config_for_json


CONFIG = step5_config_for_json()


def synthetic_poses(count=20, seed=0):
    rng = np.random.default_rng(seed)
    skeleton = rng.normal(scale=0.3, size=(14, 3))
    centroids = rng.normal(loc=[2.0, 1.5, 3.0], scale=0.5, size=(count, 3))
    noise = rng.normal(scale=0.05, size=(count, 14, 3))
    return (centroids[:, None, :] + skeleton[None] + noise)[:, None]


def test_codec_stats_definition():
    poses = synthetic_poses()
    stats = compute_codec_stats(poses)
    flat = poses[:, 0]
    centroids = flat.mean(axis=1)
    relative = flat - centroids[:, None]
    np.testing.assert_allclose(stats["mu_c"], centroids.mean(axis=0))
    np.testing.assert_allclose(stats["sigma_c"], centroids.std(axis=0))
    np.testing.assert_allclose(stats["mu_r"], relative.mean(axis=0))
    assert stats["sigma_r"] == pytest.approx(float((relative - relative.mean(axis=0)).std()))
    np.testing.assert_allclose(np.mean(stats["mu_r"], axis=0), 0.0, atol=1e-12)


def test_codec_stats_rejects_multi_person():
    with pytest.raises(ValueError):
        compute_codec_stats(np.zeros((4, 2, 14, 3)))


def test_codec_zero_output_is_train_mean_pose_and_exact_target_has_zero_loss():
    poses = synthetic_poses()
    codec = PoseTargetCodec(compute_codec_stats(poses))
    pose, c_hat, r_hat = codec.decode(torch.zeros(3, 45))
    mean_pose = torch.tensor(poses.mean(axis=0), dtype=torch.float32)
    torch.testing.assert_close(pose, mean_pose.expand(3, 1, 14, 3), atol=1e-5, rtol=0)
    target = torch.tensor(poses[:2], dtype=torch.float32)
    centroid = target[:, 0].mean(dim=1)
    loss = codec.loss(centroid, target[:, 0] - centroid[:, None], target)
    assert float(loss) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.parametrize(
    "model_name,token_count",
    [("simple_cnn", 60), ("dtpose_encoder_mlp", 900), ("dtpose_legacy_decoder", 900)],
)
def test_model_interface_shapes_and_backward(model_name, token_count):
    torch.manual_seed(0)
    poses = synthetic_poses()
    model = build_research_model(model_name, compute_codec_stats(poses), CONFIG, torch.device("cpu"))
    x = torch.randn(2, 3, 180, 20)
    tokens, pooled = model.forward_features(x)
    assert tokens.shape == (2, token_count, 256)
    assert pooled.shape == (2, 256)
    pose = model(x)
    assert pose.shape == (2, 1, 14, 3)
    target = torch.tensor(poses[:2], dtype=torch.float32)
    loss, predicted = model.loss(x, target)
    assert torch.isfinite(loss)
    assert predicted.shape == (2, 1, 14, 3)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert all(p.requires_grad for p in model.parameters())


@pytest.mark.parametrize("model_name", ["simple_cnn", "dtpose_encoder_mlp"])
def test_untrained_codec_model_outputs_exact_train_mean_pose(model_name):
    torch.manual_seed(1)
    poses = synthetic_poses()
    model = build_research_model(model_name, compute_codec_stats(poses), CONFIG, torch.device("cpu"))
    model.eval()
    with torch.no_grad():
        pose = model(torch.randn(4, 3, 180, 20))
    mean_pose = torch.tensor(poses.mean(axis=0), dtype=torch.float32).expand(4, 1, 14, 3)
    torch.testing.assert_close(pose, mean_pose, atol=1e-5, rtol=0)
    assert model.target_mode == "codec"


def test_legacy_model_uses_absolute_step4b_r_loss():
    torch.manual_seed(0)
    model = build_research_model(
        "dtpose_legacy_decoder", compute_codec_stats(synthetic_poses()), CONFIG, torch.device("cpu")
    )
    model.eval()
    x = torch.randn(2, 3, 180, 20)
    target = torch.randn(2, 1, 14, 3)
    with torch.no_grad():
        loss, pose = model.loss(x, target)
    assert model.target_mode == "absolute"
    torch.testing.assert_close(loss, torch.mean(torch.norm(pose - target, dim=-1)))
