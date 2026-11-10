"""Step-5 research models (B3). New file; ``model/model.py`` is not modified.

Every model exposes ``forward_features(x) -> (tokens (B,N,D), pooled (B,D))``,
``forward(x) -> pose (B,1,14,3)`` and ``loss(x, target_pose) -> (loss, pose)``.
"""

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from einops import rearrange

from model.model import MAE_ViT


KEYPOINTS = 14
COORDS = 3
OUTPUT_DIM = COORDS + KEYPOINTS * COORDS
MODEL_NAMES = ("simple_cnn", "dtpose_encoder_mlp", "dtpose_legacy_decoder")


# ----------------------------------------------------------------------------- codec

def compute_codec_stats(poses):
    """Target statistics from train-split poses of shape (N,1,14,3) or (N,14,3) (float64)."""
    poses = np.asarray(poses, dtype=np.float64)
    if poses.ndim == 4:
        if poses.shape[1] != 1:
            raise ValueError("Codec statistics require single-person poses")
        poses = poses[:, 0]
    if poses.ndim != 3 or poses.shape[1:] != (KEYPOINTS, COORDS):
        raise ValueError(f"Poses must have shape (N,{KEYPOINTS},{COORDS}); got {poses.shape}")
    if not np.isfinite(poses).all():
        raise ValueError("Poses must be finite")
    centroids = poses.mean(axis=1)
    relative = poses - centroids[:, None, :]
    mu_c = centroids.mean(axis=0)
    sigma_c = centroids.std(axis=0)
    mu_r = relative.mean(axis=0)
    sigma_r = float((relative - mu_r[None]).std())
    if np.any(sigma_c <= 0) or sigma_r <= 0:
        raise ValueError("Degenerate codec statistics (zero spread)")
    return {
        "mu_c": mu_c.tolist(),
        "sigma_c": sigma_c.tolist(),
        "mu_r": mu_r.tolist(),
        "sigma_r": sigma_r,
        "sample_count": int(poses.shape[0]),
    }


def save_codec_stats(path, stats, train_id_sha256):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite codec statistics: {path}")
    record = dict(stats)
    record["train_id_sha256"] = train_id_sha256
    record["source"] = "train split labels only"
    path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def load_codec_stats(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


class PoseTargetCodec(nn.Module):
    def __init__(self, stats):
        super().__init__()
        self.register_buffer("mu_c", torch.tensor(stats["mu_c"], dtype=torch.float32))
        self.register_buffer("sigma_c", torch.tensor(stats["sigma_c"], dtype=torch.float32))
        self.register_buffer("mu_r", torch.tensor(stats["mu_r"], dtype=torch.float32))
        self.register_buffer("sigma_r", torch.tensor(float(stats["sigma_r"]), dtype=torch.float32))
        if self.mu_c.shape != (COORDS,) or self.mu_r.shape != (KEYPOINTS, COORDS):
            raise ValueError("Codec statistics have the wrong shape")

    def decode(self, output):
        if output.ndim != 2 or output.shape[1] != OUTPUT_DIM:
            raise ValueError(f"Codec output must be (B,{OUTPUT_DIM})")
        c_hat = self.mu_c + self.sigma_c * output[:, :COORDS]
        r_hat = self.mu_r + self.sigma_r * output[:, COORDS:].reshape(-1, KEYPOINTS, COORDS)
        pose = (c_hat[:, None, :] + r_hat).unsqueeze(1)
        return pose, c_hat, r_hat

    def loss(self, c_hat, r_hat, target_pose):
        target = target_pose[:, 0]
        centroid = target.mean(dim=1)
        relative = target - centroid[:, None, :]
        centroid_term = torch.mean(torch.norm(c_hat - centroid, dim=-1)) / torch.mean(self.sigma_c)
        relative_term = torch.mean(torch.norm(r_hat - relative, dim=-1)) / self.sigma_r
        return centroid_term + relative_term


def _mlp_head(dim):
    head = nn.Sequential(nn.Linear(dim, 256), nn.GELU(), nn.Linear(256, OUTPUT_DIM))
    nn.init.zeros_(head[-1].weight)
    nn.init.zeros_(head[-1].bias)
    return head


class _CodecModel(nn.Module):
    target_mode = "codec"

    def __init__(self, codec_stats):
        super().__init__()
        self.codec = PoseTargetCodec(codec_stats)

    def predict(self, x):
        _, pooled = self.forward_features(x)
        pose, c_hat, r_hat = self.codec.decode(self.head(pooled))
        return pose, c_hat, r_hat

    def forward(self, x):
        return self.predict(x)[0]

    def loss(self, x, target_pose):
        pose, c_hat, r_hat = self.predict(x)
        return self.codec.loss(c_hat, r_hat, target_pose), pose


# ----------------------------------------------------------------------------- models

class SimpleCNNRegressor(_CodecModel):
    def __init__(self, codec_stats, conv_specs):
        super().__init__(codec_stats)
        layers = []
        for spec in conv_specs:
            layers.extend(
                [
                    nn.Conv2d(
                        spec["in"],
                        spec["out"],
                        kernel_size=tuple(spec["kernel"]),
                        stride=tuple(spec["stride"]),
                        padding=tuple(spec["padding"]),
                    ),
                    nn.GroupNorm(8, spec["out"]),
                    nn.GELU(),
                ]
            )
        self.backbone = nn.Sequential(*layers)
        self.emb_dim = conv_specs[-1]["out"]
        self.head = _mlp_head(self.emb_dim)

    def forward_features(self, x):
        feature_map = self.backbone(x)
        tokens = rearrange(feature_map, "b c h w -> b (h w) c")
        return tokens, tokens.mean(dim=1)


def _unmasked_encoder_tokens(x, patchify, pos_embedding, cls_token, transformer, layer_norm):
    patches = patchify(x)
    patches = rearrange(patches, "b c h w -> (h w) b c")
    patches = patches + pos_embedding
    patches = torch.cat([cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
    patches = rearrange(patches, "t b c -> b t c")
    features = layer_norm(transformer(patches))
    tokens = features[:, 1:, :]
    return tokens, tokens.mean(dim=1)


class DTPoseEncoderMLP(_CodecModel):
    def __init__(self, codec_stats, model_config):
        super().__init__(codec_stats)
        mae = MAE_ViT(
            image_size=tuple(model_config["image_size"]),
            patch_size=tuple(model_config["patch_size"]),
            encoder_layer=model_config["encoder_layer"],
            encoder_head=model_config["encoder_head"],
            decoder_layer=model_config["decoder_layer"],
            decoder_head=model_config["decoder_head"],
            emb_dim=model_config["emb_dim"],
            input_dim=model_config["input_dim"],
        )
        self.encoder = mae.encoder
        self.head = _mlp_head(model_config["emb_dim"])

    def forward_features(self, x):
        encoder = self.encoder
        return _unmasked_encoder_tokens(
            x, encoder.patchify, encoder.pos_embedding, encoder.cls_token, encoder.transformer, encoder.layer_norm
        )


class LegacyDecoderPoseModel(nn.Module):
    """Unmodified ViT_Pose_Decoder; absolute targets and the Step-4B-R loss."""

    target_mode = "absolute"

    def __init__(self, legacy_decoder):
        super().__init__()
        self.legacy = legacy_decoder

    def forward_features(self, x):
        legacy = self.legacy
        return _unmasked_encoder_tokens(
            x, legacy.patchify, legacy.pos_embedding, legacy.cls_token, legacy.transformer, legacy.layer_norm
        )

    def forward(self, x):
        pose, _ = self.legacy(x)
        return pose

    def loss(self, x, target_pose):
        pose = self.forward(x)
        return torch.mean(torch.norm(pose - target_pose, dim=-1)), pose


def build_research_model(model_name, codec_stats, config, device):
    """Build a Step-5 model from the frozen config. Seed the RNGs before calling."""
    if model_name == "simple_cnn":
        model = SimpleCNNRegressor(codec_stats, config["models"]["simple_cnn"]["convs"])
    elif model_name == "dtpose_encoder_mlp":
        model = DTPoseEncoderMLP(codec_stats, config["models"]["dtpose_encoder_mlp"])
    elif model_name == "dtpose_legacy_decoder":
        import step4b_r_baseline
        from step4b_r_config import step4b_r_config_for_json

        step4b_r_baseline.load_runtime_modules()
        legacy = step4b_r_baseline.build_step4b_r_model(step4b_r_config_for_json(), torch.device("cpu"))
        model = LegacyDecoderPoseModel(legacy)
    else:
        raise ValueError(f"Unknown Step-5 model: {model_name!r}")
    return model.to(device)
