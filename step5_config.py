"""Frozen Step-5 configuration (docs/NEXT_PHASE_PLAN.md section 4, docs/STEP_5_PROTOCOL.md).

Every path is relative to a repository root that is supplied on the command line
(``--repo-root``), so the code can live in a worktree while data and artifacts stay in
the main checkout. Nothing in this file may be changed after the protocol freeze; the
runner refuses to run when the frozen JSON copy or the recorded source hashes differ.
"""

import json
from pathlib import Path


STEP5_VERSION = "compwifi-step5-v1"

RUN_IDS = ("R1", "R2", "R3", "R4")
SIMPLICITY_ORDER = ("R1", "R2", "R3", "R4")

STEP5_CONFIG = {
    "algorithm_version": STEP5_VERSION,
    "purpose": (
        "bounded single-person baseline that must provably learn (or fail with a final "
        "diagnosis); Phase B of docs/NEXT_PHASE_PLAN.md"
    ),
    "paths": {
        "train_manifest": "data/_artifacts/piw3d_candidate_group_manifests/internal_train_ids.txt",
        "development_manifest": (
            "data/_artifacts/piw3d_candidate_group_manifests/internal_development_ids.txt"
        ),
        "source_root": "data/person_in_wifi_3d/wifi_dataset",
        "freeze_dir": "data/_artifacts/step5_single_person/freeze_v1",
        "data_root": "data/_artifacts/step5_single_person/data_v1",
        "artifact_root": "data/_artifacts/step5_single_person",
        "protocol_freeze_dir": "data/_artifacts/step5_single_person/protocol_freeze",
    },
    "subset": {
        "selection_rule": (
            "Per single-person candidate group, frames sorted by frame index (n = count). "
            "Temporal-holdout block = sorted positions [n//5, n//5 + h) with h = min(49, n//10); "
            "guard = 5 positions on each side of the block, excluded from every split; "
            "train = up to 64 evenly spaced frames from the remaining positions; "
            "temporal_holdout = up to 16 evenly spaced frames from the block. "
            "Every single-person internal-development group: group_dev = up to 64 evenly "
            "spaced frames. Evenly spaced = step4b_subset.select_evenly_spaced_frames "
            "(np.linspace over sorted positions, integer truncation). No randomness."
        ),
        "holdout_start_divisor": 5,
        "holdout_length_cap": 49,
        "holdout_length_divisor": 10,
        "guard_positions": 5,
        "train_frames_per_group": 64,
        "temporal_holdout_frames_per_group": 16,
        "group_dev_frames_per_group": 64,
        "expected_train_groups": 51,
        "expected_development_groups": 8,
        "split_names": ["train", "temporal_holdout", "group_dev"],
        "split_labels": {
            "train": "internal-train single-person groups, frames outside holdout block and guards",
            "temporal_holdout": (
                "same-group temporal holdout — NOT group-disjoint (official-like)"
            ),
            "group_dev": (
                "internal-development single-person groups — candidate-filename-group disjoint "
                "from train"
            ),
        },
        "official_test_used": False,
        "multi_person_samples_used": False,
    },
    "input_norm": {
        "options": ["legacy_minmax", "train_zscore"],
        "legacy_minmax": (
            "identical to SinglePersonPiW3DDataset.read_csi: per-sample min-max over the whole "
            "amplitude block and the whole phase block"
        ),
        "train_zscore": (
            "(x - mean) / max(std, 1e-6); mean/std of shape (3,180,1) over train-split samples "
            "and the 20 time steps, computed once from the stored csi_ap of the train split"
        ),
        "std_floor": 1e-6,
        "stats_file": "train_input_stats_v1.npz",
    },
    "target_codec": {
        "stats_file": "pose_codec_stats_v1.json",
        "definition": (
            "centroid c = mean of 14 joints; r = pose - c; mu_c, sigma_c per axis over train; "
            "mu_r (14x3) over train; scalar sigma_r = std of (r - mu_r) over all train "
            "samples/joints/axes. Output o in R^45: c_hat = mu_c + sigma_c*o[:3], "
            "r_hat = mu_r + sigma_r*o[3:], pose = c_hat + r_hat. Last linear layer zero-init."
        ),
        "loss": "mean||c_hat-c||_2 / mean(sigma_c) + mean_j||r_hat_j-r_j||_2 / sigma_r",
        "legacy_loss": "torch.mean(torch.norm(pred - target, dim=-1)) on absolute stored coordinates",
    },
    "models": {
        "simple_cnn": {
            "convs": [
                {"in": 3, "out": 32, "kernel": [5, 3], "stride": [2, 1], "padding": [2, 1]},
                {"in": 32, "out": 64, "kernel": [3, 3], "stride": [2, 1], "padding": [1, 1]},
                {"in": 64, "out": 128, "kernel": [3, 3], "stride": [2, 2], "padding": [1, 1]},
                {"in": 128, "out": 256, "kernel": [3, 3], "stride": [2, 2], "padding": [1, 1]},
            ],
            "norm": "GroupNorm(8) + GELU after every conv",
            "tokens": "flattened final map (12x5 = 60 tokens of 256)",
            "pooled": "mean over tokens",
            "head": "Linear(256,256)-GELU-Linear(256,45, zero-init)",
            "target": "codec",
        },
        "dtpose_encoder_mlp": {
            "encoder": "MAE_ViT(...).encoder as in step4b_r_baseline.build_step4b_r_model",
            "image_size": [180, 20],
            "patch_size": [2, 2],
            "encoder_layer": 4,
            "encoder_head": 4,
            "decoder_layer": 2,
            "decoder_head": 4,
            "emb_dim": 256,
            "input_dim": 3,
            "forward": "unmasked: patchify + pos-embedding + cls -> transformer -> layer_norm",
            "tokens": "900 patch tokens (cls dropped)",
            "pooled": "mean of patch tokens",
            "head": "Linear(256,256)-GELU-Linear(256,45, zero-init)",
            "all_parameters_trainable": True,
            "target": "codec",
        },
        "dtpose_legacy_decoder": {
            "model": "unmodified ViT_Pose_Decoder via step4b_r_baseline.build_step4b_r_model",
            "train_encoder": True,
            "target": "absolute",
        },
    },
    "runs": {
        "R1": {"model": "simple_cnn", "input_norm": "train_zscore", "isolates": "reference"},
        "R2": {"model": "simple_cnn", "input_norm": "legacy_minmax", "isolates": "input normalization"},
        "R3": {
            "model": "dtpose_encoder_mlp",
            "input_norm": "train_zscore",
            "isolates": "DT-Pose encoder with a sane head",
        },
        "R4": {
            "model": "dtpose_legacy_decoder",
            "input_norm": "legacy_minmax",
            "isolates": "legacy decoder under stabilized optimization",
        },
    },
    "training": {
        "optimizer": "AdamW",
        "peak_learning_rate": 3e-4,
        "weight_decay": 0.01,
        "weight_decay_scope": "uniform over all trainable parameters",
        "warmup_steps": 200,
        "schedule": "linear warmup from 0 over 200 steps, then cosine to 0 at the final step",
        "grad_clip_norm": 1.0,
        "batch_size": 32,
        "eval_batch_size": 64,
        "epochs": 40,
        "num_workers": 0,
        "drop_last": False,
        "shuffle": "torch DataLoader shuffle with a torch.Generator seeded with the run seed",
        "early_stopping": False,
        "checkpoint_selection": "final epoch only",
        "first_epoch_budget_hours": 4.0,
    },
    "overfit": {
        "runs": ["R1", "R3"],
        "sample_count": 64,
        "sample_rule": "train IDs (sorted by group, frame) taken at every k-th position, k = len(train)//64, first 64",
        "steps": 300,
        "batch": "all 64 samples every step, fixed order",
        "learning_rate": 1e-3,
        "learning_rate_schedule": "constant",
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "seed": 0,
        "pass_ratio": 0.20,
        "reference": "MPJPE of the mean pose of the same 64 samples (eval mode prediction)",
    },
    "seeds": {"first_pass": [0], "selected_run_additional": [1, 2]},
    "evaluation": {
        "splits": ["train", "temporal_holdout", "group_dev"],
        "distance_units": "stored units",
        "mm_conversion": "stored_unit_to_mm=1000, provisional, assumes stored unit = metre",
        "translation_reference": "14-joint centroid (root joint identity unproven)",
        "centroid_relative_mpjpe": "MPJPE after subtracting each skeleton's own 14-joint centroid",
        "collapse_monitors": [
            "prediction std across samples (mean over joints and axes)",
            "prediction std across joints (mean over samples and axes)",
        ],
        "shuffled_pairing_seed": 20261110,
        "research_metrics": {
            "stored_unit_to_mm": 1000.0,
            "presence_threshold": 0.5,
            "match_threshold_mm": 500.0,
            "pck_threshold": 0.5,
            "pck_scale_joint_indices": [6, 4],
            "root_index": 0,
            "people_dim": 1,
        },
    },
    "gates": {
        "G1": "all smoke checks pass for every model",
        "G2": "overfit final MPJPE <= 0.20 x mean-pose MPJPE of the same 64 samples",
        "G3_centroid_std_ratio": 0.25,
        "G3_shuffle_ratio": 0.70,
        "G4_train_ratio": 0.50,
        "G5_holdout_ratio": 0.50,
        "G6_dev_ratio": 0.95,
        "G6_dev_relative_ratio": 0.95,
        "selection_tie_relative": 0.01,
    },
    "determinism": {
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "settings": "step4b_r_baseline.configure_deterministic_runtime (identical block)",
    },
    "label_access": {
        "training_labels": "single-person internal-train labels of the train split only",
        "temporal_holdout_labels": "evaluation only",
        "group_dev_labels": (
            "single-person internal-development labels, evaluation and the predeclared run "
            "selection only (allowed under development and strict protocol)"
        ),
        "codec_and_input_stats_source": "train split only",
        "official_test_used": False,
        "multi_person_labels_used": False,
    },
}

STEP5_SOURCE_FILES = (
    "step5_config.py",
    "step5_subset.py",
    "make_step5_data.py",
    "feeder/piw3d_step5.py",
    "model/research_models.py",
    "step5_single_person_baseline.py",
)


def step5_config_for_json():
    return json.loads(json.dumps(STEP5_CONFIG))


def validate_step5_config(config):
    if config != step5_config_for_json():
        raise ValueError("Frozen Step-5 configuration does not match step5_config.py")
    return config


def resolve_path(repo_root, key):
    return Path(repo_root) / STEP5_CONFIG["paths"][key]
