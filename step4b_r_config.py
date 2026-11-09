import json


STEP4B_R_CONFIG = {
    "algorithm_version": "compwifi-step4b-r-v1",
    "baseline_name": "step4b-r-deterministic-single-person-real-subset-baseline",
    "source_algorithm_version": "compwifi-step4b-v1",
    "purpose": (
        "bounded deterministic single-person real-data baseline with internal-development "
        "evaluation and a training-derived mean-pose comparison; not a Step-4B rerun"
    ),
    "seed": 0,
    "data": {
        "freeze_config": "data/_artifacts/step4b_pilot/freeze_v2/step4b_freeze_config.json",
        "pilot_root": "data/_artifacts/step4b_pilot/pilot_data_v2",
        "preprocessing_validation": "data/_artifacts/step4b_pilot/preprocessing_validation_v2.json",
        "train_ids": "data/_artifacts/step4b_pilot/pilot_data_v2/train_data/selected_train_ids.txt",
        "development_ids": "data/_artifacts/step4b_pilot/pilot_data_v2/train_data/selected_development_ids.txt",
        "expected_train_count": 2048,
        "expected_development_count": 512,
        "official_test_used": False,
        "multi_person_samples_used": False,
        "unlabelled_or_compositional_training_used": False,
    },
    "model": {
        "family": "MAE_ViT encoder + ViT_Pose_Decoder",
        "q": 1,
        "image_size": [180, 20],
        "patch_size": [2, 2],
        "encoder_layer": 4,
        "encoder_head": 4,
        "decoder_layer": 2,
        "decoder_head": 4,
        "emb_dim": 256,
        "input_dim": 3,
        "keypoints": 14,
        "coordinate_dim": 3,
        "dataset": "person-in-wifi-3d",
        "train_encoder": True,
        "train_from_scratch": True,
        "external_checkpoint_used": False,
        "target_representation": "stored (1,14,3) coordinates; order unchanged",
        "checkpoint_format": "state_dict",
    },
    "training": {
        "optimizer": "AdamW",
        "learning_rate": 0.001,
        "weight_decay": 0.0,
        "max_epochs": 10,
        "batch_size": 8,
        "development_batch_size": 8,
        "num_workers": 0,
        "loss": "mean Euclidean pose error over people, joints, and coordinate triples",
        "stored_units_only": True,
        "early_stopping": False,
        "learning_rate_tuning": False,
        "architecture_search": False,
        "checkpoint_selection": "final epoch only; no development-checkpoint selection",
        "mean_pose_source": "internal-training labels only",
    },
    "determinism": {
        "PYTHONHASHSEED": "0",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "python_seed": 0,
        "numpy_seed": 0,
        "torch_seed": 0,
        "cuda_seed_all": 0,
        "torch_deterministic_algorithms": True,
        "torch_deterministic_warn_only": False,
        "cudnn_deterministic": True,
        "cudnn_benchmark": False,
        "cuda_matmul_allow_tf32": False,
        "cudnn_allow_tf32": False,
        "float32_matmul_precision": "highest",
        "loader_generator_seed": 0,
        "stop_if_deterministic_support_missing": True,
    },
    "measurements": {
        "distance_units": "stored units",
        "convert_to_millimetres": False,
        "make_axis_or_unit_claims": False,
        "per_epoch": [
            "finite training losses",
            "finite gradients",
            "optimizer coverage",
            "parameter updates",
            "train-mode training MPJPE",
            "no-gradient eval-mode development MPJPE",
        ],
        "final": [
            "checkpoint save/reload verification",
            "training-derived constant mean pose",
            "mean-pose training MPJPE",
            "mean-pose development MPJPE",
            "final-model versus mean-pose comparison on training data",
            "final-model versus mean-pose comparison on development data",
            "runtime",
        ],
    },
    "label_access": {
        "training_labels": "corrected frozen internal-train single-person labels only",
        "development_labels": "corrected frozen internal-development single-person labels for evaluation only",
        "development_labels_in_training_or_loss": False,
        "development_labels_in_checkpoint_or_hyperparameter_selection": False,
        "mean_pose_uses_training_labels_only": True,
        "official_test_used": False,
        "multi_person_labels_used": False,
    },
}


def step4b_r_config_for_json():
    return json.loads(json.dumps(STEP4B_R_CONFIG))


def validate_step4b_r_config(config):
    expected = step4b_r_config_for_json()
    if config != expected:
        raise ValueError("Frozen Step-4B-R configuration does not match step4b_r_config.py")
    return config
