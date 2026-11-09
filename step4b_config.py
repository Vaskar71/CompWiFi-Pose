import json


STEP4B_CONFIG = {
    "algorithm_version": "compwifi-step4b-v1",
    "seed": 0,
    "subset": {
        "train_group_count": 8,
        "development_group_count": 4,
        "train_max_frames_per_group": 256,
        "development_max_frames_per_group": 128,
        "selection_rule": (
            "Within each existing internal manifest, retain candidate groups having at "
            "least one filename-marked one-person ID. Sort eligible group names "
            "lexicographically, shuffle that list independently per partition with "
            "random.Random(0), and select the first 8 internal-train and first 4 "
            "internal-development groups. Within each selected group, sort one-person "
            "IDs by numeric frame and retain at most 256 train or 128 development "
            "frames at zero-based linspace positions. Frame caps are resource limits."
        ),
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
        "train_from_scratch": True,
        "checkpoint_format": "state_dict",
    },
    "training": {
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "epochs": 10,
        "train_batch_size": 16,
        "development_batch_size": 16,
        "num_workers": 0,
        "selection_metric": "development matched MPJPE in millimetres",
    },
    "overfit": {
        "batch_size": 8,
        "steps": 200,
        "criterion": {
            "all_losses_finite": True,
            "final_loss_less_than_initial": True,
            "max_final_mpjpe_stored_units": 0.10,
            "max_final_to_initial_mpjpe_ratio": 0.30,
        },
    },
    "metrics": {
        "stored_unit_to_mm": 1000.0,
        "presence_threshold": 0.5,
        "match_threshold_mm": 500.0,
        "pck_threshold": 0.5,
        "pck_scale_joint_indices": [6, 4],
        "pck_guard_epsilon_stored": 1e-6,
        "root_index": 0,
        "pa_degenerate_epsilon_stored": 1e-12,
        "evaluation_name": "inferred-count",
    },
    "presence_score": 1.0,
    "label_access": {
        "training_labels": "frozen internal-train one-person keypoints only",
        "development_labels": "frozen internal-development one-person keypoints only",
        "official_test_used": False,
        "multi_person_labels_used": False,
    },
}


def step4b_config_for_json():
    return json.loads(json.dumps(STEP4B_CONFIG))


def validate_step4b_config(config):
    expected = step4b_config_for_json()
    if config != expected:
        raise ValueError("Frozen Step-4B configuration does not match step4b_config.py")
    return config
