"""Frozen Step-5b configuration (docs/STEP_5B_PLAN.md, docs/STEP_5B_PROTOCOL.md).

Paths are relative to ``--repo-root`` (the main checkout that holds ``data/``). Nothing in
this file may change after the protocol freeze; the runner refuses to run otherwise.
"Recording" always means a candidate filename group (e.g. ``S11_06``); no subject,
session or environment claim is made.
"""

import json
from pathlib import Path

from step5_config import step5_config_for_json


STEP5B_VERSION = "compwifi-step5b-v1"

STAGE1_RUN_IDS = ("D1", "D2", "D3", "D4")
SIMPLICITY_ORDER = ("D1", "D2", "D3", "D4")
VIEWS = (
    "fit_train_dense",
    "fit_train_sparse",
    "fit_holdout",
    "group_val",
    "all_train_dense",
    "all_train_sparse",
    "all_holdout",
    "group_dev",
)
ROLES = ("train", "temporal_holdout", "group_dev")

_STEP5 = step5_config_for_json()

STEP5B_CONFIG = {
    "algorithm_version": STEP5B_VERSION,
    "purpose": (
        "Does CSI carry a learnable body-pose signal beyond recording identity, first within seen "
        "recordings (T1), then on unseen recordings (X1)? docs/STEP_5B_PLAN.md"
    ),
    "paths": {
        "train_manifest": "data/_artifacts/piw3d_candidate_group_manifests/internal_train_ids.txt",
        "development_manifest": "data/_artifacts/piw3d_candidate_group_manifests/internal_development_ids.txt",
        "source_root": "data/person_in_wifi_3d/wifi_dataset",
        "step5_freeze_dir": "data/_artifacts/step5_single_person/freeze_v1",
        "artifact_root": "data/_artifacts/step5b_pose_signal",
        "freeze_dir": "data/_artifacts/step5b_pose_signal/freeze_v2",
        "data_root": "data/_artifacts/step5b_pose_signal/data_v2",
        "protocol_freeze_dir": "data/_artifacts/step5b_pose_signal/protocol_freeze",
    },
    "subset": {
        "holdout_start_divisor": 5,
        "holdout_length_cap": 49,
        "holdout_length_divisor": 10,
        "guard_positions": 16,
        "sparse_frames_per_recording": 64,
        "validation_recordings_per_scene": 3,
        "validation_rule": (
            "within each scene prefix, sort recording names by sha256(name.encode('utf-8')).hexdigest() "
            "ascending; the first 3 are validation recordings; names only, never the person-count character"
        ),
        "expected_validation_recordings": [
            "S11_32", "S11_03", "S11_10", "S21_07", "S21_29", "S21_27", "S51_11", "S51_32", "S51_23",
        ],
        "expected_counts": {
            "fit_train_dense": 16235,
            "fit_train_sparse": 2687,
            "fit_holdout": 1815,
            "group_val": 3775,
            "all_train_dense": 20010,
            "all_train_sparse": 3263,
            "all_holdout": 2232,
            "group_dev": 512,
            "guard": 1632,
            "train_recordings": 51,
            "fit_recordings": 42,
            "validation_recordings": 9,
        },
        "group_dev_source": "freeze_v1/group_dev_ids.txt, verified against step5_freeze.json id_list_sha256",
        "roles": {
            "train": "train-role frames: outside the holdout block and the 16-position guards",
            "temporal_holdout": "same-recording temporal holdout — NOT recording-disjoint (official-like)",
            "group_dev": (
                "Step-5 group_dev: internal-development single-person recordings, candidate-filename-group "
                "disjoint from train"
            ),
        },
        "official_test_used": False,
        "multi_person_samples_used": False,
    },
    "input_norm": {
        "options": ["legacy_minmax", "train_zscore", "recording_centered"],
        "std_floor": 1e-6,
        "train_zscore": "Step-5 train_zscore with input_stats_fit (stage 1) / input_stats_all (stage 2)",
        "recording_centered": (
            "(x - mu_recording) / max(sigma_train, 1e-6); mu_recording from recording_input_means.npz "
            "(train-role frames for the 51 training recordings, own 64 frames for group_dev recordings); "
            "sigma_train = train_zscore std of the stage"
        ),
        "uses_label_free_per_recording_input_statistics": True,
        "transductive_note": (
            "recording_centered uses label-free CSI means of the evaluated recordings themselves "
            "(group_val, group_dev); disclosed"
        ),
    },
    "model": {
        "name": "simple_cnn",
        "builder": "model.research_models.build_research_model('simple_cnn', codec_stats, step5_config_for_json(), device)",
        "codec_stats": "codec_stats_fit.json (stage 1) / codec_stats_all.json (stage 2)",
    },
    "training": dict(_STEP5["training"], first_epoch_budget_hours=1.0),
    "stage1": {
        "seed": 0,
        "runs": {
            "D1": {"input_norm": "legacy_minmax", "train_view": "fit_train_sparse",
                   "isolates": "reference: Step-5 R2 recipe at 42 recordings"},
            "D2": {"input_norm": "legacy_minmax", "train_view": "fit_train_dense", "isolates": "frame density"},
            "D3": {"input_norm": "train_zscore", "train_view": "fit_train_dense",
                   "isolates": "normalisation under dense data"},
            "D4": {"input_norm": "recording_centered", "train_view": "fit_train_dense",
                   "isolates": "removing the recording signature"},
        },
        "eval_views": ["train_view", "fit_holdout", "group_val"],
        "holdout_view": "fit_holdout",
        "unseen_view": "group_val",
        "stats_suffix": "fit",
    },
    "stage2": {
        "seeds": [0, 1, 2],
        "train_view": "all_train_dense",
        "train_view_if_D1": "all_train_sparse",
        "per_epoch_eval_views": ["train_view", "all_holdout"],
        "final_eval_views": ["train_view", "all_holdout", "group_dev"],
        "holdout_view": "all_holdout",
        "unseen_view": "group_dev",
        "group_dev_rule": "opened only by stage 2, evaluated once at the final epoch",
        "stats_suffix": "all",
    },
    "overfit": {
        "runs": ["D2", "D4"],
        "view": "fit_train_dense",
        "sample_count": 64,
        "sample_rule": "positions i*k of the view (sorted by recording, frame), k = len(view)//64, i = 0..63",
        "steps": 300,
        "learning_rate": 1e-3,
        "learning_rate_schedule": "constant",
        "weight_decay": 0.0,
        "grad_clip_norm": 1.0,
        "seed": 0,
        "pass_ratio": 0.20,
    },
    "evaluation": {
        "distance_units": "stored units",
        "mm_conversion": "stored_unit_to_mm=1000, provisional, assumes stored unit = metre",
        "translation_reference": "14-joint centroid",
        "shuffled_pairing_seed": 20261110,
        "research_metrics": _STEP5["evaluation"]["research_metrics"],
        "recording_mean_oracle": "mean pose of the recording's frames in the run's training view",
        "mean_pose_baselines": "train-view mean pose (abs) and train-view mean relative skeleton (rel)",
        "motion_skill": (
            "per recording r in the view: dgt = rel(gt) - mean_r rel(gt); dpred = rel(pred) - mean_r rel(pred); "
            "skill_r = 1 - mean_{frames,joints}||dpred-dgt|| / mean_{frames,joints}||dgt||; reported value = "
            "frame-weighted mean of skill_r over recordings; diagnostic only"
        ),
    },
    "gates": {
        "S": "all smoke checks pass and overfit ratio <= 0.20 for D2 and D4",
        "T1_ratio": 0.95,
        "T1": "holdout-view rel MPJPE <= 0.95 x recording-mean-oracle rel MPJPE",
        "X1_ratio": 0.95,
        "X1": "unseen-view rel MPJPE <= 0.95 x train-view mean-skeleton rel MPJPE",
        "X1_abs_reported_not_gated": "unseen-view abs MPJPE / mean-pose abs MPJPE (Step-5 G6 abs, 0.95)",
        "selection_min_pred_std": 0.01,
        "selection_tie_relative": 0.01,
        "selection": (
            "among stage-1 runs that did not crash and whose training-view pred_std_across_samples >= 0.01: "
            "lowest group_val rel-MPJPE ratio to the mean skeleton; ties (<1% relative) -> simpler D1<D2<D3<D4"
        ),
    },
    "verdicts": {
        "GREEN": "T1 and X1 pass in all three stage-2 seeds",
        "YELLOW-TRACKS": "T1 passes in all three seeds, X1 fails in at least one",
        "RED-NO-TRACKING": "selected run fails T1 in all three seeds and no stage-1 run passed T1 on fit_holdout",
        "INCONCLUSIVE": "anything else",
    },
    "determinism": _STEP5["determinism"],
    "label_access": {
        "training_labels": "single-person internal-train labels of the run's training view only",
        "holdout_labels": "evaluation and the recording-mean oracle only",
        "group_val_labels": "single-person training-partition labels: evaluation and stage-1 selection",
        "group_dev_labels": "development labels: single stage-2 confirmation only, never in stage 1",
        "codec_and_input_stats_source": "fit_train_dense (stage 1) / all_train_dense (stage 2)",
        "official_test_used": False,
        "multi_person_labels_used": False,
    },
}

STEP5B_SOURCE_FILES = (
    "step5b_config.py",
    "step5b_subset.py",
    "make_step5b_data.py",
    "feeder/piw3d_step5b.py",
    "step5b_pose_signal.py",
    "step5_config.py",
    "step5_subset.py",
    "make_step5_data.py",
    "feeder/piw3d_step5.py",
    "model/research_models.py",
    "step5_single_person_baseline.py",
)


def step5b_config_for_json():
    return json.loads(json.dumps(STEP5B_CONFIG))


def validate_step5b_config(config):
    if config != step5b_config_for_json():
        raise ValueError("Frozen Step-5b configuration does not match step5b_config.py")
    return config


def resolve_path(repo_root, key):
    return Path(repo_root) / STEP5B_CONFIG["paths"][key]
