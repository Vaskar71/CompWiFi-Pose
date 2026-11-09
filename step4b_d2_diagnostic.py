import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from feeder.piw3d_single_person import SinglePersonPiW3DDataset, make_single_person_dataloader
from model.model import MAE_ViT, ViT_Pose_Decoder
from step4b_config import validate_step4b_config
from step4b_single_person_pilot import _resolve_device, _seed_everything
from step4b_subset import is_single_person_id


D2_VERSION = "compwifi-step4b-d2-v1"
D2_TYPE = "controlled encoder-trainability diagnostic after failed frozen Step-4B tiny-overfit gate"
EXPECTED_SOURCE_STEPS = 200
EXPECTED_BATCH_IDS = [
    "S11_06_10",
    "S11_06_11",
    "S11_06_13",
    "S11_06_14",
    "S11_06_16",
    "S11_06_17",
    "S11_06_19",
    "S11_06_20",
]
EXPECTED_CSI_SHAPE = (3, 180, 20)
EXPECTED_KEYPOINT_SHAPE = (1, 14, 3)
ENCODER_ROOTS = {"cls_token", "pos_embedding", "patchify", "transformer", "layer_norm"}
DEFAULT_SOURCE_ARTIFACT_ROOT = Path("data/_artifacts/step4b_pilot")
DEFAULT_D2_ROOT = Path("data/_artifacts/step4b_d2")
PRE_EDIT_AUDIT_PATH = DEFAULT_D2_ROOT / "parameter_audit_before_edit.json"


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_file_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def _resolve_source_paths(source_artifact_root, freeze_config, pilot_root):
    source_artifact_root = Path(source_artifact_root)
    freeze_config = (
        Path(freeze_config)
        if freeze_config
        else source_artifact_root / "freeze_v2" / "step4b_freeze_config.json"
    )
    pilot_root = Path(pilot_root) if pilot_root else source_artifact_root / "pilot_data_v2"
    return source_artifact_root, freeze_config, pilot_root


def _module_name(parameter_name):
    return "encoder" if parameter_name.split(".", 1)[0] in ENCODER_ROOTS else "pose_decoder"


def parameter_audit(model, optimizer=None):
    optimizer_ids = [] if optimizer is None else [
        id(parameter) for group in optimizer.param_groups for parameter in group["params"]
    ]
    optimizer_id_set = set(optimizer_ids)
    id_to_name = {id(parameter): name for name, parameter in model.named_parameters()}
    optimizer_names = [id_to_name.get(parameter_id) for parameter_id in optimizer_ids]
    records = []
    for name, parameter in model.named_parameters():
        records.append(
            {
                "name": name,
                "module": _module_name(name),
                "shape": list(parameter.shape),
                "numel": int(parameter.numel()),
                "requires_grad": bool(parameter.requires_grad),
                "in_optimizer": id(parameter) in optimizer_id_set if optimizer is not None else None,
            }
        )
    trainable_records = [record for record in records if record["requires_grad"]]
    optimizer_record_names = [record["name"] for record in records if record["in_optimizer"]]
    optimizer_exact = (
        optimizer is not None
        and len(optimizer_ids) == len(optimizer_id_set)
        and all(name is not None for name in optimizer_names)
        and sorted(optimizer_record_names) == sorted(record["name"] for record in trainable_records)
    )
    encoder_records = [record for record in records if record["module"] == "encoder"]
    decoder_records = [record for record in records if record["module"] == "pose_decoder"]
    return {
        "optimizer": None
        if optimizer is None
        else {
            "parameter_references": len(optimizer_ids),
            "unique_parameter_references": len(optimizer_id_set),
            "duplicate_optimizer_references": len(optimizer_ids) - len(optimizer_id_set),
            "unknown_optimizer_references": sum(name is None for name in optimizer_names),
            "contains_exact_trainable_parameter_set": optimizer_exact,
        },
        "parameter_counts": {
            "total_parameters": sum(record["numel"] for record in records),
            "trainable_parameters": sum(record["numel"] for record in trainable_records),
            "total_named_parameters": len(records),
            "trainable_named_parameters": len(trainable_records),
        },
        "module_counts": {
            "encoder": {
                "named_parameters": len(encoder_records),
                "trainable_named_parameters": sum(record["requires_grad"] for record in encoder_records),
                "trainable_parameter_numel": sum(
                    record["numel"] for record in encoder_records if record["requires_grad"]
                ),
            },
            "pose_decoder": {
                "named_parameters": len(decoder_records),
                "trainable_named_parameters": sum(record["requires_grad"] for record in decoder_records),
                "trainable_parameter_numel": sum(
                    record["numel"] for record in decoder_records if record["requires_grad"]
                ),
            },
        },
        "optimizer_contains_exact_trainable_parameter_set": optimizer_exact,
        "records": records,
    }


def make_sgd_optimizer(model, config):
    return torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
    )


def build_d2_model(config, device):
    model_config = config["model"]
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
    return ViT_Pose_Decoder(
        mae.encoder,
        keypoints=model_config["keypoints"],
        coor_num=model_config["coordinate_dim"],
        token_num=900,
        dataset=model_config["dataset"],
        num_person=model_config["q"],
        train_encoder=True,
    ).to(device)


def _check_original_records_unchanged(pre_edit_audit):
    expected_hashes = pre_edit_audit["summary"]["source_hashes_before_edit"]
    protected_paths = [
        path
        for path in expected_hashes
        if Path(path).name != "model.py" and Path(path).name != "model"
    ]
    checks = {}
    for path in protected_paths:
        expected = expected_hashes[path]
        actual = _sha256_file(path)
        checks[path] = {"expected": expected, "actual": actual, "unchanged": actual == expected}
    return {
        "all_unchanged": all(check["unchanged"] for check in checks.values()),
        "checks": checks,
    }


def verify_frozen_inputs(
    source_artifact_root=DEFAULT_SOURCE_ARTIFACT_ROOT,
    freeze_config=None,
    pilot_root=None,
    pre_edit_audit_path=PRE_EDIT_AUDIT_PATH,
):
    source_artifact_root, freeze_config, pilot_root = _resolve_source_paths(
        source_artifact_root, freeze_config, pilot_root
    )
    if freeze_config.resolve().parent.name != "freeze_v2":
        raise ValueError("Step-4B-D2 requires the corrected freeze_v2 configuration")
    if pilot_root.name != "pilot_data_v2":
        raise ValueError("Step-4B-D2 requires the corrected pilot_data_v2 artifact")
    pre_edit_audit_path = Path(pre_edit_audit_path)
    if not pre_edit_audit_path.is_file():
        raise FileNotFoundError(f"Missing pre-edit parameter audit: {pre_edit_audit_path}")
    pre_edit_audit = _load_json(pre_edit_audit_path)
    if pre_edit_audit["summary"].get("runtime_confirmed_original_configuration") is not True:
        raise ValueError("Pre-edit audit did not confirm the original frozen-encoder configuration")

    freeze = _load_json(freeze_config)
    config = freeze["config"]
    validate_step4b_config(config)
    if config["seed"] != 0:
        raise ValueError("Frozen seed changed")
    if config["overfit"]["steps"] != EXPECTED_SOURCE_STEPS:
        raise ValueError("Frozen original optimizer-step budget changed")
    if config["overfit"]["batch_size"] != len(EXPECTED_BATCH_IDS):
        raise ValueError("Frozen tiny-batch size changed")
    if config["training"]["optimizer"] != "SGD":
        raise ValueError("Frozen optimizer changed")
    if config["training"]["learning_rate"] != 0.01:
        raise ValueError("Frozen learning rate changed")
    if config["training"]["momentum"] != 0.9:
        raise ValueError("Frozen momentum changed")
    if config["label_access"]["official_test_used"] is not False:
        raise ValueError("Frozen configuration must not use official test data")
    if config["label_access"]["multi_person_labels_used"] is not False:
        raise ValueError("Frozen configuration must not use multi-person labels")

    validation_path = source_artifact_root / "preprocessing_validation_v2.json"
    validation = _load_json(validation_path)
    if validation["official_test_used"] is not False:
        raise ValueError("Pilot validation must not use official test data")
    if validation["source_dataset_modified"] is not False:
        raise ValueError("Pilot validation must not modify the source dataset")
    if validation["counts"] != {
        "failed_ids": 0,
        "selected_development_ids": 512,
        "selected_train_ids": 2048,
        "total_selected_ids": 2560,
        "validated_ids": 2560,
    }:
        raise ValueError("Corrected pilot preprocessing validation counts changed")
    if validation["failures"]:
        raise ValueError("Corrected pilot preprocessing validation contains failures")

    overfit_path = source_artifact_root / "overfit_result.json"
    overfit = _load_json(overfit_path)
    if overfit["status"] != "FAIL":
        raise ValueError("Original Step-4B overfit artifact is not the recorded failure")
    if overfit["steps"] != EXPECTED_SOURCE_STEPS:
        raise ValueError("Original Step-4B overfit step budget changed")
    if overfit["batch_sample_ids"] != EXPECTED_BATCH_IDS:
        raise ValueError("Original Step-4B tiny-batch sample IDs changed")
    if overfit["final_mpjpe_stored_units"] <= config["overfit"]["criterion"]["max_final_mpjpe_stored_units"]:
        raise ValueError("Original Step-4B result no longer records the threshold failure")

    decision_path = source_artifact_root / "decision.json"
    decision = _load_json(decision_path)
    if decision["status"] != "BLOCKED":
        raise ValueError("Original Step-4B decision must remain BLOCKED")

    d1_result_path = Path("data/_artifacts/step4b_d1/step4b_d1_result.json")
    d1_decision_path = Path("data/_artifacts/step4b_d1/step4b_d1_decision.json")
    d1_result = _load_json(d1_result_path)
    d1_decision = _load_json(d1_decision_path)
    if d1_result["diagnostic_status"] != "EXTENDED_STEP_DIAGNOSTIC_ABOVE_CRITERION":
        raise ValueError("Step-4B-D1 result changed from its recorded above-criterion diagnostic")
    if d1_decision["original_step4b_status"] != "BLOCKED":
        raise ValueError("Step-4B-D1 no longer preserves the original BLOCKED status")

    original_record_check = _check_original_records_unchanged(pre_edit_audit)
    if not original_record_check["all_unchanged"]:
        raise ValueError("Original Step-4B or D1 records changed after the pre-edit audit")

    selected_train_ids = list(freeze["selected_ids"]["internal_train"])
    selected_development_ids = list(freeze["selected_ids"]["internal_development"])
    if selected_train_ids[: len(EXPECTED_BATCH_IDS)] != EXPECTED_BATCH_IDS:
        raise ValueError("Frozen internal-train ordering does not reproduce the tiny batch")
    if len(selected_train_ids) != 2048 or len(selected_development_ids) != 512:
        raise ValueError("Frozen internal subset counts changed")
    if set(selected_train_ids) & set(selected_development_ids):
        raise ValueError("Frozen internal train/development IDs overlap")
    if not all(is_single_person_id(sample_id) for sample_id in selected_train_ids):
        raise ValueError("Frozen internal-train subset contains a non-single-person ID")
    if not all(freeze["split_assignments"].get(sample_id) == "internal_train" for sample_id in selected_train_ids):
        raise ValueError("Frozen internal-train split assignment changed")
    if not all(
        freeze["split_assignments"].get(sample_id) == "internal_development"
        for sample_id in selected_development_ids
    ):
        raise ValueError("Frozen internal-development split assignment changed")

    selected_train_path = pilot_root / "train_data" / "selected_train_ids.txt"
    persisted_train_ids = selected_train_path.read_text(encoding="utf-8-sig").splitlines()
    if persisted_train_ids != selected_train_ids:
        raise ValueError("Pilot selected_train_ids.txt does not exactly match freeze_v2")

    pairing_records = []
    pilot_train_root = pilot_root / "train_data"
    for sample_id in EXPECTED_BATCH_IDS:
        csi_path = pilot_train_root / "csi_ap" / f"{sample_id}.npy"
        keypoint_path = pilot_train_root / "keypoint" / f"{sample_id}.npy"
        if csi_path.stem != sample_id or keypoint_path.stem != sample_id:
            raise ValueError(f"CSI/keypoint filename pairing changed for {sample_id}")
        if not csi_path.is_file():
            raise FileNotFoundError(f"Missing pilot CSI for frozen tiny batch: {csi_path}")
        if not keypoint_path.is_file():
            raise FileNotFoundError(f"Missing pilot keypoint for frozen tiny batch: {keypoint_path}")
        csi = np.load(csi_path, allow_pickle=False)
        keypoint = np.load(keypoint_path, allow_pickle=False)
        if csi.shape != EXPECTED_CSI_SHAPE:
            raise ValueError(f"Unexpected CSI shape for {sample_id}: {csi.shape}")
        if keypoint.shape != EXPECTED_KEYPOINT_SHAPE:
            raise ValueError(f"Unexpected keypoint shape for {sample_id}: {keypoint.shape}")
        if not np.isfinite(csi).all() or not np.isfinite(keypoint).all():
            raise ValueError(f"Non-finite frozen tiny-batch arrays for {sample_id}")
        pairing_records.append(
            {
                "sample_id": sample_id,
                "csi_path": str(csi_path),
                "keypoint_path": str(keypoint_path),
                "csi_shape": list(csi.shape),
                "keypoint_shape": list(keypoint.shape),
                "paired_by_exact_sample_id": True,
            }
        )

    return {
        "diagnostic_version": D2_VERSION,
        "diagnostic_type": D2_TYPE,
        "source_algorithm_version": config["algorithm_version"],
        "source_seed": config["seed"],
        "source_batch_size": config["overfit"]["batch_size"],
        "source_optimizer_steps": config["overfit"]["steps"],
        "source_optimizer": config["training"]["optimizer"],
        "source_learning_rate": config["training"]["learning_rate"],
        "source_momentum": config["training"]["momentum"],
        "source_loss": "mean Euclidean distance over people, joints, and coordinate triples",
        "source_coordinate_convention": (
            "stored coordinates unchanged; coordinate axes are reported by zero-based index only; "
            "no axis reordering or official axis-name interpretation"
        ),
        "source_model": config["model"],
        "source_parameter_counts": overfit["model_parameters"],
        "source_initial_mpjpe_stored_units": overfit["initial_mpjpe_stored_units"],
        "source_final_mpjpe_stored_units": overfit["final_mpjpe_stored_units"],
        "source_final_to_initial_mpjpe_ratio": overfit["final_to_initial_mpjpe_ratio"],
        "source_absolute_mpjpe_criterion_stored_units": config["overfit"]["criterion"][
            "max_final_mpjpe_stored_units"
        ],
        "source_overfit_status": overfit["status"],
        "source_overfit_failures": overfit["failures"],
        "source_step4b_decision": decision,
        "d1_diagnostic_status": d1_result["diagnostic_status"],
        "d1_final_mpjpe_stored_units": d1_result["final_mpjpe_stored_units"],
        "d1_decision": d1_decision,
        "pre_edit_parameter_audit": _source_file_record(pre_edit_audit_path),
        "original_records_unchanged_since_pre_edit_audit": original_record_check,
        "input_manifests": freeze["input_manifests"],
        "subset_counts": freeze["counts"],
        "label_access": config["label_access"],
        "preprocessing_validation_counts": validation["counts"],
        "selected_internal_train_ids_count": len(selected_train_ids),
        "selected_internal_development_ids_count": len(selected_development_ids),
        "tiny_batch_sample_ids": list(EXPECTED_BATCH_IDS),
        "tiny_batch_exact_pairing": pairing_records,
        "official_test_used": False,
        "development_samples_loaded": False,
    }


def write_diagnostic_freeze(source_artifact_root, d2_root, freeze_config, pilot_root, verification):
    source_artifact_root = Path(source_artifact_root)
    d2_root = Path(d2_root)
    d2_root.mkdir(parents=True, exist_ok=True)
    freeze_path = d2_root / "step4b_d2_freeze.json"
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen D2 diagnostic: {freeze_path}")
    record = {
        "diagnostic_version": D2_VERSION,
        "diagnostic_type": D2_TYPE,
        "source_records": {
            "step4b_freeze_v2": _source_file_record(freeze_config),
            "preprocessing_validation_v2": _source_file_record(
                source_artifact_root / "preprocessing_validation_v2.json"
            ),
            "overfit_result": _source_file_record(source_artifact_root / "overfit_result.json"),
            "step4b_decision": _source_file_record(source_artifact_root / "decision.json"),
            "selected_train_ids": _source_file_record(
                Path(pilot_root) / "train_data" / "selected_train_ids.txt"
            ),
            "step4b_d1_result": _source_file_record("data/_artifacts/step4b_d1/step4b_d1_result.json"),
            "step4b_d1_decision": _source_file_record("data/_artifacts/step4b_d1/step4b_d1_decision.json"),
        },
        "frozen_source_summary": verification,
        "diagnostic_freeze": {
            "diagnostic_steps": EXPECTED_SOURCE_STEPS,
            "original_optimizer_steps": EXPECTED_SOURCE_STEPS,
            "experimental_change": {
                "encoder_trainability": "frozen -> trainable",
            },
            "only_encoder_trainability_changed": True,
            "seed": verification["source_seed"],
            "batch_size": verification["source_batch_size"],
            "batch_sample_ids": list(EXPECTED_BATCH_IDS),
            "model_family": verification["source_model"]["family"],
            "initialization_procedure": "unchanged; same from-scratch seed-0 model construction and train_encoder setting applied without consuming RNG",
            "optimizer": {
                "type": verification["source_optimizer"],
                "learning_rate": verification["source_learning_rate"],
                "momentum": verification["source_momentum"],
            },
            "loss": verification["source_loss"],
            "absolute_final_mpjpe_criterion_stored_units": verification[
                "source_absolute_mpjpe_criterion_stored_units"
            ],
            "unchanged_fields": [
                "corrected freeze_v2 internal-train sample IDs",
                "ordered fixed batch of 8 samples",
                "seed",
                "model family and dimensions",
                "random initialization values",
                "CSI preprocessing and loader normalization",
                "pose loss",
                "SGD optimizer",
                "learning rate",
                "momentum",
                "optimizer-step budget",
                "batch size",
                "device selection mode",
                "coordinate order",
                "absolute final-MPJPE criterion",
            ],
            "development_samples_loaded": False,
            "official_test_used": False,
            "coordinate_order": verification["source_coordinate_convention"],
        },
    }
    _write_json(freeze_path, record)
    return freeze_path, record


def _capture_rng_state():
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state().clone(),
    }
    if torch.cuda.is_available():
        state["cuda"] = [value.clone() for value in torch.cuda.get_rng_state_all()]
    return state


def _restore_rng_state(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if "cuda" in state:
        torch.cuda.set_rng_state_all(state["cuda"])


def _clone_model_state(model):
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _restore_model_state(model, state):
    with torch.no_grad():
        for name, value in model.state_dict().items():
            value.copy_(state[name])


def _gradient_statistics(model):
    stats = {
        "gradient_tensors": 0,
        "finite_gradient_tensors": 0,
        "max_abs_gradient": 0.0,
        "encoder_gradient_tensors": 0,
        "finite_encoder_gradient_tensors": 0,
        "nonzero_encoder_gradient_tensors": 0,
        "max_abs_encoder_gradient": 0.0,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        stats["gradient_tensors"] += 1
        if parameter.grad is None:
            raise ValueError(f"Trainable parameter has no gradient: {name}")
        finite = bool(torch.isfinite(parameter.grad).all().item())
        if finite:
            stats["finite_gradient_tensors"] += 1
        max_abs = float(parameter.grad.detach().abs().max().item())
        if np.isfinite(max_abs):
            stats["max_abs_gradient"] = max(stats["max_abs_gradient"], max_abs)
        if _module_name(name) == "encoder":
            stats["encoder_gradient_tensors"] += 1
            if finite:
                stats["finite_encoder_gradient_tensors"] += 1
            if bool((parameter.grad != 0).any().item()):
                stats["nonzero_encoder_gradient_tensors"] += 1
            if np.isfinite(max_abs):
                stats["max_abs_encoder_gradient"] = max(stats["max_abs_encoder_gradient"], max_abs)
    stats["all_gradients_finite"] = stats["finite_gradient_tensors"] == stats["gradient_tensors"]
    stats["all_encoder_gradients_finite"] = (
        stats["finite_encoder_gradient_tensors"] == stats["encoder_gradient_tensors"]
    )
    stats["all_encoder_gradients_nonzero"] = (
        stats["nonzero_encoder_gradient_tensors"] == stats["encoder_gradient_tensors"]
    )
    return stats


def _capture_checkpoint(model, csi, pose, sample_ids, step):
    if not model.training:
        raise ValueError("D2 checkpoint evidence must use the same training-mode model state as the frozen gate")
    with torch.no_grad():
        predicted_pose, _ = model(csi)
        loss = torch.mean(torch.norm(predicted_pose - pose, dim=-1))
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite D2 checkpoint loss at step {step}")
    prediction = predicted_pose.detach().cpu().numpy().astype(np.float64)
    target = pose.detach().cpu().numpy().astype(np.float64)
    joint_errors = np.linalg.norm(prediction - target, axis=-1)
    return {
        "step": step,
        "loss_stored_units": float(loss.detach().cpu()),
        "mpjpe_stored_units": float(loss.detach().cpu()),
        "sample_ids": list(sample_ids),
        "per_sample_mpjpe_stored_units": [
            float(value) for value in np.mean(joint_errors, axis=(1, 2))
        ],
        "all_values_finite": bool(np.isfinite(prediction).all() and np.isfinite(target).all()),
        "prediction": prediction,
        "target": target,
    }


def summarize_parameter_changes(model, initial_state):
    summary = {
        "trainable_parameters": 0,
        "changed_trainable_parameters": 0,
        "encoder_parameters": 0,
        "changed_encoder_parameters": 0,
        "max_abs_parameter_change": 0.0,
        "max_abs_encoder_parameter_change": 0.0,
        "per_parameter": [],
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        difference = (parameter.detach() - initial_state[name].to(parameter.device)).abs()
        max_abs = float(difference.max().item()) if difference.numel() else 0.0
        changed = not torch.equal(parameter.detach(), initial_state[name].to(parameter.device))
        module = _module_name(name)
        summary["trainable_parameters"] += 1
        summary["changed_trainable_parameters"] += int(changed)
        summary["max_abs_parameter_change"] = max(summary["max_abs_parameter_change"], max_abs)
        if module == "encoder":
            summary["encoder_parameters"] += 1
            summary["changed_encoder_parameters"] += int(changed)
            summary["max_abs_encoder_parameter_change"] = max(
                summary["max_abs_encoder_parameter_change"], max_abs
            )
        summary["per_parameter"].append(
            {
                "name": name,
                "module": module,
                "changed": changed,
                "max_abs_change": max_abs,
            }
        )
    summary["all_trainable_parameters_changed"] = (
        summary["changed_trainable_parameters"] == summary["trainable_parameters"]
    )
    summary["encoder_parameters_changed"] = summary["changed_encoder_parameters"] > 0
    summary["all_encoder_parameters_changed"] = (
        summary["changed_encoder_parameters"] == summary["encoder_parameters"]
    )
    return summary


def run_diagnostic(source_artifact_root, d2_root, freeze_config, pilot_root, device_name):
    source_artifact_root, freeze_config, pilot_root = _resolve_source_paths(
        source_artifact_root, freeze_config, pilot_root
    )
    d2_root = Path(d2_root)
    verification = verify_frozen_inputs(source_artifact_root, freeze_config, pilot_root)
    _write_json(d2_root / "input_verification.json", verification)
    freeze_path, freeze_record = write_diagnostic_freeze(
        source_artifact_root, d2_root, freeze_config, pilot_root, verification
    )

    source_freeze = _load_json(freeze_config)
    config = source_freeze["config"]
    diagnostic_freeze = freeze_record["diagnostic_freeze"]
    if diagnostic_freeze["diagnostic_steps"] != EXPECTED_SOURCE_STEPS:
        raise AssertionError("D2 diagnostic step budget is not the frozen 200-step budget")

    device = _resolve_device(device_name)
    _seed_everything(config["seed"])
    train_dataset = SinglePersonPiW3DDataset(
        pilot_root,
        pilot_root / "train_data" / "selected_train_ids.txt",
        expected_split="internal_train",
        split_assignments_path=freeze_config.parent / "split_assignments.json",
    )
    if train_dataset.sample_ids[: len(EXPECTED_BATCH_IDS)] != EXPECTED_BATCH_IDS:
        raise AssertionError("D2 loader did not reproduce the frozen ordered tiny batch")

    model = build_d2_model(config, device)
    model.train()
    optimizer = make_sgd_optimizer(model, config)
    trainability_audit = parameter_audit(model, optimizer)
    if not trainability_audit["optimizer_contains_exact_trainable_parameter_set"]:
        raise AssertionError("D2 optimizer does not contain exactly the intended trainable parameter set")
    if trainability_audit["parameter_counts"]["trainable_named_parameters"] != 97:
        raise AssertionError("D2 trainable encoder/decoder parameter audit failed")
    if trainability_audit["module_counts"]["encoder"]["trainable_named_parameters"] != 50:
        raise AssertionError("D2 encoder is not fully trainable")
    if trainability_audit["module_counts"]["pose_decoder"]["trainable_named_parameters"] != 47:
        raise AssertionError("D2 pose decoder is not fully trainable")

    ordered_loader = make_single_person_dataloader(
        train_dataset,
        is_training=False,
        generator=torch.Generator().manual_seed(config["seed"]),
        batch_size=config["overfit"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    batch = next(iter(ordered_loader))
    csi = batch["csi"].to(device)
    pose = batch["pose"].to(device)
    batch_sample_ids = list(train_dataset.sample_ids[: csi.shape[0]])

    start_time = time.perf_counter()
    initial_model_state = _clone_model_state(model)
    checkpoint_summaries = {}
    checkpoint_arrays = {}
    initial_checkpoint = _capture_checkpoint(model, csi, pose, batch_sample_ids, 0)
    checkpoint_summaries[0] = initial_checkpoint
    checkpoint_arrays[0] = initial_checkpoint

    trajectory = []
    for step in range(1, EXPECTED_SOURCE_STEPS + 1):
        optimizer.zero_grad()
        predicted_pose, _ = model(csi)
        loss = torch.mean(torch.norm(predicted_pose - pose, dim=-1))
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite D2 loss at step {step}")
        loss.backward()
        gradient_stats = _gradient_statistics(model)
        if not gradient_stats["all_gradients_finite"]:
            raise ValueError(f"Non-finite D2 gradients at step {step}")
        optimizer.step()
        trajectory.append(
            {
                "step": step,
                "pre_update_loss_stored_units": float(loss.detach().cpu()),
                "pre_update_mpjpe_stored_units": float(loss.detach().cpu()),
                **gradient_stats,
            }
        )
        if step in (100, EXPECTED_SOURCE_STEPS):
            model_state = _clone_model_state(model)
            rng_state = _capture_rng_state()
            checkpoint = _capture_checkpoint(model, csi, pose, batch_sample_ids, step)
            checkpoint_summaries[step] = checkpoint
            checkpoint_arrays[step] = checkpoint
            _restore_model_state(model, model_state)
            _restore_rng_state(rng_state)

    runtime_seconds = time.perf_counter() - start_time
    initial = checkpoint_summaries[0]["mpjpe_stored_units"]
    step_100 = checkpoint_summaries[100]["mpjpe_stored_units"]
    final = checkpoint_summaries[EXPECTED_SOURCE_STEPS]["mpjpe_stored_units"]
    criterion = diagnostic_freeze["absolute_final_mpjpe_criterion_stored_units"]
    criterion_passed = final <= criterion
    finite_evidence = all(record["all_gradients_finite"] for record in trajectory) and all(
        np.isfinite(record["pre_update_loss_stored_units"]) for record in trajectory
    )
    if not finite_evidence:
        raise AssertionError("D2 completed without finite-loss/finite-gradient evidence")

    parameter_change = summarize_parameter_changes(model, initial_model_state)
    if parameter_change["changed_encoder_parameters"] == 0:
        raise AssertionError("D2 completed without evidence that encoder parameters changed")

    for step, checkpoint in checkpoint_summaries.items():
        _write_json(
            d2_root / f"checkpoint_step_{step:04d}.json",
            {key: value for key, value in checkpoint.items() if key not in {"prediction", "target"}},
        )
    np.savez(
        d2_root / "checkpoint_predictions.npz",
        sample_ids=np.array(batch_sample_ids),
        **{
            f"prediction_step_{step:04d}": arrays["prediction"][:, None, :, :]
            for step, arrays in checkpoint_arrays.items()
        },
        ground_truth=checkpoint_arrays[0]["target"][:, None, :, :],
    )
    _write_json(d2_root / "loss_trajectory.json", {"records": trajectory})

    checkpoint_path = d2_root / "step4b_d2_final_state_dict.pt"
    torch.save(model.state_dict(), checkpoint_path)
    reloaded_model = build_d2_model(config, device)
    reloaded_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    reloaded_model.eval()
    with torch.no_grad():
        reloaded_prediction, _ = reloaded_model(csi)
        reloaded_loss = torch.mean(torch.norm(reloaded_prediction - pose, dim=-1))
    if not torch.isfinite(reloaded_loss):
        raise ValueError("Non-finite D2 checkpoint reload evaluation loss")

    if criterion_passed:
        diagnostic_status = "TRAINABLE_ENCODER_DIAGNOSTIC_PASSED"
        interpretation = (
            "D2 passed with trainable encoder; original Step-4B remains historically blocked."
        )
    else:
        diagnostic_status = "TRAINABLE_ENCODER_DIAGNOSTIC_ABOVE_CRITERION"
        improvement = verification["source_final_mpjpe_stored_units"] - final
        interpretation = (
            "D2 did not pass the unchanged absolute criterion with a trainable encoder; "
            f"final MPJPE changed by {improvement:.17g} stored units relative to the original "
            "200-step frozen-encoder result. This controlled result is reported without another fix."
        )

    result = {
        "diagnostic_version": D2_VERSION,
        "diagnostic_type": D2_TYPE,
        "diagnostic_run_status": "COMPLETE",
        "diagnostic_status": diagnostic_status,
        "interpretation": interpretation,
        "original_result_preserved": True,
        "source_step4b_decision": verification["source_step4b_decision"],
        "d1_result_preserved": True,
        "freeze_record": str(freeze_path),
        "device": str(device),
        "runtime_seconds": runtime_seconds,
        "seed": config["seed"],
        "batch_size": int(csi.shape[0]),
        "batch_sample_ids": batch_sample_ids,
        "optimizer_steps": EXPECTED_SOURCE_STEPS,
        "optimizer": config["training"]["optimizer"],
        "learning_rate": config["training"]["learning_rate"],
        "momentum": config["training"]["momentum"],
        "loss": verification["source_loss"],
        "absolute_final_mpjpe_criterion_stored_units": criterion,
        "criterion_passed": criterion_passed,
        "all_losses_finite": bool(all(np.isfinite(record["pre_update_loss_stored_units"]) for record in trajectory)),
        "all_gradients_finite": bool(all(record["all_gradients_finite"] for record in trajectory)),
        "all_encoder_gradients_finite": bool(all(record["all_encoder_gradients_finite"] for record in trajectory)),
        "all_encoder_gradients_nonzero": bool(all(record["all_encoder_gradients_nonzero"] for record in trajectory)),
        "encoder_gradient_evidence": {
            "min_nonzero_encoder_gradient_tensors": min(
                record["nonzero_encoder_gradient_tensors"] for record in trajectory
            ),
            "max_nonzero_encoder_gradient_tensors": max(
                record["nonzero_encoder_gradient_tensors"] for record in trajectory
            ),
            "max_abs_encoder_gradient": max(record["max_abs_encoder_gradient"] for record in trajectory),
        },
        "initial_mpjpe_stored_units": initial,
        "step_100_mpjpe_stored_units": step_100,
        "final_mpjpe_stored_units": final,
        "final_to_initial_mpjpe_ratio": final / initial if initial > 0 else None,
        "final_minus_original_200_step_mpjpe_stored_units": (
            final - verification["source_final_mpjpe_stored_units"]
        ),
        "initial": {key: value for key, value in checkpoint_summaries[0].items() if key not in {"prediction", "target"}},
        "step_100": {key: value for key, value in checkpoint_summaries[100].items() if key not in {"prediction", "target"}},
        "final": {
            key: value
            for key, value in checkpoint_summaries[EXPECTED_SOURCE_STEPS].items()
            if key not in {"prediction", "target"}
        },
        "trajectory_summary": {
            "first": trajectory[0]["pre_update_loss_stored_units"],
            "step_100": trajectory[99]["pre_update_loss_stored_units"],
            "last": trajectory[-1]["pre_update_loss_stored_units"],
            "min": min(record["pre_update_loss_stored_units"] for record in trajectory),
            "max": max(record["pre_update_loss_stored_units"] for record in trajectory),
        },
        "trainability_audit": {
            "optimizer": trainability_audit["optimizer"],
            "parameter_counts": trainability_audit["parameter_counts"],
            "module_counts": trainability_audit["module_counts"],
            "optimizer_contains_exact_trainable_parameter_set": trainability_audit[
                "optimizer_contains_exact_trainable_parameter_set"
            ],
        },
        "parameter_change_evidence": {
            key: value for key, value in parameter_change.items() if key != "per_parameter"
        },
        "model_parameters": verification["source_parameter_counts"],
        "coordinate_convention": verification["source_coordinate_convention"],
        "development_samples_loaded": False,
        "official_test_used": False,
        "checkpoint": {
            "path": str(checkpoint_path),
            "format": "state_dict",
            "reloaded_loss_stored_units": float(reloaded_loss.detach().cpu()),
            "reloaded_mpjpe_stored_units": float(reloaded_loss.detach().cpu()),
            "reload_finite": True,
        },
    }
    _write_json(d2_root / "step4b_d2_result.json", result)
    _write_json(
        d2_root / "step4b_d2_decision.json",
        {
            "status": diagnostic_status,
            "reason": interpretation,
            "final_mpjpe_stored_units": final,
            "original_absolute_criterion_stored_units": criterion,
            "original_step4b_status": verification["source_step4b_decision"]["status"],
            "d1_status": verification["d1_diagnostic_status"],
            "original_result_preserved": True,
            "d1_result_preserved": True,
            "result": str(d2_root / "step4b_d2_result.json"),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"d2_result={d2_root / 'step4b_d2_result.json'}")
    print(f"d2_decision={d2_root / 'step4b_d2_decision.json'}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run the frozen Step-4B-D2 encoder-trainability diagnostic.")
    parser.add_argument("--source-artifact-root", type=Path, default=DEFAULT_SOURCE_ARTIFACT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_D2_ROOT)
    parser.add_argument("--freeze-config", type=Path, default=None)
    parser.add_argument("--pilot-root", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    source_artifact_root, freeze_config, pilot_root = _resolve_source_paths(
        args.source_artifact_root, args.freeze_config, args.pilot_root
    )
    verification = verify_frozen_inputs(source_artifact_root, freeze_config, pilot_root)
    if args.verify_only:
        path = _write_json(args.artifact_root / "input_verification.json", verification)
        print(json.dumps(verification, indent=2, sort_keys=True))
        print(f"input_verification={path}")
        return 0
    return run_diagnostic(
        source_artifact_root,
        args.artifact_root,
        freeze_config,
        pilot_root,
        args.device,
    )


if __name__ == "__main__":
    raise SystemExit(main())
