import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from step4b_r_config import step4b_r_config_for_json, validate_step4b_r_config


R_VERSION = "compwifi-step4b-r-v1"
DEFAULT_ARTIFACT_ROOT = Path("data/_artifacts/step4b_r")
PACKAGE_NAMES = [
    "numpy",
    "torch",
    "torchvision",
    "scipy",
    "scikit-learn",
    "opencv-python",
    "pandas",
    "h5py",
    "PyWavelets",
    "timm",
    "einops",
    "PyYAML",
    "pytest",
]
RELEVANT_ENVIRONMENT_VARIABLES = [
    "PYTHONHASHSEED",
    "CUDA_VISIBLE_DEVICES",
    "CUDA_DEVICE_ORDER",
    "CUDA_LAUNCH_BLOCKING",
    "CUBLAS_WORKSPACE_CONFIG",
    "PYTORCH_CUDA_ALLOC_CONF",
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
]
ENCODER_ROOTS = {"cls_token", "pos_embedding", "patchify", "transformer", "layer_norm"}
EXPECTED_TRAINABLE_NAMED_PARAMETERS = 97
EXPECTED_ENCODER_NAMED_PARAMETERS = 50
EXPECTED_DECODER_NAMED_PARAMETERS = 47

np = None
torch = None
SinglePersonPiW3DDataset = None
make_single_person_dataloader = None
MAE_ViT = None
ViT_Pose_Decoder = None
validate_step4b_config = None
is_single_person_id = None
verify_protected_records = None


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _file_record(path):
    path = Path(path)
    return {"path": str(path.resolve()), "sha256": _sha256_file(path)}


def validate_process_environment():
    checks = {
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
    }
    expected = step4b_r_config_for_json()["determinism"]
    failures = []
    if checks["PYTHONHASHSEED"] != expected["PYTHONHASHSEED"]:
        failures.append(
            "PYTHONHASHSEED must be set to "
            f"{expected['PYTHONHASHSEED']!r} before the Python process starts"
        )
    if checks["CUBLAS_WORKSPACE_CONFIG"] != expected["CUBLAS_WORKSPACE_CONFIG"]:
        failures.append(
            "CUBLAS_WORKSPACE_CONFIG must be set to "
            f"{expected['CUBLAS_WORKSPACE_CONFIG']!r} before CUDA/cuBLAS initialization"
        )
    return {"checks": checks, "failures": failures, "valid": not failures}


def load_runtime_modules():
    global np, torch
    global SinglePersonPiW3DDataset, make_single_person_dataloader
    global MAE_ViT, ViT_Pose_Decoder
    global validate_step4b_config, is_single_person_id
    global verify_protected_records

    import numpy as numpy_module
    import torch as torch_module
    from feeder.piw3d_single_person import (
        SinglePersonPiW3DDataset as DatasetClass,
        make_single_person_dataloader as dataloader_factory,
    )
    from model.model import MAE_ViT as MaeClass, ViT_Pose_Decoder as PoseDecoderClass
    from step4b_config import validate_step4b_config as validate_original_config
    from step4b_subset import is_single_person_id as single_person_predicate
    from step4b_d4_reproducibility import verify_protected_records as protected_records_verifier

    np = numpy_module
    torch = torch_module
    SinglePersonPiW3DDataset = DatasetClass
    make_single_person_dataloader = dataloader_factory
    MAE_ViT = MaeClass
    ViT_Pose_Decoder = PoseDecoderClass
    validate_step4b_config = validate_original_config
    is_single_person_id = single_person_predicate
    verify_protected_records = protected_records_verifier


def configure_deterministic_runtime():
    if np is None or torch is None:
        raise RuntimeError("Runtime modules must be loaded before deterministic configuration")
    deterministic = step4b_r_config_for_json()["determinism"]
    torch.use_deterministic_algorithms(
        deterministic["torch_deterministic_algorithms"],
        warn_only=deterministic["torch_deterministic_warn_only"],
    )
    torch.backends.cudnn.deterministic = deterministic["cudnn_deterministic"]
    torch.backends.cudnn.benchmark = deterministic["cudnn_benchmark"]
    torch.backends.cuda.matmul.allow_tf32 = deterministic["cuda_matmul_allow_tf32"]
    torch.backends.cudnn.allow_tf32 = deterministic["cudnn_allow_tf32"]
    torch.set_float32_matmul_precision(deterministic["float32_matmul_precision"])
    _seed_everything(deterministic["torch_seed"])
    return {
        "torch_deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "torch_deterministic_warn_only": bool(torch.is_deterministic_algorithms_warn_only_enabled()),
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
        "cuda_matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
        "cudnn_allow_tf32": bool(torch.backends.cudnn.allow_tf32),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def _package_versions():
    versions = {}
    for name in PACKAGE_NAMES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def _nvidia_smi_summary():
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,driver_version",
                "--format=csv,noheader",
            ],
            capture_output=True,
            check=False,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def capture_environment(device, deterministic_settings):
    cuda_devices = []
    if torch.cuda.is_available():
        cuda_devices = [
            {
                "index": index,
                "name": torch.cuda.get_device_name(index),
                "capability": list(torch.cuda.get_device_capability(index)),
                "total_memory_bytes": int(torch.cuda.get_device_properties(index).total_memory),
            }
            for index in range(torch.cuda.device_count())
        ]
    return {
        "baseline_version": R_VERSION,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "python": {
            "version": sys.version,
            "version_info": list(sys.version_info[:5]),
            "executable": sys.executable,
            "implementation": platform.python_implementation(),
        },
        "packages": _package_versions(),
        "torch": {
            "version": torch.__version__,
            "cuda_build": torch.version.cuda,
            "cuda_available": bool(torch.cuda.is_available()),
            "cuda_device_count": int(torch.cuda.device_count()),
            "cuda_devices": cuda_devices,
            "cuda_architectures": list(torch.cuda.get_arch_list()),
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "deterministic_algorithms_enabled": bool(torch.are_deterministic_algorithms_enabled()),
            "deterministic_algorithms_warn_only_enabled": bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn": {
                "enabled": bool(torch.backends.cudnn.enabled),
                "benchmark": bool(torch.backends.cudnn.benchmark),
                "deterministic": bool(torch.backends.cudnn.deterministic),
                "allow_tf32": bool(torch.backends.cudnn.allow_tf32),
                "version": int(torch.backends.cudnn.version()),
            },
            "cuda": {
                "matmul_allow_tf32": bool(torch.backends.cuda.matmul.allow_tf32),
                "preferred_linalg_library": str(torch.backends.cuda.preferred_linalg_library()),
            },
            "num_threads": int(torch.get_num_threads()),
            "num_interop_threads": int(torch.get_num_interop_threads()),
        },
        "environment_variables": {
            name: os.environ.get(name) for name in RELEVANT_ENVIRONMENT_VARIABLES
        },
        "nvidia_smi": _nvidia_smi_summary(),
        "selected_device": str(device),
        "configured_deterministic_settings": deterministic_settings,
    }


def _resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def _resolve_source_paths(freeze_config, pilot_root):
    config = step4b_r_config_for_json()
    freeze_config = (
        Path(freeze_config) if freeze_config else Path(config["data"]["freeze_config"])
    )
    pilot_root = Path(pilot_root) if pilot_root else Path(config["data"]["pilot_root"])
    return freeze_config, pilot_root


def verify_frozen_inputs(freeze_config, pilot_root):
    freeze_config, pilot_root = _resolve_source_paths(freeze_config, pilot_root)
    if freeze_config.resolve().parent.name != "freeze_v2":
        raise ValueError("Step-4B-R requires the corrected freeze_v2 configuration")
    if pilot_root.name != "pilot_data_v2":
        raise ValueError("Step-4B-R requires the corrected pilot_data_v2 artifact")

    r_config = step4b_r_config_for_json()
    validate_step4b_r_config(r_config)
    source_freeze = _load_json(freeze_config)
    source_config = source_freeze["config"]
    validate_step4b_config(source_config)
    if source_config["algorithm_version"] != r_config["source_algorithm_version"]:
        raise ValueError("Corrected source freeze is not the protected Step-4B algorithm version")
    if source_config["seed"] != r_config["seed"]:
        raise ValueError("Corrected source freeze seed changed")
    for key, value in source_config["model"].items():
        if r_config["model"].get(key) != value:
            raise ValueError(f"Step-4B-R model setting diverged from the existing model: {key}")
    if r_config["model"]["train_encoder"] is not True:
        raise ValueError("Step-4B-R requires the existing trainable encoder")
    if source_config["label_access"]["official_test_used"] is not False:
        raise ValueError("Source freeze must not use official test data")
    if source_config["label_access"]["multi_person_labels_used"] is not False:
        raise ValueError("Source freeze must not use multi-person labels")

    validation_path = freeze_config.parent.parent / "preprocessing_validation_v2.json"
    validation = _load_json(validation_path)
    if validation["official_test_used"] is not False:
        raise ValueError("Corrected preprocessing validation must not use official test data")
    if validation["source_dataset_modified"] is not False:
        raise ValueError("Corrected preprocessing validation must not modify the source dataset")
    expected_counts = {
        "failed_ids": 0,
        "selected_development_ids": r_config["data"]["expected_development_count"],
        "selected_train_ids": r_config["data"]["expected_train_count"],
        "total_selected_ids": r_config["data"]["expected_train_count"]
        + r_config["data"]["expected_development_count"],
        "validated_ids": r_config["data"]["expected_train_count"]
        + r_config["data"]["expected_development_count"],
    }
    if validation["counts"] != expected_counts:
        raise ValueError("Corrected preprocessing validation counts changed")
    if validation["failures"]:
        raise ValueError("Corrected preprocessing validation contains failures")

    decision_path = freeze_config.parent.parent / "decision.json"
    decision = _load_json(decision_path)
    if decision["status"] != "BLOCKED":
        raise ValueError("Original Step-4B decision must remain BLOCKED")

    selected_train_ids = list(source_freeze["selected_ids"]["internal_train"])
    selected_development_ids = list(source_freeze["selected_ids"]["internal_development"])
    if len(selected_train_ids) != r_config["data"]["expected_train_count"]:
        raise ValueError("Corrected internal-train subset count changed")
    if len(selected_development_ids) != r_config["data"]["expected_development_count"]:
        raise ValueError("Corrected internal-development subset count changed")
    if len(set(selected_train_ids)) != len(selected_train_ids):
        raise ValueError("Corrected internal-train IDs contain duplicates")
    if len(set(selected_development_ids)) != len(selected_development_ids):
        raise ValueError("Corrected internal-development IDs contain duplicates")
    if set(selected_train_ids) & set(selected_development_ids):
        raise ValueError("Corrected internal train/development ID overlap must be zero")
    if not all(is_single_person_id(sample_id) for sample_id in selected_train_ids):
        raise ValueError("Corrected internal-train subset contains a non-single-person ID")
    if not all(is_single_person_id(sample_id) for sample_id in selected_development_ids):
        raise ValueError("Corrected internal-development subset contains a non-single-person ID")
    if not all(
        source_freeze["split_assignments"].get(sample_id) == "internal_train"
        for sample_id in selected_train_ids
    ):
        raise ValueError("Corrected internal-train split assignment changed")
    if not all(
        source_freeze["split_assignments"].get(sample_id) == "internal_development"
        for sample_id in selected_development_ids
    ):
        raise ValueError("Corrected internal-development split assignment changed")
    selected_train_groups = {
        sample_id.rsplit("_", 1)[0] for sample_id in selected_train_ids
    }
    selected_development_groups = {
        sample_id.rsplit("_", 1)[0] for sample_id in selected_development_ids
    }
    if selected_train_groups & selected_development_groups:
        raise ValueError("Corrected train/development group overlap must be zero")

    pilot_train_root = pilot_root / "train_data"
    persisted_train_ids = [
        line
        for line in (pilot_train_root / "selected_train_ids.txt")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line
    ]
    persisted_development_ids = [
        line
        for line in (pilot_train_root / "selected_development_ids.txt")
        .read_text(encoding="utf-8-sig")
        .splitlines()
        if line
    ]
    if persisted_train_ids != selected_train_ids:
        raise ValueError("Pilot selected_train_ids.txt does not exactly match freeze_v2")
    if persisted_development_ids != selected_development_ids:
        raise ValueError("Pilot selected_development_ids.txt does not exactly match freeze_v2")

    protected_records = verify_protected_records()
    if not protected_records["all_unchanged"]:
        changed = [
            path for path, check in protected_records["checks"].items() if not check["unchanged"]
        ]
        raise ValueError("Protected historical records changed: " + ", ".join(changed))

    return {
        "baseline_version": R_VERSION,
        "config": r_config,
        "source_freeze_config": _file_record(freeze_config),
        "source_preprocessing_validation": _file_record(validation_path),
        "source_step4b_decision": _file_record(decision_path),
        "pilot_root": str(pilot_root.resolve()),
        "split_verification": {
            "train_count": len(selected_train_ids),
            "development_count": len(selected_development_ids),
            "train_development_id_intersection": 0,
            "train_development_group_intersection": 0,
            "all_train_single_person_filename_marked": True,
            "all_development_single_person_filename_marked": True,
            "all_train_split_assignments_verified": True,
            "all_development_split_assignments_verified": True,
            "pilot_train_ids_match_freeze": True,
            "pilot_development_ids_match_freeze": True,
        },
        "source_subset_counts": source_freeze["counts"],
        "label_access": r_config["label_access"],
        "protected_records": protected_records,
        "official_test_used": False,
    }


def write_baseline_freeze(artifact_root, verification):
    freeze_path = Path(artifact_root) / "step4b_r_freeze.json"
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen Step-4B-R record: {freeze_path}")
    record = {
        "baseline_version": R_VERSION,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol": _file_record("docs/STEP_4B_R_PROTOCOL.md"),
        "configuration_source": _file_record("step4b_r_config.py"),
        "configuration": verification["config"],
        "source_records": {
            "step4b_freeze_v2": verification["source_freeze_config"],
            "preprocessing_validation_v2": verification["source_preprocessing_validation"],
            "step4b_decision": verification["source_step4b_decision"],
        },
        "source_verification": verification,
        "frozen_before_model_construction": True,
        "development_labels_evaluation_only": True,
        "mean_pose_training_labels_only": True,
        "official_test_used": False,
    }
    _write_json(freeze_path, record)
    return freeze_path, record


def build_step4b_r_model(config, device):
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
        train_encoder=model_config["train_encoder"],
    ).to(device)


def make_adamw_optimizer(model, config):
    return torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )


def _module_name(parameter_name):
    return "encoder" if parameter_name.split(".", 1)[0] in ENCODER_ROOTS else "pose_decoder"


def parameter_audit(model, optimizer):
    optimizer_ids = [
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
                "in_optimizer": id(parameter) in optimizer_id_set,
            }
        )
    trainable_records = [record for record in records if record["requires_grad"]]
    optimizer_record_names = [record["name"] for record in records if record["in_optimizer"]]
    optimizer_exact = (
        len(optimizer_ids) == len(optimizer_id_set)
        and all(name is not None for name in optimizer_names)
        and sorted(optimizer_record_names) == sorted(record["name"] for record in trainable_records)
    )
    encoder_records = [record for record in records if record["module"] == "encoder"]
    decoder_records = [record for record in records if record["module"] == "pose_decoder"]
    return {
        "optimizer": {
            "type": "AdamW",
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
                "trainable_named_parameters": sum(
                    record["requires_grad"] for record in encoder_records
                ),
                "trainable_parameter_numel": sum(
                    record["numel"] for record in encoder_records if record["requires_grad"]
                ),
            },
            "pose_decoder": {
                "named_parameters": len(decoder_records),
                "trainable_named_parameters": sum(
                    record["requires_grad"] for record in decoder_records
                ),
                "trainable_parameter_numel": sum(
                    record["numel"] for record in decoder_records if record["requires_grad"]
                ),
            },
        },
        "optimizer_contains_exact_trainable_parameter_set": optimizer_exact,
        "records": records,
    }


def _mean_per_joint_error(predicted_pose, target_pose):
    return torch.mean(torch.norm(predicted_pose - target_pose, dim=-1))


def _gradient_statistics(model):
    stats = {
        "gradient_tensors": 0,
        "finite_gradient_tensors": 0,
        "nonzero_gradient_tensors": 0,
        "max_abs_gradient": 0.0,
        "encoder_gradient_tensors": 0,
        "finite_encoder_gradient_tensors": 0,
        "nonzero_encoder_gradient_tensors": 0,
        "max_abs_encoder_gradient": 0.0,
    }
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise ValueError(f"Trainable parameter has no gradient: {name}")
        stats["gradient_tensors"] += 1
        finite = bool(torch.isfinite(parameter.grad).all().item())
        nonzero = bool((parameter.grad != 0).any().item())
        max_abs = float(parameter.grad.detach().abs().max().item()) if parameter.grad.numel() else 0.0
        if finite:
            stats["finite_gradient_tensors"] += 1
        if nonzero:
            stats["nonzero_gradient_tensors"] += 1
        if math.isfinite(max_abs):
            stats["max_abs_gradient"] = max(stats["max_abs_gradient"], max_abs)
        if _module_name(name) == "encoder":
            stats["encoder_gradient_tensors"] += 1
            if finite:
                stats["finite_encoder_gradient_tensors"] += 1
            if nonzero:
                stats["nonzero_encoder_gradient_tensors"] += 1
            if math.isfinite(max_abs):
                stats["max_abs_encoder_gradient"] = max(
                    stats["max_abs_encoder_gradient"], max_abs
                )
    stats["all_gradients_finite"] = (
        stats["finite_gradient_tensors"] == stats["gradient_tensors"]
    )
    stats["all_encoder_gradients_finite"] = (
        stats["finite_encoder_gradient_tensors"] == stats["encoder_gradient_tensors"]
    )
    return stats


def _make_datasets(freeze_config, pilot_root):
    split_assignments_path = Path(freeze_config).parent / "split_assignments.json"
    train_dataset = SinglePersonPiW3DDataset(
        pilot_root,
        pilot_root / "train_data" / "selected_train_ids.txt",
        expected_split="internal_train",
        split_assignments_path=split_assignments_path,
    )
    development_dataset = SinglePersonPiW3DDataset(
        pilot_root,
        pilot_root / "train_data" / "selected_development_ids.txt",
        expected_split="internal_development",
        split_assignments_path=split_assignments_path,
    )
    return train_dataset, development_dataset


def evaluate_train_mode_mpjpe(model, loader, device):
    was_training = model.training
    rng_state = _capture_rng_state()
    model.train()
    total_error = 0.0
    sample_count = 0
    try:
        with torch.no_grad():
            for batch in loader:
                csi = batch["csi"].to(device)
                pose = batch["pose"].to(device)
                predicted_pose, _ = model(csi)
                if predicted_pose.shape != pose.shape:
                    raise AssertionError(
                        f"Prediction shape {tuple(predicted_pose.shape)} does not match "
                        f"label shape {tuple(pose.shape)}"
                    )
                per_sample = torch.mean(
                    torch.norm(predicted_pose - pose, dim=-1), dim=(1, 2)
                )
                if not torch.isfinite(per_sample).all():
                    raise ValueError("Non-finite train-mode evaluation MPJPE")
                total_error += float(per_sample.sum().detach().cpu())
                sample_count += int(per_sample.numel())
    finally:
        _restore_rng_state(rng_state)
        model.train(was_training)
    if sample_count == 0:
        raise ValueError("Train-mode evaluation loader is empty")
    return total_error / sample_count, sample_count


def evaluate_eval_mode_mpjpe(model, loader, device):
    was_training = model.training
    rng_state = _capture_rng_state()
    model.eval()
    total_error = 0.0
    sample_count = 0
    try:
        with torch.no_grad():
            for batch in loader:
                csi = batch["csi"].to(device)
                pose = batch["pose"].to(device)
                predicted_pose, _ = model(csi)
                if predicted_pose.shape != pose.shape:
                    raise AssertionError(
                        f"Prediction shape {tuple(predicted_pose.shape)} does not match "
                        f"label shape {tuple(pose.shape)}"
                    )
                per_sample = torch.mean(
                    torch.norm(predicted_pose - pose, dim=-1), dim=(1, 2)
                )
                if not torch.isfinite(per_sample).all():
                    raise ValueError("Non-finite eval-mode development MPJPE")
                total_error += float(per_sample.sum().detach().cpu())
                sample_count += int(per_sample.numel())
    finally:
        _restore_rng_state(rng_state)
        model.train(was_training)
    if sample_count == 0:
        raise ValueError("Development evaluation loader is empty")
    return total_error / sample_count, sample_count


def collect_development_predictions(model, loader, device):
    was_training = model.training
    rng_state = _capture_rng_state()
    model.eval()
    predictions = []
    ground_truths = []
    total_error = 0.0
    sample_count = 0
    try:
        with torch.no_grad():
            for batch in loader:
                csi = batch["csi"].to(device)
                pose = batch["pose"].to(device)
                predicted_pose, _ = model(csi)
                if predicted_pose.shape != pose.shape:
                    raise AssertionError(
                        f"Prediction shape {tuple(predicted_pose.shape)} does not match "
                        f"label shape {tuple(pose.shape)}"
                    )
                per_sample = torch.mean(
                    torch.norm(predicted_pose - pose, dim=-1), dim=(1, 2)
                )
                if not torch.isfinite(per_sample).all():
                    raise ValueError("Non-finite development prediction")
                total_error += float(per_sample.sum().detach().cpu())
                sample_count += int(per_sample.numel())
                predictions.append(predicted_pose.detach().cpu().numpy().astype(np.float64))
                ground_truths.append(pose.detach().cpu().numpy().astype(np.float64))
    finally:
        _restore_rng_state(rng_state)
        model.train(was_training)
    return {
        "mpjpe_stored_units": total_error / sample_count,
        "sample_count": sample_count,
        "predictions": np.concatenate(predictions, axis=0),
        "ground_truths": np.concatenate(ground_truths, axis=0),
    }


def compute_mean_pose(train_dataset):
    poses = []
    for sample in train_dataset.samples:
        pose = train_dataset.read_pose(sample["keypoint_path"]).numpy().astype(np.float64)
        poses.append(pose[0])
    return np.mean(np.stack(poses, axis=0), axis=0), len(poses)


def constant_pose_mpjpe(dataset, mean_pose):
    total_error = 0.0
    sample_count = 0
    for sample in dataset.samples:
        pose = dataset.read_pose(sample["keypoint_path"]).numpy().astype(np.float64)
        error = float(np.mean(np.linalg.norm(mean_pose[None, :, :] - pose, axis=-1)))
        if not math.isfinite(error):
            raise ValueError("Non-finite constant-pose MPJPE")
        total_error += error
        sample_count += 1
    return total_error / sample_count, sample_count


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
    summary["all_encoder_parameters_changed"] = (
        summary["changed_encoder_parameters"] == summary["encoder_parameters"]
    )
    return summary


def state_dicts_equal(first_model, second_model):
    first_state = first_model.state_dict()
    second_state = second_model.state_dict()
    if set(first_state) != set(second_state):
        return False
    return all(torch.equal(first_state[name], second_state[name]) for name in first_state)


def run_baseline(artifact_root, freeze_config, pilot_root, device_name):
    artifact_root = Path(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    environment_validation = validate_process_environment()
    if not environment_validation["valid"]:
        raise RuntimeError("; ".join(environment_validation["failures"]))

    load_runtime_modules()
    deterministic_settings = configure_deterministic_runtime()
    freeze_config, pilot_root = _resolve_source_paths(freeze_config, pilot_root)
    verification = verify_frozen_inputs(freeze_config, pilot_root)
    _write_json(artifact_root / "input_verification.json", verification)
    _write_json(artifact_root / "split_verification.json", verification["split_verification"])
    freeze_path, freeze_record = write_baseline_freeze(artifact_root, verification)
    config = freeze_record["configuration"]

    device = _resolve_device(device_name)
    environment = capture_environment(device, deterministic_settings)
    _write_json(artifact_root / "environment.json", environment)
    _seed_everything(config["seed"])

    train_dataset, development_dataset = _make_datasets(freeze_config, pilot_root)
    if len(train_dataset) != config["data"]["expected_train_count"]:
        raise AssertionError("Loaded training dataset does not match the frozen count")
    if len(development_dataset) != config["data"]["expected_development_count"]:
        raise AssertionError("Loaded development dataset does not match the frozen count")
    if set(train_dataset.sample_ids) & set(development_dataset.sample_ids):
        raise AssertionError("Loaded train/development datasets overlap")

    generator = torch.Generator().manual_seed(config["determinism"]["loader_generator_seed"])
    train_loader = make_single_person_dataloader(
        train_dataset,
        is_training=True,
        generator=generator,
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    train_eval_loader = make_single_person_dataloader(
        train_dataset,
        is_training=False,
        generator=torch.Generator().manual_seed(config["determinism"]["loader_generator_seed"]),
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    development_loader = make_single_person_dataloader(
        development_dataset,
        is_training=False,
        generator=torch.Generator().manual_seed(config["determinism"]["loader_generator_seed"]),
        batch_size=config["training"]["development_batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    model = build_step4b_r_model(config, device)
    model.train()
    optimizer = make_adamw_optimizer(model, config)
    audit = parameter_audit(model, optimizer)
    if not audit["optimizer_contains_exact_trainable_parameter_set"]:
        raise AssertionError("AdamW does not contain exactly the intended trainable parameter set")
    if audit["parameter_counts"]["trainable_named_parameters"] != EXPECTED_TRAINABLE_NAMED_PARAMETERS:
        raise AssertionError("Step-4B-R trainable parameter audit failed")
    if audit["module_counts"]["encoder"]["trainable_named_parameters"] != EXPECTED_ENCODER_NAMED_PARAMETERS:
        raise AssertionError("Step-4B-R encoder is not fully trainable")
    if audit["module_counts"]["pose_decoder"]["trainable_named_parameters"] != EXPECTED_DECODER_NAMED_PARAMETERS:
        raise AssertionError("Step-4B-R pose decoder is not fully trainable")
    _write_json(artifact_root / "parameter_audit.json", audit)

    initial_model_state = {
        name: value.detach().clone() for name, value in model.state_dict().items()
    }
    initial_train_mpjpe, initial_train_count = evaluate_train_mode_mpjpe(
        model, train_eval_loader, device
    )
    initial_development_mpjpe, initial_development_count = evaluate_eval_mode_mpjpe(
        model, development_loader, device
    )
    initial_evaluation = {
        "train_mode_training_mpjpe_stored_units": initial_train_mpjpe,
        "training_samples": initial_train_count,
        "eval_mode_development_mpjpe_stored_units": initial_development_mpjpe,
        "development_samples": initial_development_count,
        "all_values_finite": all(
            math.isfinite(value)
            for value in (initial_train_mpjpe, initial_development_mpjpe)
        ),
    }
    _write_json(artifact_root / "initial_evaluation.json", initial_evaluation)

    curves = []
    trajectory = []
    optimizer_steps = 0
    start_time = time.perf_counter()

    for epoch in range(1, config["training"]["max_epochs"] + 1):
        epoch_start = time.perf_counter()
        model.train()
        total_training_error = 0.0
        training_samples = 0
        batches = 0
        for batch in train_loader:
            csi = batch["csi"].to(device)
            pose = batch["pose"].to(device)
            optimizer.zero_grad(set_to_none=True)
            predicted_pose, _ = model(csi)
            if predicted_pose.shape != pose.shape:
                raise AssertionError(
                    f"Prediction shape {tuple(predicted_pose.shape)} does not match "
                    f"label shape {tuple(pose.shape)}"
                )
            loss = _mean_per_joint_error(predicted_pose, pose)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite training loss in epoch {epoch}")
            loss.backward()
            gradient_stats = _gradient_statistics(model)
            if not gradient_stats["all_gradients_finite"]:
                raise ValueError(f"Non-finite gradients in epoch {epoch}")
            optimizer.step()
            optimizer_steps += 1
            batches += 1
            per_sample = torch.mean(torch.norm(predicted_pose - pose, dim=-1), dim=(1, 2))
            total_training_error += float(per_sample.sum().detach().cpu())
            training_samples += int(per_sample.numel())
            trajectory.append(
                {
                    "epoch": epoch,
                    "batch": batches,
                    "optimizer_step": optimizer_steps,
                    "samples": int(csi.shape[0]),
                    "pre_update_loss_stored_units": float(loss.detach().cpu()),
                    **gradient_stats,
                }
            )

        train_mpjpe, train_eval_samples = evaluate_train_mode_mpjpe(
            model, train_eval_loader, device
        )
        development_mpjpe, development_eval_samples = evaluate_eval_mode_mpjpe(
            model, development_loader, device
        )
        epoch_record = {
            "epoch": epoch,
            "batches": batches,
            "optimizer_steps_in_epoch": batches,
            "cumulative_optimizer_steps": optimizer_steps,
            "train_update_mpjpe_stored_units": total_training_error / training_samples,
            "train_update_samples": training_samples,
            "train_mode_training_mpjpe_stored_units": train_mpjpe,
            "train_mode_training_samples": train_eval_samples,
            "eval_mode_development_mpjpe_stored_units": development_mpjpe,
            "development_samples": development_eval_samples,
            "all_training_losses_finite": all(
                math.isfinite(record["pre_update_loss_stored_units"])
                for record in trajectory
                if record["epoch"] == epoch
            ),
            "all_training_gradients_finite": all(
                record["all_gradients_finite"]
                for record in trajectory
                if record["epoch"] == epoch
            ),
            "runtime_seconds": time.perf_counter() - epoch_start,
        }
        curves.append(epoch_record)
        _write_json(
            artifact_root / "learning_curves.json",
            {"initial": initial_evaluation, "epochs": curves},
        )
        print(json.dumps(epoch_record, indent=2, sort_keys=True))

    final_epoch = curves[-1]
    final_state_path = artifact_root / "final_state_dict.pt"
    torch.save(model.state_dict(), final_state_path)
    reloaded_model = build_step4b_r_model(config, device)
    reloaded_model.load_state_dict(
        torch.load(final_state_path, map_location=device, weights_only=True)
    )
    reloaded_model.to(device)
    checkpoint_state_dict_exact = state_dicts_equal(model, reloaded_model)

    _seed_everything(config["seed"])
    original_final_train_rng_state = _capture_rng_state()
    final_model_train_mpjpe, final_model_train_samples = evaluate_train_mode_mpjpe(
        model, train_eval_loader, device
    )
    _restore_rng_state(original_final_train_rng_state)
    reloaded_final_train_mpjpe, reloaded_final_train_samples = evaluate_train_mode_mpjpe(
        reloaded_model, train_eval_loader, device
    )
    _restore_rng_state(original_final_train_rng_state)

    reloaded_development_mpjpe, reloaded_development_samples = evaluate_eval_mode_mpjpe(
        reloaded_model, development_loader, device
    )
    if not math.isfinite(reloaded_development_mpjpe):
        raise ValueError("Non-finite reloaded development MPJPE")
    if not checkpoint_state_dict_exact:
        raise AssertionError("Reloaded final state_dict differs from the trained final state_dict")
    if not math.isclose(
        reloaded_development_mpjpe,
        final_epoch["eval_mode_development_mpjpe_stored_units"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Reloaded development MPJPE does not match the final epoch measurement")
    if not math.isclose(
        reloaded_final_train_mpjpe,
        final_model_train_mpjpe,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Reloaded train-mode MPJPE does not match the final trained model")

    development_collection = collect_development_predictions(
        reloaded_model, development_loader, device
    )
    mean_pose, mean_pose_source_count = compute_mean_pose(train_dataset)
    mean_pose_train_mpjpe, mean_pose_train_samples = constant_pose_mpjpe(
        train_dataset, mean_pose
    )
    mean_pose_development_mpjpe, mean_pose_development_samples = constant_pose_mpjpe(
        development_dataset, mean_pose
    )
    mean_pose_comparison = {
        "mean_pose_source": "internal-training labels only",
        "mean_pose_source_count": mean_pose_source_count,
        "mean_pose_train_mpjpe_stored_units": mean_pose_train_mpjpe,
        "mean_pose_train_samples": mean_pose_train_samples,
        "mean_pose_development_mpjpe_stored_units": mean_pose_development_mpjpe,
        "mean_pose_development_samples": mean_pose_development_samples,
        "final_model_train_mpjpe_stored_units": final_model_train_mpjpe,
        "final_model_train_samples": final_model_train_samples,
        "final_model_development_mpjpe_stored_units": reloaded_development_mpjpe,
        "final_model_development_samples": reloaded_development_samples,
        "final_model_beats_mean_predictor_on_training": (
            final_model_train_mpjpe < mean_pose_train_mpjpe
        ),
        "final_model_beats_mean_predictor_on_development": (
            reloaded_development_mpjpe < mean_pose_development_mpjpe
        ),
        "distance_units": "stored units",
        "unit_conversion": "none",
    }
    _write_json(artifact_root / "mean_pose_comparison.json", mean_pose_comparison)

    np.savez(
        artifact_root / "development_predictions.npz",
        sample_ids=np.array(development_dataset.sample_ids),
        ground_truths=development_collection["ground_truths"],
        model_predictions=development_collection["predictions"],
        mean_pose_predictions=np.stack(
            [mean_pose[None, :, :] for _ in range(len(development_dataset))], axis=0
        ),
        mean_pose=np.stack([mean_pose], axis=0),
    )

    parameter_changes = summarize_parameter_changes(model, initial_model_state)
    _write_json(artifact_root / "parameter_changes.json", parameter_changes)
    _write_json(artifact_root / "training_trajectory.json", {"records": trajectory})

    all_training_losses_finite = all(
        math.isfinite(record["pre_update_loss_stored_units"]) for record in trajectory
    )
    all_training_gradients_finite = all(
        record["all_gradients_finite"] for record in trajectory
    )
    if not all_training_losses_finite or not all_training_gradients_finite:
        raise AssertionError("Training completed without finite-loss/finite-gradient evidence")
    if optimizer_steps != len(trajectory):
        raise AssertionError("Optimizer-step evidence does not match the training trajectory")
    if not parameter_changes["all_trainable_parameters_changed"]:
        raise AssertionError("Training completed without evidence that all trainable parameters changed")
    if not parameter_changes["all_encoder_parameters_changed"]:
        raise AssertionError("Training completed without evidence that encoder parameters changed")

    runtime_seconds = time.perf_counter() - start_time
    post_run_protected_records = verify_protected_records()
    result = {
        "baseline_version": R_VERSION,
        "status": "BASELINE_COMPLETE",
        "acceptance": "bounded protocol completion and measurement reporting, not a predetermined accuracy threshold",
        "freeze_record": str(freeze_path),
        "device": str(device),
        "runtime_seconds": runtime_seconds,
        "epochs_completed": len(curves),
        "optimizer_steps": optimizer_steps,
        "seed": config["seed"],
        "model": config["model"],
        "training": config["training"],
        "deterministic_execution": {
            "required_environment": environment_validation,
            "configured_settings": deterministic_settings,
            "torch_deterministic_algorithms_enabled": bool(
                torch.are_deterministic_algorithms_enabled()
            ),
            "torch_deterministic_warn_only_enabled": bool(
                torch.is_deterministic_algorithms_warn_only_enabled()
            ),
            "cudnn_deterministic": bool(torch.backends.cudnn.deterministic),
            "cudnn_benchmark": bool(torch.backends.cudnn.benchmark),
            "silent_deterministic_disable": False,
        },
        "split_verification": verification["split_verification"],
        "train_dataset_size": len(train_dataset),
        "development_dataset_size": len(development_dataset),
        "initial": initial_evaluation,
        "learning_curves": curves,
        "training_checks": {
            "all_training_losses_finite": all_training_losses_finite,
            "all_training_gradients_finite": all_training_gradients_finite,
            "optimizer_contains_exact_trainable_parameter_set": audit[
                "optimizer_contains_exact_trainable_parameter_set"
            ],
            "optimizer_parameter_references": audit["optimizer"]["parameter_references"],
            "optimizer_steps": optimizer_steps,
            "all_trainable_parameters_changed": parameter_changes[
                "all_trainable_parameters_changed"
            ],
            "all_encoder_parameters_changed": parameter_changes[
                "all_encoder_parameters_changed"
            ],
            "trainable_named_parameters": audit["parameter_counts"]["trainable_named_parameters"],
            "trainable_parameters": audit["parameter_counts"]["trainable_parameters"],
        },
        "final_model": {
            "train_mode_training_mpjpe_stored_units": final_model_train_mpjpe,
            "train_mode_training_samples": final_model_train_samples,
            "eval_mode_development_mpjpe_stored_units": reloaded_development_mpjpe,
            "development_samples": reloaded_development_samples,
            "checkpoint_path": str(final_state_path),
            "checkpoint_format": "state_dict",
            "checkpoint_state_dict_exact_after_reload": checkpoint_state_dict_exact,
            "reloaded_train_mode_training_mpjpe_stored_units": reloaded_final_train_mpjpe,
            "reloaded_train_mode_training_samples": reloaded_final_train_samples,
        },
        "mean_pose": {
            "source": "internal-training labels only",
            "source_count": mean_pose_source_count,
            "train_mpjpe_stored_units": mean_pose_train_mpjpe,
            "development_mpjpe_stored_units": mean_pose_development_mpjpe,
        },
        "model_versus_mean_pose": {
            "beats_mean_predictor_on_training": mean_pose_comparison[
                "final_model_beats_mean_predictor_on_training"
            ],
            "beats_mean_predictor_on_development": mean_pose_comparison[
                "final_model_beats_mean_predictor_on_development"
            ],
            "training_difference_model_minus_mean_stored_units": (
                final_model_train_mpjpe - mean_pose_train_mpjpe
            ),
            "development_difference_model_minus_mean_stored_units": (
                reloaded_development_mpjpe - mean_pose_development_mpjpe
            ),
        },
        "distance_units": "stored units",
        "coordinate_order": "unchanged; no axis reordering performed",
        "unit_conversion": "none; no millimetre conversion is claimed by this baseline",
        "label_access": config["label_access"],
        "official_test_used": False,
        "source_step4b_status": "BLOCKED",
        "protected_records_unchanged_after_run": post_run_protected_records["all_unchanged"],
        "protected_record_count": post_run_protected_records["record_count"],
    }
    _write_json(artifact_root / "result.json", result)

    decision = {
        "status": "BASELINE_COMPLETE",
        "reason": (
            "The bounded deterministic Step-4B-R protocol completed and all required measurements "
            "were recorded without a predetermined accuracy threshold. Original Step-4B remains "
            "BLOCKED and was not rerun or retroactively changed."
        ),
        "final_model_train_mpjpe_stored_units": final_model_train_mpjpe,
        "final_model_development_mpjpe_stored_units": reloaded_development_mpjpe,
        "mean_pose_train_mpjpe_stored_units": mean_pose_train_mpjpe,
        "mean_pose_development_mpjpe_stored_units": mean_pose_development_mpjpe,
        "beats_mean_predictor_on_training": mean_pose_comparison[
            "final_model_beats_mean_predictor_on_training"
        ],
        "beats_mean_predictor_on_development": mean_pose_comparison[
            "final_model_beats_mean_predictor_on_development"
        ],
        "deterministic_execution_enabled": True,
        "result": str(artifact_root / "result.json"),
    }
    _write_json(artifact_root / "decision.json", decision)
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"step4b_r_result={artifact_root / 'result.json'}")
    print(f"step4b_r_decision={artifact_root / 'decision.json'}")
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run the frozen Step-4B-R bounded deterministic baseline.")
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--freeze-config", type=Path, default=None)
    parser.add_argument("--pilot-root", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        return run_baseline(args.artifact_root, args.freeze_config, args.pilot_root, args.device)
    except Exception as exc:
        error_text = str(exc)
        deterministic_enabled = (
            torch is not None
            and hasattr(torch, "are_deterministic_algorithms_enabled")
            and bool(torch.are_deterministic_algorithms_enabled())
        )
        blocker = deterministic_enabled and "deterministic" in error_text.lower()
        artifact = {
            "baseline_version": R_VERSION,
            "status": "DETERMINISTIC_BLOCKER" if blocker else "RUN_ERROR",
            "error_type": type(exc).__name__,
            "error": error_text,
            "deterministic_algorithms_enabled_at_error": deterministic_enabled,
            "traceback": traceback.format_exc(),
        }
        output_name = "deterministic_blocker.json" if blocker else "run_error.json"
        try:
            _write_json(Path(args.artifact_root) / output_name, artifact)
        except Exception:
            pass
        print(json.dumps(artifact, indent=2, sort_keys=True))
        return 4 if blocker else 5


if __name__ == "__main__":
    raise SystemExit(main())
