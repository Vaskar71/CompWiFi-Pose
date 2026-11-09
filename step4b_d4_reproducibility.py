import ast
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from step4b_d3_diagnostic import EXPECTED_PROTECTED_RECORDS


D4_VERSION = "compwifi-step4b-d4-v1"
D4_TYPE = "bounded reproducibility audit of the Step-4B-D2 trainable-encoder procedure"
D2_RUNNER_PATH = Path("step4b_d2_diagnostic.py")
D3_RUNNER_PATH = Path("step4b_d3_diagnostic.py")
D4_ROOT = Path("data/_artifacts/step4b_d4")
REPEAT_A_ROOT = D4_ROOT / "repeat_a"
REPEAT_B_ROOT = D4_ROOT / "repeat_b"
D2_SEED = 0
D2_STEPS = 200
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
D2_MPJPE_BY_STEP = {
    0: 4.920130252838135,
    100: 0.5151628255844116,
    200: 0.46943652629852295,
}
D3_MPJPE_BY_STEP = {
    0: 4.920130252838135,
    100: 0.40826788544654846,
    200: 0.3430270552635193,
}
TRAJECTORY_FIELDS = [
    "step",
    "pre_update_loss_stored_units",
    "pre_update_mpjpe_stored_units",
    "gradient_tensors",
    "finite_gradient_tensors",
    "max_abs_gradient",
    "encoder_gradient_tensors",
    "finite_encoder_gradient_tensors",
    "nonzero_encoder_gradient_tensors",
    "max_abs_encoder_gradient",
    "all_gradients_finite",
    "all_encoder_gradients_finite",
    "all_encoder_gradients_nonzero",
]
RESULT_COMPARISON_FIELDS = [
    "device",
    "seed",
    "batch_size",
    "batch_sample_ids",
    "optimizer_steps",
    "optimizer",
    "learning_rate",
    "momentum",
    "loss",
    "all_losses_finite",
    "all_gradients_finite",
    "all_encoder_gradients_finite",
    "all_encoder_gradients_nonzero",
    "initial_mpjpe_stored_units",
    "step_100_mpjpe_stored_units",
    "final_mpjpe_stored_units",
    "final_to_initial_mpjpe_ratio",
]
PROTECTED_RECORDS = dict(EXPECTED_PROTECTED_RECORDS)
PROTECTED_RECORDS.update(
    {
        "step4b_d3_diagnostic.py": "5a5278f760ee699106b0752a2282d8179a7cf76b80a97bf29d3daf5ebf5d2624",
        "docs/STEP_3_GATE_GOAL.md": "adf24ef4ddedb6a86be78709357984d8435ed78d3116a5632ab148522df303b4",
        "docs/STEP_4A_GOAL.md": "54992d450fa5190403db2dc7ec114a15af7d2c8c9cf048f45141e58f147d9168",
        "docs/STEP_4A_REPORT.md": "ab80055113070931634d95842d88cb5c70780b0f9db2fb095a4081e627cb9411",
        "docs/STEP_4B_REPORT.md": "fe623b9d1611a4ecb53d182303435d99d2aa940a086eb81573f1d6e430a84a21",
        "docs/STEP_4B_D1_REPORT.md": "5111d8cd31400e649cfb2d6561df20576e3650ad91311fae1ace85892ca4406c",
        "docs/STEP_4B_D2_REPORT.md": "bec140e0beea5cb999f54eac4f20a846c5e1e28ddf6f5ca613dd2c3ead3a6b5d",
        "docs/STEP_4B_D3_PROTOCOL.md": "57e6d4ef83ec06795206f9f83c6cc495af85c4b5dbd738ce143d80b9ceb0c08a",
        "docs/STEP_4B_D3_REPORT.md": "bee834aaa46053c495451548cf22a48f302ccd0bd4036458d161c538cdfc6b1d",
        "docs/DATA_AUDIT.md": "71ef69b472cdbd4173a2362d1707642f58cfc8398768430efff154ef48b4e2d3",
        "docs/POSE_SPLIT_METRIC_AUDIT.md": "83d0a61f369e1b70aaa6802d27a0a8ec76a94edb6c00a1012a164e05d530e40d",
        "docs/PROJECT_CONTEXT.md": "79382973fe9ed109eaba82478f055295a8b4925335a581e4bbdd4303ce1e8091",
        "docs/RESEARCH_PROTOCOL.md": "683eb9b5731bf97740c7501a29b90b3f8e24f7ea15bde5f55d65809c1d554214",
        "data/_artifacts/step4b_d3/input_verification.json": "a5289ed0c273b9fdb7c92657288b985a2cd1df184933fe5af93b770948dab800",
        "data/_artifacts/step4b_d3/step4b_d3_freeze.json": "aa1770e69cb31b456f3a5a40eeb0f4c6a430145f80cc737ddae7058ad9c696fb",
        "data/_artifacts/step4b_d3/step4b_d3_reproduction_mismatch.json": "85b2b91df13f1a6ccf39971279423f77848b96c161f8cff5df4ba52b13ebe3d8",
    }
)
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
FORBIDDEN_DETERMINISM_CALLS = {
    "torch.use_deterministic_algorithms",
    "torch.set_deterministic_algorithms",
    "torch.set_deterministic_debug_mode",
    "torch.are_deterministic_algorithms_enabled",
}


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def verify_protected_records(expected_records=None):
    expected_records = PROTECTED_RECORDS if expected_records is None else expected_records
    checks = {}
    for path, expected in expected_records.items():
        actual = _sha256_file(path)
        checks[path] = {
            "actual": actual,
            "expected": expected,
            "unchanged": actual == expected,
        }
    return {
        "record_count": len(checks),
        "all_unchanged": all(check["unchanged"] for check in checks.values()),
        "checks": checks,
    }


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


def capture_environment():
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
        "audit_version": D4_VERSION,
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
        "random_seed_procedure": {
            "frozen_seed": D2_SEED,
            "seed_call": "random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed) when CUDA is available",
            "ordered_loader_generator_seed": D2_SEED,
        },
    }


def _parse_source(path):
    return ast.parse(Path(path).read_text(encoding="utf-8"), filename=str(path))


def _called_names(tree):
    names = set()

    def visit(node):
        if isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                names.add(function.id)
            elif isinstance(function, ast.Attribute):
                value = function.value
                parts = [function.attr]
                while isinstance(value, ast.Attribute):
                    parts.append(value.attr)
                    value = value.value
                if isinstance(value, ast.Name):
                    parts.append(value.id)
                names.add(".".join(reversed(parts)))
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return names


def _d3_alias_map(tree):
    aliases = {}
    for node in tree.body:
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
            if isinstance(node.value, ast.Name):
                aliases[node.targets[0].id] = node.value.id
    return aliases


def _d2_imports_from_d3(tree):
    imported = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module == "step4b_d2_diagnostic":
            for alias in node.names:
                imported.add(alias.asname or alias.name)
    return imported


def static_execution_path_comparison():
    d2_tree = _parse_source(D2_RUNNER_PATH)
    d3_tree = _parse_source(D3_RUNNER_PATH)
    d2_source = Path(D2_RUNNER_PATH).read_text(encoding="utf-8")
    d3_source = Path(D3_RUNNER_PATH).read_text(encoding="utf-8")
    d2_calls = _called_names(d2_tree)
    d3_calls = _called_names(d3_tree)
    d3_aliases = _d3_alias_map(d3_tree)
    required_d2_imports = {
        "EXPECTED_BATCH_IDS",
        "build_d2_model",
        "make_sgd_optimizer",
        "parameter_audit",
        "summarize_parameter_changes",
        "_capture_checkpoint",
        "_capture_rng_state",
        "_clone_model_state",
        "_restore_model_state",
        "_restore_rng_state",
        "_resolve_source_paths",
    }
    d2_loop_fragments = [
        "optimizer.zero_grad()",
        "predicted_pose, _ = model(csi)",
        "loss = torch.mean(torch.norm(predicted_pose - pose, dim=-1))",
        "loss.backward()",
        "optimizer.step()",
        "pre_update_loss_stored_units",
        "pre_update_mpjpe_stored_units",
        "_clone_model_state(model)",
        "_capture_rng_state()",
        "_capture_checkpoint(model, csi, pose, batch_sample_ids, step)",
        "_restore_model_state(model, model_state)",
        "_restore_rng_state(rng_state)",
    ]
    d3_loop_fragments = [
        "optimizer.zero_grad()",
        "predicted_pose, _ = model(csi)",
        "loss = torch.mean(torch.norm(predicted_pose - pose, dim=-1))",
        "loss.backward()",
        "optimizer.step()",
        "pre_update_loss_stored_units",
        "pre_update_mpjpe_stored_units",
        "_clone_model_state(model)",
        "_capture_rng_state()",
        "_capture_checkpoint(model, csi, pose, batch_sample_ids, step)",
        "_restore_model_state(model, model_state)",
        "_restore_rng_state(rng_state)",
    ]
    shared_checks = {
        "same_frozen_seed_call": "_seed_everything(config[\"seed\"])" in d2_source
        and "_seed_everything(config[\"seed\"])" in d3_source,
        "same_train_dataset_class": "SinglePersonPiW3DDataset(" in d2_source
        and "SinglePersonPiW3DDataset(" in d3_source,
        "same_ordered_loader_factory": "make_single_person_dataloader(" in d2_source
        and "make_single_person_dataloader(" in d3_source,
        "same_non_shuffling_fixed_batch": "is_training=False" in d2_source
        and "is_training=False" in d3_source,
        "same_loader_generator_seed": "torch.Generator().manual_seed(config[\"seed\"])" in d2_source
        and "torch.Generator().manual_seed(config[\"seed\"])" in d3_source,
        "same_train_mode": "model.train()" in d2_source and "model.train()" in d3_source,
        "same_parameter_audit": "parameter_audit(model, optimizer)" in d2_source
        and "parameter_audit(model, optimizer)" in d3_source,
        "same_checkpoint_helper": "_capture_checkpoint(model, csi, pose, batch_sample_ids, step)" in d2_source
        and "_capture_checkpoint(model, csi, pose, batch_sample_ids, step)" in d3_source,
        "d2_optimizer_zero_grad_forward_loss_backward_step": all(
            fragment in d2_source for fragment in d2_loop_fragments
        ),
        "d3_optimizer_zero_grad_forward_loss_backward_step": all(
            fragment in d3_source for fragment in d3_loop_fragments
        ),
        "no_deterministic_algorithm_enable_call_in_d2": not bool(
            d2_calls & {"torch.use_deterministic_algorithms", "torch.set_deterministic_algorithms"}
        ),
        "no_deterministic_algorithm_enable_call_in_d3": not bool(
            d3_calls & {"torch.use_deterministic_algorithms", "torch.set_deterministic_algorithms"}
        ),
        "no_cudnn_deterministic_assignment": "torch.backends.cudnn.deterministic = True" not in d2_source
        and "torch.backends.cudnn.deterministic = True" not in d3_source,
    }
    differences = [
        {
            "area": "wrapper verification and freeze",
            "description": "D3 additionally hashes protected D2 records and requires the D3 protocol document before writing its own freeze.",
        },
        {
            "area": "optimizer-step budget",
            "description": "D2 uses 200 steps; D3 defines 800 steps and checkpoints at 0, 100, 200, and 800.",
        },
        {
            "area": "gradient audit implementation",
            "description": "D3 duplicates the D2 gradient-statistics logic in a local helper instead of importing D2's helper; the finite/nonzero accounting is equivalent when gradients are finite.",
        },
        {
            "area": "post-checkpoint control flow",
            "description": "After the step-200 checkpoint, D3 runs an exact D2 reproduction gate and either stops before step 201 or continues; D2 serializes its completed 200-step artifacts.",
        },
        {
            "area": "artifact fields",
            "description": "D3 adds D2 reproduction, 800-step, late-trajectory, and prior-comparison fields that D2 does not have.",
        },
    ]
    unchanged_semantics = [
        "model construction",
        "seed use",
        "loader order",
        "forward/backward call order",
        "loss calculation",
        "optimizer construction and updates",
        "train-mode checkpoint measurements at steps 0, 100, and 200",
        "optimizer coverage audit",
        "parameter-change summary",
        "batch size",
        "learning rate",
        "momentum",
        "device-selection mode",
    ]
    return {
        "audit_version": D4_VERSION,
        "source_hashes": {
            str(D2_RUNNER_PATH): _sha256_file(D2_RUNNER_PATH),
            str(D3_RUNNER_PATH): _sha256_file(D3_RUNNER_PATH),
            "model/model.py": _sha256_file("model/model.py"),
        },
        "d2_imports_used_by_d3": sorted(_d2_imports_from_d3(d3_tree)),
        "d3_aliases": d3_aliases,
        "same_model_builder_alias": d3_aliases.get("build_d3_model") == "build_d2_model",
        "same_optimizer_builder_alias": d3_aliases.get("make_d3_optimizer") == "make_sgd_optimizer",
        "required_d2_imports_present": sorted(required_d2_imports),
        "all_required_d2_imports_present": required_d2_imports.issubset(
            _d2_imports_from_d3(d3_tree)
        ),
        "shared_execution_checks": shared_checks,
        "all_shared_execution_checks_passed": all(shared_checks.values()),
        "unchanged_semantics_through_step_200": unchanged_semantics,
        "observed_code_differences": differences,
        "determinism_source_scan": {
            "forbidden_determinism_calls": sorted(FORBIDDEN_DETERMINISM_CALLS),
            "called_in_d2": sorted(d2_calls & FORBIDDEN_DETERMINISM_CALLS),
            "called_in_d3": sorted(d3_calls & FORBIDDEN_DETERMINISM_CALLS),
        },
    }


def write_static_execution_comparison(path):
    return _write_json(path, static_execution_path_comparison())


def write_environment_record(path):
    return _write_json(path, capture_environment())


def _trajectory_from_root(artifact_root):
    return _load_json(Path(artifact_root) / "loss_trajectory.json")["records"]


def _result_from_root(artifact_root):
    return _load_json(Path(artifact_root) / "step4b_d2_result.json")


def _first_record_divergence(reference_records, actual_records):
    if len(reference_records) != len(actual_records):
        return {
            "step": None,
            "field": "record_count",
            "reference": len(reference_records),
            "actual": len(actual_records),
        }
    for reference, actual in zip(reference_records, actual_records):
        for field in TRAJECTORY_FIELDS:
            if reference.get(field) != actual.get(field):
                return {
                    "step": reference.get("step"),
                    "field": field,
                    "reference": reference.get(field),
                    "actual": actual.get(field),
                }
    return None


def compare_trajectories(reference_root, actual_root):
    reference_records = _trajectory_from_root(reference_root)
    actual_records = _trajectory_from_root(actual_root)
    first_divergence = _first_record_divergence(reference_records, actual_records)
    first_loss_divergence = None
    for reference, actual in zip(reference_records, actual_records):
        if reference["pre_update_loss_stored_units"] != actual["pre_update_loss_stored_units"]:
            first_loss_divergence = {
                "step": reference["step"],
                "reference": reference["pre_update_loss_stored_units"],
                "actual": actual["pre_update_loss_stored_units"],
            }
            break
    return {
        "reference_root": str(reference_root),
        "actual_root": str(actual_root),
        "reference_record_count": len(reference_records),
        "actual_record_count": len(actual_records),
        "trajectory_exact": first_divergence is None,
        "first_recorded_divergence": first_divergence,
        "first_recorded_loss_divergence": first_loss_divergence,
    }


def compare_result_measurements(reference_root, actual_root):
    reference = _result_from_root(reference_root)
    actual = _result_from_root(actual_root)
    field_differences = {}
    for field in RESULT_COMPARISON_FIELDS:
        if reference.get(field) != actual.get(field):
            field_differences[field] = {
                "reference": reference.get(field),
                "actual": actual.get(field),
            }
    checkpoints = {}
    for step, field in ((0, "initial_mpjpe_stored_units"), (100, "step_100_mpjpe_stored_units"), (200, "final_mpjpe_stored_units")):
        checkpoints[str(step)] = {
            "reference": reference.get(field),
            "actual": actual.get(field),
            "exact_match": reference.get(field) == actual.get(field),
        }
    return {
        "reference_root": str(reference_root),
        "actual_root": str(actual_root),
        "checkpoint_mpjpe": checkpoints,
        "all_checkpoint_mpjpe_exact": all(value["exact_match"] for value in checkpoints.values()),
        "compared_result_fields": list(RESULT_COMPARISON_FIELDS),
        "field_differences": field_differences,
        "result_fields_exact": not field_differences,
    }


def classify_repeat_evidence(d2_root, repeat_a_root, repeat_b_root):
    a_vs_b_result = compare_result_measurements(repeat_a_root, repeat_b_root)
    b_vs_a_result = compare_result_measurements(repeat_b_root, repeat_a_root)
    a_vs_d2 = compare_result_measurements(d2_root, repeat_a_root)
    b_vs_d2 = compare_result_measurements(d2_root, repeat_b_root)
    a_vs_b_trajectory = compare_trajectories(repeat_a_root, repeat_b_root)
    a_vs_d2_trajectory = compare_trajectories(d2_root, repeat_a_root)
    b_vs_d2_trajectory = compare_trajectories(d2_root, repeat_b_root)
    repeats_match_each_other = (
        a_vs_b_result["all_checkpoint_mpjpe_exact"]
        and a_vs_b_result["result_fields_exact"]
        and a_vs_b_trajectory["trajectory_exact"]
    )
    both_match_historical_d2 = (
        a_vs_d2["all_checkpoint_mpjpe_exact"]
        and a_vs_d2["result_fields_exact"]
        and a_vs_d2_trajectory["trajectory_exact"]
        and b_vs_d2["all_checkpoint_mpjpe_exact"]
        and b_vs_d2["result_fields_exact"]
        and b_vs_d2_trajectory["trajectory_exact"]
    )
    if both_match_historical_d2:
        classification = "both repeats exactly reproduce historical D2"
    elif not repeats_match_each_other:
        classification = "repeats differ from each other"
    elif not both_match_historical_d2:
        classification = "repeats match each other but not historical D2"
    else:
        classification = "another explicitly described outcome"
    return {
        "classification": classification,
        "repeats_match_each_other": repeats_match_each_other,
        "both_match_historical_d2": both_match_historical_d2,
        "repeat_a_vs_repeat_b": {
            "result_a_reference_b_actual": a_vs_b_result,
            "result_b_reference_a_actual": b_vs_a_result,
            "trajectory_a_reference_b_actual": a_vs_b_trajectory,
        },
        "repeat_a_vs_historical_d2": {
            "result": a_vs_d2,
            "trajectory": a_vs_d2_trajectory,
        },
        "repeat_b_vs_historical_d2": {
            "result": b_vs_d2,
            "trajectory": b_vs_d2_trajectory,
        },
    }


def write_repeat_comparison(path, d2_root, repeat_a_root, repeat_b_root):
    return _write_json(path, classify_repeat_evidence(d2_root, repeat_a_root, repeat_b_root))
