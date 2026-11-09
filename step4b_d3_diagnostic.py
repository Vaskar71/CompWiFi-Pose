import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from feeder.piw3d_single_person import SinglePersonPiW3DDataset, make_single_person_dataloader
from step4b_config import validate_step4b_config
from step4b_single_person_pilot import _resolve_device, _seed_everything
from step4b_d2_diagnostic import (
    D2_VERSION,
    EXPECTED_BATCH_IDS,
    build_d2_model,
    make_sgd_optimizer,
    parameter_audit,
    summarize_parameter_changes,
    verify_frozen_inputs as verify_d2_frozen_inputs,
    _capture_checkpoint,
    _capture_rng_state,
    _clone_model_state,
    _load_json,
    _restore_model_state,
    _restore_rng_state,
    _resolve_source_paths,
    _sha256_file,
    _source_file_record,
    _write_json,
)


D3_VERSION = "compwifi-step4b-d3-v1"
D3_TYPE = "controlled trainable-encoder extended-step diagnostic after failed frozen Step-4B tiny-overfit gate"
EXPECTED_D2_STEPS = 200
D3_STEPS = 800
D3_BATCH_IDS = list(EXPECTED_BATCH_IDS)
D3_CHECKPOINT_STEPS = (0, 100, 200, 800)
ABSOLUTE_CRITERION_STORED_UNITS = 0.10
ORIGINAL_STEP4B_FINAL_MPJPE = 0.34014594554901123
D1_FINAL_MPJPE = 0.32848620414733887
D2_INITIAL_MPJPE = 4.920130252838135
D2_STEP_100_MPJPE = 0.5151628255844116
D2_FINAL_MPJPE = 0.46943652629852295
DEFAULT_SOURCE_ARTIFACT_ROOT = Path("data/_artifacts/step4b_pilot")
DEFAULT_D2_ROOT = Path("data/_artifacts/step4b_d2")
DEFAULT_D3_ROOT = Path("data/_artifacts/step4b_d3")
D3_PROTOCOL_PATH = Path("docs/STEP_4B_D3_PROTOCOL.md")

EXPECTED_PROTECTED_RECORDS = {
    "step4b_config.py": "8532d6fe0e017e40679819b19c5e14754d237279b2deb352725a2829be41a239",
    "step4b_single_person_pilot.py": "2848d7b8cc56429da6d74c51bafc7782229bf9fbf9b38eaba2149688d83a1169",
    "step4b_d1_diagnostic.py": "c095dac5e400b12e766390065a29e258b177b636d769d50923d99a14c9a0327b",
    "step4b_d2_diagnostic.py": "cd8aa4e36ea398769983c273ae3fdbc2f211650b8e35e392669848f8c2d8e161",
    "model/model.py": "7164ad53d4d9cf71316e3e379983576b82bfced17ec58ff67c7febadc121840f",
    "data/_artifacts/step4b_pilot/freeze_v2/step4b_freeze_config.json": "807b3e8bc37136e2c2000baff77363dc16761fc02e19d90ec58b6ab3710fdb16",
    "data/_artifacts/step4b_pilot/preprocessing_validation_v2.json": "32b7b1e4360abbe76b71cdc01a6001d8b4d7348bb48fba2677a96421cd118d41",
    "data/_artifacts/step4b_pilot/overfit_result.json": "e30a5aad15a2bab5844ab5c14e2738d30d26fef2aba956adb9015794c3253a06",
    "data/_artifacts/step4b_pilot/decision.json": "2d73224dedbf9c3f869b040f290376df9d582f5865ab13296352eb86e73f97e3",
    "data/_artifacts/step4b_d1/step4b_d1_freeze.json": "a2f683741cc5f8616a0ad374dbaa3a3ea17ade893974b611b18f30fa5a4a3ffc",
    "data/_artifacts/step4b_d1/step4b_d1_result.json": "92a6673ae63439161427a5888cfb60b960d124b37c1b6803cf1095a627266586",
    "data/_artifacts/step4b_d1/step4b_d1_decision.json": "155a6c9580ae8a22642588730da4c6a8e482bef81c93406790ad85b20ec0a1b8",
    "data/_artifacts/step4b_d1/loss_trajectory.json": "e8afa9486c5a4bd9eda1cacff667d556cc38cf41e16f33788f224022127a30d4",
    "data/_artifacts/step4b_d2/parameter_audit_before_edit.json": "0a516ad84a8c57fc7584f6122baca3b52e99e8eb7081432a3d4ae9f020db33e6",
    "data/_artifacts/step4b_d2/input_verification.json": "4ebbfd5594092cee4b1af996d0ba1cf37f05ff2deaaf0b3e508dcccc7921014e",
    "data/_artifacts/step4b_d2/step4b_d2_freeze.json": "9e04365bac612149314f6122a369def82f5cc49a53d6c37f242e3f9d6739229f",
    "data/_artifacts/step4b_d2/step4b_d2_result.json": "f83620f8736a3e35ef4ee9f3adefa6aa6aaa8628049058b5520b20c7a146e0f4",
    "data/_artifacts/step4b_d2/step4b_d2_decision.json": "9508f2bdad75f6d19b7c9572a29401db45b36c2db0f2dba89ee30eb9ab222e95",
    "data/_artifacts/step4b_d2/loss_trajectory.json": "3f8446c25747715952ec804d4e4e25e1c6d85846bdd2f63d7e8c96d80a63b737",
    "data/_artifacts/step4b_d2/checkpoint_step_0000.json": "c4b8d9011470fca2b670f497f4bd3d5c61b48188d63909b2217c9da0801f7413",
    "data/_artifacts/step4b_d2/checkpoint_step_0100.json": "dfa215722be0b385ca9000848b8036b8a5d0af2fe92233c3dc13b0239b6844cf",
    "data/_artifacts/step4b_d2/checkpoint_step_0200.json": "178ea9a31d856a7fd43f68fb6c30989209d60e01afddc2138664d463ed462c6b",
    "data/_artifacts/step4b_d2/checkpoint_predictions.npz": "6af328ed2cbc5141890eca05ba4dc59487395d262b91cd273287d8716f48b4cf",
    "data/_artifacts/step4b_d2/step4b_d2_final_state_dict.pt": "58819efca0119463a8f7904ee0ef8a2c22b8e760f118ce375726ee0e663caa6a",
}

build_d3_model = build_d2_model
make_d3_optimizer = make_sgd_optimizer


class D2ReproductionMismatchError(RuntimeError):
    def __init__(self, record):
        self.record = record
        super().__init__(
            "Step-4B-D3 stopped because the deterministic D2 reproduction gate failed: "
            + json.dumps(record, sort_keys=True)
        )


def d3_protocol_difference():
    return {
        "d2_version": D2_VERSION,
        "d3_version": D3_VERSION,
        "d2_steps": EXPECTED_D2_STEPS,
        "d3_steps": D3_STEPS,
        "changed_fields": ["optimizer_step_budget"],
        "same_ordered_batch": D3_BATCH_IDS == list(EXPECTED_BATCH_IDS),
        "same_model_builder": build_d3_model is build_d2_model,
        "same_optimizer_builder": make_d3_optimizer is make_sgd_optimizer,
        "same_loss_and_optimizer_settings": True,
        "train_encoder": True,
        "fresh_seed0_initialization": True,
        "resume_from_d2_checkpoint": False,
        "criterion_stored_units": ABSOLUTE_CRITERION_STORED_UNITS,
        "only_optimizer_step_budget_changed": True,
    }


def _hash_checks(paths_and_expected):
    checks = {}
    for path, expected in paths_and_expected.items():
        actual = _sha256_file(path)
        checks[path] = {
            "actual": actual,
            "expected": expected,
            "unchanged": actual == expected,
        }
    return checks


def _verify_d2_records(d2_root):
    d2_root = Path(d2_root)
    d2_freeze_path = d2_root / "step4b_d2_freeze.json"
    d2_result_path = d2_root / "step4b_d2_result.json"
    d2_decision_path = d2_root / "step4b_d2_decision.json"
    d2_freeze = _load_json(d2_freeze_path)
    d2_result = _load_json(d2_result_path)
    d2_decision = _load_json(d2_decision_path)

    value_checks = {
        "d2_freeze_version": d2_freeze["diagnostic_version"] == D2_VERSION,
        "d2_freeze_steps": d2_freeze["diagnostic_freeze"]["diagnostic_steps"] == EXPECTED_D2_STEPS,
        "d2_result_version": d2_result["diagnostic_version"] == D2_VERSION,
        "d2_result_status": d2_result["diagnostic_status"] == "TRAINABLE_ENCODER_DIAGNOSTIC_ABOVE_CRITERION",
        "d2_result_steps": d2_result["optimizer_steps"] == EXPECTED_D2_STEPS,
        "d2_result_batch": d2_result["batch_sample_ids"] == D3_BATCH_IDS,
        "d2_initial_mpjpe": d2_result["initial_mpjpe_stored_units"] == D2_INITIAL_MPJPE,
        "d2_step_100_mpjpe": d2_result["step_100_mpjpe_stored_units"] == D2_STEP_100_MPJPE,
        "d2_final_mpjpe": d2_result["final_mpjpe_stored_units"] == D2_FINAL_MPJPE,
        "d2_all_losses_finite": d2_result["all_losses_finite"] is True,
        "d2_all_gradients_finite": d2_result["all_gradients_finite"] is True,
        "d2_optimizer_coverage": d2_result["trainability_audit"]["optimizer_contains_exact_trainable_parameter_set"] is True,
        "d2_decision_status": d2_decision["status"] == "TRAINABLE_ENCODER_DIAGNOSTIC_ABOVE_CRITERION",
        "d2_preserves_original_block": d2_decision["original_step4b_status"] == "BLOCKED",
        "d2_preserves_d1": d2_decision["d1_result_preserved"] is True,
    }
    if not all(value_checks.values()):
        raise ValueError("Protected Step-4B-D2 records changed: " + json.dumps(value_checks, sort_keys=True))

    source_records = {
        "d2_freeze": _source_file_record(d2_freeze_path),
        "d2_result": _source_file_record(d2_result_path),
        "d2_decision": _source_file_record(d2_decision_path),
    }
    return {
        "d2_value_checks": value_checks,
        "d2_source_records": source_records,
        "d2_initial_mpjpe_stored_units": D2_INITIAL_MPJPE,
        "d2_step_100_mpjpe_stored_units": D2_STEP_100_MPJPE,
        "d2_final_mpjpe_stored_units": D2_FINAL_MPJPE,
        "d2_diagnostic_status": d2_result["diagnostic_status"],
        "d2_decision": d2_decision,
    }


def verify_frozen_inputs(
    source_artifact_root=DEFAULT_SOURCE_ARTIFACT_ROOT,
    freeze_config=None,
    pilot_root=None,
    d2_artifact_root=DEFAULT_D2_ROOT,
):
    d2_artifact_root = Path(d2_artifact_root)
    verification = verify_d2_frozen_inputs(
        source_artifact_root=source_artifact_root,
        freeze_config=freeze_config,
        pilot_root=pilot_root,
        pre_edit_audit_path=d2_artifact_root / "parameter_audit_before_edit.json",
    )
    protected_checks = _hash_checks(EXPECTED_PROTECTED_RECORDS)
    if not all(check["unchanged"] for check in protected_checks.values()):
        raise ValueError("Protected original Step-4B, D1, or D2 records changed")

    d2_record_checks = _verify_d2_records(d2_artifact_root)
    if not D3_PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"Missing frozen D3 protocol artifact: {D3_PROTOCOL_PATH}")

    verification.update(
        {
            "diagnostic_version": D3_VERSION,
            "diagnostic_type": D3_TYPE,
            "protocol_document": _source_file_record(D3_PROTOCOL_PATH),
            "protected_record_checks": protected_checks,
            "d2_record_checks": d2_record_checks,
            "d2_initial_mpjpe_stored_units": d2_record_checks["d2_initial_mpjpe_stored_units"],
            "d2_step_100_mpjpe_stored_units": d2_record_checks["d2_step_100_mpjpe_stored_units"],
            "d2_final_mpjpe_stored_units": d2_record_checks["d2_final_mpjpe_stored_units"],
            "d2_diagnostic_status": d2_record_checks["d2_diagnostic_status"],
            "d2_decision": d2_record_checks["d2_decision"],
            "d1_final_mpjpe_stored_units": D1_FINAL_MPJPE,
            "absolute_final_mpjpe_criterion_stored_units": ABSOLUTE_CRITERION_STORED_UNITS,
        }
    )
    return verification


def write_diagnostic_freeze(source_artifact_root, d3_root, freeze_config, pilot_root, d2_artifact_root, verification):
    source_artifact_root = Path(source_artifact_root)
    d3_root = Path(d3_root)
    d3_root.mkdir(parents=True, exist_ok=True)
    freeze_path = d3_root / "step4b_d3_freeze.json"
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to overwrite frozen D3 diagnostic: {freeze_path}")
    record = {
        "diagnostic_version": D3_VERSION,
        "diagnostic_type": D3_TYPE,
        "protocol_document": verification["protocol_document"],
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
            **verification["d2_record_checks"]["d2_source_records"],
        },
        "protected_record_checks": verification["protected_record_checks"],
        "frozen_source_summary": verification,
        "diagnostic_freeze": {
            "diagnostic_steps": D3_STEPS,
            "d2_optimizer_steps": EXPECTED_D2_STEPS,
            "experimental_change": {
                "optimizer_step_budget": "200 -> 800",
            },
            "only_optimizer_step_budget_changed": True,
            "seed": verification["source_seed"],
            "fresh_seed0_initialization": True,
            "resume_from_d2_checkpoint": False,
            "train_encoder": True,
            "batch_size": verification["source_batch_size"],
            "batch_sample_ids": list(D3_BATCH_IDS),
            "model_family": verification["source_model"]["family"],
            "initialization_procedure": "unchanged from D2; same from-scratch seed-0 model construction without loading a D2 checkpoint",
            "optimizer": {
                "type": verification["source_optimizer"],
                "learning_rate": verification["source_learning_rate"],
                "momentum": verification["source_momentum"],
            },
            "loss": verification["source_loss"],
            "absolute_final_mpjpe_criterion_stored_units": ABSOLUTE_CRITERION_STORED_UNITS,
            "d2_reproduction_gate": {
                "initial_mpjpe_stored_units": D2_INITIAL_MPJPE,
                "step_100_mpjpe_stored_units": D2_STEP_100_MPJPE,
                "step_200_mpjpe_stored_units": D2_FINAL_MPJPE,
                "comparison": "exact train-mode checkpoint equality before continuing past step 200",
            },
            "unchanged_fields": [
                "corrected freeze_v2 internal-train sample IDs",
                "ordered fixed batch of 8 samples",
                "seed",
                "model family and dimensions",
                "random initialization values",
                "train_encoder=True",
                "CSI preprocessing and loader normalization",
                "pose loss",
                "SGD optimizer",
                "learning rate",
                "momentum",
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


def _verify_d2_reproduction(checkpoints):
    expected_by_step = {
        0: ("initial", D2_INITIAL_MPJPE),
        100: ("step_100", D2_STEP_100_MPJPE),
        200: ("step_200", D2_FINAL_MPJPE),
    }
    mismatches = []
    for step, (name, expected) in expected_by_step.items():
        actual = checkpoints[step]["mpjpe_stored_units"]
        if actual != expected:
            mismatches.append(
                {
                    "step": step,
                    "checkpoint": name,
                    "expected_mpjpe_stored_units": expected,
                    "actual_mpjpe_stored_units": actual,
                }
            )
    if mismatches:
        raise D2ReproductionMismatchError(
            {
                "status": "D2_REPRODUCTION_MISMATCH_STOPPED",
                "mismatches": mismatches,
                "stop_rule": "D3 stopped before extending past the D2 200-step checkpoint; no protocol adjustment was made.",
            }
        )
    return {
        "initial_reproduced": checkpoints[0]["mpjpe_stored_units"] == D2_INITIAL_MPJPE,
        "step_100_reproduced": checkpoints[100]["mpjpe_stored_units"] == D2_STEP_100_MPJPE,
        "step_200_reproduced": checkpoints[200]["mpjpe_stored_units"] == D2_FINAL_MPJPE,
        "comparison": "exact train-mode checkpoint equality",
    }


def _window_mean(records, start_step, end_step):
    values = [
        record["pre_update_loss_stored_units"]
        for record in records
        if start_step <= record["step"] <= end_step
    ]
    return float(np.mean(values)) if values else None


def _linear_slope(records):
    if len(records) < 2:
        return None
    steps = np.array([record["step"] for record in records], dtype=np.float64)
    losses = np.array([record["pre_update_loss_stored_units"] for record in records], dtype=np.float64)
    return float(np.polyfit(steps, losses, 1)[0])


def summarize_late_trajectory(records):
    post_original = [record for record in records if record["step"] > EXPECTED_D2_STEPS]
    late_600 = [record for record in records if 601 <= record["step"] <= 800]
    return {
        "mean_steps_1_100": _window_mean(records, 1, 100),
        "mean_steps_101_200": _window_mean(records, 101, 200),
        "mean_steps_201_400": _window_mean(records, 201, 400),
        "mean_steps_401_600": _window_mean(records, 401, 600),
        "mean_steps_601_800": _window_mean(records, 601, 800),
        "previous_100_mean_steps_601_700": _window_mean(records, 601, 700),
        "last_100_mean_steps_701_800": _window_mean(records, 701, 800),
        "last_100_minus_previous_100_mean": (
            _window_mean(records, 701, 800) - _window_mean(records, 601, 700)
        ),
        "linear_slope_all_post_original_steps_per_step": _linear_slope(post_original),
        "linear_slope_steps_601_800_per_step": _linear_slope(late_600),
        "minimum_pre_update_loss_stored_units": min(record["pre_update_loss_stored_units"] for record in records),
        "minimum_pre_update_step": min(records, key=lambda record: record["pre_update_loss_stored_units"])["step"],
        "maximum_pre_update_loss_stored_units": max(record["pre_update_loss_stored_units"] for record in records),
        "final_pre_update_loss_stored_units": records[-1]["pre_update_loss_stored_units"],
    }


def compare_prior_results(initial, final):
    return {
        "original_step4b_final_mpjpe_stored_units": ORIGINAL_STEP4B_FINAL_MPJPE,
        "d1_final_mpjpe_stored_units": D1_FINAL_MPJPE,
        "d2_initial_mpjpe_stored_units": D2_INITIAL_MPJPE,
        "d2_step_100_mpjpe_stored_units": D2_STEP_100_MPJPE,
        "d2_final_mpjpe_stored_units": D2_FINAL_MPJPE,
        "d3_initial_mpjpe_stored_units": initial,
        "d3_final_mpjpe_stored_units": final,
        "d3_final_minus_original_step4b_final_stored_units": final - ORIGINAL_STEP4B_FINAL_MPJPE,
        "d3_final_minus_d1_final_stored_units": final - D1_FINAL_MPJPE,
        "d3_final_minus_d2_final_stored_units": final - D2_FINAL_MPJPE,
        "d3_final_to_initial_ratio": final / initial if initial > 0 else None,
        "d3_final_to_d1_final_ratio": final / D1_FINAL_MPJPE,
        "d3_final_to_d2_final_ratio": final / D2_FINAL_MPJPE,
    }


def run_diagnostic(
    source_artifact_root,
    d3_root,
    freeze_config,
    pilot_root,
    d2_artifact_root,
    device_name,
):
    source_artifact_root, freeze_config, pilot_root = _resolve_source_paths(
        source_artifact_root, freeze_config, pilot_root
    )
    d3_root = Path(d3_root)
    verification = verify_frozen_inputs(
        source_artifact_root=source_artifact_root,
        freeze_config=freeze_config,
        pilot_root=pilot_root,
        d2_artifact_root=d2_artifact_root,
    )
    _write_json(d3_root / "input_verification.json", verification)
    freeze_path, freeze_record = write_diagnostic_freeze(
        source_artifact_root,
        d3_root,
        freeze_config,
        pilot_root,
        d2_artifact_root,
        verification,
    )

    source_freeze = _load_json(freeze_config)
    config = source_freeze["config"]
    validate_step4b_config(config)
    diagnostic_freeze = freeze_record["diagnostic_freeze"]
    if diagnostic_freeze["diagnostic_steps"] != D3_STEPS:
        raise AssertionError("D3 diagnostic step budget is not the frozen 800-step budget")
    if diagnostic_freeze["only_optimizer_step_budget_changed"] is not True:
        raise AssertionError("D3 freeze does not record only the optimizer-step budget change")

    device = _resolve_device(device_name)
    _seed_everything(config["seed"])
    train_dataset = SinglePersonPiW3DDataset(
        pilot_root,
        pilot_root / "train_data" / "selected_train_ids.txt",
        expected_split="internal_train",
        split_assignments_path=Path(freeze_config).parent / "split_assignments.json",
    )
    if train_dataset.sample_ids[: len(D3_BATCH_IDS)] != D3_BATCH_IDS:
        raise AssertionError("D3 loader did not reproduce the frozen ordered tiny batch")

    model = build_d3_model(config, device)
    model.train()
    optimizer = make_d3_optimizer(model, config)
    trainability_audit = parameter_audit(model, optimizer)
    if not trainability_audit["optimizer_contains_exact_trainable_parameter_set"]:
        raise AssertionError("D3 optimizer does not contain exactly the intended trainable parameter set")
    if trainability_audit["parameter_counts"]["trainable_named_parameters"] != 97:
        raise AssertionError("D3 trainable encoder/decoder parameter audit failed")
    if trainability_audit["module_counts"]["encoder"]["trainable_named_parameters"] != 50:
        raise AssertionError("D3 encoder is not fully trainable")
    if trainability_audit["module_counts"]["pose_decoder"]["trainable_named_parameters"] != 47:
        raise AssertionError("D3 pose decoder is not fully trainable")

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
    reproduction_check = None

    trajectory = []
    for step in range(1, D3_STEPS + 1):
        optimizer.zero_grad()
        predicted_pose, _ = model(csi)
        loss = torch.mean(torch.norm(predicted_pose - pose, dim=-1))
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite D3 loss at step {step}")
        loss.backward()
        gradient_stats = _replace_audit_with_gradient_stats(model)
        if not gradient_stats["all_gradients_finite"]:
            raise ValueError(f"Non-finite D3 gradients at step {step}")
        optimizer.step()
        trajectory.append(
            {
                "step": step,
                "pre_update_loss_stored_units": float(loss.detach().cpu()),
                "pre_update_mpjpe_stored_units": float(loss.detach().cpu()),
                **gradient_stats,
            }
        )
        if step in (100, 200, D3_STEPS):
            model_state = _clone_model_state(model)
            rng_state = _capture_rng_state()
            checkpoint = _capture_checkpoint(model, csi, pose, batch_sample_ids, step)
            checkpoint_summaries[step] = checkpoint
            checkpoint_arrays[step] = checkpoint
            _restore_model_state(model, model_state)
            _restore_rng_state(rng_state)
        if step == EXPECTED_D2_STEPS:
            reproduction_check = _verify_d2_reproduction(checkpoint_summaries)

    runtime_seconds = time.perf_counter() - start_time
    initial = checkpoint_summaries[0]["mpjpe_stored_units"]
    step_100 = checkpoint_summaries[100]["mpjpe_stored_units"]
    step_200 = checkpoint_summaries[200]["mpjpe_stored_units"]
    final = checkpoint_summaries[D3_STEPS]["mpjpe_stored_units"]
    criterion = diagnostic_freeze["absolute_final_mpjpe_criterion_stored_units"]
    criterion_passed = final <= criterion
    finite_evidence = all(record["all_gradients_finite"] for record in trajectory) and all(
        np.isfinite(record["pre_update_loss_stored_units"]) for record in trajectory
    )
    if not finite_evidence:
        raise AssertionError("D3 completed without finite-loss/finite-gradient evidence")
    if reproduction_check is None:
        raise AssertionError("D3 completed without running the D2 reproduction gate")

    parameter_change = summarize_parameter_changes(model, initial_model_state)
    late_trajectory = summarize_late_trajectory(trajectory)
    prior_comparisons = compare_prior_results(initial, final)

    for step, checkpoint in checkpoint_summaries.items():
        _write_json(
            d3_root / f"checkpoint_step_{step:04d}.json",
            {key: value for key, value in checkpoint.items() if key not in {"prediction", "target"}},
        )
    np.savez(
        d3_root / "checkpoint_predictions.npz",
        sample_ids=np.array(batch_sample_ids),
        **{
            f"prediction_step_{step:04d}": arrays["prediction"][:, None, :, :]
            for step, arrays in checkpoint_arrays.items()
        },
        ground_truth=checkpoint_arrays[0]["target"][:, None, :, :],
    )
    _write_json(d3_root / "loss_trajectory.json", {"records": trajectory})

    checkpoint_path = d3_root / "step4b_d3_final_state_dict.pt"
    torch.save(model.state_dict(), checkpoint_path)
    reloaded_model = build_d3_model(config, device)
    reloaded_model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    reloaded_model.eval()
    with torch.no_grad():
        reloaded_prediction, _ = reloaded_model(csi)
        reloaded_loss = torch.mean(torch.norm(reloaded_prediction - pose, dim=-1))
    if not torch.isfinite(reloaded_loss):
        raise ValueError("Non-finite D3 checkpoint reload evaluation loss")

    if criterion_passed:
        diagnostic_status = "TRAINABLE_ENCODER_EXTENDED_STEP_CRITERION_MET"
        interpretation = (
            "D3 final train-mode MPJPE met the unchanged absolute criterion with the "
            "trainable-encoder 800-step budget; original Step-4B remains historically blocked."
        )
    else:
        diagnostic_status = "TRAINABLE_ENCODER_EXTENDED_STEP_ABOVE_CRITERION"
        interpretation = (
            "D3 did not pass the unchanged absolute criterion with the trainable-encoder "
            f"800-step budget; final MPJPE was {final:.17g} stored units. Original Step-4B "
            "remains BLOCKED and no further fix was attempted."
        )

    result = {
        "diagnostic_version": D3_VERSION,
        "diagnostic_type": D3_TYPE,
        "diagnostic_run_status": "COMPLETE",
        "diagnostic_status": diagnostic_status,
        "interpretation": interpretation,
        "protocol_document": verification["protocol_document"],
        "original_result_preserved": True,
        "d1_result_preserved": True,
        "d2_result_preserved": True,
        "source_step4b_decision": verification["source_step4b_decision"],
        "d1_diagnostic_status": verification["d1_diagnostic_status"],
        "d2_diagnostic_status": verification["d2_diagnostic_status"],
        "freeze_record": str(freeze_path),
        "device": str(device),
        "runtime_seconds": runtime_seconds,
        "seed": config["seed"],
        "fresh_seed0_initialization": True,
        "resume_from_d2_checkpoint": False,
        "train_encoder": True,
        "batch_size": int(csi.shape[0]),
        "batch_sample_ids": batch_sample_ids,
        "optimizer_steps": D3_STEPS,
        "d2_optimizer_steps": EXPECTED_D2_STEPS,
        "only_optimizer_step_budget_changed_from_d2": True,
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
        "d2_reproduction": reproduction_check,
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
        "step_200_mpjpe_stored_units": step_200,
        "final_mpjpe_stored_units": final,
        "step_800_mpjpe_stored_units": final,
        "final_to_initial_mpjpe_ratio": final / initial if initial > 0 else None,
        "initial": {key: value for key, value in checkpoint_summaries[0].items() if key not in {"prediction", "target"}},
        "step_100": {key: value for key, value in checkpoint_summaries[100].items() if key not in {"prediction", "target"}},
        "step_200": {key: value for key, value in checkpoint_summaries[200].items() if key not in {"prediction", "target"}},
        "step_800": {
            key: value
            for key, value in checkpoint_summaries[D3_STEPS].items()
            if key not in {"prediction", "target"}
        },
        "trajectory_summary": {
            "first": trajectory[0]["pre_update_loss_stored_units"],
            "step_100": trajectory[99]["pre_update_loss_stored_units"],
            "step_200": trajectory[199]["pre_update_loss_stored_units"],
            "last": trajectory[-1]["pre_update_loss_stored_units"],
            "min": min(record["pre_update_loss_stored_units"] for record in trajectory),
            "max": max(record["pre_update_loss_stored_units"] for record in trajectory),
        },
        "late_trajectory_summary": late_trajectory,
        "prior_comparisons": prior_comparisons,
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
        "source_step4b_model_parameters": verification["source_parameter_counts"],
        "model_parameters": trainability_audit["parameter_counts"],
        "coordinate_convention": verification["source_coordinate_convention"],
        "development_samples_loaded": False,
        "official_test_used": False,
        "checkpoint": {
            "path": str(checkpoint_path),
            "format": "state_dict",
            "reloaded_loss_stored_units": float(reloaded_loss.detach().cpu()),
            "reloaded_mpjpe_stored_units": float(reloaded_loss.detach().cpu()),
            "reload_finite": True,
            "measurement_mode": "eval-mode secondary diagnostic only; the frozen criterion uses train-mode checkpoint MPJPE",
        },
    }
    _write_json(d3_root / "step4b_d3_result.json", result)
    _write_json(
        d3_root / "step4b_d3_decision.json",
        {
            "status": diagnostic_status,
            "reason": interpretation,
            "final_mpjpe_stored_units": final,
            "step_800_mpjpe_stored_units": final,
            "original_absolute_criterion_stored_units": criterion,
            "original_step4b_status": verification["source_step4b_decision"]["status"],
            "d1_status": verification["d1_diagnostic_status"],
            "d2_status": verification["d2_diagnostic_status"],
            "original_result_preserved": True,
            "d1_result_preserved": True,
            "d2_result_preserved": True,
            "result": str(d3_root / "step4b_d3_result.json"),
        },
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"d3_result={d3_root / 'step4b_d3_result.json'}")
    print(f"d3_decision={d3_root / 'step4b_d3_decision.json'}")
    return 0


def _replace_audit_with_gradient_stats(model):
    stats = {
        "gradient_tensors": 0,
        "finite_gradient_tensors": 0,
        "max_abs_gradient": 0.0,
        "encoder_gradient_tensors": 0,
        "finite_encoder_gradient_tensors": 0,
        "nonzero_encoder_gradient_tensors": 0,
        "max_abs_encoder_gradient": 0.0,
    }
    encoder_roots = {"cls_token", "pos_embedding", "patchify", "transformer", "layer_norm"}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if parameter.grad is None:
            raise ValueError(f"Trainable parameter has no gradient: {name}")
        stats["gradient_tensors"] += 1
        finite = bool(torch.isfinite(parameter.grad).all().item())
        stats["finite_gradient_tensors"] += int(finite)
        max_abs = float(parameter.grad.detach().abs().max().item())
        if np.isfinite(max_abs):
            stats["max_abs_gradient"] = max(stats["max_abs_gradient"], max_abs)
        if name.split(".", 1)[0] in encoder_roots:
            stats["encoder_gradient_tensors"] += 1
            stats["finite_encoder_gradient_tensors"] += int(finite)
            stats["nonzero_encoder_gradient_tensors"] += int(bool((parameter.grad != 0).any().item()))
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


def parse_args():
    parser = argparse.ArgumentParser(description="Run the frozen Step-4B-D3 trainable-encoder extended-step diagnostic.")
    parser.add_argument("--source-artifact-root", type=Path, default=DEFAULT_SOURCE_ARTIFACT_ROOT)
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_D3_ROOT)
    parser.add_argument("--d2-artifact-root", type=Path, default=DEFAULT_D2_ROOT)
    parser.add_argument("--freeze-config", type=Path, default=None)
    parser.add_argument("--pilot-root", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument("--verify-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.verify_only:
        verification = verify_frozen_inputs(
            source_artifact_root=args.source_artifact_root,
            freeze_config=args.freeze_config,
            pilot_root=args.pilot_root,
            d2_artifact_root=args.d2_artifact_root,
        )
        path = _write_json(args.artifact_root / "input_verification.json", verification)
        print(json.dumps(verification, indent=2, sort_keys=True))
        print(f"input_verification={path}")
        return 0
    try:
        return run_diagnostic(
            args.source_artifact_root,
            args.artifact_root,
            args.freeze_config,
            args.pilot_root,
            args.d2_artifact_root,
            args.device,
        )
    except D2ReproductionMismatchError as exc:
        args.artifact_root.mkdir(parents=True, exist_ok=True)
        path = _write_json(args.artifact_root / "step4b_d3_reproduction_mismatch.json", exc.record)
        print(json.dumps(exc.record, indent=2, sort_keys=True))
        print(f"reproduction_mismatch={path}")
        return 3


if __name__ == "__main__":
    raise SystemExit(main())
