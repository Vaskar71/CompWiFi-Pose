import json

import torch

from step4b_d2_diagnostic import EXPECTED_BATCH_IDS as D2_BATCH_IDS
from step4b_d3_diagnostic import (
    D2_FINAL_MPJPE,
    D2_INITIAL_MPJPE,
    D2_STEP_100_MPJPE,
)
from step4b_d4_reproducibility import (
    D2_MPJPE_BY_STEP,
    D2_SEED,
    D2_STEPS,
    D3_MPJPE_BY_STEP,
    D4_ROOT,
    EXPECTED_BATCH_IDS,
    PROTECTED_RECORDS,
    REPEAT_A_ROOT,
    REPEAT_B_ROOT,
    RESULT_COMPARISON_FIELDS,
    capture_environment,
    classify_repeat_evidence,
    static_execution_path_comparison,
    verify_protected_records,
)


def _synthetic_result(initial, step_100, final):
    return {
        "device": "cuda",
        "seed": D2_SEED,
        "batch_size": 8,
        "batch_sample_ids": list(EXPECTED_BATCH_IDS),
        "optimizer_steps": D2_STEPS,
        "optimizer": "SGD",
        "learning_rate": 0.01,
        "momentum": 0.9,
        "loss": "mean Euclidean distance over people, joints, and coordinate triples",
        "all_losses_finite": True,
        "all_gradients_finite": True,
        "all_encoder_gradients_finite": True,
        "all_encoder_gradients_nonzero": True,
        "initial_mpjpe_stored_units": initial,
        "step_100_mpjpe_stored_units": step_100,
        "final_mpjpe_stored_units": final,
        "final_to_initial_mpjpe_ratio": final / initial,
    }


def _synthetic_record(loss):
    return {
        "step": 1,
        "pre_update_loss_stored_units": loss,
        "pre_update_mpjpe_stored_units": loss,
        "gradient_tensors": 97,
        "finite_gradient_tensors": 97,
        "max_abs_gradient": 0.1,
        "encoder_gradient_tensors": 50,
        "finite_encoder_gradient_tensors": 50,
        "nonzero_encoder_gradient_tensors": 50,
        "max_abs_encoder_gradient": 0.1,
        "all_gradients_finite": True,
        "all_encoder_gradients_finite": True,
        "all_encoder_gradients_nonzero": True,
    }


def _write_synthetic_root(root, loss):
    root.mkdir(parents=True)
    (root / "step4b_d2_result.json").write_text(
        json.dumps(
            _synthetic_result(D2_INITIAL_MPJPE, D2_STEP_100_MPJPE, D2_FINAL_MPJPE),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "loss_trajectory.json").write_text(
        json.dumps({"records": [_synthetic_record(loss)]}, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_d4_frozen_scope_repeat_roots_and_historical_comparators():
    assert D2_STEPS == 200
    assert D2_SEED == 0
    assert EXPECTED_BATCH_IDS == D2_BATCH_IDS
    assert len(EXPECTED_BATCH_IDS) == 8
    assert D2_MPJPE_BY_STEP == {
        0: D2_INITIAL_MPJPE,
        100: D2_STEP_100_MPJPE,
        200: D2_FINAL_MPJPE,
    }
    assert D3_MPJPE_BY_STEP == {
        0: D2_INITIAL_MPJPE,
        100: 0.40826788544654846,
        200: 0.3430270552635193,
    }
    assert str(REPEAT_A_ROOT).startswith(str(D4_ROOT))
    assert str(REPEAT_B_ROOT).startswith(str(D4_ROOT))
    assert REPEAT_A_ROOT != REPEAT_B_ROOT
    assert REPEAT_A_ROOT.parent == D4_ROOT
    assert REPEAT_B_ROOT.parent == D4_ROOT


def test_d4_static_execution_comparison_identifies_shared_d2_d3_path_and_differences():
    comparison = static_execution_path_comparison()

    assert comparison["same_model_builder_alias"] is True
    assert comparison["same_optimizer_builder_alias"] is True
    assert comparison["all_required_d2_imports_present"] is True
    assert comparison["all_shared_execution_checks_passed"] is True
    assert "model construction" in comparison["unchanged_semantics_through_step_200"]
    assert "optimizer construction and updates" in comparison["unchanged_semantics_through_step_200"]
    assert "train-mode checkpoint measurements at steps 0, 100, and 200" in comparison[
        "unchanged_semantics_through_step_200"
    ]
    difference_areas = {difference["area"] for difference in comparison["observed_code_differences"]}
    assert "optimizer-step budget" in difference_areas
    assert "post-checkpoint control flow" in difference_areas
    assert comparison["determinism_source_scan"]["called_in_d2"] == []
    assert comparison["determinism_source_scan"]["called_in_d3"] == []


def test_d4_environment_record_contains_versions_and_determinism_settings():
    environment = capture_environment()

    assert environment["python"]["version_info"][:2] == [3, 11]
    assert environment["packages"]["numpy"] == "1.26.4"
    assert environment["packages"]["torch"] == "2.5.1+cu121"
    assert environment["packages"]["torchvision"] == "0.20.1+cu121"
    assert environment["torch"]["cuda_build"] == "12.1"
    assert environment["torch"]["cuda_available"] is True
    assert environment["torch"]["cuda_devices"][0]["name"] == "NVIDIA GeForce RTX 3080 Laptop GPU"
    assert environment["torch"]["cuda_devices"][0]["capability"] == [8, 6]
    assert environment["torch"]["deterministic_algorithms_enabled"] is False
    assert environment["torch"]["deterministic_algorithms_warn_only_enabled"] is False
    assert environment["torch"]["cudnn"]["deterministic"] is False
    assert environment["random_seed_procedure"]["frozen_seed"] == 0
    assert environment["random_seed_procedure"]["ordered_loader_generator_seed"] == 0


def test_d4_protected_records_include_d3_stop_artifacts_and_match():
    assert "data/_artifacts/step4b_d3/step4b_d3_reproduction_mismatch.json" in PROTECTED_RECORDS
    assert "docs/STEP_4B_D3_REPORT.md" in PROTECTED_RECORDS
    verification = verify_protected_records()

    assert verification["record_count"] == len(PROTECTED_RECORDS)
    assert verification["all_unchanged"] is True
    assert verification["checks"]["data/_artifacts/step4b_d3/step4b_d3_reproduction_mismatch.json"][
        "unchanged"
    ] is True


def test_d4_repeat_classification_reports_first_recorded_divergence(tmp_path):
    reference = tmp_path / "d2"
    repeat_a = tmp_path / "a"
    repeat_b = tmp_path / "b"
    _write_synthetic_root(reference, 0.5)
    _write_synthetic_root(repeat_a, 0.5)
    _write_synthetic_root(repeat_b, 0.5)

    matching = classify_repeat_evidence(reference, repeat_a, repeat_b)
    assert matching["classification"] == "both repeats exactly reproduce historical D2"
    assert matching["repeats_match_each_other"] is True
    assert matching["both_match_historical_d2"] is True
    assert matching["repeat_a_vs_repeat_b"]["trajectory_a_reference_b_actual"][
        "first_recorded_divergence"
    ] is None

    changed_b_records = json.loads((repeat_b / "loss_trajectory.json").read_text(encoding="utf-8"))
    changed_b_records["records"][0]["pre_update_loss_stored_units"] = 0.6
    (repeat_b / "loss_trajectory.json").write_text(
        json.dumps(changed_b_records, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    divergent = classify_repeat_evidence(reference, repeat_a, repeat_b)
    assert divergent["classification"] == "repeats differ from each other"
    assert divergent["repeats_match_each_other"] is False
    assert divergent["both_match_historical_d2"] is False
    assert divergent["repeat_a_vs_repeat_b"]["trajectory_a_reference_b_actual"][
        "first_recorded_divergence"
    ] == {
        "step": 1,
        "field": "pre_update_loss_stored_units",
        "reference": 0.5,
        "actual": 0.6,
    }
    assert RESULT_COMPARISON_FIELDS == [
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


def test_d4_does_not_change_model_builder_identity_or_initialization():
    from step4b_config import step4b_config_for_json
    from step4b_d2_diagnostic import build_d2_model
    from step4b_d3_diagnostic import build_d3_model

    config = step4b_config_for_json()
    torch.manual_seed(config["seed"])
    d2_model = build_d2_model(config, torch.device("cpu"))
    d2_state = {name: value.detach().clone() for name, value in d2_model.state_dict().items()}

    torch.manual_seed(config["seed"])
    repeat_model = build_d3_model(config, torch.device("cpu"))

    assert build_d3_model is build_d2_model
    for name, value in repeat_model.state_dict().items():
        assert torch.equal(value, d2_state[name])
    assert all(parameter.requires_grad for parameter in repeat_model.parameters())
