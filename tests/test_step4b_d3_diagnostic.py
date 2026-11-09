import inspect

import torch

from model.model import ViT_Pose_Decoder
from step4b_config import step4b_config_for_json
from step4b_d2_diagnostic import (
    D2_VERSION,
    EXPECTED_BATCH_IDS,
    EXPECTED_SOURCE_STEPS,
    build_d2_model,
    make_sgd_optimizer,
    parameter_audit,
)
from step4b_d3_diagnostic import (
    ABSOLUTE_CRITERION_STORED_UNITS,
    D2_FINAL_MPJPE,
    D2_INITIAL_MPJPE,
    D2_STEP_100_MPJPE,
    D3_CHECKPOINT_STEPS,
    D3_STEPS,
    D3_VERSION,
    build_d3_model,
    d3_protocol_difference,
    make_d3_optimizer,
)
from step4b_single_person_pilot import build_model


def frozen_config():
    return step4b_config_for_json()


def encoder_parameter_names(model):
    roots = {"cls_token", "pos_embedding", "patchify", "transformer", "layer_norm"}
    return [name for name, _ in model.named_parameters() if name.split(".", 1)[0] in roots]


def test_d3_differs_from_d2_only_in_optimizer_step_budget():
    difference = d3_protocol_difference()

    assert D3_VERSION == "compwifi-step4b-d3-v1"
    assert D3_STEPS == 800
    assert EXPECTED_SOURCE_STEPS == 200
    assert D3_STEPS == 4 * EXPECTED_SOURCE_STEPS
    assert D3_CHECKPOINT_STEPS == (0, 100, 200, 800)
    assert ABSOLUTE_CRITERION_STORED_UNITS == 0.10
    assert difference == {
        "d2_version": D2_VERSION,
        "d3_version": D3_VERSION,
        "d2_steps": 200,
        "d3_steps": 800,
        "changed_fields": ["optimizer_step_budget"],
        "same_ordered_batch": True,
        "same_model_builder": True,
        "same_optimizer_builder": True,
        "same_loss_and_optimizer_settings": True,
        "train_encoder": True,
        "fresh_seed0_initialization": True,
        "resume_from_d2_checkpoint": False,
        "criterion_stored_units": 0.10,
        "only_optimizer_step_budget_changed": True,
    }


def test_d3_uses_the_d2_trainable_encoder_builder_and_optimizer_procedure():
    config = frozen_config()

    assert build_d3_model is build_d2_model
    assert make_d3_optimizer is make_sgd_optimizer

    torch.manual_seed(config["seed"])
    model = build_d3_model(config, torch.device("cpu"))
    optimizer = make_d3_optimizer(model, config)
    audit = parameter_audit(model, optimizer)

    assert audit["module_counts"]["encoder"]["named_parameters"] == 50
    assert audit["module_counts"]["encoder"]["trainable_named_parameters"] == 50
    assert audit["module_counts"]["pose_decoder"]["named_parameters"] == 47
    assert audit["module_counts"]["pose_decoder"]["trainable_named_parameters"] == 47
    assert audit["parameter_counts"]["trainable_named_parameters"] == 97
    assert audit["parameter_counts"]["trainable_parameters"] == 4795395
    assert audit["optimizer"]["parameter_references"] == 97
    assert audit["optimizer"]["unique_parameter_references"] == 97
    assert audit["optimizer"]["duplicate_optimizer_references"] == 0
    assert audit["optimizer"]["unknown_optimizer_references"] == 0
    assert audit["optimizer_contains_exact_trainable_parameter_set"] is True


def test_d3_builder_preserves_seed0_d2_initialization_and_trainability_only():
    config = frozen_config()
    torch.manual_seed(config["seed"])
    legacy_model = build_model(config, torch.device("cpu"))
    legacy_state = {name: value.detach().clone() for name, value in legacy_model.state_dict().items()}

    torch.manual_seed(config["seed"])
    d3_model = build_d3_model(config, torch.device("cpu"))

    for name, value in d3_model.state_dict().items():
        assert torch.equal(value, legacy_state[name])
    for name in encoder_parameter_names(legacy_model):
        assert not dict(legacy_model.named_parameters())[name].requires_grad
        assert dict(d3_model.named_parameters())[name].requires_grad
    decoder_names = [
        name for name, _ in d3_model.named_parameters() if name not in encoder_parameter_names(d3_model)
    ]
    assert decoder_names
    assert all(dict(d3_model.named_parameters())[name].requires_grad for name in decoder_names)

    d3_model.set_encoder_trainable(False)
    assert all(not dict(d3_model.named_parameters())[name].requires_grad for name in encoder_parameter_names(d3_model))
    d3_model.set_encoder_trainable(True)
    assert all(dict(d3_model.named_parameters())[name].requires_grad for name in encoder_parameter_names(d3_model))


def test_legacy_model_default_remains_frozen_encoder_decoder_only():
    config = frozen_config()
    signature = inspect.signature(ViT_Pose_Decoder.__init__)
    assert signature.parameters["train_encoder"].default is False

    torch.manual_seed(config["seed"])
    model = build_model(config, torch.device("cpu"))
    optimizer = make_sgd_optimizer(model, config)
    audit = parameter_audit(model, optimizer)

    assert audit["module_counts"]["encoder"]["named_parameters"] == 50
    assert audit["module_counts"]["encoder"]["trainable_named_parameters"] == 0
    assert audit["module_counts"]["pose_decoder"]["named_parameters"] == 47
    assert audit["module_counts"]["pose_decoder"]["trainable_named_parameters"] == 47
    assert audit["parameter_counts"]["total_parameters"] == 4795395
    assert audit["parameter_counts"]["trainable_parameters"] == 1404931
    assert audit["optimizer"]["parameter_references"] == 47
    assert audit["optimizer"]["unique_parameter_references"] == 47
    assert audit["optimizer"]["duplicate_optimizer_references"] == 0
    assert audit["optimizer_contains_exact_trainable_parameter_set"] is True
    assert all(
        not record["requires_grad"]
        for record in audit["records"]
        if record["module"] == "encoder"
    )
    assert all(
        record["requires_grad"] and record["in_optimizer"]
        for record in audit["records"]
        if record["module"] == "pose_decoder"
    )


def test_d3_reproduction_targets_are_the_recorded_d2_train_mode_checkpoints():
    assert EXPECTED_BATCH_IDS == [
        "S11_06_10",
        "S11_06_11",
        "S11_06_13",
        "S11_06_14",
        "S11_06_16",
        "S11_06_17",
        "S11_06_19",
        "S11_06_20",
    ]
    assert D2_INITIAL_MPJPE == 4.920130252838135
    assert D2_STEP_100_MPJPE == 0.5151628255844116
    assert D2_FINAL_MPJPE == 0.46943652629852295
