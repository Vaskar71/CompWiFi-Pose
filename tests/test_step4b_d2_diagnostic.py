import torch

from step4b_config import step4b_config_for_json
from step4b_d2_diagnostic import (
    D2_VERSION,
    ENCODER_ROOTS,
    EXPECTED_BATCH_IDS,
    EXPECTED_SOURCE_STEPS,
    build_d2_model,
    make_sgd_optimizer,
    parameter_audit,
    summarize_parameter_changes,
)
from step4b_single_person_pilot import build_model


def frozen_config():
    return step4b_config_for_json()


def encoder_parameter_names(model):
    return [
        name
        for name, _ in model.named_parameters()
        if name.split(".", 1)[0] in ENCODER_ROOTS
    ]


def test_d2_protocol_constants_are_frozen():
    assert D2_VERSION == "compwifi-step4b-d2-v1"
    assert EXPECTED_SOURCE_STEPS == 200
    assert len(EXPECTED_BATCH_IDS) == 8
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


def test_original_step4b_mode_keeps_encoder_frozen_and_optimizer_covers_decoder_only():
    config = frozen_config()
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


def test_d2_trainable_encoder_mode_makes_every_model_parameter_trainable():
    config = frozen_config()
    torch.manual_seed(config["seed"])
    model = build_d2_model(config, torch.device("cpu"))
    optimizer = make_sgd_optimizer(model, config)
    audit = parameter_audit(model, optimizer)

    assert audit["module_counts"]["encoder"]["named_parameters"] == 50
    assert audit["module_counts"]["encoder"]["trainable_named_parameters"] == 50
    assert audit["module_counts"]["pose_decoder"]["named_parameters"] == 47
    assert audit["module_counts"]["pose_decoder"]["trainable_named_parameters"] == 47
    assert audit["parameter_counts"]["total_parameters"] == 4795395
    assert audit["parameter_counts"]["trainable_parameters"] == 4795395
    assert audit["parameter_counts"]["trainable_named_parameters"] == 97
    assert audit["optimizer"]["parameter_references"] == 97
    assert audit["optimizer"]["unique_parameter_references"] == 97
    assert audit["optimizer"]["duplicate_optimizer_references"] == 0
    assert audit["optimizer_contains_exact_trainable_parameter_set"] is True
    assert all(record["requires_grad"] and record["in_optimizer"] for record in audit["records"])


def test_train_encoder_option_preserves_initialization_and_changes_only_trainability():
    config = frozen_config()
    torch.manual_seed(config["seed"])
    frozen_model = build_model(config, torch.device("cpu"))
    frozen_state = {name: value.detach().clone() for name, value in frozen_model.state_dict().items()}

    torch.manual_seed(config["seed"])
    trainable_model = build_d2_model(config, torch.device("cpu"))

    for name, value in trainable_model.state_dict().items():
        assert torch.equal(value, frozen_state[name])
    for name in encoder_parameter_names(frozen_model):
        assert not dict(frozen_model.named_parameters())[name].requires_grad
        assert dict(trainable_model.named_parameters())[name].requires_grad
    assert all(
        dict(trainable_model.named_parameters())[name].requires_grad
        for name, _ in trainable_model.named_parameters()
        if name.split(".", 1)[0] not in ENCODER_ROOTS
    )

    trainable_model.set_encoder_trainable(False)
    assert all(not dict(trainable_model.named_parameters())[name].requires_grad for name in encoder_parameter_names(trainable_model))
    trainable_model.set_encoder_trainable(True)
    assert all(dict(trainable_model.named_parameters())[name].requires_grad for name in encoder_parameter_names(trainable_model))


def test_parameter_change_summary_reports_encoder_and_decoder_updates():
    config = frozen_config()
    torch.manual_seed(config["seed"])
    model = build_d2_model(config, torch.device("cpu"))
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}

    with torch.no_grad():
        encoder_name = encoder_parameter_names(model)[0]
        decoder_name = next(
            name for name, _ in model.named_parameters() if name.split(".", 1)[0] not in ENCODER_ROOTS
        )
        dict(model.named_parameters())[encoder_name].add_(0.01)
        dict(model.named_parameters())[decoder_name].add_(0.02)

    summary = summarize_parameter_changes(model, initial_state)

    assert summary["trainable_parameters"] == 97
    assert summary["changed_trainable_parameters"] == 2
    assert summary["encoder_parameters"] == 50
    assert summary["changed_encoder_parameters"] == 1
    assert summary["all_trainable_parameters_changed"] is False
    assert summary["encoder_parameters_changed"] is True
    assert summary["all_encoder_parameters_changed"] is False
