import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch

from feeder.piw3d_single_person import SinglePersonPiW3DDataset, make_single_person_dataloader
from model.model import MAE_ViT, ViT_Pose_Decoder
from research_metrics import ResearchMetricConfig, evaluate_set_metrics
from step4b_config import step4b_config_for_json, validate_step4b_config


DEFAULT_ARTIFACT_ROOT = Path("data/_artifacts/step4b_pilot")


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _resolve_paths(artifact_root, freeze_config, pilot_root):
    artifact_root = Path(artifact_root)
    freeze_config = (
        Path(freeze_config)
        if freeze_config
        else artifact_root / "freeze_v2" / "step4b_freeze_config.json"
    )
    pilot_root = Path(pilot_root) if pilot_root else artifact_root / "pilot_data_v2"
    return artifact_root, freeze_config, pilot_root


def _load_frozen_config(freeze_config):
    freeze = _load_json(freeze_config)
    validate_step4b_config(freeze["config"])
    return freeze


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(device_name):
    if device_name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_model(config, device):
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
    model = ViT_Pose_Decoder(
        mae.encoder,
        keypoints=model_config["keypoints"],
        coor_num=model_config["coordinate_dim"],
        token_num=900,
        dataset=model_config["dataset"],
        num_person=model_config["q"],
    ).to(device)
    return model


def _make_datasets(freeze, pilot_root, freeze_config):
    pilot_train_root = Path(pilot_root) / "train_data"
    split_assignments_path = Path(freeze_config).parent / "split_assignments.json"
    train_dataset = SinglePersonPiW3DDataset(
        pilot_root,
        pilot_train_root / "selected_train_ids.txt",
        expected_split="internal_train",
        split_assignments_path=split_assignments_path,
    )
    development_dataset = SinglePersonPiW3DDataset(
        pilot_root,
        pilot_train_root / "selected_development_ids.txt",
        expected_split="internal_development",
        split_assignments_path=split_assignments_path,
    )
    return train_dataset, development_dataset


def _metric_config(config):
    return ResearchMetricConfig(**config["metrics"])


def _mean_per_joint_error(predicted_pose, target_pose):
    return torch.mean(torch.norm(predicted_pose - target_pose, dim=-1))


def _parameter_counts(model):
    return {
        "total_parameters": sum(parameter.numel() for parameter in model.parameters()),
        "trainable_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
    }


def collect_predictions(model, loader, device, presence_score):
    model.eval()
    ground_truths = []
    predictions = []
    presence_scores = []
    sample_losses = []
    with torch.no_grad():
        for batch in loader:
            csi = batch["csi"].to(device)
            pose = batch["pose"].to(device)
            predicted_pose, _ = model(csi)
            if predicted_pose.shape != pose.shape:
                raise AssertionError(
                    f"Prediction shape {tuple(predicted_pose.shape)} does not match label shape {tuple(pose.shape)}"
                )
            loss = _mean_per_joint_error(predicted_pose, pose)
            if not torch.isfinite(loss):
                raise ValueError(f"Non-finite evaluation loss: {loss}")
            sample_losses.append(float(loss.detach().cpu()))
            predicted_np = predicted_pose.detach().cpu().numpy().astype(np.float64)
            target_np = pose.detach().cpu().numpy().astype(np.float64)
            for sample_index in range(target_np.shape[0]):
                ground_truths.append(target_np[sample_index : sample_index + 1])
                predictions.append(predicted_np[sample_index : sample_index + 1])
                presence_scores.append(np.array([presence_score], dtype=np.float64))
    average_loss = float(np.mean(sample_losses)) if sample_losses else None
    return ground_truths, predictions, presence_scores, average_loss


def evaluate_predictions(ground_truths, predictions, presence_scores, config):
    return evaluate_set_metrics(
        ground_truths,
        predictions,
        presence_scores,
        config=_metric_config(config),
    )


def evaluate_constant_pose(ground_truths, constant_pose, config, presence_score):
    predictions = [constant_pose[None, :, :].astype(np.float64) for _ in ground_truths]
    presence_scores = [np.array([presence_score], dtype=np.float64) for _ in ground_truths]
    return evaluate_predictions(ground_truths, predictions, presence_scores, config)


def compute_mean_pose(train_dataset):
    poses = []
    for sample in train_dataset.samples:
        pose = train_dataset.read_pose(sample["keypoint_path"]).numpy().astype(np.float64)
        poses.append(pose[0])
    return np.mean(np.stack(poses, axis=0), axis=0), len(poses)


def save_state_dict(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    return path


def load_state_dict(path, config, device):
    model = build_model(config, device)
    state_dict = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(device)
    return model


def _write_decision(artifact_root, status, reason, evidence):
    decision = {"status": status, "reason": reason, "evidence": evidence}
    path = _write_json(Path(artifact_root) / "decision.json", decision)
    print(json.dumps(decision, indent=2, sort_keys=True))
    print(f"decision={path}")


def run_overfit(artifact_root, freeze_config, pilot_root, device_name):
    artifact_root, freeze_config, pilot_root = _resolve_paths(artifact_root, freeze_config, pilot_root)
    freeze = _load_frozen_config(freeze_config)
    config = freeze["config"]
    validation_path = artifact_root / "preprocessing_validation_v2.json"
    if not validation_path.is_file():
        raise FileNotFoundError(f"Pilot preprocessing validation is required first: {validation_path}")
    validation = _load_json(validation_path)
    if validation["counts"]["failed_ids"] != 0:
        raise ValueError("Pilot preprocessing validation contains failures; training is blocked")

    device = _resolve_device(device_name)
    _seed_everything(config["seed"])
    train_dataset, development_dataset = _make_datasets(freeze, pilot_root, freeze_config)
    model = build_model(config, device)
    model.train()
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
    )

    overfit_config = config["overfit"]
    ordered_loader = make_single_person_dataloader(
        train_dataset,
        is_training=False,
        generator=torch.Generator().manual_seed(config["seed"]),
        batch_size=overfit_config["batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    batch = next(iter(ordered_loader))
    csi = batch["csi"].to(device)
    pose = batch["pose"].to(device)
    batch_sample_ids = train_dataset.sample_ids[: csi.shape[0]]

    with torch.no_grad():
        initial_prediction, _ = model(csi)
        initial_loss = _mean_per_joint_error(initial_prediction, pose)
    if not torch.isfinite(initial_loss):
        raise ValueError(f"Non-finite initial overfit loss: {initial_loss}")
    initial_loss_value = float(initial_loss.detach().cpu())
    initial_mpjpe_value = initial_loss_value

    losses = [initial_loss_value]
    for step in range(1, overfit_config["steps"] + 1):
        optimizer.zero_grad()
        predicted_pose, _ = model(csi)
        loss = _mean_per_joint_error(predicted_pose, pose)
        if not torch.isfinite(loss):
            raise ValueError(f"Non-finite overfit loss at step {step}: {loss}")
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu()))

    with torch.no_grad():
        final_prediction, _ = model(csi)
        final_loss = _mean_per_joint_error(final_prediction, pose)
    if not torch.isfinite(final_loss):
        raise ValueError(f"Non-finite final overfit loss: {final_loss}")
    final_loss_value = float(final_loss.detach().cpu())
    final_mpjpe_value = final_loss_value

    criterion = overfit_config["criterion"]
    failures = []
    if not all(np.isfinite(losses)):
        failures.append("non-finite loss sequence")
    if final_loss_value >= initial_loss_value:
        failures.append("final loss is not less than initial loss")
    if final_mpjpe_value > criterion["max_final_mpjpe_stored_units"]:
        failures.append(
            f"final MPJPE {final_mpjpe_value:.9f} exceeds "
            f"{criterion['max_final_mpjpe_stored_units']:.9f} stored units"
        )
    if initial_mpjpe_value <= 0:
        failures.append("initial MPJPE is not positive, so the relative criterion is undefined")
    elif final_mpjpe_value / initial_mpjpe_value > criterion["max_final_to_initial_mpjpe_ratio"]:
        failures.append(
            f"final/initial MPJPE ratio {final_mpjpe_value / initial_mpjpe_value:.9f} exceeds "
            f"{criterion['max_final_to_initial_mpjpe_ratio']:.9f}"
        )
    passed = not failures

    result = {
        "status": "PASS" if passed else "FAIL",
        "device": str(device),
        "steps": overfit_config["steps"],
        "batch_size": int(csi.shape[0]),
        "batch_sample_ids": batch_sample_ids,
        "initial_loss": initial_loss_value,
        "final_loss": final_loss_value,
        "initial_mpjpe_stored_units": initial_mpjpe_value,
        "final_mpjpe_stored_units": final_mpjpe_value,
        "final_to_initial_mpjpe_ratio": (
            final_mpjpe_value / initial_mpjpe_value if initial_mpjpe_value > 0 else None
        ),
        "criterion": criterion,
        "failures": failures,
        "loss_trajectory": {
            "first": losses[0],
            "last": losses[-1],
            "min": float(np.min(losses)),
            "max": float(np.max(losses)),
        },
        "model_parameters": _parameter_counts(model),
        "train_dataset_size": len(train_dataset),
        "development_dataset_size": len(development_dataset),
    }
    _write_json(artifact_root / "overfit_result.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        _write_decision(
            artifact_root,
            "BLOCKED",
            "The frozen numerical overfit criterion failed; Step-4B training is blocked.",
            {"overfit_result": str(artifact_root / "overfit_result.json"), "failures": failures},
        )
        return 2
    return 0


def run_training(artifact_root, freeze_config, pilot_root, device_name):
    artifact_root, freeze_config, pilot_root = _resolve_paths(artifact_root, freeze_config, pilot_root)
    freeze = _load_frozen_config(freeze_config)
    config = freeze["config"]
    overfit_path = artifact_root / "overfit_result.json"
    if not overfit_path.is_file():
        _write_decision(
            artifact_root,
            "BLOCKED",
            "The overfit gate result is missing; run the overfit mode before training.",
            {"expected": str(overfit_path)},
        )
        return 2
    overfit_result = _load_json(overfit_path)
    if overfit_result["status"] != "PASS":
        _write_decision(
            artifact_root,
            "BLOCKED",
            "The overfit gate did not pass; the epoch run is blocked.",
            {"overfit_result": str(overfit_path), "failures": overfit_result["failures"]},
        )
        return 2

    device = _resolve_device(device_name)
    _seed_everything(config["seed"])
    train_dataset, development_dataset = _make_datasets(freeze, pilot_root, freeze_config)
    train_loader = make_single_person_dataloader(
        train_dataset,
        is_training=True,
        generator=torch.Generator().manual_seed(config["seed"]),
        batch_size=config["training"]["train_batch_size"],
        num_workers=config["training"]["num_workers"],
    )
    development_loader = make_single_person_dataloader(
        development_dataset,
        is_training=False,
        generator=torch.Generator().manual_seed(config["seed"]),
        batch_size=config["training"]["development_batch_size"],
        num_workers=config["training"]["num_workers"],
    )

    model = build_model(config, device)
    optimizer = torch.optim.SGD(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=config["training"]["learning_rate"],
        momentum=config["training"]["momentum"],
    )
    presence_score = float(config["presence_score"])
    stored_unit_to_mm = float(config["metrics"]["stored_unit_to_mm"])
    checkpoint_path = artifact_root / "checkpoints" / "best_development_state_dict.pt"
    curves = []
    best_selection_error_mm = float("inf")
    best_epoch = None
    start_time = time.perf_counter()

    try:
        for epoch in range(1, config["training"]["epochs"] + 1):
            model.train()
            batch_losses = []
            for batch in train_loader:
                csi = batch["csi"].to(device)
                pose = batch["pose"].to(device)
                optimizer.zero_grad()
                predicted_pose, _ = model(csi)
                if predicted_pose.shape != pose.shape:
                    raise AssertionError(
                        f"Prediction shape {tuple(predicted_pose.shape)} does not match label shape {tuple(pose.shape)}"
                    )
                loss = _mean_per_joint_error(predicted_pose, pose)
                if not torch.isfinite(loss):
                    raise ValueError(f"Non-finite training loss in epoch {epoch}: {loss}")
                loss.backward()
                optimizer.step()
                batch_losses.append(float(loss.detach().cpu()))

            average_train_loss = float(np.mean(batch_losses))
            ground_truths, predictions, presence_scores, development_loss = collect_predictions(
                model, development_loader, device, presence_score
            )
            metrics = evaluate_predictions(ground_truths, predictions, presence_scores, config)
            matched_mpjpe_mm = metrics["overall"]["matched_mpjpe_mm"]
            selection_fallback = matched_mpjpe_mm is None
            selection_error_mm = (
                development_loss * stored_unit_to_mm if selection_fallback else matched_mpjpe_mm
            )
            improved = selection_error_mm < best_selection_error_mm
            if improved:
                best_selection_error_mm = selection_error_mm
                best_epoch = epoch
                save_state_dict(model, checkpoint_path)

            epoch_record = {
                "epoch": epoch,
                "train_loss_stored_units": average_train_loss,
                "development_loss_stored_units": development_loss,
                "development_matched_mpjpe_mm": matched_mpjpe_mm,
                "development_matched_pampjpe_mm": metrics["overall"]["matched_pampjpe_mm"],
                "development_pck_percent": metrics["overall"]["pck_percent"],
                "development_root_error_mm": metrics["overall"]["root_error_mm"],
                "development_exact_count_accuracy_percent": metrics["overall"][
                    "exact_count_accuracy_percent"
                ],
                "development_person_f1": metrics["overall"]["person_f1"],
                "selection_error_mm": selection_error_mm,
                "selection_fallback_to_development_loss": selection_fallback,
                "best_so_far": improved,
            }
            curves.append(epoch_record)
            _write_json(artifact_root / "learning_curves.json", {"epochs": curves})
            print(json.dumps(epoch_record, indent=2, sort_keys=True))
    except ValueError as exc:
        _write_decision(
            artifact_root,
            "BLOCKED",
            "Training produced a non-finite or invalid value; Step-4B is blocked for engineering diagnosis.",
            {"error": str(exc), "learning_curves": str(artifact_root / "learning_curves.json")},
        )
        return 3

    runtime_seconds = time.perf_counter() - start_time
    reloaded_model = load_state_dict(checkpoint_path, config, device)
    reloaded_ground_truths, reloaded_predictions, reloaded_scores, reloaded_loss = collect_predictions(
        reloaded_model, development_loader, device, presence_score
    )
    reloaded_metrics = evaluate_predictions(
        reloaded_ground_truths, reloaded_predictions, reloaded_scores, config
    )
    reloaded_selection = reloaded_metrics["overall"]["matched_mpjpe_mm"]
    if reloaded_selection is None:
        reloaded_selection = reloaded_loss * stored_unit_to_mm
    if not np.isclose(reloaded_selection, best_selection_error_mm, rtol=1e-6, atol=1e-9):
        raise AssertionError(
            f"Reloaded checkpoint selection error {reloaded_selection} does not match best "
            f"recorded error {best_selection_error_mm}"
        )

    mean_pose, mean_pose_source_count = compute_mean_pose(train_dataset)
    mean_metrics = evaluate_constant_pose(
        reloaded_ground_truths, mean_pose, config, presence_score
    )
    model_overall = reloaded_metrics["overall"]
    mean_overall = mean_metrics["overall"]
    model_mpjpe = model_overall["matched_mpjpe_mm"]
    mean_mpjpe = mean_overall["matched_mpjpe_mm"]
    preliminary_go = (
        model_mpjpe is not None
        and mean_mpjpe is not None
        and model_mpjpe < mean_mpjpe
    )

    np.savez(
        artifact_root / "development_predictions.npz",
        sample_ids=np.array(development_dataset.sample_ids),
        ground_truths=np.stack(reloaded_ground_truths, axis=0),
        model_predictions=np.stack(reloaded_predictions, axis=0),
        mean_pose_predictions=np.stack(
            [mean_pose[None, :, :] for _ in reloaded_ground_truths], axis=0
        ),
        presence_scores=np.stack(reloaded_scores, axis=0),
        mean_pose=np.stack([mean_pose], axis=0),
    )

    evaluation = {
        "status": "PASS",
        "device": str(device),
        "runtime_seconds": runtime_seconds,
        "best_epoch": best_epoch,
        "best_development_checkpoint": str(checkpoint_path),
        "checkpoint_format": "state_dict",
        "checkpoint_reload_verified": True,
        "selection_error_mm": best_selection_error_mm,
        "selection_basis": "lowest development matched MPJPE in millimetres",
        "evaluated_people": len(reloaded_ground_truths),
        "mean_pose_source": "frozen internal-train labels only",
        "mean_pose_source_count": mean_pose_source_count,
        "model_overall": model_overall,
        "model_per_person_count": reloaded_metrics["per_person_count"],
        "mean_pose_overall": mean_overall,
        "mean_pose_per_person_count": mean_metrics["per_person_count"],
        "metric_config": config["metrics"],
        "coordinate_order": "unchanged; no axis reordering performed",
        "unit_conversion": "stored coordinates interpreted as metres and multiplied by exactly 1000 for research distance metrics",
        "learning_curves": curves,
        "overfit": overfit_result,
        "model_parameters": _parameter_counts(model),
    }
    _write_json(artifact_root / "evaluation_result.json", evaluation)

    if preliminary_go:
        status = "GO"
        reason = (
            "Preliminary Step-4C composition-control GO: finite one-seed training completed, "
            "label access followed the frozen internal split, and development matched MPJPE "
            "was strictly lower than the frozen-train mean-pose predictor."
        )
    else:
        status = "NOT_READY"
        reason = (
            "Step-4B is NOT READY for composition controls: finite training completed, but "
            "development matched MPJPE did not strictly beat the frozen-train mean-pose predictor."
        )
    _write_decision(
        artifact_root,
        status,
        reason,
        {
            "model_development_matched_mpjpe_mm": model_mpjpe,
            "mean_pose_development_matched_mpjpe_mm": mean_mpjpe,
            "evaluation_result": str(artifact_root / "evaluation_result.json"),
            "learning_curves": str(artifact_root / "learning_curves.json"),
        },
    )
    return 0


def parse_args():
    parser = argparse.ArgumentParser(description="Run the bounded Step-4B single-person sanity gate.")
    parser.add_argument("mode", choices=("overfit", "train"))
    parser.add_argument("--artifact-root", type=Path, default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--freeze-config", type=Path, default=None)
    parser.add_argument("--pilot-root", type=Path, default=None)
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    return parser.parse_args()


def main():
    args = parse_args()
    if args.mode == "overfit":
        return run_overfit(args.artifact_root, args.freeze_config, args.pilot_root, args.device)
    return run_training(args.artifact_root, args.freeze_config, args.pilot_root, args.device)


if __name__ == "__main__":
    raise SystemExit(main())
