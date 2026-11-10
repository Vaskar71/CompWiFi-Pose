"""Step-5 runner (B4/B5): protocol freeze, smoke, overfit, runs, selection, verdict.

Subcommands (all take ``--repo-root`` pointing at the checkout that holds ``data/``):
  freeze-protocol | smoke | overfit --run R1 | train --run R1 --seed 0 | select | verdict

The process must be started with PYTHONHASHSEED=0 and CUBLAS_WORKSPACE_CONFIG=:4096:8.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

import step4b_r_baseline as r_base
from feeder.piw3d_single_person import make_single_person_dataloader
from step5_config import (
    RUN_IDS,
    SIMPLICITY_ORDER,
    STEP5_SOURCE_FILES,
    resolve_path,
    step5_config_for_json,
    validate_step5_config,
)


CODE_ROOT = Path(__file__).resolve().parent
PROTOCOL_DOC = "docs/STEP_5_PROTOCOL.md"
FROZEN_CONFIG_NAME = "step5_config_frozen.json"
SOURCE_HASHES_NAME = "source_hashes.json"
SPLITS = ("train", "temporal_holdout", "group_dev")


# ----------------------------------------------------------------------------- io helpers

def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def source_hashes(code_root=CODE_ROOT):
    files = list(STEP5_SOURCE_FILES) + [PROTOCOL_DOC]
    return {name: _sha256_file(Path(code_root) / name) for name in files}


# ----------------------------------------------------------------------------- protocol freeze

def freeze_protocol(repo_root, code_root=CODE_ROOT):
    freeze_dir = resolve_path(repo_root, "protocol_freeze_dir")
    if freeze_dir.exists():
        raise FileExistsError(f"Protocol freeze already exists: {freeze_dir}")
    if not (Path(code_root) / PROTOCOL_DOC).is_file():
        raise FileNotFoundError(f"{PROTOCOL_DOC} must be written before the protocol freeze")
    config = step5_config_for_json()
    record = {
        "frozen_at_utc": _now(),
        "config_sha256": hashlib.sha256(json.dumps(config, sort_keys=True).encode("utf-8")).hexdigest(),
        "source_hashes": source_hashes(code_root),
    }
    freeze_dir.mkdir(parents=True)
    _write_json(freeze_dir / FROZEN_CONFIG_NAME, config)
    _write_json(freeze_dir / SOURCE_HASHES_NAME, record)
    return record


def verify_protocol_freeze(repo_root, code_root=CODE_ROOT):
    freeze_dir = resolve_path(repo_root, "protocol_freeze_dir")
    frozen_config = _load_json(freeze_dir / FROZEN_CONFIG_NAME)
    validate_step5_config(frozen_config)
    record = _load_json(freeze_dir / SOURCE_HASHES_NAME)
    current = source_hashes(code_root)
    changed = sorted(name for name, digest in record["source_hashes"].items() if current.get(name) != digest)
    if changed:
        raise RuntimeError("Step-5 sources changed after the protocol freeze: " + ", ".join(changed))
    return {"frozen_at_utc": record["frozen_at_utc"], "source_hashes_unchanged": True}


# ----------------------------------------------------------------------------- runtime

def setup_runtime(device_name="auto"):
    environment = r_base.validate_process_environment()
    if not environment["valid"]:
        raise RuntimeError("; ".join(environment["failures"]))
    r_base.load_runtime_modules()
    deterministic = r_base.configure_deterministic_runtime()
    device = r_base._resolve_device(device_name)
    return device, deterministic


class CachedSplit(Dataset):
    """In-memory copy of a Step-5 dataset: identical tensors, loaded once."""

    batch_keys = ("csi", "pose")

    def __init__(self, dataset):
        items = [dataset[index] for index in range(len(dataset))]
        self.csi = torch.stack([item["csi"] for item in items])
        self.pose = torch.stack([item["pose"] for item in items])
        self.sample_ids = list(dataset.sample_ids)
        self.source_dataset = dataset

    def __len__(self):
        return self.csi.shape[0]

    def __getitem__(self, index):
        return {"csi": self.csi[index], "pose": self.pose[index]}


def load_splits(repo_root, input_norm, splits=SPLITS):
    from feeder.piw3d_step5 import Step5SinglePersonDataset, load_input_stats

    config = step5_config_for_json()
    data_root = resolve_path(repo_root, "data_root")
    freeze_dir = resolve_path(repo_root, "freeze_dir")
    stats = None
    if input_norm == "train_zscore":
        stats = load_input_stats(freeze_dir / config["input_norm"]["stats_file"])
    loaded = {}
    for split in splits:
        dataset = Step5SinglePersonDataset(
            data_root,
            data_root / "train_data" / f"{split}_ids.txt",
            expected_split=split,
            split_assignments_path=freeze_dir / "split_assignments.json",
            input_norm=input_norm,
            input_stats=stats,
        )
        loaded[split] = CachedSplit(dataset)
    seen = set()
    for split in loaded.values():
        if seen & set(split.sample_ids):
            raise AssertionError("Step-5 splits share IDs")
        seen |= set(split.sample_ids)
    return loaded


def load_codec_stats(repo_root):
    from model.research_models import load_codec_stats as _load

    config = step5_config_for_json()
    return _load(resolve_path(repo_root, "freeze_dir") / config["target_codec"]["stats_file"])


def build_model(run_id, repo_root, device, seed):
    from model.research_models import build_research_model

    config = step5_config_for_json()
    r_base._seed_everything(seed)
    return build_research_model(config["runs"][run_id]["model"], load_codec_stats(repo_root), config, device)


# ----------------------------------------------------------------------------- metrics

def pose_statistics(predictions, ground_truths, shuffle_seed=None):
    """Distances in stored units. Inputs (N,1,14,3)."""
    pred = np.asarray(predictions, dtype=np.float64)[:, 0]
    gt = np.asarray(ground_truths, dtype=np.float64)[:, 0]
    pred_c = pred.mean(axis=1)
    gt_c = gt.mean(axis=1)
    result = {
        "samples": int(pred.shape[0]),
        "abs_mpjpe": float(np.mean(np.linalg.norm(pred - gt, axis=-1))),
        "centroid_error": float(np.mean(np.linalg.norm(pred_c - gt_c, axis=-1))),
        "rel_mpjpe": float(
            np.mean(np.linalg.norm((pred - pred_c[:, None]) - (gt - gt_c[:, None]), axis=-1))
        ),
        "pred_std_across_samples": float(np.mean(pred.std(axis=0))),
        "pred_std_across_joints": float(np.mean(pred.std(axis=1))),
        "pred_centroid_std_mean_axes": float(np.mean(pred_c.std(axis=0))),
        "gt_centroid_std_mean_axes": float(np.mean(gt_c.std(axis=0))),
    }
    if shuffle_seed is not None:
        permutation = np.random.default_rng(shuffle_seed).permutation(gt.shape[0])
        result["shuffled_pairing_abs_mpjpe"] = float(
            np.mean(np.linalg.norm(pred - gt[permutation], axis=-1))
        )
    return result


def research_metrics(predictions, ground_truths):
    from research_metrics import ResearchMetricAccumulator

    settings = dict(step5_config_for_json()["evaluation"]["research_metrics"])
    settings.pop("people_dim")
    settings["pck_scale_joint_indices"] = tuple(settings["pck_scale_joint_indices"])
    accumulator = ResearchMetricAccumulator(**settings)
    accumulator.update(np.asarray(ground_truths, dtype=np.float64), np.asarray(predictions, dtype=np.float64))
    metrics = accumulator.compute()
    metrics["mm_label"] = "provisional, assumes stored unit = metre"
    return metrics


def predict(model, split, device, batch_size):
    was_training = model.training
    model.eval()
    outputs = []
    try:
        with torch.no_grad():
            for start in range(0, len(split), batch_size):
                csi = split.csi[start : start + batch_size].to(device)
                pose = model(csi)
                if pose.shape != (csi.shape[0], 1, 14, 3):
                    raise AssertionError(f"Unexpected prediction shape {tuple(pose.shape)}")
                if not torch.isfinite(pose).all():
                    raise ValueError("Non-finite prediction")
                outputs.append(pose.detach().cpu().numpy().astype(np.float64))
    finally:
        model.train(was_training)
    return np.concatenate(outputs, axis=0)


def mean_pose_baselines(splits):
    train_dataset = splits["train"].source_dataset
    mean_pose, count = r_base.compute_mean_pose(train_dataset)
    baselines = {"mean_pose_source": f"train split labels ({count} samples)", "mean_pose": mean_pose.tolist()}
    for name, split in splits.items():
        gt = split.pose.numpy().astype(np.float64)
        constant = np.broadcast_to(mean_pose[None, None], gt.shape)
        stats = pose_statistics(constant, gt)
        reused_abs, _ = r_base.constant_pose_mpjpe(split.source_dataset, mean_pose)
        if not math.isclose(reused_abs, stats["abs_mpjpe"], rel_tol=1e-9, abs_tol=1e-12):
            raise AssertionError("Mean-pose MPJPE disagrees with step4b_r_baseline.constant_pose_mpjpe")
        baselines[name] = {
            "mean_pose_abs_mpjpe": stats["abs_mpjpe"],
            "mean_pose_centroid_error": stats["centroid_error"],
            "mean_skeleton_rel_mpjpe": stats["rel_mpjpe"],
            "gt_centroid_std_mean_axes": stats["gt_centroid_std_mean_axes"],
        }
    return baselines


# ----------------------------------------------------------------------------- training

def lr_factor(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def make_optimizer(model, learning_rate, weight_decay):
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    return torch.optim.AdamW(parameters, lr=learning_rate, weight_decay=weight_decay)


def training_step(model, optimizer, csi, pose, clip_norm):
    loss, _ = model.loss(csi, pose)
    if not torch.isfinite(loss):
        raise ValueError(f"Non-finite training loss: {loss.item()}")
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for name, parameter in model.named_parameters():
        if parameter.requires_grad and parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise ValueError(f"Non-finite gradient in {name}")
    grad_norm = torch.nn.utils.clip_grad_norm_(
        [parameter for parameter in model.parameters() if parameter.requires_grad], clip_norm
    )
    optimizer.step()
    return float(loss.detach().cpu()), float(grad_norm.detach().cpu())


def evaluate_all(model, splits, device, eval_batch_size, shuffle_seed, with_research=False):
    evaluation = {}
    predictions = {}
    for name, split in splits.items():
        pred = predict(model, split, device, eval_batch_size)
        gt = split.pose.numpy().astype(np.float64)
        evaluation[name] = pose_statistics(pred, gt, shuffle_seed=shuffle_seed)
        if with_research:
            evaluation[name]["research_metrics"] = research_metrics(pred, gt)
        predictions[name] = pred
    return evaluation, predictions


def gate_decision(evaluation, baselines, gates):
    train = evaluation["train"]
    holdout = evaluation["temporal_holdout"]
    dev = evaluation["group_dev"]
    g3_centroid = train["pred_centroid_std_mean_axes"] >= gates["G3_centroid_std_ratio"] * train["gt_centroid_std_mean_axes"]
    g3_shuffle = train["abs_mpjpe"] <= gates["G3_shuffle_ratio"] * train["shuffled_pairing_abs_mpjpe"]
    g4 = train["abs_mpjpe"] <= gates["G4_train_ratio"] * baselines["train"]["mean_pose_abs_mpjpe"]
    g5_abs = holdout["abs_mpjpe"] <= gates["G5_holdout_ratio"] * baselines["temporal_holdout"]["mean_pose_abs_mpjpe"]
    g5_rel = holdout["rel_mpjpe"] < baselines["temporal_holdout"]["mean_skeleton_rel_mpjpe"]
    g6_abs = dev["abs_mpjpe"] <= gates["G6_dev_ratio"] * baselines["group_dev"]["mean_pose_abs_mpjpe"]
    g6_rel = dev["rel_mpjpe"] <= gates["G6_dev_relative_ratio"] * baselines["group_dev"]["mean_skeleton_rel_mpjpe"]
    return {
        "G3": {"pass": bool(g3_centroid and g3_shuffle), "centroid_std": bool(g3_centroid), "shuffle": bool(g3_shuffle),
               "pred_centroid_std": train["pred_centroid_std_mean_axes"], "gt_centroid_std": train["gt_centroid_std_mean_axes"],
               "train_abs_mpjpe": train["abs_mpjpe"], "train_shuffled_abs_mpjpe": train["shuffled_pairing_abs_mpjpe"]},
        "G4": {"pass": bool(g4), "train_abs_mpjpe": train["abs_mpjpe"],
               "ratio_to_mean_pose": train["abs_mpjpe"] / baselines["train"]["mean_pose_abs_mpjpe"]},
        "G5": {"pass": bool(g5_abs and g5_rel), "abs": bool(g5_abs), "rel": bool(g5_rel),
               "abs_ratio_to_mean_pose": holdout["abs_mpjpe"] / baselines["temporal_holdout"]["mean_pose_abs_mpjpe"],
               "rel_ratio_to_mean_skeleton": holdout["rel_mpjpe"] / baselines["temporal_holdout"]["mean_skeleton_rel_mpjpe"]},
        "G6": {"pass": bool(g6_abs and g6_rel), "abs": bool(g6_abs), "rel": bool(g6_rel),
               "abs_ratio_to_mean_pose": dev["abs_mpjpe"] / baselines["group_dev"]["mean_pose_abs_mpjpe"],
               "rel_ratio_to_mean_skeleton": dev["rel_mpjpe"] / baselines["group_dev"]["mean_skeleton_rel_mpjpe"]},
    }


def write_crashed_run(run_dir, run_id, seed, epoch, step, reason, curves):
    failed = {"pass": False, "crashed": True}
    result = {"run_id": run_id, "seed": seed, "crashed": True, "reason": reason, "epoch": epoch,
              "steps_completed": step, "completed_at_utc": _now()}
    decision = {
        "run_id": run_id,
        "seed": seed,
        "crashed": True,
        "reason": reason,
        "gates": {gate: dict(failed) for gate in ("G3", "G4", "G5", "G6")},
        "G3_to_G5_pass": False,
        "G3_to_G6_pass": False,
        "group_dev_abs_mpjpe": None,
        "checkpoint": "none (run crashed)",
    }
    _write_json(run_dir / "learning_curves.json", curves)
    _write_json(run_dir / "result.json", result)
    _write_json(run_dir / "decision.json", decision)
    return result, decision


def run_training(repo_root, run_id, seed, device_name="auto", epochs_override=None):
    config = step5_config_for_json()
    protocol = verify_protocol_freeze(repo_root)
    training = config["training"]
    run_dir = resolve_path(repo_root, "artifact_root") / "runs" / f"{run_id}_seed{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    device, deterministic = setup_runtime(device_name)
    splits = load_splits(repo_root, config["runs"][run_id]["input_norm"])
    baselines = mean_pose_baselines(splits)
    run_dir.mkdir(parents=True)
    environment = r_base.capture_environment(device, deterministic)

    model = build_model(run_id, repo_root, device, seed)
    optimizer = make_optimizer(model, training["peak_learning_rate"], training["weight_decay"])
    audit = r_base.parameter_audit(model, optimizer)
    initial_state = {name: value.detach().clone() for name, value in model.state_dict().items()}
    generator = torch.Generator().manual_seed(seed)
    loader = make_single_person_dataloader(
        splits["train"], is_training=True, generator=generator, batch_size=training["batch_size"],
        num_workers=training["num_workers"],
    )
    epochs = epochs_override or training["epochs"]
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, training["warmup_steps"], total_steps)
    )
    shuffle_seed = config["evaluation"]["shuffled_pairing_seed"]
    curves = []
    started = time.time()
    step = 0
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        losses = []
        grad_norms = []
        for batch in loader:
            try:
                loss, grad_norm = training_step(
                    model, optimizer, batch["csi"].to(device), batch["pose"].to(device), training["grad_clip_norm"]
                )
            except ValueError as exc:  # predeclared: a non-finite run fails every gate
                return write_crashed_run(run_dir, run_id, seed, epoch, step, str(exc), curves)
            scheduler.step()
            step += 1
            losses.append(loss)
            grad_norms.append(grad_norm)
        evaluation, _ = evaluate_all(model, splits, device, training["eval_batch_size"], shuffle_seed)
        record = {
            "epoch": epoch,
            "steps": step,
            "learning_rate_end": optimizer.param_groups[0]["lr"],
            "train_loss_mean": float(np.mean(losses)),
            "grad_norm_mean_preclip": float(np.mean(grad_norms)),
            "epoch_seconds": time.time() - epoch_started,
            **{name: evaluation[name] for name in evaluation},
        }
        curves.append(record)
        print(
            f"[{run_id} seed{seed}] epoch {epoch}/{epochs} loss {record['train_loss_mean']:.4f} "
            + " ".join(
                f"{name}:abs={evaluation[name]['abs_mpjpe']:.4f}/rel={evaluation[name]['rel_mpjpe']:.4f}"
                f"/std={evaluation[name]['pred_std_across_samples']:.3g}"
                for name in SPLITS if name in evaluation
            )
            + f" ({record['epoch_seconds']:.1f}s)",
            flush=True,
        )
        _write_json(run_dir / "learning_curves.json", curves)
        if epoch == 1:
            projected_hours = record["epoch_seconds"] * epochs / 3600.0
            if projected_hours > training["first_epoch_budget_hours"]:
                raise RuntimeError(
                    f"First epoch projects {projected_hours:.2f} h > budget; stop and ask the user"
                )

    runtime_seconds = time.time() - started
    evaluation, predictions = evaluate_all(
        model, splits, device, training["eval_batch_size"], shuffle_seed, with_research=True
    )
    state_path = run_dir / "final_state_dict.pt"
    torch.save(model.state_dict(), state_path)
    reloaded = build_model(run_id, repo_root, device, seed + 1000)
    reloaded.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    reload_evaluation, reload_predictions = evaluate_all(
        reloaded, splits, device, training["eval_batch_size"], shuffle_seed, with_research=True
    )
    reload_exact = all(np.array_equal(predictions[name], reload_predictions[name]) for name in predictions)
    reload_exact = reload_exact and reload_evaluation == evaluation and r_base.state_dicts_equal(model, reloaded)
    for name, pred in predictions.items():
        np.savez(
            run_dir / f"predictions_{name}.npz",
            predictions=pred,
            ground_truths=splits[name].pose.numpy().astype(np.float64),
            sample_ids=np.asarray(splits[name].sample_ids),
        )
    changes = r_base.summarize_parameter_changes(model, initial_state)
    changes.pop("per_parameter")
    gates = gate_decision(evaluation, baselines, config["gates"])
    result = {
        "run_id": run_id,
        "seed": seed,
        "run_config": config["runs"][run_id],
        "protocol": protocol,
        "epochs": epochs,
        "total_steps": step,
        "runtime_seconds": runtime_seconds,
        "final_epoch_evaluation": evaluation,
        "baselines": baselines,
        "reload_reproduces_final_metrics_exactly": bool(reload_exact),
        "parameter_audit": {key: value for key, value in audit.items() if key != "records"},
        "parameter_changes": changes,
        "environment": environment,
        "label_access": config["label_access"],
        "distance_units": "stored units; *_mm fields provisional (assumes stored unit = metre)",
        "completed_at_utc": _now(),
    }
    _write_json(run_dir / "result.json", result)
    decision = {
        "run_id": run_id,
        "seed": seed,
        "gates": gates,
        "G3_to_G5_pass": bool(gates["G3"]["pass"] and gates["G4"]["pass"] and gates["G5"]["pass"]),
        "G3_to_G6_pass": bool(all(gates[g]["pass"] for g in ("G3", "G4", "G5", "G6"))),
        "group_dev_abs_mpjpe": evaluation["group_dev"]["abs_mpjpe"],
        "reload_exact": bool(reload_exact),
        "checkpoint": "final epoch only",
    }
    _write_json(run_dir / "decision.json", decision)
    return result, decision


# ----------------------------------------------------------------------------- smoke / overfit

def smoke_run(repo_root, run_id, device, smoke_dir, batch_size=8):
    config = step5_config_for_json()
    checks = {}
    splits = load_splits(repo_root, config["runs"][run_id]["input_norm"], splits=("train", "group_dev"))
    generator = torch.Generator().manual_seed(0)
    loader = make_single_person_dataloader(splits["train"], True, generator, batch_size)
    batch = next(iter(loader))
    checks["batch_keys"] = sorted(batch) == ["csi", "pose"]
    checks["csi_shape"] = tuple(batch["csi"].shape) == (batch_size, 3, 180, 20)
    checks["pose_shape"] = tuple(batch["pose"].shape) == (batch_size, 1, 14, 3)
    model = build_model(run_id, repo_root, device, 0)
    csi = batch["csi"].to(device)
    pose = batch["pose"].to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(csi)
        tokens, pooled = model.forward_features(csi)
    checks["forward_shape"] = tuple(prediction.shape) == (batch_size, 1, 14, 3)
    checks["forward_features_shapes"] = tokens.ndim == 3 and pooled.shape == (batch_size, tokens.shape[2])
    if model.target_mode == "codec":
        mean_pose = (model.codec.mu_c[None, :] + model.codec.mu_r).detach()
        checks["untrained_equals_train_mean_pose"] = bool(
            torch.allclose(prediction[:, 0], mean_pose.expand_as(prediction[:, 0]), atol=1e-5, rtol=0)
        )
    model.train()
    optimizer = make_optimizer(model, 1e-3, 0.0)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    loss, grad_norm = training_step(model, optimizer, csi, pose, 1.0)
    checks["loss_finite"] = math.isfinite(loss)
    checks["backward_grad_norm_positive"] = grad_norm > 0
    changed = sum(not torch.equal(before[name], value.detach()) for name, value in model.named_parameters())
    checks["optimizer_updates_parameters"] = changed > 0
    state_path = Path(smoke_dir) / f"{run_id}_state_dict.pt"
    torch.save(model.state_dict(), state_path)
    reloaded = build_model(run_id, repo_root, device, 123)
    reloaded.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    checks["state_dict_reload_exact"] = r_base.state_dicts_equal(model, reloaded)
    checks["reload_predictions_exact"] = bool(
        np.array_equal(predict(model, splits["group_dev"], device, 64), predict(reloaded, splits["group_dev"], device, 64))
    )
    evaluation = pose_statistics(predict(model, splits["group_dev"], device, 64), splits["group_dev"].pose.numpy())
    checks["evaluation_runs"] = math.isfinite(evaluation["abs_mpjpe"])
    return {
        "run_id": run_id,
        "checks": checks,
        "all_pass": all(checks.values()),
        "first_step_loss": loss,
        "first_step_grad_norm": grad_norm,
        "changed_parameter_tensors": int(changed),
        "total_parameter_tensors": len(before),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "token_shape": list(tokens.shape[1:]),
    }


def run_smoke(repo_root, device_name="auto"):
    protocol = verify_protocol_freeze(repo_root)
    smoke_dir = resolve_path(repo_root, "artifact_root") / "smoke"
    if smoke_dir.exists():
        raise FileExistsError(f"Refusing to overwrite smoke directory: {smoke_dir}")
    smoke_dir.mkdir(parents=True)
    device, deterministic = setup_runtime(device_name)
    results = {run_id: smoke_run(repo_root, run_id, device, smoke_dir) for run_id in RUN_IDS}
    record = {"protocol": protocol, "G1_pass": all(r["all_pass"] for r in results.values()), "runs": results,
              "device": str(device), "completed_at_utc": _now()}
    _write_json(smoke_dir / "smoke.json", record)
    return record


def overfit_indices(sample_count_total, sample_count):
    k = sample_count_total // sample_count
    if k < 1:
        raise ValueError("Not enough training samples for the overfit subset")
    return [index * k for index in range(sample_count)]


def run_overfit(repo_root, run_id, device_name="auto"):
    config = step5_config_for_json()
    overfit = config["overfit"]
    if run_id not in overfit["runs"]:
        raise ValueError(f"Overfit check is predeclared only for {overfit['runs']}")
    protocol = verify_protocol_freeze(repo_root)
    out_path = resolve_path(repo_root, "artifact_root") / "overfit" / f"{run_id}.json"
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite overfit result: {out_path}")
    device, _ = setup_runtime(device_name)
    splits = load_splits(repo_root, config["runs"][run_id]["input_norm"], splits=("train",))
    train = splits["train"]
    indices = overfit_indices(len(train), overfit["sample_count"])
    ids = [train.sample_ids[index] for index in indices]
    groups_all = {sample_id.rsplit("_", 1)[0] for sample_id in train.sample_ids}
    groups_subset = {sample_id.rsplit("_", 1)[0] for sample_id in ids}
    csi = train.csi[indices].to(device)
    pose = train.pose[indices].to(device)
    gt = pose.cpu().numpy().astype(np.float64)
    subset_mean = gt.mean(axis=0, keepdims=True)
    mean_pose_mpjpe = pose_statistics(np.broadcast_to(subset_mean, gt.shape), gt)["abs_mpjpe"]
    model = build_model(run_id, repo_root, device, overfit["seed"])
    optimizer = make_optimizer(model, overfit["learning_rate"], overfit["weight_decay"])
    curve = []
    for step in range(1, overfit["steps"] + 1):
        model.train()
        loss, grad_norm = training_step(model, optimizer, csi, pose, overfit["grad_clip_norm"])
        if step % 10 == 0 or step == 1:
            curve.append({"step": step, "loss": loss, "grad_norm_preclip": grad_norm})
    model.eval()
    with torch.no_grad():
        prediction = model(csi).cpu().numpy().astype(np.float64)
    final = pose_statistics(prediction, gt)
    ratio = final["abs_mpjpe"] / mean_pose_mpjpe
    record = {
        "run_id": run_id,
        "protocol": protocol,
        "sample_ids": ids,
        "all_train_groups_present": groups_subset == groups_all,
        "groups_present": len(groups_subset),
        "mean_pose_mpjpe_same_64": mean_pose_mpjpe,
        "final": final,
        "ratio": ratio,
        "pass_ratio": overfit["pass_ratio"],
        "G2_pass": bool(ratio <= overfit["pass_ratio"]),
        "curve": curve,
        "completed_at_utc": _now(),
    }
    _write_json(out_path, record)
    return record


# ----------------------------------------------------------------------------- selection / verdict

def _run_decision(repo_root, run_id, seed):
    path = resolve_path(repo_root, "artifact_root") / "runs" / f"{run_id}_seed{seed}" / "decision.json"
    return _load_json(path) if path.is_file() else None


def select_run(decisions, tie_relative):
    """decisions: {run_id: decision for seed 0}. Returns selected run id or None."""
    eligible = {run_id: d for run_id, d in decisions.items() if d is not None and d["G3_to_G5_pass"]}
    if not eligible:
        return None, eligible
    best = min(d["group_dev_abs_mpjpe"] for d in eligible.values())
    tied = [run_id for run_id, d in eligible.items() if d["group_dev_abs_mpjpe"] <= best * (1.0 + tie_relative)]
    selected = min(tied, key=SIMPLICITY_ORDER.index)
    return selected, eligible


def run_select(repo_root):
    config = step5_config_for_json()
    decisions = {run_id: _run_decision(repo_root, run_id, 0) for run_id in RUN_IDS}
    missing = [run_id for run_id, d in decisions.items() if d is None]
    if missing:
        raise FileNotFoundError("Seed-0 decisions missing for: " + ", ".join(missing))
    selected, eligible = select_run(decisions, config["gates"]["selection_tie_relative"])
    record = {
        "rule": "lowest final-epoch group_dev absolute MPJPE among runs passing G3-G5; ties (<1% relative) -> simpler (R1<R2<R3<R4)",
        "label_use": "single-person internal-development labels only; allowed under development and strict protocol",
        "eligible_runs": sorted(eligible),
        "group_dev_abs_mpjpe": {run_id: d["group_dev_abs_mpjpe"] for run_id, d in decisions.items()},
        "selected_run": selected,
        "completed_at_utc": _now(),
    }
    _write_json(resolve_path(repo_root, "artifact_root") / "selection.json", record)
    return record


def overfit_gate_for(repo_root, run_id):
    source = {"R1": "R1", "R2": "R1", "R3": "R3"}.get(run_id)
    if source is None:
        return {"applicable": False, "pass": None, "source": None}
    path = resolve_path(repo_root, "artifact_root") / "overfit" / f"{source}.json"
    return {"applicable": True, "pass": bool(_load_json(path)["G2_pass"]), "source": source}


def compute_verdict(g1_pass, g2, seed0_decisions, selected, selected_decisions):
    g4_fail_all = all(d is not None and not d["gates"]["G4"]["pass"] for d in seed0_decisions.values())
    if g4_fail_all:
        return "RED"
    if selected is None or not g1_pass or (g2["applicable"] and not g2["pass"]):
        return "INCONCLUSIVE"
    if any(d is None for d in selected_decisions.values()):
        return "INCONCLUSIVE"
    if not all(d["G3_to_G5_pass"] for d in selected_decisions.values()):
        return "INCONCLUSIVE"
    if all(d["G3_to_G6_pass"] for d in selected_decisions.values()):
        return "GREEN"
    return "YELLOW-POSITION"


def run_verdict(repo_root):
    artifact_root = resolve_path(repo_root, "artifact_root")
    smoke = _load_json(artifact_root / "smoke" / "smoke.json")
    selection = _load_json(artifact_root / "selection.json")
    seed0 = {run_id: _run_decision(repo_root, run_id, 0) for run_id in RUN_IDS}
    selected = selection["selected_run"]
    seeds = [0] + step5_config_for_json()["seeds"]["selected_run_additional"]
    selected_decisions = {seed: _run_decision(repo_root, selected, seed) for seed in seeds} if selected else {}
    g2 = overfit_gate_for(repo_root, selected) if selected else {"applicable": False, "pass": None}
    verdict = compute_verdict(smoke["G1_pass"], g2, seed0, selected, selected_decisions)
    record = {
        "verdict": verdict,
        "G1_pass": smoke["G1_pass"],
        "G2": g2,
        "selected_run": selected,
        "selected_run_seeds": {
            str(seed): (d["gates"] if d else None) for seed, d in selected_decisions.items()
        },
        "G4_fail_all_seed0": all(d is not None and not d["gates"]["G4"]["pass"] for d in seed0.values()),
        "completed_at_utc": _now(),
    }
    _write_json(artifact_root / "verdict.json", record)
    return record


# ----------------------------------------------------------------------------- cli

def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Step-5 single-person baseline runner")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-protocol")
    sub.add_parser("smoke")
    overfit = sub.add_parser("overfit")
    overfit.add_argument("--run", required=True, choices=RUN_IDS)
    train = sub.add_parser("train")
    train.add_argument("--run", required=True, choices=RUN_IDS)
    train.add_argument("--seed", required=True, type=int)
    sub.add_parser("select")
    sub.add_parser("verdict")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.command == "freeze-protocol":
        output = freeze_protocol(args.repo_root)
    elif args.command == "smoke":
        output = run_smoke(args.repo_root, args.device)
    elif args.command == "overfit":
        output = run_overfit(args.repo_root, args.run, args.device)
        output = {k: output[k] for k in ("run_id", "G2_pass", "ratio", "mean_pose_mpjpe_same_64", "all_train_groups_present")} | {"final_abs_mpjpe": output["final"]["abs_mpjpe"]}
    elif args.command == "train":
        _, output = run_training(args.repo_root, args.run, args.seed, args.device)
    elif args.command == "select":
        output = run_select(args.repo_root)
    else:
        output = run_verdict(args.repo_root)
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    sys.exit(main())
