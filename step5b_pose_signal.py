"""Step-5b runner: pose signal beyond recording identity (docs/STEP_5B_PROTOCOL.md).

Subcommands (all take ``--repo-root`` pointing at the checkout that holds ``data/``):
  freeze-protocol | smoke | overfit --run D2 | stage1 --run D1 | select | stage2 --seed 0 | verdict

Start processes with PYTHONHASHSEED=0 and CUBLAS_WORKSPACE_CONFIG=:4096:8.
"""

import argparse
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

import step4b_r_baseline as r_base
from feeder.piw3d_single_person import make_single_person_dataloader
from step5_config import step5_config_for_json
from step5_single_person_baseline import (
    CachedSplit,
    _write_json,
    lr_factor,
    make_optimizer,
    pose_statistics,
    predict,
    research_metrics,
    setup_runtime,
    training_step,
)
from step5b_config import (
    SIMPLICITY_ORDER,
    STAGE1_RUN_IDS,
    STEP5B_SOURCE_FILES,
    resolve_path,
    step5b_config_for_json,
    validate_step5b_config,
)


CODE_ROOT = Path(__file__).resolve().parent
PROTOCOL_DOC = "docs/STEP_5B_PROTOCOL.md"
PLAN_DOC = "docs/STEP_5B_PLAN.md"
FROZEN_CONFIG_NAME = "step5b_config_frozen.json"
SOURCE_HASHES_NAME = "source_hashes.json"
VIEW_ROLE = {
    "fit_train_dense": "train",
    "fit_train_sparse": "train",
    "fit_holdout": "temporal_holdout",
    "group_val": "train",
    "all_train_dense": "train",
    "all_train_sparse": "train",
    "all_holdout": "temporal_holdout",
    "group_dev": "group_dev",
}


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _sha256_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat()


def recording_of(sample_id):
    return sample_id.rsplit("_", 1)[0]


# ----------------------------------------------------------------------------- protocol freeze

def source_hashes(code_root=CODE_ROOT):
    files = list(STEP5B_SOURCE_FILES) + [PROTOCOL_DOC, PLAN_DOC]
    return {name: _sha256_file(Path(code_root) / name) for name in files}


def freeze_protocol(repo_root, code_root=CODE_ROOT):
    freeze_dir = resolve_path(repo_root, "protocol_freeze_dir")
    if freeze_dir.exists():
        raise FileExistsError(f"Protocol freeze already exists: {freeze_dir}")
    if not (Path(code_root) / PROTOCOL_DOC).is_file():
        raise FileNotFoundError(f"{PROTOCOL_DOC} must be written before the protocol freeze")
    config = step5b_config_for_json()
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
    validate_step5b_config(_load_json(freeze_dir / FROZEN_CONFIG_NAME))
    record = _load_json(freeze_dir / SOURCE_HASHES_NAME)
    current = source_hashes(code_root)
    changed = sorted(name for name, digest in record["source_hashes"].items() if current.get(name) != digest)
    if changed:
        raise RuntimeError("Step-5b sources changed after the protocol freeze: " + ", ".join(changed))
    return {"frozen_at_utc": record["frozen_at_utc"], "source_hashes_unchanged": True}


# ----------------------------------------------------------------------------- data

def load_view(repo_root, view, input_norm, stats_suffix, id_list_path=None):
    """CachedSplit of one view. ``stats_suffix`` in {"fit", "all"} selects the stage's statistics."""
    from feeder.piw3d_step5 import load_input_stats
    from feeder.piw3d_step5b import Step5bDataset, load_recording_means

    if view == "group_dev" and stats_suffix != "all":
        raise PermissionError("group_dev may be opened only by stage 2")
    data_root = resolve_path(repo_root, "data_root")
    freeze_dir = resolve_path(repo_root, "freeze_dir")
    stats = None
    means = None
    if input_norm in ("train_zscore", "recording_centered"):
        stats = load_input_stats(freeze_dir / f"input_stats_{stats_suffix}.npz")
    if input_norm == "recording_centered":
        means = load_recording_means(freeze_dir / "recording_input_means.npz")
    dataset = Step5bDataset(
        data_root,
        Path(id_list_path) if id_list_path else data_root / "train_data" / f"{view}_ids.txt",
        expected_split=VIEW_ROLE[view],
        split_assignments_path=freeze_dir / "split_assignments.json",
        input_norm=input_norm,
        input_stats=stats,
        recording_means=means,
    )
    return CachedSplit(dataset)


def load_codec(repo_root, stats_suffix):
    from model.research_models import load_codec_stats

    return load_codec_stats(resolve_path(repo_root, "freeze_dir") / f"codec_stats_{stats_suffix}.json")


def build_model(repo_root, stats_suffix, device, seed):
    from model.research_models import build_research_model

    r_base._seed_everything(seed)
    return build_research_model("simple_cnn", load_codec(repo_root, stats_suffix), step5_config_for_json(), device)


# ----------------------------------------------------------------------------- metrics

def recording_mean_poses(sample_ids, poses):
    groups = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        groups[recording_of(sample_id)].append(index)
    poses = np.asarray(poses, dtype=np.float64)
    return {name: poses[idx].mean(axis=0) for name, idx in groups.items()}


def oracle_statistics(train_ids, train_poses, eval_ids, eval_poses):
    means = recording_mean_poses(train_ids, train_poses)
    missing = sorted({recording_of(s) for s in eval_ids} - set(means))
    if missing:
        raise KeyError(f"Recording-mean oracle undefined for recordings {missing[:5]}")
    oracle = np.stack([means[recording_of(s)] for s in eval_ids])
    stats = pose_statistics(oracle, eval_poses)
    return {"oracle_abs_mpjpe": stats["abs_mpjpe"], "oracle_rel_mpjpe": stats["rel_mpjpe"]}


def mean_pose_baseline(train_poses, eval_poses):
    mean_pose = np.asarray(train_poses, dtype=np.float64).mean(axis=0)
    constant = np.broadcast_to(mean_pose[None], np.asarray(eval_poses).shape)
    stats = pose_statistics(constant, eval_poses)
    return {"mean_pose_abs_mpjpe": stats["abs_mpjpe"], "mean_skeleton_rel_mpjpe": stats["rel_mpjpe"]}


def motion_skill(sample_ids, predictions, ground_truths):
    """Frame-weighted mean over recordings of 1 - mean||dpred-dgt|| / mean||dgt|| (relative poses)."""
    pred = np.asarray(predictions, dtype=np.float64)[:, 0]
    gt = np.asarray(ground_truths, dtype=np.float64)[:, 0]
    pred = pred - pred.mean(axis=1, keepdims=True)
    gt = gt - gt.mean(axis=1, keepdims=True)
    groups = defaultdict(list)
    for index, sample_id in enumerate(sample_ids):
        groups[recording_of(sample_id)].append(index)
    skills = {}
    weights = {}
    for name, idx in sorted(groups.items()):
        dg = gt[idx] - gt[idx].mean(axis=0)
        dp = pred[idx] - pred[idx].mean(axis=0)
        denominator = float(np.linalg.norm(dg, axis=-1).mean())
        if denominator <= 0:
            continue
        skills[name] = 1.0 - float(np.linalg.norm(dp - dg, axis=-1).mean()) / denominator
        weights[name] = len(idx)
    total = sum(weights.values())
    value = sum(skills[n] * weights[n] for n in skills) / total if total else float("nan")
    return {"motion_skill": value, "per_recording": skills}


def evaluate_views(model, views, train_view_name, device, batch_size, shuffle_seed, with_research=False):
    train = views[train_view_name]
    train_gt = train.pose.numpy().astype(np.float64)
    evaluation = {}
    predictions = {}
    for name, split in views.items():
        pred = predict(model, split, device, batch_size)
        gt = split.pose.numpy().astype(np.float64)
        record = pose_statistics(pred, gt, shuffle_seed=shuffle_seed)
        record.update(mean_pose_baseline(train_gt, gt))
        record["rel_ratio_to_mean_skeleton"] = record["rel_mpjpe"] / record["mean_skeleton_rel_mpjpe"]
        record["abs_ratio_to_mean_pose"] = record["abs_mpjpe"] / record["mean_pose_abs_mpjpe"]
        seen = {recording_of(s) for s in train.sample_ids}
        if {recording_of(s) for s in split.sample_ids} <= seen:
            record.update(oracle_statistics(train.sample_ids, train_gt, split.sample_ids, gt))
            record["rel_ratio_to_oracle"] = record["rel_mpjpe"] / record["oracle_rel_mpjpe"]
            record["abs_ratio_to_oracle"] = record["abs_mpjpe"] / record["oracle_abs_mpjpe"]
        skill = motion_skill(split.sample_ids, pred, gt)
        record["motion_skill"] = skill["motion_skill"]
        if with_research:
            record["motion_skill_per_recording"] = skill["per_recording"]
            record["research_metrics"] = research_metrics(pred, gt)
        evaluation[name] = record
        predictions[name] = pred
    return evaluation, predictions


def gate_values(evaluation, holdout_view, unseen_view, gates):
    holdout = evaluation[holdout_view]
    unseen = evaluation[unseen_view]
    t1_ratio = holdout["rel_mpjpe"] / holdout["oracle_rel_mpjpe"]
    x1_ratio = unseen["rel_mpjpe"] / unseen["mean_skeleton_rel_mpjpe"]
    return {
        "T1": {"pass": bool(t1_ratio <= gates["T1_ratio"]), "view": holdout_view, "rel_mpjpe": holdout["rel_mpjpe"],
               "oracle_rel_mpjpe": holdout["oracle_rel_mpjpe"], "ratio": t1_ratio,
               "abs_ratio_to_oracle_reported": holdout["abs_mpjpe"] / holdout["oracle_abs_mpjpe"]},
        "X1": {"pass": bool(x1_ratio <= gates["X1_ratio"]), "view": unseen_view, "rel_mpjpe": unseen["rel_mpjpe"],
               "mean_skeleton_rel_mpjpe": unseen["mean_skeleton_rel_mpjpe"], "ratio": x1_ratio,
               "abs_ratio_to_mean_pose_reported": unseen["abs_mpjpe"] / unseen["mean_pose_abs_mpjpe"]},
    }


# ----------------------------------------------------------------------------- training

def _crashed(run_dir, record, reason, curves):
    record = dict(record, crashed=True, reason=reason, completed_at_utc=_now())
    decision = {
        "run_id": record["run_id"], "seed": record["seed"], "stage": record["stage"], "crashed": True,
        "reason": reason, "gates": {"T1": {"pass": False, "crashed": True}, "X1": {"pass": False, "crashed": True}},
        "train_view_pred_std_across_samples": None, "unseen_rel_ratio_to_mean_skeleton": None,
        "checkpoint": "none (run crashed)",
    }
    _write_json(run_dir / "learning_curves.json", curves)
    _write_json(run_dir / "result.json", record)
    _write_json(run_dir / "decision.json", decision)
    return record, decision


def train_and_evaluate(repo_root, stage, run_id, seed, device_name="auto", epochs_override=None, out_root=None):
    config = step5b_config_for_json()
    protocol = verify_protocol_freeze(repo_root)
    training = config["training"]
    stage_config = config[stage]
    run = config["stage1"]["runs"][run_id]
    suffix = stage_config["stats_suffix"]
    if stage == "stage1":
        train_view = run["train_view"]
        per_epoch = [train_view, "fit_holdout", "group_val"]
        final = per_epoch
    else:
        train_view = stage_config["train_view_if_D1"] if run_id == "D1" else stage_config["train_view"]
        per_epoch = [train_view, "all_holdout"]
        final = per_epoch + ["group_dev"]
    root = Path(out_root) if out_root else resolve_path(repo_root, "artifact_root")
    run_dir = root / stage / f"{run_id}_seed{seed}"
    if run_dir.exists():
        raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    device, deterministic = setup_runtime(device_name)
    views = {name: load_view(repo_root, name, run["input_norm"], suffix) for name in per_epoch}
    for name in views:
        if name != train_view and set(views[name].sample_ids) & set(views[train_view].sample_ids):
            raise AssertionError(f"{name} shares IDs with the training view")
    run_dir.mkdir(parents=True)
    environment = r_base.capture_environment(device, deterministic)
    model = build_model(repo_root, suffix, device, seed)
    optimizer = make_optimizer(model, training["peak_learning_rate"], training["weight_decay"])
    generator = torch.Generator().manual_seed(seed)
    loader = make_single_person_dataloader(views[train_view], is_training=True, generator=generator,
                                           batch_size=training["batch_size"], num_workers=training["num_workers"])
    epochs = epochs_override or training["epochs"]
    total_steps = epochs * len(loader)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_factor(step, training["warmup_steps"], total_steps)
    )
    shuffle_seed = config["evaluation"]["shuffled_pairing_seed"]
    base_record = {"run_id": run_id, "seed": seed, "stage": stage, "run_config": run, "train_view": train_view,
                   "stats_suffix": suffix, "protocol": protocol, "epochs": epochs}
    curves = []
    step = 0
    started = time.time()
    for epoch in range(1, epochs + 1):
        epoch_started = time.time()
        model.train()
        losses = []
        grad_norms = []
        for batch in loader:
            if sorted(batch) != ["csi", "pose"]:
                raise AssertionError("Training batch exposes keys other than csi/pose")
            try:
                loss, grad_norm = training_step(model, optimizer, batch["csi"].to(device), batch["pose"].to(device),
                                                training["grad_clip_norm"])
            except ValueError as exc:
                return _crashed(run_dir, dict(base_record, steps_completed=step, epoch=epoch), str(exc), curves)
            scheduler.step()
            step += 1
            losses.append(loss)
            grad_norms.append(grad_norm)
        train_seconds = time.time() - epoch_started
        evaluation, _ = evaluate_views(model, views, train_view, device, training["eval_batch_size"], shuffle_seed)
        record = {"epoch": epoch, "steps": step, "learning_rate_end": optimizer.param_groups[0]["lr"],
                  "train_loss_mean": float(np.mean(losses)), "grad_norm_mean_preclip": float(np.mean(grad_norms)),
                  "train_seconds": train_seconds, "epoch_seconds": time.time() - epoch_started, **evaluation}
        curves.append(record)
        print(f"[{stage} {run_id} seed{seed}] epoch {epoch}/{epochs} loss {record['train_loss_mean']:.4f} "
              + " ".join(f"{n}:abs={evaluation[n]['abs_mpjpe']:.4f}/rel={evaluation[n]['rel_mpjpe']:.4f}"
                         f"/skill={evaluation[n]['motion_skill']:.3f}" for n in evaluation)
              + f" ({record['epoch_seconds']:.1f}s)", flush=True)
        _write_json(run_dir / "learning_curves.json", curves)
        if epoch == 1:
            projected = record["epoch_seconds"] * epochs / 3600.0
            if projected > training["first_epoch_budget_hours"]:
                raise RuntimeError(f"First epoch projects {projected:.2f} h > budget; stop and ask the user")

    runtime = time.time() - started
    if stage == "stage2":
        views["group_dev"] = load_view(repo_root, "group_dev", run["input_norm"], suffix)
        recordings_dev = {recording_of(s) for s in views["group_dev"].sample_ids}
        if recordings_dev & {recording_of(s) for s in views[train_view].sample_ids}:
            raise AssertionError("group_dev shares recordings with the training view")
    evaluation, predictions = evaluate_views(model, views, train_view, device, training["eval_batch_size"],
                                             shuffle_seed, with_research=True)
    state_path = run_dir / "final_state_dict.pt"
    torch.save(model.state_dict(), state_path)
    reloaded = build_model(repo_root, suffix, device, seed + 1000)
    reloaded.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    reload_predictions = {name: predict(reloaded, views[name], device, training["eval_batch_size"]) for name in views}
    reload_exact = all(np.array_equal(predictions[n], reload_predictions[n]) for n in predictions)
    reload_exact = bool(reload_exact and r_base.state_dicts_equal(model, reloaded))
    for name, pred in predictions.items():
        np.savez(run_dir / f"predictions_{name}.npz", predictions=pred,
                 ground_truths=views[name].pose.numpy().astype(np.float64), sample_ids=np.asarray(views[name].sample_ids))
    holdout_view = stage_config["holdout_view"]
    unseen_view = stage_config["unseen_view"]
    gates = gate_values(evaluation, holdout_view, unseen_view, config["gates"])
    result = dict(base_record, total_steps=step, runtime_seconds=runtime, final_epoch_evaluation=evaluation,
                  reload_reproduces_final_predictions_exactly=reload_exact, environment=environment,
                  label_access=config["label_access"], distance_units="stored units", completed_at_utc=_now(),
                  input_norm_disclosure=(config["input_norm"]["transductive_note"]
                                         if run["input_norm"] == "recording_centered" else None))
    _write_json(run_dir / "result.json", result)
    decision = {
        "run_id": run_id, "seed": seed, "stage": stage, "crashed": False, "train_view": train_view, "gates": gates,
        "train_view_pred_std_across_samples": evaluation[train_view]["pred_std_across_samples"],
        "unseen_rel_ratio_to_mean_skeleton": gates["X1"]["ratio"], "reload_exact": reload_exact,
        "checkpoint": "final epoch only",
    }
    _write_json(run_dir / "decision.json", decision)
    return result, decision


# ----------------------------------------------------------------------------- smoke / overfit

def smoke_run(repo_root, run_id, device, smoke_dir, batch_size=8):
    config = step5b_config_for_json()
    run = config["stage1"]["runs"][run_id]
    data_root = resolve_path(repo_root, "data_root")
    checks = {}
    subsets = {}
    for view in (run["train_view"], "group_val"):
        ids = [line.strip() for line in (data_root / "train_data" / f"{view}_ids.txt").read_text(encoding="utf-8").splitlines() if line.strip()]
        path = Path(smoke_dir) / f"{run_id}_{view}_ids.txt"
        path.write_text("".join(f"{s}\n" for s in ids[:: max(1, len(ids) // 64)][:64]), encoding="utf-8")
        subsets[view] = load_view(repo_root, view, run["input_norm"], "fit", id_list_path=path)
    loader = make_single_person_dataloader(subsets[run["train_view"]], True, torch.Generator().manual_seed(0), batch_size)
    batch = next(iter(loader))
    checks["batch_keys"] = sorted(batch) == ["csi", "pose"]
    checks["csi_shape"] = tuple(batch["csi"].shape) == (batch_size, 3, 180, 20)
    checks["pose_shape"] = tuple(batch["pose"].shape) == (batch_size, 1, 14, 3)
    checks["csi_finite"] = bool(torch.isfinite(batch["csi"]).all())
    model = build_model(repo_root, "fit", device, 0)
    csi = batch["csi"].to(device)
    pose = batch["pose"].to(device)
    model.eval()
    with torch.no_grad():
        prediction = model(csi)
        tokens, pooled = model.forward_features(csi)
    checks["forward_shape"] = tuple(prediction.shape) == (batch_size, 1, 14, 3)
    checks["forward_features_shapes"] = tokens.ndim == 3 and pooled.shape == (batch_size, tokens.shape[2])
    codec = load_codec(repo_root, "fit")
    mean_pose = torch.tensor(np.asarray(codec["mu_c"])[None, :] + np.asarray(codec["mu_r"]), dtype=torch.float32, device=device)
    checks["untrained_equals_train_mean_pose"] = bool(
        torch.allclose(prediction[:, 0], mean_pose.expand_as(prediction[:, 0]), atol=1e-5, rtol=0)
    )
    model.train()
    optimizer = make_optimizer(model, 1e-3, 0.0)
    before = {name: value.detach().clone() for name, value in model.named_parameters()}
    loss, grad_norm = training_step(model, optimizer, csi, pose, 1.0)
    checks["loss_finite"] = math.isfinite(loss)
    checks["backward_grad_norm_positive"] = grad_norm > 0
    changed = sum(not torch.equal(before[n], v.detach()) for n, v in model.named_parameters())
    checks["optimizer_updates_parameters"] = changed > 0
    state_path = Path(smoke_dir) / f"{run_id}_state_dict.pt"
    torch.save(model.state_dict(), state_path)
    reloaded = build_model(repo_root, "fit", device, 123)
    reloaded.load_state_dict(torch.load(state_path, map_location=device, weights_only=True))
    checks["state_dict_reload_exact"] = r_base.state_dicts_equal(model, reloaded)
    checks["reload_predictions_exact"] = bool(np.array_equal(predict(model, subsets["group_val"], device, 64),
                                                             predict(reloaded, subsets["group_val"], device, 64)))
    evaluation = pose_statistics(predict(model, subsets["group_val"], device, 64), subsets["group_val"].pose.numpy())
    checks["evaluation_runs"] = math.isfinite(evaluation["abs_mpjpe"])
    return {"run_id": run_id, "checks": checks, "all_pass": all(checks.values()), "first_step_loss": loss,
            "first_step_grad_norm": grad_norm, "changed_parameter_tensors": int(changed),
            "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad))}


def run_smoke(repo_root, device_name="auto"):
    protocol = verify_protocol_freeze(repo_root)
    smoke_dir = resolve_path(repo_root, "artifact_root") / "smoke"
    if smoke_dir.exists():
        raise FileExistsError(f"Refusing to overwrite smoke directory: {smoke_dir}")
    smoke_dir.mkdir(parents=True)
    device, _ = setup_runtime(device_name)
    results = {run_id: smoke_run(repo_root, run_id, device, smoke_dir) for run_id in STAGE1_RUN_IDS}
    record = {"protocol": protocol, "all_pass": all(r["all_pass"] for r in results.values()), "runs": results,
              "device": str(device), "completed_at_utc": _now()}
    _write_json(smoke_dir / "smoke.json", record)
    return record


def overfit_indices(total, count):
    k = total // count
    if k < 1:
        raise ValueError("Not enough samples for the overfit subset")
    return [index * k for index in range(count)]


def run_overfit(repo_root, run_id, device_name="auto"):
    config = step5b_config_for_json()
    overfit = config["overfit"]
    if run_id not in overfit["runs"]:
        raise ValueError(f"Overfit check is predeclared only for {overfit['runs']}")
    protocol = verify_protocol_freeze(repo_root)
    out_path = resolve_path(repo_root, "artifact_root") / "overfit" / f"{run_id}.json"
    if out_path.exists():
        raise FileExistsError(f"Refusing to overwrite overfit result: {out_path}")
    device, _ = setup_runtime(device_name)
    view = load_view(repo_root, overfit["view"], config["stage1"]["runs"][run_id]["input_norm"], "fit")
    indices = overfit_indices(len(view), overfit["sample_count"])
    ids = [view.sample_ids[i] for i in indices]
    csi = view.csi[indices].to(device)
    pose = view.pose[indices].to(device)
    gt = pose.cpu().numpy().astype(np.float64)
    reference = pose_statistics(np.broadcast_to(gt.mean(axis=0, keepdims=True), gt.shape), gt)["abs_mpjpe"]
    model = build_model(repo_root, "fit", device, overfit["seed"])
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
    ratio = final["abs_mpjpe"] / reference
    record = {"run_id": run_id, "protocol": protocol, "sample_ids": ids,
              "recordings_present": len({recording_of(s) for s in ids}), "mean_pose_mpjpe_same_64": reference,
              "final": final, "ratio": ratio, "pass_ratio": overfit["pass_ratio"],
              "pass": bool(ratio <= overfit["pass_ratio"]), "curve": curve, "completed_at_utc": _now()}
    _write_json(out_path, record)
    return record


def gate_s(repo_root):
    root = resolve_path(repo_root, "artifact_root")
    smoke = _load_json(root / "smoke" / "smoke.json")
    overfits = {run_id: _load_json(root / "overfit" / f"{run_id}.json") for run_id in step5b_config_for_json()["overfit"]["runs"]}
    return {"pass": bool(smoke["all_pass"] and all(o["pass"] for o in overfits.values())),
            "smoke_all_pass": smoke["all_pass"], "overfit_ratios": {k: o["ratio"] for k, o in overfits.items()}}


# ----------------------------------------------------------------------------- selection / verdict

def _decision(repo_root, stage, run_id, seed):
    path = resolve_path(repo_root, "artifact_root") / stage / f"{run_id}_seed{seed}" / "decision.json"
    return _load_json(path) if path.is_file() else None


def select_run(decisions, min_std, tie_relative):
    eligible = {
        run_id: d for run_id, d in decisions.items()
        if d is not None and not d["crashed"] and d["train_view_pred_std_across_samples"] >= min_std
    }
    if not eligible:
        return None, eligible
    best = min(d["unseen_rel_ratio_to_mean_skeleton"] for d in eligible.values())
    tied = [r for r, d in eligible.items() if d["unseen_rel_ratio_to_mean_skeleton"] <= best * (1.0 + tie_relative)]
    return min(tied, key=SIMPLICITY_ORDER.index), eligible


def run_select(repo_root):
    config = step5b_config_for_json()
    decisions = {run_id: _decision(repo_root, "stage1", run_id, 0) for run_id in STAGE1_RUN_IDS}
    missing = [r for r, d in decisions.items() if d is None]
    if missing:
        raise FileNotFoundError("Stage-1 decisions missing for: " + ", ".join(missing))
    selected, eligible = select_run(decisions, config["gates"]["selection_min_pred_std"],
                                    config["gates"]["selection_tie_relative"])
    record = {"rule": config["gates"]["selection"],
              "label_use": "single-person training-partition labels (group_val) only; group_dev untouched",
              "eligible_runs": sorted(eligible),
              "group_val_rel_ratio_to_mean_skeleton": {r: d["unseen_rel_ratio_to_mean_skeleton"] for r, d in decisions.items()},
              "fit_holdout_T1": {r: d["gates"]["T1"] for r, d in decisions.items()},
              "selected_run": selected, "completed_at_utc": _now()}
    _write_json(resolve_path(repo_root, "artifact_root") / "selection.json", record)
    return record


def compute_verdict(stage1_decisions, selected, stage2_decisions):
    if selected is None or any(d is None for d in stage2_decisions.values()):
        return "INCONCLUSIVE"
    t1 = [bool(d["gates"]["T1"]["pass"]) for d in stage2_decisions.values()]
    x1 = [bool(d["gates"]["X1"]["pass"]) for d in stage2_decisions.values()]
    any_stage1_t1 = any(d is not None and d["gates"]["T1"]["pass"] for d in stage1_decisions.values())
    if all(t1) and all(x1):
        return "GREEN"
    if all(t1):
        return "YELLOW-TRACKS"
    if not any(t1) and not any_stage1_t1:
        return "RED-NO-TRACKING"
    return "INCONCLUSIVE"


def run_verdict(repo_root):
    config = step5b_config_for_json()
    root = resolve_path(repo_root, "artifact_root")
    selection = _load_json(root / "selection.json")
    selected = selection["selected_run"]
    stage1 = {r: _decision(repo_root, "stage1", r, 0) for r in STAGE1_RUN_IDS}
    stage2 = {s: _decision(repo_root, "stage2", selected, s) for s in config["stage2"]["seeds"]} if selected else {}
    verdict = compute_verdict(stage1, selected, stage2)
    record = {"verdict": verdict, "gate_S": gate_s(repo_root), "selected_run": selected,
              "stage2_gates": {str(s): (d["gates"] if d else None) for s, d in stage2.items()},
              "stage1_T1_any_pass": any(d["gates"]["T1"]["pass"] for d in stage1.values() if d),
              "completed_at_utc": _now()}
    _write_json(root / "verdict.json", record)
    return record


# ----------------------------------------------------------------------------- cli

def main(argv=None):
    parser = argparse.ArgumentParser(description="Step-5b runner")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--device", default="auto")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("freeze-protocol")
    sub.add_parser("smoke")
    overfit = sub.add_parser("overfit")
    overfit.add_argument("--run", required=True, choices=STAGE1_RUN_IDS)
    stage1 = sub.add_parser("stage1")
    stage1.add_argument("--run", required=True, choices=STAGE1_RUN_IDS)
    sub.add_parser("select")
    stage2 = sub.add_parser("stage2")
    stage2.add_argument("--seed", required=True, type=int)
    sub.add_parser("verdict")
    args = parser.parse_args(argv)
    if args.command == "freeze-protocol":
        output = freeze_protocol(args.repo_root)
    elif args.command == "smoke":
        output = run_smoke(args.repo_root, args.device)
    elif args.command == "overfit":
        output = run_overfit(args.repo_root, args.run, args.device)
        output = {k: output[k] for k in ("run_id", "pass", "ratio", "mean_pose_mpjpe_same_64", "recordings_present")}
    elif args.command == "stage1":
        s_gate = gate_s(args.repo_root)
        if not s_gate["pass"]:
            raise RuntimeError(f"Gate S failed; stop and report: {s_gate}")
        _, output = train_and_evaluate(args.repo_root, "stage1", args.run, 0, args.device)
    elif args.command == "stage2":
        selection = _load_json(resolve_path(args.repo_root, "artifact_root") / "selection.json")
        if selection["selected_run"] is None:
            raise RuntimeError("No stage-1 run was selected; stop and report")
        if args.seed not in step5b_config_for_json()["stage2"]["seeds"]:
            raise ValueError("Stage-2 seeds are predeclared as 0, 1, 2")
        _, output = train_and_evaluate(args.repo_root, "stage2", selection["selected_run"], args.seed, args.device)
    elif args.command == "select":
        output = run_select(args.repo_root)
    else:
        output = run_verdict(args.repo_root)
    print(json.dumps(output, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    sys.exit(main())
