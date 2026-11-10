import builtins
import io
import json
import shutil

import numpy as np
import pytest
import torch

import step5_single_person_baseline as runner
from feeder.piw3d_step5 import compute_train_input_stats, save_input_stats
from model.research_models import compute_codec_stats, save_codec_stats
from step5_config import STEP5_SOURCE_FILES, step5_config_for_json


CONFIG = step5_config_for_json()
SPLIT_IDS = {
    "train": [f"S11_01_{frame}" for frame in range(10, 50)] + [f"S01_02_{frame}" for frame in range(10, 34)],
    "temporal_holdout": [f"S11_01_{frame}" for frame in range(60, 68)],
    "group_dev": [f"S21_09_{frame}" for frame in range(10, 18)],
}
DECOY_MULTI_PERSON = "S02_03_10"


@pytest.fixture(autouse=True)
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


def make_repo(tmp_path):
    rng = np.random.default_rng(0)
    data_root = tmp_path / "data/_artifacts/step5_single_person/data_v1"
    freeze_dir = tmp_path / "data/_artifacts/step5_single_person/freeze_v1"
    train_root = data_root / "train_data"
    (train_root / "csi_ap").mkdir(parents=True)
    (train_root / "keypoint").mkdir()
    freeze_dir.mkdir(parents=True)
    all_ids = [i for ids in SPLIT_IDS.values() for i in ids] + [DECOY_MULTI_PERSON]
    for index, sample_id in enumerate(all_ids):
        np.save(train_root / "csi_ap" / f"{sample_id}.npy", rng.normal(size=(3, 180, 20)))
        pose = rng.normal(scale=0.2, size=(1, 14, 3)) + np.array([2.0, 1.5, 3.0]) + 0.01 * index
        np.save(train_root / "keypoint" / f"{sample_id}.npy", pose)
    (train_root / "train_data_list.txt").write_text("".join(f"{i}\n" for i in all_ids), encoding="utf-8")
    for split, ids in SPLIT_IDS.items():
        (train_root / f"{split}_ids.txt").write_text("".join(f"{i}\n" for i in ids), encoding="utf-8")
    assignments = {i: split for split, ids in SPLIT_IDS.items() for i in ids}
    (freeze_dir / "split_assignments.json").write_text(json.dumps(assignments), encoding="utf-8")
    train_paths = [train_root / "csi_ap" / f"{i}.npy" for i in SPLIT_IDS["train"]]
    save_input_stats(freeze_dir / "train_input_stats_v1.npz", compute_train_input_stats(train_paths), "x")
    poses = np.stack([np.load(train_root / "keypoint" / f"{i}.npy") for i in SPLIT_IDS["train"]])
    save_codec_stats(freeze_dir / "pose_codec_stats_v1.json", compute_codec_stats(poses), "x")
    decoy_test = tmp_path / "data/person_in_wifi_3d/wifi_dataset/test_data"
    decoy_test.mkdir(parents=True)
    (decoy_test / "test_data_list.txt").write_text("S11_01_301\n", encoding="utf-8")
    return tmp_path


def test_lr_schedule_warmup_and_cosine():
    assert runner.lr_factor(0, 200, 4080) == pytest.approx(1 / 200)
    assert runner.lr_factor(199, 200, 4080) == pytest.approx(1.0)
    assert runner.lr_factor(200, 200, 4080) == pytest.approx(1.0)
    assert runner.lr_factor(2140, 200, 4080) == pytest.approx(0.5)
    assert runner.lr_factor(4079, 200, 4080) < 1e-5


def test_pose_statistics_on_toy_case():
    gt = np.zeros((4, 1, 14, 3))
    gt[:, 0, :, 0] = np.arange(4)[:, None]
    gt[:, 0, :, 1] = np.arange(14)[None, :]
    constant = np.broadcast_to(gt.mean(axis=0, keepdims=True), gt.shape)
    stats = runner.pose_statistics(constant, gt, shuffle_seed=0)
    assert stats["pred_std_across_samples"] == 0.0
    assert stats["pred_centroid_std_mean_axes"] == 0.0
    assert stats["rel_mpjpe"] == pytest.approx(0.0)
    assert stats["abs_mpjpe"] == pytest.approx(1.0)
    assert stats["centroid_error"] == pytest.approx(1.0)
    perfect = runner.pose_statistics(gt, gt, shuffle_seed=0)
    assert perfect["abs_mpjpe"] == 0.0
    assert perfect["shuffled_pairing_abs_mpjpe"] > 0.0


def decision(g3=True, g4=True, g5=True, g6=True, dev=1.0):
    gates = {"G3": {"pass": g3}, "G4": {"pass": g4}, "G5": {"pass": g5}, "G6": {"pass": g6}}
    return {"gates": gates, "G3_to_G5_pass": g3 and g4 and g5, "G3_to_G6_pass": g3 and g4 and g5 and g6,
            "group_dev_abs_mpjpe": dev}


def test_selection_lowest_dev_with_simplicity_tiebreak():
    decisions = {"R1": decision(dev=0.800), "R2": decision(g5=False, dev=0.5), "R3": decision(dev=0.795),
                 "R4": decision(dev=0.9)}
    selected, eligible = runner.select_run(decisions, 0.01)
    assert sorted(eligible) == ["R1", "R3", "R4"]
    assert selected == "R1"  # R3 is within 1% -> tie -> simpler R1
    decisions["R3"] = decision(dev=0.70)
    assert runner.select_run(decisions, 0.01)[0] == "R3"
    assert runner.select_run({k: decision(g4=False) for k in "R1 R2 R3 R4".split()}, 0.01)[0] is None


def test_verdict_rules():
    g2 = {"applicable": True, "pass": True}
    seed0 = {k: decision() for k in ("R1", "R2", "R3", "R4")}
    all_good = {0: decision(), 1: decision(), 2: decision()}
    assert runner.compute_verdict(True, g2, seed0, "R1", all_good) == "GREEN"
    yellow = {0: decision(), 1: decision(g6=False), 2: decision()}
    assert runner.compute_verdict(True, g2, seed0, "R1", yellow) == "YELLOW-POSITION"
    red_seed0 = {k: decision(g4=False) for k in seed0}
    assert runner.compute_verdict(True, g2, red_seed0, None, {}) == "RED"
    assert runner.compute_verdict(True, {"applicable": True, "pass": False}, seed0, "R1", all_good) == "INCONCLUSIVE"
    assert runner.compute_verdict(True, g2, seed0, "R1", {0: decision(), 1: decision(g3=False), 2: decision()}) == "INCONCLUSIVE"
    assert runner.compute_verdict(True, {"applicable": False, "pass": None}, seed0, "R4", all_good) == "GREEN"


def test_overfit_indices_cover_all_groups():
    indices = runner.overfit_indices(3264, 64)
    assert indices[:3] == [0, 51, 102] and len(indices) == 64
    groups = {index // 64 for index in indices}
    assert groups == set(range(51))


def test_protocol_freeze_detects_source_change(tmp_path):
    code_root = tmp_path / "code"
    for name in list(STEP5_SOURCE_FILES) + [runner.PROTOCOL_DOC]:
        target = code_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runner.CODE_ROOT / name, target)
    repo = tmp_path / "repo"
    runner.freeze_protocol(repo, code_root=code_root)
    assert runner.verify_protocol_freeze(repo, code_root=code_root)["source_hashes_unchanged"]
    with pytest.raises(FileExistsError):
        runner.freeze_protocol(repo, code_root=code_root)
    (code_root / "step5_config.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        runner.verify_protocol_freeze(repo, code_root=code_root)


def test_end_to_end_tiny_run_label_access_and_reload(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runner.freeze_protocol(repo)
    opened = []
    real_open = builtins.open
    real_io_open = io.open

    def recording_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_open(file, *args, **kwargs)

    def recording_io_open(file, *args, **kwargs):
        opened.append(str(file))
        return real_io_open(file, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", recording_open)
    monkeypatch.setattr(io, "open", recording_io_open)
    result, run_decision = runner.run_training(repo, "R1", 0, device_name="cpu", epochs_override=2)
    monkeypatch.setattr(builtins, "open", real_open)
    monkeypatch.setattr(io, "open", real_io_open)

    joined = "\n".join(opened).replace("\\", "/")
    assert "test_data" not in joined
    assert DECOY_MULTI_PERSON not in joined
    assert result["reload_reproduces_final_metrics_exactly"] is True
    assert result["total_steps"] == 2 * 2  # 64 train samples, batch 32
    assert set(run_decision["gates"]) == {"G3", "G4", "G5", "G6"}
    run_dir = repo / "data/_artifacts/step5_single_person/runs/R1_seed0"
    for name in ("learning_curves.json", "result.json", "decision.json", "final_state_dict.pt",
                 "predictions_train.npz", "predictions_temporal_holdout.npz", "predictions_group_dev.npz"):
        assert (run_dir / name).is_file()
    with np.load(run_dir / "predictions_group_dev.npz") as stored:
        assert list(stored["sample_ids"]) == SPLIT_IDS["group_dev"]
    metrics = result["final_epoch_evaluation"]["group_dev"]["research_metrics"]["overall"]
    assert metrics["samples"] == len(SPLIT_IDS["group_dev"])
    assert result["baselines"]["train"]["mean_pose_abs_mpjpe"] > 0
    with pytest.raises(FileExistsError):
        runner.run_training(repo, "R1", 0, device_name="cpu", epochs_override=1)


def test_smoke_all_runs_on_cpu(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runner.freeze_protocol(repo)
    record = runner.run_smoke(repo, device_name="cpu")
    failing = {run: {k: v for k, v in r["checks"].items() if not v} for run, r in record["runs"].items()}
    assert record["G1_pass"], failing
    assert record["runs"]["R1"]["checks"]["untrained_equals_train_mean_pose"]
    assert "untrained_equals_train_mean_pose" not in record["runs"]["R4"]["checks"]


def test_crashed_run_fails_every_gate_and_is_never_selected(tmp_path):
    _, crashed = runner.write_crashed_run(tmp_path, "R2", 0, 3, 250, "Non-finite training loss: nan", [])
    assert crashed["crashed"] and not crashed["G3_to_G5_pass"]
    assert all(not gate["pass"] for gate in crashed["gates"].values())
    selected, eligible = runner.select_run({"R1": decision(dev=0.9), "R2": crashed}, 0.01)
    assert selected == "R1" and "R2" not in eligible
    assert (tmp_path / "decision.json").is_file()
