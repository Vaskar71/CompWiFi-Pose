import builtins
import io
import json
import shutil

import numpy as np
import pytest
import torch

import step5b_pose_signal as runner
from feeder.piw3d_step5 import compute_train_input_stats, load_input_stats, save_input_stats
from feeder.piw3d_step5b import Step5bDataset, load_recording_means
from model.research_models import compute_codec_stats, save_codec_stats
from step5b_config import STEP5B_SOURCE_FILES


def _ids(group, frames):
    return [f"{group}_{f}" for f in frames]


VIEWS = {
    "fit_train_dense": _ids("S11_01", range(10, 42)) + _ids("S21_02", range(10, 42)),
    "fit_holdout": _ids("S11_01", range(60, 68)) + _ids("S21_02", range(60, 68)),
    "group_val": _ids("S51_03", range(10, 26)),
    "all_holdout": _ids("S11_01", range(60, 68)) + _ids("S21_02", range(60, 68)) + _ids("S51_03", range(60, 64)),
    "group_dev": _ids("S11_90", range(10, 18)),
}
VIEWS["fit_train_sparse"] = VIEWS["fit_train_dense"][::2]
VIEWS["all_train_dense"] = VIEWS["fit_train_dense"] + VIEWS["group_val"]
VIEWS["all_train_sparse"] = VIEWS["all_train_dense"][::2]
DECOY = "S12_05_10"


@pytest.fixture(autouse=True)
def restore_determinism():
    yield
    torch.use_deterministic_algorithms(False)


def make_repo(tmp_path):
    rng = np.random.default_rng(0)
    base = tmp_path / "data/_artifacts/step5b_pose_signal"
    train = base / "data_v2" / "train_data"
    freeze = base / "freeze_v2"
    (train / "csi_ap").mkdir(parents=True)
    (train / "keypoint").mkdir()
    freeze.mkdir(parents=True)
    roles = {"train": VIEWS["all_train_dense"], "temporal_holdout": VIEWS["all_holdout"], "group_dev": VIEWS["group_dev"]}
    everything = [s for ids in roles.values() for s in ids] + [DECOY]
    offsets = {}
    for index, sample_id in enumerate(everything):
        rec = sample_id.rsplit("_", 1)[0]
        offsets.setdefault(rec, rng.normal(scale=3.0, size=(3, 180, 1)))
        np.save(train / "csi_ap" / f"{sample_id}.npy", offsets[rec] + rng.normal(size=(3, 180, 20)))
        pose = rng.normal(scale=0.1, size=(1, 14, 3)) + np.array([2.0, 1.5, 3.0]) + 0.01 * index
        np.save(train / "keypoint" / f"{sample_id}.npy", pose)
    (train / "train_data_list.txt").write_text("".join(f"{s}\n" for s in everything), encoding="utf-8")
    for view, ids in VIEWS.items():
        (train / f"{view}_ids.txt").write_text("".join(f"{s}\n" for s in ids), encoding="utf-8")
    assignments = {s: role for role, ids in roles.items() for s in ids}
    (freeze / "split_assignments.json").write_text(json.dumps(assignments), encoding="utf-8")
    for suffix, view in (("fit", "fit_train_dense"), ("all", "all_train_dense")):
        save_input_stats(freeze / f"input_stats_{suffix}.npz",
                         compute_train_input_stats(train / "csi_ap" / f"{s}.npy" for s in VIEWS[view]), "x")
        poses = np.stack([np.load(train / "keypoint" / f"{s}.npy") for s in VIEWS[view]])
        save_codec_stats(freeze / f"codec_stats_{suffix}.json", compute_codec_stats(poses), "x")
    names = sorted({s.rsplit("_", 1)[0] for s in roles["train"] + roles["group_dev"]})
    means = []
    for name in names:
        frames = [s for s in roles["train"] + roles["group_dev"] if s.rsplit("_", 1)[0] == name]
        means.append(np.mean([np.load(train / "csi_ap" / f"{s}.npy").mean(axis=2, keepdims=True) for s in frames], axis=0))
    np.savez(freeze / "recording_input_means.npz", recordings=np.asarray(names), means=np.stack(means),
             frame_counts=np.ones(len(names)), sources=np.asarray(["x"] * len(names)))
    decoy_test = tmp_path / "data/person_in_wifi_3d/wifi_dataset/test_data"
    decoy_test.mkdir(parents=True)
    (decoy_test / "test_data_list.txt").write_text("S11_01_301\n", encoding="utf-8")
    return tmp_path


# ----------------------------------------------------------------------------- feeder

def test_recording_centered_subtracts_recording_mean(tmp_path):
    repo = make_repo(tmp_path)
    freeze = repo / "data/_artifacts/step5b_pose_signal/freeze_v2"
    data_root = repo / "data/_artifacts/step5b_pose_signal/data_v2"
    stats = load_input_stats(freeze / "input_stats_fit.npz")
    means = load_recording_means(freeze / "recording_input_means.npz")
    dataset = Step5bDataset(data_root, data_root / "train_data" / "group_val_ids.txt", expected_split="train",
                            split_assignments_path=freeze / "split_assignments.json",
                            input_norm="recording_centered", input_stats=stats, recording_means=means)
    sample_id = dataset.sample_ids[0]
    raw = np.load(data_root / "train_data" / "csi_ap" / f"{sample_id}.npy")
    expected = (raw - means["S51_03"]) / np.maximum(stats["std"], 1e-6)
    np.testing.assert_allclose(dataset[0]["csi"].numpy(), expected.astype(np.float32), rtol=1e-6, atol=1e-6)
    assert sorted(dataset[0]) == ["csi", "pose"]
    stacked = np.stack([dataset[i]["csi"].numpy() for i in range(len(dataset))])
    assert np.abs(stacked.mean(axis=(0, 3))).max() < 1e-4  # all group_val frames define the mean


def test_feeder_argument_validation(tmp_path):
    repo = make_repo(tmp_path)
    data_root = repo / "data/_artifacts/step5b_pose_signal/data_v2"
    path = data_root / "train_data" / "group_val_ids.txt"
    with pytest.raises(ValueError):
        Step5bDataset(data_root, path, input_norm="recording_centered")
    with pytest.raises(ValueError):
        Step5bDataset(data_root, path, input_norm="legacy_minmax", recording_means={})
    with pytest.raises(ValueError):
        Step5bDataset(data_root, path, input_norm="other")


def test_group_dev_refused_in_stage1(tmp_path):
    repo = make_repo(tmp_path)
    with pytest.raises(PermissionError):
        runner.load_view(repo, "group_dev", "legacy_minmax", "fit")
    assert len(runner.load_view(repo, "group_dev", "legacy_minmax", "all")) == 8


# ----------------------------------------------------------------------------- metrics

def test_oracle_mean_pose_and_motion_skill():
    train_ids = _ids("S11_01", [1, 2]) + _ids("S11_02", [1, 2])
    train = np.zeros((4, 1, 14, 3))
    train[2:] += 1.0
    eval_ids = _ids("S11_01", [5]) + _ids("S11_02", [5])
    gt = np.concatenate([np.zeros((1, 1, 14, 3)), np.ones((1, 1, 14, 3))])
    oracle = runner.oracle_statistics(train_ids, train, eval_ids, gt)
    assert oracle["oracle_abs_mpjpe"] == 0.0
    baseline = runner.mean_pose_baseline(train, gt)
    assert baseline["mean_pose_abs_mpjpe"] == pytest.approx(np.sqrt(3 * 0.25))
    with pytest.raises(KeyError):
        runner.oracle_statistics(train_ids, train, ["S99_01_1"], gt[:1])

    rng = np.random.default_rng(0)
    ids = _ids("S11_01", range(10))
    truth = rng.normal(size=(10, 1, 14, 3))
    assert runner.motion_skill(ids, truth + 5.0, truth)["motion_skill"] == pytest.approx(1.0)
    still = np.broadcast_to(truth.mean(axis=0, keepdims=True), truth.shape)
    assert runner.motion_skill(ids, still, truth)["motion_skill"] == pytest.approx(0.0)
    assert runner.motion_skill(ids, -truth, truth)["motion_skill"] < 0


def _decision(t1=True, x1=True, ratio=0.9, std=0.3, crashed=False):
    return {"crashed": crashed, "gates": {"T1": {"pass": t1}, "X1": {"pass": x1}},
            "train_view_pred_std_across_samples": None if crashed else std,
            "unseen_rel_ratio_to_mean_skeleton": None if crashed else ratio}


def test_selection_rule():
    decisions = {"D1": _decision(ratio=1.10), "D2": _decision(ratio=1.005), "D3": _decision(ratio=1.0),
                 "D4": _decision(ratio=0.5, std=0.001)}
    selected, eligible = runner.select_run(decisions, 0.01, 0.01)
    assert sorted(eligible) == ["D1", "D2", "D3"]
    assert selected == "D2"  # D3 best, D2 within 1% -> simpler
    decisions["D2"] = _decision(crashed=True)
    assert runner.select_run(decisions, 0.01, 0.01)[0] == "D3"
    assert runner.select_run({k: _decision(crashed=True) for k in "D1 D2".split()}, 0.01, 0.01)[0] is None


def test_verdict_rules():
    stage1_t1 = {"D1": _decision(t1=False), "D2": _decision(t1=True)}
    stage1_no = {"D1": _decision(t1=False), "D2": _decision(t1=False)}
    good = {s: _decision() for s in range(3)}
    assert runner.compute_verdict(stage1_t1, "D2", good) == "GREEN"
    assert runner.compute_verdict(stage1_t1, "D2", {0: _decision(), 1: _decision(x1=False), 2: _decision()}) == "YELLOW-TRACKS"
    fail = {s: _decision(t1=False, x1=False) for s in range(3)}
    assert runner.compute_verdict(stage1_no, "D1", fail) == "RED-NO-TRACKING"
    assert runner.compute_verdict(stage1_t1, "D1", fail) == "INCONCLUSIVE"
    assert runner.compute_verdict(stage1_t1, "D2", {0: _decision(), 1: _decision(t1=False), 2: _decision()}) == "INCONCLUSIVE"
    assert runner.compute_verdict(stage1_t1, None, {}) == "INCONCLUSIVE"
    assert runner.compute_verdict(stage1_t1, "D2", {0: _decision(), 1: None, 2: _decision()}) == "INCONCLUSIVE"


def test_protocol_freeze_detects_source_change(tmp_path):
    code_root = tmp_path / "code"
    for name in list(STEP5B_SOURCE_FILES) + [runner.PROTOCOL_DOC, runner.PLAN_DOC]:
        target = code_root / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(runner.CODE_ROOT / name, target)
    repo = tmp_path / "repo"
    runner.freeze_protocol(repo, code_root=code_root)
    assert runner.verify_protocol_freeze(repo, code_root=code_root)["source_hashes_unchanged"]
    with pytest.raises(FileExistsError):
        runner.freeze_protocol(repo, code_root=code_root)
    (code_root / "step5_single_person_baseline.py").write_text("# changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError):
        runner.verify_protocol_freeze(repo, code_root=code_root)


# ----------------------------------------------------------------------------- end to end

def _record_opens(monkeypatch):
    opened = []
    real_open, real_io_open = builtins.open, io.open

    def rec_open(file, *a, **k):
        opened.append(str(file))
        return real_open(file, *a, **k)

    def rec_io_open(file, *a, **k):
        opened.append(str(file))
        return real_io_open(file, *a, **k)

    monkeypatch.setattr(builtins, "open", rec_open)
    monkeypatch.setattr(io, "open", rec_io_open)
    return opened, lambda: (monkeypatch.setattr(builtins, "open", real_open), monkeypatch.setattr(io, "open", real_io_open))


@pytest.mark.parametrize("run_id", ["D1", "D4"])
def test_stage1_tiny_run_label_access_and_reload(tmp_path, monkeypatch, run_id):
    repo = make_repo(tmp_path)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runner.freeze_protocol(repo)
    opened, restore = _record_opens(monkeypatch)
    result, decision = runner.train_and_evaluate(repo, "stage1", run_id, 0, device_name="cpu", epochs_override=2)
    restore()
    joined = "\n".join(opened).replace("\\", "/")
    assert "test_data" not in joined
    assert "group_dev" not in joined and "S11_90" not in joined
    assert DECOY not in joined
    assert result["reload_reproduces_final_predictions_exactly"] is True
    assert set(decision["gates"]) == {"T1", "X1"}
    evaluation = result["final_epoch_evaluation"]
    assert set(evaluation) == {result["train_view"], "fit_holdout", "group_val"}
    assert "oracle_rel_mpjpe" in evaluation["fit_holdout"] and "oracle_rel_mpjpe" not in evaluation["group_val"]
    run_dir = repo / f"data/_artifacts/step5b_pose_signal/stage1/{run_id}_seed0"
    for name in ("result.json", "decision.json", "learning_curves.json", "final_state_dict.pt", "predictions_group_val.npz"):
        assert (run_dir / name).is_file()
    with pytest.raises(FileExistsError):
        runner.train_and_evaluate(repo, "stage1", run_id, 0, device_name="cpu", epochs_override=1)


def test_stage2_opens_group_dev_once_at_the_end(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runner.freeze_protocol(repo)
    result, decision = runner.train_and_evaluate(repo, "stage2", "D3", 1, device_name="cpu", epochs_override=1)
    assert set(result["final_epoch_evaluation"]) == {"all_train_dense", "all_holdout", "group_dev"}
    curves = json.loads((repo / "data/_artifacts/step5b_pose_signal/stage2/D3_seed1/learning_curves.json").read_text())
    assert all("group_dev" not in epoch for epoch in curves)
    assert decision["gates"]["X1"]["view"] == "group_dev"
    assert decision["gates"]["T1"]["view"] == "all_holdout"


def test_smoke_all_runs_on_cpu(tmp_path, monkeypatch):
    repo = make_repo(tmp_path)
    monkeypatch.setenv("PYTHONHASHSEED", "0")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    runner.freeze_protocol(repo)
    record = runner.run_smoke(repo, device_name="cpu")
    failing = {r: {k: v for k, v in rec["checks"].items() if not v} for r, rec in record["runs"].items()}
    assert record["all_pass"], failing
