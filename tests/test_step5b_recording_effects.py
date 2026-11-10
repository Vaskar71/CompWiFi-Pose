import numpy as np
import pytest

import analyze_step5_recording_effects as ana


def _ids(recording, frames):
    return [f"{recording}_{f}" for f in frames]


def test_recording_of_and_means():
    ids = _ids("S11_01", [1, 2]) + _ids("S21_03", [5])
    values = np.array([[0.0], [2.0], [10.0]])
    means = ana.recording_means(ids, values)
    assert ana.recording_of("S11_01_17") == "S11_01"
    assert means["S11_01"].tolist() == [1.0]
    assert means["S21_03"].tolist() == [10.0]
    assert ana.oracle_predictions(["S21_03_9", "S11_01_4"], means).ravel().tolist() == [10.0, 1.0]


def test_relative_and_mpjpe():
    poses = np.zeros((2, 14, 3))
    poses[1] += 5.0
    assert np.allclose(ana.relative(poses), 0.0)
    assert ana.mpjpe(poses[0], poses[0] + np.array([3.0, 4.0, 0.0])) == pytest.approx(5.0)


def test_share_beating_oracle():
    ids = _ids("S11_01", [1, 2]) + _ids("S11_02", [1, 2])
    gt = np.zeros((4, 14, 3))
    oracle = np.full((4, 14, 3), 1.0)
    pred = np.concatenate([np.full((2, 14, 3), 0.5), np.full((2, 14, 3), 2.0)])
    share, count = ana.share_recordings_beating_oracle(ids, pred, gt, oracle)
    assert count == 2 and share == pytest.approx(0.5)


def test_deviation_correlation_perfect_and_anti():
    rng = np.random.default_rng(0)
    gt = rng.normal(size=(6, 14, 3))
    ids = _ids("S11_01", range(6))
    median, _ = ana.median_deviation_correlation(ids, gt + 7.0, gt)
    assert median == pytest.approx(1.0)
    median, _ = ana.median_deviation_correlation(ids, -gt, gt)
    assert median == pytest.approx(-1.0)


def test_nearest_mean_identification():
    rng = np.random.default_rng(1)
    centers = {"S11_01": 0.0, "S11_02": 10.0, "S21_01": 20.0}
    train_x, train_r, eval_x, eval_r = [], [], [], []
    for name, center in centers.items():
        for _ in range(4):
            train_x.append(center + rng.normal(scale=0.1, size=(3, 4, 2)))
            train_r.append(name)
        eval_x.append(center + rng.normal(scale=0.1, size=(3, 4, 2)))
        eval_r.append(name)
    accuracy, chance = ana.nearest_mean_identification(train_x, train_r, eval_x, eval_r)
    assert accuracy == 1.0 and chance == pytest.approx(1 / 3)


def test_lag_autocorrelation_uses_frame_index_pairs():
    frames = [1, 2, 3, 10, 11]
    deviations = np.array([1.0, 2.0, 3.0, -1.0, -2.0])[:, None]
    # lag 1 pairs: (1,2),(2,3),(10,11) -> a=[1,2,-1], b=[2,3,-2]
    expected = np.corrcoef([1, 2, -1], [2, 3, -2])[0, 1]
    assert ana.lag_autocorrelation(deviations, frames, 1) == pytest.approx(expected)
    assert ana.lag_autocorrelation(deviations, frames, 50) is None


def test_motion_statistics_constant_recordings():
    base = np.arange(42, dtype=float).reshape(14, 3)
    rng = np.random.default_rng(2)
    recordings = {}
    for index in range(3):
        poses = np.stack([base * (1 + index) + rng.normal(scale=0.01, size=(14, 3)) for _ in range(20)])
        recordings[f"S11_0{index}"] = (list(range(1, 21)), poses)
    stats = ana.motion_statistics(recordings, lags=(1, 2))
    assert stats["recordings"] == 3
    assert stats["within_recording_deviation"] < 0.05
    assert stats["between_recording_mean_skeleton_deviation"] > 1.0


def test_compare_and_tolerances():
    tolerance = ana._tolerances()
    assert tolerance["oracle_abs"] == pytest.approx(0.0005, abs=1e-8)
    assert tolerance["R1_seed0_share_beating_oracle"] == pytest.approx(0.005, abs=1e-8)
    measured = dict(ana.PLAN_VALUES)
    measured["oracle_abs"] = 0.1394
    rows = ana.compare(measured, tolerance)
    assert all(row["reproduces"] for row in rows.values())
    measured["autocorrelation_lag_8"] = 0.14
    assert not ana.compare(measured, tolerance)["autocorrelation_lag_8"]["reproduces"]
