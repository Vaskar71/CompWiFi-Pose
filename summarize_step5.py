"""Read-only Step-5 summary: prints markdown tables from run artifacts (no computation of new metrics)."""

import argparse
import json
from pathlib import Path

from step5_config import resolve_path


SPLITS = ("train", "temporal_holdout", "group_dev")


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _f(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def summarize(repo_root):
    root = resolve_path(repo_root, "artifact_root")
    lines = []
    run_dirs = sorted((root / "runs").glob("R*_seed*"))
    baselines = None
    lines.append("| Run | Seed | Split | Abs MPJPE | Centroid err | Rel MPJPE | Pred std (samples) | Pred std (joints) | Shuffled abs | PA-MPJPE mm* | PCK@0.5 % | Matched (of N) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|")
    gate_lines = ["| Run | Seed | G3 | G4 (ratio) | G5 abs ratio / rel ratio | G6 abs ratio / rel ratio | G3–G5 | G3–G6 | Reload exact | Runtime s |",
                  "|---|---|---|---|---|---|---|---|---|---|"]
    for run_dir in run_dirs:
        result = _load(run_dir / "result.json")
        decision = _load(run_dir / "decision.json")
        if result.get("crashed"):
            gate_lines.append(f"| {result['run_id']} | {result['seed']} | crashed: {result['reason']} | | | | no | no | — | — |")
            continue
        baselines = result["baselines"]
        for split in SPLITS:
            e = result["final_epoch_evaluation"][split]
            rm = e["research_metrics"]["overall"]
            lines.append(
                f"| {result['run_id']} | {result['seed']} | {split} | {_f(e['abs_mpjpe'])} | {_f(e['centroid_error'])} | "
                f"{_f(e['rel_mpjpe'])} | {_f(e['pred_std_across_samples'])} | {_f(e['pred_std_across_joints'])} | "
                f"{_f(e['shuffled_pairing_abs_mpjpe'])} | {_f(rm['matched_pampjpe_mm'], 1)} | {_f(rm['pck_percent'], 1)} | "
                f"{rm['matched_people']} of {rm['samples']} |"
            )
        g = decision["gates"]
        gate_lines.append(
            f"| {result['run_id']} | {result['seed']} | {'pass' if g['G3']['pass'] else 'FAIL'} "
            f"(c-std {g['G3']['pred_centroid_std']:.3f}/{g['G3']['gt_centroid_std']:.3f}; "
            f"abs/shuf {g['G3']['train_abs_mpjpe']:.3f}/{g['G3']['train_shuffled_abs_mpjpe']:.3f}) | "
            f"{'pass' if g['G4']['pass'] else 'FAIL'} ({g['G4']['ratio_to_mean_pose']:.3f}) | "
            f"{'pass' if g['G5']['pass'] else 'FAIL'} ({g['G5']['abs_ratio_to_mean_pose']:.3f} / {g['G5']['rel_ratio_to_mean_skeleton']:.3f}) | "
            f"{'pass' if g['G6']['pass'] else 'FAIL'} ({g['G6']['abs_ratio_to_mean_pose']:.3f} / {g['G6']['rel_ratio_to_mean_skeleton']:.3f}) | "
            f"{decision['G3_to_G5_pass']} | {decision['G3_to_G6_pass']} | {result['reload_reproduces_final_metrics_exactly']} | "
            f"{result['runtime_seconds']:.0f} |"
        )
    out = ["## Final-epoch metrics", *lines, "", "*mm values provisional (assumes stored unit = metre).", "",
           "## Gates", *gate_lines, ""]
    if baselines:
        out += ["## Baselines (train-mean pose / mean skeleton)", "| Split | Mean-pose abs | Mean-pose centroid err | Mean-skeleton rel |", "|---|---|---|---|"]
        for split in SPLITS:
            b = baselines[split]
            out.append(f"| {split} | {_f(b['mean_pose_abs_mpjpe'])} | {_f(b['mean_pose_centroid_error'])} | {_f(b['mean_skeleton_rel_mpjpe'])} |")
    return "\n".join(out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    print(summarize(parser.parse_args().repo_root))
