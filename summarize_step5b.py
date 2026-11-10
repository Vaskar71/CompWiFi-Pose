"""Print the Step-5b report tables from run artifacts (read-only)."""

import argparse
import json
from pathlib import Path

from step5b_config import resolve_path


def _load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _f(value, digits=4):
    return "—" if value is None else f"{value:.{digits}f}"


def run_rows(root, stage):
    rows = []
    for run_dir in sorted((root / stage).glob("*_seed*")):
        result = _load(run_dir / "result.json")
        decision = _load(run_dir / "decision.json")
        rows.append((run_dir.name, result, decision))
    return rows


def metrics_table(rows):
    lines = [
        "| Run | View | Abs | Centroid | Rel | Oracle abs / rel | Mean pose abs | Mean skel rel | Rel/oracle | Rel/mean skel | Abs/mean pose | Motion skill | Pred std (samples) | PA-MPJPE mm* | PCK@0.5 % | Matched |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for name, result, _ in rows:
        if result.get("crashed"):
            lines.append(f"| {name} | CRASHED: {result['reason']} |" + " |" * 15)
            continue
        for view, ev in result["final_epoch_evaluation"].items():
            rm = ev.get("research_metrics", {}).get("overall", {})
            oracle = (f"{_f(ev.get('oracle_abs_mpjpe'))} / {_f(ev.get('oracle_rel_mpjpe'))}"
                      if "oracle_rel_mpjpe" in ev else "—")
            pa = rm.get("matched_pampjpe_mm")
            pck = rm.get("pck_percent")
            matched = rm.get("matched_people")
            matched_text = "—" if matched is None else f"{matched} of {ev['samples']}"
            lines.append(
                f"| {name} | {view} | {_f(ev['abs_mpjpe'])} | {_f(ev['centroid_error'])} | {_f(ev['rel_mpjpe'])} | {oracle} | "
                f"{_f(ev['mean_pose_abs_mpjpe'])} | {_f(ev['mean_skeleton_rel_mpjpe'])} | {_f(ev.get('rel_ratio_to_oracle'), 3)} | "
                f"{_f(ev['rel_ratio_to_mean_skeleton'], 3)} | {_f(ev['abs_ratio_to_mean_pose'], 3)} | {_f(ev['motion_skill'], 3)} | "
                f"{_f(ev['pred_std_across_samples'])} | {_f(pa, 1)} | {_f(pck, 1)} | "
                f"{matched_text} |"
            )
    return "\n".join(lines)


def gate_table(rows):
    lines = ["| Run | T1 view | T1 ratio (rel/oracle) | T1 | abs/oracle (reported) | X1 view | X1 ratio (rel/mean skel) | X1 | abs/mean pose (reported) | Reload exact | Runtime s |",
             "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, result, decision in rows:
        if decision.get("crashed"):
            lines.append(f"| {name} | — | — | FAIL (crashed) | — | — | — | FAIL (crashed) | — | — | — |")
            continue
        t1, x1 = decision["gates"]["T1"], decision["gates"]["X1"]
        lines.append(
            f"| {name} | {t1['view']} | {t1['ratio']:.3f} | {'pass' if t1['pass'] else 'FAIL'} | {t1['abs_ratio_to_oracle_reported']:.3f} | "
            f"{x1['view']} | {x1['ratio']:.3f} | {'pass' if x1['pass'] else 'FAIL'} | {x1['abs_ratio_to_mean_pose_reported']:.3f} | "
            f"{decision['reload_exact']} | {result['runtime_seconds']:.0f} |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    args = parser.parse_args(argv)
    root = resolve_path(args.repo_root, "artifact_root")
    for stage in ("stage1", "stage2"):
        rows = run_rows(root, stage)
        if not rows:
            continue
        print(f"## {stage}\n\n### Final-epoch metrics\n{metrics_table(rows)}\n\n### Gates\n{gate_table(rows)}\n")


if __name__ == "__main__":
    main()
