"""POST-HOC addendum to Step 5b (user-approved after the INCONCLUSIVE verdict).

Runs the frozen stage-2 recipe for D3 (train_zscore, all_train_dense, seeds 0/1/2) through the
unchanged, hash-verified ``step5b_pose_signal.train_and_evaluate``. Outputs go to
``data/_artifacts/step5b_pose_signal/posthoc_d3/stage2/`` so that they are never confused with the
protocol's selection, stage-2 runs or verdict, all of which stay unchanged.

This is a deviation from the frozen selection rule and must be reported as such. It uses
``group_dev`` (development labels) a second time, once per seed at the final epoch.
"""

import argparse
import json
from pathlib import Path

from step5b_config import resolve_path
from step5b_pose_signal import _write_json, train_and_evaluate

RUN_ID = "D3"
SEEDS = (0, 1, 2)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--seed", type=int, choices=SEEDS, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args(argv)
    out_root = resolve_path(args.repo_root, "artifact_root") / "posthoc_d3"
    out_root.mkdir(parents=True, exist_ok=True)
    note = out_root / "POSTHOC_NOTE.json"
    if not note.exists():
        _write_json(note, {
            "status": "post-hoc, not part of the frozen Step-5b protocol verdict",
            "reason": "protocol tie rule selected D1; user approved confirming the dense recipe D3",
            "run": RUN_ID, "seeds": list(SEEDS),
            "group_dev_reuse": "second use of development labels, evaluation only",
        })
    _, decision = train_and_evaluate(args.repo_root, "stage2", RUN_ID, args.seed, args.device, out_root=out_root)
    print(json.dumps(decision["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
