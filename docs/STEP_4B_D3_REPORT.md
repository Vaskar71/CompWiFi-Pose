# Step-4B-D3 Trainable-Encoder Extended-Step Diagnostic Report

Date: 2026-11-07

## Verdict

**D2_REPRODUCTION_MISMATCH_STOPPED — the 800-step D3 diagnostic did not proceed past the frozen D2 reproduction gate.**

The D3 verifier and frozen protocol were written before model construction, and the pre-run protected-record checks passed. The fresh seed-0 trainable-encoder run reproduced the recorded D2 initial train-mode MPJPE exactly, but it did **not** reproduce the recorded D2 step-100 or step-200 train-mode MPJPE values. Under the frozen stop rule, D3 stopped before extending past step 200. No protocol adjustment, rerun, or other fix was attempted.

No `TRAINABLE_ENCODER_EXTENDED_STEP_CRITERION_MET` or `TRAINABLE_ENCODER_EXTENDED_STEP_ABOVE_CRITERION` status was assigned, because the mandatory D2 reproduction gate prevented the 800-step measurement from being validly started/completed.

Original Step-4B remains:

```json
{
  "status": "BLOCKED",
  "reason": "The frozen numerical overfit criterion failed; Step-4B training is blocked."
}
```

D1 remains `EXTENDED_STEP_DIAGNOSTIC_ABOVE_CRITERION`; D2 remains `TRAINABLE_ENCODER_DIAGNOSTIC_ABOVE_CRITERION`. This report makes no scientific feasibility, accuracy, novelty, generalization, or state-of-the-art claim.

## Files added or changed for Step-4B-D3

Source and tests:

- `step4b_d3_diagnostic.py`
- `tests/test_step4b_d3_diagnostic.py`

Documentation:

- `docs/STEP_4B_D3_PROTOCOL.md`
- `docs/STEP_4B_D3_REPORT.md`
- `docs/CURRENT_TASK.md`

Generated ignored D3 artifacts:

- `data/_artifacts/step4b_d3/input_verification.json`
- `data/_artifacts/step4b_d3/step4b_d3_freeze.json`
- `data/_artifacts/step4b_d3/step4b_d3_reproduction_mismatch.json`

Because D3 stopped at the reproduction gate, it did **not** write D3 checkpoint summaries, prediction arrays, per-step trajectory JSON, final state-dict, result JSON, or decision JSON. No D2 or D1 artifact was overwritten.

`model/model.py` was not changed for D3. The D3 runner reuses the opt-in D2 trainable-encoder builder and the legacy default remains `train_encoder=False`.

## Scope and label access

- Training was limited to the corrected `freeze_v2` / `pilot_data_v2` internal-training pilot path.
- The fixed ordered eight-sample batch remained:
  `S11_06_10`, `S11_06_11`, `S11_06_13`, `S11_06_14`,
  `S11_06_16`, `S11_06_17`, `S11_06_19`, `S11_06_20`.
- Development samples were not loaded.
- Official test samples and arrays were not read.
- No multi-person, unlabelled, compositional, epoch, or development-evaluation run was started.
- No packages were installed or removed; no files were downloaded or deleted; no commit, push, or remote operation was performed.

## Frozen protocol and protected records

The separate frozen protocol was written first at `docs/STEP_4B_D3_PROTOCOL.md`. The D3 freeze at `data/_artifacts/step4b_d3/step4b_d3_freeze.json` records:

- diagnostic version `compwifi-step4b-d3-v1`;
- exactly `800` D3 optimizer steps;
- D2 comparison budget `200` steps;
- the only experimental change: optimizer-step budget `200 -> 800`;
- `train_encoder=True`;
- fresh seed-0 initialization with no D2 checkpoint resume;
- the same ordered eight-sample batch;
- the unchanged absolute criterion `<=0.10` stored units;
- the D2 reproduction gate values.

The pre-implementation hash audit confirmed the protected original Step-4B and D1 hashes unchanged and recorded the current D2 record hashes. The D3 pre-run verifier then rechecked the protected original, D1, and D2 source/result/decision records; all checks printed `unchanged: true`. The current post-D2 `model/model.py` hash also matched the D3 freeze expectation.

One prior-record transcription issue was observed and preserved without editing: the D2 report/freeze display a 65-character hash for `step4b_single_person_pilot.py`, while the D2 pre-edit audit contains the valid 64-character hash. The actual file matched the valid pre-edit value. This did not cause the D3 numerical reproduction mismatch, but it is disclosed as a record-format discrepancy.

## Exact commands and observed outputs

### 1. Focused D3 tests

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_step4b_d3_diagnostic.py -q -p no:cacheprovider
```

Observed result:

```text
5 passed, 1 warning in 6.25s
```

The warning was the known external `timm.models.layers` deprecation warning. The focused tests verify that D3 uses the D2 trainable-encoder builder/optimizer procedure, differs only in optimizer-step budget, preserves seed-0 initialization, and keeps the legacy default encoder frozen with decoder-only optimizer coverage.

### 2. Complete fixture-backed focused suite

```bash
PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

Observed result:

```text
62 passed, 1 warning in 10.69s
```

There were 0 skips. The warning was the same external `timm.models.layers` deprecation warning.

### 3. D3 pre-run verifier

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_d3_diagnostic.py --verify-only --artifact-root data/_artifacts/step4b_d3 --device auto
```

Observed result: exit code `0`. The verifier printed the frozen seed, exact batch pairings, original `BLOCKED` decision, D1 above-criterion status, D2 above-criterion status, protected hash checks, and:

```text
input_verification=data\_artifacts\step4b_d3\input_verification.json
```

### 4. Frozen D3 run

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_d3_diagnostic.py --artifact-root data/_artifacts/step4b_d3 --device auto
```

Observed result: exit code `3`. The run wrote the D3 freeze, constructed the fresh seed-0 trainable-encoder configuration, reproduced the initial checkpoint, reached the D2 reproduction gate, and stopped. It printed:

```json
{
  "mismatches": [
    {
      "actual_mpjpe_stored_units": 0.40826788544654846,
      "checkpoint": "step_100",
      "expected_mpjpe_stored_units": 0.5151628255844116,
      "step": 100
    },
    {
      "actual_mpjpe_stored_units": 0.3430270552635193,
      "checkpoint": "step_200",
      "expected_mpjpe_stored_units": 0.46943652629852295,
      "step": 200
    }
  ],
  "status": "D2_REPRODUCTION_MISMATCH_STOPPED",
  "stop_rule": "D3 stopped before extending past the D2 200-step checkpoint; no protocol adjustment was made."
}
```

The mismatch artifact is:

- `data/_artifacts/step4b_d3/step4b_d3_reproduction_mismatch.json`

The initial D2 checkpoint did not appear in the mismatch list; the D3 initial train-mode MPJPE therefore matched the recorded D2 initial value `4.920130252838135` exactly.

The step-100 actual value was `-0.10689494013786314` stored units below the recorded D2 checkpoint. The step-200 actual value was `-0.12640947103500365` stored units below the recorded D2 checkpoint.

## Required D3 measurements not available

Because the run correctly stopped at the reproduction gate, the following requested 800-step measurements were not produced and are not guessed:

| Required measurement | Status |
|---|---|
| Step-800 train-mode MPJPE | NOT PRODUCED — stopped before valid 800-step continuation |
| Final/initial MPJPE ratio | NOT PRODUCED |
| Late-trajectory D3 behavior | NOT PRODUCED |
| D3 final comparison with D1 | NOT PRODUCED |
| D3 final comparison with D2 | NOT PRODUCED |
| D3 per-step trajectory artifact | NOT WRITTEN — stop rule raised before trajectory serialization |
| D3 checkpoint/prediction/final-state artifacts | NOT WRITTEN |
| D3 criterion terminal status | NOT ASSIGNED |

The in-loop finite-loss and finite-gradient checks had not failed when the step-200 reproduction mismatch was raised, but the complete per-step finite-evidence artifact was not written because the run stopped before serialization.

## Interpretation and uncertainty

The only supported interpretation is procedural:

- the D3 freeze and pre-run protected-record verification succeeded;
- the D3 run reproduced the D2 initial measurement;
- the same fresh trainable-encoder procedure did not reproduce D2 at steps 100 and 200 under the D3 run;
- the frozen stop rule therefore prevented a valid extended-budget measurement.

No root cause is claimed. In particular, this report does not claim that D2 was wrong, that D3 was wrong, that CUDA nondeterminism caused the mismatch, or that extending the trainable-encoder run would pass/fail the criterion. Establishing a cause would require a separately approved reproducibility/determinism investigation; none was started here.

## Boundaries honored

Not performed:

- Development-sample loading or evaluation.
- Official-test access.
- Epoch training.
- Step-4C or compositional experiments.
- Architecture, optimizer, data, criterion, or threshold changes after seeing the mismatch.
- Another D3 run or automatic fix attempt.
- Package installation/removal.
- Dataset download or deletion.
- Source dataset modification.
- Commit, push, or remote changes.

## Current status and recommended next action

Step-4B remains **BLOCKED**. D1 and D2 remain above-criterion diagnostics. D3 is stopped at `D2_REPRODUCTION_MISMATCH_STOPPED`; no D3 criterion status is available.

The recommended next task is a separately approved reproducibility/determinism audit of the D2 procedure and records before any new D3 attempt. Do not begin Step-4C and do not treat this stopped D3 run as evidence for or against scientific feasibility.
