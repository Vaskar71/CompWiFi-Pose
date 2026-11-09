# Step-4B-D3 Frozen Protocol

Date: 2026-11-07

## Purpose

Step-4B-D3 is a separately frozen, 800-step extended-budget diagnostic of the existing trainable-encoder configuration from Step-4B-D2. The purpose is honest measurement of whether the unchanged absolute train-mode MPJPE criterion is met when the D2 optimizer-step budget is extended from 200 to 800 steps.

D3 is not a Step-4B rerun with post-hoc tuning, not a scientific feasibility experiment, and not a Step-4C compositional control. Meeting the MPJPE threshold is not required for correct execution of this protocol; the permitted terminal records are:

- `TRAINABLE_ENCODER_EXTENDED_STEP_CRITERION_MET` when final train-mode MPJPE is `<= 0.10` stored units.
- `TRAINABLE_ENCODER_EXTENDED_STEP_ABOVE_CRITERION` when final train-mode MPJPE is `> 0.10` stored units.

Original Step-4B remains `BLOCKED` regardless of the D3 measurement.

## Lineage

| Record | Reported result |
|---|---:|
| Original Step-4B, frozen encoder, 200 steps | final MPJPE `0.34014594554901123` stored units; `BLOCKED` because the value exceeded `0.10` |
| Step-4B-D1, frozen encoder, 800 steps | final MPJPE `0.32848620414733887` stored units; above criterion |
| Step-4B-D2, trainable encoder, 200 steps | initial MPJPE `4.920130252838135`; step-100 MPJPE `0.5151628255844116`; final MPJPE `0.46943652629852295`; above criterion |

## Scope and label access

Training uses only the corrected `freeze_v2` / `pilot_data_v2` internal-training pilot artifacts and the same ordered eight-sample internal-training batch as D2:

```text
S11_06_10, S11_06_11, S11_06_13, S11_06_14,
S11_06_16, S11_06_17, S11_06_19, S11_06_20
```

Development samples are not loaded. Official test samples and arrays are not read. No multi-person, unlabelled, compositional, epoch, or development-evaluation run is started. No package installation/removal, download, deletion, source-dataset modification, commit, push, or remote operation is part of this protocol.

## Frozen experimental change

The only experimental change from Step-4B-D2 is the optimizer-step budget:

```text
D2: 200 optimizer steps
D3: 800 optimizer steps
```

Everything else in the D2 runtime configuration remains unchanged:

- corrected `freeze_v2` / `pilot_data_v2` training data;
- the exact ordered eight-sample batch above;
- fresh seed-0 initialization, with no resume from the D2 checkpoint;
- `train_encoder=True`;
- model family and dimensions: from-scratch `MAE_ViT` encoder plus `ViT_Pose_Decoder`, `Q=1`, 14 keypoints, 3 stored coordinate dimensions, image `(180,20)`, patch `(2,2)`, encoder layers/heads `4/4`, MAE decoder layers/heads `2/2`, embedding width `256`, input channels `3`;
- CSI preprocessing, loader normalization, target representation, loss, dropout, batch size, and device-selection procedure;
- SGD with learning rate `0.01` and momentum `0.9`;
- all 97 D2 trainable parameter tensors covered exactly once;
- train-mode checkpoint measurement procedure;
- stored coordinate order and the unchanged absolute criterion `final train-mode MPJPE <= 0.10` stored units.

## Protected-record verification before training

Before model construction, the D3 verifier must verify the protected original Step-4B and D1 records, load and check the protected D2 records, and write the D3 freeze only after those checks pass.

The original and D1 expected SHA-256 values below were checked before D3 implementation and matched. Current D2 records were also hashed for the D3 freeze.

| Record | SHA-256 |
|---|---|
| `step4b_config.py` | `8532d6fe0e017e40679819b19c5e14754d237279b2deb352725a2829be41a239` |
| `step4b_single_person_pilot.py` | `2848d7b8cc56429da6d74c51bafc7782229bf9fbf9b38eaba2149688d83a1169` |
| `step4b_d1_diagnostic.py` | `c095dac5e400b12e766390065a29e258b177b636d769d50923d99a14c9a0327b` |
| `step4b_d2_diagnostic.py` | `cd8aa4e36ea398769983c273ae3fdbc2f211650b8e35e392669848f8c2d8e161` |
| `model/model.py` (current post-D2) | `7164ad53d4d9cf71316e3e379983576b82bfced17ec58ff67c7febadc121840f` |
| `data/_artifacts/step4b_pilot/freeze_v2/step4b_freeze_config.json` | `807b3e8bc37136e2c2000baff77363dc16761fc02e19d90ec58b6ab3710fdb16` |
| `data/_artifacts/step4b_pilot/preprocessing_validation_v2.json` | `32b7b1e4360abbe76b71cdc01a6001d8b4d7348bb48fba2677a96421cd118d41` |
| `data/_artifacts/step4b_pilot/overfit_result.json` | `e30a5aad15a2bab5844ab5c14e2738d30d26fef2aba956adb9015794c3253a06` |
| `data/_artifacts/step4b_pilot/decision.json` | `2d73224dedbf9c3f869b040f290376df9d582f5865ab13296352eb86e73f97e3` |
| `data/_artifacts/step4b_d1/step4b_d1_result.json` | `92a6673ae63439161427a5888cfb60b960d124b37c1b6803cf1095a627266586` |
| `data/_artifacts/step4b_d1/step4b_d1_decision.json` | `155a6c9580ae8a22642588730da4c6a8e482bef81c93406790ad85b20ec0a1b8` |
| `data/_artifacts/step4b_d2/parameter_audit_before_edit.json` | `0a516ad84a8c57fc7584f6122baca3b52e99e8eb7081432a3d4ae9f020db33e6` |
| `data/_artifacts/step4b_d2/step4b_d2_freeze.json` | `9e04365bac612149314f6122a369def82f5cc49a53d6c37f242e3f9d6739229f` |
| `data/_artifacts/step4b_d2/step4b_d2_result.json` | `f83620f8736a3e35ef4ee9f3adefa6aa6aaa8628049058b5520b20c7a146e0f4` |
| `data/_artifacts/step4b_d2/step4b_d2_decision.json` | `9508f2bdad75f6d19b7c9572a29401db45b36c2db0f2dba89ee30eb9ab222e95` |

The D2 report/freeze contain a 65-character transcribed value for the original entry-point hash. The valid 64-character value from the D2 pre-edit audit is recorded above and matched the file. The prior D2 report/freeze were not edited.

## In-run D2 reproduction gate

The D3 run must use the same deterministic fresh seed-0 trainable-encoder procedure as D2 and must reproduce these train-mode checkpoint measurements exactly before it is allowed to continue beyond step 200:

| Checkpoint | Required train-mode MPJPE, stored units |
|---|---:|
| Initial | `4.920130252838135` |
| After 100 optimizer steps | `0.5151628255844116` |
| After 200 optimizer steps | `0.46943652629852295` |

If any value does not reproduce, D3 must stop and report the mismatch. It must not modify the protocol, adjust the model, or force agreement.

## Measurement and artifact procedure

The D3 runner must:

1. Verify the corrected freeze/pilot records, original `BLOCKED` decision, D1 above-criterion result, and D2 source/result/decision records.
2. Write `data/_artifacts/step4b_d3/step4b_d3_freeze.json` before model construction and refuse to overwrite it.
3. Construct a fresh seed-0 trainable-encoder model, SGD optimizer, ordered eight-sample loader, and selected device without loading a prior checkpoint.
4. Audit that the optimizer covers exactly the 97 intended trainable parameter tensors once.
5. Record complete per-step pre-update losses/MPJPE and finite-gradient evidence for all 800 steps.
6. Record train-mode checkpoints at steps 0, 100, 200, and 800 using the D2 checkpoint measurement procedure.
7. Verify the D2 initial, step-100, and step-200 reproduction gate.
8. Record parameter-change evidence for all 97 trainable parameter tensors.
9. Save checkpoint summaries, prediction arrays, trajectory JSON, final `state_dict`, result JSON, and decision JSON separately under `data/_artifacts/step4b_d3/`.
10. Save an eval-mode reload measurement only as a secondary diagnostic; it must not replace the frozen train-mode criterion.

## Reporting requirements

`docs/STEP_4B_D3_REPORT.md` must report the actual commands and outcomes, including:

- step-800 train-mode MPJPE;
- final/initial MPJPE ratio;
- finite-loss, finite-gradient, optimizer-coverage, and parameter-change evidence;
- the complete per-step trajectory artifact;
- late-trajectory segment means and short-window behavior;
- exact comparisons with D1 and D2;
- any reproduction mismatch or blocker, if one occurs;
- the required terminal status, exactly one of the two permitted D3 statuses.

The report must not alter the original `BLOCKED` decision and must make no scientific feasibility, generalization, accuracy, novelty, or state-of-the-art claim.

## Planned commands

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_step4b_d3_diagnostic.py -q -p no:cacheprovider
```

Fixture-backed suite:

```bash
PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

D3 verification and frozen diagnostic:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_d3_diagnostic.py --verify-only --artifact-root data/_artifacts/step4b_d3 --device auto
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_d3_diagnostic.py --artifact-root data/_artifacts/step4b_d3 --device auto
```
