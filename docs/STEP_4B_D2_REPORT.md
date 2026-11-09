# Step-4B-D2 Encoder-Trainability Diagnostic Report

Date: 2026-11-07

## Verdict

**TRAINABLE_ENCODER_DIAGNOSTIC_ABOVE_CRITERION — original Step-4B remains BLOCKED.**

The controlled D2 run made the encoder and pose decoder trainable while preserving the corrected `freeze_v2` tiny batch, seed `0`, initialization procedure, data, preprocessing, loss, SGD optimizer, learning rate `0.01`, momentum `0.9`, batch size `8`, and exactly `200` optimizer steps. All losses and gradients were finite. All `97` intended trainable parameters were covered exactly once by SGD, all `50` encoder gradients were nonzero at every recorded step, and all `50` encoder parameters changed.

Final train-mode MPJPE after 200 optimizer steps was `0.46943652629852295` stored units. This did not meet the unchanged absolute criterion of `<=0.10` stored units and was `0.12929058074951172` stored units worse than the original frozen-encoder Step-4B endpoint (`0.34014594554901123`). The measured result is reported without trying another fix.

Original Step-4B remains:

```json
{
  "status": "BLOCKED",
  "reason": "The frozen numerical overfit criterion failed; Step-4B training is blocked."
}
```

The D1 result also remains `EXTENDED_STEP_DIAGNOSTIC_ABOVE_CRITERION` with final MPJPE `0.32848620414733887` stored units. Original Step-4B and D1 artifacts were not rewritten.

This report makes no scientific feasibility, accuracy, novelty, generalization, or state-of-the-art claim.

## Scope and label access

- Only corrected `freeze_v2` / `pilot_data_v2` internal-training pilot artifacts were used for D2 training.
- The fixed ordered batch was loaded from `selected_train_ids.txt`, whose full 2,048-ID content exactly matches `freeze_v2`.
- CSI and keypoint arrays were paired by exact sample ID for all eight tiny-batch samples.
- Development samples were not loaded by the D2 runner.
- Official test samples and arrays were not read or evaluated.
- No multi-person, unlabelled, compositional, epoch, or development-evaluation run was started.
- No packages were installed or removed; no files were downloaded or deleted; no commit, push, or remote operation was performed.

## Pre-edit named-parameter audit

Artifact:

- `data/_artifacts/step4b_d2/parameter_audit_before_edit.json`

Before editing, a seed-0 CUDA runtime audit constructed the original Step-4B model and optimizer from `data/_artifacts/step4b_pilot/freeze_v2/step4b_freeze_config.json`. It directly confirmed:

| Component | Named parameters | Parameter count | Trainable named parameters | Trainable parameter count |
|---|---:|---:|---:|---:|
| Encoder | 50 | 3,390,464 | 0 | 0 |
| Pose decoder | 47 | 1,404,931 | 47 | 1,404,931 |
| Full pose model | 97 | 4,795,395 | 47 | 1,404,931 |

Optimizer audit:

- SGD learning rate `0.01`, momentum `0.9`.
- `47` optimizer parameter references.
- `47` unique optimizer parameter references.
- `0` duplicate optimizer references.
- `0` unknown optimizer references.
- The optimizer contained exactly the `47` trainable pose-decoder parameters and no frozen encoder parameter.

**Diagnosed original configuration:** original Step-4B runtime constructed a randomly initialized from-scratch encoder, froze every encoder parameter, trained every pose-decoder parameter, and passed exactly those trainable parameters to SGD.

The audit also recorded SHA-256 hashes for the original Step-4B freeze, preprocessing validation, overfit result, decision, Step-4B configuration/entry point, D1 script, D1 result, and D1 decision. Those hashes were rechecked after the D2 run.

## Implementation change and legacy behavior

Source change:

- `model/model.py` now accepts opt-in `train_encoder=False` on `ViT_Pose_Decoder` and adds `set_encoder_trainable()`.
- The default remains the legacy behavior: copied encoder parameters are frozen before pose-decoder parameters are added.
- `train_encoder=True` makes exactly the copied encoder parameters trainable; it does not change model dimensions, dropout, target representation, loss, data, preprocessing, seed, batch, optimizer type, learning rate, momentum, or step budget.
- `step4b_single_person_pilot.py` was not changed. Its original construction and optimizer filtering remain available.
- The D2 runner uses a separate `build_d2_model()` function that passes `train_encoder=True`.

Focused tests verify that the option changes `requires_grad` state but preserves seed-0 initial parameter values.

## Frozen D2 protocol

Freeze artifact, written before model construction:

- `data/_artifacts/step4b_d2/step4b_d2_freeze.json`

The only experimental setting change from original Step-4B was:

| Setting | Original Step-4B | Step-4B-D2 |
|---|---|---|
| Encoder trainability | frozen | trainable |

Explicitly unchanged:

- Corrected `freeze_v2` internal-train sample IDs.
- Ordered fixed batch of eight samples:
  `S11_06_10`, `S11_06_11`, `S11_06_13`, `S11_06_14`,
  `S11_06_16`, `S11_06_17`, `S11_06_19`, `S11_06_20`.
- Seed `0`.
- Model family and dimensions.
- Random initialization values.
- CSI preprocessing and loader normalization.
- Mean Euclidean pose loss.
- SGD optimizer.
- Learning rate `0.01`.
- Momentum `0.9`.
- Optimizer budget of exactly `200` steps.
- Batch size `8`.
- Device-selection mode.
- Stored coordinate order and the unchanged `<=0.10` stored-unit criterion.

D2 trainability audit before training:

| Component | Named parameters | Trainable named parameters | Trainable parameter count |
|---|---:|---:|---:|
| Encoder | 50 | 50 | 3,390,464 |
| Pose decoder | 47 | 47 | 1,404,931 |
| Full pose model | 97 | 97 | 4,795,395 |

SGD contained all `97` trainable parameters exactly once.

## Exact commands and observed outputs

### 1. Focused D2 unit tests

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_step4b_d2_diagnostic.py -q -p no:cacheprovider
```

Observed result:

```text
5 passed, 1 warning in 5.77s
```

The warning was the known external `timm.models.layers` deprecation warning.

### 2. Existing Step-4B and D1 tests

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_step4b_pilot.py tests/test_step4b_d1_diagnostic.py -q -p no:cacheprovider
```

Observed result:

```text
7 passed, 1 warning in 5.69s
```

### 3. Pre-run frozen-input verification

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_d2_diagnostic.py --verify-only --artifact-root data/_artifacts/step4b_d2 --device auto
```

Observed result: exit code `0`; the verifier printed the frozen seed, batch IDs, exact pairings, model/optimizer settings, original metrics, original failure, unchanged `BLOCKED` decision, unchanged D1 above-criterion result, and:

```text
input_verification=data\_artifacts\step4b_d2\input_verification.json
```

### 4. Frozen 200-step D2 diagnostic

```bash
test ! -e data/_artifacts/step4b_d2/step4b_d2_freeze.json && PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_d2_diagnostic.py --artifact-root data/_artifacts/step4b_d2 --device auto
```

Observed summary:

```text
diagnostic_run_status=COMPLETE
diagnostic_status=TRAINABLE_ENCODER_DIAGNOSTIC_ABOVE_CRITERION
device=cuda
runtime_seconds=19.235578899999382
all_losses_finite=true
all_gradients_finite=true
all_encoder_gradients_finite=true
all_encoder_gradients_nonzero=true
criterion_passed=false
initial_mpjpe_stored_units=4.920130252838135
step_100_mpjpe_stored_units=0.5151628255844116
final_mpjpe_stored_units=0.46943652629852295
final_to_initial_mpjpe_ratio=0.09541140217329257
```

The run printed:

```text
d2_result=data\_artifacts\step4b_d2\step4b_d2_result.json
d2_decision=data\_artifacts\step4b_d2\step4b_d2_decision.json
```

### 5. Complete focused suite

```bash
PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

Observed result:

```text
57 passed, 1 warning in 9.42s
```

There were **0 skips**. The warning was the same external `timm.models.layers` deprecation warning.

## Learning evidence

### Checkpoints

| Checkpoint | Loss = MPJPE, stored units |
|---|---:|
| Initial | 4.920130252838135 |
| Step 100, after 100 optimizer steps | 0.5151628255844116 |
| Step 200, after 200 optimizer steps | 0.46943652629852295 |

All checkpoint predictions, targets, and recorded losses were finite. The final checkpoint state_dict was saved and reloaded; eval-mode reloaded MPJPE was `0.43328580260276794` stored units and finite. The train-mode checkpoint value above is the criterion value, matching the original Step-4B measurement procedure.

### Trajectory evidence

The complete per-step pre-update records and gradient evidence are in:

- `data/_artifacts/step4b_d2/loss_trajectory.json`

Summary:

| Evidence | Value |
|---|---:|
| First pre-update loss | 4.924509525299072 |
| Step-100 pre-update loss | 0.588363766670227 |
| Step-200 pre-update loss | 0.3613433539867401 |
| Minimum pre-update loss | 0.34691399335861206 |
| Minimum pre-update step | 153 |
| Mean steps 1–100 | 0.9375664529204368 |
| Mean steps 101–200 | 0.47824851781129835 |
| Mean steps 151–200 | 0.43299115061759946 |
| Last-50 minus previous-50 mean | -0.09051473438739782 |
| Linear slope, all 200 steps | -0.005144628876730657 stored units/step |
| Linear slope, steps 101–200 | -0.0017588922076492814 stored units/step |

D2 was still improving over its own late trajectory on average, but the measured endpoint remained far above `0.10` and worse than the original frozen-encoder endpoint. It should not be described as a D1-style flat plateau, nor as evidence that the criterion was nearly satisfied.

### Gradient and parameter-change evidence

| Evidence | Result |
|---|---:|
| Trainable gradient tensors per step | 97 |
| All trainable gradients finite | true |
| Encoder gradient tensors per step | 50 |
| All encoder gradients finite | true |
| Encoder gradients nonzero per step | 50 of 50 at every step |
| Maximum absolute encoder gradient observed | 0.2116079032421112 |
| Trainable parameters changed by step 200 | 97 of 97 |
| Encoder parameters changed by step 200 | 50 of 50 |
| Maximum absolute encoder parameter change | 0.018428806215524673 |
| Maximum absolute any-trainable parameter change | 0.7603052854537964 |

Per-sample final MPJPE ranged from `0.4619998193189736` to `0.4756385945634379` stored units, tightly clustered around the final mean.

## Interpretation only

The controlled result is:

- a trainable encoder was correctly constructed, optimized, and directly updated;
- every loss and intended gradient was finite;
- D2 reduced initial MPJPE to `9.54%` of its initial value but ended at `0.4694365263` stored units;
- this was an improvement of `4.450693726539612` stored units from D2's own initial value, but a regression of `0.12929058074951172` stored units relative to the original 200-step frozen-encoder endpoint;
- therefore, making the encoder trainable under the otherwise frozen settings did **not** explain the original Step-4B failure as caused by an unavailable frozen-encoder optimization pathway;
- this result does not identify a root cause and does not justify changing the architecture, optimizer, normalization, data checks, or criterion after seeing the outcome.

No scientific feasibility, accuracy, novelty, generalization, or state-of-the-art claim is made.

## Files added or changed for Step-4B-D2

Source and tests:

- `model/model.py`
- `step4b_d2_diagnostic.py`
- `tests/test_step4b_d2_diagnostic.py`

Documentation:

- `docs/STEP_4B_D2_REPORT.md`
- `docs/CURRENT_TASK.md`

Generated ignored D2 artifacts:

- `data/_artifacts/step4b_d2/parameter_audit_before_edit.json`
- `data/_artifacts/step4b_d2/input_verification.json`
- `data/_artifacts/step4b_d2/step4b_d2_freeze.json`
- `data/_artifacts/step4b_d2/loss_trajectory.json`
- `data/_artifacts/step4b_d2/checkpoint_step_0000.json`
- `data/_artifacts/step4b_d2/checkpoint_step_0100.json`
- `data/_artifacts/step4b_d2/checkpoint_step_0200.json`
- `data/_artifacts/step4b_d2/checkpoint_predictions.npz`
- `data/_artifacts/step4b_d2/step4b_d2_result.json`
- `data/_artifacts/step4b_d2/step4b_d2_decision.json`
- `data/_artifacts/step4b_d2/step4b_d2_final_state_dict.pt`

`git check-ignore` verified that the D2 artifacts are ignored by the existing `data/_artifacts/` rule.

## Original-record preservation

After the D2 run, the following SHA-256 hashes still matched the pre-edit audit:

| Record | SHA-256 |
|---|---|
| `step4b_config.py` | `8532d6fe0e017e40679819b19c5e14754d237279b2deb352725a2829be41a239` |
| `step4b_single_person_pilot.py` | `2848d7b8cc56429da6d74c51bafc77822229bf9fbf9b38eaba2149688d83a1169` |
| `freeze_v2/step4b_freeze_config.json` | `807b3e8bc37136e2c2000baff77363dc16761fc02e19d90ec58b6ab3710fdb16` |
| `preprocessing_validation_v2.json` | `32b7b1e4360abbe76b71cdc01a6001d8b4d7348bb48fba2677a96421cd118d41` |
| `overfit_result.json` | `e30a5aad15a2bab5844ab5c14e2738d30d26fef2aba956adb9015794c3253a06` |
| `decision.json` | `2d73224dedbf9c3f869b040f290376df9d582f5865ab13296352eb86e73f97e3` |
| `step4b_d1_diagnostic.py` | `c095dac5e400b12e766390065a29e258b177b636d769d50923d99a14c9a0327b` |
| `step4b_d1_result.json` | `92a6673ae63439161427a5888cfb60b960d124b37c1b6803cf1095a627266586` |
| `step4b_d1_decision.json` | `155a6c9580ae8a22642588730da4c6a8e482bef81c93406790ad85b20ec0a1b8` |

`model/model.py` changed only to add the opt-in encoder-trainability mechanism.

## Remaining hypotheses and uncertainties

- The original frozen-encoder gate failure is not explained by the encoder being unable to receive gradients; D2 received and applied encoder gradients but performed worse under the frozen optimizer setting.
- A trainable encoder may make the fixed SGD(`lr=0.01`, `momentum=0.9`) optimization problem harder rather than easier.
- Small-batch training-mode dropout and representation co-adaptation may contribute to checkpoint variability; the frozen-encoder original endpoint remains better under the identical measurement procedure.
- The unresolved coordinate/unit and target-representation uncertainties recorded in the Step-3/4A audits remain unresolved.
- D2 used one seed and one fixed tiny batch only. It is an engineering diagnostic, not a scientific feasibility result.

## Recommended next action

Do not begin Step-4C and do not run another unapproved diagnostic. Original Step-4B remains **BLOCKED**; D1 remains an above-criterion extended-step diagnostic; D2 is an above-criterion trainable-encoder diagnostic. Any future engineering diagnosis or baseline rerun requires a separately approved, newly frozen protocol.
