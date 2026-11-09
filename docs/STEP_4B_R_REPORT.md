# Step-4B-R Deterministic Bounded Baseline Report

Date: 2026-11-08

## Verdict

**BASELINE_COMPLETE — the bounded protocol finished and all required measurements were recorded.**

This is a completion status, not an accuracy pass. The frozen run completed 10
epochs, saved and reloaded the epoch-10 state dictionary, and reported the
training-derived mean-pose comparison. Under the frozen stored-unit metric, the
final model did **not** beat the constant training-mean predictor on either
partition:

| Comparison | Result |
|---|---|
| Final model vs mean pose, training | **did not beat mean predictor** |
| Final model vs mean pose, development | **did not beat mean predictor** |

Original Step-4B remains:

```json
{
  "status": "BLOCKED",
  "reason": "The frozen numerical overfit criterion failed; Step-4B training is blocked."
}
```

Step-4B-R is a separately named baseline. It is not a rerun, retroactive pass,
or reversal of Step-4B, D1, D2, D3, or D4. It makes no scientific
feasibility, accuracy, novelty, generalization, or state-of-the-art claim.

## Scope and label access

The run used only the corrected existing artifacts:

- `data/_artifacts/step4b_pilot/freeze_v2/`
- `data/_artifacts/step4b_pilot/pilot_data_v2/`
- `data/_artifacts/step4b_pilot/preprocessing_validation_v2.json`

Verified data boundaries:

- Training: 2,048 labelled filename-marked single-person internal-train samples.
- Development evaluation: 512 labelled filename-marked single-person
  internal-development samples.
- Train/development ID intersection: `0`.
- Train/development filename-group intersection: `0`.
- Every selected split assignment matched `freeze_v2`.
- The persisted pilot selected-ID files exactly matched `freeze_v2`.
- Corrected preprocessing validation remained 2,560 validated IDs and zero
  failures.
- Development labels were used only for no-gradient evaluation and were never
  used for training, loss computation, early stopping, checkpoint selection,
  or hyperparameter selection.
- The constant mean pose was computed from the 2,048 internal-training labels
  only.
- Official-test CSI/keypoint arrays were not read.
- Multi-person samples, unlabelled training CSI, compositional training, and
  external/pretrained checkpoints were not used.

All 40 protected historical source/report/result records verified by the D4
record were unchanged before the run and remained unchanged afterward.

## Files and frozen configuration

Added source and tests:

- `step4b_r_config.py`
- `step4b_r_baseline.py`
- `tests/test_step4b_r_baseline.py`

Added documentation:

- `docs/STEP_4B_R_PROTOCOL.md`
- `docs/STEP_4B_R_REPORT.md`

Generated ignored evidence:

- `data/_artifacts/step4b_r/deterministic_smoke.json`
- `data/_artifacts/step4b_r/deterministic_smoke_state_dict.pt`
- `data/_artifacts/step4b_r/run.log`
- `data/_artifacts/step4b_r/run_error.json`
- `data/_artifacts/step4b_r/baseline_run.log`
- complete final run evidence under `data/_artifacts/step4b_r/baseline/`

The frozen source configuration is `compwifi-step4b-r-v1` with seed `0`.
Its separate generated freeze is:

- `data/_artifacts/step4b_r/baseline/step4b_r_freeze.json`

Frozen experimental settings:

| Setting | Value |
|---|---|
| Model | existing from-scratch `MAE_ViT` encoder plus `ViT_Pose_Decoder` |
| Encoder | trainable |
| Q / keypoints / coordinate dimensions | `1 / 14 / 3` |
| Input and patch | `(180,20)` and `(2,2)` |
| Encoder layers/heads | `4/4` |
| MAE decoder layers/heads | `2/4` as preserved from the existing Step-4B configuration |
| Embedding width / input channels | `256 / 3` |
| Target | stored `(1,14,3)` coordinates, order unchanged |
| Optimizer | AdamW |
| Learning rate / weight decay | `0.001 / 0.0` |
| Batch size | `8` |
| Maximum epochs | `10` |
| Workers | `0` |
| Loss | mean Euclidean pose error in stored units |
| Checkpoint | epoch 10 only; no development-checkpoint selection |
| Learning-rate tuning / early stopping / architecture search | none |

## Deterministic execution evidence

The Python process was launched with:

```bash
PYTHONHASHSEED=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The runner validated both values before importing PyTorch. The final
environment record reports:

| Setting | Recorded value |
|---|---|
| Python | `3.11.0` |
| PyTorch | `2.5.1+cu121` |
| CUDA build | `12.1` |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB |
| `torch.are_deterministic_algorithms_enabled()` | `true` |
| deterministic warn-only mode | `false` |
| `torch.backends.cudnn.deterministic` | `true` |
| `torch.backends.cudnn.benchmark` | `false` |
| CUDA matmul TF32 | `false` |
| cuDNN TF32 | `false` |
| float32 matmul precision | `highest` |
| `PYTHONHASHSEED` | `0` |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` |

A separate CUDA deterministic preflight completed before the real run. It
loaded an `(8,3,180,20)` batch with `(8,1,14,3)` targets, completed a finite
forward/backward/AdamW update, observed 97 finite gradient tensors and 97
changed parameters, and verified state-dict save/reload plus finite eval-mode
loss. Its artifact is:

- `data/_artifacts/step4b_r/deterministic_smoke.json`

No deterministic blocker was encountered, and deterministic mode was not
silently disabled.

## Exact commands and observed outputs

### 1. Focused Step-4B-R tests

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_step4b_r_baseline.py -q -p no:cacheprovider
```

Final observed result:

```text
6 passed, 1 warning in 7.34s
```

The warning was the known external `timm.models.layers` deprecation warning.

### 2. Complete fixture-backed suite before the real run

```bash
PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

Observed result:

```text
74 passed, 1 warning in 11.79s
```

There were 0 skips.

### 3. Initial pre-model run attempt

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_r_baseline.py --artifact-root data/_artifacts/step4b_r --device auto > data/_artifacts/step4b_r/run.log 2>&1
```

Observed result: exit code `5` with a pre-model `TypeError` because the new
runner declared but did not assign the imported historical-record verifier.
The error artifact is `data/_artifacts/step4b_r/run_error.json`. No freeze,
dataset, model, optimizer, training, evaluation, checkpoint, or result artifact
from the intended baseline had yet been written in that attempt. The focused
file passed again after assigning the verifier.

### 4. Bounded deterministic baseline

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_r_baseline.py --artifact-root data/_artifacts/step4b_r/baseline --device auto > data/_artifacts/step4b_r/baseline_run.log 2>&1
```

Observed result: exit code `0`. The run wrote:

```text
step4b_r_result=data\_artifacts\step4b_r\baseline\result.json
step4b_r_decision=data\_artifacts\step4b_r\baseline\decision.json
```

The separate `baseline/` root was used solely to preserve the failed pre-model
tooling log while keeping all final artifacts under `data/_artifacts/step4b_r/`.

### 5. Complete fixture-backed suite after final implementation

```bash
PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

Observed result:

```text
74 passed, 1 warning in 11.69s
```

There were 0 skips.

## Learning curves

Every epoch had 256 optimizer batches, finite training losses, and finite
gradients. Train-mode training MPJPE was measured after epoch updates;
development MPJPE was measured afterward in no-gradient eval mode. All values
below are stored units.

| Epoch | Train update MPJPE | Post-epoch train-mode training MPJPE | Development MPJPE | Epoch runtime, seconds |
|---:|---:|---:|---:|---:|
| Initial | — | 4.5129113011062145 | 4.787727847695351 | — |
| 1 | 1.0938618569634855 | 1.0517671528505161 | 0.8322722939774394 | 47.74712200000067 |
| 2 | 1.01327839307487 | 0.9593870609533042 | 0.7336507886648178 | 47.950904600002104 |
| 3 | 1.0199106358923018 | 0.9655520044034347 | 0.6920720627531409 | 48.06008770002518 |
| 4 | 0.9566357919247821 | 0.9616586291231215 | 0.6864092582836747 | 50.97588730001007 |
| 5 | 0.9537556637078524 | 0.9491216223104857 | 0.6982559422031045 | 51.42712939999183 |
| 6 | 0.9501694134669378 | 0.9470359600381926 | 0.7108422992751002 | 51.69859700000961 |
| 7 | 0.9518748765112832 | 0.9392734606517479 | 0.703606627881527 | 51.29919660001178 |
| 8 | 0.9519303693668917 | 0.948007303988561 | 0.6581724304705858 | 48.8603209000139 |
| 9 | 0.9806043859571218 | 1.0972500991774723 | 0.9321989882737398 | 48.46907349998946 |
| 10 | 1.0691768904216588 | 1.0594319407828152 | 0.8333281083032489 | 48.29408009999315 |

The lower epoch-8 development value was not used for checkpoint selection. The
frozen protocol required the epoch-10 final model regardless of the
development curve.

The post-epoch train-mode measurements use the training RNG state at the end
of each epoch. For final checkpoint verification, the original and reloaded
final models were both evaluated from an explicitly restored seed-0 RNG state;
that fixed-state value is `1.0592086446704343` for both. The tiny difference
from the epoch-10 post-epoch curve is therefore measurement-state/dropout-state
difference, not a parameter or checkpoint mismatch: the reloaded state dict
matched exactly.

## Final model and mean-pose comparison

| Predictor | Training MPJPE, stored units | Development MPJPE, stored units |
|---|---:|---:|
| Initial model | 4.5129113011062145 | 4.787727847695351 |
| Final epoch-10 model | 1.0592086446704343 | 0.8333281083032489 |
| Constant mean pose from 2,048 training labels | 0.989848210435728 | 0.8013955073852118 |

Model-minus-mean differences:

| Partition | Final model minus mean pose, stored units | Final model beats mean? |
|---|---:|---|
| Training | `0.06936043423470628` | No |
| Development | `0.03193260091803707` | No |

## Training and checkpoint evidence

| Check | Result |
|---|---|
| Epochs completed | 10 |
| Optimizer steps | 2,560 |
| Training losses | All finite |
| Training gradients | All finite |
| Trainable named parameters | 97 |
| Trainable scalar parameters | 4,795,395 |
| AdamW parameter references | 97 unique, exact trainable-set coverage |
| Trainable parameters changed | 97 of 97 |
| Encoder parameters changed | 50 of 50 |
| Final checkpoint format | `state_dict` |
| Final checkpoint reload | Exact state-dict equality |
| Reloaded train-mode final MPJPE | `1.0592086446704343` |
| Reloaded eval-mode development MPJPE | `0.8333281083032489` |
| Training/evaluation runtime | `521.2547732999956` seconds |
| Full process wall time reported by the background task | `540.664` seconds |

## Generated final artifacts

Under `data/_artifacts/step4b_r/baseline/`:

- `input_verification.json`
- `split_verification.json`
- `step4b_r_freeze.json`
- `environment.json`
- `initial_evaluation.json`
- `parameter_audit.json`
- `learning_curves.json`
- `training_trajectory.json`
- `parameter_changes.json`
- `final_state_dict.pt`
- `development_predictions.npz`
- `mean_pose_comparison.json`
- `result.json`
- `decision.json`

The pre-run CUDA deterministic smoke and preserved pre-model tooling error are
also under `data/_artifacts/step4b_r/`.

## Interpretation and limitations

- The frozen AdamW run produced finite learning curves and a final development
  MPJPE of `0.8333281083032489` stored units.
- The final model was worse than the simple training-mean predictor by
  `0.06936043423470628` stored units on training and `0.03193260091803707`
  stored units on development under this frozen configuration.
- Development MPJPE was nonmonotonic across epochs, but no early stopping,
  learning-rate tuning, architecture search, or checkpoint selection was
  introduced after observing the curve.
- The internal split is candidate filename-group disjoint. It does not establish
  recording-, subject-, or environment-disjointness.
- Stored coordinate units, joint ordering, coordinate frame, and root identity
  remain unproven. This report intentionally makes no millimetre conversion,
  axis interpretation, unit claim, or coordinate-order claim.
- This is one seed and one bounded real-subset baseline. It is not a multi-seed
  scientific result and does not establish generalization or infeasibility.
- The first run attempt failed before model construction because of the new
  runner's unassigned protected-record verifier. It was corrected as a
  pre-model implementation repair and disclosed here; no historical record was
  changed.

## Current status and stop condition

Step-4B-R is complete. Stop here as requested: do not start Step-4C, full
preprocessing/training, multi-seed experiments, automatic hyperparameter
changes, package installation/removal, downloads, deletion, official-test
access, commit/push, or remote changes. Original Step-4B remains **BLOCKED**;
D1/D2 remain above-criterion diagnostics; D3 remains stopped at its
reproduction gate; D4 remains the completed reproducibility audit.
