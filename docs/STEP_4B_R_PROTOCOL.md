# Step-4B-R Frozen Protocol

Date: 2026-11-08

## Purpose

Step-4B-R is a separately named, deterministic, bounded single-person
real-data baseline. Its purpose is to obtain useful learning evidence on the
existing corrected real-subset pilot and to stop the sequence of eight-sample
micro-diagnostics. It is not a rerun or retroactive pass of original Step-4B,
not full training, not a compositional experiment, and not a scientific
feasibility claim.

Acceptance means completing this bounded protocol and reporting the
measurements. No predetermined accuracy threshold must be met.

Original Step-4B remains **BLOCKED** regardless of the Step-4B-R outcome.

## Lineage and protected status

| Record | Status used by this protocol |
|---|---|
| Original Step-4B | **BLOCKED** at the frozen overfit gate |
| Step-4B-D1 | Extended-step diagnostic above criterion |
| Step-4B-D2 | Trainable-encoder diagnostic above criterion |
| Step-4B-D3 | Stopped at its D2 reproduction gate |
| Step-4B-D4 | Reproducibility audit complete; unchanged seed-0 repeats differed from each other |

Step-4B-R does not rewrite, relax, or reverse any historical result. Before
model construction, its runner verifies the protected Step-4B/D1/D2/D3/D4
source and result records and refuses to continue if they changed.

## Data and label access

Only the corrected existing artifacts may be used:

- Freeze: `data/_artifacts/step4b_pilot/freeze_v2/`
- Preprocessed pilot: `data/_artifacts/step4b_pilot/pilot_data_v2/`
- Preprocessing validation: `data/_artifacts/step4b_pilot/preprocessing_validation_v2.json`

Training uses exactly the existing 2,048 labelled single-person
internal-train samples. Development evaluation uses exactly the existing
512 single-person internal-development samples.

Before training, the runner must verify:

- selected train count is 2,048;
- selected development count is 512;
- all selected IDs are filename-marked single-person IDs;
- split assignments match the corrected freeze for every selected ID;
- the intersection of train and development IDs is empty;
- pilot train selected-ID files exactly match `freeze_v2`; and
- corrected preprocessing validation reports 2,560 validated IDs and zero
  failures without official-test use or source-dataset modification.

Development labels are used only for no-gradient development evaluation.
They must never enter a training loss, optimizer update, checkpoint choice,
hyperparameter decision, early-stopping decision, or mean-pose construction.
The constant mean pose must be computed from internal-training labels only.

The runner must not read official-test CSI or keypoint arrays. It must not
load multi-person samples, unlabelled training CSI, compositional training
data, or external/pretrained checkpoints.

## Frozen configuration

The separate frozen source configuration is `step4b_r_config.py`; the runner
copies it into a generated JSON freeze before model construction and refuses
to overwrite that freeze.

Frozen settings:

| Setting | Value |
|---|---|
| Baseline version | `compwifi-step4b-r-v1` |
| Seed | `0` |
| Encoder | trainable, existing from-scratch `MAE_ViT` encoder |
| Decoder | existing `ViT_Pose_Decoder` |
| Model dimensions | Q=1; 14 keypoints; 3 coordinate dimensions; image `(180,20)`; patch `(2,2)`; encoder layers/heads `4/4`; MAE decoder layers/heads `2/2`; embedding width `256`; input channels `3` |
| Target representation | stored `(1,14,3)` coordinates, order unchanged |
| Optimizer | AdamW |
| Learning rate | `0.001` |
| Weight decay | `0.0` |
| Batch size | `8` |
| Maximum epochs | `10` |
| Workers | `0` |
| Loss | mean Euclidean pose error in stored units |
| Checkpoint | final epoch only; no development-checkpoint selection |
| Learning-rate tuning | none |
| Early stopping | none |
| Architecture search | none |
| Additional diagnostic branches | none |

This is deliberately a new baseline configuration rather than a claim that
the original SGD Step-4B/D1/D2/D3/D4 procedure was rerun.

## Deterministic execution requirements

The process environment must set these variables before Python starts and
before CUDA/cuBLAS initialization:

```bash
PYTHONHASHSEED=0
CUBLAS_WORKSPACE_CONFIG=:4096:8
```

The actual training command must set both in the environment of the Python
process. The runner must refuse to proceed if either expected value is absent.

After importing PyTorch but before model construction, training, evaluation,
or CUDA tensor creation, the runner must:

- seed Python `random`, NumPy, PyTorch, and all visible CUDA devices with `0`;
- call `torch.use_deterministic_algorithms(True)` with warn-only mode off;
- set `torch.backends.cudnn.deterministic=True`;
- set `torch.backends.cudnn.benchmark=False`;
- disable CUDA matmul TF32 and cuDNN TF32;
- set float32 matmul precision to `highest`; and
- record Python/PyTorch/CUDA/cuDNN/package versions, GPU identity, all
  relevant deterministic settings, environment variables, and thread counts.

If any operation raises because it lacks a deterministic implementation, the
runner must stop, write a deterministic blocker artifact, and report the
error. It must not silently disable deterministic mode, switch to warn-only
mode, or continue on a nondeterministic path.

## Measurement procedure

For each of at most 10 epochs:

1. Iterate the shuffled 2,048-sample training partition with batch size 8
   and execute the normal train-mode forward, stored-unit mean Euclidean
   loss, backward, finite-gradient check, and AdamW update.
2. After the epoch's updates, measure a train-mode training MPJPE pass in
   stored units with `model.train()` and no optimizer update.
3. Restore model state and RNG state after that train-mode evaluation so
   the extra evaluation does not consume training RNG.
4. Measure a no-gradient eval-mode development MPJPE pass over all 512
   development samples in stored units with `model.eval()`.
5. Record finite losses, finite gradients, optimizer coverage, parameter
   updates, train MPJPE, development MPJPE, and epoch runtime.

At the end:

- save the epoch-10 state dictionary;
- reload it into a fresh model and verify checkpoint round-trip;
- construct one constant mean pose from all 2,048 internal-training labels;
- measure that predictor's training MPJPE and development MPJPE in stored
  units;
- save model and mean-pose predictions for the development partition;
- record total runtime, environment, per-epoch curves, final model results,
  mean-pose results, and whether the final model beats the mean predictor
  separately on training and development data; and
- assign a terminal record that completion occurred without silently
  changing any frozen setting.

All distances are reported in stored units. This protocol does not convert
results to millimetres and does not make axis, unit, joint-ordering, or
coordinate-frame claims.

## Required artifacts

All run artifacts must be written under `data/_artifacts/step4b_r/`,
including at least:

- `input_verification.json`
- `step4b_r_freeze.json`
- `environment.json`
- `split_verification.json`
- `parameter_audit.json`
- `learning_curves.json`
- `training_trajectory.json`
- `parameter_changes.json`
- `final_state_dict.pt`
- `development_predictions.npz`
- `mean_pose_comparison.json`
- `result.json`
- `decision.json`

No historical Step-4B, D1, D2, D3, or D4 artifact may be overwritten.

## Planned commands

Focused tests:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_step4b_r_baseline.py -q -p no:cacheprovider
```

Fixture-backed suite before training:

```bash
PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

Bounded deterministic training run:

```bash
PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_r_baseline.py --artifact-root data/_artifacts/step4b_r --device auto
```

## Stop rules

Stop and report rather than changing the protocol when:

- a required corrected artifact is missing or invalid;
- split verification fails;
- historical protected records changed;
- the required deterministic environment variables are absent;
- deterministic algorithms cannot be enabled or an operation lacks
  deterministic support;
- a loss, gradient, prediction, label, or metric is non-finite; or
- the final state-dict checkpoint cannot be reloaded and verified.

Stop after the Step-4B-R report. Do not start Step-4C, full preprocessing or
full training, multi-seed experiments, hyperparameter changes, package
installation/removal, downloads, deletion, official-test access, commit/push,
or remote changes.
