# Step-4B Single-Person Sanity Gate Report

Date: 2026-11-07

## Verdict

**BLOCKED — do not proceed to Step-4C composition controls.**

The corrected frozen subset was selected, preprocessed, validated, loaded, and used in the frozen tiny-overfit gate. All values were finite and the fixed batch loss improved from `4.920130252838135` to `0.34014594554901123` stored units, but the frozen absolute overfit criterion required final MPJPE at most `0.10` stored units. The observed final MPJPE was `0.34014594554901123`, so the gate failed before the bounded 10-epoch run.

Per the frozen stop rule, no epoch training, development evaluation, mean-pose comparison, compositional training, or official test evaluation was run after this failure. No hyperparameter, architecture, criterion, or data check was changed after seeing the outcome.

This report makes no claim of scientific feasibility, accuracy, novelty, generalization, or state of the art.

## Label-access and scope statement

- Training labels were restricted to the frozen internal-train one-person pilot IDs.
- Evaluation/model-selection labels were restricted to the frozen internal-development one-person pilot IDs.
- No official-test list or array was read.
- No multi-person, unlabelled, or compositional training was run.
- The fixed model used `Q=1`; all selected development labels were verified as exactly one person.
- Count metrics for this run would be structural for a single always-active slot and are not evidence of multi-person counting.

One protocol-sensitive correction occurred before training: an initial candidate freeze used an overly broad filename test and selected some `S12_*`/`S13_*` IDs. The bounded preprocessing shape checks rejected those IDs because their keypoint person dimensions were not one. That pass did not use their pose coordinates for training or evaluation. The rule was corrected to the Person-in-WiFi-3D filename count character (`S?1_*` = one person), and a completely new freeze was generated and used. The rejected freeze and partial pilot root remain ignored artifacts and were not deleted.

## Files added for Step-4B

Source and tests:

- `step4b_config.py`
- `step4b_subset.py`
- `make_step4b_pilot_data.py`
- `feeder/piw3d_single_person.py`
- `step4b_single_person_pilot.py`
- `tests/test_step4b_subset.py`
- `tests/test_step4b_pilot_data.py`
- `tests/test_piw3d_single_person_loader.py`
- `tests/test_step4b_pilot.py`

Documentation:

- `docs/STEP_4B_REPORT.md`
- `docs/CURRENT_TASK.md`

Generated ignored artifacts:

- `data/_artifacts/step4b_goal.txt`
- `data/_artifacts/step4b_pilot/freeze_v2/`
- `data/_artifacts/step4b_pilot/pilot_data_v2/`
- `data/_artifacts/step4b_pilot/preprocessing_validation_v2.json`
- `data/_artifacts/step4b_pilot/overfit_result.json`
- `data/_artifacts/step4b_pilot/decision.json`

Superseded ignored artifacts retained without deletion:

- `data/_artifacts/step4b_pilot/freeze/`
- `data/_artifacts/step4b_pilot/pilot_data/`
- `data/_artifacts/step4b_pilot/preprocessing_validation.json`

## Frozen configuration

The frozen algorithm version is `compwifi-step4b-v1` with seed `0`.

Subset caps and rule:

- 8 internal-train groups and 4 internal-development groups.
- At most 256 frames per selected train group and 128 per selected development group.
- Eligible groups were those containing at least one filename-marked one-person ID (`S?1_*`).
- Eligible group names were sorted lexicographically, independently shuffled per partition with `random.Random(0)`, and the first 8 train / 4 development groups were selected.
- Within each selected group, one-person IDs were sorted by numeric frame and sampled at zero-based `linspace` positions.
- These caps were fixed resource limits, not tuned scientific parameters.

Input internal manifests:

| Partition | IDs | SHA-256 |
|---|---:|---|
| internal_train | 70,197 | `3628030d40df460829cea988e6e88c4966ae6163288f307d3720c08b7b587d15` |
| internal_development | 19,749 | `4a3dfb699e85cdf1a6782ce779ce9d401f2db6737ce96353ca0f7d706708113d` |

Selected groups and frame counts:

| Partition | Group | Selected frames |
|---|---|---:|
| internal_train | `S11_06` | 256 |
| internal_train | `S11_09` | 256 |
| internal_train | `S11_18` | 256 |
| internal_train | `S21_15` | 256 |
| internal_train | `S21_28` | 256 |
| internal_train | `S51_10` | 256 |
| internal_train | `S51_11` | 256 |
| internal_train | `S51_12` | 256 |
| internal_development | `S11_16` | 128 |
| internal_development | `S11_19` | 128 |
| internal_development | `S51_01` | 128 |
| internal_development | `S51_02` | 128 |

Totals:

- Internal train: 8 groups / 2,048 IDs.
- Internal development: 4 groups / 512 IDs.
- Total before eligibility checks: 2,560 IDs.
- No groups were moved between partitions.

Frozen model/training settings:

- Separate entry point: `step4b_single_person_pilot.py`.
- Model family: from-scratch `MAE_ViT` encoder plus `ViT_Pose_Decoder`.
- `Q=1`, 14 keypoints, 3 coordinate dimensions.
- Image size `(180,20)`, patch `(2,2)`, encoder layers/heads `4/4`, MAE decoder layers/heads `2/2`, embedding width `256`, input channels `3`.
- Existing component behavior freezes the randomly initialized encoder parameters for pose-decoder training; no pretrained or external checkpoint was loaded.
- SGD learning rate `0.01`, momentum `0.9`.
- Frozen post-gate budget: one seed, at most 10 epochs, train batch 16, development batch 16, `num_workers=0`.
- Checkpoints are `state_dict` only.

Frozen overfit criterion:

- One fixed ordered internal-training batch of 8 samples.
- 200 optimizer steps.
- Every loss finite.
- Final loss less than initial loss.
- Final MPJPE at most `0.10` stored units.
- Final/initial MPJPE ratio at most `0.30`.

Research metric settings frozen for the eventual evaluation:

- Stored-coordinate to millimetres scale: exactly `1000.0`.
- Presence threshold: `0.5`; the single `Q=1` slot used presence score `1.0` in the intended evaluation path.
- Hungarian one-to-one matching with match threshold `500 mm`.
- PCK@`0.5`, normalized by the ground-truth distance between joints `6` and `4` in stored units.
- Near-zero PCK scale guard: `1e-6` stored units.
- Root index: `0` (provisional legacy convention).
- Coordinate order was not changed.

## Exact commands and observed outputs

### 1. Corrected subset freeze

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_subset.py --train-manifest data/_artifacts/piw3d_candidate_group_manifests/internal_train_ids.txt --development-manifest data/_artifacts/piw3d_candidate_group_manifests/internal_development_ids.txt --output-dir data/_artifacts/step4b_pilot/freeze_v2 --seed 0
```

Observed output:

```text
{
  "internal_development_frames_per_group_cap": 128,
  "internal_development_groups_requested": 4,
  "internal_development_groups_selected": 4,
  "internal_development_ids_selected": 512,
  "internal_train_frames_per_group_cap": 256,
  "internal_train_groups_requested": 8,
  "internal_train_groups_selected": 8,
  "internal_train_ids_selected": 2048,
  "total_ids_selected_before_eligibility_checks": 2560
}
freeze_config=data\_artifacts\step4b_pilot\freeze_v2\step4b_freeze_config.json
```

### 2. Bounded preprocessing and validation

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe make_step4b_pilot_data.py --source-root data/person_in_wifi_3d/wifi_dataset --freeze-config data/_artifacts/step4b_pilot/freeze_v2/step4b_freeze_config.json --pilot-root data/_artifacts/step4b_pilot/pilot_data_v2 --validation-output data/_artifacts/step4b_pilot/preprocessing_validation_v2.json
```

Observed final output:

```text
{
  "failed_ids": 0,
  "selected_development_ids": 512,
  "selected_train_ids": 2048,
  "total_selected_ids": 2560,
  "validated_ids": 2560
}
validation_report=data\_artifacts\step4b_pilot\preprocessing_validation_v2.json
```

Validation checked every selected sample for:

- exact amplitude shape `(3,90,20)`;
- exact phase shape `(3,90,20)`;
- exact combined `csi_ap` shape `(3,180,20)`;
- finite values in all processed arrays;
- exact `csi_ap == concatenate(amplitude, phase)`;
- exact keypoint shape `(1,14,3)`;
- finite keypoint values; and
- the frozen internal-train/internal-development assignment.

The source dataset was not modified. The existing bounded `process_ids` path was used; full preprocessing `main()` was not run.

### 3. Frozen tiny-overfit gate

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_single_person_pilot.py overfit --artifact-root data/_artifacts/step4b_pilot --device auto
```

Observed result, exit code `2`:

```text
{
  "batch_size": 8,
  "device": "cuda",
  "steps": 200,
  "initial_loss": 4.920130252838135,
  "final_loss": 0.34014594554901123,
  "initial_mpjpe_stored_units": 4.920130252838135,
  "final_mpjpe_stored_units": 0.34014594554901123,
  "final_to_initial_mpjpe_ratio": 0.06913352453480291,
  "loss_trajectory": {
    "first": 4.920130252838135,
    "last": 0.3460858464241028,
    "min": 0.333870530128479,
    "max": 4.924509525299072
  },
  "model_parameters": {
    "total_parameters": 4795395,
    "trainable_parameters": 1404931
  },
  "status": "FAIL",
  "failures": [
    "final MPJPE 0.340145946 exceeds 0.100000000 stored units"
  ]
}
```

The fixed batch sample IDs were:

```text
S11_06_10, S11_06_11, S11_06_13, S11_06_14,
S11_06_16, S11_06_17, S11_06_19, S11_06_20
```

The generated decision artifact says:

```json
{
  "status": "BLOCKED",
  "reason": "The frozen numerical overfit criterion failed; Step-4B training is blocked."
}
```

### 4. Complete focused test suite

```bash
PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

Observed result:

```text
48 passed, 1 warning in 9.67s
```

There were **0 skips**. The one warning was the external `timm.models.layers` deprecation warning also observed in earlier steps.

## Learning evidence and diagnosis

The overfit gate produced finite losses throughout, and final loss was much lower than initial loss:

- Initial loss / MPJPE: `4.920130252838135` stored units.
- Final loss / MPJPE: `0.34014594554901123` stored units.
- Final/initial ratio: `0.06913352453480291`.
- Minimum observed loss: `0.333870530128479` stored units.
- Device: CUDA.
- Trainable parameters: `1,404,931` of `4,795,395` total.

The failure is therefore not explained by unreadable data, invalid split assignment, incorrect keypoint shape, nonfinite arrays, missing optimizer updates, or nonfinite training values in the checked batch. The corrected pilot passed all preprocessing eligibility checks, and the overfit run loaded the intended internal-train batch with the expected `(3,180,20)` CSI and `(1,14,3)` label shapes.

Under the fixed model, fixed optimizer, and frozen absolute threshold, the model reduced the batch error substantially but did not reach `0.10` stored units. Because changing the criterion, optimizer, model, or data checks after seeing this result would violate the frozen gate, the correct terminal state is **BLOCKED**, not a post-hoc tuning attempt.

## Development evaluation and mean-pose comparison

Not run. The overfit gate failed, so the bounded epoch run and evaluation were prohibited.

| Required output | Result |
|---|---|
| Development matched MPJPE | NOT RUN — blocked before epoch training |
| Development PA-MPJPE | NOT RUN — blocked before epoch training |
| Guarded PCK | NOT RUN — blocked before epoch training |
| Root-localization error | NOT RUN — blocked before epoch training |
| Exact person-count accuracy | NOT RUN — blocked before epoch training |
| Missing/spurious counts | NOT RUN — blocked before epoch training |
| Number of evaluated people | 0 model-evaluation people; 512 eligible internal-development labels were preprocessed but not used for model selection after the block |
| Frozen-train mean-pose MPJPE | NOT RUN — mean-pose comparison was not started |
| Mean-pose comparison | NOT PERFORMED |

## Test evidence

Step-4B unit tests added 15 tests covering:

- deterministic group/frame selection and fixed seed behavior;
- corrected filename one-person rule;
- split-assignment enforcement;
- labelled loader batch keys and shapes;
- rejection of wrong split, multi-person filename marker, wrong keypoint shape, nonfinite labels, unknown IDs, and missing pilot files;
- fixed Step-4B configuration serialization;
- Q=1 metric integration and millimetre conversion; and
- bounded pilot preprocessing validation/failure reporting.

The complete focused suite using the existing two-frame fixture passed with **48 tests, 0 skips**.

## Unresolved assumptions and uncertainties

- Stored coordinates are treated as metres and multiplied by exactly `1000` for research metrics based on bounded upstream evidence, but the released dataset metadata still does not formally declare units.
- Joint ordering, coordinate frame/origin, root identity, and the upstream axis-1-vertical versus local axis-2-vertical conflict remain unproven.
- Root index `0` and PCK scale joints `(6,4)` remain provisional, versioned settings.
- Candidate filename groups remain filename rules only; no recording-, subject-, or environment-disjointness is claimed.
- One-person selection used the filename count marker required by the task; no semantic person metadata is available.
- The fixed random-encoder/decoder configuration demonstrated finite learning but did not satisfy the frozen absolute overfit threshold. No claim is made about whether a different approved protocol would pass.
- The result is a single-seed engineering sanity gate only. It is not a scientific feasibility result.

## Boundaries honored

Not performed:

- Package installation or removal.
- Dataset download or deletion.
- Source dataset modification.
- Full preprocessing `main()`.
- Preprocessing beyond the corrected frozen subset.
- Multi-person, unlabelled, or compositional training.
- Architecture or hyperparameter search.
- Official-test evaluation.
- Credential access.
- Commit, push, or remote changes.

## Current status and recommended next task

Step-4B is **BLOCKED** at the frozen overfit gate. Do not begin Step-4C from this result.

A later task may either:

1. revise and re-freeze the overfit criterion/protocol before seeing new outcomes, then run a new bounded gate; or
2. perform a separately approved engineering diagnosis of the fixed configuration without tuning against development results.

Any such task must preserve label-access rules and must not treat this blocked run as evidence of scientific feasibility or infeasibility.
