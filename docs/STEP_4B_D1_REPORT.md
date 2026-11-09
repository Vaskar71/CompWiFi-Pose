# Step-4B-D1 Extended-Step Under-Training Diagnostic Report

Date: 2026-11-07

## Verdict

**EXTENDED_STEP_DIAGNOSTIC_ABOVE_CRITERION — the original Step-4B result remains BLOCKED.**

The frozen 4x optimizer-step diagnostic completed 800 finite SGD steps with finite gradients. Final train-mode MPJPE was `0.32848620414733887` stored units, which remains above the original absolute criterion of `<=0.10` stored units. The diagnostic therefore **does not** retroactively pass Step-4B and makes no generalization claim.

Original Step-4B remains:

```json
{
  "status": "BLOCKED",
  "reason": "The frozen numerical overfit criterion failed; Step-4B training is blocked."
}
```

The original `data/_artifacts/step4b_pilot/decision.json`, threshold, report, and overfit result were not rewritten.

## Scope and label access

- Only corrected `freeze_v2` / `pilot_data_v2` internal-training pilot artifacts were used for training.
- The fixed ordered batch was loaded from `selected_train_ids.txt`, whose full 2,048-ID content exactly matches `freeze_v2`.
- CSI and keypoint arrays were paired by exact sample ID for all eight tiny-batch samples.
- Development samples were not loaded by the D1 runner.
- Official test samples and arrays were not read or evaluated.
- No multi-person, unlabelled, compositional, epoch, or development-evaluation run was started.
- No packages were installed or removed; no files were downloaded or deleted; no commit, push, or remote operation was performed.

## Frozen source configuration verified before the D1 run

| Setting | Frozen value |
|---|---|
| Algorithm version | `compwifi-step4b-v1` |
| Diagnostic version | `compwifi-step4b-d1-v1` |
| Seed | `0` |
| Ordered tiny-batch IDs | `S11_06_10`, `S11_06_11`, `S11_06_13`, `S11_06_14`, `S11_06_16`, `S11_06_17`, `S11_06_19`, `S11_06_20` |
| Batch size | `8` |
| Model family | from-scratch `MAE_ViT` encoder plus `ViT_Pose_Decoder` |
| Model settings | `Q=1`; 14 keypoints; 3 coordinate dimensions; image `(180,20)`; patch `(2,2)`; encoder layers/heads `4/4`; MAE decoder layers/heads `2/2`; embedding width `256`; input channels `3` |
| Parameter count | `4,795,395` total; `1,404,931` trainable |
| Loss | mean Euclidean distance over people, joints, and coordinate triples |
| Optimizer | SGD, learning rate `0.01`, momentum `0.9` |
| Original optimizer budget | `200` steps |
| D1 optimizer budget | exactly `4 * 200 = 800` steps |
| Absolute criterion | final MPJPE `<=0.10` stored units |
| Coordinate handling | stored coordinate order unchanged; axes are identified only by zero-based coordinate index |
| Original initial loss/MPJPE | `4.920130252838135` stored units |
| Original final loss/MPJPE | `0.34014594554901123` stored units |
| Original final/initial ratio | `0.06913352453480291` |
| Original failure | `final MPJPE 0.340145946 exceeds 0.100000000 stored units` |

Input manifest hashes verified from the freeze:

| Manifest | IDs | SHA-256 |
|---|---:|---|
| internal_train | 70,197 | `3628030d40df460829cea988e6e88c4966ae6163288f307d3720c08b7b587d15` |
| internal_development | 19,749 | `4a3dfb699e85cdf1a6782ce779ce9d401f2db6737ce96353ca0f7d706708113d` |

Pilot preprocessing validation was confirmed as 2,048 internal-train and 512 internal-development IDs, 2,560 validated, zero failures, official test unused, and source data unmodified. The D1 run itself loaded only the internal-training dataset.

Exact tiny-batch pairing evidence is saved in `data/_artifacts/step4b_d1/input_verification.json`. Each pair has CSI shape `(3,180,20)`, keypoint shape `(1,14,3)`, finite values, and the same exact sample ID in the CSI path, keypoint path, loader order, and original Step-4B overfit artifact.

## Frozen D1 change

The diagnostic freeze is `data/_artifacts/step4b_d1/step4b_d1_freeze.json`. It was written before model construction and records the source artifact SHA-256 values. The only experimental change from Step-4B was:

- optimizer-step budget: `200 -> 800`.

The following were explicitly unchanged: seed, ordered batch, data, preprocessing, model and dimensions, random initialization procedure, loss, SGD optimizer, learning rate, momentum, batch size, device-selection mode, coordinate order, and the original `<=0.10` absolute criterion.

## Exact commands and observed outputs

### 1. Focused D1 unit tests

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_step4b_d1_diagnostic.py -q -p no:cacheprovider
```

Observed result after the final plot-label update:

```text
4 passed, 1 warning in 5.75s
```

The warning was the known external `timm.models.layers` deprecation warning.

### 2. Complete focused suite

```bash
PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

Observed final result after the D1 implementation and report:

```text
52 passed, 1 warning in 9.47s
```

There were **0 skips**. The warning was the same external `timm.models.layers` deprecation warning.

### 3. Pre-run frozen-input verification

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_d1_diagnostic.py --verify-only --artifact-root data/_artifacts/step4b_d1 --device auto
```

Observed result: exit code `0`; the verifier printed the frozen seed, batch IDs, exact pairings, model/optimizer settings, original metrics, original failure, unchanged `BLOCKED` decision, and:

```text
input_verification=data\_artifacts\step4b_d1\input_verification.json
```

### 4. Frozen 800-step diagnostic

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_d1_diagnostic.py --artifact-root data/_artifacts/step4b_d1 --device auto
```

Observed summary:

```text
diagnostic_run_status=COMPLETE
diagnostic_status=EXTENDED_STEP_DIAGNOSTIC_ABOVE_CRITERION
device=cuda
runtime_seconds=36.858981000012136
all_losses_finite=true
all_gradients_finite=true
criterion_passed=false
initial_mpjpe_stored_units=4.920130252838135
original_stop_mpjpe_stored_units=0.34014594554901123
final_mpjpe_stored_units=0.32848620414733887
final_minus_original_stop_mpjpe_stored_units=-0.011659741401672363
final_to_initial_mpjpe_ratio=0.066763721134792
```

The run printed:

```text
d1_result=data\_artifacts\step4b_d1\step4b_d1_result.json
d1_decision=data\_artifacts\step4b_d1\step4b_d1_decision.json
```

The D1 decision is:

```json
{
  "status": "EXTENDED_STEP_DIAGNOSTIC_ABOVE_CRITERION",
  "final_mpjpe_stored_units": 0.32848620414733887,
  "original_absolute_criterion_stored_units": 0.1,
  "original_step4b_status": "BLOCKED",
  "original_result_preserved": true
}
```

## Trajectory evidence

MPJPE equals the training loss in this frozen implementation. The full 800-step pre-update loss/MPJPE and gradient-finite records are in `data/_artifacts/step4b_d1/loss_trajectory.json`.

Every 25 recorded optimizer steps:

| Step | Pre-update loss = MPJPE, stored units |
|---:|---:|
| 25 | 0.6646754741668701 |
| 50 | 0.42089027166366577 |
| 75 | 0.36026182770729065 |
| 100 | 0.3424915075302124 |
| 125 | 0.35564225912094116 |
| 150 | 0.35861560702323914 |
| 175 | 0.34348565340042114 |
| 200 | 0.3460858464241028 |
| 225 | 0.3447013795375824 |
| 250 | 0.343589186668396 |
| 275 | 0.33407261967658997 |
| 300 | 0.3468073010444641 |
| 325 | 0.3536248505115509 |
| 350 | 0.3465113341808319 |
| 375 | 0.34564483165740967 |
| 400 | 0.3412717878818512 |
| 425 | 0.333574116230011 |
| 450 | 0.35543355345726013 |
| 475 | 0.34177955985069275 |
| 500 | 0.33866018056869507 |
| 525 | 0.3409993350505829 |
| 550 | 0.33230671286582947 |
| 575 | 0.33420857787132263 |
| 600 | 0.33363085985183716 |
| 625 | 0.33273425698280334 |
| 650 | 0.33533161878585815 |
| 675 | 0.3332032263278961 |
| 700 | 0.34047386050224304 |
| 725 | 0.3291252553462982 |
| 750 | 0.3527730107307434 |
| 775 | 0.33258056640625 |
| 800 | 0.3282155990600586 |

The step-200 value above is the pre-update loss at the 200th backward. The post-update original stopping checkpoint, matching the original Step-4B final value exactly, was `0.34014594554901123` stored units.

Trend summary:

| Evidence | Value |
|---|---:|
| First recorded pre-update loss | 4.924509525299072 |
| Minimum pre-update loss | 0.3212946951389313 |
| Pre-update minimum step | 793 |
| Final recorded pre-update loss | 0.3282155990600586 |
| Mean steps 200–400 | 0.3430550715223474 |
| Mean steps 401–600 | 0.3384132421016693 |
| Mean steps 601–800 | 0.3348188555240631 |
| Adjacent previous-100 mean in saved summary | 0.33421005934476855 |
| Adjacent last-100 mean in saved summary | 0.3354276517033577 |
| Adjacent last-100 change in saved summary | +0.001217592358589148 |
| Linear slope over all post-original pre-update records | -1.9627956772565248e-05 per step |
| Linear slope over steps 600–800 | -1.0860303189210098e-06 per step |

The broad segment means decreased slowly, but short windows are nonmonotonic and near flat. The post-update endpoint improved by only `0.011659741401672363` stored units from the original 200-step endpoint. This is best characterized as a plateau around `0.33–0.34`, not continued strong convergence.

## Original-stop and final detailed measurements

The complete machine-readable checkpoint records are:

- `data/_artifacts/step4b_d1/checkpoint_step_0200.json`
- `data/_artifacts/step4b_d1/checkpoint_step_0800.json`

Coordinate indices below are storage indices only; no axis name or coordinate convention is declared official.

### Per-coordinate-index error, stored units

| Coordinate index | Step-200 MAE | Step-800 MAE | Step-200 RMSE | Step-800 RMSE | Step-200 max abs | Step-800 max abs |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.12922437489032745 | 0.12585235280650003 | 0.1552648706442361 | 0.15277737164557206 | 0.2910330295562744 | 0.3173537254333496 |
| 1 | 0.038868914757456095 | 0.041200916681970866 | 0.049701737085480625 | 0.0567951777242466 | 0.13115191459655762 | 0.15546584129333496 |
| 2 | 0.2888043203524181 | 0.27095783182552885 | 0.3295903714941573 | 0.31950822636746223 | 0.7144074440002441 | 0.5464539527893066 |

Coordinate index 2 has the largest MAE/RMSE at both checkpoints. Its MAE improved by `0.01784648852688925`, while coordinate index 1 worsened by `0.002332001924514771`.

### Per-joint MPJPE, stored units

Joint indices are zero-based storage indices; joint ordering remains unverified.

| Joint index | Step 200 | Step 800 | Final minus original stop |
|---:|---:|---:|---:|
| 0 | 0.19738767713358665 | 0.3127400567822519 | +0.11535238 |
| 1 | 0.6775408229610733 | 0.4560138685372646 | -0.221526954 |
| 2 | 0.2025926841893294 | 0.2703915245034126 | +0.06779884 |
| 3 | 0.22178170225432975 | 0.2872127936886292 | +0.065431091 |
| 4 | 0.22017941478487324 | 0.2248631161348384 | +0.004683701 |
| 5 | 0.31743009533674943 | 0.20118371403746454 | -0.116246381 |
| 6 | 0.3131167904191763 | 0.24307904708644756 | -0.070037743 |
| 7 | 0.3209104925476894 | 0.22008032992264226 | -0.100830163 |
| 8 | 0.4473697356718127 | 0.5598210573336049 | +0.112451322 |
| 9 | 0.39803502599205504 | 0.5189711150063675 | +0.120936089 |
| 10 | 0.48067001374050455 | 0.4498886688193836 | -0.030781345 |
| 11 | 0.4580052674971869 | 0.5233436143430489 | +0.065338347 |
| 12 | 0.2216006573296742 | 0.1541079099915004 | -0.067492747 |
| 13 | 0.28542258745399435 | 0.17711006743295377 | -0.10831252 |

At step 800, error concentrates most at joint indices 8 (`0.5598210573`), 11 (`0.5233436143`), 9 (`0.5189711150`), 1 (`0.4560138685`), and 10 (`0.4498886688`). Joint 1 improved substantially, while joints 8 and 9 worsened relative to step 200.

### Per-sample MPJPE, stored units

| Sample ID | Step 200 | Step 800 | Final minus original stop |
|---|---:|---:|---:|
| S11_06_10 | 0.34627953701602554 | 0.3250211843897693 | -0.021258353 |
| S11_06_11 | 0.33117702444357355 | 0.3383076548141167 | +0.00713063 |
| S11_06_13 | 0.3404385585337027 | 0.337184259398251 | -0.003254299 |
| S11_06_14 | 0.3448414572382296 | 0.3262540782784053 | -0.018587379 |
| S11_06_16 | 0.35293101640095836 | 0.3360111243543215 | -0.016919892 |
| S11_06_17 | 0.3318764074515788 | 0.3099975113443832 | -0.021878896 |
| S11_06_19 | 0.32324605259386857 | 0.3448515346069672 | +0.021605482 |
| S11_06_20 | 0.3503773562146542 | 0.3102623005965346 | -0.040115056 |

Per-sample errors remain tightly clustered, from `0.3099975113` to `0.3448515346` at step 800. Six samples improved, while `S11_06_11` and `S11_06_19` worsened. No single sample dominates the remaining error.

### Coordinate ranges and variance

Ground-truth target arrays are unchanged across checkpoints.

| Array | Coordinate index | Step | Min | Max | Variance |
|---|---:|---:|---:|---:|---:|
| Prediction | 0 | 200 | 2.2490930557250977 | 2.935269832611084 | 0.0717255702978683 |
| Prediction | 1 | 200 | 2.248540163040161 | 3.329423427581787 | 0.1775499690715334 |
| Prediction | 2 | 200 | 2.2390801906585693 | 2.99222731590271 | 0.07974656762344484 |
| Prediction | 0 | 800 | 2.2595055103302 | 3.027073383331299 | 0.09069878011038508 |
| Prediction | 1 | 800 | 2.2329957485198975 | 3.096296787261963 | 0.11425612847581078 |
| Prediction | 2 | 800 | 2.260741949081421 | 3.0191471576690674 | 0.08932533110898543 |
| Target | 0 | both | 2.263493061065674 | 2.7313339710235596 | 0.04163014604614861 |
| Target | 1 | both | 2.258230686187744 | 2.7353720664978027 | 0.037267213713675924 |
| Target | 2 | both | 2.2236804962158203 | 2.8812742233276367 | 0.07965962669662909 |

Overall variance was `0.1903774831686976` for step-200 predictions, `0.19485459940408414` for step-800 predictions, and `0.22225718120111973` for targets.

### Mean-pose / collapsed-output comparison

| Evidence, stored units | Step 0 | Step 200 | Step 800 |
|---|---:|---:|---:|
| Fixed-batch mean-pose baseline MPJPE | 0.008724883778774263 | 0.008724883778774263 | 0.008724883778774263 |
| Model-to-fixed-batch-mean MPJPE | 4.920122517342274 | 0.34006716428171035 | 0.3282895416366456 |
| Frozen-train mean-pose baseline MPJPE | 0.804225435680532 | 0.804225435680532 | 0.804225435680532 |
| Model-to-frozen-train-mean MPJPE | 4.393603953077288 | 0.9221599618071887 | 0.9138911478026657 |
| Target distance to its own cross-sample mean | 0.008724883778774261 | 0.008724883778774261 | 0.008724883778774261 |
| Prediction distance to its own cross-sample mean | 0.08672763432073134 | 0.04737772180790182 | 0.0392643593134629 |

Prediction-to-target cross-sample variance ratios:

| Coordinate index | Step 0 | Step 200 | Step 800 |
|---:|---:|---:|---:|
| 0 | 111.17840542128093 | 6.861937279003772 | 4.898329671050698 |
| 1 | 34.74982062899039 | 2.5007898943568017 | 1.9846071511324965 |
| 2 | 270.6585960676046 | 246.5608252275827 | 272.40411502662204 |

The predictions did **not** collapse toward the fixed-batch mean or the frozen-train mean. At step 800 they remain much farther from the fixed-batch mean (`0.3282895416`) than the mean pose itself (`0.0087248838`), and their cross-sample dispersion remains above the target dispersion for every coordinate index.

## Prediction-versus-ground-truth plots

All eight fixed tiny-batch samples have labelled plots comparing target, initial, step-200, and step-800 predictions by coordinate index:

- `data/_artifacts/step4b_d1/plots/S11_06_10_prediction_vs_ground_truth.png`
- `data/_artifacts/step4b_d1/plots/S11_06_11_prediction_vs_ground_truth.png`
- `data/_artifacts/step4b_d1/plots/S11_06_13_prediction_vs_ground_truth.png`
- `data/_artifacts/step4b_d1/plots/S11_06_14_prediction_vs_ground_truth.png`
- `data/_artifacts/step4b_d1/plots/S11_06_16_prediction_vs_ground_truth.png`
- `data/_artifacts/step4b_d1/plots/S11_06_17_prediction_vs_ground_truth.png`
- `data/_artifacts/step4b_d1/plots/S11_06_19_prediction_vs_ground_truth.png`
- `data/_artifacts/step4b_d1/plots/S11_06_20_prediction_vs_ground_truth.png`

Additional summary plots:

- `data/_artifacts/step4b_d1/plots/mean_pose_comparison.png`
- `data/_artifacts/step4b_d1/plots/per_sample_mpjpe.png`

The prediction arrays used for the plots and checkpoint statistics are saved in `data/_artifacts/step4b_d1/checkpoint_predictions.npz`.

## Engineering evidence

- One fixed batch loaded with expected `(8,3,180,20)` CSI and `(8,1,14,3)` pose shapes.
- Forward and backward passes succeeded for all 800 steps.
- Every loss was finite.
- Every trainable gradient tensor was finite.
- SGD updated parameters throughout.
- Final `state_dict` checkpoint saved to `data/_artifacts/step4b_d1/step4b_d1_final_state_dict.pt`.
- Checkpoint reload evaluation matched the saved model; eval-mode reloaded MPJPE was `0.33505353331565857` stored units.
- The step-200 D1 post-update MPJPE exactly matched the original Step-4B final MPJPE: `0.34014594554901123` stored units. The initial D1 MPJPE also exactly matched the original initial value: `4.920130252838135` stored units.

## Diagnostic interpretation only

Because final MPJPE `0.32848620414733887` is greater than `0.10`, the required interpretation is:

- the extended-step diagnostic remained above the original criterion;
- broad trajectory means improved slightly from `0.3430550715` to `0.3348188555`, but the endpoint improved only `0.0116597414` from the original 200-step endpoint and short-window trends were near flat/nonmonotonic;
- the global minimum pre-update loss was `0.3212946951389313`, still far above `0.10`;
- predictions did not collapse toward the mean; they retained greater cross-sample variance than the targets;
- residual error was concentrated primarily in coordinate index 2 and joint indices 8, 11, 9, 1, and 10;
- the fourfold optimizer budget alone did not explain the frozen gate failure as merely insufficient steps. This diagnostic does not identify the root cause and does not justify an architecture, normalization, optimizer, or data change within Step-4B-D1.

No scientific feasibility, accuracy, novelty, generalization, or state-of-the-art claim is made.

## Files added or changed for Step-4B-D1

Source and tests:

- `step4b_d1_diagnostic.py`
- `tests/test_step4b_d1_diagnostic.py`

Documentation:

- `docs/STEP_4B_D1_REPORT.md`
- `docs/CURRENT_TASK.md`

Generated ignored D1 artifacts:

- `data/_artifacts/step4b_d1/input_verification.json`
- `data/_artifacts/step4b_d1/step4b_d1_freeze.json`
- `data/_artifacts/step4b_d1/loss_trajectory.json`
- `data/_artifacts/step4b_d1/checkpoint_step_0000.json`
- `data/_artifacts/step4b_d1/checkpoint_step_0200.json`
- `data/_artifacts/step4b_d1/checkpoint_step_0800.json`
- `data/_artifacts/step4b_d1/checkpoint_predictions.npz`
- `data/_artifacts/step4b_d1/step4b_d1_result.json`
- `data/_artifacts/step4b_d1/step4b_d1_decision.json`
- `data/_artifacts/step4b_d1/step4b_d1_final_state_dict.pt`
- ten PNG plots under `data/_artifacts/step4b_d1/plots/`

## Recommended next action

Open one separately approved D2 engineering-diagnosis task to investigate the fixed model's optimization/capacity or target-representation mismatch using a new pre-registered diagnostic protocol. Do not begin Step-4C composition controls and do not treat this D1 result as a Step-4B pass.
