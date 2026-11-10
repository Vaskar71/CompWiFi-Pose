# Current Task and Project Status

## Status at a glance (updated 2026-11-10, end of Step 5b)

- **Phase:** Step 5b (`docs/STEP_5B_PLAN.md`, inserted between Phase B and Phase C) is complete with
  verdict **INCONCLUSIVE** (`docs/STEP_5B_REPORT.md`). Phase C ("Step 6") is **not** started and needs
  the user's decision. Binding rules: `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/RESEARCH_PROTOCOL.md`,
  `docs/NEXT_PHASE_PLAN.md`, `docs/STEP_5B_PLAN.md`, and the frozen `docs/STEP_5B_PROTOCOL.md`.
- **Step-5b findings** (stored units; "recording" = candidate filename group):
  - **Premises:** 17/18 reproduce; lag-8 pose autocorrelation is 0.14 vs the plan's 0.17 (disclosed).
  - **Dense frames give within-recording tracking:** stage 1 seed 0, D2/D3/D4 beat the recording-mean
    oracle on `fit_holdout` (T1 ratios 0.853/0.858/0.905). Sparse D1 does not (1.109).
  - **No transfer to unseen recordings:** body pose never beats the train-mean skeleton (X1 0.993-1.074
    in stage 1; D1 stage 2 on group_dev 1.093-1.108 over 3 seeds). Motion skill there is negative in
    every run.
  - **Why INCONCLUSIVE:** the predeclared tie rule selected D1, because all X1 ratios were ≈ 1. Stage 2
    therefore confirmed the one run that fails T1.
- **Post-hoc addendum** (user-approved after the verdict; `docs/STEP_5B_REPORT.md` Addendum A): the frozen
  stage-2 recipe for dense D3 (`all_train_dense`, 3 seeds) passes T1 in 3/3 (0.837/0.839/0.846) and fails
  X1 in 3/3 (1.155/1.189/1.156; worse than sparse D1). That is the YELLOW-TRACKS pattern, post-hoc. It is
  the second use of group_dev labels. The protocol verdict stays INCONCLUSIVE.
- **Previous verdict:** Step 5 = YELLOW-POSITION (`docs/STEP_5_REPORT.md`).
- **Data:** `data/_artifacts/step5b_pose_signal/freeze_v2` (dense views; 42 fit + 9 validation recordings;
  1,632 guard frames) and `data_v2` (22,754 validated samples, 4.5 GB). Step-5 `freeze_v1`/`data_v1`
  unchanged.
- **Repository:** Step-5b code, tests, plan, protocol and report are on local branch
  `research/step5b-pose-signal` (from `research/step5-single-person-baseline` `11fad75`; worktree
  `.claude/worktrees/charming-napier-69e92f`). **Not pushed, not merged.**
- **Tests:** fixture-backed suite -> 146 passed, 1 skipped, 1 failed (118 baseline + 28 new). The failure
  is the known worktree-only `test_d4_protected_records_include_d3_stop_artifacts_and_match`.
- **Next task (needs a user decision):** the post-hoc D3 check is done. Choose between redesigning Phase C
  around label-free adaptation to the target recordings (their unlabelled CSI is permitted) and switching to
  the label-efficiency track. Do not start either without the user's go-ahead.
- Detailed history below is unchanged; the Step-5b entry is at the top of the progress log.

## Dataset acquisition and audit status (2026-11-05)

The public Wiception3D / Person-in-WiFi-3D dataset has been downloaded,
extracted, and audited read-only. Download, extraction, placement, list/file
pairing checks, and bounded real-array inspection are complete.

- Source URL: https://www.kaggle.com/datasets/xjtuqianbo/wiception3d
  (public, authentication-free download; no credentials used or requested).
- Download date: 2026-11-04 (local time, Asia/Dhaka).
- Archive: `data/_downloads/wiception3d_archive.zip`
  (server-served name `archive.zip`), 16,809,368,215 bytes
  (~15.65 GiB), verified byte-identical to the server Content-Range total.
  Method: 10 resumable HTTP range chunks via standard-library Python
  (threaded urllib), concatenated in order; `zipfile.is_zipfile` passed.
- Extraction location: `data/person_in_wifi_3d/` using a skip-if-exists
  extractor (no overwrites). Result: 293,312 files extracted, 0 skipped,
  18,689,284,877 bytes decompressed. `data/person_in_wifi_3d/process.py`
  is unchanged (4421 bytes, original timestamp) and remains tracked by Git.
- Confirmed actual PIW3D_DATA_ROOT (directory directly containing `train_data`
  and `test_data`): `data/person_in_wifi_3d/wifi_dataset`. The archive introduces
  one wrapper directory level (`wifi_dataset/`); nothing was renamed.
- Actual inventory: train has 89,946 unique listed IDs and test has 7,824. Each
  listed ID has its raw CSI `.mat` and keypoint `.npy`; there are 0 missing or
  unlisted raw CSI/keypoint files and 0 listed CSI-without-keypoint cases. Each
  split also has one SMPL `.npz` per listed ID. Total inventory is 293,312 files.
- Filename-derived audit-only person-count categories: train one/two/three =
  28,121/36,242/25,583; test = 2,586/3,184/2,054. The 12 sampled IDs per split
  cover all three categories, and every sampled pose's person dimension agrees
  with its filename-derived count. These counts must never be exposed to an
  unlabelled training loader.
- All 24 sampled raw CSI files have stored shape `(20, 30, 3, 3)` and compound
  float64 real/imag fields; audit interpretation gives `(3, 3, 30, 20)`. All 24
  sampled poses are float64 `(people, 14, 3)`. No sampled raw CSI or pose array
  contains NaN/Inf. These are bounded content checks, not a full-dataset scan.
- At audit time, `csi_ap`, `csi_amplitude`, and `csi_phase` were absent for every
  listed ID in both splits. Official full-dataset processed arrays remain absent.
  The later bounded Step-3 fixture verified finite shapes and exact amplitude/phase
  concatenation for two official training IDs only; it did not measure full-dataset
  constant-case frequency, ranges, or integrity.
- Sequence analysis found 215 train and 195 test candidate keys, with 195 keys
  shared across the official split and 0 exact listed-ID overlap. Grouped
  train/development splitting is provisionally possible by filename key, but
  recording/subject/environment semantics and pose units/joint ordering/
  coordinate conventions remain unknown.
- Full evidence, exact commands, ranges, next-frame gaps, and the **YELLOW**
  readiness assessment are in `docs/DATA_AUDIT.md`.
- Git: `data/_downloads` added to `.gitignore`; the archive and the extracted
  tree are ignored (`git check-ignore` verified). `process.py` negations
  intact; both `data/*/process.py` remain tracked.
- Disk: ~70 GB free on J: after download + extraction.

## Current phase

Phase 0 focused-test environment setup is complete. The dedicated environment at
`C:\Users\vaska\venvs\compwifi-piw3d-py311` uses Python 3.11.0, pip 24.3.1,
PyTorch 2.5.1+cu121, and torchvision 0.20.1+cu121. `pip check` reported no broken
requirements; CUDA is available on the NVIDIA GeForce RTX 3080 Laptop GPU. The
complete focused suite produced **7 passed, 1 skipped** with no import-related
failures. The exact tested package set is saved in
`requirements/focused-test-py311-lock.txt`.

The Phase 1 read-only structural, bounded-content, pose, split, and metric audits
are complete with a **YELLOW** readiness assessment. The latest audit parsed all
official list filenames, checked 90 official training poses (30 per person-count
category), generated and visually inspected six skeleton figures, and inspected
metric source without editing it. All 90 shape/count/finiteness/nonconstant checks
passed. The local audit interprets axis 2 as vertical and stored units as plausibly
metres, but the original authors' per-axis evaluation instead treats axis 1 as
vertical. This conflict is unresolved; coordinates have not been reordered and
neither interpretation is declared official. Official units, joint ordering,
coordinate frame, and candidate temporal group semantics remain unproven. The paper
documents a first-550/last-50 protocol, but released IDs do not verify its literal
filename mapping: observed train/test frame ranges are 10..599 and 301..349, with
no test IDs in 551..600. There are 215 train and 195 test candidate temporal groups,
with all 195 test groups shared with train and zero exact listed-ID overlap. The
original authors' evaluation supports metre-to-millimetre conversion, while local
legacy MPJPE/PA-MPJPE retain stored units and omit `*1000`; PiW3D PCK uses joints 6
and 4 (the elbows under the provisional ordering); ground-truth-count trimming and
slot-wise pairing hide count/miss/duplicate behavior. Full evidence and the shared
corrections are in `docs/POSE_SPLIT_METRIC_AUDIT.md`.

The bounded Step-3 gate is **GREEN for the specified engineering checks and
protocol documentation only** (2026-11-06). A deterministic ignored fixture now
contains only official training IDs `S11_01_10` and `S11_01_11`. Both processed
samples have finite amplitude `(3,90,20)`, phase `(3,90,20)`, and `csi_ap`
`(3,180,20)`, with exact amplitude/phase concatenation. The Step-3 focused suite
reported **8 passed, 0 skipped** (one external `timm` deprecation warning). A
separate from-scratch training-only smoke completed exactly one forward, backward,
and SGD step on CUDA with `num_workers=0`: loss `3.750966311`, 47 finite
nonzero-gradient tensors, and 47 changed nonzero-gradient parameters (first:
`pose_prompt`). No Step-3 validation/test loader, evaluation, checkpoint
selection, full preprocessing, full training, or pilot was run.

Step-4A is **complete for label-safe data infrastructure and corrected research
metrics only** (2026-11-06). A bounded upstream source check found code-level
support for `*1000` millimetre reporting and axis-1 vertical evaluation, but no
filename-group semantics; unit, axis, and group uncertainties remain recorded
rather than silently resolved. Deterministic ignored manifests were generated from
the official training list only: 215 candidate groups split into 172 internal-train
groups/70,197 IDs and 43 internal-development groups/19,749 IDs, with zero ID
overlap, zero group overlap, and zero immediate-next-frame crossings. A separate
unlabelled CSI loader returns only `csi` tensors, never keypoints, poses,
`person_num`, raw IDs, or filename-count-derived signals. Corrected research
metrics are implemented in a new module with Hungarian matching, matched
MPJPE/PA-MPJPE in millimetres, guarded PCK, exact count accuracy, missed and
duplicate/spurious counts, person precision/recall/F1, root error, and
denominator-weighted global aggregation. The complete focused suite now reports
**33 passed, 0 skipped** with the same external `timm` deprecation warning.
Scientific feasibility, accuracy, novelty, and scientific readiness remain
**unproven**. Official full-dataset processed arrays remain absent, no pretrained
checkpoint is present, TensorBoard, `mat73`, and training-entry-point imports
remain separately unverified, and the corrected evaluator is not wired into
training. Full evidence and exact commands are in `docs/STEP_4A_REPORT.md`.

Step-4B is **BLOCKED at the frozen overfit gate** (2026-11-07). The corrected seed-0 freeze selected 8 internal-train groups/2,048 IDs and 4 internal-development groups/512 IDs (2,560 total), using only filename-marked one-person official-training IDs from the existing internal manifests. Bounded `process_ids` preprocessing into `data/_artifacts/step4b_pilot/pilot_data_v2/` validated all 2,560 samples with zero failures: finite `(3,90,20)` amplitude/phase, exact `(3,180,20)` amplitude/phase concatenation, finite `(1,14,3)` keypoints, and correct internal split assignment. The separate Q=1 research loader/entry point was implemented without running `train_pose.py` or touching official test data. The frozen one-batch overfit check on CUDA completed 200 finite SGD steps and reduced stored-unit MPJPE from `4.920130252838135` to `0.34014594554901123` (ratio `0.06913352453480291`), but failed the predeclared absolute criterion `<=0.10`; decision `BLOCKED` is recorded in `data/_artifacts/step4b_pilot/decision.json` and `docs/STEP_4B_REPORT.md`. No epoch run, development evaluation, mean-pose comparison, compositional training, package change, download, deletion, commit, push, or remote change occurred after the failure. The complete focused suite with the existing two-frame fixture now reports **48 passed, 0 skipped**, with the same external `timm` deprecation warning. Scientific feasibility remains unproven.

Step-4B-D1 is **complete as a disclosed extended-step diagnostic** (2026-11-07). The verifier confirmed the corrected `freeze_v2`/`pilot_data_v2` internal-training artifacts, exact CSI/keypoint pairing for the fixed eight-sample batch, seed `0`, SGD(`lr=0.01`, `momentum=0.9`), 200 original steps, unchanged coordinates, and the original `BLOCKED` decision. The separate D1 freeze changed only the optimizer budget to `800` steps. On CUDA, all losses and gradients remained finite; initial and step-200 MPJPE exactly matched the original values (`4.920130252838135` and `0.34014594554901123` stored units), but final step-800 train-mode MPJPE remained `0.32848620414733887`, above the unchanged `<=0.10` criterion. The trajectory plateaued around `0.33–0.34`; predictions did not collapse toward the fixed-batch or frozen-train mean; residual error concentrated in coordinate index 2 and joint indices 8, 11, 9, 1, and 10. Original Step-4B remains **BLOCKED**. Full measurements and plots are in `docs/STEP_4B_D1_REPORT.md` and `data/_artifacts/step4b_d1/`; no development samples, official test data, epoch training, package change, download, deletion, commit, push, or remote change occurred.

Step-4B-D2 is **complete as a disclosed encoder-trainability diagnostic** (2026-11-07). Before editing, a named-parameter audit directly confirmed the original runtime configuration: all 50 encoder parameters (`3,390,464` values) were frozen, all 47 pose-decoder parameters (`1,404,931` values) were trainable, and SGD covered exactly those 47 parameters. The minimal opt-in `train_encoder` mechanism preserved default legacy behavior. The frozen D2 protocol changed only encoder trainability from frozen to trainable while preserving the corrected tiny batch, seed `0`, initialization values, 200 SGD(`lr=0.01`, `momentum=0.9`) steps, loss, data, preprocessing, and `<=0.10` stored-unit criterion. CUDA training completed with finite losses/gradients; all 97 intended parameters were optimized exactly once, all 50 encoder gradients were nonzero every step, and all 50 encoder parameters changed. Final train-mode MPJPE was `0.46943652629852295`, above criterion and `0.12929058074951172` worse than the original frozen-encoder endpoint. Original Step-4B remains **BLOCKED**; D1 remains above criterion. Full measurements are in `docs/STEP_4B_D2_REPORT.md` and `data/_artifacts/step4b_d2/`; no development samples, official test data, epoch training, package change, download, deletion, commit, push, or remote change occurred.

Step-4B-D3 is **stopped at its frozen D2 reproduction gate** (2026-11-07). The separate protocol, D3-only runner, and five focused tests were added before training; the focused file passed `5 passed, 1 warning`, and the complete fixture-backed suite passed **62 passed, 0 skipped**, with the same external `timm` warning. The pre-run verifier confirmed the protected original Step-4B, D1, and D2 records and wrote `data/_artifacts/step4b_d3/step4b_d3_freeze.json` before model construction. The fresh seed-0 trainable-encoder run reproduced the D2 initial train-mode MPJPE exactly, but produced step-100 MPJPE `0.40826788544654846` instead of recorded D2 `0.5151628255844116`, and step-200 MPJPE `0.3430270552635193` instead of recorded D2 `0.46943652629852295`. Per the frozen stop rule, D3 exited with `D2_REPRODUCTION_MISMATCH_STOPPED` before extending past step 200. Consequently there is no valid step-800 MPJPE, final/initial ratio, late-trajectory summary, D3 criterion status, checkpoint/prediction trajectory, final state-dict, result JSON, or decision JSON. Original Step-4B remains **BLOCKED**; D1/D2 remain above-criterion diagnostics. Full evidence is in `docs/STEP_4B_D3_PROTOCOL.md`, `docs/STEP_4B_D3_REPORT.md`, and `data/_artifacts/step4b_d3/`; no development samples, official test data, epoch training, package change, download, deletion, commit, push, remote change, rerun, or automatic fix attempt occurred.

Step-4B-D4 is **complete as a bounded reproducibility audit** (2026-11-08). Before new runs, the static comparison found that D3 aliases the D2 model/optimizer builders and shares the seed, ordered eight-sample loader, forward/backward/loss sequence, SGD update, parameter audits, and step-0/100/200 train-mode checkpoint procedure; observed D3 differences were wrapper verification/bookkeeping, the 800-step budget, duplicate gradient-statistics bookkeeping, and the post-step-200 reproduction gate. The environment record captured Python `3.11.0`, PyTorch `2.5.1+cu121`, CUDA `12.1`, RTX 3080 Laptop GPU, seed `0`, and deterministic algorithms/cuDNN/cuBLAS-related settings without enabling deterministic mode. Two independent unchanged seed-0 200-step D2 repeats completed under `data/_artifacts/step4b_d4/repeat_a/` and `repeat_b/`: both exactly reproduced D2's initial MPJPE `4.920130252838135`, had finite losses/gradients, exact 97-parameter optimizer coverage, and 97/97 trainable plus 50/50 encoder parameter changes. Repeat A measured step-100 MPJPE `0.4460887312889099` and step-200 `0.46392330527305603`; repeat B measured `0.49196043610572815` and `0.37702471017837524`. Exact comparisons classify the evidence as **`repeats differ from each other`**; the first A/B full-record divergence is step 1 in `max_abs_encoder_gradient`, and the first A/B pre-update-loss divergence is step 3. The D4 focused file passed `6 passed, 1 warning`, and the complete fixture-backed suite passed **68 passed, 0 skipped**, with the same external `timm` warning. All 40 pre-selected protected historical records remained unchanged before and after D4. Full evidence is in `docs/STEP_4B_D4_REPORT.md` and `data/_artifacts/step4b_d4/`; no D3 rerun, 800-step extension, deterministic-setting change, optimizer/criterion/data change, development/test access, epoch training, package change, download, deletion, commit, push, remote change, or automatic fix attempt occurred.

Step-4B-R is **complete as a deterministic single-person baseline, verdict `BASELINE_COMPLETE`** (2026-11-08; recorded here during Phase A, numbers from `docs/STEP_4B_R_REPORT.md` and `data/_artifacts/step4b_r/baseline/`). It is a separately named baseline, not a rerun or reversal of Step-4B/D1-D4. Data: 2,048 single-person internal-train IDs (8 candidate groups) and 512 internal-development IDs (4 groups, zero ID/group overlap); official test, multi-person samples, unlabelled data, and pretrained checkpoints unused. Model: from-scratch `MAE_ViT` encoder (trainable) + legacy `ViT_Pose_Decoder`, stored `(1,14,3)` targets, AdamW lr `0.001`, weight decay `0.0`, batch 8, 10 epochs, final-epoch checkpoint only, deterministic execution enabled (`PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Result in stored units: final model train/development MPJPE `1.0592086446704343`/`0.8333281083032489` versus training-derived constant mean pose `0.989848210435728`/`0.8013955073852118`; the model did **not** beat the mean predictor on either partition. The report records a focused file of `6 passed` and the fixture-backed suite at `74 passed` before the run. No accuracy, generalization, or feasibility claim follows.

Step-4B-R collapse diagnosis is **complete (Phase A, 2026-11-10)** and reproduced every premise of `docs/NEXT_PHASE_PLAN.md` section 1 (`diagnose_step4b_r_collapse.py`, read-only on artifacts, CPU; report `docs/STEP_4B_R_COLLAPSE_DIAGNOSIS.md`, data `data/_artifacts/step4b_r_diagnosis/diagnosis.json`). (1) The 512 stored development predictions are a single common point `[2.7970, 1.7842, 3.2779]` (max deviation `4.3e-06`, std across samples `6.6e-07`; ground-truth std across samples `0.3186`). (2) In eval mode on 16 train + 16 development samples, std across samples is `0.1959` at the pooled encoder feature and `0.4761` after `gconv3` (std across joints `3.1694`), but `4.240e-04` after `transformer_encoder1`, `5.362e-06` after `transformer_encoder2`, and `4.746e-07` after `transformer_encoder3`: the collapse is inside the legacy decoder's post-norm transformer layers. The activation checks are pattern checks within 3x (the plan quotes about 1-2 significant figures and no std convention); the target and prediction checks are exact-to-rounding. (3) Target decomposition from train labels only: absolute mean-pose MPJPE `0.9898`/`0.8014`, centroid distance to the train-mean centroid `0.9683`/`0.7874`, centroid-relative mean-skeleton MPJPE `0.1549`/`0.1279` (train/development, stored units, 14-joint centroid as translation reference). (4) The pilot contains exactly 8 training and 4 development candidate groups, each essentially one position (per-axis within-group centroid std 0.007-0.117 train, 0.008-0.031 development). These are descriptive measurements of one run; they do not prove the cause of the collapse beyond what the activation trace shows.

### Step-3 bounded gate commands and outputs

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_piw3d_preprocessing.py -q -p no:cacheprovider
```

Result: `2 passed in 3.10s`. Before implementation, both tests failed with the
expected `ImportError` for missing `process_id`/`process_ids`.

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe make_piw3d_smoke_fixture.py --source-root data/person_in_wifi_3d/wifi_dataset --output-root data/_fixtures/piw3d_two_frame
```

The final run selected `S11_01_10, S11_01_11`, processed only those two files, and
printed the required shapes, `finite=True`, and `exact_concatenation=True` for
both. The first attempt stopped before processing with
`FileNotFoundError: ... data\_fixtures\piw3d_two_frame\train_data\csi\S11_01_10.mat`;
copying the same two official raw training files into the ignored fixture resolved
that tooling error. No original dataset file was modified.

```bash
PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_legacy_baseline.py -q -p no:cacheprovider
```

Result: `8 passed, 1 warning in 20.27s`; no tests were skipped, weakened, or added
to this eight-test focused file.

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe smoke_piw3d_training.py --data-root data/_fixtures/piw3d_two_frame
```

Result: one eligible training sample; IDs `S11_01_10, S11_01_11`; device `cuda`;
one forward/backward/optimizer step; finite loss `3.750966311`; 47 finite and 47
nonzero-gradient tensors; 47 changed nonzero-gradient parameters, first
`pose_prompt`. The model was created from scratch and no checkpoint was loaded.

### Latest audit command and outputs

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe audit_piw3d_pose_and_split.py --data_root data/person_in_wifi_3d/wifi_dataset --output docs/POSE_SPLIT_METRIC_AUDIT.md --figure_dir docs/figures/pose_audit
```

The command exited successfully and printed YELLOW, the verified/unresolved facts,
rating reasons, and generated files. Syntax compilation and `--help` also completed
successfully before execution. The legacy pytest suite was deliberately not run
because this read-only task prohibits its synthetic optimization/training behavior.

Generated/changed by this audit: `audit_piw3d_pose_and_split.py`,
`docs/POSE_SPLIT_METRIC_AUDIT.md`, six PNGs under `docs/figures/pose_audit/`, and
this handoff. No preprocessing/training/test-array access, package installation,
metric/source change, dataset write, commit, or push occurred.

## Completed work

- Research direction, research protocol, and phased plan documented.
- Read `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, and this handoff completely.
- Inspected the repository structure, entry points, YAML configurations,
  dataset feeders, preprocessing scripts, models, and evaluation utilities
  without modifying them.
- Documented the code-implied dataset layouts, path settings, tensor shapes,
  documented commands, batching, checkpointing, output paths, and metric
  definitions.
- Confirmed that no mixed-precision/AMP option is implemented.
- Confirmed that the existing evaluation implements MPJPE, PA-MPJPE, and
  torso-normalized PCK, but not the complete set-based evaluation required by
  the research specification.
- Queried the Git commit, local dataset directories, Python/PyTorch/CUDA
  availability, GPU, Conda availability, and installed package metadata.
- Delivered the read-only Phase 0 repository/environment report.
- Traced the complete Person-in-WiFi-3D import closure, including the other
  feeders and alternative models imported eagerly by the unmodified entry
  points.
- Proposed a dedicated Python 3.11.11 environment, PyTorch 2.5.1/torchvision
  0.20.1 with CUDA 12.1, exact direct-dependency candidate pins, and separate
  baseline/optional/development manifests.
- Documented exact environment-creation, dry-run, installation, import, timm,
  and RTX 3080 verification commands without executing them.
- Checked the official PyTorch Windows/CUDA installation pairing and PyPI
  `mat73==0.65` metadata; recorded unresolved compatibility items explicitly.
- Added a standalone read-only Person-in-WiFi-3D audit command with full-list
  file-pairing checks and bounded array-content sampling. It does not import or
  modify any training loader.
- Verified syntax, CLI help, clean absent-dataset behavior, report generation,
  and refusal to write a report inside the dataset root.
- Initially generated `docs/DATA_AUDIT.md` for the absent-dataset state, then
  completed the real-data audit after extraction. The report now separates full
  list/file measurements, bounded array-content measurements, source-code
  expectations, and unknown units/conventions. It assigns YELLOW readiness and
  records the exact dedicated-environment commands.
- Replaced the undefined pose-decoder `feature.device` reference with `x.device`.
  A pre-fix runtime reproduction was attempted but stopped at the missing `timm`
  import, so the expected `NameError` was not observed.
- Aligned the PiW3D pose YAML with the existing pretraining checkpoint directory.
- Enabled the processor's existing combined amplitude/phase `csi_ap` output.
- Added the missing pose `step_count` increment and divided accumulated losses by
  `steps_per_update` for `steps_per_update > 1`. The helper returns the original
  loss object unchanged when `steps_per_update == 1`; the synthetic
  effective-batch accumulation test passes.
- Guarded constant amplitude/phase normalization and added an optional PiW3D
  loader `num_workers` argument whose default remains `8`.
- Added focused tests for MAE backward, pose output, one optimization step,
  effective-batch accumulation, checkpoint round-trip, constant CSI
  normalization, checkpoint-path consistency, and a conditional real
  training-sample load using `num_workers=0`.
- Inspected the focused-test import closure and documented a smaller
  approval-gated Python 3.11 virtual-environment installation command set.
- Created the dedicated environment, upgraded pip to 24.3.1 after separate
  approval, installed the approved focused dependencies, verified the runtime,
  and reran the complete suite: 7 passed and 1 skipped, with no import failures.
- Saved the exact successfully tested package set as
  `requirements/focused-test-py311-lock.txt`.
- Completed the bounded Step-3 documentation corrections, two-frame real-training
  fixture, focused acceptance run, one-step training-only smoke, and design-only
  research protocol. The bounded engineering/protocol gate is GREEN; scientific
  readiness remains unproven and no pilot or full research model was started.

## Files changed

- Read-only inspection: no files changed.
- Environment proposal and earlier status handoff: `docs/CURRENT_TASK.md` only.
- Dataset audit tooling: added `audit_person_in_wifi_3d.py` and
  `docs/DATA_AUDIT.md`; updated `docs/CURRENT_TASK.md`.
- Real-data audit: updated `docs/DATA_AUDIT.md` and
  `docs/CURRENT_TASK.md` only. No source, test, or dataset file was changed.
- Legacy baseline repair: changed `model/model.py`, `train_pose.py`, `utils.py`,
  `config/person_in_wifi_3d/pose_config.yaml`,
  `data/person_in_wifi_3d/process.py`, and `feeder/person_in_wifi_3d.py`; added
  `tests/test_legacy_baseline.py`; updated this handoff.
- Focused-test setup: added `requirements/focused-test-py311-lock.txt`; updated
  `docs/CURRENT_TASK.md` and the static `csi_ap` statement in
  `docs/DATA_AUDIT.md`. No source or test changes were needed after installation.
- Bounded Step-3 gate: added `data/_fixtures/` to `.gitignore`; refactored
  `data/person_in_wifi_3d/process.py`; added `make_piw3d_smoke_fixture.py`,
  `smoke_piw3d_training.py`, and `tests/test_piw3d_preprocessing.py`; added
  `docs/RESEARCH_PROTOCOL.md` and `docs/STEP_3_GATE_GOAL.md`; updated
  `docs/PROJECT_CONTEXT.md`, `docs/POSE_SPLIT_METRIC_AUDIT.md`, and this handoff.
  The generated fixture is ignored and is not a source change.
- Step-4A infrastructure: added `piw3d_manifest.py`,
  `feeder/piw3d_unlabelled.py`, `research_metrics.py`, and 23 new tests across
  `tests/test_piw3d_manifest.py`, `tests/test_piw3d_unlabelled_loader.py`, and
  `tests/test_research_metrics.py`; added the `data/_artifacts/` ignore rule and
  generated ignored candidate-group manifests under
  `data/_artifacts/piw3d_candidate_group_manifests/`; added
  `docs/STEP_4A_GOAL.md` and `docs/STEP_4A_REPORT.md`; updated this handoff.
  Legacy `utils.py` and `feeder/person_in_wifi_3d.py` were not changed.
- Step-4B-D1 diagnostic: added `step4b_d1_diagnostic.py`,
  `tests/test_step4b_d1_diagnostic.py`, and `docs/STEP_4B_D1_REPORT.md`; updated
  this handoff. Generated ignored evidence under `data/_artifacts/step4b_d1/`.
  Original `step4b_config.py`, the Step-4B entry point, and the original
  `data/_artifacts/step4b_pilot/decision.json` were not changed.
- Step-4B-D2 diagnostic: changed `model/model.py` only to add the opt-in
  `train_encoder` mechanism while preserving the default legacy frozen-encoder
  behavior; added `step4b_d2_diagnostic.py`,
  `tests/test_step4b_d2_diagnostic.py`, and `docs/STEP_4B_D2_REPORT.md`; updated
  this handoff. Generated ignored evidence under `data/_artifacts/step4b_d2/`.
  Original `step4b_config.py`, the Step-4B entry point, original Step-4B decision
  artifacts, and D1 source/results were hash-verified unchanged.
- Step-4B-D3 diagnostic: added `step4b_d3_diagnostic.py`,
  `tests/test_step4b_d3_diagnostic.py`, `docs/STEP_4B_D3_PROTOCOL.md`, and
  `docs/STEP_4B_D3_REPORT.md`; updated this handoff. Generated ignored evidence
  under `data/_artifacts/step4b_d3/`. Original Step-4B, D1, and D2 records were
  hash-verified unchanged. No model source file was changed for D3.
- Step-4B-D4 reproducibility audit: added `step4b_d4_reproducibility.py`,
  `tests/test_step4b_d4_reproducibility.py`, and `docs/STEP_4B_D4_REPORT.md`;
  updated this handoff. Generated ignored evidence under
  `data/_artifacts/step4b_d4/`, including two complete unchanged-D2 repeat roots.
  D2/D3 runners, model source, and protected historical records were not changed.
- `docs/EXPERIMENT_LOG.md` was not created because no experiments were run.

## Repository and Git findings

- Commit `aaeacd5` (`Repair legacy PiW3D baseline smoke path`) was pushed to
  `kimi/main` and verified on the remote.
- Branch: `main`, tracking `kimi/main`.
- The current Step 3 follow-up changes to `utils.py`, `train_pose.py`,
  `tests/test_legacy_baseline.py`, and this handoff are local until separately
  committed and pushed.
- There is no standalone evaluation command; `train_pose.py` evaluates after
  each training epoch.

## Actual commands run and outcomes

The following shell commands were run read-only from the repository root.
File reads and file/content searches were also performed with dedicated
read-only tools. Project modules were not imported.

### Git commit and initial status

```bash
git rev-parse HEAD && git status --short --branch
```

Output:

```text
2fbbef271454e6fbb95b1a0163b8cdbb96163838
## main...origin/main
?? AGENTS.md
?? docs/
```

### Local dataset-directory inspection

```bash
ls -la data && ls -la data/person_in_wifi_3d && ls -la data/wipose
```

Outcome: `data/person_in_wifi_3d/` contained only `process.py`;
`data/wipose/` contained only `process.py`. No `train_data`, `test_data`,
WiPose `Train`/`Test`, or MM-Fi dataset directory was present. Data could
exist elsewhere, but no other location was inspected or verified.

### Python, PyTorch, and CUDA query

```bash
python --version && python -c "import torch; print('torch:', torch.__version__); print('torch CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('CUDA device count:', torch.cuda.device_count())"
```

Output:

```text
Python 3.14.0
torch: 2.11.0+cu128
torch CUDA build: 12.8
CUDA available: True
CUDA device count: 1
```

This establishes runtime availability only, not DT-Pose compatibility.

### GPU query

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
```

Output:

```text
NVIDIA GeForce RTX 3080 Laptop GPU, 16384 MiB, 581.29
```

### Conda query

```bash
conda --version
```

Outcome: failed with exit code 127; `conda: command not found`. Conda is not
on this Bash shell's `PATH`. This does not prove that Anaconda is not
installed elsewhere on the system.

### Installed-package metadata query

```bash
python -c "from importlib.metadata import version, PackageNotFoundError; names=['numpy','torch','torchvision','tensorboard','PyYAML','scikit-learn','scipy','opencv-python','pandas','h5py','PyWavelets','mat73','timm','einops','pywin32']; [(print(f'{n}: {version(n)}') if True else None) for n in []]; exec('for n in names:\n try: print(f\"{n}: {version(n)}\")\n except PackageNotFoundError: print(f\"{n}: NOT INSTALLED\")')"
```

Output:

```text
numpy: 2.5.2
torch: 2.11.0+cu128
torchvision: NOT INSTALLED
tensorboard: NOT INSTALLED
PyYAML: 6.0.3
scikit-learn: 1.9.0
scipy: 1.18.1
opencv-python: NOT INSTALLED
pandas: 3.0.5
h5py: NOT INSTALLED
PyWavelets: NOT INSTALLED
mat73: NOT INSTALLED
timm: NOT INSTALLED
einops: NOT INSTALLED
pywin32: NOT INSTALLED
```

### Initial-inspection final Git status

```bash
git status --short --branch
```

Output:

```text
## main...origin/main
?? AGENTS.md
?? docs/
```

### Environment-proposal read-only checks

The following interpreter query did not import project modules:

```bash
python -c "import importlib.util, sys; print(sys.version); print('xxsubtype:', importlib.util.find_spec('xxsubtype'))"
```

Output:

```text
3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)]
xxsubtype: ModuleSpec(name='xxsubtype', loader=<class '_frozen_importlib.BuiltinImporter'>, origin='built-in')
```

The proposed documentation diff was checked with:

```bash
git diff --check -- docs/CURRENT_TASK.md
```

Outcome: exit code 0 with no output. This checks whitespace/conflict markers in
the documentation diff only; it is not a runtime or dependency test.

```bash
git diff --stat -- docs/CURRENT_TASK.md
```

Output before this status-only update:

```text
 docs/CURRENT_TASK.md | 226 ++++++++++++++++++++++++++++++++++++++++++++++++---
 1 file changed, 216 insertions(+), 10 deletions(-)
```

### Current handoff Git checks

```bash
git rev-parse HEAD
```

Output:

```text
8432b2a6547c59b92be323a3a38284dd88d1392c
```

```bash
git status --short --branch
```

Output before this status-only update:

```text
## main...kimi/main
 M docs/CURRENT_TASK.md
```

The official PyTorch previous-versions page and PyPI `mat73==0.65` JSON metadata
were also read through the browser. No package index command or installation
command was executed.

### Dataset-audit implementation checks

A syntax-only check that does not create bytecode was run:

```bash
python -c "from pathlib import Path; p = Path('audit_person_in_wifi_3d.py'); compile(p.read_text(encoding='utf-8'), str(p), 'exec'); print('Syntax OK')"
```

Output:

```text
Syntax OK
```

```bash
python audit_person_in_wifi_3d.py --help
```

Output began with the expected usage and showed required `--data_root` and
`--output` options plus the bounded `--max_samples` option (default 6).

```bash
python audit_person_in_wifi_3d.py --data_root data/person_in_wifi_3d --output docs/DATA_AUDIT.md
```

Output:

```text
Audit report written to: D:\Zero-shot 3D Multi-person Pose Estimation\docs\DATA_AUDIT.md
Person-in-WiFi-3D dataset content was not found at: D:\Zero-shot 3D Multi-person Pose Estimation\data\person_in_wifi_3d
Place the existing dataset under train_data/ and test_data/, or pass its actual root with --data_root.
No dataset files were created or modified. Rerun the same audit command after supplying the dataset.
```

The report-path guard was checked with an output under `data/person_in_wifi_3d`.
It refused the write with `Refusing to write the report inside --data_root`, and
the shell checks confirmed a nonzero command status and no created report file.
After adding shell quoting to the generated rerun instructions, the syntax check
and audit were rerun with the quoted absolute paths and completed with the same
clean absent-dataset output. Directory inspection still showed only `process.py`.

### Real-data audit execution (2026-11-05)

The following commands were run from the repository root using the existing
dedicated environment. The script and CLI were inspected before data loading.
Root confirmation succeeded and verified that the supplied root directly
contains both splits and both list files:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -c "from pathlib import Path; root = Path('data/person_in_wifi_3d/wifi_dataset').resolve(); print('PIW3D_DATA_ROOT:', root); print('root is directory:', root.is_dir()); print('train_data is directory:', (root / 'train_data').is_dir()); print('test_data is directory:', (root / 'test_data').is_dir()); print('train list is file:', (root / 'train_data' / 'train_data_list.txt').is_file()); print('test list is file:', (root / 'test_data' / 'test_data_list.txt').is_file()); assert root.is_dir() and (root / 'train_data').is_dir() and (root / 'test_data').is_dir()"
```

Actual root: `J:\Zero-shot 3D Multi-person Pose Estimation\data\person_in_wifi_3d\wifi_dataset`; all five printed existence checks were true.

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe audit_person_in_wifi_3d.py --help
```

Help completed successfully and showed required `--data_root`/`--output` and
bounded `--max_samples` (default 6).

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -c "from pathlib import Path; import audit_person_in_wifi_3d as a; root = Path('data/person_in_wifi_3d/wifi_dataset'); exec('for split, dirname in a.SPLITS.items():\n sample_list = a.load_sample_list(root / dirname, split)\n selected = a.select_samples(sample_list[\"unique_ids\"], 12)\n print(split, \"filename-derived counts:\", dict(sorted(sample_list[\"filename_counts\"].items(), key=lambda item: (item[0] is None, str(item[0])))))\n print(split, \"selected IDs:\", selected)\n print(split, \"selected counts:\", [a.filename_info(sample_id)[\"person_count\"] for sample_id in selected])\n assert {1, 2, 3}.issubset({a.filename_info(sample_id)[\"person_count\"] for sample_id in selected})')"
```

The representative-selection assertions passed. Both 12-ID selections contain
one-, two-, and three-person filename categories. The exact selected IDs and
count output are recorded in `docs/DATA_AUDIT.md`.

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe audit_person_in_wifi_3d.py --data_root data/person_in_wifi_3d/wifi_dataset --output docs/DATA_AUDIT.md --max_samples 12
```

Actual output: `Audit report written to: J:\Zero-shot 3D Multi-person Pose Estimation\docs\DATA_AUDIT.md`. Exit code was 0. Only the requested report was
written; no dataset file was created, modified, or preprocessed.

## Actual test and execution outcomes

- The dedicated Python 3.11 environment was created and all approved focused-test
  dependencies were installed after the separately approved pip upgrade.
- The earlier complete focused suite produced 7 passed, 1 skipped, and one
  external `timm` deprecation warning, as itemized below. With the Step-3 fixture,
  the same eight-test file now reports **8 passed, 0 skipped**, with one warning.
- Synthetic forward/backward, optimizer update, checkpoint save/reload, gradient
  accumulation, and normalization checks passed. Two additional synthetic bounded
  preprocessing regression tests passed.
- The deterministic fixture verified the real training sample load for
  `S11_01_10` and `S11_01_11`. Official full-dataset processed arrays remain
  absent; no tiny-overfitting or full training run was performed.
- The real-data audit completed full-list existence and pairing checks, and all
  24 sampled raw CSI plus 24 sampled keypoint files loaded with finite values.
  Its documentation-only task did not rerun pytest.
- Step 3 ran bounded preprocessing for exactly two training IDs and one separate
  forward/backward/optimizer smoke step. No pretraining, full pose training,
  evaluation, checkpoint selection, or experiment was run. No baseline performance
  or research result is claimed.
- Training-entry-point imports, TensorBoard, and `mat73` were not verified; the
  environment must not yet be described as generally training-ready.
- Step-4B-D2 added five focused trainability/optimizer tests; the new file passed
  `5 passed, 1 warning`, and the existing Step-4B/D1 focused files still passed
  `7 passed, 1 warning` before the frozen diagnostic. The one-shot D2 CUDA run
  completed 200 finite steps with all 97 intended gradients finite, all 50 encoder
  gradients nonzero, all 50 encoder parameters changed, and final train-mode MPJPE
  `0.46943652629852295` stored units. After implementation, the complete
  fixture-backed focused suite reported **57 passed, 0 skipped**, with one
  external `timm` deprecation warning.
- Step-4B-D3 added five focused protocol/default tests, passed
  `5 passed, 1 warning`, and then passed the complete fixture-backed suite with
  **62 passed, 0 skipped** and the same external warning. The pre-run verifier
  exited `0`. The frozen CUDA run stopped by design at the D2 reproduction gate:
  initial MPJPE reproduced exactly, step-100 MPJPE was `0.40826788544654846`
  instead of `0.5151628255844116`, and step-200 MPJPE was `0.3430270552635193`
  instead of `0.46943652629852295`. It exited `3` with
  `D2_REPRODUCTION_MISMATCH_STOPPED`; no valid 800-step result exists.
- Step-4B-D4 added six focused static/environment/protection/comparison tests and
  passed `6 passed, 1 warning`. Two independent unchanged seed-0 D2 repeats
  completed 200 finite steps in separate artifact roots. Exact comparisons
  classified the evidence as **`repeats differ from each other`**; repeat A/B
  checkpoint MPJPE values were `0.4460887312889099`/`0.46392330527305603` and
  `0.49196043610572815`/`0.37702471017837524` at steps 100/200. The complete
  fixture-backed suite passed **68 passed, 0 skipped**, and all 40 protected
  historical records were hash-verified unchanged after the audit.

## Newly discovered dataset facts

- The actual root is `data/person_in_wifi_3d/wifi_dataset/`; train/test list
  sizes are 89,946/7,824 unique IDs with no duplicate entries. The archive is
  kept in `data/_downloads/`.
- All listed raw CSI and keypoint files are present with zero missing, unlisted,
  or CSI-without-keypoint cases. SMPL `.npz` files also match the physical counts,
  but their contents were outside this audit's array-content checks.
- Filename-derived train one/two/three-person counts are 28,121/36,242/25,583;
  test counts are 2,586/3,184/2,054. All sampled pose person dimensions match.
  This encoding remains audit-only and forbidden as an unlabelled training signal.
- Sampled raw CSI uses HDF5 variable `csi_out`, stored shape `(20, 30, 3, 3)`, and
  compound little-endian float64 real/imag fields. Audit interpretation produces
  `(3, 3, 30, 20)` complex128. Across both 12-sample sets, real/imag/magnitude
  ranges are `[-67, 65]`, `[-69, 63]`, and `[0, 73.736]`, respectively.
- Sampled poses are float64 `(people, 14, 3)` with combined range
  `[-0.171579, 4.10975]`. No sampled raw CSI or pose contains NaN/Inf. Content
  statistics are bounded to the samples, not dataset-wide integrity claims.
- Official full-dataset `csi_ap`, amplitude, and phase arrays remain absent. The
  ignored two-ID fixture contains finite amplitude `(3,90,20)`, phase `(3,90,20)`,
  and combined `csi_ap` `(3,180,20)` arrays, with exact concatenation and a
  successful real feeder load. Full-dataset ranges, constant-case frequency, and
  integrity were not measured.
- Candidate sequence keys produce 215 train groups and 195 test groups. Train has
  2,032 IDs without an immediate successor on the unfiltered list. All 195 test
  candidate keys overlap train keys, while exact listed-ID overlap is zero.
  Filename-grouped train/development splitting is provisionally possible, but no
  recording-, subject-, or environment-disjoint claim is supported yet.
- Pose units, joint ordering, coordinate axes/origin/frame, and candidate temporal
  group semantics remain unproven. Local visualization interprets axis 2 as
  vertical, while the original authors' per-axis evaluation interprets axis 1 as
  vertical; neither is declared settled and coordinates were not reordered. See
  `docs/DATA_AUDIT.md` and `docs/POSE_SPLIT_METRIC_AUDIT.md`.

## Newly discovered environment facts

- The system interpreter remains Python 3.14.0 with PyTorch 2.11.0+cu128; the
  dedicated focused-test environment separately uses Python 3.11.0, pip 24.3.1,
  PyTorch 2.5.1+cu121, and torchvision 0.20.1+cu121.
- In the dedicated environment, `pip check` found no broken requirements, CUDA
  12.1 availability was true, and the GPU name was NVIDIA GeForce RTX 3080
  Laptop GPU. The Step-3 smoke subsequently completed one bounded CUDA training
  step; this is not a general training-readiness result.
- The exact tested focused package set is materialized in
  `requirements/focused-test-py311-lock.txt`. The older candidate manifests below
  remain historical proposals and were not materialized or installed.
- Conda is unavailable on the current Bash `PATH`; the approved environment used
  the existing 64-bit CPython 3.11.0 interpreter and `venv` instead.
- The unmodified PiW3D entry points import all three dataset feeders;
  `train_pose.py` additionally imports MetaFi and HPE-Li. Consequently,
  `mat73` and TensorBoard remain outside this focused-test setup even though they
  belong to the full training-entry-point import closure.
- The focused tests verified the legacy `timm` import paths and whole-model
  `torch.load` checkpoint round-trip on CPU. They emitted one deprecation warning
  for `timm.models.layers`; this did not fail the suite.
- Full training-entry-point imports, `mat73`, TensorBoard, and real-data
  compatibility beyond the bounded fixture remain unverified.

## Current errors, blockers, and uncertainties

### Legacy findings resolved in this task

1. `ViT_Pose_Decoder.forward()` used undefined `feature.device`; it now uses
   `x.device`. Synthetic output, optimization, and checkpoint tests now pass.
2. The PiW3D pose checkpoint path omitted `all-person`; it now matches the
   pretraining output path, and its consistency test passes.
3. The processor wrote amplitude and phase but not the feeder's required
   `csi_ap`; it now saves the existing combined `(3, 180, 20)` array as well.
   Compatibility is verified only for the two-ID fixture.
4. Pose gradient accumulation did not increment `step_count`, and its unscaled
   backward calls summed microbatch means for `steps_per_update > 1`. The loop
   now increments the counter and divides only accumulated losses by
   `steps_per_update`; `steps_per_update == 1` returns the original loss object
   and keeps legacy behavior. The synthetic effective-batch regression passes.
5. Constant amplitude/phase normalization previously divided by zero. A constant
   component now maps to zeros instead of NaNs; nonconstant min-max behavior is
   unchanged, and other NaN/Inf handling was not broadened. Its regression test
   passes.

### Remaining code/protocol issues

1. The Person-in-WiFi `validation` loader reads `test_data`, and checkpoint
   selection uses its labels. Development validation and official test are not
   separated as required by the research protocol.
2. Ground-truth `person_num` trims predictions during training and evaluation;
   the current model does not predict presence or infer count.
3. The existing pretraining feeder loads pose labels and exposes padded poses
   and `person_num`; it is not a CSI-only unlabelled loader.
4. MM-Fi contains hardcoded `/root/autodl-tmp/...` paths that are incompatible
   with the current Windows layout.
5. No Hungarian matching, root-localization error, exact count accuracy,
   missed/duplicate counts, or person-level precision/recall/F1 are implemented.

### Runtime verification blockers

- The focused dependency import failures are resolved. The fixture-backed focused
  suite passed all eight tests, and the separate one-step smoke passed.
- Official full-dataset preprocessing was not run and no pretrained checkpoint is
  available. The bounded Step-3 gate is complete, but these results do not
  establish full baseline reproduction or general training readiness.
- TensorBoard, `mat73`, and training-entry-point imports remain separately
  unverified. No full training-readiness claim is made.

### Still unknown

- Full-dataset processed ranges, constant-case frequency, NaN/Inf status, file
  readability, and feeder compatibility beyond the verified two-ID fixture.
- Official pose units, joint ordering, and coordinate conventions; visualization
  has been generated and inspected, but it does not prove those conventions. The
  exact upstream axis/unit source lines remain independently unverified.
- Recording, subject, environment, and clip semantics of candidate temporal groups.
- Whether the dedicated environment can run all entry points and whether further
  runtime blockers remain once dependencies are available.
- Baseline reproduction accuracy and zero-label compositional viability.

## Proposed Phase 0 environment plan (not executed)

No environment was created and no installation, data download, preprocessing,
project-module import, model execution, or training was performed for this
proposal. The filenames and commands below are planned, not existing files or
completed steps.

### Person-in-WiFi-3D import audit

The unmodified `pretrain.py` and `train_pose.py` entry points eagerly import all
three feeders. `train_pose.py` also eagerly imports MetaFi and HPE-Li. Therefore,
an environment for the unmodified Person-in-WiFi-3D baseline must satisfy the
whole import closure, including packages that a PiW3D code path never calls.

| Source | Third-party imports and distribution names | Reason |
|---|---|---|
| `pretrain.py`, `train_pose.py`, `utils.py` | `yaml` (`PyYAML`), `numpy`, `torch`, `torchvision`, `tensorboard`, `sklearn` (`scikit-learn`) | Entry-point, logging, model, and metric imports |
| `feeder/person_in_wifi_3d.py` | `scipy`, `cv2` (`opencv-python`), `pandas`, `h5py`, `pywt` (`PyWavelets`), plus NumPy/Torch | All are imported eagerly, although several are unused by the current PiW3D loader |
| `model/model.py` | `timm`, `einops`, NumPy, Torch, torchvision | MAE/ViT and pose-decoder definitions |
| Eager `feeder.mmfi` and `feeder.wipose` imports | SciPy, OpenCV, scikit-learn, pandas, h5py, PyWavelets, and `mat73` | `mat73` is specifically pulled in by `feeder/wipose.py` even for a PiW3D run |
| Eager MetaFi/HPE-Li imports in `train_pose.py` | Torch, torchvision, OpenCV, and `xxsubtype` | `xxsubtype` is an interpreter-provided CPython module, not a PyPI distribution to install |
| `data/person_in_wifi_3d/process.py` | h5py, PyWavelets, Torch, NumPy | Raw PiW3D preprocessing; h5py and PyWavelets perform the file/wavelet work |

Standard-library imports need no requirements entries. Commented-out
`torch_geometric` imports are not dependencies. No additional unique third-party
distribution was found in the current MM-Fi, WiPose, MetaFi, or HPE-Li modules.

### 1. Required baseline dependencies

Proposed dedicated environment: `compwifi-piw3d-py311`, using 64-bit CPython
3.11.11 on Windows. Python 3.11 is preferred over 3.12 for this legacy stack; no
claim is made that 3.12 is unsupported.

Use the official PyTorch CUDA 12.1 wheel index with the version-matched pair
below. The official archive lists this pair and index for Linux and Windows:
https://pytorch.org/get-started/previous-versions/ . The RTX 3080 Laptop GPU must
still be checked for compute capability 8.6 and an `sm_86` kernel after install.
A separate system CUDA Toolkit installation is not part of this plan.

Planned `requirements/torch-cu121-py311.txt` (installed with the CUDA index
specified in the command, not from default PyPI):

```text
torch==2.5.1
torchvision==0.20.1
```

Planned `requirements/piw3d-py311.txt`:

```text
numpy==1.26.4
scipy==1.14.1
scikit-learn==1.5.2
opencv-python==4.9.0.80
pandas==2.2.3
h5py==3.12.1
PyWavelets==1.8.0
mat73==0.65
timm==0.9.16
einops==0.8.0
PyYAML==6.0.2
tensorboard==2.18.0
```

These are exact direct-dependency candidate pins, not yet a resolved transitive
lock file. NumPy 1.26.4 is intentionally proposed instead of NumPy 2.x to reduce
legacy binary/API risk. Dependency resolution and Windows/Python wheel availability
have not been executed or verified.

### 2. Optional dependencies for other datasets/models

There are no unique additional pip distributions to list for the current MM-Fi,
WiPose, MetaFi, or HPE-Li code: their third-party imports are already in the
required baseline import closure. In particular, SciPy/OpenCV/pandas support
MM-Fi code, mat73 supports WiPose, and OpenCV is imported by MetaFi/HPE-Li.
They cannot currently be omitted from a PiW3D environment because the entry
points import those modules eagerly.

Planned `requirements/other-datasets-models-py311.txt` is therefore
intentionally comments-only and must not be used as a reason to remove packages
from the baseline manifest:

```text
# No unique additions beyond piw3d-py311.txt for the current source tree.
# Reclassify import-only packages only if a later, approved lazy-import change
# makes the PiW3D entry points independent of the other feeders/models.
```

MetaFi construction calls `torchvision.models.resnet34(pretrained=True)`, which
can request pretrained weights when that alternative model is actually created.
That is a separate download decision, not an environment dependency, and no such
model construction or download is authorized here.

### 3. Development and testing dependencies

These are not needed merely to import or run the baseline. No repository tests
currently exist, so their presence must not be reported as test coverage.

Planned `requirements/dev-py311.txt`:

```text
pytest==8.3.4
pytest-cov==5.0.0
pip-tools==7.4.1
```

`pip-tools` is proposed only for producing a fully resolved lock after the
candidate stack passes verification. It is not required for the first install.

### Python/package compatibility uncertainties

- No proposed package is confirmed here to be incompatible with Python 3.11,
  but the complete set remains unverified until the dry-run and imports below
  succeed in the new environment.
- `mat73==0.65` needs explicit verification. Its PyPI 0.65 metadata provides a
  `py3-none-any` wheel but sets no `Requires-Python` constraint; that metadata is
  not proof of Python 3.11 runtime compatibility. Source checked:
  https://pypi.org/pypi/mat73/0.65/json .
- `xxsubtype` is not a package to add with pip. It resolved as a built-in under
  the current CPython 3.14 interpreter, but availability under the proposed
  CPython 3.11 interpreter remains to be checked. Do not substitute an unrelated
  third-party package if it is missing.
- `timm==0.9.16` must be checked against the two legacy import paths
  `timm.models.layers.trunc_normal_` and `timm.models.vision_transformer.Block`,
  including the `Block(emb_dim, num_head)` constructor used by the baseline.
  This is an API compatibility check, separate from Python-version support.
- Native Windows wheels and mutual constraints for NumPy, SciPy, scikit-learn,
  OpenCV, pandas, h5py, and PyWavelets must be checked by pip's resolver. The
  `--only-binary=:all:` dry-runs below deliberately fail rather than silently
  build an unverified native package from source.
- The exact CPython 3.11/CUDA wheel installation, RTX 3080 execution, legacy
  whole-model `torch.load` behavior, and all transitive dependency versions are
  still unknown. Do not infer baseline correctness from successful imports.

### Exact creation and installation commands (do not run before approval)

After approval, first materialize the four manifest files with exactly the
contents above. Run the following in an **Anaconda Prompt**, because `conda` is
not currently on the Git Bash `PATH`. If `where conda` fails, stop and resolve
the Anaconda installation/PATH rather than falling back to Python 3.14.

```bat
where conda
conda --version
conda create --name compwifi-piw3d-py311 python=3.11.11 pip=24.3.1 --yes
conda activate compwifi-piw3d-py311
cd /d "D:\Zero-shot 3D Multi-person Pose Estimation"

python -m pip install --dry-run --only-binary=:all: --index-url https://download.pytorch.org/whl/cu121 -r requirements\torch-cu121-py311.txt
python -m pip install --index-url https://download.pytorch.org/whl/cu121 -r requirements\torch-cu121-py311.txt

python -m pip install --dry-run --only-binary=:all: -r requirements\piw3d-py311.txt
python -m pip install --dry-run --only-binary=:all: -r requirements\dev-py311.txt
python -m pip install -r requirements\piw3d-py311.txt
python -m pip install -r requirements\dev-py311.txt
```

The comments-only optional manifest has no separate installation command. Do
not run preprocessing or either training entry point as part of installation.

### Exact verification commands (after the approved installation)

```bat
where python
python --version
python -m pip --version
python -m pip check
python -c "import sys; assert sys.version_info[:2] == (3, 11), sys.version; print(sys.version)"
python -c "import yaml, numpy, scipy, sklearn, cv2, pandas, h5py, pywt, mat73, torch, torchvision, tensorboard, timm, einops, xxsubtype; print('Dependency imports OK')"
python -c "from timm.models.layers import trunc_normal_; from timm.models.vision_transformer import Block; import inspect; print('timm Block signature:', inspect.signature(Block.__init__))"
python -c "import pretrain, train_pose; print('Unmodified baseline entry-point imports OK')"
python -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('GPU:', torch.cuda.get_device_name(0)); print('capability:', torch.cuda.get_device_capability(0)); print('architectures:', torch.cuda.get_arch_list()); assert torch.cuda.is_available(); assert torch.version.cuda == '12.1'; assert torch.cuda.get_device_capability(0) == (8, 6); assert 'sm_86' in torch.cuda.get_arch_list(); x = torch.ones((1024, 1024), device='cuda'); y = x @ x; torch.cuda.synchronize(); assert y.is_cuda and y.shape == (1024, 1024); print('CUDA matrix multiplication OK')"
python -m pip freeze
```

The entry-point import command only imports modules; their `if __name__ ==
'__main__'` blocks must not execute. A successful CUDA matrix multiplication is
an environment check, not training or evidence that the baseline forward pass
works. Save the `pip freeze` output as the full lock only after every check
passes, and record any failure instead of changing pins silently.

Phase 0 environment readiness exits only when the interpreter, `pip check`, all
imports, both entry-point imports, the timm API check, and the RTX 3080 CUDA
check pass. Dataset readiness and the previously documented source/data issues
remain separate blockers.

### Read-only check performed for this proposal

In addition to static file reads/searches and the official metadata pages above,
the following command was run without importing project modules:

```bash
python -c "import importlib.util, sys; print(sys.version); print('xxsubtype:', importlib.util.find_spec('xxsubtype'))"
```

Output:

```text
3.14.0 (tags/v3.14.0:ebf955d, Oct  7 2025, 10:15:03) [MSC v.1944 64 bit (AMD64)]
xxsubtype: ModuleSpec(name='xxsubtype', loader=<class '_frozen_importlib.BuiltinImporter'>, origin='built-in')
```

This confirms only the current interpreter. No tests were available or run.

## Immediate task

The approved Step-4B-D4 bounded reproducibility audit is complete. Stop here as
requested: original Step-4B remains **BLOCKED**; D1 and D2 remain above-criterion
diagnostics only; D3 remains stopped at `D2_REPRODUCTION_MISMATCH_STOPPED` with no
valid 800-step criterion status. D4 established that two fresh unchanged seed-0
200-step D2 repeats differ from each other and from historical D2 after exactly
matching the initial checkpoint. No root cause is claimed. Do not begin Step-4C
composition controls, rerun D3, extend to 800 steps, selectively enable
deterministic mode, change the optimizer, relax thresholds, or attempt an
automatic fix. Any future diagnostic requires a separately approved, newly frozen
protocol. Scientific readiness and feasibility remain unproven; full-dataset
preprocessing, pretrained checkpoint availability, TensorBoard, `mat73`, and full
training-entry-point imports remain outside these bounded results.

## Focused-test inspection

Executed full-suite command in the dedicated environment:

```bash
VENV="/c/Users/vaska/venvs/compwifi-piw3d-py311"
PYTHONDONTWRITEBYTECODE=1 "$VENV/Scripts/python.exe" -m pytest tests/test_legacy_baseline.py -q -p no:cacheprovider
```

Actual summary:

```text
.......s                                                                 [100%]
7 passed, 1 skipped, 1 warning in 23.33s
```

The warning was an external `timm.models.layers` deprecation warning, not an
import error or test failure. Individual results:

| Test | Result | Evidence/reason |
|---|---|---|
| `test_mae_forward_backward_on_synthetic_input` | passed | Synthetic forward, finite loss, backward, and finite-gradient assertions passed |
| `test_pose_decoder_output_shape_and_finite` | passed | Pose/feature shapes and finite outputs passed |
| `test_pose_decoder_one_optimization_step` | passed | Finite loss, backward, optimizer step, and parameter-change assertions passed |
| `test_pose_gradient_accumulation_matches_effective_batch` | passed | Accumulated gradients and optimizer result matched the full effective batch; single-step identity passed |
| `test_checkpoint_save_and_load` | passed | Whole-model checkpoint round-trip and restored-decoder output checks passed |
| `test_constant_csi_normalization` | passed | Constant amplitude mapped to zero; phase normalization and finite-output checks passed |
| `test_pretrained_checkpoint_path_is_consistent` | passed | Pose YAML path matched the pretraining output path |
| `test_real_training_sample_preprocessing_and_load` | skipped | `train_data/train_data_list.txt` is absent; no `test_data` was read |

There were no remaining import-related failures. No changed-code assertion
failed, so no post-install repair, affected-test rerun, or second full-suite
rerun was necessary. No source or test file was modified during verification.

### Minimal dependencies for this focused suite

The tests directly require NumPy, PyTorch, pytest, and PyYAML. Importing
`model.model` additionally requires `timm`, `einops`, and torchvision. Importing
`feeder.person_in_wifi_3d` eagerly requires SciPy, OpenCV, scikit-learn, pandas,
h5py, and PyWavelets. The focused suite does not import `mat73` or TensorBoard.

The environment was created outside the repository from the existing 64-bit
Python 3.11.0 interpreter at `C:\Users\vaska\venvs\compwifi-piw3d-py311`.

### Executed installation commands

```bash
PY311="/c/Users/vaska/AppData/Local/Programs/Python/Python311/python.exe"
VENV="/c/Users/vaska/venvs/compwifi-piw3d-py311"
"$PY311" -m venv "$VENV"
"$VENV/Scripts/python.exe" -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
"$VENV/Scripts/python.exe" -m pip install --only-binary=:all: numpy==1.26.4 pytest==8.3.4 PyYAML==6.0.2 timm==0.9.16 einops==0.8.0 scipy==1.14.1 opencv-python==4.9.0.80 scikit-learn==1.5.2 pandas==2.2.3 h5py==3.12.1 PyWavelets==1.8.0
```

Environment creation succeeded. PyTorch installation initially failed as recorded
below; after the separately approved pip upgrade, the unchanged PyTorch command
and then the remaining focused-dependency command both completed with exit code
0. The second command replaced initially resolved NumPy 2.4.6 with approved NumPy
1.26.4. Pip installed the required transitive dependencies.

### Executed post-install verification commands

These commands ran from the repository root:

```bash
VENV="/c/Users/vaska/venvs/compwifi-piw3d-py311"
"$VENV/Scripts/python.exe" --version
"$VENV/Scripts/python.exe" -m pip --version
"$VENV/Scripts/python.exe" -m pip check
"$VENV/Scripts/python.exe" -m pip freeze
"$VENV/Scripts/python.exe" -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
PYTHONDONTWRITEBYTECODE=1 "$VENV/Scripts/python.exe" -m pytest tests/test_legacy_baseline.py -q -p no:cacheprovider
```

Actual diagnostic output, excluding the freeze listing saved in the lockfile:

```text
Python 3.11.0
pip 24.3.1 from C:\Users\vaska\venvs\compwifi-piw3d-py311\Lib\site-packages\pip (python 3.11)
No broken requirements found.
torch: 2.5.1+cu121
CUDA build: 12.1
CUDA available: True
GPU: NVIDIA GeForce RTX 3080 Laptop GPU
```

`pip freeze` completed successfully. Its complete 43-distribution output is saved
verbatim in `requirements/focused-test-py311-lock.txt`, the exact package set used
by the successful suite above.

### Approved setup attempt: actual outcome

The exact environment-creation command completed successfully with exit code 0:

```bash
PY311="/c/Users/vaska/AppData/Local/Programs/Python/Python311/python.exe"
VENV="/c/Users/vaska/venvs/compwifi-piw3d-py311"
"$PY311" -m venv "$VENV"
```

The exact approved PyTorch installation command was attempted three times:

```bash
"$VENV/Scripts/python.exe" -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
```

- First attempt: failed with exit code 2 after a network read timeout while
  downloading `torch-2.5.1+cu121-cp311-cp311-win_amd64.whl` (about 0.3 of 2.4 GB
  received). No package installation completed.
- Second attempt: downloaded the 2.4 GB torch wheel and 6.1 MB torchvision wheel,
  then failed with exit code 1. Pip 23.2.1 discarded every available
  `typing_extensions>=4.8.0` candidate with `inconsistent Name: expected
  'typing-extensions', but metadata has 'typing_extensions'`, ending with
  `ERROR: No matching distribution found for typing-extensions>=4.8.0`.

After separate user approval, the minimal unblock below succeeded with exit code
0 and installed pip 24.3.1:

```bash
"$VENV/Scripts/python.exe" -m pip install pip==24.3.1
```

The unchanged PyTorch command then succeeded on its third attempt with exit code
0, followed by the unchanged remaining focused-dependency command with exit code
0. Post-install diagnostics and the full suite completed as recorded above, and
`requirements/focused-test-py311-lock.txt` now contains the tested freeze output.

No dataset content was downloaded, read, or preprocessed, and no training or
evaluation was started. TensorBoard, `mat73`, and training-entry-point imports
were deliberately not verified. The environment is not claimed training-ready.

## Legacy baseline repair checks

Before the dedicated environment was available, the complete focused suite in the
old system interpreter produced `5 failed, 2 passed, 1 skipped` because of missing
imports. The dedicated-environment rerun now produces `7 passed, 1 skipped` as
itemized above. Earlier, the new accumulation test was also run alone:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_legacy_baseline.py::test_pose_gradient_accumulation_matches_effective_batch -q -p no:cacheprovider
```

Actual result: `1 passed`. It compares two scaled microbatch backward passes
with one full-effective-batch backward pass and optimizer step, and verifies the
`steps_per_update=1` path is the identity.

Syntax was checked without importing project dependencies or writing bytecode:

```bash
python -c "from pathlib import Path; paths = ['model/model.py', 'feeder/person_in_wifi_3d.py', 'data/person_in_wifi_3d/process.py', 'train_pose.py', 'utils.py', 'tests/test_legacy_baseline.py']; [compile(Path(path).read_text(encoding='utf-8'), path, 'exec') for path in paths]; print('Syntax OK:', ', '.join(paths))"
```

Actual result: `Syntax OK` for all six files. `git diff --check` completed with
no output. Those earlier checks ran no preprocessing, training entry point, or
evaluation; the later approved environment attempt is recorded separately above.

### Normalization numerical behavior

Amplitude and phase are still normalized independently with the legacy min-max
formula when nonconstant. If either component is constant, that component now
returns a same-shaped zero array. Previously this case divided by zero and
produced NaNs with a runtime warning. No additional NaN/Inf sanitization was
introduced.

### Gradient accumulation behavior

For `steps_per_update > 1`, each microbatch mean loss is divided by
`steps_per_update` before backward, so equal-sized microbatches reproduce the
full-effective-batch mean gradient instead of a sum of means. For
`steps_per_update == 1`, the helper returns the original loss tensor unchanged,
preserving the legacy single-device-batch path.

## Next recommended task

Phase B ("Step 5") ended **YELLOW-POSITION** (`docs/STEP_5_REPORT.md`). Per the frozen protocol the step
stops here, and the next move is the user's choice between the plan's two options:
(a) Phase C ("Step 6", `docs/NEXT_PHASE_PLAN.md` section 5) under an explicitly disclosed **official-like
temporal protocol** (same groups, held-out frame blocks), with group-disjoint results always reported
alongside; the Step-5 encoder candidate would be R2's `simple_cnn` (or R1, with the z-score input).
(b) A data-scaling run on all 23,874 single-person internal-train IDs (~5-6 GB preprocessing plus full
training; ask first). Caveat: it adds frames, not groups, and the 51 training groups are the likely limit
on G6. The Phase C protocol (`docs/STEP_6_PROTOCOL.md`) must itself be approved before any Phase C
training. Also open: push/merge `research/step5-single-person-baseline`.

## Progress log

- 2026-11-10: Completed **Step 5b** (`docs/STEP_5B_PLAN.md`); verdict **INCONCLUSIVE**. Branch
  `research/step5b-pose-signal` from `11fad75`, worktree `.claude/worktrees/charming-napier-69e92f`,
  `--repo-root` pointing at the main checkout. The user approved the dense preprocessing and the training
  runs in the kickoff message. New files: `analyze_step5_recording_effects.py`, `step5b_config.py`,
  `step5b_subset.py`, `make_step5b_data.py`, `feeder/piw3d_step5b.py`, `step5b_pose_signal.py`,
  `summarize_step5b.py`, `tests/test_step5b_{recording_effects,subset,runner}.py`,
  `docs/STEP_5B_{PLAN,PROTOCOL,REPORT}.md`. No protected file was edited. Order:
  1. Premises: 17/18 reproduce; lag-8 autocorrelation is 0.142 vs 0.17 (disclosed; it strengthens the
     premise).
  2. `freeze_v2`: all view counts and the 9 validation recordings exactly as planned.
  3. Code + protocol committed (`6a2a97b`), then the protocol was frozen at 2026-11-10T11:46:07Z.
  4. `data_v2` preprocessing: 22,754/0 failures, 1,742 s.
  5. Gate S pass (smoke D1-D4; overfit D2 0.106, D4 0.051).
  6. Stage 1: T1 D1 FAIL 1.109, D2 0.853, D3 0.858, D4 0.905; X1 fails for all (0.995/0.998/0.993/1.074).
  7. Selection: D1 (tie band; simplest).
  8. Stage 2 D1 seeds 0-2: T1 FAIL (1.103/1.115/1.103), X1 FAIL (1.108/1.097/1.093).
  9. Verdict: INCONCLUSIVE.
  Every reload was exact and the source hashes still verify. Nothing pushed or merged; Phase C not started.
  Post-hoc addendum (user-approved): `posthoc_step5b_dense_check.py` ran D3 stage-2 seeds 0-2 into
  `posthoc_d3/`. T1 passes 0.837/0.839/0.846 and X1 fails 1.155/1.189/1.156 (YELLOW-TRACKS pattern, post-hoc).

- 2026-11-10: Completed Phase B ("Step 5") of `docs/NEXT_PHASE_PLAN.md` only; verdict
  **YELLOW-POSITION**. Executed in worktree `.claude/worktrees/phase-b-execution-ce4d08` on new branch
  `research/step5-single-person-baseline` (from `a47b47a`). There are no junctions this time: every new
  script takes `--repo-root "J:/Zero-shot 3D Multi-person Pose Estimation"`, so data and artifacts are
  read and written in the main checkout's git-ignored `data/`. New files: `step5_config.py`,
  `step5_subset.py`, `make_step5_data.py`, `feeder/piw3d_step5.py`, `model/research_models.py`,
  `step5_single_person_baseline.py`, `summarize_step5.py`, `tests/test_step5_{subset,models,feeder,runner}.py`,
  `docs/STEP_5_PROTOCOL.md`, `docs/STEP_5_REPORT.md`. No protected file was edited. Order:
  1. Subset freeze: 51/8 groups; 3,264 / 804 / 512 IDs; 510 guard IDs.
  2. Preprocessing: 4,580/0 failures, 445 s.
  3. Code + protocol committed (`3a71989`), then the protocol was frozen at 2026-11-10T08:50:03Z.
     Working-tree files were normalised to LF first so the frozen hashes equal the git blobs.
  4. Smoke G1 pass (R1-R4, CUDA).
  5. Overfit G2 pass: R1 ratio 0.076, R3 ratio 0.066.
  6. Seed 0 for R1-R4: all pass G3-G5 and fail G6 on the centroid-relative criterion.
  7. Selection: R2 (group_dev 0.6478; R1 0.6589 is 1.7% higher, outside the 1% tie).
  8. R2 seeds 1-2, then the verdict.
  Every run's reloaded state dict reproduced its final predictions bit-exactly. Total training time
  about 50 min. Official test data was never opened, and no multi-person sample was used.

- 2026-11-10: Completed Phase A of `docs/NEXT_PHASE_PLAN.md` only (no new science; Phase B/C not
  started). Verified stash `9585ba9375a2218a9d3af5ae52c6c640cb247d32` exists and the main checkout had
  only untracked files; found branch `consolidate/step3-to-step4b-r` already existing at `c494564`
  with no extra commits and reused it. Applied the stash (exactly the six expected modified files:
  `.gitignore`, `data/person_in_wifi_3d/process.py`, `docs/CURRENT_TASK.md`,
  `docs/POSE_SPLIT_METRIC_AUDIT.md`, `docs/PROJECT_CONTEXT.md`, `model/model.py`); the stash entry
  was left in place. Executed in the git worktree `.claude/worktrees/phase-a-execution-d4e33c` instead of
  the main checkout: the 43 untracked source/doc/test files were copied from the main checkout (originals
  untouched) and `data/_artifacts` and `data/_fixtures` were exposed as NTFS junctions to the main
  checkout's directories (git-ignored), so relative paths resolve; diagnosis outputs therefore land in the
  main checkout's ignored `data/_artifacts/step4b_r_diagnosis/`. Added `.anchor/`, `archive-7U4NdL/`,
  `archive-AzzHuZ/`, `archive-GzqAl5/`, `archive-KjVUak/` to the shared `.git/info/exclude` (the plan
  named three; two more archive directories also exist). `docs/NEXT_PHASE_PLAN.md` was copied from the
  main checkout and is byte-identical (`cmp`) to
  `C:\Users\vaska\.claude\plans\c-users-vaska-downloads-zero-shot-3d-mu-fancy-whistle.md`.
  **Test gate:** the first full-suite run gave `1 failed, 73 passed`
  (`test_d4_protected_records_include_d3_stop_artifacts_and_match`). Cause: the protected-record check
  hashes `docs/DATA_AUDIT.md`, `docs/POSE_SPLIT_METRIC_AUDIT.md`, and `docs/PROJECT_CONTEXT.md` as stored in
  the original Windows working copy with CRLF endings, while git stores LF blobs and this checkout wrote
  LF; converting each to CRLF reproduces all three expected SHA-256 values exactly. After the user chose
  to continue, a narrow `.gitattributes` (`text eol=crlf` for those three files only) was added and those
  three files were rewritten with CRLF; `git status` still shows only the six expected tracked
  modifications. The full suite then gave `92 passed` (74 + 18 new). Added
  `diagnose_step4b_r_collapse.py` and `tests/test_step4b_r_collapse_diagnosis.py`; the diagnosis ran on
  real artifacts with all premises reproduced (see the Step-4B-R collapse diagnosis paragraph above and
  `docs/STEP_4B_R_COLLAPSE_DIAGNOSIS.md`). Updated this file (status block, Step-4B-R entry, next task).
  No training, preprocessing, package change, download, deletion, push, remote change, or stash drop
  occurred; protected `step4b_*` files, `model/model.py` beyond the stashed change, feeders, and dataset
  contents were not edited.

- 2026-11-08: Completed the approved Step-4B-D4 bounded reproducibility audit of
  the unchanged D2 trainable-encoder procedure. Added the D4 static/environment/
  comparison module, six focused tests, `docs/STEP_4B_D4_REPORT.md`, and ignored
  evidence under `data/_artifacts/step4b_d4/`. Before training, statically
  compared D2/D3 and recorded the environment without changing either runner or
  enabling deterministic mode. Executed two independent fresh seed-0 200-step D2
  repeats in separate roots. Both exactly matched D2's initial MPJPE and retained
  finite-loss/finite-gradient, exact 97-parameter optimizer coverage, and 97/97
  trainable plus 50/50 encoder parameter-change evidence. Repeat A measured
  step-100/200 train-mode MPJPE `0.4460887312889099`/`0.46392330527305603`;
  repeat B measured `0.49196043610572815`/`0.37702471017837524`; historical D2
  recorded `0.5151628255844116`/`0.46943652629852295`. Exact comparisons
  classified the evidence as **`repeats differ from each other`**, with the first
  A/B full-record divergence at step 1 in `max_abs_encoder_gradient` and first
  A/B loss divergence at step 3. The D4 focused tests passed `6 passed,
  1 warning`; the complete fixture-backed suite passed **68 passed, 0 skipped**;
  all 40 protected historical records remained unchanged. No D3 rerun, 800-step
  extension, deterministic-setting change, optimizer/criterion/data change,
  development/test access, epoch training, package change, download, deletion,
  commit, push, remote change, or automatic fix attempt occurred. Original
  Step-4B remains **BLOCKED**; D1/D2 remain above-criterion diagnostics; D3
  remains stopped with no valid 800-step status; no root cause is claimed.

- 2026-11-07: Stopped the approved Step-4B-D3 trainable-encoder extended-step
  diagnostic at its frozen D2 reproduction gate. Added the separate D3 protocol,
  D3-only runner, and five focused tests; the focused file reported `5 passed,
  1 warning`, and the complete fixture-backed suite reported **62 passed,
  0 skipped**, with the same external `timm` warning. The D3 verifier confirmed
  the protected original Step-4B, D1, and D2 records, then wrote
  `data/_artifacts/step4b_d3/step4b_d3_freeze.json` before model construction.
  The fresh seed-0 trainable-encoder run reproduced the D2 initial train-mode
  MPJPE exactly, but did not reproduce D2 at step 100 or 200
  (`0.40826788544654846` versus `0.5151628255844116`, and
  `0.3430270552635193` versus `0.46943652629852295` stored units). The runner
  exited `3` with `D2_REPRODUCTION_MISMATCH_STOPPED` and saved the required stop
  artifact; no valid 800-step D3 result, criterion status, checkpoint trajectory,
  predictions, or final state-dict exists. Original Step-4B remains **BLOCKED**;
  D1/D2 remain above-criterion diagnostics. No development/test access, epoch
  training, package change, preprocessing, download, deletion, commit, push,
  remote change, rerun, or automatic fix attempt occurred.

- 2026-11-07: Completed the approved Step-4B-D2 encoder-trainability diagnostic.
  Before editing, a seed-0 CUDA named-parameter audit confirmed the original
  configuration directly: 50 encoder parameters (`3,390,464` values) frozen, 47
  pose-decoder parameters (`1,404,931` values) trainable, and SGD coverage exactly
  matching those 47 parameters. Added an opt-in `train_encoder` mechanism to
  `model/model.py` while preserving default legacy behavior, plus a separate D2
  runner and five focused tests covering both modes, initialization preservation,
  encoder `requires_grad`, optimizer coverage, and parameter-change reporting. The
  frozen D2 protocol changed only encoder trainability from frozen to trainable and
  preserved the corrected tiny batch, seed `0`, initialization values, data,
  preprocessing, loss, SGD optimizer, learning rate `0.01`, momentum `0.9`, batch,
  200-step budget, coordinate order, and `<=0.10` stored-unit criterion. CUDA
  training completed with finite losses/gradients; all 97 intended parameters were
  optimized exactly once, all 50 encoder gradients were nonzero at every step, and
  all 50 encoder parameters changed. Final train-mode MPJPE was
  `0.46943652629852295`, above criterion and `0.12929058074951172` worse than the
  original frozen-encoder endpoint. Saved the pre-edit audit, input verification,
  D2 freeze, per-step trajectory, checkpoints/predictions, final state_dict, result,
  and decision under ignored `data/_artifacts/step4b_d2/`; wrote
  `docs/STEP_4B_D2_REPORT.md`. The complete fixture-backed focused suite reported
  **57 passed, 0 skipped**, with one external `timm` deprecation warning. Protected
  original Step-4B/D1 source/result hashes remained unchanged. No development
  samples, official test data, epoch training, package change, preprocessing,
  download, deletion, commit, push, or remote change occurred. Original Step-4B
  remains BLOCKED; scientific feasibility remains unproven.

- 2026-11-07: Completed the approved Step-4B-D1 bounded under-training diagnostic
  after the failed Step-4B tiny-overfit gate. Added a separate D1 runner and tests
  that verify the corrected `freeze_v2`/`pilot_data_v2` source records, exact
  eight-sample CSI/keypoint pairing, unchanged original `BLOCKED` decision, and a
  D1-only freeze before model construction. The diagnostic used the same data,
  seed, model, preprocessing, loss, SGD optimizer, learning rate, momentum, batch,
  device-selection mode, coordinate order, and `<=0.10` stored-unit criterion;
  only the optimizer budget changed, from 200 to exactly 800 steps. CUDA training
  completed with finite losses and gradients; the initial and step-200 MPJPE
  exactly reproduced `4.920130252838135` and `0.34014594554901123`, while final
  step-800 MPJPE remained `0.32848620414733887`. The trajectory plateaued around
  `0.33–0.34`, predictions did not collapse toward the fixed-batch or train mean,
  and error concentrated in coordinate index 2 and joint indices 8, 11, 9, 1, and
  10. Saved checkpoint JSON/NPZ evidence, a verified final state-dict reload, and
  ten labelled PNG plots under `data/_artifacts/step4b_d1/`; wrote
  `docs/STEP_4B_D1_REPORT.md`. The complete focused suite subsequently reported
  **52 passed, 0 skipped**, with the same external `timm` deprecation warning. No
  development samples, official test data, epoch training, package change,
  preprocessing, download, deletion, commit, push, or remote change occurred.
  Original Step-4B remains BLOCKED; scientific feasibility remains unproven.

- 2026-11-07: Completed the approved Step-4B bounded sanity gate up to its frozen
  stop rule. Added the frozen Step-4B configuration, deterministic seed-0 subset
  selector, bounded pilot preprocessing/validation, separate Q=1 single-person
  loader and research entry point, and focused tests. An initial freeze exposed a
  filename-count selector error; its shape validation rejected the affected IDs,
  the rule was corrected to `S?1_*`, and the superseding `freeze_v2` subset was
  used. Corrected preprocessing validated 2,048 internal-train and 512
  internal-development samples with zero failures and no source/test-data access.
  The frozen CUDA overfit check completed 200 finite steps, reducing stored-unit
  MPJPE from `4.920130252838135` to `0.34014594554901123`, but failed the
  predeclared `<=0.10` absolute criterion. Recorded `BLOCKED` in
  `data/_artifacts/step4b_pilot/decision.json` and `docs/STEP_4B_REPORT.md`; no
  epoch run, development metrics, mean-pose comparison, compositional training,
  package change, download, deletion, commit, push, or remote change occurred. The
  complete focused suite reported 48 passed and 0 skipped with one external `timm`
  deprecation warning. Scientific feasibility remains unproven.

- 2026-11-06: Completed only the approved Step-4A infrastructure task. Added the
  deterministic candidate-group-disjoint manifest generator, generated ignored
  internal train/development manifests from all 89,946 official training IDs only,
  and independently verified zero ID/group overlap and zero immediate-next-frame
  crossings. Added the flagship unlabelled CSI-only loader with no keypoint,
  pose/count, raw-ID, or filename-count exposure. Added corrected research metrics
  in a new module without changing legacy `utils.py`, including Hungarian matching,
  millimetre conversion exactly once, guarded PCK, count/miss/duplicate/spurious
  detection, person precision/recall/F1, root error, and unequal-batch-safe
  aggregation. The complete focused suite with the existing two-frame fixture
  produced 33 passed and 0 skipped, with one external `timm` deprecation warning.
  Updated `docs/STEP_4A_REPORT.md` and this handoff. Scientific feasibility,
  accuracy, novelty, and readiness remain unproven; no preprocessing beyond the
  existing fixture, training, evaluation, pilot, package change, download,
  deletion, official test access, commit, push, or remote change occurred.

- 2026-11-06: Completed only the approved bounded Step-3 gate. Corrected shared
  split/axis/unit/comparator documentation; added reusable explicit-ID bounded
  preprocessing and its two passing synthetic regression tests; generated the
  ignored two-frame fixture from official training IDs `S11_01_10` and
  `S11_01_11`; verified finite `(3,90,20)` amplitude/phase and `(3,180,20)` exact
  concatenation; and reran the focused suite with 8 passed, 0 skipped, and one
  external `timm` deprecation warning. A separate from-scratch CUDA smoke ran
  exactly one forward/backward/SGD step with finite loss `3.750966311`, 47 finite
  nonzero-gradient tensors, and 47 changed nonzero-gradient parameters. Added the
  design-only `docs/RESEARCH_PROTOCOL.md`. The bounded engineering/protocol gate
  is GREEN; scientific feasibility, accuracy, novelty, and readiness remain
  unproven. No full preprocessing/training, evaluation, official test-array read,
  package change, download, deletion, commit, push, or pilot occurred.

- 2026-11-05: Completed the read-only pose/split/metric audit. Parsed all 89,946
  train and 7,824 test list IDs without reading test arrays; validated 90 official
  training poses; measured coordinates and major-limb lengths; generated and
  inspected six equal-scale skeleton figures. Axis 2 appears vertical and metre
  scale is plausible, but official conventions remain unproven. Train/test frame
  ranges are 10..599/301..349, rejecting a literal first-550/last-50 interpretation;
  all 195 test candidate temporal groups overlap train. Confirmed stored-unit
  MPJPE/PA-MPJPE, inter-elbow PCK normalization under the provisional ordering, and
  count-trimming/slot-pairing evaluation limitations. Rating is YELLOW. Added
  `audit_piw3d_pose_and_split.py`, `docs/POSE_SPLIT_METRIC_AUDIT.md`, and six PNGs;
  updated this handoff. No preprocessing, training, evaluation, metric change,
  dataset modification, commit, or push occurred.

- 2026-11-05: Completed only the requested read-only real-data audit using the
  dedicated Python 3.11 environment. Confirmed root
  `data/person_in_wifi_3d/wifi_dataset`, 89,946 train and 7,824 test unique IDs,
  complete listed raw CSI/keypoint pairing, and one-/two-/three-person coverage in
  12 samples per split. Sampled raw CSI is `(20, 30, 3, 3)` compound float64 and
  interpreted as `(3, 3, 30, 20)`; poses are float64 `(people, 14, 3)`. No sampled
  NaN/Inf was found. Processed arrays are entirely absent, so normalization and
  concatenation checks remain unmeasured. Documented 215/195 candidate sequence
  groups, 2,032 train IDs without immediate successors, 195 train/test candidate
  key overlaps, unknown pose conventions/metadata, YELLOW readiness, and exact
  commands in both audit/status documents. No preprocessing, training, evaluation,
  source change, or Step 4 work was performed.

- 2026-11-03: Completed the read-only Phase 0 repository/environment
  inspection and updated this handoff. No experiments were run and no
  implementation task had begun.
- 2026-11-03: Traced the complete unmodified Person-in-WiFi-3D entry-point
  import closure and proposed a Python 3.11.11/CUDA 12.1 dependency plan with
  exact candidate pins and unexecuted creation/verification commands. Only this
  handoff changed; no packages, environments, data, model code, or training
  processes were installed, downloaded, modified, or started.
- 2026-11-03: Implemented the standalone read-only Person-in-WiFi-3D audit,
  generated a factual absent-dataset report, and verified syntax, CLI help,
  clean placement/rerun instructions, and the dataset-root write guard. No fake
  data, preprocessing, package installation, or training was used; real-data
  audit results remain unavailable.
- 2026-11-04: Implemented the scoped legacy decoder, checkpoint-path,
  preprocessing, accumulation, normalization, and loader repairs plus focused
  tests. Recorded the constant-to-zero normalization change. Syntax, path, and
  diff checks passed; the real-data test skipped and model/normalization runtime
  tests remained blocked by missing dependencies. No full training was started.
- 2026-11-04: Reopened Step 3, inspected every focused test result, identified
  the minimal test import closure, and documented exact approval-gated Python
  3.11 environment commands without installing anything. Added and passed a
  synthetic effective-batch gradient-accumulation test while preserving the
  `steps_per_update=1` identity path.
- 2026-11-04: Created the approved dedicated Python 3.11 virtual environment.
  The first PyTorch wheel download timed out; the retry downloaded the wheels
  but failed on pip 23.2.1's `typing-extensions`/`typing_extensions` metadata
  name handling. Corrected the static `csi_ap` audit statement without claiming
  real-data verification. No data access, preprocessing, training, evaluation,
  TensorBoard import, or training-entry-point import was performed.
- 2026-11-04: After separate approval, upgraded the dedicated environment to pip
  24.3.1 and completed both unchanged approved dependency installations. Recorded
  Python 3.11.0, torch 2.5.1+cu121, CUDA 12.1 availability, and the RTX 3080 Laptop
  GPU; `pip check` passed. The full suite produced 7 passed, 1 skipped, and one
  external deprecation warning, with no import failures or assertion repairs.
  Saved `requirements/focused-test-py311-lock.txt`. Step 3 remains open for the
  unavailable real-data test; Step 4 and training were not begun.
- 2026-11-04/05: Downloaded the full public Wiception3D dataset from the approved
  Kaggle URL (public, authentication-free; no packages installed; existing curl
  and standard-library Python only). A single-stream curl download died on a
  connection reset and its retry throttled, and a bash 10-chunk downloader
  stopped on Git Bash fork failures, so the download was completed with a
  stdlib-Python threaded resumable-chunk downloader. Archive
  `data/_downloads/wiception3d_archive.zip` verified at exactly 16,809,368,215
  bytes; zip central directory valid (293,312 entries). Extracted without
  overwrites into `data/person_in_wifi_3d/`: 293,312 files, 0 skipped, 18.69 GB;
  actual PIW3D_DATA_ROOT is `data/person_in_wifi_3d/wifi_dataset` containing
  `train_data` (csi/keypoint/smpl, 89,946 each + list) and `test_data`
  (csi/keypoint/smpl, 7,824 each + list). `process.py` untouched and tracked;
  `.gitignore` gained `data/_downloads`; `git check-ignore` verified. No
  preprocessing, audit, training, or evaluation was started.
