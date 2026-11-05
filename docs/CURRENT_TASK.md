# Current Task and Project Status

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
- `csi_ap`, `csi_amplitude`, and `csi_phase` are absent for every listed ID in
  both splits. The existing feeder therefore cannot yet consume the extracted
  data. Constant normalization cases and equality between `csi_ap` and
  concatenated amplitude/phase are unmeasured; no preprocessing was run.
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
passed. Axis 2 appears vertical and stored units are plausibly metres, but official
units, joint ordering, coordinate frame, and candidate temporal group semantics
remain unproven. Observed train/test frame ranges are 10..599 and 301..349; these
reject the literal 1-based first-550/last-50 interpretation. There are 215 train
and 195 test candidate temporal groups, with all 195 test groups shared with train
and zero exact listed-ID overlap. MPJPE/PA-MPJPE retain stored units; PiW3D PCK
uses joints 6 and 4 (the elbows under the provisional ordering); ground-truth-count
trimming and slot-wise pairing hide count/miss/duplicate behavior. Full evidence is
in `docs/POSE_SPLIT_METRIC_AUDIT.md`.

Step 3 baseline smoke verification remains open: raw training data is present, but
feeder-required `csi_ap` is absent and the conditional real-sample
preprocessing/load test was not rerun. No pretrained checkpoint is present. No
baseline training, dataset preprocessing, evaluation, or experiment was started.
This is a focused-test environment only, not a training-readiness claim:
TensorBoard, `mat73`, and the training-entry-point imports have not been separately
verified. Do not begin Step 4.

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

## Files changed

- Read-only inspection: no files changed.
- Environment proposal and earlier status handoff: `docs/CURRENT_TASK.md` only.
- Dataset audit tooling: added `audit_person_in_wifi_3d.py` and
  `docs/DATA_AUDIT.md`; updated `docs/CURRENT_TASK.md`.
- Real-data audit (this task): updated `docs/DATA_AUDIT.md` and
  `docs/CURRENT_TASK.md` only. No source, test, or dataset file was changed.
- Legacy baseline repair: changed `model/model.py`, `train_pose.py`, `utils.py`,
  `config/person_in_wifi_3d/pose_config.yaml`,
  `data/person_in_wifi_3d/process.py`, and `feeder/person_in_wifi_3d.py`; added
  `tests/test_legacy_baseline.py`; updated this handoff.
- Focused-test setup: added `requirements/focused-test-py311-lock.txt`; updated
  `docs/CURRENT_TASK.md` and the static `csi_ap` statement in
  `docs/DATA_AUDIT.md`. No source or test changes were needed after installation.
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
- The complete focused suite ran successfully: 7 passed, 1 skipped, and one
  external `timm` deprecation warning. Every test is itemized below.
- Synthetic forward/backward, optimizer update, checkpoint save/reload, gradient
  accumulation, and normalization checks passed. No changed-code assertion
  failed, so no post-install source repair was made.
- The earlier real-training-sample test skipped when the training list was
  absent. It was not rerun during this audit, and the feeder-required `csi_ap`
  files remain absent. No tiny-overfitting or full training run was performed.
- The real-data audit itself completed successfully: full-list existence and
  pairing checks passed, and all 24 sampled raw CSI plus 24 sampled keypoint
  files loaded with finite values. No pytest suite was rerun because this task
  changed documentation only.
- No preprocessing, pretraining, pose training, evaluation, or experiment was
  run. No experimental results or baseline performance numbers exist yet.
- Training-entry-point imports, TensorBoard, and `mat73` were not verified; the
  environment must not yet be described as training-ready.

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
- The Person-in-WiFi-3D loader expects `train_data/csi_ap/*.npy` and
  `test_data/csi_ap/*.npy`; those and the separate amplitude/phase arrays are
  absent for every listed ID. `data/person_in_wifi_3d/process.py:99-102` contains
  an enabled `csi_ap` save, but it was not run and its real-data output remains
  unverified. Constant normalization frequency and concatenation equality are
  therefore unknown.
- Candidate sequence keys produce 215 train groups and 195 test groups. Train has
  2,032 IDs without an immediate successor on the unfiltered list. All 195 test
  candidate keys overlap train keys, while exact listed-ID overlap is zero.
  Filename-grouped train/development splitting is provisionally possible, but no
  recording-, subject-, or environment-disjoint claim is supported yet.
- Pose units, joint ordering, coordinate axes/origin/frame, and candidate temporal
  group semantics remain unproven. A separate audit validated 90 training-pose
  structures and generated/inspected six skeleton figures; axis 2 appears vertical
  and metre scale is plausible from body proportions. Overall readiness is YELLOW;
  see `docs/DATA_AUDIT.md` and `docs/POSE_SPLIT_METRIC_AUDIT.md`.

## Newly discovered environment facts

- The system interpreter remains Python 3.14.0 with PyTorch 2.11.0+cu128; the
  dedicated focused-test environment separately uses Python 3.11.0, pip 24.3.1,
  PyTorch 2.5.1+cu121, and torchvision 0.20.1+cu121.
- In the dedicated environment, `pip check` found no broken requirements, CUDA
  12.1 availability was true, and the GPU name was NVIDIA GeForce RTX 3080
  Laptop GPU. This diagnostic did not run a CUDA computation or training.
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
- Full training-entry-point imports, `mat73`, TensorBoard, GPU kernel execution,
  and real-data compatibility remain unverified.

## Current errors, blockers, and uncertainties

### Legacy findings resolved in this task

1. `ViT_Pose_Decoder.forward()` used undefined `feature.device`; it now uses
   `x.device`. Synthetic output, optimization, and checkpoint tests now pass.
2. The PiW3D pose checkpoint path omitted `all-person`; it now matches the
   pretraining output path, and its consistency test passes.
3. The processor wrote amplitude and phase but not the feeder's required
   `csi_ap`; it now saves the existing combined `(3, 180, 20)` array as well.
   Real-data output compatibility remains unverified.
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

- The focused dependency import failures are resolved; all seven runnable tests
  passed in the dedicated environment's earlier run.
- Raw train/test content is now present and audited, but no pretrained checkpoint
  or feeder-required `csi_ap` is available. The real-sample test was not rerun,
  so Step 3 remains open.
- TensorBoard, `mat73`, and training-entry-point imports remain separately
  unverified. No training-readiness claim is made.

### Still unknown

- Processed amplitude/phase/`csi_ap` shapes, ranges, constant-case frequency,
  concatenation equality, and compatibility with the real feeder.
- Full-dataset file readability and NaN/Inf status beyond the bounded samples.
- Official pose units, joint ordering, and coordinate conventions; visualization
  has been generated and inspected, but it does not prove those conventions.
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

The requested read-only pose, metric, and filename-semantics audit is complete with
a YELLOW rating. Results, commands, limitations, and six inspected visualizations
are documented in `docs/POSE_SPLIT_METRIC_AUDIT.md`; the earlier data audit remains
in `docs/DATA_AUDIT.md`. Stop after the audit as requested. Do not preprocess,
train, evaluate, or begin Step 4 automatically. Before any later training,
separately approve and verify the remaining entry-point dependencies and imports,
including TensorBoard and `mat73`.

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

First resolve official pose units/joint ordering/coordinate conventions and the
meaning of candidate temporal groups; the observed indices require checking the
actual split-selection convention rather than assuming first-550/last-50. Then,
only with separate approval, generate the required processed arrays and rerun the
real training-sample load test without using official `test_data` for training or
model selection. Before any training, verify TensorBoard, `mat73`, both entry-point
imports, and all required smoke checks. Keep Step 3 open until those checks pass;
do not begin Step 4 or full training automatically.

## Progress log

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
