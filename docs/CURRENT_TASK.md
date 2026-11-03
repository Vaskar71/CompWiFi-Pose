# Current Task and Project Status

## Current phase

Phase 0: Repository and environment inspection.

The read-only inspection is complete. Phase 0 remains open because a dedicated
runnable environment and a verified dataset location are not configured. No
baseline, smoke test, preprocessing run, training run, or experiment has been
completed.

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
- Queried the current Git commit, local dataset directories, Python/PyTorch/
  CUDA availability, GPU, Conda availability, and installed package metadata.
- Delivered the read-only Phase 0 report in the preceding response.

## Files changed

- Read-only inspection: no files changed.
- This handoff update: `docs/CURRENT_TASK.md` only.
- `docs/EXPERIMENT_LOG.md` was not created because no experiments were run.

## Repository and Git findings

- Current commit: `2fbbef271454e6fbb95b1a0163b8cdbb96163838`.
- Branch: `main`, tracking `origin/main`.
- No tracked modifications were reported during the inspection.
- `AGENTS.md` and `docs/` were untracked before and after the inspection.
- No dependency manifest or test files were found.
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

### Final Git status

```bash
git status --short --branch
```

Output:

```text
## main...origin/main
?? AGENTS.md
?? docs/
```

## Actual test and execution outcomes

- No tests were run; no test files were found during the repository search.
- No preprocessing, pretraining, pose training, evaluation, or full training
  command was run.
- No model forward pass, loss calculation, backward pass, optimizer update,
  checkpoint save/reload, or tiny-overfitting test was run.
- No experimental results or baseline performance numbers exist yet.
- The command failures and static code findings below must not be represented
  as successful runtime reproductions.

## Newly discovered dataset facts

- The configured local dataset directories do not contain dataset content;
  they contain only preprocessing scripts.
- Person-in-WiFi-3D YAML defaults to `data/person_in_wifi_3d`, but no verified
  external dataset path is configured.
- The Person-in-WiFi-3D loader expects `train_data/csi_ap/*.npy` and
  `test_data/csi_ap/*.npy`.
- `data/person_in_wifi_3d/process.py` writes `csi_amplitude` and `csi_phase`,
  while its `csi_ap` writing block is commented out. Unless the original
  package already supplies `csi_ap`, the documented preprocessing and loader
  are incompatible.
- The loader derives person-count filtering and `person_num` from filenames
  and loaded pose arrays. This encoding has not been validated against real
  data and must not be used as a forbidden zero-label training signal.
- Actual dataset schema, MATLAB variables, tensor shapes on disk, joint order,
  coordinate units, person-count representation, and recording/subject/
  environment metadata remain unknown until real data is audited.

## Newly discovered environment facts

- The current interpreter is Python 3.14.0 with PyTorch 2.11.0+cu128.
- PyTorch reports CUDA availability on one 16 GB RTX 3080 Laptop GPU.
- The current interpreter lacks several required package distributions, so it
  cannot run the repository as-is.
- The repository provides no Python, package, PyTorch, or CUDA version pins.
- Conda is unavailable on the current Bash `PATH`.
- Package/version compatibility, including `timm` APIs and whole-model
  `torch.load` behavior, remains unverified.

## Current errors, blockers, and uncertainties

### Confirmed by static code inspection

1. `model/model.py` uses undefined `feature.device` in
   `ViT_Pose_Decoder.forward()`; pose execution is expected to raise
   `NameError`, but this was not runtime-tested.
2. Person-in-WiFi pretraining writes
   `pretrain_weights/person-in-wifi-3d/all-person/pretrain_our_ratio80.pt`,
   while the pose YAML requests
   `pretrain_weights/person-in-wifi-3d/pretrain_our_ratio80.pt`.
3. The Person-in-WiFi `validation` loader reads `test_data`, and checkpoint
   selection uses its labels. Development validation and official test are not
   separated as required by the research protocol.
4. Ground-truth `person_num` trims predictions during training and evaluation;
   the current model does not predict presence or infer count.
5. The existing pretraining feeder loads pose labels and exposes padded poses
   and `person_num`; it is not a CSI-only unlabelled loader.
6. MM-Fi contains hardcoded `/root/autodl-tmp/...` paths that are incompatible
   with the current Windows layout.
7. Pose gradient accumulation does not increment `step_count`; shipped pose
   configs avoid the issue only because their effective batch fits in one
   device batch.
8. No Hungarian matching, root-localization error, exact count accuracy,
   missed/duplicate counts, or person-level precision/recall/F1 are implemented.

### Still unknown

- Whether a dataset is downloaded elsewhere and whether the public package
  matches DT-Pose preprocessing.
- Actual dataset schema, joint ordering, coordinate units, and metadata.
- Whether a dedicated compatible environment can run all entry points.
- Whether the static forward-pass and checkpoint issues are the only runtime
  blockers.
- Baseline reproduction accuracy and zero-label compositional viability.

## Immediate task

Update this handoff with the completed inspection and verified findings. Do
not begin implementation or experiments automatically.

## Next recommended task

After approval, create a small read-only Phase 1 dataset-audit script. It
should accept an explicit dataset root; record missing/corrupt files, actual
schemas, shapes, dtypes, NaN/Inf values, sample IDs, and available metadata;
check raw and preprocessed CSI compatibility; and support CSI/skeleton
visualization without modifying source data or exposing labels to a training
loader.

The audit script has not been started.

## Progress log

- 2026-11-03: Completed the read-only Phase 0 repository/environment
  inspection and updated this handoff. No experiments were run and no
  implementation task has begun.
