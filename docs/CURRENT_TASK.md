# Current Task and Project Status

## Current phase

Phase 0 environment setup and Step 3 baseline smoke verification remain open.
The current repairs are provisional: dependency-independent checks now pass for
checkpoint-path consistency and effective-batch gradient accumulation, but model
and feeder runtime tests remain blocked by missing packages. The real-data audit
and real-sample test remain blocked because no Person-in-WiFi-3D split content is
available locally; no pretrained checkpoint is present either.

No package installation, environment creation, baseline training, dataset
preprocessing, evaluation, or experiment was started. Step 3 must not close until
the complete focused suite runs in the approved dedicated environment and every
dependency-independent test genuinely passes.

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
- Generated `docs/DATA_AUDIT.md` from the actual local state. It records the
  absent dataset and static processor/feeder incompatibility without inventing
  real-data schema, values, units, or metadata.
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

## Files changed

- Read-only inspection: no files changed.
- Environment proposal and earlier status handoff: `docs/CURRENT_TASK.md` only.
- Dataset audit: added `audit_person_in_wifi_3d.py` and `docs/DATA_AUDIT.md`;
  updated `docs/CURRENT_TASK.md`.
- Legacy baseline repair: changed `model/model.py`, `train_pose.py`, `utils.py`,
  `config/person_in_wifi_3d/pose_config.yaml`,
  `data/person_in_wifi_3d/process.py`, and `feeder/person_in_wifi_3d.py`; added
  `tests/test_legacy_baseline.py`; updated this handoff.
- No requirement manifest files were materialized; installation commands are
  documented for approval instead.
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

## Actual test and execution outcomes

- No tests were run; no test files were found during the repository search.
- No environment creation, dependency resolver dry-run, package installation,
  Python 3.11 import, or proposed-environment CUDA verification was run.
- No preprocessing, pretraining, pose training, evaluation, or full training
  command was run.
- No model forward pass, loss calculation, backward pass, optimizer update,
  checkpoint save/reload, or tiny-overfitting test was run.
- No experimental results or baseline performance numbers exist yet.
- `git diff --check -- docs/CURRENT_TASK.md` completed successfully with no
  output; this validates only the documentation diff.
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
- No new dataset files, schema, or metadata were inspected during the
  environment-proposal task; the dataset findings above remain unchanged.

## Newly discovered environment facts

- The current interpreter is Python 3.14.0 with PyTorch 2.11.0+cu128.
- PyTorch reports CUDA availability on one 16 GB RTX 3080 Laptop GPU.
- The current interpreter lacks several required package distributions, so it
  cannot run the repository as-is.
- The repository contains no materialized dependency manifest or environment
  pins. The exact pins below are only a proposal in this handoff.
- Conda is unavailable on the current Bash `PATH`.
- The unmodified PiW3D entry points import all three dataset feeders;
  `train_pose.py` additionally imports MetaFi and HPE-Li. Consequently,
  `mat73`, SciPy, OpenCV, pandas, h5py, and PyWavelets belong to the unmodified
  baseline import closure even when their functionality is not called.
- `xxsubtype` is provided as a built-in by the current CPython 3.14
  interpreter. This does not verify its availability under the proposed
  CPython 3.11 environment, and it is not a PyPI package to install.
- The official PyTorch archive lists torch 2.5.1/torchvision 0.20.1 with the
  CUDA 12.1 wheel index for Windows. No local installation or GPU execution of
  that pairing has been attempted.
- PyPI metadata for `mat73==0.65` provides a `py3-none-any` wheel but no
  `Requires-Python` constraint; Python 3.11 runtime compatibility is unverified.
- Package/version compatibility, including native Windows wheel resolution,
  `timm` APIs, transitive dependencies, and whole-model `torch.load` behavior,
  remains unverified.

## Current errors, blockers, and uncertainties

### Legacy findings resolved in this task

1. `ViT_Pose_Decoder.forward()` used undefined `feature.device`; it now uses
   `x.device`. Runtime verification remains blocked by the missing `timm` import.
2. The PiW3D pose checkpoint path omitted `all-person`; it now matches the
   pretraining output path.
3. The processor wrote amplitude and phase but not the feeder's required
   `csi_ap`; it now saves the existing combined `(3, 180, 20)` array as well.
4. Pose gradient accumulation did not increment `step_count`, and its unscaled
   backward calls summed microbatch means for `steps_per_update > 1`. The loop
   now increments the counter and divides only accumulated losses by
   `steps_per_update`; `steps_per_update == 1` returns the original loss object
   and keeps legacy behavior. The synthetic effective-batch regression passes.
5. Constant amplitude/phase normalization previously divided by zero. A constant
   component now maps to zeros instead of NaNs; nonconstant min-max behavior is
   unchanged, and other NaN/Inf handling was not broadened.

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

- The current interpreter lacks `timm`, `cv2`, and other focused-test
  dependencies. The four model tests fail at `timm`; the normalization test
  fails at `cv2`. These import failures are not passing smoke tests.
- No local training split content or pretrained checkpoint is available.
- No package has been installed for this follow-up; installation awaits explicit
  approval of the exact commands below.

### Still unknown

- Whether a dataset is downloaded elsewhere and whether the public package
  matches DT-Pose preprocessing.
- Actual dataset schema, joint ordering, coordinate units, and metadata.
- Whether a dedicated compatible environment can run all entry points.
- Whether additional runtime blockers remain once dependencies are available.
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

Step 3 remains open and the current repairs are provisional. Do not proceed to
Step 4, implement research-model behavior, preprocess data, download the full
dataset, or start full training. The next action is user approval or rejection
of the exact focused-test environment commands below; nothing has been installed
for this follow-up.

## Focused-test inspection

Current full-suite command:

```bash
PYTHONDONTWRITEBYTECODE=1 python -m pytest tests/test_legacy_baseline.py -q -p no:cacheprovider
```

Actual result in the current interpreter: **5 failed, 2 passed, 1 skipped**.

| Test | Current result | Evidence/reason |
|---|---|---|
| `test_mae_forward_backward_on_synthetic_input` | failed | `ModuleNotFoundError: No module named 'timm'` while importing `model.model` |
| `test_pose_decoder_output_shape_and_finite` | failed | same missing `timm` import |
| `test_pose_decoder_one_optimization_step` | failed | same missing `timm` import |
| `test_pose_gradient_accumulation_matches_effective_batch` | passed | accumulated gradients and optimizer result match the full effective batch; `steps_per_update=1` returns the original loss object |
| `test_checkpoint_save_and_load` | failed | same missing `timm` import |
| `test_constant_csi_normalization` | failed | `ModuleNotFoundError: No module named 'cv2'` while importing the PiW3D feeder |
| `test_pretrained_checkpoint_path_is_consistent` | passed | pose YAML path matches the pretraining output path |
| `test_real_training_sample_preprocessing_and_load` | skipped | `train_data/train_data_list.txt` is absent; no `test_data` was read |

The pre-fix decoder reproduction attempt also stopped at missing `timm`, so the
expected `NameError` has not been runtime-observed. The current import failures
are dependency blockers, not evidence that the changed-code assertions pass.

### Minimal dependencies for this focused suite

The tests directly require NumPy, PyTorch, pytest, and PyYAML. Importing
`model.model` additionally requires `timm`, `einops`, and torchvision. Importing
`feeder.person_in_wifi_3d` eagerly requires SciPy, OpenCV, scikit-learn, pandas,
h5py, and PyWavelets. The focused suite does not import `mat73` or TensorBoard.

Use the existing 64-bit Python 3.11.0 interpreter (`pip 23.2.1`, `venv`
available) to create a dedicated environment outside the repository:
`C:\Users\vaska\venvs\compwifi-piw3d-py311`.

### Exact approval-gated installation commands

Do not run these commands until approved:

```bash
PY311="/c/Users/vaska/AppData/Local/Programs/Python/Python311/python.exe"
VENV="/c/Users/vaska/venvs/compwifi-piw3d-py311"

"$PY311" -m venv "$VENV"
"$VENV/Scripts/python.exe" -m pip install --only-binary=:all: --index-url https://download.pytorch.org/whl/cu121 torch==2.5.1 torchvision==0.20.1
"$VENV/Scripts/python.exe" -m pip install --only-binary=:all: numpy==1.26.4 pytest==8.3.4 PyYAML==6.0.2 timm==0.9.16 einops==0.8.0 scipy==1.14.1 opencv-python==4.9.0.80 scikit-learn==1.5.2 pandas==2.2.3 h5py==3.12.1 PyWavelets==1.8.0
```

The listed direct packages are the complete focused-test set. Pip will also
install their required transitive dependencies; actual versions must be recorded
with `pip freeze` after installation.

### Exact post-install verification commands

After approval and installation, run from the repository root:

```bash
VENV="/c/Users/vaska/venvs/compwifi-piw3d-py311"
"$VENV/Scripts/python.exe" --version
"$VENV/Scripts/python.exe" -m pip --version
"$VENV/Scripts/python.exe" -m pip check
"$VENV/Scripts/python.exe" -m pip freeze
"$VENV/Scripts/python.exe" -c "import torch; print('torch:', torch.__version__); print('CUDA build:', torch.version.cuda); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
PYTHONDONTWRITEBYTECODE=1 "$VENV/Scripts/python.exe" -m pytest tests/test_legacy_baseline.py -q -p no:cacheprovider
```

Report every test individually. If any changed-code assertion fails, show the
failure and propose the smallest fix before editing. Do not remove imports,
weaken tests, use official `test_data`, preprocess the full dataset, or start
training.

## Legacy baseline repair checks

The complete focused suite was attempted with the command above and produced
`5 failed, 2 passed, 1 skipped`. The new accumulation test was also run alone:

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
no output. No preprocessing, training entry point, evaluation, package
installation, or environment creation was run.

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

Wait for approval of the exact installation commands above. After approval,
install only that approved set, record actual Python/package versions, rerun the
complete focused suite, and report each test individually. If all synthetic
checks pass and training data is supplied, run only the conditional real
training-sample check; do not begin Step 4 or full training automatically.

## Progress log

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
