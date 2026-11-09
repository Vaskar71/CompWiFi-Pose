# Step-4B-D4 D2 Reproducibility Audit Report

Date: 2026-11-08

## Verdict

**D4 AUDIT COMPLETE — evidence classification: `repeats differ from each other`.**

Two independent fresh seed-0 repeats of the unchanged 200-step D2 procedure both completed on CUDA with finite losses and gradients, exact 97-parameter optimizer coverage, and observed parameter updates. Both reproduced the historical D2 initial train-mode MPJPE exactly. Neither repeat reproduced the historical D2 step-100 or step-200 values, and the two repeats did not reproduce each other.

The first full recorded repeat-to-repeat trajectory divergence is at step 1 in `max_abs_encoder_gradient`; the first repeat-to-repeat pre-update loss divergence is at step 3. These facts are reported without a post-hoc tolerance and without assigning a root cause.

Acceptance of D4 is completion of the audit with preserved records and reported measurements. Exact reproduction was not required, and no MPJPE criterion is changed or retroactively passed.

## Scope and label access

- Work was limited to Step-4B-D4.
- Training used only the corrected `freeze_v2` / `pilot_data_v2` internal-training path and the fixed eight-sample batch:
  `S11_06_10`, `S11_06_11`, `S11_06_13`, `S11_06_14`,
  `S11_06_16`, `S11_06_17`, `S11_06_19`, `S11_06_20`.
- Development samples were not loaded.
- Official test samples and arrays were not read.
- No multi-person, unlabelled, compositional, epoch, or development-evaluation run was started.
- No D3 rerun or 800-step extension was performed.
- No package installation/removal, download, deletion, source-dataset modification, commit, push, or remote operation was performed.

## Files added or changed for Step-4B-D4

Source and tests:

- `step4b_d4_reproducibility.py`
- `tests/test_step4b_d4_reproducibility.py`

Documentation:

- `docs/STEP_4B_D4_REPORT.md`
- `docs/CURRENT_TASK.md`

Generated ignored D4 artifacts:

- `data/_artifacts/step4b_d4/static_execution_comparison.json`
- `data/_artifacts/step4b_d4/environment.json`
- `data/_artifacts/step4b_d4/pre_run_protected_records.json`
- `data/_artifacts/step4b_d4/post_run_protected_records.json`
- `data/_artifacts/step4b_d4/final_protected_records.json`
- `data/_artifacts/step4b_d4/repeat_comparison.json`
- `data/_artifacts/step4b_d4/repeat_evidence_summary.json`
- complete repeat A artifacts under `data/_artifacts/step4b_d4/repeat_a/`
- complete repeat B artifacts under `data/_artifacts/step4b_d4/repeat_b/`

The historical D2 and D3 runners, model source, reports, decisions, results, and artifact roots were not edited or overwritten.

## Static D2/D3 execution-path comparison

Static evidence is saved in:

- `data/_artifacts/step4b_d4/static_execution_comparison.json`

### Shared execution path established before new runs

The D3 runner is a wrapper around D2 components rather than a second model implementation:

- `build_d3_model is build_d2_model`.
- `make_d3_optimizer is make_sgd_optimizer`.
- D3 imports the D2 frozen-input verifier, batch constants, model builder, SGD builder, parameter audit, checkpoint helper, RNG-state helpers, state clone/restore helpers, and parameter-change summary.
- Both runners call `_seed_everything(config["seed"])` with the frozen seed `0` after input verification and before dataset/model construction.
- Both use `SinglePersonPiW3DDataset`, the same corrected pilot root, the same split assignments, and `make_single_person_dataloader(..., is_training=False, generator=torch.Generator().manual_seed(config["seed"]), batch_size=8, num_workers=0)`.
- Both construct a fresh trainable-encoder `MAE_ViT`/`ViT_Pose_Decoder`, call `model.train()`, construct SGD(`lr=0.01`, `momentum=0.9`) over exactly the 97 trainable parameters, and audit optimizer coverage.
- Through step 200, both execute `optimizer.zero_grad()`, a train-mode forward on the fixed CSI batch, mean Euclidean pose loss, `loss.backward()`, gradient-statistics collection, and `optimizer.step()`.
- Both capture train-mode checkpoints at steps 0, 100, and 200 with the same helper and restore model/RNG state after those checkpoint forwards.
- Neither runner enables PyTorch deterministic algorithms or assigns cuDNN deterministic mode in source.

No static source-path difference was found in model construction, seed use, loader order, forward/backward call order, loss calculation, optimizer construction/update order, optimizer coverage audit, parameter-change summary, batch size, learning rate, momentum, device-selection mode, or the step-0/100/200 train-mode checkpoint procedure.

### Observed D2/D3 code differences

The established differences are bookkeeping or post-step-200 control flow:

1. D3 additionally verifies protected D2 source/result/decision hashes and requires the D3 protocol document before writing its own freeze.
2. D2 defines a 200-step run; D3 defines an 800-step run and an 800-step checkpoint.
3. D3 duplicates the gradient-statistics logic in `_replace_audit_with_gradient_stats()` instead of importing D2's helper. When gradients are finite, the recorded finite/nonzero semantics are equivalent; this helper records evidence and does not update gradients or parameters.
4. After the step-200 checkpoint, D3 checks exact D2 reproduction and either stops or continues. D2 serializes its completed 200-step artifacts.
5. D3 adds reproduction-gate, late-trajectory, and prior-comparison artifact fields.

These differences do not establish a cause for the observed numerical divergence. In particular, the static comparison does not prove or disprove CUDA nondeterminism, library-level kernel selection, environmental drift, or any other mechanism.

## Environment and determinism record

Environment evidence is saved in:

- `data/_artifacts/step4b_d4/environment.json`

| Item | Recorded value |
|---|---|
| OS | Windows 10, AMD64 |
| Python | `3.11.0` |
| Interpreter | `C:\Users\vaska\venvs\compwifi-piw3d-py311\Scripts\python.exe` |
| PyTorch | `2.5.1+cu121` |
| CUDA build | `12.1` |
| GPU | NVIDIA GeForce RTX 3080 Laptop GPU, 16 GiB |
| Driver | `581.29` |
| Compute capability | `8.6`; compiled architectures include `sm_86` |
| cuDNN | `90100` |
| NumPy | `1.26.4` |
| torchvision | `0.20.1+cu121` |
| SciPy | `1.14.1` |
| scikit-learn | `1.5.2` |
| OpenCV | `4.9.0.80` |
| pandas | `2.2.3` |
| h5py | `3.12.1` |
| PyWavelets | `1.8.0` |
| timm | `0.9.16` |
| einops | `0.8.0` |
| PyYAML | `6.0.2` |
| pytest | `8.3.4` |

Determinism-related settings recorded before the repeats:

| Setting | Recorded value |
|---|---|
| Frozen run seed | `0` |
| Ordered-loader generator seed | `0` |
| `torch.are_deterministic_algorithms_enabled()` | `false` |
| deterministic-algorithm warn-only mode | `false` |
| `torch.backends.cudnn.enabled` | `true` |
| `torch.backends.cudnn.benchmark` | `false` |
| `torch.backends.cudnn.deterministic` | `false` |
| `torch.backends.cudnn.allow_tf32` | `true` |
| `torch.backends.cuda.matmul.allow_tf32` | `false` |
| `torch.get_float32_matmul_precision()` | `highest` |
| `CUBLAS_WORKSPACE_CONFIG` | unset |
| `PYTHONHASHSEED` | unset |
| `CUDA_VISIBLE_DEVICES` | unset |
| CPU/inter-op threads | `8` / `8` |

No deterministic mode was selectively enabled for D4.

## Exact commands and observed outputs

### 1. Static comparison and environment capture

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe - <<'PY'
from step4b_d4_reproducibility import (
    write_environment_record,
    write_static_execution_comparison,
    verify_protected_records,
    _write_json,
)
static_path = write_static_execution_comparison('data/_artifacts/step4b_d4/static_execution_comparison.json')
environment_path = write_environment_record('data/_artifacts/step4b_d4/environment.json')
protected_path = _write_json('data/_artifacts/step4b_d4/pre_run_protected_records.json', verify_protected_records())
print(static_path)
print(environment_path)
print(protected_path)
PY
```

Observed output:

```text
data\_artifacts\step4b_d4\static_execution_comparison.json
data\_artifacts\step4b_d4\environment.json
data\_artifacts\step4b_d4\pre_run_protected_records.json
```

The static record reports all shared execution checks passed and no explicit deterministic-algorithm enable call in either D2 or D3.

### 2. Focused D4 tests

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_step4b_d4_reproducibility.py -q -p no:cacheprovider
```

Observed result:

```text
6 passed, 1 warning in 6.10s
```

The warning was the known external `timm.models.layers` deprecation warning.

### 3. Repeat A of the unchanged D2 procedure

```bash
mkdir -p data/_artifacts/step4b_d4/repeat_a && PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_d2_diagnostic.py --artifact-root data/_artifacts/step4b_d4/repeat_a --device auto > data/_artifacts/step4b_d4/repeat_a/run.log 2>&1
```

Observed result: exit code `0`; `diagnostic_run_status=COMPLETE`; `diagnostic_status=TRAINABLE_ENCODER_DIAGNOSTIC_ABOVE_CRITERION`; device `cuda`; 200 optimizer steps.

### 4. Repeat B of the unchanged D2 procedure

```bash
mkdir -p data/_artifacts/step4b_d4/repeat_b && PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe step4b_d2_diagnostic.py --artifact-root data/_artifacts/step4b_d4/repeat_b --device auto > data/_artifacts/step4b_d4/repeat_b/run.log 2>&1
```

Observed result: exit code `0`; `diagnostic_run_status=COMPLETE`; `diagnostic_status=TRAINABLE_ENCODER_DIAGNOSTIC_ABOVE_CRITERION`; device `cuda`; 200 optimizer steps.

### 5. Repeat comparison

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe - <<'PY'
from step4b_d4_reproducibility import classify_repeat_evidence, _write_json
comparison = classify_repeat_evidence(
    'data/_artifacts/step4b_d2',
    'data/_artifacts/step4b_d4/repeat_a',
    'data/_artifacts/step4b_d4/repeat_b',
)
path = _write_json('data/_artifacts/step4b_d4/repeat_comparison.json', comparison)
print(path)
print('classification=', comparison['classification'])
PY
```

Observed output:

```text
data\_artifacts\step4b_d4\repeat_comparison.json
classification= repeats differ from each other
```

### 6. Full fixture-backed suite after D4 implementation

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_step4b_d4_reproducibility.py -q -p no:cacheprovider && PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

Observed result:

```text
6 passed, 1 warning in 6.17s
68 passed, 1 warning in 10.63s
```

There were 0 skips. The single warning was the known external `timm.models.layers` deprecation warning.

### 7. Post-run protected-record verification

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe - <<'PY'
from step4b_d4_reproducibility import verify_protected_records, _write_json
verification = verify_protected_records()
path = _write_json('data/_artifacts/step4b_d4/post_run_protected_records.json', verification)
print(path)
print('record_count=', verification['record_count'])
print('all_unchanged=', verification['all_unchanged'])
PY
```

Observed output:

```text
data\_artifacts\step4b_d4\post_run_protected_records.json
record_count= 40
all_unchanged= True
```

## Repeat measurements

All checkpoint MPJPE values below are train-mode values after the indicated number of optimizer steps. MPJPE equals the frozen mean Euclidean pose loss in this procedure.

| Run | Step 0 | Step 100 | Step 200 |
|---|---:|---:|---:|
| Historical D2 | `4.920130252838135` | `0.5151628255844116` | `0.46943652629852295` |
| D3 stop-gate context | `4.920130252838135` | `0.40826788544654846` | `0.3430270552635193` |
| D4 repeat A | `4.920130252838135` | `0.4460887312889099` | `0.46392330527305603` |
| D4 repeat B | `4.920130252838135` | `0.49196043610572815` | `0.37702471017837524` |

D3 remains context only: it stopped at the reproduction gate, has no valid 800-step measurement, and its lower recorded values are not treated as improvement.

### Exact checkpoint equality

| Comparison | Step 0 | Step 100 | Step 200 |
|---|---|---|---|
| Repeat A vs repeat B | exact | different | different |
| Repeat A vs historical D2 | exact | different | different |
| Repeat B vs historical D2 | exact | different | different |

### Numerical differences

These are descriptive decimal differences from the recorded checkpoint values, not tolerances.

| Left minus right | Step-100 difference | Step-200 difference |
|---|---:|---:|
| Repeat A − repeat B | `-0.04587170481681825` | `0.08689859509468079` |
| Repeat A − historical D2 | `-0.0690740942955017` | `-0.00551322102546692` |
| Repeat B − historical D2 | `-0.02320238947868345` | `-0.09241181612014771` |
| D3 stop context − historical D2 | `-0.10689494013786314` | `-0.12640947103500365` |

All step-0 differences in these comparisons are zero.

## First recorded trajectory divergence

The complete per-step records for repeats A and B are preserved in their `loss_trajectory.json` files. Historical D2's complete record remains under `data/_artifacts/step4b_d2/loss_trajectory.json`. D3 wrote no complete trajectory artifact because it stopped at its gate.

| Comparison | First full-record divergence | First pre-update-loss divergence |
|---|---|---|
| Repeat A vs repeat B | Step `1`, `max_abs_encoder_gradient`: A `0.2116079181432724`, B `0.2116079032421112` | Step `3`: A `3.352538585662842`, B `3.3525383472442627` |
| Repeat A vs historical D2 | Step `1`, `max_abs_encoder_gradient`: A `0.2116079181432724`, D2 `0.2116079032421112` | Step `6`: A `1.616533637046814`, D2 `1.616533875465393` |
| Repeat B vs historical D2 | Step `3`, `pre_update_loss_stored_units`: B `3.3525383472442627`, D2 `3.352538585662842`; the same record also differs in maximum gradient statistics | Step `3`: B `3.3525383472442627`, D2 `3.352538585662842` |

The exact comparison artifact is:

- `data/_artifacts/step4b_d4/repeat_comparison.json`

## Finite-loss, gradient, optimizer, and parameter-change evidence

| Evidence | Historical D2 | Repeat A | Repeat B |
|---|---:|---:|---:|
| Complete per-step records | 200 | 200 | 200 |
| All pre-update losses finite | true | true | true |
| All 97 trainable gradient tensors finite per step | true | true | true |
| Nonzero encoder gradient tensors per step | 50 of 50 | 50 of 50 | 50 of 50 |
| SGD parameter references | 97 unique | 97 unique | 97 unique |
| Exact trainable-set optimizer coverage | true | true | true |
| Trainable parameters changed by step 200 | 97 of 97 | 97 of 97 | 97 of 97 |
| Encoder parameters changed by step 200 | 50 of 50 | 50 of 50 | 50 of 50 |
| Maximum absolute encoder gradient observed | `0.2116079032421112` | `0.2116079181432724` | `0.2116079032421112` |
| Maximum absolute parameter change | `0.7603052854537964` | `0.7877951860427856` | `0.810707688331604` |

Per-step loss trajectory summary:

| Run | First pre-update loss | Step-100 pre-update loss | Last pre-update loss | Minimum pre-update loss |
|---|---:|---:|---:|---:|
| Historical D2 | `4.924509525299072` | `0.588363766670227` | `0.3613433539867401` | `0.34691399335861206` |
| Repeat A | `4.924509525299072` | `0.5627881288528442` | `0.43507394194602966` | `0.3553522527217865` |
| Repeat B | `4.924509525299072` | `0.5701178908348083` | `0.3906129002571106` | `0.3546094298362732` |

Secondary eval-mode reload measurements, not the frozen criterion, were:

- Historical D2: `0.43328580260276794` stored units.
- Repeat A: `0.5134801268577576` stored units.
- Repeat B: `0.39538922905921936` stored units.

Both repeat final train-mode MPJPE values remain above the unchanged `0.10` stored-unit criterion. No criterion, threshold, optimizer, model, data, or measurement setting was changed after seeing these outcomes.

## Descriptive evidence classification

The exact-match comparisons classify the D4 evidence as:

```text
repeats differ from each other
```

This is not a claim that historical D2 was wrong, that D3 was wrong, that either D4 repeat is correct, or that CUDA nondeterminism caused the differences. The audit directly establishes that:

1. the same protected source/result records were present before and after D4;
2. the static D2 and D3 execution paths share the model, seed, data-order, forward/backward, loss, optimizer, audit, and step-0/100/200 checkpoint procedures;
3. two fresh seed-0 executions of the unchanged D2 command diverged from each other in recorded gradient and loss trajectories after the initial checkpoint;
4. both diverged from historical D2 at steps 100 and 200; and
5. all required finite-update and optimizer-coverage checks remained true.

No root cause is directly established. A lower numerical endpoint in repeat B or D3 is not interpreted as improvement.

## Historical protected-record preservation

Pre-run, post-run, and final protected-record artifacts:

- `data/_artifacts/step4b_d4/pre_run_protected_records.json`
- `data/_artifacts/step4b_d4/post_run_protected_records.json`
- `data/_artifacts/step4b_d4/final_protected_records.json`

All checks covered 40 protected source, report, decision, result, trajectory, checkpoint, prediction, state-dict, D3-freeze, and D3-stop records. The final result was:

```text
record_count=40
all_unchanged=True
```

Representative protected source hashes remain:

| Record | SHA-256 |
|---|---|
| `step4b_d2_diagnostic.py` | `cd8aa4e36ea398769983c273ae3fdbc2f211650b8e35e392669848f8c2d8e161` |
| `step4b_d3_diagnostic.py` | `5a5278f760ee699106b0752a2282d8179a7cf76b80a97bf29d3daf5ebf5d2624` |
| `model/model.py` | `7164ad53d4d9cf71316e3e379983576b82bfced17ec58ff67c7febadc121840f` |

`docs/CURRENT_TASK.md` was intentionally updated as the living handoff and is therefore not part of the fixed unchanged-hash table.

## Current status and recommended next action

Original Step-4B remains **BLOCKED**. D1 remains an above-criterion extended-step diagnostic. D2 remains an above-criterion trainable-encoder diagnostic. D3 remains stopped at `D2_REPRODUCTION_MISMATCH_STOPPED` with no valid 800-step measurement. D4 is complete and establishes that fresh unchanged-procedure repeats differ from each other.

Do not begin Step-4C, rerun D3, extend a run to 800 steps, enable deterministic mode selectively, change the optimizer, relax the criterion, or start development/epoch evaluation from this result. Any next diagnostic or protocol revision requires a separately approved, newly frozen task.

No scientific feasibility, accuracy, novelty, generalization, or state-of-the-art claim is made.
