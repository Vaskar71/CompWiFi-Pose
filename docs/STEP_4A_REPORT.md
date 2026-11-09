# Step-4A Infrastructure Report

Date: 2026-11-06

## Verdict

Step-4A is complete for its bounded engineering scope. The repository now has:

- Deterministic candidate-group-disjoint internal train/development manifests generated from the official Person-in-WiFi-3D training list only.
- A separate flagship unlabelled-training CSI loader that returns CSI tensors only.
- A corrected research metric module using Hungarian person matching and explicit configurable conventions.
- Deterministic synthetic tests for manifest invariants, label guards, and metric behavior.

This establishes infrastructure correctness only. It does not establish model feasibility, pose accuracy, novelty, or scientific readiness. No research model or feasibility pilot was implemented.

## Files changed in Step-4A

Source and tests:

- `.gitignore`
- `piw3d_manifest.py`
- `feeder/piw3d_unlabelled.py`
- `research_metrics.py`
- `tests/test_piw3d_manifest.py`
- `tests/test_piw3d_unlabelled_loader.py`
- `tests/test_research_metrics.py`

Documentation and goal artifacts:

- `docs/CURRENT_TASK.md`
- `docs/STEP_4A_GOAL.md`
- `docs/STEP_4A_REPORT.md`

Generated ignored artifacts:

- `data/_artifacts/piw3d_candidate_group_manifests/internal_train_ids.txt`
- `data/_artifacts/piw3d_candidate_group_manifests/internal_development_ids.txt`
- `data/_artifacts/piw3d_candidate_group_manifests/manifest_summary.json`

Legacy `utils.py` was not changed. Legacy `feeder/person_in_wifi_3d.py` was not changed.

The worktree also contains previously reported Step-3 modifications and documents; those are not Step-4A source changes.

## Bounded upstream source check

The source pass was deliberately bounded and did not read local dataset arrays.

Upstream references checked:

- [Person-in-WiFi 3D repository README](https://raw.githubusercontent.com/aiotgroup/Person-in-WiFi3D/main/README.md)
- [Upstream `opera/datasets/wifi_pose.py`](https://raw.githubusercontent.com/aiotgroup/Person-in-WiFi3D/main/opera/datasets/wifi_pose.py)
- [Public repository page](https://github.com/aiotgroup/Person-in-WiFi-3D-repo)

Findings:

- Upstream README reports millimetre-scale errors and describes the public file layout.
- Upstream evaluation multiplies MPJPE and per-axis errors by `1000`, supporting the configured stored-unit-to-mm convention.
- Upstream per-axis evaluation labels axis 1 as `mpjpev`, supporting the documented upstream axis-1-vertical convention. This conflicts with the local visual audit's axis-2 interpretation; the conflict remains unresolved and coordinates were not reordered.
- Upstream dataset code reads `train_data_list.txt` or `test_data_list.txt` and treats each listed entry as a filename. It does not define recording, subject, environment, or clip semantics for the first two filename tokens.
- Upstream evaluation passes `root=[5,7]` to its matching helper, but the executed list-root branch does not perform separate root localization. Step-4A therefore keeps root localization explicit and configurable instead of inheriting that ambiguous behavior.

Unresolved conventions are handled explicitly and remain documented assumptions, not claims.

## Candidate-group-disjoint manifests

Implementation: `piw3d_manifest.py`.

Conventions:

- Candidate group: first two underscore-separated ID tokens.
- Hash: SHA-256 over `compwifi-step3-v1:<group>`.
- Groups are sorted by hash and the first `floor(0.8 * group_count)` groups are assigned to internal train; the remainder to internal development.
- IDs within each manifest are sorted by candidate group and numeric frame.
- The split label is exactly `candidate-group-disjoint`.
- No person-count stratification, filtering, balancing, or filename-count use is performed.
- The generator accepts a training list path only and has no test-list or test-array input path.

Generation command:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe piw3d_manifest.py --train-list data/person_in_wifi_3d/wifi_dataset/train_data/train_data_list.txt --output-dir data/_artifacts/piw3d_candidate_group_manifests --train-fraction 0.8
```

Verified output:

```text
official_training_id_count: 89946
candidate_group_count: 215
internal_train: 172 groups, 70197 IDs
internal_development: 43 groups, 19749 IDs
person_count_stratified: false
filename_person_counts_used: false
official_test_list_read: false
official_test_array_read: false
id_overlap: 0
candidate_group_overlap: 0
cross_partition_next_frame_pairs: 0
source_list_sha256: 89e09086e39bd8a2287e45206696cd0d3473b04ccaecea7b29625f6f9bb0148d
internal_train_manifest_sha256: 3628030d40df460829cea988e6e88c4966ae6163288f307d3720c08b7b587d15
internal_development_manifest_sha256: 4a3dfb699e85cdf1a6782ce779ce9d401f2db6737ce96353ca0f7d706708113d
```

Independent validation command:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe - <<'PY'
from pathlib import Path
from piw3d_manifest import parse_sample_id

train_list = Path('data/person_in_wifi_3d/wifi_dataset/train_data/train_data_list.txt')
train_ids = [line.split()[0] for line in train_list.read_text(encoding='utf-8-sig').splitlines() if line.split()]
train_set = set(train_ids)
internal_train = set(Path('data/_artifacts/piw3d_candidate_group_manifests/internal_train_ids.txt').read_text(encoding='utf-8').splitlines())
internal_dev = set(Path('data/_artifacts/piw3d_candidate_group_manifests/internal_development_ids.txt').read_text(encoding='utf-8').splitlines())
assignments = {}
for sample_id in internal_train:
    assignments[sample_id] = 'internal_train'
for sample_id in internal_dev:
    assignments[sample_id] = 'internal_development'
assert len(train_ids) == 89946
assert len(internal_train) + len(internal_dev) == 89946
assert set(assignments) == train_set
assert len(assignments) == 89946
assert internal_train.isdisjoint(internal_dev)
parsed = {sample_id: parse_sample_id(sample_id) for sample_id in train_ids}
train_groups = {parsed[sample_id]['group'] for sample_id in internal_train}
dev_groups = {parsed[sample_id]['group'] for sample_id in internal_dev}
assert train_groups.isdisjoint(dev_groups)
ids_by_group_frame = {(value['group'], value['frame']): sample_id for sample_id, value in parsed.items()}
next_frame_crossings = 0
for sample_id, value in parsed.items():
    successor = ids_by_group_frame.get((value['group'], value['frame'] + 1))
    if successor is not None and assignments[successor] != assignments[sample_id]:
        next_frame_crossings += 1
assert next_frame_crossings == 0
print('independent_manifest_validation: ids=89946 assigned_once=True id_overlap=0 group_overlap=0 next_frame_crossings=0 test_read=False')
PY
```

Independent validation output:

```text
independent_manifest_validation: ids=89946 assigned_once=True id_overlap=0 group_overlap=0 next_frame_crossings=0 test_read=False
```

Ignore check:

```bash
git check-ignore -v data/_artifacts/piw3d_candidate_group_manifests/manifest_summary.json
```

Output:

```text
.gitignore:3:data/_artifacts/ data/_artifacts/piw3d_candidate_group_manifests/manifest_summary.json
```

The official test split was not read or modified.

## Flagship unlabelled CSI loader

Implementation: `feeder/piw3d_unlabelled.py`.

Label-guard properties:

- Reads only beneath `train_data/csi_ap` for a supplied training root.
- Can restrict to an explicit ID list, but every selected ID must occur in that root's official `train_data_list.txt`.
- Never constructs a keypoint path and never returns poses, padded poses, `person_num`, presence targets, SMPL values, raw IDs, or filenames in batches.
- Does not parse filename-derived person counts and does not filter, balance, or sample by them.
- Filters only technical CSI eligibility: file presence during initialization, then expected shape `(3,180,20)` and finite values when a sample is read.
- Normalizes amplitude and phase independently; constant components map to zero, matching the repaired legacy convention.
- Batches contain only the declared `csi` tensor (`BATCH_KEYS = ("csi",)` at `feeder/piw3d_unlabelled.py:11`).

The existing labelled single-person loader and legacy evaluation path remain separate and unchanged.

Fixture command:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe - <<'PY'
from pathlib import Path
import torch
from feeder.piw3d_unlabelled import UnlabelledPiW3DCSIDataset, make_unlabelled_csi_dataloader

data_root = Path('data/_fixtures/piw3d_two_frame')
dataset = UnlabelledPiW3DCSIDataset(data_root)
loader = make_unlabelled_csi_dataloader(dataset, is_training=False, generator=torch.manual_seed(0), batch_size=2, num_workers=0)
batches = list(loader)
assert len(dataset) == 2
assert dataset.skipped_count == 0
assert [tuple(batch['csi'].shape) for batch in batches] == [(2, 3, 180, 20)]
assert all(set(batch) == {'csi'} for batch in batches)
assert all(torch.isfinite(batch['csi']).all() for batch in batches)
print('fixture_unlabelled_loader: samples=', len(dataset), 'skipped_missing_csi=', dataset.skipped_count)
print('batch_keys=', sorted(batches[0].keys()), 'batch_shape=', tuple(batches[0]['csi'].shape), 'finite=True')
PY
```

Fixture output:

```text
fixture_unlabelled_loader: samples= 2 skipped_missing_csi= 0
batch_keys= ['csi'] batch_shape= (2, 3, 180, 20) finite=True
```

A direct attempt to pair the full real internal-train manifest with the two-ID fixture was rejected because the manifest IDs are absent from the fixture's official training list. That is the intended training-membership guard; the fixture run above uses the matching fixture list.

## Corrected research metrics

Implementation: `research_metrics.py`.

Defaults and definitions:

- Predicted count: number of prediction slots with presence score greater than or equal to the frozen threshold; default `0.5`.
- Matching: rectangular Hungarian one-to-one assignment over all above-threshold predictions and all ground-truth people.
- Assignment cost: mean unaligned per-joint Euclidean distance after conversion to millimetres.
- Valid match threshold: configurable, default `500 mm`.
- Matched MPJPE: mean unaligned matched-joint distance in millimetres; no root or Procrustes alignment.
- Matched PA-MPJPE: per-person similarity alignment with translation, rotation, and isotropic scale; result remains in millimetres. Degenerate or constant skeletons are excluded and counted.
- Stored-unit-to-mm scale: explicit configurable value, default `1000.0`; applied once to distance and root metrics.
- PCK: default PCK@0.5, normalized per matched ground-truth person by `||joint_6 - joint_4||` in stored coordinate units. If that scale is below `1e-6` stored units, the person's PCK is undefined, excluded from the denominator, and counted.
- Root error: Euclidean distance between configurable root indices on valid matches; default root index `0`, explicit provisional convention.
- Exact person-count accuracy: percentage of samples whose inferred count equals ground-truth count.
- Missed people: unmatched ground-truth people.
- Duplicate people: unmatched predictions whose nearest ground-truth person is already validly matched and lies within the same match threshold.
- Spurious predictions: other unmatched above-threshold predictions.
- Person precision, recall, and F1: `TP/(TP+FP)`, `TP/(TP+FN)`, and harmonic mean. Zero denominators return `None` while numerator and denominator counts remain explicit.
- Aggregation: global sums and denominators across samples and joints, not equal averaging of unequal batches.
- Oracle count: separately named `evaluate_oracle_count_diagnostic`; it is marked `diagnostic_only=True` and `headline_comparison_allowed=False`.

Legacy `utils.py` remains unchanged.

## Tests and actual results

Focused manifest tests:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_piw3d_manifest.py -q -p no:cacheprovider
```

```text
6 passed in 0.10s
```

Focused unlabelled-loader tests:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_piw3d_unlabelled_loader.py -q -p no:cacheprovider
```

```text
6 passed in 4.52s
```

Focused research-metric tests:

```bash
PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests/test_research_metrics.py -q -p no:cacheprovider
```

```text
11 passed in 0.56s
```

Complete focused suite:

```bash
PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider
```

```text
.................................                                        [100%]
============================== warnings summary ===============================
tests/test_legacy_baseline.py::test_mae_forward_backward_on_synthetic_input
  C:\Users\vaska\venvs\compwifi-piw3d-py311\Lib\site-packages\timm\models\layers\__init__.py:49: DeprecationWarning: Importing from timm.models.layers is deprecated, please import via timm.layers
    warnings.warn(f"Importing from {__name__} is deprecated, please import via timm.layers", DeprecationWarning)

-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html
33 passed, 1 warning in 9.78s
```

Test accounting:

- Original legacy focused tests: 8 passed, 0 skipped.
- Existing preprocessing tests: 2 passed.
- New manifest tests: 6 passed.
- New unlabelled-loader tests: 6 passed.
- New research-metric tests: 11 passed.
- Total: **33 passed, 0 skipped, 1 external timm deprecation warning**.

No assertions were weakened or replaced with unconditional passes.

## Boundaries honored

Not performed:

- Package installation or removal.
- Dataset download.
- File deletion.
- Full preprocessing or full training.
- Compositional model implementation.
- Feasibility pilot.
- Official test evaluation or official test-array reading.
- Credential access.
- Commit, push, or remote changes.

## Remaining uncertainties and blockers

- Official keypoint units, joint ordering, coordinate frame, and root semantics remain unproven. The upstream `*1000` evaluation supports the configured metre-to-millimetre convention, but local metadata does not formally declare units.
- The upstream axis-1-vertical convention conflicts with the local visual axis-2 interpretation. No coordinate reordering was performed.
- Candidate filename groups remain filename rules only. No recording-, subject-, or environment-disjointness is claimed.
- Official full-dataset processed `csi_ap` arrays remain absent. The unlabelled loader therefore cannot produce a real full-dataset batch without a separately approved preprocessing run; the two-frame fixture and synthetic fixtures verify the loader contract.
- Corrected metrics are implemented as a separate module but are not wired into `train_pose.py`.
- Scientific feasibility, accuracy, novelty, and readiness remain unproven.

## Recommended next task

With separate approval, wire a bounded development-evaluation path that reads only the generated internal-development manifest and uses `research_metrics.py`; do not begin the compositional model, pilot, full preprocessing, or official test evaluation automatically.
