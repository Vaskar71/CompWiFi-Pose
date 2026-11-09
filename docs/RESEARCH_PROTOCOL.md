# CompWiFi-Pose Research Protocol

Status: **design only; not executed**. This document defines the later leakage-safe
benchmark and early feasibility pilot. It does not authorize implementation of the
full compositional model, creation of research manifests, training, or evaluation.
The bounded Step-3 fixture and one-step engineering smoke are not pilot results.

## 1. Claim and scope

The flagship claim is **zero real multi-person training labels**. It is not
“annotation-free learning”: labelled single-person CSI-pose pairs are used, and
separately labelled supervised references have different data access. Development
label use must be disclosed and does not broaden the claim beyond training labels.
Make no first-ever novelty claim or Q1/venue-quality claim.

The output is a variable-sized set of 3D skeletons with one presence score per
prediction slot. Person count is inferred from presence predictions, never supplied
as an inference-time input. The initial scope is the existing PiW3D hardware and
its supported one-, two-, and three-person scenes.

## 2. Development and strict protocols

These are distinct protocols and must not be mixed in reports.

| Data/resource | Development protocol | Strict protocol |
| --- | --- | --- |
| Labelled official-training single-person pairs | Training permitted | Training permitted |
| Official-training CSI without labels | Training permitted through a CSI-only loader | Training permitted through a CSI-only loader |
| Real multi-person official-training pose/count labels | Never used for training | Never used for training |
| Internal development multi-person labels | Disclosed development evaluation and model/checkpoint/hyperparameter selection only; never training | Forbidden for evaluation, checkpoint selection, and hyperparameter selection |
| Single-person validation or synthetic validation | Permitted | Permitted for all selection |
| Previously frozen development decisions | Permitted if disclosed | Permitted; no new multi-person-label-driven decisions |
| Official test CSI/labels | Untouched until final frozen evaluation | Untouched until final frozen evaluation |
| Fully supervised multi-person training labels | Only in a separately labelled upper-reference experiment | Only in a separately labelled upper-reference experiment, never in the strict flagship run |

Under both protocols, official test CSI and labels are excluded from pretraining,
training, validation, checkpoint selection, and hyperparameter tuning. No test
samples may enter through caches, preprocessing outputs, wrappers, or filenames.
The current legacy `validation` loader reads official `test_data`; it must not be
used as internal development validation for this research.

A development-protocol result may retain the precise claim “zero real multi-person
training labels” only when its development-label evaluation and selection are
reported alongside it. A strict-protocol result must additionally document the
single-person, synthetic, or previously frozen basis for every selection decision.

## 3. Provisional internal manifests — design only

No manifest is created or executed under this protocol-writing task.

1. Start only from the official training list. For an ID such as `S11_01_10`, define
   the candidate filename group as the first two underscore-separated fields,
   `S11_01`. This is a filename rule, not a claim about recording boundaries.
2. Assign each whole candidate group to internal train or internal development with
   a deterministic hash of `compwifi-step3-v1:<group>` using SHA-256. Sort groups by
   hash and assign the first 80% to internal train and the remainder to development.
   Freeze the algorithm, source-list hash, assignments, and exact IDs before use.
3. Assert zero group overlap and zero exact-ID overlap between the two manifests.
   Record all IDs used at every label budget. Prefer whole-group label budgets; do
   not split adjacent frames from one group across labelled and unlabelled subsets.
4. Single-person metadata may select the labelled single-person subset. Development
   labels may be attached only to the development evaluation view, never to a
   training batch.
5. Claim only **candidate-filename-group disjointness**. The metadata does not
   establish recording-, subject-, or environment-disjoint generalization. Shared
   candidate groups across the official train/test split are a further reason not
   to claim independent generalization.

## 4. Label-access guards

The unlabelled training loader returns CSI tensors only. It must not return or
compute poses, padded poses, `person_num`, presence targets, count targets, SMPL
values, or keypoint paths. Audit manifests may retain IDs for traceability, but raw
filename person-count encodings must not enter batches or model features.

The unlabelled pool must not use filename-derived counts for filtering, stratifying,
balancing, weighting, or sampling. Eligibility filters are limited to split
membership and CSI technical validity, such as file presence, expected shape, and
finite CSI values. Do not infer labels through caches, preprocessing metadata,
adjacent labelled frames, or dataset-wrapper side effects.

The labelled single-person loader is separate and may expose only the permitted
single-person pose supervision. The supervised upper reference has its own clearly
labelled data path so its multi-person labels cannot enter either flagship protocol.

## 5. Early feasibility pilot — not executed

Use the simplest feature-space scene-to-set prototype for the compositional arm.
Do not add a large temporal Transformer, differentiable RF renderer, temporal loss,
or physics component in this pilot. Do not synthesize people by adding raw CSI.

| Arm | Labelled supervision | Unlabelled official-training CSI | Purpose |
| --- | --- | --- | --- |
| A: single-person-only | Single-person pose labels only | None | Direct-transfer baseline |
| B: reconstruction-only | Same single-person labels as A | Same CSI pool as C, reconstruction/latent objective only | Controls for self-supervised representation learning |
| C: minimal compositional | Same single-person labels as A/B | Same CSI pool as B, minimal slot composition and reconstruction/consistency | Tests the research hypothesis |
| Fully supervised upper reference | Separately labelled multi-person training supervision | Optional and disclosed | Upper reference only; not a fair zero-label competitor |

A, B, and C use the same encoder family, single-person label IDs and budget,
unlabelled CSI IDs where applicable, training-step budget, presence rule, and
corrected evaluation implementation. Changes in model capacity or compute must be
reported and matched by an additional control. Equal-label-budget supervised
baselines are required for any later low-label comparison; the upper reference is
always reported separately.

### Collapse controls

- A duplicate-slot control repeats one predicted person across active slots.
- A mean-pose control predicts the mean skeleton computed only from permitted
  labelled single-person training poses, repeated across active slots.
- Report active-slot count, pairwise prediction similarity, duplicate count,
  per-slot usage, and prediction variance across different CSI inputs.
- Compare reconstruction error and pose/set metrics together. Good reconstruction
  with mean poses, duplicated slots, or shuffled/unrelated CSI is a failure, not
  evidence of successful decomposition.

## 6. Predeclared development-only go/no-go rules

Run the eventual pilot with three frozen seeds and report every seed plus mean and
standard deviation. No official test result participates in these decisions.

**Go** to limited method development only when all of the following hold on internal
development data, for both two- and three-person scenes:

1. C's mean person-level F1 is at least 5% relatively higher than the stronger of A
   and B under the same frozen presence and matching rules.
2. C's matched MPJPE is no more than 5% relatively worse than that stronger
   baseline, and C does not increase the sum of missed and duplicate people.
3. C beats both collapse controls on person F1 and matched MPJPE, and neither
   duplicate predictions nor a CSI-independent mean-pose output explains the gain.

**Yellow** when pose error improves but count accuracy, slot separation, or the
miss/duplicate criterion fails. Investigate only the implicated component within a
new bounded task; do not describe this as successful scene decomposition.

**No-go** for the zero-label prototype when C fails the primary F1 criterion, matches
the collapse controls, or persistently collapses within the frozen pilot budget.
Report the failed result. A later 0.1–1% multi-person-label experiment requires a
separate explicitly labelled task and must not silently replace the flagship result.

These thresholds are fixed before pilot execution. Changing them after seeing
development outcomes requires a new disclosed protocol version and invalidates
comparisons with runs selected under the old thresholds. Official test remains
untouched in every case.

## 7. Corrected research metrics

Research evaluation is set-based and does not trim predictions using ground-truth
count. Freeze the presence threshold before the corresponding evaluation. Predicted
count is the number of slots whose presence score passes that threshold.

Use millimetres for research distance metrics: multiply stored metre-scale
coordinates by 1000, consistent with the original authors' evaluation, and state
this conversion with every result. Keep coordinate order unchanged. The local
axis-2-vertical audit interpretation conflicts with upstream axis-1-vertical
per-axis evaluation; this protocol does not declare either settled.

### Matching and pose error

1. For each sample, construct the cost between every above-threshold prediction and
   every ground-truth person as their unaligned mean per-joint Euclidean distance
   in mm. Solve the rectangular one-to-one assignment with Hungarian matching.
2. A valid person match additionally requires assignment cost **at most 500 mm**.
   Reject assignments above that threshold. Freeze this threshold before evaluation.
3. **Matched MPJPE (mm)** is the mean joint Euclidean distance over valid matches,
   without root alignment or Procrustes alignment.
4. **Matched PA-MPJPE (mm)** applies per-person similarity Procrustes alignment
   (translation, rotation, and isotropic scale) before the same joint average.
   Degenerate/constant skeletons make PA-MPJPE undefined; exclude and count them.
5. **Root-localization error (mm)** is root-joint Euclidean distance on valid
   matches. Candidate root index 0 follows the legacy hip-alignment convention, but
   it remains provisional until the joint map is validated and frozen.
6. Report matched pose metrics together with all count, miss, duplicate, and person
   detection metrics below. Matched MPJPE alone can hide missed people and is not a
   sufficient headline result.

### PCK

Report **PCK@0.5** under the current provisional joint map: a matched joint is
correct when its unaligned Euclidean error is at most `0.5 * s`, where
`s = ||gt_joint_6 - gt_joint_4||_2`. This preserves the legacy PiW3D normalization
reference while making its threshold explicit. PCK is dimensionless; report it as a
percentage and aggregate over valid-match joints.

If `s < 1e-6` in stored coordinate units (equivalently `s < 0.001` mm), do not
divide by the scale or silently substitute a body-scale estimate. Mark that
person's PCK undefined, exclude it from the PCK denominator, and report the number
of guarded cases. Revisit the joint indices only after joint-order validation, with
a versioned metric definition.

### Counts, misses, duplicates, and person detection

- **Exact person-count accuracy:** percentage of samples with predicted count equal
  to ground-truth count.
- **Missed people:** ground-truth people without a valid match (false negatives).
- **Duplicate people:** unmatched above-threshold predictions whose nearest
  ground-truth person is already validly matched and lies within the same 500 mm
  mean-joint-distance rule. Other unmatched predictions are separate false-positive
  background/spurious slots.
- **Person precision:** `TP / (TP + FP)`; **person recall:** `TP / (TP + FN)`;
  **person F1:** their harmonic mean. TP is the number of valid one-to-one matches,
  FP includes duplicate and other unmatched predictions, and FN is missed people.
  Report zero denominators explicitly rather than hiding them.
- Report metrics overall and separately for one-, two-, and three-person scenes,
  with counts of samples and matched/guarded people.

**Oracle-count evaluation is diagnostic only.** If ground-truth count is used to
trim or select slots for an ablation, label that result `oracle-count`, keep it out
of headline comparisons, and never substitute it for inferred-count evaluation.

## 8. Legacy reproduction is a separate track

Legacy reproduction retains the existing implementation's stored-unit
MPJPE/PA-MPJPE (no `*1000`), its PCK thresholds and normalization, ground-truth
count trimming, and slot-wise pairing solely to reproduce baseline behavior. Do not
mix those numbers with corrected research metrics or rename them as millimetre
set-based results. Any controlled research comparison uses the corrected evaluator
above for every arm.

## 9. Required records and stopping rule

For each eventual run, record the protocol type, exact train/development IDs and
group hashes, label budget, seeds, source/config hashes, presence and matching
thresholds, package versions, commands, checkpoint-selection basis, and all metric
denominators/guard counts. Keep supervised-reference and low-label logs separate
from flagship logs.

This Step-3 task stops after documentation, bounded fixture verification, and the
one-step training smoke. Passing those engineering checks proves neither research
feasibility, pose accuracy, novelty, nor scientific readiness. Do not start the
pilot or full compositional implementation automatically.
