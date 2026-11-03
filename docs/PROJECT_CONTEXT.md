# CompWiFi-Pose — Complete Project Context

## 1. Project title

CompWiFi-Pose: Multi-Person 3D Pose Estimation Without Real
Multi-Person Training Labels.

## 2. Research problem

Current multi-person WiFi pose systems require expensive synchronized
CSI and human-pose annotations.

The research question is:

Can two- and three-person 3D pose be learned using labelled
single-person CSI-pose pairs and unlabelled multi-person CSI, without
using real multi-person pose labels for training?

The intended contribution is not generic self-supervised learning.
The specific contribution is removing or greatly reducing paired
multi-person supervision through compositional scene learning.

## 3. Important novelty boundaries

Do not describe the project as merely:
- A WiFi foundation model.
- Generic self-supervised WiFi pose estimation.
- Adding a Transformer.
- Generic cross-environment learning.
- Fully supervised multi-person pose improvement.

Literature anchors:
- Person-in-WiFi 3D: supervised multi-person set-based pose estimation.
- DT-Pose: self-supervised representation learning followed by
  supervised pose decoding.
- WiFi-JEPA: label-free CSI pretraining, followed by supervised
  downstream pose learning.
- WiSPER: predictive pretraining and residual refinement using paired
  CSI-pose supervision.

Our precise gap:
Learning multi-person pose without real multi-person training labels,
or with extremely small multi-person label budgets.

This is a research hypothesis, not a guarantee that the method will work
or that no concurrent work exists.

## 4. Public resources

Base code:
https://github.com/cseeyangchen/DT-Pose

Original Person-in-WiFi 3D code:
https://github.com/aiotgroup/Person-in-WiFi-3D-repo

Public dataset entry:
https://www.kaggle.com/datasets/xjtuqianbo/wiception3d

WiFi-JEPA paper:
https://arxiv.org/abs/2607.11064

WiSPER paper:
https://arxiv.org/abs/2610.07025

The public Wiception3D package may not exactly match the package expected
by DT-Pose. Inspect and document compatibility instead of assuming it.

## 5. Local environment

- OS: Windows.
- Environment manager: Anaconda.
- Editor: VS Code.
- Coding agent: Kimi Code Desktop.
- GPU: RTX 3080 16 GB.
- Dataset location: not configured yet.
- Conda environment: not configured yet.
- Repository baseline commit: not recorded yet.

The GPU should be used first for smoke tests and moderate baselines.
Do not assume that every published full-scale configuration fits in
16 GB. Use smaller batches or mixed precision only after inspecting
the implementation, and record every such change.

## 6. Current verified status

Planning has been completed.

Not yet verified:
- Whether the repository runs locally.
- Whether the dataset has been downloaded.
- Whether the public dataset matches DT-Pose preprocessing.
- Whether any baseline has been reproduced.
- Whether recording, subject, and environment metadata are available.
- Whether the proposed zero-label model works.

No experimental result should be assumed to exist.

## 7. Exact model task

Input:
A WiFi CSI observation or temporal CSI window.

Output:
A set of candidate predictions:
{(Y_hat_q, presence_q)}, q = 1...Q.

Y_hat_q is a 3D skeleton.
presence_q indicates whether the slot represents a real person.

The person count must not be supplied as an input at inference.

Initial scope:
- PiW3D/Wiception3D.
- 3D skeletons, not meshes.
- Fixed dataset hardware.
- At most the maximum person count supported by the dataset.
- No cross-device or cross-band claims.

## 8. Training-data roles

single_labeled_train:
- CSI and single-person pose labels available.
- May be used for single-person pose supervision.

unlabeled_train:
- Only CSI available to the model.
- Pose and person-count labels forbidden.
- May be used for masking, reconstruction, latent prediction,
  compositional consistency, or temporal consistency.

development_val:
- Used for development evaluation and model selection only.
- Label use must be disclosed.
- Never used for training.

official_test:
- Reserved for final evaluation.
- Not used for pretraining, training, model selection, or iterative tuning.

supervised_full_train:
- Used only for fully supervised upper-reference experiments.

## 9. Experimental settings

Main setting:
- 0% real multi-person training labels.

Fallback and label-efficiency settings:
- 0.1%
- 1%
- 5%
- 10%
- 100% fully supervised reference

Label budgets should be sampled by recording or sequence when metadata
allows. Do not use near-duplicate adjacent frames to make a low-label
setting artificially easy.

## 10. Required baselines

Where implementation permits:

B0. Fully supervised Person-in-WiFi 3D reference.
Purpose: upper-reference performance, not a fair zero-label competitor.

B1. Single-person-only transfer.
Same model family trained using only labelled single-person data.

B2. Self-supervised pretraining plus single-person-only supervision.
Uses the same unlabelled training CSI as our method.
This is a critical controlled comparison.

B3. Generic consistency or teacher-student learning.
Uses the same labelled and unlabelled resources.

B4. Our compositional model without individual components.
These are the ablation baselines.

B5. Low-label supervised training.
Used for 0.1%, 1%, 5%, and 10% comparisons.

Published numbers from different splits must not be presented as direct
controlled comparisons.

## 11. Model hypothesis

Start with a minimal compositional scene-to-set prototype:

1. Reuse the existing CSI encoder.
2. Pretrain it with an established CSI self-supervised objective.
3. Train a shared single-person pose decoder using single-person labels.
4. Add a slot-based scene-decomposition module.
5. Use one background slot and enough human slots for the maximum
   supported person count.
6. Apply the shared pose decoder to each human slot.
7. Predict a presence score for every human slot.
8. Reconstruct observed CSI features from the combined slots.
9. Train unlabelled scenes using reconstruction or latent consistency.

Possible losses:
- Single-person pose loss.
- CSI self-supervised loss.
- Feature-space scene-composition loss.
- Temporal consistency loss, added later.
- Human pose-prior loss, added later.
- Duplicate-prediction suppression, added later.

No multi-person pose loss is allowed in the zero-label setting.

Feature-space composition is the first approach.
Do not begin with a full differentiable RF renderer.
Do not directly add raw CSI arrays to simulate multiple people.

## 12. Evaluation specification

Report separately for one-, two-, and three-person scenes where relevant:
- MPJPE.
- PA-MPJPE.
- PCK with an explicitly documented definition.
- Root-localization error.
- Person-count accuracy.
- Missed people.
- Duplicate predictions.
- Person-level precision, recall, and F1 if a matching threshold is defined.

Use Hungarian matching for set-based evaluation where appropriate.

Inspect failure modes:
- Slot collapse.
- Only one active human slot.
- Duplicate people.
- Background predicted as a person.
- Plausible skeletons unrelated to CSI.
- Good reconstruction but poor pose.
- Performance degradation with increasing person count.

Final important experiments should use at least three seeds and report
mean and standard deviation.

Record inference latency, parameter count, and peak GPU memory when
the implementation is stable.

## 13. Development phases

Phase 0: Repository and environment inspection.
Exit when dependencies, paths, commands, and compatibility issues are known.

Phase 1: Dataset audit.
Exit when real CSI and pose files can be loaded and visualized correctly.

Phase 2: Baseline reproduction.
Exit when preprocessing, training, checkpointing, and evaluation work
end-to-end and a tiny-overfitting test succeeds.

Phase 3: Leakage-safe benchmark.
Exit when manifests, label-access guards, metrics, and leakage tests work.

Phase 4: Minimal compositional feasibility experiment.
Exit with evidence about whether slot decomposition is viable.

Phase 5: Full method development.
Add composition, pose priors, temporal consistency, and optional simulation
incrementally with ablations.

Phase 6: Final experiments.
Run controlled baselines, label budgets, seeds, ablations, efficiency,
qualitative evaluation, and failure analysis.

Phase 7: Paper and reproducibility package.
Release configurations, manifests, evaluation code, commands, and actual
results. Do not redistribute data unless permitted.

## 14. Phase 4 decision gate

Green:
The prototype beats the equally supervised single-person/SSL baselines
on development validation. Continue method development.

Yellow:
Pose improves but person counting or slot decomposition fails.
Focus on presence prediction, slot separation, and duplicate suppression.

Red:
Slots collapse and reasonable debugging produces no improvement.
Move to 0.1-1% multi-person-label learning within the same project.

Do not hide a failed zero-label result by silently changing the task.

## 15. Initial scope exclusions

Do not currently work on:
- Mesh reconstruction.
- Cross-device generalization.
- Cross-band generalization.
- New hardware.
- New data collection.
- Real-time deployment.
- Full physics ray tracing.

These can be considered only after the core result works.

## 16. Unknowns that must not be guessed

- Actual dataset schema.
- Actual joint ordering.
- Actual coordinate units.
- Actual person-count representation.
- Actual recording metadata.
- Compatibility of public data with DT-Pose.
- Local package and CUDA compatibility.
- Baseline reproduction accuracy.
- Whether zero-label compositional learning is viable.

Inspect, measure, and record these instead of assuming them.