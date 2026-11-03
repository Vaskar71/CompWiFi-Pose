# CompWiFi-Pose — Permanent Agent Instructions

## Required context

Before starting work, read:
1. AGENTS.md
2. docs/PROJECT_CONTEXT.md
3. docs/CURRENT_TASK.md

Treat docs/CURRENT_TASK.md as the current source of truth for progress.
Treat docs/PROJECT_CONTEXT.md as the research and evaluation specification.

## Research objective

Develop multi-person 3D human pose estimation from WiFi CSI without
using real multi-person CSI-pose training labels in the flagship
zero-label experiment.

The model should predict:
- A variable-sized set of 3D human skeletons.
- A presence score for every prediction slot.
- Person count implicitly through presence predictions.

## Starting point

- Start from the existing DT-Pose repository.
- Preserve existing baseline functionality.
- Add research functionality behind new configuration options.
- Do not rewrite unrelated baseline code.
- Do not reorganize the repository unless explicitly requested.

## Label-access rules

For the flagship zero-label experiment:

Allowed:
- Labelled single-person CSI-pose pairs from the training partition.
- Unlabelled CSI from the training partition.
- Synthetic data only in explicitly marked experiments.

Forbidden:
- Real multi-person pose labels in training.
- Real multi-person person-count labels in training.
- Test samples in training or pretraining.
- Test labels for checkpoint selection or hyperparameter tuning.
- Indirect label access through caches, filenames, preprocessing outputs,
  validation code, or dataset wrappers.

Single-person metadata may be used to construct the labelled
single-person training subset.

The unlabelled training loader must not expose pose or person-count
labels, even if those values exist in the original files.

## Validation protocols

Development protocol:
- Multi-person validation labels may be used for development evaluation
  and model selection if this is explicitly recorded and disclosed.
- They must never enter training.

Strict protocol:
- No multi-person labels may be used for training, validation,
  checkpoint selection, or hyperparameter selection.
- Hyperparameters must be fixed using single-person validation,
  synthetic validation, or previously frozen development decisions.

Do not confuse these protocols in reports.

## Data integrity

- Inspect the real dataset before writing schema-dependent code.
- Do not assume .mat variable names, tensor shapes, joint ordering,
  coordinate units, or directory structure.
- Do not modify or delete original dataset files.
- Keep datasets outside the Git repository.
- Use configurable dataset paths.
- Record missing files, corrupted files, NaNs, and unsupported metadata.
- Do not claim that recording-disjoint, subject-disjoint, or
  environment-disjoint splits exist unless metadata supports them.

## Low-label experiments

- Prefer recording-level or sequence-level label-budget sampling.
- Do not randomly select isolated adjacent frames when this would leak
  near-duplicate samples across labelled and unlabelled subsets.
- Record the exact sample and recording IDs used for every label budget.

## Engineering workflow

For every task:
1. Read the relevant existing implementation.
2. State the intended files and changes.
3. Make the smallest reasonable change.
4. Add or update tests.
5. Run the tests.
6. Run a small smoke test before any full training.
7. Report actual commands and outputs.
8. Update docs/CURRENT_TASK.md.

Never claim that a command, experiment, or test ran successfully unless
its actual output confirms it.

Do not invent results, metrics, dataset metadata, or baseline performance.

## Required smoke tests

Before full model training, verify:
- One batch loads correctly.
- Input and label shapes are correct.
- Forward pass succeeds.
- Loss is finite.
- Backward pass succeeds.
- Optimizer updates parameters.
- Checkpoint saves and reloads.
- Evaluation runs.
- A tiny subset can be overfit, where applicable.

## Evaluation requirements

Use the same evaluation implementation for controlled comparisons.

Report:
- MPJPE.
- PA-MPJPE.
- PCK with its exact threshold and normalization definition.
- Root-localization error.
- Exact person-count accuracy.
- Missed-person and duplicate-person counts.
- Person-level precision/recall/F1 when a matching rule is defined.

Use Hungarian matching where appropriate.

Do not use matched-pose MPJPE alone, because it can hide missed people.

## Model-development restrictions

- Start with the simplest compositional prototype.
- Do not begin with a large temporal Transformer or differentiable
  ray-tracing renderer.
- Do not create pseudo-multi-person data by directly adding raw CSI arrays.
- Add temporal, simulation, and physics components separately so their
  contributions can be ablated.
- Do not tune against the final test set.

## Permission requirements

Ask before:
- Installing or removing packages.
- Downloading the full dataset.
- Starting full training.
- Deleting files.
- Changing the Git remote.
- Publishing or uploading anything.

## Completion reporting

After each task, report:
- What was completed.
- Files changed.
- Commands actually run.
- Test results.
- Remaining errors and uncertainties.
- Updated current status.
- Recommended next task.

Do not automatically begin the next phase.