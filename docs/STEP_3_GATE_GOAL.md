# Approved Step-3 Goal

Continue CompWiFi-Pose from the existing YELLOW audit and close only the bounded Step-3 engineering and scientific-readiness gate. First read `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/CURRENT_TASK.md`, `docs/DATA_AUDIT.md`, and `docs/POSE_SPLIT_METRIC_AUDIT.md`. Preserve the baseline and make small, tested changes.

## 1. Correct the shared documentation before execution

- State that the paper documents a first-550/last-50 protocol, but released IDs do not verify its literal filename mapping.
- Record the conflict between the audit’s axis-2-vertical interpretation and the original authors’ axis-1-vertical per-axis evaluation. Do not reorder coordinates or declare either interpretation settled.
- Record that the original authors’ evaluation supports metre-to-millimetre conversion, while local legacy metrics currently omit it.
- Add WiFi-JEPA (arXiv:2607.11064), WiSPER (arXiv:2610.07025), WiFi-InstancePose, and *Self-supervised One-Stage Learning for RF-based Multi-Person Pose Estimation* to the literature/comparator review.
- Make no first-ever or Q1 claim. Use the precise claim “zero real multi-person training labels,” not “annotation-free learning.”

## 2. Create a deterministic two-frame real-training smoke fixture without modifying the dataset

- Select one eligible labelled one-person official-training ID whose immediate successor exists.
- Refactor preprocessing into a reusable single-ID function and an explicit-ID bounded entry point; do not call the current full `main()`.
- Process only those two raw training CSI files into an ignored temporary fixture root with the expected `train_data` structure and required keypoint copies.
- Verify amplitude shape `(3,90,20)`, phase shape `(3,90,20)`, `csi_ap` shape `(3,180,20)`, finite values, and exact `csi_ap` concatenation.
- Run the focused suite with `PIW3D_DATA_ROOT` pointing to the fixture.
- Acceptance: **8 passed, 0 skipped**. Do not weaken or skip tests.

## 3. Add a separate training-only smoke path; do not run `train_pose.py` unchanged

- Using only the fixture’s labelled one-person training sample, perform exactly one forward pass, backward pass, and optimizer step from scratch, with `num_workers=0`.
- Do not construct validation or test loaders, evaluate, or select checkpoints.
- Record the sample IDs, device, finite loss, finite gradients, and evidence that at least one parameter with a nonzero gradient changed.
- Any failure keeps the gate YELLOW or makes it BLOCKED with the exact error.

## 4. Write `docs/RESEARCH_PROTOCOL.md` without implementing the full research model

- Define development and strict protocols separately.
- Multi-person development labels, if used, may be used only for disclosed development evaluation/model selection, never training; the claim remains limited to training labels.
- Design—but do not execute—provisional candidate-filename-group-disjoint internal train/development manifests. Claim only candidate-group disjointness, not recording, subject, or environment generalization.
- Official test CSI/labels remain untouched.
- The unlabelled training pool must not expose poses/counts or use filename counts for filtering or balancing.
- Define an early feasibility pilot comparing single-person-only, reconstruction-only, and minimal compositional learning, with duplicate/mean-pose collapse controls, equal-label-budget baselines, and a separately labelled fully supervised upper reference.
- Predeclare development-only go/no-go criteria.
- Specify corrected research metrics separately from legacy reproduction: Hungarian matching; matched MPJPE/PA-MPJPE with explicit units; documented PCK with a near-zero-scale guard; exact count accuracy; missed/duplicate people; person precision/recall/F1 under a documented matching rule; and root error. Oracle-count evaluation is diagnostic only.

## Completion and stop rule

Complete only these four tasks, update `docs/CURRENT_TASK.md` with the resulting gate status, and report changed files, commands actually run, test counts, smoke evidence, and unresolved scientific risks. Mark the gate complete only when every acceptance check passes; otherwise leave it YELLOW or mark it BLOCKED with the exact error and stop rather than widening scope.

Do not install/remove packages, download files, delete data, run full preprocessing/training, read official test arrays, access credentials, commit/push, or change remotes. Do not start the full compositional model or pilot. Passing engineering tests does not prove feasibility, accuracy, novelty, or scientific readiness.
