# Step-4A Goal

Implement Step-4A for CompWiFi-Pose: label-safe data infrastructure and corrected research metrics, not the research model.

Read `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/CURRENT_TASK.md`, `docs/RESEARCH_PROTOCOL.md`, `docs/STEP_3_GATE_GOAL.md`, `docs/DATA_AUDIT.md`, and `docs/POSE_SPLIT_METRIC_AUDIT.md`. Treat Step 3 as GREEN only for its reported engineering checks. Preserve legacy behavior behind existing paths; put research functionality in new modules or options.

1. Bounded source check: make one bounded check of existing upstream references for coordinate units, vertical axis, and filename/group semantics. Do not guess or wait indefinitely. If unresolved, record the uncertainty and continue with explicit configurable conventions. Do not reorder coordinates or claim recording-, subject-, or environment-level semantics without metadata support.

2. Provisional manifest generator: implement a deterministic generator using only the official training list. Assign whole candidate filename groups, defined by the first two ID tokens, to internal train/development partitions without person-count stratification. Write generated manifests and summaries only to an ignored artifact location. Validate that every official-training ID is assigned exactly once across the two partitions; train/development ID overlap is zero; candidate-group overlap is zero; immediate-next-frame pairs cannot cross partitions; and no official test list or array is read. Label these partitions only “candidate-group-disjoint.” Keep the official test split untouched.

3. Flagship unlabelled-training loader: implement a separate CSI dataset/loader for flagship unlabelled training. It may read only allowed training CSI and must not read keypoints; return poses or `person_num`; expose raw IDs to training batches; or filter, balance, or sample using filename-derived person counts. Keep labelled single-person and development-evaluation loaders separate. Add tests proving unlabelled loading works when keypoint files are absent; mixed 1/2/3-person filename fixtures are not count-filtered; and batches contain only declared CSI tensors. Reuse the existing two-frame fixture where appropriate; do not preprocess more real data.

4. Corrected research metrics: implement research set metrics in a new module without silently changing legacy `utils.py`. Follow `docs/RESEARCH_PROTOCOL.md`; make ambiguous choices explicit and configurable. Use Hungarian person matching rather than GT-count trimming or slot pairing. Include matched MPJPE and PA-MPJPE with an explicit stored-unit-to-millimetre scale; documented PCK normalization, thresholds, and near-zero-scale handling; exact person-count accuracy; missed-person and unmatched/spurious-prediction counts; person precision, recall, and F1 under the documented matching rule; root error with an explicit configurable root definition; and correct global aggregation across unequal batches. If oracle-count evaluation is present, expose it only as a separately named diagnostic.

5. Deterministic tests and documentation: add deterministic synthetic tests covering perfect predictions, person permutation, missing people, extra/spurious predictions, unequal counts, root error, millimetre conversion exactly once, PCK near-zero scale, Hungarian matching, and aggregation across unequal batches. Update `docs/CURRENT_TASK.md`. Write `docs/STEP_4A_REPORT.md` containing changed files, exact commands, actual test counts and results, manifest checks, label-guard evidence, metric definitions, and unresolved assumptions.

Passing these tests establishes infrastructure correctness only, not model feasibility, accuracy, novelty, or scientific readiness.

Boundaries: do not install or remove packages; download data; delete files; run full preprocessing or training; implement the compositional model; run the feasibility pilot; evaluate official test data; access credentials; commit or push; or change remotes. Stop after Step-4A. If a required check is blocked, stop and report the blocker rather than weakening assertions, fabricating evidence, or widening scope. Unresolved upstream conventions alone do not block progress: document them and use explicit configurable conventions as specified above.

## Completion criterion

All Step-4A implementation and documentation above is complete, every manifest invariant has been validated, and the complete focused suite has been run with `PIW3D_DATA_ROOT` pointing to the existing fixture. The original eight tests and every new test pass, with zero skips and no weakened assertions.
