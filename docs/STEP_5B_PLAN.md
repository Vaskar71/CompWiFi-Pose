# CompWiFi-Pose — Step 5b Execution Plan: pose signal beyond recording identity

Written 2026-11-10 for an executor agent that has **not** seen the planning conversation.
Read `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/CURRENT_TASK.md`, `docs/RESEARCH_PROTOCOL.md`,
`docs/NEXT_PHASE_PLAN.md`, `docs/STEP_5_PROTOCOL.md` and `docs/STEP_5_REPORT.md` first. They stay
binding. This plan inserts one step between Phase B (Step 5, done) and Phase C (Step 6, not started).
**Phase C must not start until this step has a verdict and the user has decided.**

"Recording" below always means a *candidate filename group* (e.g. `S11_06`). No subject, session or
environment claim is made.

---

## 1. Context

**Where Phase B ended.** Step 5 verdict `YELLOW-POSITION`: the single-person baseline learns (G1–G5
pass for every run) but on unseen recordings (G6) its centroid-relative body pose is worse than the
train-mean skeleton in every run and seed (1.09–1.25×). Phase B offered two follow-ups: (a) Phase C under
the within-recording temporal protocol, (b) a data-scaling run.

**Why neither is taken as written.** Read-only checks after Phase B (on the saved Step-5 predictions,
`data_v1` CSI and the dense single-person training labels) found:

1. **On seen recordings the model does not track the person.** On `temporal_holdout`, predicting each
   recording's own train-mean pose ("recording-mean oracle") gives abs MPJPE **0.139** / rel **0.095**.
   R1 gives 0.204 / 0.102, R2 gives 0.231 / 0.104. Within-recording correlation between predicted and
   true deviations: median 0.05 (R1), 0.04 (R2). The model beats its own recording-mean oracle in 16 %
   (R1) / 10 % (R2) of recordings. G5 passed only because the model recognises the recording.
2. **Recordings are easy to recognise from raw CSI.** Assigning each `temporal_holdout` frame to the
   recording with the nearest mean stored amplitude block is right for **72 %** of frames (chance 2 %).
3. **The G6 failure is not missing position coverage.** Every `group_dev` recording has a training
   recording whose mean centroid is within 0.10–0.24 stored units, yet R2's per-recording centroid error
   is 0.14–1.31, and body pose is worse than the mean skeleton in 5 of 8 dev recordings.
4. **Pose motion inside a recording is real and fast.** From all labelled frames of the 51 single-person
   internal-train recordings, the relative-pose deviation from the recording mean has autocorrelation
   (median over recordings) **0.94 / 0.83 / 0.59 / 0.17 / −0.08** at frame lags 1 / 2 / 4 / 8 / 16.
   Mean per-joint deviation within a recording is 0.092; between recording mean skeletons 0.104.
5. **Step 5 could not have learned that motion.** Its `train` split takes 64 evenly spaced frames per
   recording, about 8 frames apart, where the pose is already almost decorrelated (0.17). Each recording
   contributed ~64 nearly independent poses.
6. **Phase C needs exactly what is missing.** Pose labels come only from single-person recordings; the
   multi-person recordings to be evaluated are always *different* recordings, however frames are split.
   So option (a) does not avoid cross-recording transfer, and option (b) is necessary (dense frames) but
   was not framed as a test of anything.

**Question this step answers.** Does CSI carry a learnable body-pose signal beyond recording identity —
first within seen recordings (gate T1), then on unseen recordings (gate X1)?

**Intended outcome.** One bounded, predeclared step with a three-way verdict that decides whether
Phase C is run as planned, redesigned, or replaced by the label-efficiency track.

---

## 2. Ground rules for the executor

- Everything in `docs/NEXT_PHASE_PLAN.md` §2 still applies (no package installs, stored units, centroid
  as translation reference, candidate-filename-group wording only, never official `test_data`, never
  `git add -A`, never `git stash pop`, real command output only, ask-first list).
- **Code location.** Create branch `research/step5b-pose-signal` **from
  `research/step5-single-person-baseline`** (commit `11fad75`; that branch is local only and is *not*
  merged into `main`). Work in whatever checkout the session gives you. Data and artifacts live only in
  the main checkout: pass `--repo-root "J:/Zero-shot 3D Multi-person Pose Estimation"` to every script,
  exactly as the Step-5 scripts do.
- **Environment.** `C:\Users\vaska\venvs\compwifi-piw3d-py311\Scripts\python.exe`; start processes with
  `PYTHONHASHSEED=0` and `CUBLAS_WORKSPACE_CONFIG=:4096:8`.
- **Test baseline on the Step-5 branch (observed 2026-11-10 in a worktree):** `118 passed, 1 skipped,
  1 failed`. The one failure is
  `tests/test_step4b_d4_reproducibility.py::test_d4_protected_records_include_d3_stop_artifacts_and_match`,
  which looks up `data/_artifacts` by relative path and therefore fails outside the main checkout. Any
  other failure → stop and report. Command:
  `PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`
- **Additional protected files (do not edit):** every `step5_*.py`, `make_step5_data.py`,
  `summarize_step5.py`, `feeder/piw3d_step5.py`, `model/research_models.py`, `docs/STEP_5_*`, and
  everything under `data/_artifacts/step5_single_person/`. The Step-5 runner hashes its sources. Add new
  files; import from the old ones.
- **One protocol doc + one runner + one report.** No follow-up experiments beyond the matrix in §5. If a
  gate fails, stop, write the report, ask.
- **Line endings.** Write new files with LF. Before freezing hashes, confirm
  `git ls-files --eol` shows `w/lf` for the hashed files (Phases A and B both lost time to CRLF).
- **[ASK] points:** (1) before the dense preprocessing in §4 (≈4.5 GB, ≈40 min), unless the user's
  kickoff message already approves it; (2) if the first epoch of any run projects more than 1 hour for
  that run; (3) before pushing or merging anything.
- After the step: completion report per `AGENTS.md`, update `docs/CURRENT_TASK.md`, then **stop**.

---

## 3. Step 0 — record the premises (read-only, CPU, minutes)

1. Copy this plan verbatim into the branch as `docs/STEP_5B_PLAN.md` (source:
   `J:\Zero-shot 3D Multi-person Pose Estimation\.claude\worktrees\zero-shot-3d-pose-plan-38e14a\docs\STEP_5B_PLAN.md`,
   fallback `C:\Users\vaska\.claude\plans\compwifi-step5b-plan.md`). Do not paraphrase it.
2. Add `analyze_step5_recording_effects.py` + `tests/test_step5b_recording_effects.py` (unit-test the
   helpers on synthetic arrays). It reads only existing artifacts and single-person training-partition
   labels, and writes `data/_artifacts/step5b_pose_signal/premises/premises.json` plus the tables for
   the report. It must reproduce, within rounding:
   - recording-mean-oracle comparison on `temporal_holdout` for `R1_seed0` and `R2_seed0`
     (oracle 0.139 / 0.095; R1 0.204 / 0.102; R2 0.231 / 0.104; share of recordings beating the oracle
     16 % / 10 %). Oracle = mean of that recording's poses in `predictions_train.npz` ground truths;
     recording = `sample_id.rsplit("_", 1)[0]`;
   - nearest-recording-mean identification on stored `csi_ap` amplitude rows `[:, :90, :]` (72 %);
   - pose autocorrelation by frame lag from `wifi_dataset/train_data/keypoint/<id>.npy` over all frames
     of the 51 single-person internal-train recordings (0.94 / 0.83 / 0.59 / 0.17 / −0.08 at lags
     1 / 2 / 4 / 8 / 16), and the within/between deviations (0.092 / 0.104).
   If these do not reproduce, **stop and report**: the premises of this plan would be wrong.

---

## 4. Data: dense freeze `freeze_v2` and `data_v2`

New files `step5b_subset.py`, `make_step5b_data.py`, tests. Read `step5_subset.py` and
`make_step5_data.py` first and reuse their pattern and helpers by import (`single_person_groups`,
`holdout_layout`, the freeze/validation layout, `build_step5_data`'s copy → `process_ids` → validate
flow). Sources: the same internal manifests and `source_root` as Step 5.

**Frame roles**, for each of the 51 single-person internal-train recordings, frames sorted by frame
index, `n` = count:
- holdout block = sorted positions `[n//5, n//5 + h)`, `h = min(49, n//10)` — identical to Step 5;
- guard = **16** positions on each side of the block, excluded from everything (Step 5 used 5; the pose
  autocorrelation is 0.59 at lag 4 and ≈0 at lag 16, so 16 removes leakage from dense neighbours);
- role `temporal_holdout` = **every** frame in the block; role `train` = every remaining frame.

**Recording partition.** Within each scene prefix (`S11`, `S21`, `S51`), sort recording names by
`hashlib.sha256(name.encode("utf-8")).hexdigest()` ascending; the first 3 are **validation recordings**.
Expected: `S11_32, S11_03, S11_10, S21_07, S21_29, S21_27, S51_11, S51_32, S51_23`. The other 42 are
**fit recordings**. The selection reads names only and never branches on the person-count character.

**ID lists (views)** written to `data/_artifacts/step5b_pose_signal/freeze_v2/`:

| View | Content | Expected frames |
|---|---|---|
| `fit_train_dense` | `train`-role frames of the 42 fit recordings | 16,235 |
| `fit_train_sparse` | up to 64 evenly spaced `train`-role frames per fit recording (`select_evenly_spaced_frames`) | 2,687 |
| `fit_holdout` | `temporal_holdout`-role frames of the 42 fit recordings | 1,815 |
| `group_val` | `train`-role frames of the 9 validation recordings | 3,775 |
| `all_train_dense` | `train`-role frames of all 51 recordings | 20,010 |
| `all_train_sparse` | up to 64 evenly spaced `train`-role frames per recording, all 51 (used only if D1 is selected) | 3,263 |
| `all_holdout` | `temporal_holdout`-role frames of all 51 recordings | 2,232 |
| `group_dev` | the 512 frozen Step-5 `group_dev` IDs, read from `freeze_v1/group_dev_ids.txt` and verified against its recorded SHA-256 | 512 |

Guard frames (1,632) are not preprocessed. If any expected number differs, stop and report.
`split_assignments.json` maps every ID to its role (`train`, `temporal_holdout`, `group_dev`); views are
ID lists over those roles, so `SinglePersonPiW3DDataset(expected_split=...)` keeps working.

**Preprocessing [ASK].** `process_ids` into `data/_artifacts/step5b_pose_signal/data_v2/`
(22,242 + 512 = 22,754 samples; ≈196 KB each → ≈4.5 GB; Step 5 measured 445 s for 4,580 samples →
≈40 min; 68 GB free on J:). Originals untouched. Validate shape and finiteness of every output.

**Statistics, label-free unless stated, saved in `freeze_v2/` and never overwritten:**
- `input_stats_{fit,all}.npz`: mean/std `(3,180,1)` from `fit_train_dense` / `all_train_dense`
  (`compute_train_input_stats`);
- `codec_stats_{fit,all}.json`: `compute_codec_stats` on the same two views (uses single-person
  training labels — permitted);
- `recording_input_means.npz`: per recording, mean of stored `csi_ap` over frames and time, shape
  `(3,180,1)`. For the 51 training recordings use their `train`-role frames; for the 8 `group_dev`
  recordings use their own 64 frames. This is label-free but **transductive** for unseen recordings:
  record `uses_label_free_per_recording_input_statistics: true` in the config and disclose it.

Tests: view disjointness where required (`fit_*` vs `group_val`; train vs holdout vs guard),
`fit_train_sparse ⊂ fit_train_dense`, guard respected, no multi-person ID, no ID outside the internal
manifests, deterministic re-selection, expected validation recordings.

---

## 5. Models, inputs and the frozen run matrix

New files `step5b_config.py`, `feeder/piw3d_step5b.py`, `step5b_pose_signal.py`, tests.

**Model.** `simple_cnn` only, built with
`model.research_models.build_research_model("simple_cnn", codec_stats, step5_config_for_json(), device)`.
No ViT, no temporal module, no new architecture.

**Input normalisations** (`Step5bDataset(Step5SinglePersonDataset)` overriding `read_csi`):
- `legacy_minmax` and `train_zscore` — inherited unchanged (`train_zscore` uses `input_stats_fit` in
  stage 1 and `input_stats_all` in stage 2);
- `recording_centered` — `(x − μ_recording) / max(σ_train, 1e-6)`, with `μ_recording` from
  `recording_input_means.npz` and `σ_train` the `train_zscore` std of the stage. The recording of a
  sample is resolved once in the dataset constructor from its ID (`parse_sample_id(...)["group"]`),
  never inside the training loop.

**Optimiser recipe.** Identical to Step 5 (`STEP5_CONFIG["training"]`): AdamW, peak lr 3e-4, weight
decay 0.01, 200-step linear warmup → cosine, grad clip 1.0, batch 32, 40 epochs, final-epoch checkpoint,
same determinism block. Reuse by import from `step5_single_person_baseline`: `CachedSplit`,
`pose_statistics`, `research_metrics`, `predict`, `lr_factor`, `make_optimizer`, `training_step`,
`setup_runtime`, `_write_json`. Do not call its `load_splits` / `run_training` (they are bound to the
Step-5 config).

**Stage 1 — selection on validation recordings (seed 0, trained on the 42 fit recordings):**

| Run | Input | Training view | Isolates |
|---|---|---|---|
| D1 | `legacy_minmax` | `fit_train_sparse` | reference: Step-5 R2 recipe at 42 recordings |
| D2 | `legacy_minmax` | `fit_train_dense` | frame density |
| D3 | `train_zscore` | `fit_train_dense` | normalisation under dense data |
| D4 | `recording_centered` | `fit_train_dense` | removing the recording signature |

Evaluated every epoch and at the end on: the run's training view, `fit_holdout`, `group_val`.

**Stage 2 — confirmation (selected run only, seeds 0, 1, 2, trained on all 51 recordings):** training
view `all_train_dense` (`all_train_sparse` if D1 is selected), evaluated on
`all_holdout` and **once** on `group_dev`. `group_dev` is not touched in stage 1.

**Order of execution:** write `docs/STEP_5B_PROTOCOL.md` → commit → freeze protocol (config JSON +
source hashes, as in Step 5) → smoke checks (AGENTS.md list; untrained model equals train-mean pose) →
overfit check on 64 samples of `fit_train_dense` for D2 and D4 (300 steps, lr 1e-3, pass ≤ 0.20 ×
mean-pose MPJPE of the same samples) → stage 1 (D1–D4) → selection → stage 2 → verdict → report.
Expected GPU time ≈ 10–15 min per dense run, ≈ 1.5 h in total.

---

## 6. Metrics, gates, selection, verdict (write into `STEP_5B_PROTOCOL.md` before any run)

For every run and evaluation view report: abs MPJPE, centroid error, centroid-relative ("rel") MPJPE,
collapse monitors, `ResearchMetricAccumulator` numbers (people dim 1, provisional mm), plus:

- **Recording-mean oracle** (seen recordings): predict the mean pose of that recording's frames in the
  run's training view. Report its abs and rel MPJPE on the holdout view.
- **Mean-pose / mean-skeleton baselines** (unseen recordings): from the run's training view, as in
  Step 5.
- **Motion skill** (diagnostic, every evaluation view): per recording, subtract the recording's mean
  relative skeleton from the ground truth and the recording's mean *predicted* relative skeleton from the
  prediction; `skill = 1 − mean‖Δpred − Δgt‖ / mean‖Δgt‖`, averaged over frames. Positive = predicted
  motion follows true motion. Uses evaluation labels for evaluation only.

**Gates**
- **S** — all smoke checks pass; overfit ratio ≤ 0.20.
- **T1, tracking on seen recordings** — on the holdout view: `rel MPJPE ≤ 0.95 × recording-mean-oracle
  rel MPJPE`. (Abs ratio to the oracle is reported, not gated.)
- **X1, body-pose transfer to unseen recordings** — on `group_val` (stage 1) / `group_dev` (stage 2):
  `rel MPJPE ≤ 0.95 × mean-skeleton rel MPJPE`. The absolute ratio to the mean pose (Step-5 G6 abs,
  ≤ 0.95) is reported next to it but is **not** part of the verdict, because `recording_centered`
  removes static position information by construction.

**Selection (stage 1, seed 0).** Among runs that did not crash and whose training-view
`pred_std_across_samples` is ≥ 0.01: lowest `group_val` rel-MPJPE ratio to the mean skeleton. Ties
(< 1 % relative) → simpler run, order D1 < D2 < D3 < D4. Selection uses labels of single-person
*training-partition* recordings only.

**Verdict (stage 2, all three seeds of the selected run)**
- **GREEN** — T1 and X1 pass in all three seeds. Phase C may be proposed with the selected input and
  data recipe and group-disjoint evaluation as primary.
- **YELLOW-TRACKS** — T1 passes in all three seeds, X1 fails in at least one. Pose sensing exists but is
  recording-specific. Stop. Recommend to the user: redesign Phase C around label-free adaptation to the
  target recordings (their unlabelled CSI is permitted), or switch to the label-efficiency track.
- **RED-NO-TRACKING** — the selected run fails T1 in all three seeds **and** no stage-1 run passed T1 on
  `fit_holdout`. Even with dense labels from the same recording the model cannot beat the recording
  mean. Stop. The only permitted follow-up inside this step is a *read-only* audit of CSI↔label time
  alignment (frame index vs CSI window; `process_ids` output vs raw `.mat`). Recommend, as a separate
  task needing approval, reproducing the upstream DT-Pose recipe on the official split as a sanity check.
- **INCONCLUSIVE** — anything else (e.g. T1 passes in some seeds, or a stage-1 run passed T1 but the
  selected run does not). Report every number and ask.

`docs/STEP_5B_REPORT.md` must contain: the premises table (§3), every stage-1 and stage-2 run including
failed ones, what D2 vs D1 shows (density), what D4 vs D2 shows (recording signature), motion skill on
seen and unseen recordings, the verdict, and the disclosure lines (development labels used only for the
single stage-2 confirmation; transductive per-recording input statistics where D4 is involved; stored
units; candidate-filename-group wording).

---

## 7. Critical files

Reuse (import, do not edit): `step5_config.py` (`step5_config_for_json`, recipe and model blocks),
`step5_subset.py` (`single_person_groups`, `holdout_layout`), `make_step5_data.py`
(`build_step5_data` pattern), `feeder/piw3d_step5.py` (`Step5SinglePersonDataset`,
`compute_train_input_stats`, `save_input_stats`, `load_input_stats`), `model/research_models.py`
(`build_research_model`, `compute_codec_stats`, `save_codec_stats`, `load_codec_stats`),
`step5_single_person_baseline.py` (helpers listed in §5), `step4b_subset.py`
(`select_evenly_spaced_frames`, `is_single_person_id`), `piw3d_manifest.py` (`parse_sample_id`),
`data/person_in_wifi_3d/process.py` (`process_ids`), `research_metrics.py`.

New: `analyze_step5_recording_effects.py`, `step5b_config.py`, `step5b_subset.py`,
`make_step5b_data.py`, `feeder/piw3d_step5b.py`, `step5b_pose_signal.py`, `tests/test_step5b_*.py`,
`docs/STEP_5B_PLAN.md`, `docs/STEP_5B_PROTOCOL.md`, `docs/STEP_5B_REPORT.md`; edit
`docs/CURRENT_TASK.md` (status block and progress entry only).

---

## 8. Verification

- Tests after every code change; count ≥ baseline + new tests; only the one known environment failure.
- Premises reproduce (§3) before anything else is built.
- Freeze counts match the table in §4; protocol freeze timestamp precedes the first run; source hashes
  still verify at the end.
- Untrained-model-equals-mean-pose test passes; every run's reloaded state dict reproduces its final
  predictions exactly; a crashed run fails every gate and still writes its decision record.
- Label access: no multi-person ID anywhere; `group_dev` opened only by stage 2; official `test_data`
  never opened (assert by monkeypatching file opens in a test); training loop sees only
  `{"csi","pose"}` batches.
- Final message: completion report per `AGENTS.md` (what was done, files changed, commands actually
  run, test results, uncertainties, status, recommended next task) and updated `docs/CURRENT_TASK.md`.
  Do not start Phase C.
