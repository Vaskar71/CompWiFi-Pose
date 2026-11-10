# Step 5b Protocol — pose signal beyond recording identity

Status: **frozen before any training run.** Source: `docs/STEP_5B_PLAN.md` (copied verbatim). The
machine-readable copy is `step5b_config.py` (`STEP5B_CONFIG`). `step5b_pose_signal.py freeze-protocol`
writes a JSON copy plus SHA-256 hashes of this document, the plan and every Step-5b (and reused Step-5)
source file to `data/_artifacts/step5b_pose_signal/protocol_freeze/`. Every later command refuses to run
if any of them changed. No threshold, recipe or rule may change after seeing results. Final-epoch
checkpoints only.

Binding rules: `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/RESEARCH_PROTOCOL.md`,
`docs/NEXT_PHASE_PLAN.md` §2, `docs/STEP_5B_PLAN.md` §2. "Recording" = candidate filename group (e.g.
`S11_06`). No subject, session or environment claim is made. All distances are **stored units**; "mm"
values are **provisional, assuming stored unit = metre**. The 14-joint centroid is the translation
reference.

**Question.** Does CSI carry a learnable body-pose signal beyond recording identity: first within seen
recordings (gate T1), then on unseen recordings (gate X1)?

## 0. Premises (Step 0, done before this freeze)

`analyze_step5_recording_effects.py` → `premises/premises.json`. 17 of 18 quoted values reproduce within
rounding. The lag-8 pose autocorrelation is **0.14** (plan: 0.17). Lags 1/2/4/16 reproduce (0.94 / 0.83 /
0.59 / −0.08). The deviation is in the direction that strengthens premise 5: at Step-5's ~8-frame
spacing the pose is even more decorrelated. The plan's guard choice (16, autocorrelation ≈ 0) is
unaffected. Recorded as a disclosed deviation; the step proceeds. The definitions used (where the plan
gives none) are recorded in `premises.json` → `details.definitions`.

## 1. Data: `freeze_v2` / `data_v2`

`step5b_subset.py` (deterministic, no seed, names only) on the internal candidate-group manifests.
Single-person filter: `step4b_subset.is_single_person_id` (permitted, selects the labelled single-person
subset). For each of the 51 single-person internal-train recordings, frames sorted by frame index, `n`
frames:
- holdout block = sorted positions `[n//5, n//5 + h)`, `h = min(49, n//10)` (identical to Step 5);
- guard = 16 positions on each side, excluded from everything and **not preprocessed**;
- role `temporal_holdout` = every frame in the block. Label: **same-recording temporal holdout — NOT
  recording-disjoint (official-like)**;
- role `train` = every remaining frame.

Recording partition: within each scene prefix (`S11`, `S21`, `S51`), sort recording names by
`sha256(name.encode("utf-8")).hexdigest()` ascending; the first 3 are validation recordings. Result
(matches the plan): `S11_32, S11_03, S11_10, S21_07, S21_29, S21_27, S51_11, S51_32, S51_23`. The other 42
are fit recordings. Role `group_dev` = the 512 frozen Step-5 `group_dev` IDs (8 internal-development
recordings, **candidate-filename-group disjoint**). The list is verified against its recorded SHA-256.

| View | Content | Frames (frozen) |
|---|---|---|
| `fit_train_dense` | train-role frames of the 42 fit recordings | 16,235 |
| `fit_train_sparse` | ≤ 64 evenly spaced train-role frames per fit recording | 2,687 |
| `fit_holdout` | holdout-role frames of the 42 fit recordings | 1,815 |
| `group_val` | train-role frames of the 9 validation recordings | 3,775 |
| `all_train_dense` | train-role frames of all 51 recordings | 20,010 |
| `all_train_sparse` | ≤ 64 evenly spaced train-role frames per recording (only if D1 is selected) | 3,263 |
| `all_holdout` | holdout-role frames of all 51 recordings | 2,232 |
| `group_dev` | frozen Step-5 group_dev | 512 |

Guard frames: 1,632. Preprocessing (`make_step5b_data.py`): copy the raw files, then run `process_ids`
on 22,754 samples in `data_v2/`. Every output is validated for shape, finiteness, exact amplitude/phase
concatenation and role. Originals are untouched; official `test_data` is never opened.

Statistics, written once to `freeze_v2/`:
- `input_stats_{fit,all}.npz` (z-score mean/std `(3,180,1)`) from `fit_train_dense` / `all_train_dense`.
- `codec_stats_{fit,all}.json` (target codec) from the same views. These use single-person training
  labels, which is permitted.
- `recording_input_means.npz`: per recording, the mean of stored `csi_ap` over frames and time. It uses
  train-role frames for the 51 training recordings and their own 64 frames for the 8 `group_dev`
  recordings. These means are label-free but **transductive** for the evaluated recordings (group_val,
  group_dev): `uses_label_free_per_recording_input_statistics: true`.

## 2. Model, inputs, recipe

Model: `simple_cnn` only, via `build_research_model("simple_cnn", codec_stats, step5_config_for_json(),
device)`, with the Step-5 codec and loss. The last layer is zero-initialised, so the untrained model
outputs the training-view mean pose.

Input normalisations (`feeder/piw3d_step5b.Step5bDataset`, overriding `read_csi` only):
- `legacy_minmax`: Step-5 per-sample min-max.
- `train_zscore`: the Step-5 z-score with `input_stats_fit` (stage 1) / `input_stats_all` (stage 2).
- `recording_centered`: `(x − μ_recording) / max(σ_train, 1e-6)`, with `σ_train` the z-score std of the
  stage. Each sample's recording is resolved once in the dataset constructor; the training loop never
  sees IDs.

Recipe: identical to Step 5 (AdamW, peak lr 3e-4, weight decay 0.01, 200-step linear warmup then cosine,
grad clip 1.0, batch 32, 40 epochs, `num_workers=0`, seeded shuffle generator, final-epoch checkpoint,
Step-5 determinism block). [ASK] if the first epoch of any run projects > 1 h for that run.

## 3. Run matrix

**Stage 1**: seed 0, trained on the 42 fit recordings, with stats/codec `fit`. Each run is evaluated
every epoch and at the end on its training view, `fit_holdout` and `group_val`. `group_dev` is not
opened (enforced in code).

| Run | Input | Training view | Isolates |
|---|---|---|---|
| D1 | `legacy_minmax` | `fit_train_sparse` | reference: Step-5 R2 recipe at 42 recordings |
| D2 | `legacy_minmax` | `fit_train_dense` | frame density |
| D3 | `train_zscore` | `fit_train_dense` | normalisation under dense data |
| D4 | `recording_centered` | `fit_train_dense` | removing the recording signature |

**Stage 2**: the selected run only, seeds 0, 1 and 2, trained on all 51 recordings with stats/codec
`all`. The training view is `all_train_dense` (`all_train_sparse` if D1 is selected). It is evaluated
each epoch on the training view and `all_holdout`, and **once**, at the final epoch, on `group_dev`.

Order: protocol doc → commit → freeze → smoke → overfit (D2, D4) → stage 1 → selection → stage 2 →
verdict → report.

## 4. Checks, metrics, gates

**Smoke** (every D run config, 64-sample subsets of its training view and of group_val): batch keys
`{csi,pose}`; shapes `(B,3,180,20)` / `(B,1,14,3)`; finite inputs; forward and `forward_features`
shapes; the untrained model equals the train-mean pose (codec `fit`); finite loss; backward with
non-zero gradient norm; parameter update; bit-exact state-dict and prediction reload; evaluation runs.

**Overfit** (D2 and D4): 64 samples of `fit_train_dense` at positions `i·k`, `k = 16235//64`; 300 steps
on the full batch; AdamW at constant lr 1e-3, no weight decay, clip 1.0, seed 0. Pass if the final
eval-mode abs MPJPE ≤ 0.20 × the abs MPJPE of the mean pose of the same 64 samples.

**Gate S** = all smoke checks pass and both overfit ratios ≤ 0.20. Stage 1 refuses to start otherwise.

**Per run and evaluation view** (stored units): abs MPJPE, centroid error, rel MPJPE (centroid-relative),
collapse monitors (prediction std across samples / joints), shuffled-pairing MPJPE, and
`ResearchMetricAccumulator` (people dim 1, presence 1, 500 "mm" match threshold, PCK@0.5 with scale =
the distance between GT joints 6 and 4, root joint 0 with unproven identity; final epoch only). Also:
- **Recording-mean oracle** (views whose recordings all appear in the training view): predict the mean
  pose of that recording's frames in the run's training view. Abs and rel MPJPE.
- **Mean pose / mean skeleton**: the training-view mean pose (abs) and its centroid-relative skeleton
  (rel).
- **Motion skill** (diagnostic, every view). Per recording r: `dgt = rel(gt) − mean_r rel(gt)`,
  `dpred = rel(pred) − mean_r rel(pred)`, and
  `skill_r = 1 − mean_{frames,joints}‖dpred − dgt‖ / mean_{frames,joints}‖dgt‖`. The reported value is
  the frame-weighted mean of `skill_r` over recordings. Positive means the predicted motion follows the
  true motion. It uses evaluation labels for evaluation only.

**Gates**
- **T1, tracking on seen recordings**: on the holdout view (`fit_holdout` / `all_holdout`),
  `rel MPJPE ≤ 0.95 × recording-mean-oracle rel MPJPE`. The abs ratio to the oracle is reported but not
  gated.
- **X1, body-pose transfer to unseen recordings**: on `group_val` (stage 1) / `group_dev` (stage 2),
  `rel MPJPE ≤ 0.95 × training-view mean-skeleton rel MPJPE`. The abs ratio to the mean pose is reported
  next to it but is not part of the verdict.

A run that stops on a non-finite loss or gradient is recorded as crashed: its `decision.json` has
`crashed: true` and it fails every gate. It is reported and never re-run with changed settings.

**Selection** (stage 1, seed 0): among runs that did not crash and whose training-view
`pred_std_across_samples` ≥ 0.01, pick the lowest `group_val` rel-MPJPE ratio to the mean skeleton.
Ties (< 1 % relative) go to the simpler run (D1 < D2 < D3 < D4). This uses single-person
training-partition labels only.

**Verdict** (stage 2, all three seeds of the selected run):
- **GREEN**: T1 and X1 pass in all three seeds → Phase C may be proposed with the selected input/data
  recipe and group-disjoint evaluation as primary.
- **YELLOW-TRACKS**: T1 passes in all three seeds and X1 fails in at least one → stop. Recommend either
  redesigning Phase C around label-free adaptation to the target recordings, or the label-efficiency
  track.
- **RED-NO-TRACKING**: the selected run fails T1 in all three seeds **and** no stage-1 run passed T1 on
  `fit_holdout` → stop. Only a read-only CSI↔label time-alignment audit is permitted inside this step.
- **INCONCLUSIVE**: anything else → report every number and ask.

## 5. Label access

- **Training loss**: single-person labels of the run's training view only.
- **Holdout labels**: evaluation and the oracle only.
- **group_val labels** (single-person training partition): evaluation and stage-1 selection.
- **group_dev labels** (development): the single stage-2 confirmation only. The loader refuses
  `group_dev` with stage-1 statistics.
- **Codec / z-score statistics**: training view of the stage.
- **Per-recording input means**: label-free; transductive for unseen recordings (disclosed).
- No multi-person IDs, no official `test_data`, and no sample-ID parsing inside the training loop.
  Batches are `{csi, pose}` only (asserted per batch).

## 6. Execution environment

Code and commits live on branch `research/step5b-pose-signal` (from `research/step5-single-person-baseline`
`11fad75`). Data and artifacts live in the main checkout's git-ignored
`data/_artifacts/step5b_pose_signal/`, addressed with `--repo-root "J:/Zero-shot 3D Multi-person Pose
Estimation"`.
