# Step 5b Report — pose signal beyond recording identity

**Verdict: INCONCLUSIVE** (`data/_artifacts/step5b_pose_signal/verdict.json`). Protocol:
`docs/STEP_5B_PROTOCOL.md`, committed as `6a2a97b`, frozen 2026-11-10T11:46:07Z (config sha256
`8b480522…`) before any run. Source hashes still verify after all runs. No threshold, recipe or rule
changed after seeing results. Final-epoch checkpoints only. "Recording" = candidate filename group; no
subject, session or environment claim. Distances are **stored units**; "mm" values are **provisional,
assuming stored unit = metre**.

## 1. Bottom line

- **With dense frames, the model tracks pose within seen recordings.** In stage 1, D2, D3 and D4
  (dense, 16,235 frames from 42 recordings) beat the recording-mean oracle on `fit_holdout`. Their T1
  ratios (rel MPJPE / oracle rel MPJPE) are 0.853, 0.858 and 0.905, so T1 passes. The sparse reference
  D1 (2,687 frames, the Step-5 recipe) does not (1.109). This is the first evidence in the project that
  the CSI→pose mapping carries information beyond "which recording is this".
- **Nothing transfers body pose to unseen recordings.** X1 fails for every run. On `group_val` the rel
  MPJPE / mean-skeleton ratios are 0.993–0.998 for D1–D3 and 1.074 for D4. Motion skill on unseen
  recordings is negative in every run (−0.08 to −0.15).
- **The verdict is INCONCLUSIVE because of the selection rule, not because the data are ambiguous.** The
  predeclared rule picks the lowest `group_val` X1 ratio, with ties (< 1 % relative) going to the simpler
  run. D1, D2 and D3 all sit at ≈ 0.99–1.00, inside one tie band, so the **simplest run, D1, was
  selected**. D1 is the only stage-1 run that failed T1. Stage 2 therefore confirmed D1, which fails T1
  and X1 in all three seeds. Because stage-1 runs did pass T1, this is neither YELLOW-TRACKS nor
  RED-NO-TRACKING; the protocol defines it as INCONCLUSIVE. The rule was written so that X1 would
  discriminate between runs. Since no run transfers at all, the X1 ratio carried no signal and the tie
  rule decided.

## 2. Premises (Step 0)

`analyze_step5_recording_effects.py` → `premises/premises.json`. Definitions not given in the plan are
recorded in `details.definitions`: abs-MPJPE comparison per recording for "beats the oracle"; Pearson r
of absolute deviations from the recording's own mean for the correlation; Pearson r over frame pairs
exactly *k* frame indices apart for the autocorrelation; medians over recordings.

| Quantity | Plan | Measured | Reproduces |
|---|---|---|---|
| oracle abs / rel (temporal_holdout) | 0.139 / 0.095 | 0.1388 / 0.0954 | yes |
| R1 seed 0 abs / rel | 0.204 / 0.102 | 0.2041 / 0.1017 | yes |
| R2 seed 0 abs / rel | 0.231 / 0.104 | 0.2315 / 0.1035 | yes |
| share of recordings beating the oracle (R1 / R2) | 16 % / 10 % | 15.7 % / 9.8 % | yes |
| median within-recording deviation correlation (R1 / R2) | 0.05 / 0.04 | 0.051 / 0.037 | yes |
| nearest-recording-mean identification (chance 2 %) | 72 % | 71.6 % | yes |
| pose autocorrelation, lags 1 / 2 / 4 / 16 | 0.94 / 0.83 / 0.59 / −0.08 | 0.939 / 0.828 / 0.591 / −0.077 | yes |
| pose autocorrelation, lag 8 | 0.17 | **0.142** | **no** |
| within / between deviation | 0.092 / 0.104 | 0.0918 / 0.1041 | yes |

17 of 18 values reproduce. The lag-8 autocorrelation is 0.14 rather than 0.17. That deviation
strengthens premise 5 (Step-5 frames ~8 apart are even less correlated) and does not affect the guard
choice (lag 16 ≈ 0). It was disclosed in the protocol before the freeze, and the step proceeded.

## 3. Data

`freeze_v2` reproduced every expected number exactly. The views are:
- `fit_train_dense` 16,235, `fit_train_sparse` 2,687, `fit_holdout` 1,815;
- `group_val` 3,775 (validation recordings S11_32, S11_03, S11_10, S21_07, S21_29, S21_27, S51_11,
  S51_32, S51_23);
- `all_train_dense` 20,010, `all_train_sparse` 3,263, `all_holdout` 2,232;
- `group_dev` 512 (verified against the Step-5 SHA-256);
- 1,632 guard frames, not preprocessed.

Preprocessing into `data_v2`: 22,754 / 22,754 validated, 0 failures, 1,742 s, 4.5 GB; originals
untouched. Statistics:
- input std range 0.343–15.12 (fit) and 0.344–14.96 (all);
- codec σ_r 0.1090 (fit) and 0.1105 (all);
- per-recording input means for 59 recordings (51 from train-role frames, 8 from group_dev's own frames).

## 4. Gate S

- **Smoke**: D1–D4 pass every check on CUDA, including batch keys `{csi,pose}`, shapes, finite input,
  untrained = train-mean pose, finite loss, backward, parameter update, bit-exact state and prediction
  reload, and evaluation.
- **Overfit** on 64 `fit_train_dense` samples from 40 recordings (mean-pose MPJPE 0.8686): D2 ratio
  **0.106**, D4 ratio **0.051**; both ≤ 0.20.

## 5. Stage 1 (seed 0, 42 fit recordings; `group_dev` not opened)

| Run | View | Abs | Centroid | Rel | Oracle abs / rel | Mean pose abs | Mean skel rel | Rel/oracle | Rel/mean skel | Abs/mean pose | Motion skill | Pred std | PA-MPJPE mm* | PCK@0.5 % | Matched |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| D1 | fit_train_sparse | 0.1526 | 0.1137 | 0.0923 | 0.1110 / 0.0872 | 0.8708 | 0.1396 | 1.058 | 0.661 | 0.175 | −0.001 | 0.4621 | 71.8 | 82.7 | 2618 of 2687 |
| D1 | fit_holdout | 0.2081 | 0.1720 | 0.1018 | 0.1167 / 0.0918 | 0.8456 | 0.1451 | 1.109 | 0.702 | 0.246 | −0.043 | 0.4395 | 78.8 | 73.9 | 1715 of 1815 |
| D1 | group_val | 0.7053 | 0.6754 | 0.1479 | — | 1.0874 | 0.1486 | — | 0.995 | 0.649 | −0.076 | 0.3570 | 131.6 | 49.2 | 1571 of 3775 |
| D2 | fit_train_dense | 0.0772 | 0.0470 | 0.0608 | 0.1127 / 0.0888 | 0.8439 | 0.1405 | 0.685 | 0.433 | 0.091 | 0.276 | 0.4698 | 50.5 | 94.3 | 16220 of 16235 |
| D2 | fit_holdout | 0.1183 | 0.0876 | 0.0780 | 0.1164 / 0.0915 | 0.8448 | 0.1451 | 0.853 | 0.538 | 0.140 | 0.071 | 0.4632 | 62.7 | 87.8 | 1795 of 1815 |
| D2 | group_val | 0.6187 | 0.5855 | 0.1483 | — | 1.0741 | 0.1485 | — | 0.998 | 0.576 | −0.099 | 0.3580 | 135.2 | 55.6 | 1880 of 3775 |
| D3 | fit_train_dense | 0.0674 | 0.0385 | 0.0552 | 0.1127 / 0.0888 | 0.8439 | 0.1405 | 0.622 | 0.393 | 0.080 | 0.338 | 0.4716 | 46.6 | 95.9 | 16228 of 16235 |
| D3 | fit_holdout | 0.1142 | 0.0824 | 0.0785 | 0.1164 / 0.0915 | 0.8448 | 0.1451 | 0.858 | 0.541 | 0.135 | 0.063 | 0.4641 | 63.5 | 88.8 | 1798 of 1815 |
| D3 | group_val | 0.5995 | 0.5655 | 0.1476 | — | 1.0741 | 0.1485 | — | 0.993 | 0.558 | −0.096 | 0.3653 | 132.4 | 56.2 | 1972 of 3775 |
| D4 | fit_train_dense | 0.0687 | 0.0370 | 0.0568 | 0.1127 / 0.0888 | 0.8439 | 0.1405 | 0.640 | 0.404 | 0.081 | 0.319 | 0.4716 | 48.0 | 96.3 | 16233 of 16235 |
| D4 | fit_holdout | 0.1607 | 0.1304 | 0.0828 | 0.1164 / 0.0915 | 0.8448 | 0.1451 | 0.905 | 0.571 | 0.190 | −0.016 | 0.4527 | 66.0 | 84.8 | 1736 of 1815 |
| D4 | group_val | 0.8802 | 0.8533 | 0.1595 | — | 1.0741 | 0.1485 | — | 1.074 | 0.820 | −0.153 | 0.3644 | 130.6 | 38.3 | 1161 of 3775 |

*PA-MPJPE in mm is provisional. "Matched" = predictions within the 500 "mm" threshold (people dim 1, so
each unmatched sample is one miss plus one spurious prediction). PCK scale = distance between GT joints
6 and 4. The D1 oracle is built from the sparse training view, hence slightly different oracle values.

| Run | T1 ratio (fit_holdout) | T1 | abs/oracle (reported) | X1 ratio (group_val) | X1 | abs/mean pose (reported) | Reload exact | Runtime s |
|---|---|---|---|---|---|---|---|---|
| D1 | 1.109 | FAIL | 1.783 | 0.995 | FAIL | 0.649 | True | 71 |
| D2 | 0.853 | pass | 1.016 | 0.998 | FAIL | 0.576 | True | 359 |
| D3 | 0.858 | pass | 0.981 | 0.993 | FAIL | 0.558 | True | 356 |
| D4 | 0.905 | pass | 1.381 | 1.074 | FAIL | 0.820 | True | 352 |

No run crashed. Training-view prediction std across samples is 0.46–0.47 (≥ 0.01), so all four are
eligible. **Selection** (`selection.json`): best X1 ratio D3 0.9934; the tie band (≤ 1.0033) contains
D1 0.9954 and D2 0.9982; simplest = **D1**.

## 6. Stage 2 (selected D1, `all_train_sparse`, 51 recordings, seeds 0/1/2; `group_dev` evaluated once)

| Seed | View | Abs | Centroid | Rel | Oracle abs / rel | Mean skel rel | Rel/oracle | Rel/mean skel | Abs/mean pose | Motion skill | PA-MPJPE mm* | PCK@0.5 % | Matched |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | all_holdout | 0.2368 | 0.1993 | 0.1063 | 0.1430 / 0.0964 | 0.1447 | 1.103 | 0.735 | 0.269 | −0.049 | 84.1 | 72.3 | 2031 of 2232 |
| 0 | group_dev | 0.6734 | 0.6466 | 0.1496 | — | 0.1350 | — | 1.108 | 0.712 | −0.154 | 111.0 | 43.5 | 223 of 512 |
| 1 | all_holdout | 0.2343 | 0.1960 | 0.1075 | 0.1430 / 0.0964 | 0.1447 | 1.115 | 0.743 | 0.266 | −0.056 | 85.2 | 72.4 | 2052 of 2232 |
| 1 | group_dev | 0.6844 | 0.6598 | 0.1481 | — | 0.1350 | — | 1.097 | 0.723 | −0.139 | 111.0 | 38.3 | 215 of 512 |
| 2 | all_holdout | 0.2396 | 0.2014 | 0.1064 | 0.1430 / 0.0964 | 0.1447 | 1.103 | 0.735 | 0.272 | −0.052 | 83.8 | 71.2 | 2045 of 2232 |
| 2 | group_dev | 0.6175 | 0.5898 | 0.1475 | — | 0.1350 | — | 1.093 | 0.653 | −0.143 | 115.4 | 43.2 | 260 of 512 |

T1 fails in all three seeds (1.103 / 1.115 / 1.103) and X1 fails in all three (1.108 / 1.097 / 1.093).
Every reloaded state dict reproduced its predictions exactly. Runtimes 90 / 78 / 92 s. Training-view
tables are in `summary_tables.md`. Stage 2 essentially reproduces Step-5 R2 (same recipe; group_dev rel
0.148 vs 0.148), which is a useful consistency check.

## 7. What the controlled comparisons show

- **D2 vs D1 (frame density, same input):** going from 64 to all train-role frames per recording cuts
  `fit_holdout` abs MPJPE from 0.208 to 0.118 and rel from 0.102 to 0.078. The T1 ratio drops from
  1.109 to 0.853, and holdout motion skill rises from −0.04 to +0.07. Density is what turns
  "recognise the recording" into "beat the recording mean". On unseen recordings, density helps only the
  absolute part (group_val abs 0.705 → 0.619, centroid 0.675 → 0.586); the rel ratio is unchanged
  (0.995 → 0.998).
- **D3 vs D2 (normalisation under dense data):** train z-score fits the training data better (train rel
  0.055 vs 0.061, skill 0.34 vs 0.28) but is indistinguishable on holdout (T1 0.858 vs 0.853) and on
  group_val (0.993 vs 0.998).
- **D4 vs D2 (removing the recording signature):** subtracting each recording's mean CSI still allows
  within-recording fitting (train rel 0.057) and passes T1 (0.905). It loses most of the static position
  information (holdout abs 0.161 vs 0.118; group_val abs/mean pose 0.820 vs 0.576) and makes unseen-
  recording body pose *worse* (X1 1.074, skill −0.15). The recording signature is not a nuisance that
  hides a transferable pose signal: once removed, nothing transferable remains in this model.
- **Motion skill, seen vs unseen recordings:** on seen-recording holdout it is small even when T1
  passes (+0.06 to +0.07 for D2/D3). On unseen recordings it is negative everywhere. Combined with
  T1 < 1, this suggests (interpretation, not tested here) that much of the T1 gain comes from following
  slow, block-level pose changes inside a recording. The oracle uses the training-frame mean, while the
  holdout block may sit in a different part of the recording. Frame-to-frame motion is followed only
  weakly.

## 8. Verdict and what it means

Under the predeclared rule the verdict is **INCONCLUSIVE**: the selected run (D1) fails T1 in all seeds,
but D2, D3 and D4 passed T1 in stage 1. Reported as measured, with no post-hoc re-selection:
- Seen recordings: dense supervision gives within-recording tracking that beats the recording-mean
  oracle (stage 1, seed 0, three of three dense runs). This was **not** confirmed over seeds or on all 51
  recordings, because the selection rule sent D1 to stage 2.
- Unseen recordings: no input normalisation or data density gave centroid-relative body pose better than
  the train-mean skeleton (X1 fails in 4/4 stage-1 runs and 3/3 stage-2 seeds). This half of the
  question has a consistent negative answer within this model family.

Weakness of the protocol: the selection criterion (X1 ratio) could not discriminate once every run sat
at ≈ 1.0, so the simplicity tie-break chose the run that the step was designed to improve on.

## 9. Options (stop and ask; nothing started)

1. **Disclosed confirmation of the dense recipe:** run D2 (or D3) stage 2, seeds 0/1/2 on
   `all_train_dense` (~6 min per seed), as an explicitly *post-hoc*, labelled addendum. It would settle
   whether T1 holds across seeds on all 51 recordings, i.e. whether this is YELLOW-TRACKS in substance.
   It must be reported as a deviation from the frozen selection, not as the protocol verdict.
2. **Accept the X1 result as decisive for Phase C as planned:** single-person training on these
   recordings does not yield body pose that transfers to new recordings. Then follow the YELLOW-TRACKS
   recommendation: redesign Phase C around label-free adaptation to the target recordings (their
   unlabelled CSI is permitted), or switch to the label-efficiency track.
3. Either way, Phase C must not start without your decision.

## 10. Disclosures

- Development labels (`group_dev`) were used only for the single stage-2 confirmation, evaluated once at
  the final epoch of each seed. Stage 1 never opened `group_dev`; this is enforced in code and tested.
- `group_val` labels are single-person training-partition labels, used for evaluation and stage-1
  selection.
- D4 uses label-free but **transductive** per-recording input means of the evaluated recordings
  (group_val in stage 1; group_dev would have been used in stage 2, but D4 was not selected).
- No multi-person ID and no official `test_data` were used.
- Wording: candidate filename groups only; stored units; mm values provisional.

## 11. Reproduction

```
export PYTHONHASHSEED=0 CUBLAS_WORKSPACE_CONFIG=:4096:8
R="J:/Zero-shot 3D Multi-person Pose Estimation"
python analyze_step5_recording_effects.py --repo-root "$R"
python step5b_subset.py --repo-root "$R"
python make_step5b_data.py --repo-root "$R"
python step5b_pose_signal.py --repo-root "$R" freeze-protocol
python step5b_pose_signal.py --repo-root "$R" smoke
python step5b_pose_signal.py --repo-root "$R" overfit --run D2   # and D4
python step5b_pose_signal.py --repo-root "$R" stage1 --run D1    # D1..D4
python step5b_pose_signal.py --repo-root "$R" select
python step5b_pose_signal.py --repo-root "$R" stage2 --seed 0    # 0, 1, 2
python step5b_pose_signal.py --repo-root "$R" verdict
python summarize_step5b.py --repo-root "$R"
```

Environment: torch 2.5.1+cu121, RTX 3080 Laptop GPU, deterministic algorithms. Artifacts (git-ignored,
main checkout): `data/_artifacts/step5b_pose_signal/{premises,freeze_v2,data_v2,protocol_freeze,smoke,overfit,stage1,stage2,logs}`,
`selection.json`, `verdict.json`, `summary_tables.md`.

---

## Addendum A — POST-HOC dense-recipe confirmation (D3, user-approved after the verdict)

**Not part of the frozen protocol.** The protocol verdict above stays **INCONCLUSIVE**, and
`selection.json`, `stage2/` and `verdict.json` are unchanged. After seeing the verdict, the user
approved running option 1 of §9. `posthoc_step5b_dense_check.py` calls the unchanged, hash-verified
`train_and_evaluate("stage2", "D3", seed)` with the frozen stage-2 recipe: `train_zscore`,
`all_train_dense` (20,010 frames, 51 recordings), stats/codec `all`, 40 epochs. Outputs go to
`data/_artifacts/step5b_pose_signal/posthoc_d3/stage2/`.

D3 was chosen because it had the lowest stage-1 X1 ratio among the dense runs. Choosing it uses
`group_val` (training-partition) labels only. This addendum is a **second use of `group_dev`
development labels**: one final-epoch evaluation per seed. The source hashes still verify after these
runs.

| Seed | View | Abs | Centroid | Rel | Oracle abs / rel | Mean skel rel | Motion skill | PA-MPJPE mm* | PCK@0.5 % | Matched |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | all_train_dense | 0.0708 | 0.0416 | 0.0565 | 0.1295 / 0.0932 | 0.1416 | 0.349 | 47.7 | 95.7 | 19996 of 20010 |
| 0 | all_holdout | 0.1287 | 0.0971 | 0.0806 | 0.1428 / 0.0962 | 0.1444 | 0.061 | 63.6 | 86.9 | 2194 of 2232 |
| 0 | group_dev | 0.6766 | 0.6497 | 0.1561 | — | 0.1351 | −0.240 | 102.5 | 48.7 | 209 of 512 |
| 1 | all_train_dense | 0.0706 | 0.0417 | 0.0561 | 0.1295 / 0.0932 | 0.1416 | 0.354 | 47.4 | 95.7 | 19990 of 20010 |
| 1 | all_holdout | 0.1316 | 0.1000 | 0.0808 | 0.1428 / 0.0962 | 0.1444 | 0.051 | 64.1 | 86.9 | 2191 of 2232 |
| 1 | group_dev | 0.6430 | 0.6139 | 0.1606 | — | 0.1351 | −0.253 | 110.1 | 47.1 | 235 of 512 |
| 2 | all_train_dense | 0.0710 | 0.0415 | 0.0570 | 0.1295 / 0.0932 | 0.1416 | 0.343 | 48.2 | 95.6 | 20000 of 20010 |
| 2 | all_holdout | 0.1293 | 0.0976 | 0.0814 | 0.1428 / 0.0962 | 0.1444 | 0.053 | 65.0 | 87.1 | 2193 of 2232 |
| 2 | group_dev | 0.6394 | 0.6107 | 0.1562 | — | 0.1351 | −0.222 | 113.4 | 47.7 | 227 of 512 |

| Seed | T1 ratio (all_holdout) | T1 | abs/oracle (reported) | X1 ratio (group_dev) | X1 | abs/mean pose (reported) | Reload exact | Runtime s |
|---|---|---|---|---|---|---|---|---|
| 0 | 0.837 | pass | 0.902 | 1.155 | FAIL | 0.711 | True | 437 |
| 1 | 0.839 | pass | 0.922 | 1.189 | FAIL | 0.676 | True | 442 |
| 2 | 0.846 | pass | 0.906 | 1.156 | FAIL | 0.672 | True | 432 |

**Reading (post-hoc).** Applying the verdict definitions to D3, T1 passes in all three seeds and X1
fails in all three. That is the **YELLOW-TRACKS pattern**: pose sensing exists but is
recording-specific. The pattern rests on a disclosed post-hoc choice, not on the frozen selection.
Points to note:
- **Seen recordings:** within-recording tracking is stable over seeds. The rel/oracle ratio is
  0.84 ± 0.005, and the dense model also beats the oracle in absolute terms (abs/oracle 0.90–0.92).
- **Unseen recordings:** the dense model is *worse* than sparse D1 on `group_dev`. Its X1 ratios are
  1.16–1.19, against 1.09–1.11 for D1. Motion skill there is −0.22 to −0.25, against about −0.14 for D1.
  More frames per recording fit recording-specific pose more tightly and transfer less. The absolute
  ratio to the mean pose is about the same (0.67–0.71).
- **Frame-to-frame motion:** on held-out frames of seen recordings, motion skill is small (+0.05 to
  +0.06), so the model follows fast motion only weakly. As in §7, much of the T1 gain probably comes from
  slower within-recording pose changes (interpretation, not tested).

**Recommendation (unchanged; for the user to decide).** The YELLOW-TRACKS recommendation applies in
substance. Either redesign Phase C around label-free adaptation to the target recordings (their
unlabelled CSI is permitted), or switch to the label-efficiency track. Phase C has not been started.
