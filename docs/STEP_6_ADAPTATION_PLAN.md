# CompWiFi-Pose — Step 6 Plan (redesign): label-free adaptation to target recordings

Written 2026-11-10 after Step 5b. The user chose the **adaptation redesign** of Phase C. This plan
replaces the Phase C design in `docs/NEXT_PHASE_PLAN.md` §5 by inserting a bounded single-person gate
(**Step 6a**) before the compositional pilot (**Step 6b**). `docs/RESEARCH_PROTOCOL.md` §5–§7 (arms
A/B/C, collapse controls, Go/Yellow/No-go thresholds, corrected metrics) stay binding for 6b and are
not changed. Read `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/CURRENT_TASK.md`,
`docs/RESEARCH_PROTOCOL.md`, `docs/NEXT_PHASE_PLAN.md` §2/§5, `docs/STEP_5B_PLAN.md` §2 and
`docs/STEP_5B_REPORT.md` first.

**Status: plan only. Nothing here is approved for execution until the user says so.** The
[ASK] points are listed in §6.

"Recording" = candidate filename group. Stored units throughout; "mm" values are provisional
(metre assumption).

---

## 1. Why the redesign

What Step 5 and 5b established (`docs/STEP_5B_REPORT.md`, including post-hoc Addendum A):
- **Within seen recordings, CSI tracks the person.** With dense frames, single-person models beat each
  recording's own mean pose on held-out frames: T1 0.84 over 3 seeds (D3, post-hoc).
- **Across recordings, nothing transfers.** On unseen recordings, centroid-relative body pose is never
  better than the train-mean skeleton: X1 0.99–1.19 across all Step-5b runs. Motion skill there is
  negative, and dense training makes it worse. Removing the recording signature at the input (D4) also
  removes what little transfers.
- **Phase C needs exactly this transfer.** Pose labels exist only for single-person recordings; the
  multi-person recordings to be evaluated are always different recordings. A compositional model built
  on a single-person pose head that does not transfer would at best reproduce the train-mean skeleton in
  each slot. The `RESEARCH_PROTOCOL` collapse controls would then correctly call it a failure, with no
  way to tell "composition does not work" from "nothing transfers".

What adaptation can use, inside the label rules:
- Unlabelled CSI from the training partition is permitted (`AGENTS.md`). Every internal-development
  recording is part of the official training partition and has 188–539 frames. A disjoint block of each
  target recording can be held out for evaluation, while its other frames serve as unlabelled
  adaptation data.
- The official benchmark tests frames 301–349 of recordings that are also in official train
  (`docs/POSE_SPLIT_METRIC_AUDIT.md`). Recording-level adaptation from unlabelled official-train CSI
  therefore matches how the real benchmark is built.

**Question of Step 6a.** Can label-free adaptation, using only unlabelled CSI of a target recording,
make the single-person pose head transfer to that unseen recording (X1)? This is measurable now,
because single-person target recordings have permitted labels for evaluation. If no adaptation method
gives single-person transfer, the zero-label compositional pilot has no foundation. The step then stops
and recommends the label-efficiency track instead of running 6b.

**Lesson carried over from 5b.** No selection-by-tie-break step. Every arm is predeclared and is
confirmed on its own if it passes the development gate.

---

## 2. Ground rules

Everything in `docs/NEXT_PHASE_PLAN.md` §2 and `docs/STEP_5B_PLAN.md` §2 applies:
- **Environment and data:** the same venv, `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and
  `--repo-root` pointing at the main checkout.
- **Process:** one protocol doc + one runner + one report per step; freeze before running; final-epoch
  checkpoints only; no tuning after seeing results.
- **Data and git:** never official `test_data`; never `git add -A`; LF line endings for hashed files.

Additional rules:
- Branch `research/step6a-adaptation` from `research/step5b-pose-signal` (`ac77c29`).
- **Protected:** every Step-5 and Step-5b source file, doc and artifact (the Step-5b runner hashes them).
  Import from them; do not edit.
- **Frame disjointness:** adaptation frames and evaluation frames of the same target recording are
  disjoint and separated by a 16-frame guard (the Step-5b layout). Adaptation never reads the
  evaluation frames, not even their CSI. Record
  `uses_unlabelled_csi_of_evaluation_recordings: true` (other frames) in every config, and disclose it.
- **Labels:** no target-recording label reaches training in any label-free arm. Adaptation loaders return
  `{"csi"}` only (reuse the `feeder/piw3d_unlabelled` pattern) and never IDs. Evaluation labels are
  used for evaluation only.

---

## 3. Step 6a — single-person adaptation gate

### 3.1 Data

| Use | Source recordings | Frames | Status |
|---|---|---|---|
| Labelled source, dev stage | 42 fit recordings, `fit_train_dense` | 16,235 | frozen (`freeze_v2`) |
| Target adapt, dev stage (CSI only) | 9 validation recordings, `group_val` (train-role frames) | 3,775 | frozen |
| Target eval, dev stage (labels for eval only) | the same 9 recordings' holdout blocks (`all_holdout` minus `fit_holdout`) | 417 | frozen; add as view `val_holdout` |
| Labelled source, confirm stage | all 51 recordings, `all_train_dense` | 20,010 | frozen |
| Target adapt, confirm stage (CSI only) | 8 internal-dev single-person recordings, frames outside block + guard | 3,599 | **new preprocessing [ASK]** |
| Target eval, confirm stage | the same 8 recordings, block `[n//5, n//5+min(49,n//10))` | 392 | new preprocessing (same run) |

Confirm-stage layout: the Step-5b rule applied to all 4,247 frames of the 8 internal-dev single-person
recordings (block, guard 16, rest = adapt). The 512 frozen Step-5 `group_dev` frames are *not* reused
as the eval set, because they are spread over the whole recording and would overlap the adapt pool.
New freeze `freeze_v3` and data root `data_v3` (3,991 samples, ≈ 0.8 GB, ≈ 5 min at the Step-5b
rate). Expected counts are computed and written into the protocol before preprocessing; any mismatch →
stop.

Dev-stage target labels are single-person training-partition labels. Confirm-stage target labels are
internal-development labels, used for evaluation only (development protocol, disclosed).

### 3.2 Model and arms

Base: the Step-5b D3 recipe. `simple_cnn` + Step-5 codec; `train_zscore` with **source-only** input
statistics; AdamW 3e-4; warmup + cosine; clip 1.0; batch 32 labelled; 40 epochs; seeds 0/1/2.
`simple_cnn` uses GroupNorm, so BatchNorm-statistics adaptation (AdaBN) does not apply and is not an
arm. Each step of an adaptation arm adds one unlabelled batch of 32 target-adapt frames, drawn uniformly
over the target recordings' adapt pools. Loss weights are fixed at 1 on scale-normalised losses and are
not tuned.

| Arm | Labelled loss | Unlabelled target loss | Isolates |
|---|---|---|---|
| A0 | pose (codec) | none (target batch drawn, unused) | no-adaptation control (= D3) |
| A1 | pose | masked CSI reconstruction on source **and** target batches: 50 % of 9×4 (row × time) patches masked; light conv decoder from the encoder feature map; MSE on masked cells of the z-scored input | self-supervised features that include the target recordings |
| A2 | pose | CORAL + mean alignment of pooled 256-d encoder features, labelled batch vs target batch: `‖μ_s−μ_t‖² + ‖C_s−C_t‖_F²/(4d²)` | feature-distribution alignment to the target recordings |
| R-few | pose on source **plus** 16 labelled frames per target recording, taken evenly from the adapt pool | none | **reference only, dev stage only**: how much a few target labels would help. Forbidden at the confirm stage (development labels must not train) |

Model additions go in a new `model/adaptation.py` (the reconstruction decoder and the CORAL loss). The
pose path is identical to D3.

### 3.3 Metrics and gates

On target eval frames (and, as diagnostics, on the source train view and on target adapt frames *without*
using their labels for anything but reporting at the dev stage):
- abs MPJPE, centroid error, rel MPJPE, motion skill, collapse monitors, `ResearchMetricAccumulator`;
- mean-pose / mean-skeleton baselines from the labelled source view;
- per-recording rel MPJPE.

**Gate X1a (transfer to target recordings):** target-eval
`rel MPJPE ≤ 0.95 × source mean-skeleton rel MPJPE`.

**Gate Δ (adaptation helps):** target-eval `rel MPJPE ≤ 0.97 ×` A0's value for the same seed and stage.
This keeps an arm that merely matches A0 from passing on a lucky split.

**Sanity S:** the Step-5b smoke list for each arm, plus:
- the adaptation loader returns only `{csi}`;
- a test opens files through a monkeypatched `open` and asserts that no evaluation-frame file is read
  during training;
- a 64-sample overfit check for A1 and A2 (≤ 0.20 × mean pose), confirming that the extra loss does not
  break fitting.

**Procedure.**
1. **Dev stage**: A0, A1, A2 and R-few, seed 0, source = 42 fit recordings, target = 9 validation
   recordings.
2. **Carry-forward rule (no tie-breaks):** every *label-free* arm (A1, A2) that passes X1a **and** Δ at
   the dev stage goes to the confirm stage. A0 always goes, as the control.
3. **Confirm stage**: each carried arm and A0 for seeds 0/1/2, source = 51 recordings, target = the 8
   internal-dev single-person recordings.

### 3.4 Verdict

- **ADAPT-GO:** at least one label-free arm passes X1a and Δ in all 3 confirm seeds → propose Step 6b
  with that adaptation mechanism.
- **ADAPT-PARTIAL:** a label-free arm passes at the dev stage but in only 1–2 confirm seeds, or passes
  X1a without Δ → stop, report, ask.
- **ADAPT-NO-GO:** no label-free arm passes X1a and Δ at the dev stage, or every carried arm fails all
  confirm seeds → stop. Recommend the label-efficiency track (`PROJECT_CONTEXT` §9 fallback, a separate
  and explicitly labelled task). Report R-few next to it, to show whether the gap is "no adaptation
  signal" or "transfer is impossible even with a few labels".

Expected cost: dev stage 4 runs × ≈ 8–15 min (A1 roughly doubles step cost); confirm stage up to
3 arms × 3 seeds × ≈ 10–15 min; total ≈ 2–3 GPU-h. If any first epoch projects > 1 h per run → [ASK].

---

## 4. Step 6b — compositional pilot with adaptation (only after ADAPT-GO and user approval)

This is the `docs/NEXT_PHASE_PLAN.md` §5 pilot, modified only as follows:
- **Labelled source:** `all_train_dense` (dense frames are what made T1 work), instead of the Step-5
  sparse `train` split.
- **Unlabelled pool U, count-blind:** the adapt frames of **all 43 internal-development recordings**
  (single- and multi-person alike, the same block/guard rule for every recording, never branching on
  the person-count character), plus a fixed per-recording sample of the 172 internal-train recordings.
  Preprocessing needs an [ASK]: ≈ 16,000 + new samples, ≈ 3 GB.
- **Evaluation:** the eval blocks of the 35 multi-person internal-dev recordings (21 two-person,
  14 three-person; up to 49 frames each under the block rule, at most 1,715 samples), plus the 8 single-person
  eval blocks from 6a. These frames are disjoint from U by the 16-frame guard. New
  `feeder/piw3d_dev_eval.py`, as specified in NEXT_PHASE_PLAN §5.
- **Adaptation in every arm:** the 6a-winning adaptation loss is applied identically in A, B and C on U,
  so that the arms still differ only in what RESEARCH_PROTOCOL §5 says they differ in. Add a disclosed
  control A-noadapt (arm A without the adaptation loss) to measure the adaptation effect on
  multi-person scenes.
- Go/Yellow/No-go thresholds, collapse controls, 3 seeds and corrected metrics: **unchanged** from
  `RESEARCH_PROTOCOL` §6–§7.
- Disclosure: unlabelled CSI of the evaluation recordings (other frames) is used in training. This is a
  development-protocol result under "zero real multi-person training labels". Multi-person labels are
  used only for evaluation and the go/no-go decision.

A separate `docs/STEP_6B_PROTOCOL.md` is written and approved before any 6b run.

---

## 5. Files

Reuse by import (do not edit): `step5b_config.py`, `step5b_subset.py` (`dense_layout`,
`load_view_ids`), `make_step5b_data.py` (`recording_input_means` pattern, copy/process/validate flow),
`feeder/piw3d_step5b.py`, `feeder/piw3d_unlabelled.py`, `step5b_pose_signal.py` (`load_view`,
`evaluate_views`, `motion_skill`, `mean_pose_baseline`, `gate_values` pattern), `step5_single_person_baseline.py`
helpers, `model/research_models.py`, `research_metrics.py`.

New for 6a: `step6a_config.py`, `step6a_subset.py` (`val_holdout` view + `freeze_v3`),
`make_step6a_data.py`, `model/adaptation.py`, `step6a_adaptation.py`, `tests/test_step6a_*.py`,
`docs/STEP_6A_PROTOCOL.md`, `docs/STEP_6A_REPORT.md`; edit `docs/CURRENT_TASK.md` (status block and
progress entry only).

---

## 6. [ASK] points and order

1. Approval of this plan (now).
2. Preprocessing `freeze_v3` / `data_v3`: 3,991 internal-dev single-person samples (3,599 adapt + 392 eval), ≈ 0.8 GB.
3. Training of the 6a matrix (≈ 2–3 GPU-h), unless approved together with item 1.
4. After the 6a verdict, the decision on 6b; then 6b preprocessing (≈ 3 GB) and the 6b protocol.
5. Pushing or merging anything.

Order inside 6a: protocol doc → commit → freeze → smoke/overfit → dev stage → carry-forward →
confirm stage → verdict → report → `CURRENT_TASK.md` → stop.

## 7. Risks stated up front

- **The likely outcome is ADAPT-NO-GO.** D4, a crude form of per-recording normalisation, made
  transfer worse, and dense training reduced cross-recording motion skill. Within the 5b model family
  the recording signature appears to carry most of the pose-relevant information. ADAPT-NO-GO is a
  valid, reportable result, and it is cheaper to learn it here than inside a compositional pilot.
- **CORAL and similar alignment can remove pose information** together with the recording signature.
  Gate Δ against A0 and the R-few reference make this visible.
- **Target-recording adaptation is transductive at the recording level.** It is disclosed everywhere
  and kept frame-disjoint by construction.
