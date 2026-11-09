# CompWiFi-Pose — Next-Phase Execution Plan (Phase A → B → C)

Written 2026-11-10 for an executor agent that has **not** seen the planning conversation.
Read `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/CURRENT_TASK.md`, `docs/RESEARCH_PROTOCOL.md` first; they stay binding. This plan adds to them, it does not replace them.

---

## 1. Context

**Goal of the project.** Multi-person 3D pose from WiFi CSI (Person-in-WiFi-3D) with **zero real multi-person training labels**: labelled single-person pairs + unlabelled CSI only. Output = set of skeletons + per-slot presence. Hypothesis = compositional scene-to-set learning (background slot + ≤3 human slots, shared single-person pose head, latent recomposition on unlabelled CSI).

**Where it stands.** Engineering foundation is done (data audit, bounded preprocessing `process_ids`, candidate-group manifests, CSI-only unlabelled loader, corrected set metrics, 74 tests). The science is stuck one step before the actual research: the single-person baseline (Step-4B → D1–D4 → Step-4B-R) never beat a constant mean pose (final train/dev 1.059/0.833 vs mean-pose 0.990/0.801 stored units). The compositional method (RESEARCH_PROTOCOL arms A/B/C) has not been started. The user is frustrated by repetitive micro-diagnostics.

**Why the baseline failed — measured during planning (read-only, on existing artifacts; Phase A re-runs and records these):**

1. **The Step-4B-R model is a constant function.** `development_predictions.npz`: all 512 dev predictions identical, and all 14 joints are the *same 3D point* `[2.797, 1.784, 3.278]`. Train MPJPE sat at 0.94–1.10 from epoch 1, i.e. it never left the constant regime.
2. **The collapse is inside the legacy pose decoder, not the encoder.** Forward pass of `final_state_dict.pt` on 32 real samples: pooled encoder feature std across samples 0.18; after `gconv3` std across samples 0.47 / across joints 3.28 (alive); after `transformer_encoder1` 4e-4 / 1.5e-2; after `transformer_encoder3` 4e-7 / 3e-6 (dead). The three post-norm `CustomTransformerEncoderLayer`s in `ViT_Pose_Decoder` (`model/model.py`) saturate to a bias-dominated constant under from-scratch AdamW lr 1e-3, batch 8, no warmup. (Note: the upstream DT-Pose recipe is MAE-pretrain 400 epochs → frozen encoder; Step-4B-R trained from scratch.)
3. **The target is dominated by where the person stands, not by pose.** Mean-pose error 0.99 (train) is almost all translation: centroid distance to train-mean centroid = 0.97. Centroid-relative mean-skeleton MPJPE is only **0.155 train / 0.128 dev**.
4. **The pilot subset has only 8 positions.** Each candidate group is one person at essentially one location (within-group centroid std 0.01–0.12). 8 train groups → 8 positions; dev = 4 *unseen* positions. Absolute localization cannot generalize from 8 points regardless of the model. The manifests contain **51 single-person internal-train groups (23,874 IDs)** and **8 single-person internal-dev groups (4,247 IDs)** — unused so far.
5. **The official benchmark is a within-recording temporal split.** Official test = frames 301–349 of groups that are all also in official train (frames 10–599 minus that block). Our group-disjoint internal dev is a much harder protocol than the published one. Both views must be reported, never mixed.

**Repository risk.** All work after commit `c494564` (Step-3 gate → Step-4B-R) is **uncommitted and unpushed**: ~45 untracked files in the main checkout plus six modified tracked files that live only in git stash `9585ba9375a2218a9d3af5ae52c6c640cb247d32` (`.gitignore`, `data/person_in_wifi_3d/process.py`, `docs/CURRENT_TASK.md`, `docs/POSE_SPLIT_METRIC_AUDIT.md`, `docs/PROJECT_CONTEXT.md`, `model/model.py`). The untracked code depends on the stashed changes (`process_ids`, `train_encoder`). GitHub (`kimi` remote = Vaskar71/CompWiFi-Pose) has none of it.

**Intended outcome of this plan.** (A) Make the work safe and reproducible. (B) In **one** bounded, predeclared step, obtain a single-person model that demonstrably learns (or a clear, final diagnosis of why not). (C) Run the minimal compositional pilot A/B/C with the frozen Go/Yellow/No-go rule — the first real test of the research hypothesis.

---

## 2. Ground rules for the executor

- **Work in the main checkout** `J:\Zero-shot 3D Multi-person Pose Estimation` (data and artifacts are git-ignored and exist only there; worktrees do not have them).
- Environment: `C:\Users\vaska\venvs\compwifi-piw3d-py311\Scripts\python.exe` (Py 3.11, torch 2.5.1+cu121, RTX 3080 Laptop 16 GB). **No package installs** without asking.
- Test command (baseline: **74 passed**):
  `PIW3D_DATA_ROOT=data/_fixtures/piw3d_two_frame PYTHONDONTWRITEBYTECODE=1 /c/Users/vaska/venvs/compwifi-piw3d-py311/Scripts/python.exe -m pytest tests -q -p no:cacheprovider`
- **Do not edit protected files**: `model/model.py`, `feeder/piw3d_single_person.py`, `feeder/piw3d_unlabelled.py`, `feeder/person_in_wifi_3d.py`, `train_pose.py`, `pretrain.py`, every `step4b_*.py`, every `docs/STEP_4B_*`, everything under `data/_artifacts/step4b_*`. Step-4B-R's runner verifies these records. Add **new** modules/subclasses instead. Never touch `data/person_in_wifi_3d/wifi_dataset/` contents.
- **One protocol doc + one runner + one report per step.** No new "D-n" micro-diagnostic steps. If a gate fails, stop, write the report, and ask — do not improvise follow-up experiments.
- Freeze the config (JSON copy + source hash) **before** any run; no tuning after seeing dev numbers; final-epoch checkpoint only.
- Never use official `test_data` (the legacy `validation` loader reads it — do not call it). Never let pose/count labels of multi-person samples reach training code. Never parse sample IDs inside training loops (the 3rd character of the scene prefix encodes person count).
- Report distances in **stored units**; a ×1000 "mm" figure may be shown only labelled "provisional, assumes stored unit = metre". Joint order, axes, root identity remain unproven — use the **14-joint centroid** as the translation reference, not "root joint 0".
- Claim only "candidate-filename-group disjoint", never subject/recording/environment disjoint.
- Never claim a command passed without its real output. Do not invent numbers.
- **Ask-first list (AGENTS.md):** installing/removing packages, downloading the dataset, full training, deleting anything (including dropping the stash), changing remotes, pushing/publishing. Additional ask points are marked **[ASK]** below.
- Git: never `git add -A` (unrelated `archive-7U4NdL/`, `archive-AzzHuZ/`, `.anchor/` exist); add explicit paths. Never `git stash pop` / bare `git stash`. Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>` (adapt if another model executes).
- After each phase: completion report per AGENTS.md, update `docs/CURRENT_TASK.md`, then **stop** and wait for the user.
- **Execution setup:** one fresh session per phase (A, then B, then C), each started with "Read AGENTS.md and docs/NEXT_PHASE_PLAN.md, execute Phase X only". Use a top-tier reasoning model for B and C; a mid-tier model is enough for A. All function names in §6 were verified to exist on 2026-11-10 (`process_ids(name_list, data_root, split)` is in the stashed `process.py`).

---

## 3. Phase A — Consolidate and secure the repository (no new science; ~1 hour)

1. `git -C "<main>" stash list --format='%H %gs'` → confirm `9585ba9375a2218a9d3af5ae52c6c640cb247d32` exists. `git status --short` → confirm only untracked files (no tracked modifications). If either differs, **stop and report**.
2. `git switch -c consolidate/step3-to-step4b-r` then `git stash apply 9585ba9375a2218a9d3af5ae52c6c640cb247d32`. Expect exactly six modified tracked files (list in §1). Leave the stash entry in place.
3. Run the test command → expect **74 passed, 0 skipped**. If not, stop and report.
4. Keep unrelated files out of git without changing the repo: append `.anchor/`, `archive-7U4NdL/`, `archive-AzzHuZ/` to `.git/info/exclude`. Do not delete them.
5. Bring this plan into the repo as `docs/NEXT_PHASE_PLAN.md`: copy it verbatim from `C:\Users\vaska\.claude\plans\c-users-vaska-downloads-zero-shot-3d-mu-fancy-whistle.md` (or from the file the user provides if that path is gone). Do not paraphrase it.
6. Add `diagnose_step4b_r_collapse.py` (new, read-only on artifacts, CPU) + `tests/test_step4b_r_collapse_diagnosis.py` (unit-test the statistics helpers on synthetic tensors). It must output `data/_artifacts/step4b_r_diagnosis/diagnosis.json` and `docs/STEP_4B_R_COLLAPSE_DIAGNOSIS.md` containing:
   - per-joint and per-sample std of `model_predictions` in `data/_artifacts/step4b_r/baseline/development_predictions.npz` (expected ≈0, single point `[2.797, 1.784, 3.278]`);
   - activation statistics (std across samples, std across joints, fraction exact zeros) at pooled encoder feature, `gconv1..3`, `transformer_encoder1..3`, final pose, for the reloaded `final_state_dict.pt` on 16 train + 16 dev samples (build the model with `step4b_r_baseline.build_step4b_r_model`, load data with `SinglePersonPiW3DDataset`); expected pattern in §1 item 2;
   - target decomposition on `pilot_data_v2` keypoints: absolute mean-pose MPJPE (0.9898/0.8014), centroid distance to train-mean centroid (0.968/0.787), centroid-relative mean-skeleton MPJPE (0.155/0.128), per-group centroid mean/std.
   If the numbers do not reproduce within rounding, **stop and report** (the plan's premises would be wrong).
7. Update `docs/CURRENT_TASK.md`: add a ≤40-line **"Status at a glance"** block at the top (phase, last verdict, next task, pointer to this plan); add the missing Step-4B-R entry (verdict `BASELINE_COMPLETE`, numbers from `docs/STEP_4B_R_REPORT.md`) and the collapse diagnosis; set "Next recommended task" to Phase B. Do not rewrite older entries.
8. Commit with explicit paths in 2–3 logical commits (research infrastructure + stashed tracked changes; Step-4B/D1–D4; Step-4B-R + diagnosis + docs). `data/_artifacts/`, `data/_downloads`, `data/_fixtures/` stay ignored (the stashed `.gitignore` already does this). Verify `git status --short` shows nothing unexpected and tests still pass.
9. **[ASK]** the user: (a) push the branch to `kimi`? (b) fast-forward `main`? (c) drop the stash now that it is committed? Do none of these without a yes.

**Phase A done when:** branch contains all prior work, tests pass, diagnosis reproduced and documented, CURRENT_TASK.md current.

---

## 4. Phase B — "Step 5": a single-person baseline that provably learns (the next phase proper)

Branch `research/step5-single-person-baseline` from the Phase A branch. Deliverables: `docs/STEP_5_PROTOCOL.md` (frozen before any run), code + tests, `docs/STEP_5_REPORT.md`, verdict GREEN / YELLOW-POSITION / RED.

### B1. Data subset (`step5_subset.py`, `make_step5_data.py`, tests)

Source: `data/_artifacts/piw3d_candidate_group_manifests/internal_{train,development}_ids.txt`; single-person filter `step4b_subset.is_single_person_id`; ID parsing `piw3d_manifest.parse_sample_id` (gives `group`, `frame`); frame thinning `step4b_subset.select_evenly_spaced_frames`. Read `step4b_subset.freeze_step4b_subset` and `make_step4b_pilot_data.build_step4b_pilot_data` and reuse their freeze/validate/`process_ids` pattern (import, don't edit).

For **every** single-person internal-train group (51 expected; frames per group 105–539), with `n` = number of available frames sorted by frame index:
- temporal-holdout block = sorted positions `[n//5, n//5 + h)`, `h = min(49, n//10)`; guard = 5 positions on each side, excluded from everything;
- `train` = up to **64** evenly spaced frames from the remaining positions (→ ≈3,264);
- `temporal_holdout` = up to **16** evenly spaced frames from the block (→ ≈816). Label it everywhere **"same-group temporal holdout — NOT group-disjoint (official-like)"**.

For every single-person internal-dev group (8 expected): `group_dev` = up to **64** evenly spaced frames (→ ≈512). This is the **group-disjoint** set.

Write the freeze (IDs per split, group list + hashes, counts, rule text, seed-free determinism) to `data/_artifacts/step5_single_person/freeze_v1/`; preprocess via `data.person_in_wifi_3d.process.process_ids` into `data/_artifacts/step5_single_person/data_v1/` (≈4,600 samples, ≈1 GB, ≈6–10 min; originals untouched; validate shapes/finiteness like `_validate_processed_sample`). Record per-group centroid mean/std in the freeze summary (labels of single-person training-partition samples are permitted). Tests: split disjointness, guard respected, no multi-person ID, no ID outside the internal manifests, deterministic re-selection.

### B2. Dataset wrapper (`feeder/piw3d_step5.py`, tests)

`Step5SinglePersonDataset(SinglePersonPiW3DDataset)` overriding only `read_csi` with option `input_norm`:
- `legacy_minmax` — identical to the parent (per-sample min-max over the whole amplitude block and whole phase block);
- `train_zscore` — `(x − mean)/max(std,1e-6)` with mean/std of shape `(3,180,1)` computed **once from the `train` split only**, saved to the freeze dir, loaded read-only.
Batches stay `{"csi","pose"}`; reuse `collate_single_person` / `make_single_person_dataloader`.

### B3. Models (`model/research_models.py`, tests) — new file, nothing in `model/model.py` changes

Common target parameterization (`PoseTargetCodec`, stats from the `train` split only, saved in the freeze): centroid `c` = mean of 14 joints; relative `r = pose − c`; `μ_c, σ_c` per axis; `μ_r` (14×3); scalar `σ_r` = std of `r − μ_r`. Network output `o ∈ R^45` → `ĉ = μ_c + σ_c·o[:3]`, `r̂ = μ_r + σ_r·o[3:]`, `pose = ĉ + r̂`. **The last linear layer is zero-initialised**, so the untrained model outputs exactly the train-mean pose (unit test this). Loss = `mean‖ĉ−c‖₂ / mean(σ_c) + mean_j‖r̂_j−r_j‖₂ / σ_r` (no tuned weight).

Every model exposes `forward_features(x) → (tokens (B,N,D), pooled (B,D))` and `forward(x) → pose (B,1,14,3)` (Phase C reuses the encoder).

- `simple_cnn` (reference regressor): Conv(3→32,k(5,3),s(2,1)) → Conv(32→64,k3,s(2,1)) → Conv(64→128,k3,s2) → Conv(128→256,k3,s2), each GroupNorm(8)+GELU; tokens = flattened map (≈60×256); pooled = mean; head Linear(256,256)-GELU-Linear(256,45, zero-init).
- `dtpose_encoder_mlp`: the existing `MAE_ViT(...).encoder` built exactly as in `step4b_r_baseline.build_step4b_r_model` (image (180,20), patch (2,2), 4 layers/4 heads, emb 256), all parameters trainable, from scratch, run unmasked the way `ViT_Pose_Decoder.forward` does (patchify + pos-embedding + cls → transformer → layer_norm); pooled = mean of patch tokens; same MLP head. The legacy GCN/transformer pose decoder is bypassed.
- `dtpose_legacy_decoder`: unmodified `ViT_Pose_Decoder` (`train_encoder=True`), absolute stored-coordinate targets and the Step-4B-R loss `torch.mean(torch.norm(pred−target, dim=-1))` — only the optimizer recipe differs from Step-4B-R. Purpose: single-factor check of "was it just optimization?".

### B4. Frozen run matrix (`step5_config.py`, `step5_single_person_baseline.py`)

Reuse from `step4b_r_baseline.py` by import: `validate_process_environment`, `configure_deterministic_runtime`, `_seed_everything`, `capture_environment`, `parameter_audit`, `state_dicts_equal`, `_mean_per_joint_error`, `compute_mean_pose`, `constant_pose_mpjpe`. Determinism block identical to `step4b_r_config.py` (`PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`, deterministic algorithms on).

Shared optimizer recipe (all runs): AdamW, peak lr 3e-4, weight decay 0.01, linear warmup 200 steps → cosine to 0, grad-norm clip 1.0, batch 32, 40 epochs, `num_workers=0`, no early stopping, final-epoch checkpoint only.

| Run | Model | Input norm | Isolates |
|---|---|---|---|
| R1 | `simple_cnn` | `train_zscore` | reference |
| R2 | `simple_cnn` | `legacy_minmax` | input normalization |
| R3 | `dtpose_encoder_mlp` | `train_zscore` | DT-Pose encoder with a sane head |
| R4 | `dtpose_legacy_decoder` | `legacy_minmax` | legacy decoder under stabilized optimization |

Order of execution:
1. **Smoke (AGENTS.md list), every model:** batch loads, shapes `(B,3,180,20)`/`(B,1,14,3)`, forward, finite loss, backward, parameters change, state-dict save/reload exact, evaluation runs.
2. **Overfit check, R1 and R3:** 64 fixed train samples (every k-th ID so all groups appear), 300 steps, constant lr 1e-3, no weight decay. Pass if final MPJPE ≤ **0.20 ×** the mean-pose MPJPE of those same 64 samples.
3. **Seed 0, R1–R4** on the full `train` split. Log per epoch: train loss, and on `train` / `temporal_holdout` / `group_dev`: absolute MPJPE, centroid error, centroid-relative MPJPE, plus **collapse monitors** (std of predictions across samples, std across joints). Estimated total < 2 h; if the measured first epoch projects > 4 h total, stop and **[ASK]**.
4. **Selection (predeclared):** among runs passing G3–G5 below, the lowest final-epoch `group_dev` absolute MPJPE; ties (<1% relative) → the simpler model (R1 before R3). Uses single-person dev labels only — allowed under both development and strict protocol; state this in the report.
5. **Seeds 1 and 2** for the selected run only.

Final-epoch evaluation per run and split: absolute MPJPE, centroid error, centroid-relative MPJPE, and the canonical single-person numbers from `research_metrics.ResearchMetricAccumulator` (people dim = 1; PA-MPJPE, PCK@0.5 with its documented normalization, count accuracy, P/R/F1; `stored_unit_to_mm=1000` labelled provisional). Baselines with the same code: train-mean pose; train-mean relative skeleton (for centroid-relative). Controls: shuffled-pairing MPJPE (predictions scored against a fixed permutation of labels). Save `predictions_{split}.npz`, `learning_curves.json`, `result.json`, `decision.json`, state dict under `data/_artifacts/step5_single_person/runs/<run>_seed<k>/`.

### B5. Predeclared gate (write into `STEP_5_PROTOCOL.md` before running; thresholds are relative on purpose)

- **G1** all smoke checks pass. **G2** overfit ratio ≤ 0.20.
- **G3 input dependence:** std of predicted centroids across `train` samples ≥ 0.25 × std of ground-truth centroids (mean over axes), and matched `train` MPJPE ≤ 0.70 × shuffled-pairing MPJPE.
- **G4 fits training data:** `train` absolute MPJPE ≤ 0.50 × train mean-pose MPJPE.
- **G5 unseen frames, seen groups:** `temporal_holdout` absolute MPJPE ≤ 0.50 × mean-pose on that split **and** centroid-relative MPJPE < mean-skeleton centroid-relative on that split.
- **G6 unseen groups:** `group_dev` absolute MPJPE ≤ 0.95 × mean-pose **and** centroid-relative MPJPE ≤ 0.95 × mean-skeleton centroid-relative.

Verdicts (selected run, all three seeds):
- **GREEN** = G1–G6 → Phase C may be proposed with group-disjoint dev as primary.
- **YELLOW-POSITION** = G1–G5 pass, G6 fails → the model learns but does not transfer to unseen positions/groups. Stop and report; recommend to the user either (a) Phase C under an explicitly disclosed *official-like temporal protocol* (same groups, held-out frame blocks — mirrors the public benchmark) with group-disjoint results reported alongside, or (b) a data-scaling run on all 23,874 single-person train IDs (**[ASK]** — counts as full training/preprocessing ≈5–6 GB).
- **RED** = G4 fails for all of R1–R4 → not an optimizer problem. Stop and report with evidence; the only permitted follow-up inside this step is a *read-only* CSI↔label alignment audit (ID pairing, `process_ids` output vs raw `.mat`, per-group CSI statistics). No further training variants.

`STEP_5_REPORT.md` must also state what R2 vs R1 and R4 vs Step-4B-R show (normalization; whether the legacy decoder is usable), and the full table for every run, including failed ones.

**Stop after Phase B.** Do not begin Phase C without the user's go-ahead.

---

## 5. Phase C — "Step 6": minimal compositional pilot (arms A/B/C) — specified now, executed only after user approval of Phase B

Follow `docs/RESEARCH_PROTOCOL.md` §5–§7 exactly (arms, collapse controls, three seeds, Go/Yellow/No-go thresholds — **do not change them**). First deliverable is `docs/STEP_6_PROTOCOL.md` freezing everything below; **[ASK]** the user to approve that protocol before any training. All results are **development-protocol** results (multi-person internal-dev labels used for evaluation and the go/no-go decision, never for training) and must be labelled so.

**Data**
- Labelled: the Step-5 `train` split, identical IDs for A/B/C.
- Unlabelled pool U (arms B, C): CSI only via `feeder/piw3d_unlabelled.UnlabelledPiW3DCSIDataset` (returns only `{"csi"}`). Count-blind selection: **all 172 internal-train groups**, the same rule for every group (up to 64 evenly spaced frames → ≈11,000 samples, ≈2.2 GB preprocessing **[ASK]** before running it). Selection code may read group names but must not branch on the person-count character; training code never sees IDs. Input z-score stats recomputed on U (label-free) and frozen.
- Evaluation: new `feeder/piw3d_dev_eval.py` returning `csi`, variable-size `poses (people,14,3)`, `sample_id`; constructor refuses any ID not in `internal_development_ids.txt`; imported only by evaluation code (add a test that the training module does not import it). All 43 internal-dev groups (8/21/14 with 1/2/3 people), up to 32 evenly spaced frames each. Official test untouched.

**Model** (`model/compositional.py`, new): Step-5 selected encoder → tokens; K = 3 person queries + 1 background query through a 2-layer transformer decoder (cross-attention to tokens; no iterative slot attention, no temporal module) → slot vectors `z_1..z_3`, `z_bg`; shared Step-5 pose head per slot (centroid + relative pose via `PoseTargetCodec`); presence logit per slot; latent-additive recomposition `u = φ_bg(z_bg) + Σ_k p_k·φ(z_k)`, `x̂ = Dec(u)` reconstructing the normalized input CSI (a fixed, non-learned target).

**Arms (same encoder, slots, heads, step budget, batch composition, seeds 0/1/2)**
- **A — single-person only:** labelled batch; Hungarian-assign the one GT person to the best slot; pose loss on it; presence BCE target 1 for it, 0 for the other slots (single-person status of labelled samples is permitted metadata).
- **B — reconstruction only:** A + reconstruction of unlabelled CSI from the pooled encoder feature through the same `Dec` (no slot composition).
- **C — minimal compositional:** A + (i) slot-composed reconstruction of unlabelled CSI as above; (ii) feature-space composition supervision: add the encoder token maps of 2–3 labelled single-person samples from different groups, run the slot module on the sum, supervise with the set of their poses (Hungarian) and presence = number mixed. This is **synthetic feature-space composition** — mark `uses_feature_space_synthetic_composition: true` in the config and disclose it; raw CSI arrays are never added.
- Optional disclosed control **C0** = C without (i), not part of the Go rule.
- Fully supervised upper reference: defer; needs a separate, explicitly labelled task.

**Evaluation** with `research_metrics.ResearchMetricAccumulator` only: presence threshold 0.5 and 500 "mm" match threshold frozen (provisional metre assumption stated), per-count breakdown (1/2/3 people), matched MPJPE, PA-MPJPE, centroid/root error, PCK, exact count accuracy, missed/duplicate/spurious, person P/R/F1; oracle-count numbers only as labelled diagnostics. Collapse controls from the protocol: duplicate-slot control, train-mean-pose control, active-slot count, pairwise slot similarity, per-slot usage, prediction variance across inputs, reconstruction error on shuffled CSI.

**Process:** smoke tests → tiny overfit (16 labelled + 16 composed samples) → 3 seeds × arms → apply the frozen rule → `docs/STEP_6_REPORT.md` with verdict **Go / Yellow / No-go**, every seed reported. If total projected training exceeds ~4 GPU-hours, **[ASK]** first. No-go is a valid, reportable outcome; the 0.1–1% label-efficiency track is a separate, explicitly labelled task that must not silently replace it. Stop after the report.

---

## 6. Critical files

Reuse (import, do not edit): `piw3d_manifest.py` (`parse_sample_id`, `build_candidate_group_manifests`), `step4b_subset.py` (`is_single_person_id`, `select_evenly_spaced_frames`, `freeze_step4b_subset`), `make_step4b_pilot_data.py` (`build_step4b_pilot_data`), `data/person_in_wifi_3d/process.py` (`process_ids`), `feeder/piw3d_single_person.py` (`SinglePersonPiW3DDataset`, `collate_single_person`, `make_single_person_dataloader`), `feeder/piw3d_unlabelled.py`, `research_metrics.py` (`ResearchMetricConfig`, `ResearchMetricAccumulator`, `evaluate_oracle_count_diagnostic`), `step4b_r_baseline.py` (determinism, audits, `build_step4b_r_model`), `step4b_r_config.py` (pattern for frozen configs), `model/model.py` (`MAE_ViT`, `ViT_Pose_Decoder`).

New in Phase A: `diagnose_step4b_r_collapse.py`, `tests/test_step4b_r_collapse_diagnosis.py`, `docs/STEP_4B_R_COLLAPSE_DIAGNOSIS.md`, `docs/NEXT_PHASE_PLAN.md`; edit `docs/CURRENT_TASK.md`.
New in Phase B: `step5_config.py`, `step5_subset.py`, `make_step5_data.py`, `feeder/piw3d_step5.py`, `model/research_models.py`, `step5_single_person_baseline.py`, `tests/test_step5_*.py`, `docs/STEP_5_PROTOCOL.md`, `docs/STEP_5_REPORT.md`.
New in Phase C: `model/compositional.py`, `feeder/piw3d_dev_eval.py`, `step6_config.py`, `make_step6_data.py`, `step6_compositional_pilot.py`, `tests/test_step6_*.py`, `docs/STEP_6_PROTOCOL.md`, `docs/STEP_6_REPORT.md`.

---

## 7. Verification

- **After every code change:** the pytest command in §2; count must be ≥ 74 + new tests, 0 failures; paste the real tail of the output into the report.
- **Phase A:** `git status --short` clean apart from locally-excluded paths; `git log --stat` shows the consolidation commits; `git diff c494564 --stat` covers the six formerly-stashed files; diagnosis numbers match §1.
- **Phase B:** freeze JSON exists before the first run timestamp; `decision.json` per run; untrained-model-equals-mean-pose unit test passes; collapse monitors non-zero for passing runs; a reloaded state dict reproduces the final metrics exactly; label-access tests pass (no multi-person ID, no dev ID in train, no official-test path opened — assert by monkeypatching file opens in a test).
- **Phase C:** unlabelled batch keys are exactly `{"csi"}`; training module does not import the dev-eval feeder; metrics on a hand-built toy case match expected Hungarian results (existing `tests/test_research_metrics.py` pattern); report includes all seeds, both collapse controls, and the protocol label "development protocol".
- **Each phase ends with:** completion report (what was done, files changed, commands actually run, test results, remaining uncertainties, status, recommended next task) and an updated `docs/CURRENT_TASK.md`.
