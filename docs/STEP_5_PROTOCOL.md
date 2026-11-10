# Step 5 Protocol — single-person baseline that provably learns (Phase B)

Status: **frozen before any training run.** Source: `docs/NEXT_PHASE_PLAN.md` §4. The machine-readable
copy of everything below is `step5_config.py` (`STEP5_CONFIG`); the runner
`step5_single_person_baseline.py freeze-protocol` writes a JSON copy plus SHA-256 hashes of this document
and of every Step-5 source file to `data/_artifacts/step5_single_person/protocol_freeze/`. Every later
command (`smoke`, `overfit`, `train`) refuses to run if the config or any hashed file changed. No
threshold, recipe, or selection rule may change after seeing results; final-epoch checkpoints only.

Binding rules: `AGENTS.md`, `docs/PROJECT_CONTEXT.md`, `docs/RESEARCH_PROTOCOL.md`, plan §2.

## 1. Data (B1)

Inputs: `data/_artifacts/piw3d_candidate_group_manifests/internal_{train,development}_ids.txt` (Step-3
candidate-filename-group manifests). Single-person filter: `step4b_subset.is_single_person_id`
(filename metadata; permitted for selecting the labelled single-person subset). Official `test_data` is
never opened. No multi-person sample enters any split.

Rule (deterministic, no seed). For each single-person internal-train group (51), frames sorted by frame
index, `n` = count:
- temporal-holdout block = sorted positions `[n//5, n//5 + h)`, `h = min(49, n//10)`;
- guard = 5 positions on each side of the block, excluded from every split;
- `train` = up to 64 evenly spaced frames from the remaining positions;
- `temporal_holdout` = up to 16 evenly spaced frames from the block.
For each single-person internal-development group (8): `group_dev` = up to 64 evenly spaced frames.
"Evenly spaced" = `step4b_subset.select_evenly_spaced_frames`.

Split labels (used verbatim in all reports):
- `train`: internal-train single-person groups, frames outside the holdout block and guards.
- `temporal_holdout`: **same-group temporal holdout — NOT group-disjoint (official-like).**
- `group_dev`: internal-development single-person groups — **candidate-filename-group disjoint** from
  train. Never described as subject-, recording- or environment-disjoint.

Freeze result (produced before this protocol was frozen, `freeze_v1/step5_freeze.json`):
train 3,264 IDs, temporal_holdout 804 IDs, group_dev 512 IDs, 510 guard IDs excluded; 51 + 8 groups as
expected. Preprocessing: copy selected `.mat`/keypoint files to `data_v1/`, run
`data.person_in_wifi_3d.process.process_ids`, validate shapes/finiteness/exact amplitude-phase
concatenation (`make_step4b_pilot_data._validate_processed_sample`). Originals untouched.

Train-split-only statistics written once into `freeze_v1/`: `train_input_stats_v1.npz` (z-score) and
`pose_codec_stats_v1.json` (target codec).

## 2. Input normalisation (B2)

`feeder/piw3d_step5.Step5SinglePersonDataset` overrides only `read_csi`:
- `legacy_minmax`: identical to `SinglePersonPiW3DDataset` (per-sample min-max over the whole amplitude
  block and whole phase block).
- `train_zscore`: `(x − mean)/max(std, 1e-6)`, mean/std shape `(3,180,1)` over train samples and the 20
  time steps of stored `csi_ap`, loaded read-only.
Batches are `{"csi", "pose"}`. The runner caches each split's tensors in memory once (identical values).

## 3. Models and targets (B3)

`PoseTargetCodec` (train-split labels only): centroid `c` = mean of 14 joints, `r = pose − c`;
`μ_c, σ_c` per axis; `μ_r` (14×3); scalar `σ_r` = std of `r − μ_r`. Output `o ∈ R^45`:
`ĉ = μ_c + σ_c·o[:3]`, `r̂ = μ_r + σ_r·o[3:]`, `pose = ĉ + r̂`. Last linear layer zero-initialised, so
the untrained model outputs exactly the train-mean pose (unit-tested). Loss
`mean‖ĉ−c‖₂ / mean(σ_c) + mean_j‖r̂_j−r_j‖₂ / σ_r` (no tuned weight).

- `simple_cnn`: 4 convs (3→32 k(5,3) s(2,1); 32→64 k3 s(2,1); 64→128 k3 s2; 128→256 k3 s2; padding k//2),
  each GroupNorm(8)+GELU; tokens = 12×5 = 60 tokens of 256; pooled = mean; head
  Linear(256,256)-GELU-Linear(256,45, zero-init).
- `dtpose_encoder_mlp`: `MAE_ViT(...).encoder` with the Step-4B-R settings (image (180,20), patch (2,2),
  4 layers, 4 heads, emb 256), from scratch, all trainable, unmasked forward (patchify + pos + cls →
  transformer → layer_norm); 900 patch tokens; pooled = mean of patch tokens; same head.
- `dtpose_legacy_decoder`: unmodified `ViT_Pose_Decoder` (`train_encoder=True`) from
  `step4b_r_baseline.build_step4b_r_model`; absolute stored-coordinate targets; Step-4B-R loss
  `torch.mean(torch.norm(pred−target, dim=-1))`. Only the optimiser recipe differs from Step-4B-R.

## 4. Run matrix and recipe (B4)

| Run | Model | Input norm | Isolates |
|---|---|---|---|
| R1 | `simple_cnn` | `train_zscore` | reference |
| R2 | `simple_cnn` | `legacy_minmax` | input normalisation |
| R3 | `dtpose_encoder_mlp` | `train_zscore` | DT-Pose encoder with a sane head |
| R4 | `dtpose_legacy_decoder` | `legacy_minmax` | legacy decoder under stabilised optimisation |

Shared recipe: AdamW, peak lr 3e-4, weight decay 0.01 on all trainable parameters, batch 32
(`drop_last=False`, 102 steps/epoch), 40 epochs (4,080 steps), lr factor `(s+1)/200` for step `s < 200`,
then `0.5(1+cos(π(s−200)/(S−200)))` with `S` = total steps; grad-norm clip 1.0; `num_workers=0`; shuffle
with a `torch.Generator` seeded by the run seed; no early stopping; final-epoch checkpoint only.
Determinism: `PYTHONHASHSEED=0`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
`step4b_r_baseline.configure_deterministic_runtime` (deterministic algorithms, no TF32); model built after
seeding python/numpy/torch with the run seed. Budget: if the first epoch of any run projects > 4 h, the
run aborts and the user is asked; if the whole matrix projects > 4 h, stop and ask.

Order: (1) smoke for R1–R4; (2) overfit R1 and R3; (3) seed 0 of R1–R4; (4) selection; (5) seeds 1, 2 of
the selected run; (6) verdict. If G1 fails, or G2 fails for both R1 and R3, stop after that stage and
report.

**Smoke (G1), every run config:** one batch loads with keys `{csi,pose}`; shapes `(B,3,180,20)` /
`(B,1,14,3)`; forward shape `(B,1,14,3)`; `forward_features` returns `(B,N,D)`/`(B,D)`; codec models
output the train-mean pose before training; finite loss; backward with non-zero gradient norm; one
optimiser step changes parameters; state-dict save/reload bit-exact (state and predictions); evaluation
runs.

**Overfit (G2), R1 and R3:** 64 train samples at positions `i·k`, `k = 3264//64 = 51`, `i = 0..63` (all
51 groups present — recorded); 300 steps, full 64-sample batch in fixed order, AdamW constant lr 1e-3, no
weight decay, clip 1.0, seed 0. Pass if final eval-mode MPJPE ≤ 0.20 × MPJPE of the mean pose of the same
64 samples. R2 shares R1's architecture and is assigned R1's G2 result; R4 has no overfit check (G2
recorded "not applicable").

**Per-epoch logging:** train loss; on train / temporal_holdout / group_dev (eval mode): absolute MPJPE,
centroid error, centroid-relative MPJPE, prediction std across samples (mean over joints/axes) and across
joints (mean over samples/axes).

**Final evaluation per run and split:** absolute MPJPE, centroid error, centroid-relative MPJPE (each
skeleton minus its own 14-joint centroid), collapse monitors, shuffled-pairing MPJPE (predictions vs a
fixed permutation of labels, `np.random.default_rng(20261110)`), and `research_metrics.ResearchMetricAccumulator`
with people dim = 1, presence = 1: PA-MPJPE, PCK@0.5 (scale = distance between ground-truth joints 6 and
4 in stored units), root error (joint 0 — identity unproven), exact count accuracy, missed/duplicate,
person P/R/F1 at a 500 "mm" match threshold. All distances in **stored units**; "mm" values use
`stored_unit_to_mm = 1000` and are **provisional, assuming stored unit = metre**. Baselines with the same
code: train-mean pose (absolute) and train-mean relative skeleton (centroid-relative). A reloaded final
state dict must reproduce the final predictions bit-exactly.

## 5. Predeclared gates (B5)

- **G1** all smoke checks pass for every run config.
- **G2** overfit ratio ≤ 0.20 (per architecture as above).
- **G3 input dependence** (train split): mean over axes of the std of predicted centroids ≥ 0.25 × the
  same for ground-truth centroids, **and** matched absolute MPJPE ≤ 0.70 × shuffled-pairing MPJPE.
- **G4 fits training data:** train absolute MPJPE ≤ 0.50 × train mean-pose MPJPE.
- **G5 unseen frames, seen groups:** temporal_holdout absolute MPJPE ≤ 0.50 × mean-pose MPJPE on that
  split **and** centroid-relative MPJPE < mean-skeleton centroid-relative MPJPE on that split.
- **G6 unseen groups:** group_dev absolute MPJPE ≤ 0.95 × mean-pose **and** centroid-relative MPJPE
  ≤ 0.95 × mean-skeleton centroid-relative.

A run that stops on a non-finite loss or gradient is recorded as **crashed** (`decision.json` with
`crashed: true`) and fails G3–G6; it is reported, never re-run with changed settings.

**Selection (seed 0):** among runs passing G3–G5, the lowest final-epoch group_dev absolute MPJPE; runs
within 1% relative of the best are tied and the simplest wins (R1 < R2 < R3 < R4). Uses single-person
internal-development labels only — allowed under both the development and the strict protocol.

**Verdict** (selected run, seeds 0, 1, 2):
- **GREEN** = G1, G2 (where applicable) and G3–G6 pass for all three seeds → Phase C may be proposed with
  group-disjoint dev as primary.
- **YELLOW-POSITION** = G1, G2 (where applicable), G3–G5 pass for all three seeds, G6 fails for at least
  one seed → stop; recommend (a) Phase C under an explicitly disclosed official-like temporal protocol
  with group-disjoint results alongside, or (b) a data-scaling run on all 23,874 single-person train IDs
  (**ask** first).
- **RED** = G4 fails for all of R1–R4 (seed 0) → not an optimiser problem; stop; only a read-only CSI↔label
  alignment audit is permitted inside this step.
- **INCONCLUSIVE** = none of the above (e.g. no run passes G3–G5 while some run passes G4, a selected-run
  seed fails G3–G5, or G1/G2 fail) → stop and report; no improvised follow-up.

`docs/STEP_5_REPORT.md` reports every run including failed ones, R2 vs R1 (normalisation), R4 vs Step-4B-R
(is the legacy decoder usable), and the full gate table per run and seed.

## 6. Label access

Training loss: train-split single-person labels only. Codec and z-score statistics: train split only.
temporal_holdout labels: evaluation only. group_dev labels: evaluation and the predeclared selection only.
No multi-person IDs, no official test data, no sample-ID parsing inside the training loop.

## 7. Execution environment note

Code and commits live in the worktree branch `research/step5-single-person-baseline`; data and artifacts
live in the main checkout's git-ignored `data/` and are addressed with `--repo-root
"J:/Zero-shot 3D Multi-person Pose Estimation"`.
