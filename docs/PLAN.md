# AIJ 2026 Task 4 (Vision-Action Data Factory): how to approach it

## Context

**The task.** Build `generate_dataset.py`, which turns Bridge, Fractal, Language Table (LeRobot v3) and Egocentric-100K into a ShareGPT JSONL. That JSONL fine-tunes SmolVLM2-500M (the `smolvlm2/` checkpoint), which then initializes a SmolVLA policy trained on LIBERO (the `action_expert/` checkpoint).

**Scoring.** `Score = 100·(0.4·mean(8 VLM benches) + 0.6·VLA_SR)`. The VLA success rate comes from a private LIBERO-like bench with new objects, layouts, wordings and camera positions.

**Our situation.**
- Nothing has been run yet.
- Compute: 1× RTX 5090 32 GB (always available), 2× 5090 (sometimes available), and 100 H100-hours on Colab.
- The deadline is **2026-10-20**.

**Inputs reviewed.** `docs/research.pdf`, the rules (§3.10, `docs/organizers/contest_rules.pdf`), `vendor/participant/`, `vendor/baseline/`, the demo submission (`vendor/demo_submission/`), and the organizer answers in `docs/organizers/aij_comments.md`.

---

## 1. Ground rules settled by the organizers (aij_comments.md)

**VLM SFT**
- Follow the repo README and configs, which are free to edit. They take priority over the "fixed parameters" wording in the rules.

**VLA stage**
- Allowed:
  - edits to `vla_libero.yaml`, including `train_expert_only` and `freeze_vision_encoder`, so the VLM and vision encoder can be unfrozen;
  - **any number of GPUs** (effective batch is 4×N);
  - all of LIBERO v3.0 and changing `episode_count`;
  - changing **`chunk_size` and `n_action_steps`**.
- Not allowed:
  - extra LIBERO data such as libero_90;
  - **image augmentations**;
  - paraphrasing LIBERO instructions.

  The VLA must train on the provided v3.0 data, unmodified.

**VLM data**
- Extracted-frame directories, image normalization and even generative frame editing are allowed if fully reproducible.
- **LIBERO frames are not allowed** in the VLM dataset.

**Leaderboard**
- Only the uploaded weights are run, on 1×H100 within **210 min**. That limit covers inference, not dataset generation.
- `generate_dataset.py` is run only for finalists at the end. There is no hard time limit, but it must be "reasonable", and that rules out weeks.

**ZIP size**
- **5 GB, the same for finalists**, and it must hold everything: code, checkpoints and any generator model weights. Do not include datasets or images, because unpacked images exceeded the judge's inode limit.

**The scoring bug**
- Until Sep 30 a bug scored VLA as 0. The baseline's 9.909 is therefore pure VLM, an average of **0.2477** over the 8 benches.
- The bug is fixed. **The limit is 3 submissions per day through Oct 8**, then 1 per day.

**Still unanswered** (ask again, §3): network access for the generator in the control run, and whether AGPL code is allowed. The latter matters for Ultralytics YOLOv8, which the baseline uses.

---

## 2. Facts from the repo that change or correct the research PDF

1. **Two checkpoints, two VLM copies, so two scores that can be optimized separately.**
   - The demo `action_expert/model.safetensors` holds a **full VLM copy** (`model.vlm_with_expert.vlm.*`, 460M params plus lm_head) and 32 expert layers.
   - LIBERO eval loads from that file (`scaling-lawyer/.../libero_eval/smolvla_model.py`). The 8 benches load `smolvlm2/`.
   - With `train_expert_only: false` the VLA's VLM is re-adapted on LIBERO, and our SFT matters for the VLA **only as initialization**.
   - The PDF's frozen-backbone worry holds only if the frozen config wins.
2. **All 32 VLM layers are used**, not 16 as the PDF says (`num_vlm_layers: -1`).
3. **The default VLA run is weak.** It trains an expert from scratch for 30k steps × batch 4 with the VLM frozen, about half an epoch of LIBERO by my estimate. The VLA-stage config is probably the biggest lever.
4. **Value of a point.** 1 pp of VLA success equals 0.6 score points; 1 pp on one VLM bench equals 0.05. **So 1 pp of VLA success is worth 12 pp on one benchmark.**
5. **The likely cause of the "stage 12" failures in the comments.**
   - The demo `action_expert/config.json`, `train_config.json` (`policy.vlm_model_name`) and `policy_preprocessor.json` (`tokenizer_name`) all hard-code **`/home/ubuntu22/aij_robotics_judge/submission/smolvlm2`**.
   - Our pipeline writes a local path there. Offline, that becomes an HF download and the run fails.
   - **Rewrite these three fields to the judge path before zipping.**
   - Also save `smolvlm2/` in bf16 to save about 1 GB: the demo stores it as 2.03 GB fp32, and the eval loads bf16 anyway.
6. **Image tokens.**
   - SFT and the benches see about 1,100 tokens per square image (upscaled to 2048, then 16 crops plus 1 global; see `LLaMA-Factory/.../mm_plugin.py: _get_smolvlm_rows_cols`).
   - The VLA sees one padded 512×512 view (64 tokens).
   - With `cutoff_len: 4096`, keep **3 or fewer images per sample**, or raise `cutoff_len`.
7. **Eval formats to train toward:**
   - **ERQA**: "Answer with the single letter… A, B, C, or D", with `max_new_tokens=16`.
   - **MSMU**:
     - The primary metric is `msmu_overall_accuracy`.
     - Numeric answers count if the ratio to the target is under 1.25.
     - Qualitative answers are matched leniently.
     - The suffixes are "Give a concise answer." and "output only the number and unit".
   - **CrossPoint, BLINK, EmbSpatial, MMMU**: read their upstream lmms-eval prompts and parsers after setup. CrossPoint's coordinate format is the critical one.
8. **The generator's weight budget is about 2–2.5 GB** (5 GB minus about 1.4 GB for action_expert, about 1 GB for smolvlm2 in bf16, and some headroom).
   - That rules out a 4–8B teacher VLM like the baseline's Qwen3.5-4B.
   - **Most labels must come from proprioception, geometry and colour, without any models.**
   - Use small Apache/MIT models: Florence-2-base (MIT; captions, detection and grounding in one model, about 0.5 GB), Depth Anything V2-Small (Apache, about 0.1 GB), SAM2.1-tiny or small (Apache), OWLv2 or Grounding-DINO-T (Apache).
   - Avoid AGPL YOLOv8 and CC-BY-NC models such as CoTracker until licensing is clarified.
9. **Reusable baseline code:** `baseline/src/bpc/storage.py: build_llama_factory_record` (ShareGPT records), `src/sharding.py` and `src/multigpu.py`.

---

## 3. Remaining questions for the organizers

Post these in the comments again; both are still unanswered.
1. Does `generate_dataset.py` have network or HF access in the finalist control run? If not, generator weights must fit in the 5 GB ZIP.
2. Are AGPL-3.0 dependencies (Ultralytics) allowed?

Until they answer, design for **offline use within the 5 GB limit**.

---

## 4. Strategy

- **Track A, the VLA stage (60%, highest priority right now).** Find the strongest config that is allowed and reproducible, on the **base** VLM. Knobs: freezing, batch size and GPU count, steps, `chunk_size` / `n_action_steps`.
  - The 3-per-day window through Oct 8 is the cheapest moment to calibrate against the private bench. Unfreezing may overfit to LIBERO's visuals and hurt under the private perturbations, and only the leaderboard shows that.
  - Keep inference within 210 min on 1×H100: a smaller `n_action_steps` means more policy calls.
- **Track B, the dataset (40%, plus VLA initialization).**
  - Use model-free, label-exact, in-domain data in the benchmarks' exact formats: Bridge, which is closest to LIBERO, plus Language Table geometry.
  - Then add small-model geometry: depth, boxes and correspondence.
  - With an unfrozen VLA stage, SFT can push the benches harder. Still check that "SFT'd init" is not worse than "base init" for the VLA.

---

## 5. Phase 0 (Oct 6–8): infrastructure and Track A (*set up first*; uses the 3-per-day window)

1. On the always-on 5090:
   - build `participant/Dockerfile` (cu128, works on Blackwell);
   - run the smoke test (`scripts/run_pipeline.sh configs/smoke.yaml --smoke`) and `scripts/validate_submission.sh`;
   - run `scripts/setup_score_selfcheck.sh`;
   - download `HuggingFaceVLA/libero@v3.0`.
2. **A packaging script** (new, `tools/pack_submission.py`):
   - rewrite the three judge-path fields (§2.5);
   - convert `smolvlm2/` to bf16;
   - include `generate_dataset.py`, `src/`, `config.yaml`, `annotations.jsonl`, the README and the **final SFT and VLA yaml files**;
   - assert the zip is under 5 GB, has no images, and has no `hf://` or remote references;
   - run `validate_submission.sh` and the libero self-check on the unpacked zip.
3. **VLM baseline.** Evaluate the untouched SmolVLM2 on the 8 benches with `log_samples`. Report per-subtask accuracy against chance, the parse-failure rate, answer lengths, letter bias and the pointing-output format. Compare the average with the 0.2477 baseline.
4. **Track A runs on the base VLM.** Evaluate each on local `libero_spatial` with **5 or more episodes per task** (the default of 1 is noise).
   - A0: default.
   - A1: `train_expert_only: false` with the encoder frozen.
   - A2: both unfrozen.
   - Then batch size or GPU count and steps, and `n_action_steps` (e.g. 50 versus 10–25), on the best of these.
   - Use the 2×5090 machine and Colab H100 (pip env; an unfrozen VLM needs more VRAM).
5. **Submissions (up to 3 per day on Oct 7–8):** A0, the best unfrozen variant, and the batch, steps and `n_action_steps` variants. Read the current post-fix leaderboard to see typical VLA magnitudes.

## 6. Phase 1 (Oct 8–11): data access and v0 dataset

**Downloads.** Get only `meta/` plus a few chunk files per LeRobot dataset (HF `allow_patterns`), and a handful of Ego100K clips. Episode selection is **deterministic from metadata**, so the local subset equals what the full-data control run picks.

**Code** (becomes `submission/src/`):
- `generate_dataset.py`: `--input --output --config`, global seed, auto-discovers datasets via `meta/info.json`.
- `src/readers/lerobot_v3.py`: parquet offsets plus video timestamps. Seek-decode only the chosen frames (PyAV) and save JPGs under `<output_dir>/frames/`.
- `src/readers/ego100k.py`.
- `src/generators/*.py`: one per signal family, with benchmark-style templates, balanced letters and varied phrasing.
- `src/mix.py`: quotas from `config.yaml`, dedup, ≤3 images per sample, a token-length check.
- `src/writer.py`: adapted from `baseline/src/bpc/storage.py`.

**v0 generators** (model-free):
1. **Proprio and gripper (Bridge, Fractal):**
   - gripper state, holding, phase, progress bin;
   - qualitative motion direction in image-frame words only (never numbers or waypoints, because of the rules);
   - frame-order MCQ;
   - success yes/no (start versus end frames, plus instruction-mismatch negatives).
2. **Language Table geometry:** colour-segment the 8 blocks. This gives exact left/right/above/below and closest/farthest MCQs (EmbSpatial style), counting, pointing in CrossPoint's format, and "which block moved".
3. **Bridge multi-view:** the same scene from 2 cameras. "Same object?" questions and view-change MCQs, for ERQA and BLINK multi-view.
4. **Instruction grounding:** template paraphrases of Bridge, Fractal and LT instructions mapped to the target object. This is allowed for *our* datasets; only LIBERO paraphrasing is banned.
5. **Anchor slice** (about 15–25%): Florence-2 captions, plus possibly self-distillation from the base SmolVLM2's own answers on our frames. That would put about 1 GB of weights in the budget, so decide based on §3.

**Run v0.** Use about 30–60k samples and compare LoRA with full SFT, then the 8 benches. Train the VLA with the leading Track-A config from the v0 init and from the base init, compare, and submit.

## 7. Phase 2 (Oct 11–17): targeted iteration (one submission per day)

Choose each day's experiment from the error analysis:
1. **SFT knobs:** size (20k / 60k / 150k), LoRA versus full, LR, frozen vision tower, anchor ratio.
2. **Small-model geometry:**
   - Depth Anything V2-Small for close/far and BLINK-style relative depth with drawn A/B markers;
   - OWLv2 or Florence-2 boxes, kept only where two sources agree;
   - MSMU-style size and distance comparisons;
   - counting.
3. **Correspondence:** across Bridge cameras and across time. Use an Apache or MIT tracker, or LT colour blobs and SAM2 mask propagation. This serves CrossPoint and BLINK.
4. **Failure and recovery:** templated, and kept small.

**Guardrail.** Use it only if Track A ends up frozen: per-layer hidden-state drift against the base on LIBERO-like inputs.

## 8. Phase 3 (Oct 17–20): freeze and package

- `generate_dataset.py` runs offline from vendored weights and is deterministic (seeds, sorted inputs, greedy decoding). Its runtime is reasonable and documented (hours, not days), with optional sharding across GPUs.
- Regenerate `annotations.jsonl` and check it is identical to the training version.
- Run `pack_submission.py` (≤5 GB, judge paths, bf16) and the local self-check on the unpacked zip.
- Pick the final 3; the best private score of the three counts.

## 9. Compute allocation

| Machine | Use |
|---|---|
| 5090 (always available) | Data pipeline, SFT, 8-bench VLM evals |
| 2×5090 (intermittent) | Track-A VLA runs (multi-GPU is allowed) plus libero_spatial |
| Colab H100 (100 h) | Unfrozen-VLM VLA runs, larger batches. No Docker, so use a pip env and keep jobs resumable |

## 10. Pitfalls

- The ZIP must have **no hard-coded local or HF paths**. Use the judge path for `vlm_model_name`/`tokenizer_name`, keep it under 5 GB, and include no images.
- No augmentations or LIBERO paraphrasing in the VLA stage, and no LIBERO frames in the VLM data.
- No numeric action, delta or waypoint answers in the VLM data. Never use benchmark images.
- An unfrozen VLA may win locally and lose on the private bench, so trust the leaderboard.
- Treat local differences under about 3–5 pp of success as noise.

## Verification

- The smoke pipeline and `validate_submission.sh` pass. The base-model self-check gives `leaderboard_score.json`, and the VLM average is about 0.25.
- The first Track-A submission passes the judge (no stage-12 failure) and shows nonzero VLA.
- Each experiment logs: per-bench deltas, parse failures, libero_spatial success with 5 or more episodes per task, the SFT and VLA configs, and the leaderboard score.
- Final check: rerunning `generate_dataset.py --input <raw> --output … --config config.yaml` offline produces an identical JSONL, training from it works, and the packed zip passes the self-check.
