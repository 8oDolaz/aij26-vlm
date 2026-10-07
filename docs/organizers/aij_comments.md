# Vision-Action Data Factory (AI Journey 2026) — Comments Q&A

Organizer/staff accounts: @ferry3563, @blick8624, @blanda7513. All others are participants.
Format: Q = participant question, A = reply. Dates are 2026 (publish date).

---

## Dataset generation / data rules

**Q** @conn4445 (Aug 31): Is it allowed for generate_dataset.py to create a directory of extracted frames and reference it? Are we allowed any image normalization?
**A** @ferry3563 (Sep 1): Creating frame directories is allowed if fully reproducible. Image normalization is allowed under the same condition.

**Q** @conn4445 (Sep 20): Can annotations.jsonl reference frames from HuggingFaceVLA/libero (v3.0), used for action expert training, i.e. VLM questions on both LIBERO cameras? Or only the four sources from the Data tab?
**A** @ferry3563 (Sep 21): No, referencing LIBERO when generating the VLM dataset is not allowed.

**Q** @conn4445 (Sep 20): Is generative frame editing (inpainting of background/textures) allowed in generate_dataset.py?
**A** @ferry3563 (Sep 21): Allowed if fully reproducible.

**Q** @hamill5759 (Sep 27): During the finalist control run, will generate_dataset.py have network access (internet/HuggingFace), or must it work fully offline within the 5 GB ZIP? (Affects whether to vendor annotation-model weights.)
**A** — no answer.

**Q** @hamill5759 (Sep 27): What hardware will the finalist control run use for dataset generation (GPU count/type, CPU/RAM)? Inference is known to be 1×H100.
**A** — no answer.

**Q** @hermann5422 (Sep 5): Are AGPL-3.0 libraries allowed?
**A** — no answer.

---

## Training parameters / compute budget

**Q** @hand8710 (Sep 3): The site says parameters and compute budget are fixed, but the participant README and configs say LR, steps, batch and full/LoRA can be changed. Which is the truth?
**A** @ferry3563 (Sep 3): VLM training strategy may depend on the data you generate, so follow the README and configs for allowed parameters.

**Q** @conn4445 (Sep 14): Does the "follow README and configs" answer also cover stage 2 (action expert) — can steps, batch size and LR be changed, or are they fixed as the site says ("parameters and compute budget for both training stages")? Architecture and LIBERO dataset are untouched.
**A** @ferry3563 (Sep 16): Please follow the README and configs.

**Q** @dach5656 (Sep 17): Can configs/vlm_sft.yaml be changed (LR, epochs, vision tower freezing)? The repo says "free to edit", the rules say parameters and compute budget of both stages are fixed. Which has priority in the control run?
**A** @ferry3563 (Sep 21): Follow the README and configs from the repository — they have priority.

**Q** @conn4445 (Sep 22): May train_expert_only and freeze_vision_encoder in vla_libero.yaml be changed (training the VLM/encoder in stage 2)?
**A** @blick8624 (Sep 24): Yes, both can be changed, including fine-tuning the VLM and vision encoder at the VLA stage; the config template may be edited. Unfreezing increases VRAM use and training time. After changing settings, verify the final `submission` loads and passes the local self-check.

**Q** @conn4445 (Aug 31): Action expert training is defined by steps: 30000 with batch_size: 4. In lerobot batch_size is per process, so on N GPUs effective batch is 4×N and examples seen grow N times. Does "fixed compute budget" mean single-GPU training, or is multi-GPU allowed? train_config.json doesn't record GPU count, so the cases are indistinguishable.
**A** @blick8624 (Sep 7): You can train on any GPU and any number of them.

**Q** @conn4445 (Sep 7): How are the 16×H100 on your server interconnected? Can we share model weights between them?
**A** @blick8624 (Sep 7): Inference of your weights is always on 1×H100.

**Q** @trantow2337 (Sep 23): (1) LIBERO is large; using it for VLM was forbidden. Can the whole dataset be used for VLA training? The yaml in participants.zip uses only part — can the config be changed? (2) Does the 210-minute limit cover the whole pipeline (ShareGPT building, dependency install, training) or only generate_dataset.py? Earlier answer was that dataset generation isn't run at this stage; what about finalists?
**A** @blick8624 (Sep 24): (1) Yes, for VLA training the full LIBERO v3.0 can be used and the sample size in vla_libero.yaml can be changed. The restriction on using LIBERO for separate VLM training remains. (2) generate_dataset.py is not run in the evaluation now. The 210-minute limit applies only to running your weights to get leaderboard metrics. Finalists need a fully reproducible solution; there is no explicit time limit for reproducing their results, but it should be reasonable (two weeks or a month is not good).

**Q** @conn4445 (Oct 4): (1) Can other LIBERO demos (e.g. libero_90) be added for action expert training if converted to the same format and reproducible, or only the provided v3.0 dataset (40 tasks, 1693 episodes)? (2) Are LeRobot image augmentations (dataset.image_transforms: brightness, contrast, saturation, hue, sharpness, random crop) allowed if parameters are documented? (3) Can paraphrased instructions for LIBERO v3.0 tasks be used (offline, reproducible, episodes/actions/images unchanged)? (4) Can chunk_size and n_action_steps be changed when training the policy, since they don't change weight shapes?
**A** @ferry3563 (Oct 6): (1) No, training must be exclusively on the provided v3.0 dataset. (2) Augmentations are not allowed. (3) Paraphrasing instructions is not allowed. Overall: only the provided v3.0 dataset, with no modifications. (4) Correct, chunk_size and n_action_steps can be changed.

---

## Evaluation, leaderboard, scoring

**Q** @hand8710 (Sep 3): Is only a run of the uploaded weights used for the leaderboard? What is the budget limit for dataset generation and training if that check is done only for the top N at the end?
**A** @ferry3563 (Sep 3): Correct, only uploaded weights are run for the benchmark. Asked back: what volumes are you thinking of?
**Q** @hand8710 (Sep 5): Roughly, if processing takes an hour or several on the same machine; e.g. generating a dataset for 10–20 h on one H100 — is that OK?
**A** — no answer.

**Q** @hermann5422 (Sep 6): Will the finalist control run of the full pipeline also be subject to the 210-minute limit?
**A** @blick8624 (Sep 7): Running finalist weights takes about as long as the current run. Finalists must in any case provide the full solution, including all code and weights.

**Q** @hermann5422 (Sep 12): Does the 210-minute limit apply to the full pipeline (dataset generation + VLM fine-tuning + action expert training) or to individual stages? generate_dataset.py is not run now since annotations.jsonl is submitted ready-made.
**A** @ferry3563 (Sep 12): Dataset generation is not part of the run at this stage and is not executed. It will be run at the end of the competition to verify finalists' solutions.

**Q** @schoen1993 (Sep 22): How exactly is VLA computed (environment/suites, chunked mode, number of episodes)?
**A** @blick8624 (Sep 24): In the open self-check, VLA runs in the LIBERO simulator on `libero_spatial`: 10 tasks, by default one episode per task. Per-task success rate is computed, then averaged with equal weight. Episode count for your own checks is configurable. The template has `chunk_size: 50` and `n_action_steps: 50` — model parameters, not a description of official evaluation. Environment set, chunk execution mode and episode count of the official run are not disclosed. Use self-check to compare your own experiments, not as an exact leaderboard prediction.

**Q** @pacocha6731 (Sep 25): Is the VLA component counted in the current public leaderboard, and has any submission had nonzero VLA? Three different action experts (self-check libero_spatial 0.16–0.46) give exactly the same leaderboard score, differing only in the VLM part.
**Q** @hamill5759 (Sep 27, reply): Same picture; a diagnostic submission differing only in action_expert config paths gave a Score identical to 4 decimals, while local results are much higher. Please confirm whether VLA is counted in the current LB.
**Q** @hamill5759 (Sep 27): Do you confirm VLA Score is zero for all submissions, including baseline (9.90909 = 40·0.2477, pure VLM)? Is a fix planned before the end? If VLA is counted but the hidden bench gives SR≈0 to everyone, just confirm it is expected behavior.
**Q** @schoen1993 (Sep 29): Do you confirm VLA=0 for everyone? Or is it disabled for load reduction and will follow the rules in the private stage?
**A** @blick8624 (Sep 28, to hamill5759): We will look at the logs and report back.
**Announcement** @ferry3563 (Sep 30): A bug in the evaluation script meant metrics on VLA benchmarks were not counted (weight 0.6 in the final score). It is fixed; the corrected version applies to new submissions. Please re-submit to get current scores and positions. The daily submission limit is temporarily raised to 3 per day, until Oct 8, 2026 inclusive.
**Q** @hermann5422 (Sep 30): When will the counter update, or do we wait for the timer?
**A** — no answer.

**Q** @hamill5759 (Sep 27): How are the 3 final solutions chosen: automatically best by LB, or do I specify them? Will final submissions be re-evaluated in the control run (can the score change vs LB)?
**A** @blanda7513 (Sep 28): The participant selects 3 final solutions manually, but this is optional; otherwise the 3 with the best public score are chosen automatically. When the private leaderboard is formed, a private score is computed for the 3; the team's best private score of the 3 goes on the private leaderboard.

**Q** @hamill5759 (Sep 27): Can I get the run log of my submission (stage 12+, action_expert loading), at least for finalists after results?
**A** @blanda7513 (Sep 28): Logs are available only for failed submissions.

---

## Submission archive / ZIP

**Q** @hermann5422 (Sep 12): Is there a size limit on the uploaded .zip?
**A** @ferry3563 (Sep 12): Yes, 5 GB.
**Q** @hermann5422 (Sep 15, Sep 17): Is the same ≤5 GB ZIP used for finalist checks, or can extra files (open-model weights, dependencies missing from the environment) be supplied separately?
**A** @ferry3563 (Sep 21): The same 5 GB limit applies to finalists' solutions.

---

## Submissions failing / platform issues

**Q** @renner4544 (Sep 3, team renner45448572): Why do submissions fail? The last 4 failed at "12. Статус работы скрипта оценки решения"; hard to debug without logs.
**R** @hermann5422 (Sep 6): Same problem.
**A** — no answer.

**Q** @hand8710 (Sep 5): Anyone else failing at status 12? Identical solutions differing only in model weights have failed for 2 days; failed archives were marginally larger after compression — hidden archive size limit?
**R** @renner4544 (Sep 6): Exactly the same; even replacing safetensors with the baseline's fails.
**R** @hand8710 (Sep 6): The checking system seems broken — it fails even on what passed before.
**A** — no answer.

**Q** @hand8710 (Sep 8): Solution has been stuck at step 6 for several hours.
**A** — no answer.

**Q** @haley5191 (Sep 9): Everything fails again at stage 12, even a previously passing submission; lost a whole day. When is the fix expected?
**R** @haley5191 (Sep 9): It finally works.

**Q** @schoen1993 (Oct 1): Since Oct 1 02:11 MSK the script hangs at "02. Статус очистки дискового пространства".
**A** @blick8624 (Oct 1): Do not put datasets, images or other auxiliary data into the weights archive. The uploaded archive contained a dataset with many image files, which exceeded the host's inode limit on unpacking. Keep only weights, config and other files required for inference; exclude datasets, caches, logs and intermediate data.
**R** @schoen1993 (Oct 1): Noted, but you mixed me up with someone — I have no images, 19 files in 3 directories. My solution wasn't picked up and I couldn't submit another for 24 hours until the daily timer expired. Others in the Aug 29–31 threads reported this too. (Screenshots attached.)
**R** @schoen1993 (Oct 1): Inconvenient that one upload queue is shared across competitions: files scroll down, can't select, can't re-upload under the same name, so after the VLA fix files must be renamed/rebuilt; failed ones can't be deleted.

**Q** @bins6237 (Oct 2): Three runs failed at stage 12: Oct 1 20:39 `vla_base_001_submission.zip`, Oct 2 06:48 same file, Oct 2 07:22 `vla_base_001_submission_v2.zip`. A control archive `submission (1).zip` with the same structure passed Oct 1 13:11. The new archive has no images/datasets/caches/logs; 3.45 GB. Please provide the failed run's log.
**R** @bins6237 (Oct 4): Will there be an answer? I can't upload any solution.
**R** @schoen1993 (Oct 4): Take the organizers' config (paths are hard-coded there) and change only the model.

**Q** @konopelski2330 (Oct 5): Two submissions (Oct 4 16:39 and 21:14) failed at stage 12; one from Oct 4 03:18 with the same archive structure passed (10.0253). Differences: only action_expert weights (trained on full HuggingFaceVLA/libero, 30k steps instead of mini-fixture) and config paths/step counts. Weights checked: no NaN/Inf, tensor structure identical to the passing one, VLM weights identical. Looks like an evaluator issue after the fix; please check logs. Can send the archive.
**A** @blick8624 (Oct 5): Make sure all weights are local, the submission structure matches requirements, and the configs contain no references to downloading anything from HuggingFace.

---

## Webinar / materials

**Q** @dach5656 (Sep 26): Is there a recording of the Sep 25 webinar?
**Q** @schoen1993 (Sep 29): Where can I watch the webinar recording?
**Q** @conn4445 (Sep 30): Where can I watch the task walkthrough recording?
**A** — no replies in comments (the competition overview page links the recording: https://vkvideo.ru/video-212217448_456240309?list=ln-Vn1BQTU8zkTIqAQWQt).
