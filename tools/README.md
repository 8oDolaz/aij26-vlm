# Tools

Dev-only scripts. Nothing here ships in the ZIP, and `submission/` must not import from here.

- `pack_submission.py`: build the upload ZIP from a kit run (`runs/<run>/submission`) plus the code in `submission/`.
  Saves `smolvlm2` in bf16, rewrites `vlm_model_name` / `tokenizer_name` to the judge path, refuses images,
  remote refs, NaNs and tensor-name drift vs the demo, and writes `artifacts/<name>.zip` (≤5 GB).
  `--vlm` / `--action-expert` take either half from a different run.
  With `--sam3/--da3/--sam2/--da3-src/--pydeps` it also ships the generator for the offline finalist run: weights to
  `weights/` (SAM 3 in bf16), the DA3 source and `addict` to `src/third_party/`, and rewrites `config.yaml`'s model
  refs to those paths (refuses any ref still pointing at the Hub). `--vlm-fill DIR` takes smolvlm2 files missing from
  the run (e.g. `tokenizer.json`) from a LLaMA-Factory `checkpoint-<step>/`.
- `pack_final.sh`: the final pack in one command on the GPU host (geoscan-5090 defaults): runs `pack_submission.py` in
  the organizers' image with all generator inputs. `tools/pack_final.sh NAME <vlm dir> <action_expert dir>`.
- `verify_submission.sh`: run **on the GPU server host**. Checks a ZIP in a fresh `--network none` container from the
  untouched organizers' image, with the submission mounted read-only at the judge path. Never verify in the dev container.
  `--generator RAW_DIR [--episodes N] [--det]` also runs `generate_dataset.py` from the ZIP on N episodes per dataset
  (no HF cache, no offline env vars: any hub lookup fails), requires records from every source, and with `--det`
  compares two runs byte for byte.

- `scene_qa_sheet.py`: hand check of the SAM 3 scene questions (`aijgen.scene`, docs/sam3_vqa_plan.md). Runs
  `scene_qa` on `--episodes` per dataset (GPU box; SAM 3 + DA3, cached per frame) and writes one contact sheet per
  template plus `review.jsonl` to report wrong labels by id.
- `sam3_probe.py`: SAM 3 detection probe (instruction / closed-dictionary / negative-control vocabularies); shares the
  dictionaries and box filter with `scene_qa` (`submission/src/aijgen/scene/`).

- `fetch_data.py`: download Bridge, Fractal and Language Table at the organizers' pinned commits into `/data/raw/<name>/`.
  Always fetches `meta/` and `data/` (~0.75 GB in all); video is either complete (`--full`) or a deterministic
  subset of files picked from that metadata (usable episodes, new instructions, multi-view for Bridge) within
  `--budget-gb`. Writes `selection.json` with the chosen files and episode ids for `generate_dataset.py`.
  Defaults: bridge 20 GB (half reserved for multi-view), fractal 1.1 GB, language_table 0.5 GB.
  `--no-video [NAME ...]` fetches only `meta/` + `data/` (state, action, instructions) and lists every usable episode.
  `--phases` / `--min-phase-frames`: grasp phases a Bridge/Fractal episode must go through (Robo2VLM-style segmentation
  of the measured gripper, `submission/src/aijgen/phases.py`); default `pre_grasp contact post_grasp`, 2 frames each.

```bash
python tools/pack_submission.py --name a0 --from runs/a0/submission --annotations data/gen/v0/annotations.jsonl
tools/verify_submission.sh artifacts/a0.zip --libero

# final submission on geoscan-5090 (inputs gathered in ~/Documents/lab449/pack_final/inputs, see pack_final.sh)
tools/pack_final.sh final_d4_a4 runs/d4_scene_mix/submission/smolvlm2 runs/d4_scene_mix/submission/action_expert
TASK_IDS='[0,1,2,3,4,5,6,7,8,9]' tools/verify_submission.sh ../artifacts/final_d4_a4.zip --libero \
    --generator ~/Documents/lab449/aij26-vlm/data/raw --det

docker compose run --rm dev python tools/fetch_data.py --dry-run         # plan only: metadata + selection.json
docker compose run --rm dev python tools/fetch_data.py --full fractal    # bridge, language_table subset; fractal whole
docker compose run --rm dev python tools/fetch_data.py --no-video        # proprioception only, ~0.75 GB
```
