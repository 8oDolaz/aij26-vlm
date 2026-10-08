# Tools

Dev-only scripts. Nothing here ships in the ZIP, and `submission/` must not import from here.

- `pack_submission.py`: build the upload ZIP from a kit run (`runs/<run>/submission`) plus the code in `submission/`.
  Saves `smolvlm2` in bf16, rewrites `vlm_model_name` / `tokenizer_name` to the judge path, refuses images,
  remote refs, NaNs and tensor-name drift vs the demo, and writes `artifacts/<name>.zip` (≤5 GB).
  `--vlm` / `--action-expert` take either half from a different run.
- `verify_submission.sh`: run **on the GPU server host**. Checks a ZIP in a fresh `--network none` container from the
  untouched organizers' image, with the submission mounted read-only at the judge path. Never verify in the dev container.

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

docker compose run --rm dev python tools/fetch_data.py --dry-run         # plan only: metadata + selection.json
docker compose run --rm dev python tools/fetch_data.py --full fractal    # bridge, language_table subset; fractal whole
docker compose run --rm dev python tools/fetch_data.py --no-video        # proprioception only, ~0.75 GB
```
