# Tools

Dev-only scripts. Nothing here ships in the ZIP, and `submission/` must not import from here.

- `pack_submission.py`: build the upload ZIP from a kit run (`runs/<run>/submission`) plus the code in `submission/`.
  Saves `smolvlm2` in bf16, rewrites `vlm_model_name` / `tokenizer_name` to the judge path, refuses images,
  remote refs, NaNs and tensor-name drift vs the demo, and writes `artifacts/<name>.zip` (≤5 GB).
  `--vlm` / `--action-expert` take either half from a different run.
- `verify_submission.sh`: run **on the GPU server host**. Checks a ZIP in a fresh `--network none` container from the
  untouched organizers' image, with the submission mounted read-only at the judge path. Never verify in the dev container.

```bash
python tools/pack_submission.py --name a0 --from runs/a0/submission --annotations data/gen/v0/annotations.jsonl
tools/verify_submission.sh artifacts/a0.zip --libero
```

Planned:
- `fetch_data.py`: download `meta/` plus a deterministic subset of chunks into `data/raw/`
