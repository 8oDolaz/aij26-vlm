# AIJ 2026 Task 4: Vision-Action Data Factory

`generate_dataset.py` turns Bridge / Fractal / Language Table / Egocentric-100K into a ShareGPT JSONL. That data is used to SFT SmolVLM2-500M, which then initializes a SmolVLA policy trained on LIBERO.
See [docs/PLAN.md](docs/PLAN.md) for the strategy and [docs/experiments.md](docs/experiments.md) for results.

## Layout

| Path | What | In git |
|---|---|---|
| `submission/` | **The deliverable**: exactly the code that goes into the ZIP. Must be self-contained (no imports from `tools/` or `vendor/`). | code yes, weights no |
| `configs/` | Our SFT / VLA experiment configs (overrides of `vendor/participant/configs/`) | yes |
| `tools/` | Dev-only scripts (packing, data fetch, eval reports). Never shipped. | yes |
| `tests/` | Determinism and packaging checks | yes |
| `docs/` | Plan, experiment log, research notes, organizer rules and Q&A | yes |
| `vendor/participant/` | Organizer training/eval kit (LLaMA-Factory, scaling-lawyer, lmms-eval overlay) | yes, committed as-is first; patches show as diffs |
| `vendor/baseline/` | Organizer baseline data pipeline (MIT), for reference and copying | yes |
| `vendor/demo_submission/` | Demo submission configs + weight manifest (no weights) | yes |
| `data/` | Raw dataset subsets, generated JSONL and frames | no |
| `runs/` | SFT / VLA checkpoints, eval logs | no |
| `artifacts/` | Demo weights (`artifacts/demo_submission/`), built ZIPs | no |

Weights never go into git. Move checkpoints between machines (5090s and Colab) via a private HF repo.
Demo weights can be verified with `cd artifacts/demo_submission && sha256sum -c ../../vendor/demo_submission/weights.sha256`.

## Workflow

1. Generate data: `python submission/generate_dataset.py --input data/raw --output data/gen/<name> --config submission/config.yaml`
2. Train and evaluate with the kit in `vendor/participant/` (see its README), using a config from `configs/`; outputs go to `runs/<id>/`
3. Log the run in `docs/experiments.md`
4. Pack: `tools/pack_submission.py` (planned) fills `submission/{smolvlm2,action_expert,annotations.jsonl}`, rewrites judge paths, and builds the ZIP into `artifacts/`
