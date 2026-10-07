# Tools

Dev-only scripts. Nothing here ships in the ZIP, and `submission/` must not import from here.

Planned:
- `pack_submission.py`: copy weights into `submission/`, save `smolvlm2` in bf16, rewrite judge paths
  (`vlm_model_name`, `tokenizer_name`), build the ZIP into `artifacts/`, and assert it is ≤5 GB with no images and no remote refs
- `fetch_data.py`: download `meta/` plus a deterministic subset of chunks into `data/raw/`
