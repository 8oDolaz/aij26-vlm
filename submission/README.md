# Submission

Contents of the ZIP. Built by `tools/pack_submission.py`:

- `generate_dataset.py`, `config.yaml`, `src/aijgen/`, `requirements.txt`: dataset generator (tracked)
- `smolvlm2/`, `action_expert/`, `annotations.jsonl`: added at pack time into `artifacts/<name>/`, never stored here

Run: `python generate_dataset.py --input <raw_root> --output <out_dir> --config config.yaml`
