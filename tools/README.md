# Tools

Dev-only scripts. Nothing here ships in the ZIP, and `submission/` never imports from here. Each script's docstring
has its full options and examples.

**Data**
- `fetch_data.py`: download Bridge, Fractal and Language Table at the organizers' pinned commits. Always fetches
  `meta/` and `data/`; video either whole (`--full`) or a deterministic subset within `--budget-gb`. `--no-video` fetches
  state and instructions only.
- `build_sft_mix.py`: merge generator outputs into one self-contained SFT mix (how the submitted `d4_mix` was built).
- `run_d4_chain.sh`: the d4 experiment chain (scene data, mix, VLM SFT + VLA, self-checks), unattended.

**Packing and checks**
- `pack_submission.py`: build the upload ZIP from a kit run plus `submission/`. Saves `smolvlm2` in bf16, rewrites the
  judge paths, refuses images, remote refs, NaNs and tensor-name drift vs the demo. With `--sam3/--da3/--sam2/--da3-src/--pydeps`
  it also ships the generator's weights and code so it runs offline.
- `pack_final.sh`: the final pack in one command on the GPU host, inside the organizers' image.
- `verify_submission.sh`: run on the GPU host. Checks a ZIP in a fresh `--network none` container from the untouched
  organizers' image. `--libero` rolls out the policy; `--generator RAW_DIR [--det]` runs the generator from the ZIP
  (twice with `--det`, comparing byte for byte).
- `merge_vla_lora.py`, `vla_peft_shim/`: LoRA for the VLA run (the shim adds PEFT to the kit's `lerobot-train`; the
  merge folds the adapter back into a plain SmolVLA policy).

**Inspection**
- `scene_qa_sheet.py`: contact sheets of the SAM 3 scene questions per template, for hand checks.
- `sam3_probe.py`: SAM 3 detection probe over instruction, dictionary and negative-control vocabularies.
- `inspect_annotations.py`: step through a generated dataset (image, question, answer).
- `explore_episode.py`, `rerun_episode.py`: one episode's gripper signal, phases and frames (plots, or live in rerun).

```bash
python tools/fetch_data.py --dry-run          # plan only: metadata + selection.json
python tools/pack_submission.py --name a0 --from runs/a0/submission --annotations /data/gen/v0/annotations.jsonl
tools/verify_submission.sh artifacts/a0.zip --libero --generator /path/to/raw --det
```
