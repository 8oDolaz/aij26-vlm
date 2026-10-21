# AIJ 2026 Task 4 — Vision-Action Data Factory: submission

A generator that turns the raw robot episodes (Bridge, Fractal) into ~14k ShareGPT VQA records aimed at the skills the
VLM benchmarks test (spatial relations, relative depth, pointing, counting, existence, grasp phases). That data
fine-tunes SmolVLM2-500M, and the fine-tuned VLM initialises the SmolVLA policy trained on LIBERO v3.0.

## Contents

| Path | What |
|---|---|
| `generate_dataset.py`, `config.yaml`, `src/aijgen/` | the generator and its final config (one run reproduces the training data) |
| `annotations.jsonl` | the dataset the submitted VLM was trained on (14,114 records; image paths relative to the JSONL's dir) |
| `smolvlm2/` | fine-tuned SmolVLM2-500M-Video-Instruct (bf16) |
| `action_expert/` | SmolVLA policy trained on LIBERO v3.0 with this VLM |
| `weights/` | the generator's models, loaded from here offline (table below) |
| `src/third_party/` | Depth Anything 3 source and `addict`, the one dependency missing from the organizers' image |
| `configs/` | training configs: `d4_scene_mix.yaml` (pipeline run), `d3_depth_phase_detection.yaml` (VLM SFT), `a4_full_transforms.yaml` (VLA) |

## Data sources (config.yaml)

| Source | Questions | Models | Records |
|---|---|---|---|
| `scene_qa` | object relations (EmbSpatial/BLINK/MSMU wording), closest/farthest object, ERQA pointing, BLINK box localisation, existence, counting | SAM 3 (text-prompted detection over per-dataset object dictionaries) + DA3 (depth of each mask) | ≤ per-template caps, 10.7k |
| `depth_tier1` | BLINK-style "which point is closer" pairs and 3–4 point sets | DA3 (all views of a moment in one call) + SAM 2.1 (segments for point sampling) | 3,000 |
| `phase_qa` | grasp phase of the gripper (pre-grasp / contact / post-grasp) from the measured gripper state | none | 1,000 |

After capping, two-option records are trimmed so no answer letter exceeds 27% (`letter_balance`), and all records are
shuffled with `seed`. Everything is seeded: the same inputs and config give the same output.

## Reproduce

Environment: the organizers' image (`vendor/participant/Dockerfile`, CUDA 12.8, torch 2.10, transformers 5.3.0). Nothing
to install: `addict` and the DA3 code are vendored in `src/third_party/`, all model weights are in `weights/`, and no step
uses the network. `requirements.txt` lists the versions used.

1. **Dataset** (one GPU; about 3–4 h on an RTX 5090):
   ```bash
   python generate_dataset.py --input /data/raw --output /output/annotations.jsonl --config config.yaml
   ```
   `/data/raw` holds the organizers' `bridge/` and `fractal/` LeRobot datasets. Frames and rendered marker images are
   written next to the JSONL (`frames/`, `depth_tier1/frames/`); detections and depth are cached under the output dir,
   so an interrupted run resumes. `annotations.jsonl` here came from the same generators, settings and caps run in two
   development passes (depth + phase, then scene) and merged with the same caps, so a fresh run is equivalent, not byte-identical.
2. **VLM SFT** with the organizers' kit (`run_pipeline.sh`), full fine-tuning (vision tower included), 1 epoch, LR 1e-5, batch 1 × gradient accumulation 16 on one GPU:
   `configs/d3_depth_phase_detection.yaml` as `configs/vlm_sft.yaml`, data and media dir from `configs/d4_scene_mix.yaml`
   (point `data_path` / `media_dir` at the step 1 output).
3. **VLA** with the kit: `configs/a4_full_transforms.yaml` as `configs/vla_libero.yaml` (whole backbone trainable,
   LIBERO v3.0, 55k steps, batch 16), started from the step 2 VLM:
   ```bash
   # from the unzipped submission dir; the kit lives at /workspace in the organizers' image
   ln -sf "$PWD/configs/d3_depth_phase_detection.yaml" /workspace/configs/vlm_sft.yaml
   ln -sf "$PWD/configs/a4_full_transforms.yaml" /workspace/configs/vla_libero.yaml
   cd /workspace && ./scripts/run_pipeline.sh "$OLDPWD/configs/d4_scene_mix.yaml"   # steps 2 and 3 in one run
   ```

## Generator models (offline, in `weights/`)

| Path | Source (HF id @ revision) | Stored as | License | model.safetensors sha256 |
|---|---|---|---|---|
| `weights/sam3` | `facebook/sam3` @ 3c879f3 | bf16 (runs in bf16; conversion is lossless for inference) | SAM License (`weights/sam3/LICENSE`) | 8fa6c2a1f6f1ad3d10efc1521ce7f5eef0232514b15771638e783a806d1b959b |
| `weights/da3-base` | `depth-anything/DA3-BASE` @ f4a6c9b | fp32, as released | Apache-2.0 | e01067dc1659613083d9145a9a2547ccdbe6ccbbf83c4fe7b3e8a4e2bdae78b5 |
| `weights/sam2.1-hiera-small` | `facebook/sam2.1-hiera-small` @ ee5bba1 | fp32, as released | Apache-2.0 | 0a4067b11ce1e23d5229203f11c718a823060d15a4b23fa2372a7d4b77cbbc60 |

Code: Depth Anything 3 (`github.com/ByteDance-Seed/Depth-Anything-3` @ 3d835ec, Apache-2.0) and `addict` 2.4.0 (MIT);
licenses in `src/third_party/LICENSES/`. `config.yaml` refers to the models by these relative paths; in development
the same keys hold the HF ids.
