# Configs

Experiment configs. Training ones start from the kit defaults in `vendor/participant/configs/`.

- `gen_*.yaml`: generator configs (`submission/generate_dataset.py --config`)
- `d*.yaml`, `a1_*.yaml`, `selfcheck_*.yaml`, `smoke.yaml`: kit pipeline runs (`run_pipeline.sh <config>`)
- `vlm_sft/`: LLaMA-Factory SFT configs for SmolVLM2
- `vla/`: SmolVLA LIBERO configs

The submitted model used `d4_scene_mix.yaml` with `vlm_sft/d3_depth_phase_detection.yaml` and `vla/a4_full_transforms.yaml`.
