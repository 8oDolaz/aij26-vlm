#!/usr/bin/env bash
# Pack the final submission ZIP (VLM + VLA + offline generator) on the GPU host, inside the organizers' image so the
# root-owned checkpoints are readable and torch/safetensors match the judge. Then verify it with
# tools/verify_submission.sh (printed at the end).
#
# Usage (from this repo's root on the host):
#   tools/pack_final.sh NAME VLM_DIR AE_DIR [extra pack_submission.py args]
#     VLM_DIR, AE_DIR: paths inside DATA_REPO (the training checkout), e.g.
#       runs/d4_scene_mix/submission/smolvlm2  runs/d4_scene_mix/submission/action_expert
# Environment (defaults: geoscan-5090):
#   DATA_REPO   training checkout with runs/, data/gen/, example/depth-anything-3   (~/Documents/lab449/aij26-vlm)
#   INPUTS      dir with sam2.1-hiera-small/ and pydeps/addict (+ addict-*.dist-info)  (../inputs next to this repo)
#   HF_HUB      HF cache hub dir with facebook/sam3 and depth-anything/DA3-BASE         (~/.cache/huggingface/hub)
#   ANNOTATIONS annotations.jsonl inside DATA_REPO                                       (data/gen/d4_mix/annotations.jsonl)
#   OUT         output dir for artifacts/<NAME>/ and <NAME>.zip                           (../artifacts next to this repo)
set -euo pipefail
[[ $# -ge 3 ]] || { sed -n '2,17p' "$0"; exit 2; }
NAME=$1 VLM=$2 AE=$3
shift 3

CODE="$(cd "$(dirname "$0")/.." && pwd)"
DATA_REPO="${DATA_REPO:-$HOME/Documents/lab449/aij26-vlm}"
INPUTS="${INPUTS:-$(dirname "$CODE")/inputs}"
HF_HUB="${HF_HUB:-$HOME/.cache/huggingface/hub}"
ANNOTATIONS="${ANNOTATIONS:-data/gen/d4_mix/annotations.jsonl}"
OUT="${OUT:-$(dirname "$CODE")/artifacts}"
IMAGE="${IMAGE:-aij-robotics:latest}"
mkdir -p "$OUT"

snap() { ls -d "$HF_HUB"/models--$1/snapshots/*/ | head -1 | sed "s|^$HF_HUB|/hf|"; }
SAM3=$(snap facebook--sam3)
DA3=$(snap depth-anything--DA3-BASE)

docker run --rm --network none \
  -v "$CODE:/code:ro" -v "$DATA_REPO:/repo:ro" -v "$INPUTS:/inputs:ro" -v "$HF_HUB:/hf:ro" -v "$OUT:/out" \
  -w /code "$IMAGE" bash -c "
    python3 tools/pack_submission.py --name '$NAME' --out-dir /out --force \
      --vlm '/repo/$VLM' --action-expert '/repo/$AE' --annotations '/repo/$ANNOTATIONS' \
      --configs configs/d4_scene_mix.yaml configs/vlm_sft/d3_depth_phase_detection.yaml configs/vla/a4_full_transforms.yaml \
      --sam3 '$SAM3' --da3 '$DA3' --sam2 /inputs/sam2.1-hiera-small \
      --da3-src /repo/example/depth-anything-3/src --pydeps /inputs/pydeps/addict $*
    rc=\$?
    chown -R $(id -u):$(id -g) /out
    exit \$rc"

echo "next: tools/verify_submission.sh $OUT/$NAME.zip --libero --generator $DATA_REPO/data/raw --det"
