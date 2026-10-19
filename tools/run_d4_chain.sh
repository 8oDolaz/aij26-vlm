#!/usr/bin/env bash
# d4 experiment chain, unattended. Run inside the dev container (it is started in tmux session "d4" on the host):
#   docker exec aij-dev bash /repo/tools/run_d4_chain.sh
#   1. scene questions        configs/gen_scene14k.yaml            -> /data/gen/scene14k
#   2. SFT mix                tools/build_sft_mix.py               -> /data/gen/d4_mix
#   3. VLM SFT + VLA (full)   configs/d4_scene_mix.yaml            -> /repo/runs/d4_scene_mix
#   4. self-check             configs/selfcheck_score_5ep.yaml     -> /repo/runs/d4_scene_mix/score_selfcheck
#   5. VLA with LoRA on VLM   configs/d4_scene_mix_lora.yaml       -> /repo/runs/d4_scene_mix_lora (merged afterwards)
#   6. self-check (LIBERO only; the VLM is d4's, so its VLM score is reused)
# A step whose output exists is skipped, so the script can be re-run after a failure (remove the failed step's
# half-written output first). Logs: /repo/runs/d4_logs/<step>.log; progress: /repo/runs/d4_logs/chain.log.
set -euo pipefail

LOGS=/repo/runs/d4_logs
mkdir -p "$LOGS"
export PYTORCH_ALLOC_CONF=expandable_segments:True

say() { echo "[$(date '+%F %T')] $*" | tee -a "$LOGS/chain.log"; }
disk() { df -h / | awk 'NR==2 {print $4 " free"}'; }
need_space() {  # $1 = GB needed before a step
  local free; free=$(df -BG --output=avail / | tail -1 | tr -dc 0-9)
  if [ "$free" -lt "$1" ]; then say "STOP: only ${free} GB free, step needs ${1} GB"; exit 1; fi
}
step() {  # step <name> <done-marker> <command...>
  local name=$1 marker=$2; shift 2
  if [ -e "$marker" ]; then say "skip $name ($marker exists)"; return; fi
  say "start $name ($(disk))"
  if "$@" > "$LOGS/$name.log" 2>&1; then
    say "done  $name ($(disk))"
  else
    say "FAIL  $name, see $LOGS/$name.log"; exit 1
  fi
}

gen() {
  cd /repo && python submission/generate_dataset.py --input /data/raw \
    --output /data/gen/scene14k/annotations.jsonl --config configs/gen_scene14k.yaml
}
mix() {
  python /repo/tools/build_sft_mix.py --scene /data/gen/scene14k/annotations.jsonl \
    --old /data/gen/d2_depth_phase/annotations.jsonl --depth 3000 --phase 1000 \
    --media-root /data/gen --out /data/gen/d4_mix/annotations.jsonl
}
train_full() {
  ln -sf /repo/configs/vlm_sft/d3_depth_phase_detection.yaml /workspace/configs/vlm_sft.yaml
  ln -sf /repo/configs/vla/a1_full_unfreeze.yaml /workspace/configs/vla_libero.yaml
  cd /workspace && FORCE_TORCHRUN=1 ./scripts/run_pipeline.sh /repo/configs/d4_scene_mix.yaml
}
selfcheck_full() {
  cd /workspace && ./scripts/run_score_selfcheck.sh --submission /repo/runs/d4_scene_mix/submission \
    --config /repo/configs/selfcheck_score_5ep.yaml
}
train_lora() {
  ln -sf /repo/configs/vla/a3_lora_vlm.yaml /workspace/configs/vla_libero.yaml
  cd /workspace && PATH=/repo/tools/vla_peft_shim:$PATH VLA_PEFT_CONFIG=/repo/configs/vla/a3_lora_vlm.yaml \
    ./scripts/run_pipeline.sh /repo/configs/d4_scene_mix_lora.yaml --vlm-checkpoint /repo/runs/d4_scene_mix/vlm/checkpoints
  python /repo/tools/merge_vla_lora.py /repo/runs/d4_scene_mix_lora/submission/action_expert \
    --keep-adapter /repo/runs/d4_scene_mix_lora/action_expert_adapter \
    --reference /repo/runs/d4_scene_mix/submission/action_expert
}
selfcheck_lora() {
  cd /workspace && ./scripts/run_score_selfcheck.sh --submission /repo/runs/d4_scene_mix_lora/submission \
    --config /repo/configs/selfcheck_score_5ep.yaml --skip-vlm
  # same VLM as d4_scene_mix: Score = 100 * (0.4 * VLM(d4) + 0.6 * VLA(lora))
  python3 - <<'EOF'
import json
full = json.load(open("/repo/runs/d4_scene_mix/score_selfcheck/summary/leaderboard_score.json"))
lora = json.load(open("/repo/runs/d4_scene_mix_lora/score_selfcheck/summary/leaderboard_score.json"))
part = lambda d: d["score"] if isinstance(d.get("score"), dict) else d  # report vs leaderboard_score.json layout
vlm, vla = part(full)["vlm"], part(lora)["vla"]
out = {"vlm_from": "d4_scene_mix", "vlm": vlm, "vla": vla, "score": 100 * (0.4 * vlm + 0.6 * vla)}
json.dump(out, open("/repo/runs/d4_scene_mix_lora/score_selfcheck/summary/combined_score.json", "w"), indent=1)
print(out)
EOF
}

say "chain start ($(disk))"
need_space 30
step gen_scene14k   /data/gen/scene14k/annotations.jsonl                                   gen
step build_mix      /data/gen/d4_mix/annotations.jsonl                                     mix
need_space 40
step train_full     /repo/runs/d4_scene_mix/submission/action_expert                       train_full
step selfcheck_full /repo/runs/d4_scene_mix/score_selfcheck/summary/leaderboard_score.json selfcheck_full
need_space 30
step train_lora     /repo/runs/d4_scene_mix_lora/action_expert_adapter                     train_lora
step selfcheck_lora /repo/runs/d4_scene_mix_lora/score_selfcheck/summary/combined_score.json selfcheck_lora
say "chain finished ($(disk))"
