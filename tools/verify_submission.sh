#!/usr/bin/env bash
# Check a packed submission ZIP the way the judge would see it: in a fresh, throwaway container from the
# organizers' image (never the dev container), with no network, no HF cache, and the unzipped
# submission mounted read-only at the judge path. Run on the GPU server host, from the repo root.
#
# Usage:
#   tools/verify_submission.sh artifacts/<name>.zip [--libero] [--generator RAW_DIR [--episodes N] [--det]]
#                              [--image aij-robotics:latest]
#     --libero          also roll out the policy in LIBERO sim (1 task x 1 episode; set TASK_IDS / N_EPISODES for more)
#     --generator DIR   also run generate_dataset.py from the ZIP on the raw datasets in DIR (bridge/, fractal/, mounted
#                       read-only at /data/raw), with the shipped config.yaml capped to N episodes per dataset (default 3).
#                       No HF_HUB_OFFLINE here on purpose: with --network none any hub lookup fails, so a pass proves
#                       every model loads from the ZIP. Every source (depth_tier1, phase_qa, scene_qa/*) must give records.
#     --det             run the generator twice and require byte-identical annotations.jsonl
set -euo pipefail

JUDGE_ROOT=/home/ubuntu22/aij_robotics_judge/submission
IMAGE=aij-robotics:latest
LIBERO=0
RAW=""
EPISODES=3
DET=0
ZIP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --libero) LIBERO=1 ;;
    --image) IMAGE="$2"; shift ;;
    --generator) RAW="$(realpath "$2")"; shift ;;
    --episodes) EPISODES="$2"; shift ;;
    --det) DET=1 ;;
    -h|--help) sed -n '2,17p' "$0"; exit 0 ;;
    *) ZIP="$1" ;;
  esac
  shift
done
[[ -f "${ZIP}" ]] || { echo "usage: $0 artifacts/<name>.zip [--libero] [--generator RAW_DIR]" >&2; exit 2; }
[[ -z "${RAW}" || -d "${RAW}" ]] || { echo "not a directory: ${RAW}" >&2; exit 2; }
docker image inspect "${IMAGE}" >/dev/null 2>&1 \
  || { echo "image ${IMAGE} not found; build it: docker compose build dev" >&2; exit 2; }

NAME="$(basename "${ZIP}" .zip)"
WORK="$(dirname "$(realpath "${ZIP}")")/verify_${NAME}"
rm -rf "${WORK}" 2>/dev/null \
  || docker run --rm -v "$(dirname "${WORK}"):/w" "${IMAGE}" rm -rf "/w/$(basename "${WORK}")"  # root-owned outputs
mkdir -p "${WORK}/submission" "${WORK}/out"
trap 'rm -rf "${WORK}/submission"' EXIT

echo "[verify] unzipping ${ZIP} -> ${WORK}/submission"
python3 -m zipfile -e "${ZIP}" "${WORK}/submission"

STEPS="./scripts/validate_submission.sh ${JUDGE_ROOT}"
if [[ "${LIBERO}" -eq 1 ]]; then
  STEPS+=" && OUT_DIR=/out TASK_IDS='${TASK_IDS:-[0]}' N_EPISODES=${N_EPISODES:-1} ./scripts/run_libero_selfcheck.sh ${JUDGE_ROOT}"
fi

# The LIBERO simulator's scene assets are not in the image (hf-libero downloads jadechoghari/libero-assets to
# ~/.cache/libero/assets on first use); with no network, mount a copy read-only at the package's local assets path,
# which hf-libero checks first. Copy one out of the dev container once:
#   docker cp aij-dev:/root/.cache/libero/assets <dir>
# They belong to the evaluator, not the submission.
ASSETS="${LIBERO_ASSETS:-$(dirname "$(realpath "$0")")/../../inputs/libero_assets}"
ASSET_MOUNT=()
if [[ "${LIBERO}" -eq 1 ]]; then
  [[ -f "${ASSETS}/scenes/libero_tabletop_base_style.xml" ]] \
    || { echo "LIBERO assets not found at ${ASSETS}; set LIBERO_ASSETS (see the comment above)" >&2; exit 2; }
  ASSET_MOUNT=(-v "$(realpath "${ASSETS}"):/usr/local/lib/python3.12/dist-packages/libero/libero/assets:ro")
fi

echo "[verify] image=${IMAGE} network=none submission=${JUDGE_ROOT} (read-only)"
docker run --rm --gpus all --network none --shm-size=8g \
  -e MUJOCO_GL=egl \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  "${ASSET_MOUNT[@]}" \
  -v "${WORK}/submission:${JUDGE_ROOT}:ro" \
  -v "${WORK}/out:/out" \
  "${IMAGE}" \
  bash -c "set -e; ${STEPS}"
echo "[verify] weights: PASS"
[[ "${LIBERO}" -eq 1 ]] && echo "[verify] libero metrics: ${WORK}/out/eval_info.json"

if [[ -n "${RAW}" ]]; then
  RUNS=1
  [[ "${DET}" -eq 1 ]] && RUNS=2
  echo "[verify] generator: ${EPISODES} episodes per dataset from ${RAW}, ${RUNS} run(s), network=none, no HF cache"
  docker run --rm --gpus all --network none --shm-size=8g \
    -v "${WORK}/submission:${JUDGE_ROOT}:ro" \
    -v "${RAW}:/data/raw:ro" \
    -v "${WORK}/out:/out" \
    -w "${JUDGE_ROOT}" \
    "${IMAGE}" \
    bash -c "set -e
python3 - <<'EOF'
import yaml
cfg = yaml.safe_load(open('config.yaml'))
for ds in cfg.get('depth_tier1', {}).get('datasets', {}).values():
    ds['max_episodes'] = ${EPISODES}
for ds in cfg.get('episodes', {}).get('datasets', {}).values():
    ds['max_episodes'] = ${EPISODES}
yaml.safe_dump(cfg, open('/out/smoke_config.yaml', 'w'), sort_keys=False)
EOF
for i in \$(seq 1 ${RUNS}); do
  rm -rf /out/gen\$i
  python3 generate_dataset.py --input /data/raw --output /out/gen\$i/annotations.jsonl --config /out/smoke_config.yaml \
    > /out/gen\$i.log 2>&1 || { tail -40 /out/gen\$i.log; exit 1; }
done
python3 - <<'EOF'
import json, hashlib, sys
from pathlib import Path
cfg_sources = ['depth_tier1', 'phase_qa', 'scene_qa']
errors = []
out = Path('/out/gen1')
stats = json.loads((out / 'annotations.stats.json').read_text())
kept = {k: v['kept'] for k, v in stats['mix'].items()}
print('records per source:', json.dumps(kept))
for s in cfg_sources:
    n = sum(v for k, v in kept.items() if k == s or k.startswith(s + '/'))
    if n == 0:
        errors.append(f'no {s} records')
recs = [json.loads(l) for l in (out / 'annotations.jsonl').read_text().splitlines() if l.strip()]
missing = [p for r in recs for p in r['images'] if not (out / p).is_file()]
if missing:
    errors.append(f'{len(missing)} referenced images missing, e.g. {missing[:3]}')
print(f'{len(recs)} records, all images present: {not missing}')
if Path('/out/gen2').exists():
    h = [hashlib.sha256(Path(f'/out/gen{i}/annotations.jsonl').read_bytes()).hexdigest() for i in (1, 2)]
    print('determinism:', 'identical' if h[0] == h[1] else 'DIFFERENT', h[0][:16], h[1][:16])
    if h[0] != h[1]:
        errors.append('two runs gave different annotations.jsonl')
for e in errors:
    print('ERROR:', e)
sys.exit(1 if errors else 0)
EOF"
  echo "[verify] generator: PASS (outputs and logs in ${WORK}/out/gen*)"
fi

echo "[verify] PASS: ${ZIP}"
exit 0
