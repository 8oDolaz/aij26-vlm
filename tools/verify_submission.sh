#!/usr/bin/env bash
# Check a packed submission ZIP the way the judge would see it: in a fresh, throwaway container from the
# organizers' image (never the dev container), with no network, no HF cache, and the unzipped
# submission mounted read-only at the judge path. Run on the GPU server host, from the repo root.
#
# Usage:
#   tools/verify_submission.sh artifacts/<name>.zip [--libero] [--image aij-robotics:latest]
#     --libero   also roll out the policy in LIBERO sim (1 task x 1 episode; set TASK_IDS / N_EPISODES for more)
set -euo pipefail

JUDGE_ROOT=/home/ubuntu22/aij_robotics_judge/submission
IMAGE=aij-robotics:latest
LIBERO=0
ZIP=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --libero) LIBERO=1 ;;
    --image) IMAGE="$2"; shift ;;
    -h|--help) sed -n '2,9p' "$0"; exit 0 ;;
    *) ZIP="$1" ;;
  esac
  shift
done
[[ -f "${ZIP}" ]] || { echo "usage: $0 artifacts/<name>.zip [--libero]" >&2; exit 2; }
docker image inspect "${IMAGE}" >/dev/null 2>&1 \
  || { echo "image ${IMAGE} not found; build it: docker compose build dev" >&2; exit 2; }

NAME="$(basename "${ZIP}" .zip)"
WORK="$(dirname "$(realpath "${ZIP}")")/verify_${NAME}"
rm -rf "${WORK}"
mkdir -p "${WORK}/submission" "${WORK}/out"
trap 'rm -rf "${WORK}/submission"' EXIT

echo "[verify] unzipping ${ZIP} -> ${WORK}/submission"
python3 -m zipfile -e "${ZIP}" "${WORK}/submission"

STEPS="./scripts/validate_submission.sh ${JUDGE_ROOT}"
if [[ "${LIBERO}" -eq 1 ]]; then
  STEPS+=" && OUT_DIR=/out TASK_IDS='${TASK_IDS:-[0]}' N_EPISODES=${N_EPISODES:-1} ./scripts/run_libero_selfcheck.sh ${JUDGE_ROOT}"
fi

echo "[verify] image=${IMAGE} network=none submission=${JUDGE_ROOT} (read-only)"
docker run --rm --gpus all --network none --shm-size=8g \
  -e MUJOCO_GL=egl \
  -e NVIDIA_DRIVER_CAPABILITIES=all \
  -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
  -v "${WORK}/submission:${JUDGE_ROOT}:ro" \
  -v "${WORK}/out:/out" \
  "${IMAGE}" \
  bash -c "set -e; ${STEPS}"

echo "[verify] PASS: ${ZIP}"
[[ "${LIBERO}" -eq 1 ]] && echo "[verify] libero metrics: ${WORK}/out/eval_info.json"
exit 0
