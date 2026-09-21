#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/nvs"

subjects=(388 422 443 445 475)
sequences=(GLASSES EXP-2-eyes FREE EXP-6-tongue-1 HAIR)
pids=()

for index in "${!subjects[@]}"; do
  subject="${subjects[$index]}"
  sequence="${sequences[$index]}"
  subject_root="${DATASET_ROOT}/${subject}"
  pcd="${subject_root}/sequences/${sequence}/pointclouds/frame_00000.pcd"
  output="${ROOT}/artifacts/cache/flame_fit_${subject}_landmark_anchored.npz"
  CUDA_VISIBLE_DEVICES="${index}" "${ROOT}/scripts/run_mediapipe_py310.sh" \
    "${ROOT}/scripts/fit_flame_multiview_landmarks.py" \
    --subject-root "${subject_root}" \
    --sequence "${sequence}" \
    --frame 0 \
    --pcd "${pcd}" \
    --height 512 \
    --output "${output}" \
    >"${ROOT}/artifacts/cache/flame_fit_${subject}_landmark_anchored.stdout.log" 2>&1 &
  pids+=("$!")
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  echo "landmark-anchored FLAME fitting failed_runs=${failures}" >&2
  exit 1
fi

echo "Completed landmark-anchored fits: ${#subjects[@]} cases"
