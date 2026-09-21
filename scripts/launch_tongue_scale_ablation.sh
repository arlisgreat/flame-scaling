#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-${ROOT}/artifacts/runs/tongue017_frame182_scale_caps4000}"
FIT="${ROOT}/artifacts/cache/flame_fit_subject017_tongue_frame182_landmark_anchored.npz"
PCD="${ROOT}/artifacts/cache/subject017_tongue_frame182_hull.npz"
SUBJECT_ROOT="/data1/workspace/airulan/3dv/datasets/nersemble_v2/017"

mkdir -p "${OUTPUT_ROOT}"

names=(
  hard_cap003 hard_cap006 hard_cap012
  r050_cap003 r050_cap006 r050_cap012
  free_cap003 free_cap006 free_cap012
)
radii=(0 0 0 50 50 50 -1 -1 -1)
caps=(3 6 12 3 6 12 3 6 12)

run_one() {
  local index="$1"
  local gpu="$2"
  local name="${names[$index]}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${ROOT}/scripts/run_py310.sh" \
    "${ROOT}/scripts/run_geometry_oracle.py" \
    --fit "${FIT}" \
    --subject-root "${SUBJECT_ROOT}" \
    --sequence EXP-6-tongue-1 \
    --frame 182 \
    --pcd "${PCD}" \
    --height 128 \
    --background-threshold 30 \
    --radius-mm "${radii[$index]}" \
    --max-scale-mm "${caps[$index]}" \
    --steps 4000 \
    --seed 0 \
    --output "${OUTPUT_ROOT}/${name}" \
    >"${OUTPUT_ROOT}/${name}.log" 2>&1
}

pids=()
for index in {0..7}; do
  run_one "${index}" "${index}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

run_one 8 0

echo "Completed scale-cap ablation: ${OUTPUT_ROOT}"
