#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-${ROOT}/artifacts/runs/tongue017_frame182_heldout_release4000}"
FIT="${ROOT}/artifacts/cache/flame_fit_subject017_tongue_frame182_landmark_anchored.npz"
PCD="${ROOT}/artifacts/cache/subject017_tongue_frame182_hull.npz"
SUBJECT_ROOT="/data1/workspace/airulan/3dv/datasets/nersemble_v2/017"

mkdir -p "${OUTPUT_ROOT}"

names=(hard r010 r025 r050 r075 r100 r150 r200 free)
radii=(0 10 25 50 75 100 150 200 -1)
heldout=(222200039 222200049 222200040 222200046)

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
    --heldout-camera-ids "${heldout[@]}" \
    --radius-mm "${radii[$index]}" \
    --max-scale-mm 30 \
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

echo "Completed held-out mean-release ladder: ${OUTPUT_ROOT}"
