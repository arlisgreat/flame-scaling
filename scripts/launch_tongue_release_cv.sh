#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_ROOT="${1:-${ROOT}/artifacts/runs/tongue017_frame182_release_cv4000}"
FIT="${ROOT}/artifacts/cache/flame_fit_subject017_tongue_frame182_landmark_anchored.npz"
PCD="${ROOT}/artifacts/cache/subject017_tongue_frame182_hull.npz"
SUBJECT_ROOT="/data1/workspace/airulan/3dv/datasets/nersemble_v2/017"

mkdir -p "${OUTPUT_ROOT}"

names=(hard r010 r025 r050 r075 r100 r150 r200 free)
radii=(0 10 25 50 75 100 150 200 -1)
fold_names=(fold_a fold_b fold_c fold_d)
fold_a=(222200039 222200049 222200040 222200046)
fold_b=(222200043 222200047 222200048 222200036)
fold_c=(222200038 222200037 222200041 220700191)
fold_d=(222200045 221501007 222200044 222200042)

run_one() {
  local fold_index="$1"
  local condition_index="$2"
  local gpu="$3"
  local fold_name="${fold_names[$fold_index]}"
  local heldout_name="${fold_name}[@]"
  local heldout=("${!heldout_name}")
  local name="${names[$condition_index]}"
  local output="${OUTPUT_ROOT}/${fold_name}/${name}"
  mkdir -p "${output}"
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
    --radius-mm "${radii[$condition_index]}" \
    --max-scale-mm 30 \
    --steps 4000 \
    --seed 0 \
    --output "${output}" \
    >"${output}/stdout.log" 2>&1
}

pids=()
slot=0
failures=0
wait_wave() {
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done
  pids=()
  slot=0
}

for fold_index in "${!fold_names[@]}"; do
  for condition_index in "${!names[@]}"; do
    run_one "${fold_index}" "${condition_index}" "${slot}" &
    pids+=("$!")
    slot=$((slot + 1))
    if (( slot == 8 )); then
      wait_wave
    fi
  done
done
if (( slot > 0 )); then
  wait_wave
fi
if (( failures > 0 )); then
  echo "release cross-validation failed_runs=${failures}" >&2
  exit 1
fi

echo "Completed four-fold release cross-validation: ${OUTPUT_ROOT}"
