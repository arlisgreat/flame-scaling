#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/nvs"
OUTPUT_ROOT="${1:-${ROOT}/artifacts/runs/multicase_release_cv4000}"

subjects=(388 422 443 445 475)
sequences=(GLASSES EXP-2-eyes FREE EXP-6-tongue-1 HAIR)
condition_names=(hard r075 free)
radii=(0 75 -1)
fold_names=(fold_a fold_b fold_c)
fold_a=(221501007 222200043 222200041 222200036 222200042)
fold_b=(222200045 222200047 220700191 222200040)
fold_c=(222200049 222200038 222200048 222200044)

mkdir -p "${OUTPUT_ROOT}"
pids=()
slot=0
failures=0

run_one() {
  local subject_index="$1"
  local fold_index="$2"
  local condition_index="$3"
  local gpu="$4"
  local subject="${subjects[$subject_index]}"
  local sequence="${sequences[$subject_index]}"
  local fold_name="${fold_names[$fold_index]}"
  local heldout_name="${fold_name}[@]"
  local heldout=("${!heldout_name}")
  local condition="${condition_names[$condition_index]}"
  local output="${OUTPUT_ROOT}/${subject}/${fold_name}/${condition}"
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${ROOT}/scripts/run_py310.sh" \
    "${ROOT}/scripts/run_geometry_oracle.py" \
    --fit "${ROOT}/artifacts/cache/flame_fit_${subject}_landmark_anchored.npz" \
    --subject-root "${DATASET_ROOT}/${subject}" \
    --sequence "${sequence}" \
    --frame 0 \
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

wait_wave() {
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done
  pids=()
  slot=0
}

for subject_index in "${!subjects[@]}"; do
  for fold_index in "${!fold_names[@]}"; do
    for condition_index in "${!condition_names[@]}"; do
      run_one "${subject_index}" "${fold_index}" "${condition_index}" "${slot}" &
      pids+=("$!")
      slot=$((slot + 1))
      if (( slot == 8 )); then
        wait_wave
      fi
    done
  done
done
if (( slot > 0 )); then
  wait_wave
fi
if (( failures > 0 )); then
  echo "multi-case release cross-validation failed_runs=${failures}" >&2
  exit 1
fi

echo "Completed multi-case release cross-validation: ${OUTPUT_ROOT}"
