#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/nvs"
OUTPUT_ROOT="${1:-${ROOT}/artifacts/runs/multicase_landmark_mean_extent4000}"

subjects=(388 422 443 445 475)
sequences=(GLASSES EXP-2-eyes FREE EXP-6-tongue-1 HAIR)
condition_names=(hard_cap003 hard_cap030 free_cap003 free_cap030)
radii=(0 0 -1 -1)
caps=(3 30 3 30)

mkdir -p "${OUTPUT_ROOT}"
pids=()
failures=0
slot=0

wait_wave() {
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done
  pids=()
  slot=0
}

for case_index in "${!subjects[@]}"; do
  subject="${subjects[$case_index]}"
  sequence="${sequences[$case_index]}"
  subject_root="${DATASET_ROOT}/${subject}"
  fit="${ROOT}/artifacts/cache/flame_fit_${subject}_landmark_anchored.npz"
  for condition_index in "${!condition_names[@]}"; do
    name="${condition_names[$condition_index]}"
    output="${OUTPUT_ROOT}/${subject}/${name}"
    mkdir -p "${output}"
    CUDA_VISIBLE_DEVICES="${slot}" "${ROOT}/scripts/run_py310.sh" \
      "${ROOT}/scripts/run_geometry_oracle.py" \
      --fit "${fit}" \
      --subject-root "${subject_root}" \
      --sequence "${sequence}" \
      --frame 0 \
      --height 128 \
      --background-threshold 30 \
      --radius-mm "${radii[$condition_index]}" \
      --max-scale-mm "${caps[$condition_index]}" \
      --steps 4000 \
      --seed 0 \
      --output "${output}" \
      >"${output}/stdout.log" 2>&1 &
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
  echo "mean-extent factorial failed_runs=${failures}" >&2
  exit 1
fi

echo "Completed multi-case mean-extent factorial: ${OUTPUT_ROOT}"
