#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RUN_ROOT="${1:?usage: launch_identity_scaling_head_roi_reeval.sh RUN_ROOT}"
GPU_LIST="${GPU_LIST:-0 1 2 3 4 5 6 7}"
read -r -a gpus <<<"${GPU_LIST}"
if (( ${#gpus[@]} == 0 )); then
  echo "GPU_LIST must contain at least one CUDA device index" >&2
  exit 2
fi
pids=()
labels=()
failures=()
slot=0

wait_wave() {
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      failures+=("${labels[$index]}")
    fi
  done
  pids=()
  labels=()
  slot=0
}

for run in "${RUN_ROOT}"/nid*_seed*; do
  [[ -d "${run}" ]] || continue
  label="$(basename "${run}")"
  if [[ -s "${run}/head_roi_metrics.json" && -s "${run}/head_roi_rows.csv" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${gpus[$slot]}" "${ROOT}/scripts/run_py310.sh" \
    "${ROOT}/scripts/reevaluate_identity_scaling_head_roi.py" \
    --run "${run}" \
    --audit-subjects 017 112 372 \
    >"${run}/head_roi_stdout.log" 2>&1 &
  pids+=("$!")
  labels+=("${label}")
  slot=$((slot + 1))
  if (( slot == ${#gpus[@]} )); then
    wait_wave
  fi
done

if (( slot > 0 )); then
  wait_wave
fi

if (( ${#failures[@]} > 0 )); then
  printf '%s\n' "${failures[@]}" >"${RUN_ROOT}/head_roi_failed_runs.txt"
  echo "Head ROI re-evaluation failures=${#failures[@]} labels=${failures[*]}" >&2
  exit 1
fi
rm -f "${RUN_ROOT}/head_roi_failed_runs.txt"
echo "Completed head ROI re-evaluation run_root=${RUN_ROOT}"
