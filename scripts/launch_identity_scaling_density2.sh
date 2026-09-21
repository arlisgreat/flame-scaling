#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE="${1:-${ROOT}/artifacts/runs}"
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

for n_id in 8 384; do
  for method in hard adaptive released; do
    for seed in 0 1 2; do
      label="nid${n_id}_${method}_seed${seed}"
      output="${BASE}/identity_scaling20k_density2_v1/${label}"
      if [[ -s "${output}/metrics.json" && -s "${output}/test_rows.csv" ]]; then
        continue
      fi
      mkdir -p "${output}"
      CUDA_VISIBLE_DEVICES="${slot}" "${ROOT}/scripts/run_py310.sh" \
        "${ROOT}/scripts/run_identity_scaling.py" \
        --n-id "${n_id}" --method "${method}" --seed "${seed}" \
        --steps 20000 --density-multiplier 2 \
        --output "${output}" >"${output}/stdout.log" 2>&1 &
      pids+=("$!")
      labels+=("${label}")
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

if (( ${#failures[@]} > 0 )); then
  printf '%s\n' "${failures[@]}" >"${BASE}/identity_scaling20k_density2_v1/failed_runs.txt"
  echo "Density2 failures=${#failures[@]} labels=${failures[*]}" >&2
  exit 1
fi
rm -f "${BASE}/identity_scaling20k_density2_v1/failed_runs.txt"
echo "Completed density2 identity scaling controls"
