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

launch() {
  local family="$1"
  local label="$2"
  shift 2
  local output="${BASE}/${family}/${label}"
  if [[ -s "${output}/metrics.json" && -s "${output}/test_rows.csv" ]]; then
    return
  fi
  mkdir -p "${output}"
  CUDA_VISIBLE_DEVICES="${slot}" "${ROOT}/scripts/run_py310.sh" \
    "${ROOT}/scripts/run_identity_scaling.py" \
    "$@" \
    --output "${output}" \
    >"${output}/stdout.log" 2>&1 &
  pids+=("$!")
  labels+=("${family}/${label}")
  slot=$((slot + 1))
  if (( slot == 8 )); then
    wait_wave
  fi
}

# Equal architecture at 3 mm covariance: pre-registered endpoint sensitivity.
for n_id in 8 384; do
  for method in hard adaptive released; do
    for seed in 0 1 2; do
      launch identity_scaling20k_scale3_v1 "nid${n_id}_${method}_seed${seed}" \
        --n-id "${n_id}" --method "${method}" --seed "${seed}" \
        --steps 20000 --fixed-scale-mm 3
    done
  done
done

# Decoder capacity endpoints around the 651,816-parameter primary model.
for hidden in 64 512; do
  family="identity_scaling20k_h${hidden}_v1"
  for n_id in 8 384; do
    for method in hard adaptive released; do
      for seed in 0 1 2; do
        launch "${family}" "nid${n_id}_${method}_seed${seed}" \
          --n-id "${n_id}" --method "${method}" --seed "${seed}" \
          --steps 20000 --hidden-dim "${hidden}"
      done
    done
  done
done

# Release-dose controls for APR; hard and released150 anchors come from primary.
for radius in 30 75; do
  family="identity_scaling20k_adaptive_r${radius}_v1"
  for n_id in 8 384; do
    for seed in 0 1 2; do
      launch "${family}" "nid${n_id}_adaptive_seed${seed}" \
        --n-id "${n_id}" --method adaptive --seed "${seed}" \
        --steps 20000 --adaptive-radius-mm "${radius}"
    done
  done
done

# High-data converged-budget control.
for method in hard adaptive released; do
  for seed in 0 1 2; do
    launch identity_scaling60k_v1 "nid384_${method}_seed${seed}" \
      --n-id 384 --method "${method}" --seed "${seed}" --steps 60000
  done
done

if (( slot > 0 )); then
  wait_wave
fi

if (( ${#failures[@]} > 0 )); then
  printf '%s\n' "${failures[@]}" >"${BASE}/identity_scaling_followup_failed_runs.txt"
  echo "Identity follow-up failures=${#failures[@]} labels=${failures[*]}" >&2
  exit 1
fi
rm -f "${BASE}/identity_scaling_followup_failed_runs.txt"
echo "Completed identity scaling follow-up controls base=${BASE}"
