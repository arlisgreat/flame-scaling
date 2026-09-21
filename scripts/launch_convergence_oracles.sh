#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/nvs"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/runs/geometry_converged_seed0}"
STEPS="${STEPS:-2000}"
HEIGHT="${HEIGHT:-128}"
POSITION_LR="${POSITION_LR:-0.0005}"

subjects=(388 422 443 445 475)
sequences=(GLASSES EXP-2-eyes FREE EXP-6-tongue-1 HAIR)
radii=(0 200 -1)
pids=()
failures=0
slot=0

wait_wave() {
  for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
      failures=$((failures + 1))
    fi
  done
  pids=()
  slot=0
}

for case_index in "${!subjects[@]}"; do
  subject="${subjects[$case_index]}"
  sequence="${sequences[$case_index]}"
  fit="$ROOT/artifacts/cache/flame_fit_${subject}.npz"
  if [[ "$subject" == "388" ]]; then
    fit="$ROOT/artifacts/cache/flame_fit_388_glasses.npz"
  fi
  for radius in "${radii[@]}"; do
    gpu=$((slot + 1))
    label="r${radius}mm"
    if [[ "$radius" == "-1" ]]; then
      label="free"
    fi
    output="$OUTPUT_ROOT/$subject/$label"
    mkdir -p "$output"
    CUDA_VISIBLE_DEVICES="$gpu" "$ROOT/scripts/run_py310.sh" \
      "$ROOT/scripts/run_geometry_oracle.py" \
      --fit "$fit" \
      --subject-root "$DATASET_ROOT/$subject" \
      --sequence "$sequence" \
      --frame 0 \
      --radius-mm "$radius" \
      --steps "$STEPS" \
      --height "$HEIGHT" \
      --position-lr "$POSITION_LR" \
      --seed 0 \
      --output "$output" \
      >"$output/stdout.log" 2>&1 &
    pid=$!
    pids+=("$pid")
    slot=$((slot + 1))
    echo "subject=$subject sequence=$sequence radius=$radius gpu=$gpu pid=$pid"
    if (( slot == 7 )); then
      wait_wave
    fi
  done
done

if (( slot > 0 )); then
  wait_wave
fi
if (( failures > 0 )); then
  echo "convergence oracle failed_runs=$failures" >&2
  exit 1
fi
echo "convergence oracle completed runs=15"

