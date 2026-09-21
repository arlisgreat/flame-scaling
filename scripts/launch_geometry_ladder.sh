#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIT="${FIT:-$ROOT/artifacts/cache/flame_fit_388_glasses.npz}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT/artifacts/runs/geometry_ladder_seed0}"
STEPS="${STEPS:-600}"
HEIGHT="${HEIGHT:-128}"
POSITION_LR="${POSITION_LR:-0.0005}"
SUBJECT_ROOT="${SUBJECT_ROOT:-/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/nvs/388}"
SEQUENCE="${SEQUENCE:-GLASSES}"
FRAME="${FRAME:-0}"

if [[ -n "${RADII:-}" ]]; then
  read -r -a radii <<<"$RADII"
else
  radii=(0 2 5 10 25 -1)
fi
if [[ -n "${GPUS:-}" ]]; then
  read -r -a gpus <<<"$GPUS"
else
  gpus=(1 2 3 4 5 6)
fi
if [[ "${#radii[@]}" -ne "${#gpus[@]}" ]]; then
  echo "RADII and GPUS must have the same number of entries" >&2
  exit 2
fi
pids=()

for index in "${!radii[@]}"; do
  radius="${radii[$index]}"
  gpu="${gpus[$index]}"
  label="r${radius}mm"
  if [[ "$radius" == "-1" ]]; then
    label="free"
  fi
  output="$OUTPUT_ROOT/$label"
  mkdir -p "$output"
  CUDA_VISIBLE_DEVICES="$gpu" "$ROOT/scripts/run_py310.sh" \
    "$ROOT/scripts/run_geometry_oracle.py" \
    --fit "$FIT" \
    --subject-root "$SUBJECT_ROOT" \
    --sequence "$SEQUENCE" \
    --frame "$FRAME" \
    --radius-mm "$radius" \
    --steps "$STEPS" \
    --height "$HEIGHT" \
    --position-lr "$POSITION_LR" \
    --seed 0 \
    --output "$output" \
    >"$output/stdout.log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "radius=$radius gpu=$gpu pid=$pid output=$output"
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done

if (( failures > 0 )); then
  echo "geometry ladder failed_runs=$failures" >&2
  exit 1
fi
echo "geometry ladder completed runs=${#pids[@]}"
