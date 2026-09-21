#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_ROOT="/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/nvs"

subjects=(422 443 445 475)
sequences=(EXP-2-eyes FREE EXP-6-tongue-1 HAIR)
gpus=(1 2 3 4)
pids=()

for index in "${!subjects[@]}"; do
  subject="${subjects[$index]}"
  sequence="${sequences[$index]}"
  gpu="${gpus[$index]}"
  output="$ROOT/artifacts/cache/flame_fit_${subject}.npz"
  log="$ROOT/artifacts/cache/flame_fit_${subject}.stdout.log"
  CUDA_VISIBLE_DEVICES="$gpu" "$ROOT/scripts/run_py310.sh" \
    "$ROOT/scripts/fit_flame_to_pcd.py" \
    --pcd "$DATASET_ROOT/$subject/sequences/$sequence/pointclouds/frame_00000.pcd" \
    --output "$output" \
    --outer-iterations 8 \
    --inner-steps 200 \
    --target-sample 200000 \
    >"$log" 2>&1 &
  pid=$!
  pids+=("$pid")
  echo "subject=$subject sequence=$sequence gpu=$gpu pid=$pid"
done

failures=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  echo "FLAME fits failed_runs=$failures" >&2
  exit 1
fi
echo "FLAME fits completed runs=${#pids[@]}"

