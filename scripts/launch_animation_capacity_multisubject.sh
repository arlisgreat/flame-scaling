#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/mono_flame_avatar"
OUT="$ROOT/artifacts/runs/animation_capacity_exp5"
mkdir -p "$OUT/logs"

run_one() {
  local gpu="$1"
  local subject="$2"
  local method="$3"
  local count="$4"
  local steps="$5"
  local label="$6"
  local capacity="$7"
  local tag="n${label}_${subject}_${method}_k${capacity}"
  CUDA_VISIBLE_DEVICES="$gpu" "$ROOT/scripts/run_py310.sh" "$ROOT/scripts/run_animation_oracle.py" \
    --subject-root "$DATA/$subject" \
    --sequence EXP-5-mouth \
    --method "$method" \
    --radius-mm 30 \
    --release-penalty 0.001 \
    --train-count "$count" \
    --height 128 \
    --steps "$steps" \
    --batch-size 4 \
    --latent-dimensions "$capacity" \
    --seed 0 \
    --output "$OUT/n${label}/$subject/${method}_k${capacity}" \
    > "$OUT/logs/$tag.log" 2>&1
}

launch_batch() {
  local specs=("$@")
  local gpu=0
  for spec in "${specs[@]}"; do
    read -r subject method count steps label capacity <<< "$spec"
    run_one "$gpu" "$subject" "$method" "$count" "$steps" "$label" "$capacity" &
    gpu=$((gpu + 1))
  done
  wait
}

for label in 8 all; do
  if [[ "$label" == "8" ]]; then count=8; steps=500; fi
  if [[ "$label" == "all" ]]; then count=0; steps=3500; fi
  specs=()
  for capacity in 8 32; do
    for subject in 393 404 461 477 486; do
      for method in adaptive free; do
        specs+=("$subject $method $count $steps $label $capacity")
      done
    done
  done
  launch_batch "${specs[@]:0:8}"
  launch_batch "${specs[@]:8:8}"
  launch_batch "${specs[@]:16}"
done
