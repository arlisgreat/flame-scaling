#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/mono_flame_avatar"
OUT="$ROOT/artifacts/runs/animation_multisubject_exp5"
mkdir -p "$OUT/logs"

run_one() {
  local gpu="$1"
  local subject="$2"
  local method="$3"
  local count="$4"
  local steps="$5"
  local label="$6"
  local tag="n${label}_${subject}_${method}"
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
    --latent-dimensions 16 \
    --seed 0 \
    --output "$OUT/n${label}/$subject/$method" \
    > "$OUT/logs/$tag.log" 2>&1
}

launch_batch() {
  local specs=("$@")
  local gpu=0
  for spec in "${specs[@]}"; do
    read -r subject method count steps label <<< "$spec"
    run_one "$gpu" "$subject" "$method" "$count" "$steps" "$label" &
    gpu=$((gpu + 1))
  done
  wait
}

for label in 8 24 all; do
  if [[ "$label" == "8" ]]; then count=8; steps=500; fi
  if [[ "$label" == "24" ]]; then count=24; steps=500; fi
  if [[ "$label" == "all" ]]; then count=0; steps=3500; fi

  launch_batch \
    "393 hard $count $steps $label" "393 adaptive $count $steps $label" "393 free $count $steps $label" \
    "404 hard $count $steps $label" "404 adaptive $count $steps $label" "404 free $count $steps $label" \
    "461 hard $count $steps $label" "461 adaptive $count $steps $label"
  launch_batch \
    "461 free $count $steps $label" \
    "477 hard $count $steps $label" "477 adaptive $count $steps $label" "477 free $count $steps $label" \
    "486 hard $count $steps $label" "486 adaptive $count $steps $label" "486 free $count $steps $label"
done
