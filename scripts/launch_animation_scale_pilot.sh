#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUBJECT="/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/mono_flame_avatar/393"
OUT="$ROOT/artifacts/runs/animation_scale_393_exp5"
mkdir -p "$OUT/logs"

run_condition() {
  local gpu="$1"
  local train_count="$2"
  local steps="$3"
  local method="$4"
  local label="$5"
  CUDA_VISIBLE_DEVICES="$gpu" "$ROOT/scripts/run_py310.sh" "$ROOT/scripts/run_animation_oracle.py" \
    --subject-root "$SUBJECT" \
    --sequence EXP-5-mouth \
    --method "$method" \
    --radius-mm 30 \
    --release-penalty 0.001 \
    --train-count "$train_count" \
    --height 128 \
    --steps "$steps" \
    --batch-size 4 \
    --latent-dimensions 16 \
    --seed 0 \
    --output "$OUT/n${label}_${method}" \
    > "$OUT/logs/n${label}_${method}.log" 2>&1
}

run_condition 0 8 500 hard 8 &
run_condition 1 8 500 bounded 8 &
run_condition 2 8 500 adaptive 8 &
run_condition 3 8 500 free 8 &
run_condition 4 72 1500 hard 72 &
run_condition 5 72 1500 bounded 72 &
run_condition 6 72 1500 adaptive 72 &
run_condition 7 72 1500 free 72 &
wait

run_condition 0 0 3250 hard all &
run_condition 1 0 3250 bounded all &
run_condition 2 0 3250 adaptive all &
run_condition 3 0 3250 free all &
wait
