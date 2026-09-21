#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA="/data1/workspace/airulan/3dv/datasets/nersemble_benchmark/mono_flame_avatar"
OUT="$ROOT/artifacts/runs/correspondence_multisubject"
mkdir -p "$OUT/logs"

run_one() {
  local gpu="$1"
  local sequence="$2"
  local subject="$3"
  local mode="$4"
  local fraction="$5"
  local count="$6"
  local steps="$7"
  local label="$8"
  local short="${sequence//[^A-Za-z0-9]/_}"
  local frac_tag="${fraction//./p}"
  CUDA_VISIBLE_DEVICES="$gpu" "$ROOT/scripts/run_py310.sh" "$ROOT/scripts/run_animation_oracle.py" \
    --subject-root "$DATA/$subject" \
    --sequence "$sequence" \
    --method hard \
    --correspondence-shuffle-mode "$mode" \
    --correspondence-shuffle-fraction "$fraction" \
    --correspondence-local-bin-mm 10 \
    --train-count "$count" \
    --height 128 \
    --steps "$steps" \
    --batch-size 4 \
    --latent-dimensions 16 \
    --seed 0 \
    --output "$OUT/$short/n${label}/$subject/${mode}_${frac_tag}" \
    > "$OUT/logs/${short}_n${label}_${subject}_${mode}_${frac_tag}.log" 2>&1
}

launch_batch() {
  local specs=("$@")
  local gpu=0
  for spec in "${specs[@]}"; do
    read -r sequence subject mode fraction count steps label <<< "$spec"
    run_one "$gpu" "$sequence" "$subject" "$mode" "$fraction" "$count" "$steps" "$label" &
    gpu=$((gpu + 1))
  done
  wait
}

for sequence in EXP-2-eyes EXP-5-mouth; do
  for label in 8 all; do
    if [[ "$label" == "8" ]]; then count=8; steps=500; fi
    if [[ "$label" == "all" ]]; then count=0; steps=3500; fi
    specs=()
    for subject in 393 404 461 477 486; do
      for mode in local global; do
        for fraction in 0.25 0.5 1.0; do
          specs+=("$sequence $subject $mode $fraction $count $steps $label")
        done
      done
    done
    launch_batch "${specs[@]:0:8}"
    launch_batch "${specs[@]:8:8}"
    launch_batch "${specs[@]:16:8}"
    launch_batch "${specs[@]:24}"
  done
done
