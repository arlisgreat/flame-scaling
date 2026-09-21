#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MANIFEST="${1:-${ROOT}/artifacts/audits/nersemble_identity_scaling.json}"
OUTPUT_ROOT="${2:-${ROOT}/artifacts/cache/nersemble_scale_frames}"

mkdir -p "${OUTPUT_ROOT}"
mapfile -t subjects < <(
  python3 - "${MANIFEST}" <<'PY'
import json
import sys
payload = json.load(open(sys.argv[1]))
for subject, row in sorted(payload["per_subject"].items()):
    if row["usable"]:
        print(subject)
PY
)

pids=()
pid_subjects=()
failures=()
slot=0

wait_wave() {
  for index in "${!pids[@]}"; do
    if ! wait "${pids[$index]}"; then
      failures+=("${pid_subjects[$index]}")
    fi
  done
  pids=()
  pid_subjects=()
  slot=0
}

for subject in "${subjects[@]}"; do
  output="${OUTPUT_ROOT}/${subject}.npz"
  report="${OUTPUT_ROOT}/${subject}.json"
  if [[ -s "${output}" && -s "${report}" ]]; then
    continue
  fi
  CUDA_VISIBLE_DEVICES="${slot}" "${ROOT}/scripts/run_py310.sh" \
    "${ROOT}/scripts/prepare_nersemble_scale_subject.py" \
    --subject "${subject}" \
    --sequence FREE \
    --candidate-count 7 \
    --fit-steps 1000 \
    --output "${output}" \
    >"${OUTPUT_ROOT}/${subject}.stdout.log" 2>&1 &
  pids+=("$!")
  pid_subjects+=("${subject}")
  slot=$((slot + 1))
  if (( slot == 8 )); then
    wait_wave
  fi
done

if (( slot > 0 )); then
  wait_wave
fi

if (( ${#failures[@]} > 0 )); then
  printf '%s\n' "${failures[@]}" >"${OUTPUT_ROOT}/failed_subjects.txt"
  echo "NeRSemble scale cache failures=${#failures[@]} subjects=${failures[*]}" >&2
  exit 1
fi

rm -f "${OUTPUT_ROOT}/failed_subjects.txt"
echo "Completed NeRSemble scale cache subjects=${#subjects[@]} output=${OUTPUT_ROOT}"
