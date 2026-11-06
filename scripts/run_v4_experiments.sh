#!/usr/bin/env bash
set -euo pipefail

mkdir -p artifacts/v4-logs

for track in a b c; do
  config="configs/v4_track_${track}.yaml"
  run_dir="artifacts/v4-track-${track}-run"
  adapter_dir="${run_dir}/adapter"
  track_upper=$(printf '%s' "${track}" | tr '[:lower:]' '[:upper:]')
  metadata="${run_dir}/training_metadata.json"

  for split in train validation test; do
    dataset="data/v4/track_${track}/${split}.jsonl"
    if [[ ! -s "${dataset}" ]]; then
      printf 'Missing or empty dataset: %s\n' "${dataset}" >&2
      exit 1
    fi
  done

  if [[ -s "${adapter_dir}/adapter_model.safetensors" && -s "${adapter_dir}/adapter_config.json" && -s "${metadata}" ]]; then
    printf '\n===== TRACK %s: reusing completed adapter =====\n' "${track_upper}"
  else
    printf '\n===== TRACK %s: training =====\n' "${track_upper}"
    loraflow train --config "${config}" 2>&1 | tee "artifacts/v4-logs/train_${track}.log"
  fi

  printf '\n===== TRACK %s: held-out evaluation =====\n' "${track_upper}"
  loraflow evaluate --config "${config}" --adapter "${adapter_dir}" 2>&1 | tee "artifacts/v4-logs/evaluate_${track}.log"
done

python scripts/summarize_v4_results.py --output artifacts/v4-summary.md
printf '\nCompleted. Summary: artifacts/v4-summary.md\n'
