#!/usr/bin/env bash
set -euo pipefail

command -v loraflow >/dev/null
command -v python >/dev/null

for track in a b c; do
  config="configs/v4_prompt_v2_track_${track}.yaml"
  data_dir="data/v4-prompt-v2/track_${track}"
  run_dir="artifacts/v4-prompt-v2/track-${track}-run"
  export_dir="artifacts/v4-prompt-v2/track-${track}-exported"

  test -s "$config"
  test -s "$data_dir/manifest.json"
  test -s "$data_dir/train.jsonl"
  test -s "$data_dir/validation.jsonl"
  test -s "$data_dir/test.jsonl"

  python - "$data_dir" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
for split in ("train", "validation", "test"):
    path = root / f"{split}.jsonl"
    raw = path.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    expected = manifest["sha256"][split]
    count = sum(bool(line.strip()) for line in raw.splitlines())
    expected_count = manifest["counts"][split]
    if digest != expected:
        raise SystemExit(f"{path}: SHA-256 mismatch: {digest} != {expected}")
    if count != expected_count:
        raise SystemExit(f"{path}: record count mismatch: {count} != {expected_count}")
    print(f"{path}: {count} records; SHA-256 verified")
PY

  track_upper="$(printf '%s' "$track" | tr '[:lower:]' '[:upper:]')"

  if [[ -e "$run_dir" || -e "$export_dir/adapter" ]]; then
    printf 'Refusing to overwrite existing run/export for track %s. Preserve it or move it aside first.\n' "$track" >&2
    exit 2
  fi

  printf '\n===== PROMPT-V2 TRACK %s: DATA CHECK =====\n' "$track_upper"
  loraflow data-check --config "$config"

  printf '\n===== PROMPT-V2 TRACK %s: TRAIN =====\n' "$track_upper"
  loraflow train --config "$config"

  printf '\n===== PROMPT-V2 TRACK %s: EVALUATE =====\n' "$track_upper"
  loraflow evaluate --config "$config" --adapter "$run_dir/adapter"

  printf '\n===== PROMPT-V2 TRACK %s: EXPORT + RELOAD VALIDATION =====\n' "$track_upper"
  loraflow export --config "$config" --adapter "$run_dir/adapter"
done

printf '\n===== LIVE FASTAPI SMOKE TEST =====\n'
bash scripts/smoke_prompt_v2_serving.sh

printf '\nAll Prompt-v2 tracks trained, evaluated, exported, reload-validated, and served.\n'
