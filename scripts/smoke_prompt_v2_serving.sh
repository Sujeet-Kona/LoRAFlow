#!/usr/bin/env bash
set -euo pipefail

command -v loraflow >/dev/null
command -v curl >/dev/null
command -v python >/dev/null

port_base="${PORT_BASE:-8126}"
mkdir -p artifacts/v4-prompt-v2

for track in a b c; do
  case "$track" in
    a) offset=0 ;;
    b) offset=1 ;;
    c) offset=2 ;;
  esac

  port=$((port_base + offset))
  base_url="http://127.0.0.1:$port"
  config="configs/v4_prompt_v2_track_${track}.yaml"
  adapter="artifacts/v4-prompt-v2/track-${track}-exported/adapter"
  log_file="artifacts/v4-prompt-v2/track-${track}-serve.log"
  train_file="data/v4-prompt-v2/track_${track}/train.jsonl"
  test_file="data/v4-prompt-v2/track_${track}/test.jsonl"
  health_file="$(mktemp)"
  payload_file="$(mktemp)"
  response_file="$(mktemp)"

  test -f "$config"
  test -f "$train_file"
  test -f "$test_file"
  test -s "$adapter/adapter_model.safetensors"
  test -s "$adapter/adapter_config.json"
  test -s "$adapter/tokenizer.json"

  track_upper="$(printf '%s' "$track" | tr '[:lower:]' '[:upper:]')"
  echo "===== Prompt-v2 Track $track_upper ====="
  loraflow serve --config "$config" --adapter "$adapter" --host 127.0.0.1 --port "$port" >"$log_file" 2>&1 &
  server_pid=$!

  cleanup() {
    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    rm -f "$health_file" "$payload_file" "$response_file"
  }
  trap cleanup EXIT

  ready=0
  for attempt in $(seq 1 90); do
    if curl --silent --show-error --fail "$base_url/health" -o "$health_file"; then
      ready=1
      break
    fi
    if ! kill -0 "$server_pid" 2>/dev/null; then
      cat "$log_file"
      echo "Server exited before becoming healthy for track $track" >&2
      exit 1
    fi
    sleep 2
  done

  if [ "$ready" -ne 1 ]; then
    cat "$log_file"
    echo "Server did not become healthy for track $track" >&2
    exit 1
  fi

  python - "$health_file" <<'PY'
import json
import sys

health = json.load(open(sys.argv[1], encoding="utf-8"))
assert health.get("status") == "ok", health
assert health.get("service", {}).get("adapter_applied") is True, health
print("health=ok adapter_applied=true")
PY

  python - "$train_file" "$test_file" "$payload_file" <<'PY'
import json
import sys

def first_record(path):
    with open(path, encoding="utf-8") as source:
        for line in source:
            if line.strip():
                return json.loads(line)
    raise SystemExit(f"No JSONL records in {path}")

train_record = first_record(sys.argv[1])
test_record = first_record(sys.argv[2])
payload = {
    "prompt": train_record["instruction"],
    "input": test_record["input"],
    "max_new_tokens": 24,
    "temperature": 0,
}
with open(sys.argv[3], "w", encoding="utf-8") as target:
    json.dump(payload, target)
PY

  curl --silent --show-error --fail \
    -X POST "$base_url/generate" \
    -H 'content-type: application/json' \
    --data-binary "@$payload_file" > "$response_file"

  python - "$response_file" <<'PY'
import json
import re
import sys

result = json.load(open(sys.argv[1], encoding="utf-8"))
line = result.get("completion_line", "")
pattern = (
    r"sev=(routine|watch|urgent) "
    r"cat=(mobility|feeding|respiratory|skin|behaviour|unknown) "
    r"act=(log_only|recheck_12h|flag_vet_now)"
)
assert result.get("adapter_applied") is True, result
assert re.fullmatch(pattern, line), result
print(f"schema_valid=true completion={line}")
PY

  cleanup
  trap - EXIT
done

echo "All three Prompt-v2 adapters passed live FastAPI smoke checks."
