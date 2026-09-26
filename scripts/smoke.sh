#!/usr/bin/env bash
# One-shot smoke test: lint + tests + tiny training + inference + API call.
# Run from the repository root.

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "[smoke] === ruff ==="
ruff check z_jev tests || exit 1

echo "[smoke] === pytest ==="
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -q

echo "[smoke] === training tiny model ==="
CKPT_DIR="${CKPT_DIR:-checkpoints/smoke}"
rm -rf "$CKPT_DIR"
python3 -m z_jev.train \
    --dataset mixed \
    --steps 60 \
    --n-train 96 \
    --n-val 32 \
    --batch-size 4 \
    --max-state-len 96 \
    --hidden 128 \
    --layers 2 \
    --out "$CKPT_DIR"

echo "[smoke] === CLI inference ==="
REQ_FILE=$(mktemp)
cat > "$REQ_FILE" <<'EOF'
{
  "state": "free lunch click urgent verify password now",
  "questions": {
    "category": {
      "type": "choice",
      "instructions": "Classify the message as spam or ham",
      "criteria": {"spam": "Spam", "ham": "Ham"}
    },
    "risk": {
      "type": "score",
      "instructions": "Operational risk",
      "criteria": ["low", "medium", "high"]
    },
    "is_urgent": {"type": "noul", "instructions": "Is the message urgent?"}
  }
}
EOF

OUT_FILE="$ROOT/examples/sample_output.json"
python3 -m z_jev.infer_cli --checkpoint "$CKPT_DIR/model.pt" --request "$REQ_FILE" \
    | tee "$OUT_FILE"
rm -f "$REQ_FILE"

echo "[smoke] === curl the API ==="
# Free a stale listener from any earlier run.
fuser -k 127.0.0.1:8089/tcp 2>/dev/null || true
sleep 1
ZJEV_CHECKPOINT="$CKPT_DIR/model.pt" python3 -m uvicorn z_jev.serve:app --host 127.0.0.1 --port 8089 \
    --log-level warning &
SERVER_PID=$!
for _ in $(seq 1 30); do
    if curl -sf -o /dev/null http://127.0.0.1:8089/healthz; then break; fi
    sleep 1
done
curl -sf http://127.0.0.1:8089/healthz > /dev/null

REQ=$(cat <<'EOF'
{
  "state": "free lunch click urgent verify password now",
  "questions": {
    "category": {
      "type": "choice",
      "instructions": "Classify the message as spam or ham",
      "criteria": {"spam": "Spam", "ham": "Ham"}
    },
    "risk": {
      "type": "score",
      "instructions": "Operational risk",
      "criteria": ["low", "medium", "high"]
    },
    "is_urgent": {"type": "noul", "instructions": "Is the message urgent?"}
  }
}
EOF
)

HTTP_CODE=$(curl -s -o /tmp/zjev_api.json -w "%{http_code}" \
    -H "Content-Type: application/json" \
    -X POST http://127.0.0.1:8089/v1/evaluate \
    --data "$REQ")
echo "[api] HTTP $HTTP_CODE"
cat /tmp/zjev_api.json | python3 -m json.tool

curl -s http://127.0.0.1:8089/healthz | python3 -m json.tool

kill "$SERVER_PID" || true
wait "$SERVER_PID" 2>/dev/null || true

echo "[smoke] OK"