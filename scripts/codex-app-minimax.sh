#!/bin/zsh
set -euo pipefail

ROOT_DIR="${0:A:h:h}"
TARGET_PATH="${1:-.}"
PROXY_URL="${CODEX_MINIMAX_PROXY_URL:-http://localhost:4000/health}"
PROXY_PORT="${CODEX_MINIMAX_PROXY_PORT:-4000}"
LOG_FILE="${CODEX_MINIMAX_PROXY_LOG:-/tmp/codex-minimax-proxy.log}"
USE_OPENROUTER="${CODEX_MINIMAX_USE_OPENROUTER:-0}"
CODEX_BIN="${CODEX_BIN:-codex}"
NODE_BIN="${NODE_BIN:-node}"

if ! curl -fsS "$PROXY_URL" >/dev/null 2>&1; then
  if [ "$USE_OPENROUTER" = "1" ]; then
    nohup env PROXY_PORT="$PROXY_PORT" MINIMAX_API_KEY="${MINIMAX_API_KEY:-}" OPENAI_API_KEY="${OPENAI_API_KEY:-}" OPENROUTER_API_KEY="${OPENROUTER_API_KEY:-}" "$NODE_BIN" "$ROOT_DIR/proxy.mjs" >"$LOG_FILE" 2>&1 &
  else
    nohup env PROXY_PORT="$PROXY_PORT" MINIMAX_API_KEY="${MINIMAX_API_KEY:-}" OPENAI_API_KEY="${OPENAI_API_KEY:-}" OPENROUTER_API_KEY="" "$NODE_BIN" "$ROOT_DIR/proxy.mjs" >"$LOG_FILE" 2>&1 &
  fi
  sleep 2
fi

exec "$CODEX_BIN" app -c 'model="MiniMax-M2.7"' -c 'model_provider="minimax_proxy"' "$TARGET_PATH"
