#!/bin/zsh
set -euo pipefail

TARGET_PATH="${1:-.}"
CODEX_BIN="${CODEX_BIN:-codex}"

exec "$CODEX_BIN" app "$TARGET_PATH"
