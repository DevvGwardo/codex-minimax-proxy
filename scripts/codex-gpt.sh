#!/bin/zsh
set -euo pipefail

CODEX_BIN="${CODEX_BIN:-codex}"

exec "$CODEX_BIN" "$@"
