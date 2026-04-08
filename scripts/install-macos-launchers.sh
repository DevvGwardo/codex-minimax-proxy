#!/bin/zsh
set -euo pipefail

REPO_ROOT="${0:A:h:h}"
OUTPUT_DIR="${1:-$HOME/Desktop/Codex Launchers}"
CODEX_BIN="${CODEX_BIN:-$(command -v codex || true)}"
NODE_BIN="${NODE_BIN:-$(command -v node || true)}"

if [ -z "$CODEX_BIN" ]; then
  echo "codex not found in PATH" >&2
  exit 1
fi

if [ -z "$NODE_BIN" ]; then
  echo "node not found in PATH" >&2
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

write_command_launcher() {
  local output_path="$1"
  local repo_script="$2"
  local use_openrouter="$3"
  local mode_label="$4"

  cat >"$output_path" <<EOF
#!/bin/zsh
set -euo pipefail

REPO_ROOT="$REPO_ROOT"
CODEX_BIN="\${CODEX_BIN:-$CODEX_BIN}"
NODE_BIN="\${NODE_BIN:-$NODE_BIN}"
MODE_LABEL="$mode_label"

if [ "\$#" -gt 0 ]; then
  TARGET_PATH="\$1"
else
  TARGET_PATH="\$HOME"
fi

if [ -n "\${CODEX_LAUNCHER_TARGET_PATH:-}" ]; then
  TARGET_PATH="\$CODEX_LAUNCHER_TARGET_PATH"
fi

clear
echo "Codex Launcher"
echo "Mode: \$MODE_LABEL"
echo "Folder: \$TARGET_PATH"
echo ""

if [ ! -x "\$CODEX_BIN" ] && ! command -v "\$CODEX_BIN" >/dev/null 2>&1; then
  echo "Codex CLI was not found."
  echo "Expected: \$CODEX_BIN"
  echo ""
  read -r "?Press Enter to close..."
  exit 1
fi

if [ ! -x "\$NODE_BIN" ] && ! command -v "\$NODE_BIN" >/dev/null 2>&1; then
  echo "Node.js was not found."
  echo "Expected: \$NODE_BIN"
  echo ""
  read -r "?Press Enter to close..."
  exit 1
fi

if env CODEX_BIN="\$CODEX_BIN" NODE_BIN="\$NODE_BIN" CODEX_MINIMAX_USE_OPENROUTER="$use_openrouter" /bin/zsh "\$REPO_ROOT/$repo_script" "\$TARGET_PATH"; then
  launcher_exit_code=0
else
  launcher_exit_code=\$?
  echo ""
  echo "Codex did not start successfully."
  echo "If this is MiniMax mode, check:"
  echo "  /tmp/codex-minimax-proxy.log"
  echo ""
  read -r "?Press Enter to close..."
fi

exit \$launcher_exit_code
EOF

  chmod +x "$output_path"
}

GPT_COMMAND="$OUTPUT_DIR/Codex GPT.command"
MINIMAX_COMMAND="$OUTPUT_DIR/Codex MiniMax.command"

write_command_launcher "$GPT_COMMAND" "scripts/codex-app-gpt.sh" "0" "GPT"
write_command_launcher "$MINIMAX_COMMAND" "scripts/codex-app-minimax.sh" "0" "MiniMax"

cat >"$OUTPUT_DIR/README.txt" <<EOF
Codex launchers created in:
$OUTPUT_DIR

Double-click:
- Codex GPT.command
- Codex MiniMax.command

These launchers open Codex in the user's home folder by default.
If you need a specific path, run the .command file from Terminal with a folder path argument.
If MiniMax mode fails, check:
  /tmp/codex-minimax-proxy.log
EOF

echo "Created macOS launchers in: $OUTPUT_DIR"
