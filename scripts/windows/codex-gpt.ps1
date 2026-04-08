param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

$codexBin = if ($env:CODEX_BIN) { $env:CODEX_BIN } else { (Get-Command codex -ErrorAction Stop).Source }
& $codexBin @RemainingArgs
