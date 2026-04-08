param(
  [Parameter(ValueFromRemainingArguments = $true)]
  [string[]]$RemainingArgs
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$proxyPort = if ($env:CODEX_MINIMAX_PROXY_PORT) { $env:CODEX_MINIMAX_PROXY_PORT } else { "4000" }
$proxyUrl = if ($env:CODEX_MINIMAX_PROXY_URL) { $env:CODEX_MINIMAX_PROXY_URL } else { "http://localhost:$proxyPort/health" }
$logFile = if ($env:CODEX_MINIMAX_PROXY_LOG) { $env:CODEX_MINIMAX_PROXY_LOG } else { "$env:TEMP\codex-minimax-proxy.log" }
$useOpenRouter = if ($env:CODEX_MINIMAX_USE_OPENROUTER) { $env:CODEX_MINIMAX_USE_OPENROUTER } else { "0" }
$codexBin = if ($env:CODEX_BIN) { $env:CODEX_BIN } else { (Get-Command codex -ErrorAction Stop).Source }
$nodeBin = if ($env:NODE_BIN) { $env:NODE_BIN } else { (Get-Command node -ErrorAction Stop).Source }

try {
  Invoke-WebRequest -Uri $proxyUrl -UseBasicParsing -TimeoutSec 2 | Out-Null
} catch {
  $originalOpenRouter = $env:OPENROUTER_API_KEY
  $env:PROXY_PORT = $proxyPort
  if ($useOpenRouter -ne "1") {
    $env:OPENROUTER_API_KEY = ""
  }

  Start-Process -FilePath $nodeBin -ArgumentList "`"$repoRoot\proxy.mjs`"" -WindowStyle Hidden | Out-Null
  Start-Sleep -Seconds 2

  if ($useOpenRouter -ne "1") {
    $env:OPENROUTER_API_KEY = $originalOpenRouter
  }
}

if ($RemainingArgs.Count -gt 0) {
  $subcommand = $RemainingArgs[0]
  $rest = @()
  if ($RemainingArgs.Count -gt 1) {
    $rest = $RemainingArgs[1..($RemainingArgs.Count - 1)]
  }
  & $codexBin $subcommand "-c" 'model="MiniMax-M2.7"' "-c" 'model_provider="minimax_proxy"' @rest
} else {
  & $codexBin "-c" 'model="MiniMax-M2.7"' "-c" 'model_provider="minimax_proxy"'
}
