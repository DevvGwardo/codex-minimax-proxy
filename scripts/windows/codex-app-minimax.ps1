param(
  [string]$TargetPath = "."
)

$ErrorActionPreference = "Stop"

$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$repoRoot = Split-Path -Parent (Split-Path -Parent $scriptDir)
$proxyPort = if ($env:CODEX_MINIMAX_PROXY_PORT) { $env:CODEX_MINIMAX_PROXY_PORT } else { "4000" }
$proxyUrl = if ($env:CODEX_MINIMAX_PROXY_URL) { $env:CODEX_MINIMAX_PROXY_URL } else { "http://localhost:$proxyPort/health" }
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

& $codexBin "app" "-c" 'model="MiniMax-M2.7"' "-c" 'model_provider="minimax_proxy"' $TargetPath
