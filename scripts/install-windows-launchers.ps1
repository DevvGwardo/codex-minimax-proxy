param(
  [string]$OutputDir = "$HOME\Desktop\Codex Launchers"
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$windowsScripts = Join-Path $repoRoot "scripts\windows"
$codexBin = (Get-Command codex -ErrorAction Stop).Source
$nodeBin = (Get-Command node -ErrorAction Stop).Source

New-Item -ItemType Directory -Force -Path $OutputDir | Out-Null

function New-LauncherFiles {
  param(
    [string]$BaseName,
    [string]$AppScript,
    [string]$ModeLabel
  )

  $ps1Path = Join-Path $OutputDir "$BaseName.ps1"
  $cmdPath = Join-Path $OutputDir "$BaseName.cmd"

  $ps1 = @"
Add-Type -AssemblyName System.Windows.Forms
`$dialog = New-Object System.Windows.Forms.FolderBrowserDialog
`$dialog.Description = 'Choose a folder to open in $BaseName'
if (`$dialog.ShowDialog() -ne [System.Windows.Forms.DialogResult]::OK) { exit 0 }
Write-Host 'Codex Launcher'
Write-Host 'Mode: $ModeLabel'
Write-Host ('Folder: ' + `$dialog.SelectedPath)
Write-Host ''
`$env:CODEX_BIN = '$codexBin'
`$env:NODE_BIN = '$nodeBin'
try {
  & '$windowsScripts\$AppScript' `$dialog.SelectedPath
  exit `$LASTEXITCODE
} catch {
  Write-Host ''
  Write-Host 'Codex did not start successfully.'
  Write-Host 'If this is MiniMax mode, check:'
  Write-Host '  %TEMP%\codex-minimax-proxy.log'
  Write-Host ''
  Read-Host 'Press Enter to close'
  exit 1
}
"@

  Set-Content -Path $ps1Path -Value $ps1 -Encoding UTF8

  $cmd = @"
@echo off
setlocal
powershell -NoLogo -NoProfile -ExecutionPolicy Bypass -File "$ps1Path"
"@

  Set-Content -Path $cmdPath -Value $cmd -Encoding ASCII
}

New-LauncherFiles -BaseName "Codex GPT" -AppScript "codex-app-gpt.ps1" -ModeLabel "GPT"
New-LauncherFiles -BaseName "Codex MiniMax" -AppScript "codex-app-minimax.ps1" -ModeLabel "MiniMax"

$wsh = New-Object -ComObject WScript.Shell
foreach ($name in @("Codex GPT", "Codex MiniMax")) {
  $shortcutPath = Join-Path $OutputDir "$name.lnk"
  $shortcut = $wsh.CreateShortcut($shortcutPath)
  $shortcut.TargetPath = Join-Path $OutputDir "$name.cmd"
  $shortcut.WorkingDirectory = $repoRoot
  $shortcut.Save()
}

$readme = @"
Windows Codex launchers created in:
$OutputDir

Use:
- Codex GPT.lnk
- Codex MiniMax.lnk

The .cmd and .ps1 files remain alongside the shortcuts for troubleshooting.
"@

Set-Content -Path (Join-Path $OutputDir "README.txt") -Value $readme -Encoding UTF8
Write-Host "Created Windows launchers in: $OutputDir"
