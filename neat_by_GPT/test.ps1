Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptPath = Join-Path $PSScriptRoot "scripts\regression_smoke.mjs"
if (-not (Test-Path $ScriptPath)) {
  throw "regression smoke script not found: $ScriptPath"
}

$null = Get-Command node -ErrorAction Stop
& node $ScriptPath
exit $LASTEXITCODE
