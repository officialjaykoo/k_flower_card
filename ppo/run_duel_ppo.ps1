param(
  [Parameter(Mandatory = $true)][string]$RuntimeConfig,
  [Parameter(Mandatory = $false)][string]$Python = ".\.venv\Scripts\python"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $RuntimeConfig)) {
  throw "runtime config not found: $RuntimeConfig"
}

$args = @(
  "ppo/scripts/duel_ppo_vs_v5.py",
  "--runtime-config", "$RuntimeConfig"
)

Write-Host "=== PPO Duel Start ==="
Write-Host "Python: $Python"
Write-Host "Runtime: $RuntimeConfig"

& $Python @args
if ($LASTEXITCODE -ne 0) {
  throw "PPO duel failed with exit code $LASTEXITCODE"
}
