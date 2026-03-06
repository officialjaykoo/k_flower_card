param(
  [Parameter(Mandatory = $true)][string]$RuntimeConfig,
  [Parameter(Mandatory = $false)][string]$Python = ".\.venv\Scripts\python",
  [Parameter(Mandatory = $false)][string]$Seed = "",
  [Parameter(Mandatory = $false)][string]$CheckpointPath = "",
  [Parameter(Mandatory = $false)][string]$ResultOut = "",
  [Parameter(Mandatory = $false)][Nullable[int]]$Games = $null,
  [Parameter(Mandatory = $false)][Nullable[int]]$Workers = $null,
  [Parameter(Mandatory = $false)][string]$OpponentPolicy = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $RuntimeConfig)) {
  throw "runtime config not found: $RuntimeConfig"
}

$args = @(
  "ppo_by_GPT/scripts/duel_ppo_vs_v5.py",
  "--runtime-config", "$RuntimeConfig"
)

if (-not [string]::IsNullOrWhiteSpace($Seed)) {
  $args += @("--seed", "$Seed")
}
if (-not [string]::IsNullOrWhiteSpace($CheckpointPath)) {
  $args += @("--checkpoint-path", "$CheckpointPath")
}
if (-not [string]::IsNullOrWhiteSpace($ResultOut)) {
  $args += @("--result-out", "$ResultOut")
}
if ($null -ne $Games) {
  if ($Games -le 0) {
    throw "Games must be > 0 when provided, got: $Games"
  }
  $args += @("--games", "$Games")
}
if ($null -ne $Workers) {
  if ($Workers -le 0) {
    throw "Workers must be > 0 when provided, got: $Workers"
  }
  $args += @("--workers", "$Workers")
}
if (-not [string]::IsNullOrWhiteSpace($OpponentPolicy)) {
  $args += @("--opponent-policy", "$OpponentPolicy")
}

Write-Host "=== PPO Duel Start ==="
Write-Host "Python: $Python"
Write-Host "Runtime: $RuntimeConfig"
if (-not [string]::IsNullOrWhiteSpace($Seed)) { Write-Host "Seed override: $Seed" }
if (-not [string]::IsNullOrWhiteSpace($CheckpointPath)) { Write-Host "Checkpoint override: $CheckpointPath" }
if (-not [string]::IsNullOrWhiteSpace($ResultOut)) { Write-Host "Result override: $ResultOut" }
if ($null -ne $Games) { Write-Host "Games override: $Games" }
if ($null -ne $Workers) { Write-Host "Workers override: $Workers" }
if (-not [string]::IsNullOrWhiteSpace($OpponentPolicy)) { Write-Host "Opponent override: $OpponentPolicy" }

& $Python @args
if ($LASTEXITCODE -ne 0) {
  throw "PPO duel failed with exit code $LASTEXITCODE"
}
