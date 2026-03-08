param(
  [Parameter(Mandatory = $true)][string]$Human,
  [Parameter(Mandatory = $true)][string]$Ai,
  [int]$Games = 1000,
  [string]$Seed = "model-duel-gpt",
  [int]$MaxSteps = 600,
  [ValidateSet("alternate", "fixed")][string]$FirstTurnPolicy = "alternate",
  [ValidateSet("human")][string]$FixedFirstTurn = "human",
  [ValidateSet("1", "2")][string]$ContinuousSeries = "1",
  [ValidateSet("text", "json")][string]$StdoutFormat = "text",
  [string]$ResultOut = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$CommonScript = Join-Path $PSScriptRoot "common.ps1"
if (-not (Test-Path $CommonScript)) {
  throw "common helper not found: $CommonScript"
}
. $CommonScript

$RepoRoot = Get-NeatGptRepoRoot -ScriptRoot $PSScriptRoot
$DuelWorker = Join-Path $PSScriptRoot "scripts\model_duel_worker.mjs"
if (-not (Test-Path $DuelWorker)) {
  throw "duel worker not found: $DuelWorker"
}

$null = Get-Command node -ErrorAction Stop

$cmd = @(
  $DuelWorker,
  "--human", $Human,
  "--ai", $Ai,
  "--games", "$Games",
  "--seed", $Seed,
  "--max-steps", "$MaxSteps",
  "--first-turn-policy", $FirstTurnPolicy,
  "--fixed-first-turn", $FixedFirstTurn,
  "--continuous-series", $ContinuousSeries,
  "--stdout-format", $StdoutFormat
)

if (-not [string]::IsNullOrWhiteSpace($ResultOut)) {
  $resolvedResultOut = Resolve-PathFromBase -Path $ResultOut -BasePath $RepoRoot
  $cmd += @("--result-out", $resolvedResultOut)
}

$resultLines = & node @cmd
$exitCode = $LASTEXITCODE
if ($null -ne $resultLines) {
  $resultLines | ForEach-Object { Write-Output $_ }
}
exit $exitCode
