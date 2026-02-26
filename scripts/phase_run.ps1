param(
  [Parameter(Mandatory = $true)][ValidateSet("1", "2")][string]$Phase,
  [Parameter(Mandatory = $true)][int]$Seed
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Read-JsonFile {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path $Path)) {
    throw "json file not found: $Path"
  }
  return Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function To-PositiveIntOrDefault {
  param(
    [Parameter(Mandatory = $false)]$Value,
    [Parameter(Mandatory = $true)][int]$DefaultValue
  )
  try {
    $n = [int]$Value
    if ($n -gt 0) { return $n }
  }
  catch {
  }
  return $DefaultValue
}

function Resolve-Phase1Checkpoint {
  param(
    [Parameter(Mandatory = $true)][string]$CheckpointDir,
    [Parameter(Mandatory = $true)][int]$PreferredGeneration
  )
  if (-not (Test-Path $CheckpointDir)) {
    throw "phase1 checkpoint directory not found: $CheckpointDir"
  }

  $preferred = Join-Path $CheckpointDir "neat-checkpoint-gen$PreferredGeneration"
  if (Test-Path $preferred) {
    return [ordered]@{
      path = $preferred
      generation = $PreferredGeneration
    }
  }

  $latest = Get-ChildItem -Path $CheckpointDir -File -Filter "neat-checkpoint-gen*" |
    ForEach-Object {
      $m = [regex]::Match($_.Name, "gen(\d+)$")
      if (-not $m.Success) { return $null }
      [ordered]@{
        path = $_.FullName
        generation = [int]$m.Groups[1].Value
      }
    } |
    Where-Object { $null -ne $_ } |
    Sort-Object generation -Descending |
    Select-Object -First 1

  if ($null -eq $latest) {
    throw "phase1 checkpoint not found in: $CheckpointDir"
  }
  return $latest
}

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "python not found: $python"
}

$configFeedforward = "scripts/configs/neat_feedforward.ini"
$runtimeConfig = "scripts/configs/runtime_phase$Phase.json"
$outputDir = "logs/neat_phase${Phase}_seed$Seed"

if (-not (Test-Path $configFeedforward)) {
  throw "config not found: $configFeedforward"
}
if (-not (Test-Path $runtimeConfig)) {
  throw "runtime config not found: $runtimeConfig"
}

$cmd = @(
  "scripts/neat_train.py",
  "--config-feedforward", $configFeedforward,
  "--runtime-config", $runtimeConfig,
  "--output-dir", $outputDir,
  "--seed", "$Seed",
  "--profile-name", "phase${Phase}_seed$Seed"
)

if ($Phase -eq "2") {
  $phase1RuntimePath = "scripts/configs/runtime_phase1.json"
  $phase1Runtime = Read-JsonFile -Path $phase1RuntimePath
  $phase1Generations = To-PositiveIntOrDefault -Value $phase1Runtime.generations -DefaultValue 20
  $phase1CheckpointDir = "logs/neat_phase1_seed$Seed/checkpoints"
  $resume = Resolve-Phase1Checkpoint -CheckpointDir $phase1CheckpointDir -PreferredGeneration $phase1Generations

  $cmd += @(
    "--resume", "$($resume.path)",
    "--base-generation", "$($resume.generation)"
  )
}

$result = & $python @cmd | Out-String
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
  if (-not [string]::IsNullOrWhiteSpace($result)) {
    Write-Host $result.Trim()
  }
  exit $exitCode
}

try {
  $summary = $result | ConvertFrom-Json
}
catch {
  if (-not [string]::IsNullOrWhiteSpace($result)) {
    Write-Host $result.Trim()
  }
  throw "failed to parse neat_train output as JSON"
}

Write-Host "=== Phase$Phase Summary (Seed=$Seed) ==="
Write-Host "EMA win rate:     $($summary.gate_state.ema_win_rate)"
Write-Host "EMA imitation:    $($summary.gate_state.ema_imitation)"
Write-Host "Latest win rate:  $($summary.gate_state.latest_win_rate)"
Write-Host "Win slope(5):     $($summary.gate_state.latest_win_rate_slope_5)"
Write-Host "Transition ready: $($summary.gate_state.transition_ready)"
Write-Host "Transition gen:   $($summary.gate_state.transition_generation)"
Write-Host "Best fitness:     $($summary.best_fitness)"
Write-Host "================================"

exit 0