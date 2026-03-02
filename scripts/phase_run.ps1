param(
  [Parameter(Mandatory = $true)][ValidateSet("1", "2", "3")][string]$Phase,
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

function Resolve-PreviousCheckpoint {
  param(
    [Parameter(Mandatory = $true)][string]$CheckpointDir,
    [Parameter(Mandatory = $true)][int]$PreferredGeneration,
    [Parameter(Mandatory = $true)][string]$Label
  )
  if (-not (Test-Path $CheckpointDir)) {
    throw "$Label checkpoint directory not found: $CheckpointDir"
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
    throw "$Label checkpoint not found in: $CheckpointDir"
  }
  return $latest
}

function Get-BestGenomeGoMetrics {
  param([Parameter(Mandatory = $true)]$Summary)

  $empty = [ordered]@{
    available = $false
    go_count = $null
    go_games = $null
    go_fail_count = $null
    go_fail_rate = $null
    go_rate = $null
    luck_proxy = $null
    luck_seat_win_rate_gap = $null
    luck_gold_volatility_norm = $null
  }

  $generationMetricsPathRaw = [string]$Summary.generation_metrics_log
  $evalMetricsPathRaw = [string]$Summary.eval_metrics_log
  if ([string]::IsNullOrWhiteSpace($generationMetricsPathRaw) -or [string]::IsNullOrWhiteSpace($evalMetricsPathRaw)) {
    return $empty
  }

  $generationMetricsPath = [System.IO.Path]::GetFullPath($generationMetricsPathRaw)
  $evalMetricsPath = [System.IO.Path]::GetFullPath($evalMetricsPathRaw)
  if (-not (Test-Path $generationMetricsPath) -or -not (Test-Path $evalMetricsPath)) {
    return $empty
  }

  $lastGenerationLine = Get-Content $generationMetricsPath | Select-Object -Last 1
  if ([string]::IsNullOrWhiteSpace($lastGenerationLine)) {
    return $empty
  }

  $generationRecord = $lastGenerationLine | ConvertFrom-Json
  $targetGeneration = [int]$generationRecord.generation
  $targetGenomeKey = [int]$generationRecord.best_genome_key

  $bestEvalRecord =
    Get-Content $evalMetricsPath |
      ForEach-Object { $_ | ConvertFrom-Json } |
      Where-Object { [int]$_.generation -eq $targetGeneration -and [int]$_.genome_key -eq $targetGenomeKey } |
      Select-Object -First 1

  if ($null -eq $bestEvalRecord) {
    return $empty
  }

  return [ordered]@{
    available = $true
    go_count = [int]$bestEvalRecord.go_count
    go_games = [int]$bestEvalRecord.go_games
    go_fail_count = [int]$bestEvalRecord.go_fail_count
    go_fail_rate = [double]$bestEvalRecord.go_fail_rate
    go_rate = [double]$bestEvalRecord.go_rate
    luck_proxy = [double]$bestEvalRecord.luck_proxy
    luck_seat_win_rate_gap = [double]$bestEvalRecord.luck_components.seat_win_rate_gap
    luck_gold_volatility_norm = [double]$bestEvalRecord.luck_components.gold_volatility_norm
  }
}

function Get-LatestGenerationGoAverages {
  param([Parameter(Mandatory = $true)]$Summary)

  $empty = [ordered]@{
    available = $false
    mean_go_games = $null
    mean_go_rate = $null
    mean_go_fail_rate = $null
  }

  $generationMetricsPathRaw = [string]$Summary.generation_metrics_log
  if ([string]::IsNullOrWhiteSpace($generationMetricsPathRaw)) {
    return $empty
  }

  $generationMetricsPath = [System.IO.Path]::GetFullPath($generationMetricsPathRaw)
  if (-not (Test-Path $generationMetricsPath)) {
    return $empty
  }

  $lastGenerationLine = Get-Content $generationMetricsPath | Select-Object -Last 1
  if ([string]::IsNullOrWhiteSpace($lastGenerationLine)) {
    return $empty
  }

  $generationRecord = $lastGenerationLine | ConvertFrom-Json
  return [ordered]@{
    available = $true
    mean_go_games = Get-OptionalDouble -Value $generationRecord.mean_go_games
    mean_go_rate = Get-OptionalDouble -Value $generationRecord.mean_go_rate
    mean_go_fail_rate = Get-OptionalDouble -Value $generationRecord.mean_go_fail_rate
  }
}

function Get-OptionalDouble {
  param(
    [Parameter(Mandatory = $false)]$Value,
    [Parameter(Mandatory = $false)][double]$DefaultValue = [double]::NaN
  )
  if ($null -eq $Value) {
    return $DefaultValue
  }
  try {
    return [double]$Value
  }
  catch {
    return $DefaultValue
  }
}

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "python not found: $python"
}

$configFeedforward = "scripts/configs/neat_feedforward.ini"
$runtimeConfig = "scripts/configs/runtime_phase$Phase.json"
$outputDir = "logs/NEAT/neat_phase${Phase}_seed$Seed"

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

if ($Phase -ne "1") {
  $previousPhase = [int]$Phase - 1
  $previousLabel = "phase$previousPhase"
  $previousRuntimePath = "scripts/configs/runtime_phase$previousPhase.json"
  $previousRuntime = Read-JsonFile -Path $previousRuntimePath
  $previousGenerations = To-PositiveIntOrDefault -Value $previousRuntime.generations -DefaultValue 20
  $previousCheckpointDir = "logs/NEAT/neat_phase${previousPhase}_seed$Seed/checkpoints"
  $resume = Resolve-PreviousCheckpoint -CheckpointDir $previousCheckpointDir -PreferredGeneration $previousGenerations -Label $previousLabel

  $cmd += @(
    "--resume", "$($resume.path)",
    "--base-generation", "$($resume.generation)"
  )
}

$phaseRunStartedAt = Get-Date
$result = & $python @cmd | Out-String
$phaseRunElapsedSec = [math]::Round(((Get-Date) - $phaseRunStartedAt).TotalSeconds, 3)
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

$goMetrics = Get-BestGenomeGoMetrics -Summary $summary
$goAverages = Get-LatestGenerationGoAverages -Summary $summary
$summaryElapsedSec = Get-OptionalDouble -Value $summary.run_elapsed_sec -DefaultValue $phaseRunElapsedSec

Write-Host "=== Phase$Phase Summary (Seed=$Seed) ==="
Write-Host "EMA win rate:     $($summary.gate_state.ema_win_rate)"
Write-Host "EMA imitation:    $($summary.gate_state.ema_imitation)"
Write-Host "Latest win rate:  $($summary.gate_state.latest_win_rate)"
Write-Host "Win slope(5):     $($summary.gate_state.latest_win_rate_slope_5)"
Write-Host "Transition ready: $($summary.gate_state.transition_ready)"
Write-Host "Transition gen:   $($summary.gate_state.transition_generation)"
Write-Host "Best fitness:     $($summary.best_fitness)"
Write-Host "Elapsed time:     $summaryElapsedSec s"
if ([bool]$goAverages.available) {
  Write-Host "GO mean games:    $($goAverages.mean_go_games)"
  Write-Host "GO mean fail:     $($goAverages.mean_go_fail_rate)"
  Write-Host "GO mean rate:     $($goAverages.mean_go_rate)"
}
else {
  Write-Host "GO mean games:    N/A"
  Write-Host "GO mean fail:     N/A"
  Write-Host "GO mean rate:     N/A"
}
if ([bool]$goMetrics.available) {
  Write-Host "GO count:         $($goMetrics.go_count)"
  Write-Host "GO games:         $($goMetrics.go_games)"
  Write-Host "GO fail count:    $($goMetrics.go_fail_count)"
  Write-Host "GO fail rate:     $($goMetrics.go_fail_rate)"
  Write-Host "GO rate:          $($goMetrics.go_rate)"
  Write-Host "Luck proxy:       $($goMetrics.luck_proxy)"
  Write-Host "Luck seat gap:    $($goMetrics.luck_seat_win_rate_gap)"
  Write-Host "Luck volatility:  $($goMetrics.luck_gold_volatility_norm)"
}
else {
  Write-Host "GO/Luck metrics:  unavailable"
}
Write-Host "================================"

exit 0
