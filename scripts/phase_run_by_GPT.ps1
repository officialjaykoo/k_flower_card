#
# phase_run_by_GPT.ps1
# - Train NEAT by phase using GPT runtime configs.
# - Keep runtime logic unchanged; this file is comment/structure organized.
#

param(
  [Parameter(Mandatory = $true)][ValidateSet("1", "2", "3")][string]$Phase,
  [Parameter(Mandatory = $true)][int]$Seed
)

# ---------------------------------------------------------------------------
# Section 1) Strict mode and shared helpers
# ---------------------------------------------------------------------------
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
    return [ordered]@{ path = $preferred; generation = $PreferredGeneration }
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

function Get-BestGenomeGoMetrics {
  param([Parameter(Mandatory = $true)]$Summary)

  $empty = [ordered]@{
    available = $false
    fitness_model = $null
    fitness_profile = $null
    mean_gold_delta = $null
    p10_gold_delta = $null
    p50_gold_delta = $null
    gold_core = $null
    expected_result = $null
    bankrupt_rate = $null
    bankrupt_penalty = $null
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
    fitness_model = [string]$bestEvalRecord.fitness_model
    fitness_profile = [string]$bestEvalRecord.fitness_profile
    mean_gold_delta = [double]$bestEvalRecord.mean_gold_delta
    p10_gold_delta = [double]$bestEvalRecord.p10_gold_delta
    p50_gold_delta = [double]$bestEvalRecord.p50_gold_delta
    gold_core = [double]$bestEvalRecord.fitness_components.gold_core
    expected_result = [double]$bestEvalRecord.fitness_components.expected_result
    bankrupt_rate = [double]$bestEvalRecord.fitness_components.bankrupt_rate
    bankrupt_penalty = [double]$bestEvalRecord.fitness_components.bankrupt_penalty
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
    mean_gold_delta = $null
    mean_p10_gold_delta = $null
    mean_p50_gold_delta = $null
    mean_gold_core = $null
    mean_expected_result = $null
    mean_bankrupt_rate = $null
    mean_bankrupt_penalty = $null
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
    mean_gold_delta = Get-OptionalDouble -Value $generationRecord.mean_mean_gold_delta
    mean_p10_gold_delta = Get-OptionalDouble -Value $generationRecord.mean_p10_gold_delta
    mean_p50_gold_delta = Get-OptionalDouble -Value $generationRecord.mean_p50_gold_delta
    mean_gold_core = Get-OptionalDouble -Value $generationRecord.mean_gold_core
    mean_expected_result = Get-OptionalDouble -Value $generationRecord.mean_expected_result
    mean_bankrupt_rate = Get-OptionalDouble -Value $generationRecord.mean_bankrupt_rate
    mean_bankrupt_penalty = Get-OptionalDouble -Value $generationRecord.mean_bankrupt_penalty
    mean_go_games = Get-OptionalDouble -Value $generationRecord.mean_go_games
    mean_go_rate = Get-OptionalDouble -Value $generationRecord.mean_go_rate
    mean_go_fail_rate = Get-OptionalDouble -Value $generationRecord.mean_go_fail_rate
  }
}

# ---------------------------------------------------------------------------
# Section 2) Runtime paths and validation
# ---------------------------------------------------------------------------
$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "python not found: $python"
}

$configFeedforward = "scripts/configs/neat_feedforward.ini"
$runtimeConfig = "scripts/configs/runtime_phase${Phase}_by_GPT.json"
$outputDir = "logs/NEAT_GPT/neat_phase${Phase}_seed$Seed"

if (-not (Test-Path $configFeedforward)) {
  throw "config not found: $configFeedforward"
}
if (-not (Test-Path $runtimeConfig)) {
  throw "runtime config not found: $runtimeConfig"
}

# ---------------------------------------------------------------------------
# Section 3) Build training command
# ---------------------------------------------------------------------------
$cmd = @(
  "scripts/neat_train_worker_by_GPT.py",
  "--config-feedforward", $configFeedforward,
  "--runtime-config", $runtimeConfig,
  "--output-dir", $outputDir,
  "--seed", "$Seed",
  "--profile-name", "gpt_phase${Phase}_seed$Seed"
)

if ($Phase -ne "1") {
  $previousPhase = [int]$Phase - 1
  $previousLabel = "phase$previousPhase"
  $previousRuntimePath = "scripts/configs/runtime_phase${previousPhase}_by_GPT.json"
  $previousRuntime = Read-JsonFile -Path $previousRuntimePath
  $previousGenerations = To-PositiveIntOrDefault -Value $previousRuntime.generations -DefaultValue 20
  $previousCheckpointDir = "logs/NEAT_GPT/neat_phase${previousPhase}_seed$Seed/checkpoints"
  $resume = Resolve-PreviousCheckpoint -CheckpointDir $previousCheckpointDir -PreferredGeneration $previousGenerations -Label $previousLabel

  $cmd += @(
    "--resume", "$($resume.path)",
    "--base-generation", "$($resume.generation)"
  )
}

# ---------------------------------------------------------------------------
# Section 4) Execute training and parse summary
# ---------------------------------------------------------------------------
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
  throw "failed to parse neat_train_worker_by_GPT output as JSON"
}

# ---------------------------------------------------------------------------
# Section 5) Human-readable summary output
# ---------------------------------------------------------------------------
$goMetrics = Get-BestGenomeGoMetrics -Summary $summary
$goAverages = Get-LatestGenerationGoAverages -Summary $summary
$summaryElapsedSec = Get-OptionalDouble -Value $summary.run_elapsed_sec -DefaultValue $phaseRunElapsedSec

Write-Host "=== Phase$Phase GPT Summary (Seed=$Seed) ==="
Write-Host "EMA win rate:     $($summary.gate_state.ema_win_rate)"
Write-Host "EMA imitation:    $($summary.gate_state.ema_imitation)"
Write-Host "Latest win rate:  $($summary.gate_state.latest_win_rate)"
Write-Host "Win slope(5):     $($summary.gate_state.latest_win_rate_slope_5)"
Write-Host "Transition ready: $($summary.gate_state.transition_ready)"
Write-Host "Transition gen:   $($summary.gate_state.transition_generation)"
Write-Host "Best fitness:     $($summary.best_fitness)"
Write-Host "Elapsed time:     $summaryElapsedSec s"
if ([bool]$goAverages.available) {
  Write-Host "Mean gold delta:  $($goAverages.mean_gold_delta)"
  Write-Host "Mean p10 delta:   $($goAverages.mean_p10_gold_delta)"
  Write-Host "Mean p50 delta:   $($goAverages.mean_p50_gold_delta)"
  Write-Host "Mean gold core:   $($goAverages.mean_gold_core)"
  Write-Host "Mean exp result:  $($goAverages.mean_expected_result)"
  Write-Host "Mean bankrupt:    $($goAverages.mean_bankrupt_rate)"
  Write-Host "Mean bk penalty:  $($goAverages.mean_bankrupt_penalty)"
  Write-Host "GO mean games:    $($goAverages.mean_go_games)"
  Write-Host "GO mean fail:     $($goAverages.mean_go_fail_rate)"
  Write-Host "GO mean rate:     $($goAverages.mean_go_rate)"
}
else {
  Write-Host "Mean gold delta:  N/A"
  Write-Host "Mean p10 delta:   N/A"
  Write-Host "Mean p50 delta:   N/A"
  Write-Host "Mean gold core:   N/A"
  Write-Host "Mean exp result:  N/A"
  Write-Host "Mean bankrupt:    N/A"
  Write-Host "Mean bk penalty:  N/A"
  Write-Host "GO mean games:    N/A"
  Write-Host "GO mean fail:     N/A"
  Write-Host "GO mean rate:     N/A"
}
if ([bool]$goMetrics.available) {
  Write-Host "Fitness model:    $($goMetrics.fitness_model)"
  Write-Host "Fitness profile:  $($goMetrics.fitness_profile)"
  Write-Host "Best gold delta:  $($goMetrics.mean_gold_delta)"
  Write-Host "Best p10 delta:   $($goMetrics.p10_gold_delta)"
  Write-Host "Best p50 delta:   $($goMetrics.p50_gold_delta)"
  Write-Host "Best gold core:   $($goMetrics.gold_core)"
  Write-Host "Best exp result:  $($goMetrics.expected_result)"
  Write-Host "Best bankrupt:    $($goMetrics.bankrupt_rate)"
  Write-Host "Best bk penalty:  $($goMetrics.bankrupt_penalty)"
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
