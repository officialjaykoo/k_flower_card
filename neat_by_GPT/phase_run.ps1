#
# phase_run.ps1
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

function To-BoolOrDefault {
  param(
    [Parameter(Mandatory = $false)]$Value,
    [Parameter(Mandatory = $true)][bool]$DefaultValue
  )
  if ($null -eq $Value) {
    return $DefaultValue
  }
  $s = [string]$Value
  if ([string]::IsNullOrWhiteSpace($s)) {
    return $DefaultValue
  }
  switch ($s.Trim().ToLowerInvariant()) {
    "1" { return $true }
    "true" { return $true }
    "yes" { return $true }
    "on" { return $true }
    "0" { return $false }
    "false" { return $false }
    "no" { return $false }
    "off" { return $false }
    default { return $DefaultValue }
  }
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

function Get-PropertyValue {
  param(
    [Parameter(Mandatory = $false)]$Object,
    [Parameter(Mandatory = $true)][string]$Name
  )
  if ($null -eq $Object) {
    return $null
  }
  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.$Name
  }
  return $null
}

function Invoke-TrainingRun {
  param(
    [Parameter(Mandatory = $true)][string]$Python,
    [Parameter(Mandatory = $true)][string[]]$Arguments
  )
  $result = & $Python @Arguments | Out-String
  $exitCode = $LASTEXITCODE
  if ($exitCode -ne 0) {
    if (-not [string]::IsNullOrWhiteSpace($result)) {
      Write-Host $result.Trim()
    }
    exit $exitCode
  }
  try {
    return ($result | ConvertFrom-Json)
  }
  catch {
    if (-not [string]::IsNullOrWhiteSpace($result)) {
      Write-Host $result.Trim()
    }
    throw "failed to parse neat_train_worker output as JSON"
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

  $generationMetricsPathRaw = [string](Get-PropertyValue -Object $Summary -Name "generation_metrics_log")
  $evalMetricsPathRaw = [string](Get-PropertyValue -Object $Summary -Name "eval_metrics_log")
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

  $fitnessComponents = Get-PropertyValue -Object $bestEvalRecord -Name "fitness_components"
  $luckComponents = Get-PropertyValue -Object $bestEvalRecord -Name "luck_components"

  return [ordered]@{
    available = $true
    fitness_model = [string](Get-PropertyValue -Object $bestEvalRecord -Name "fitness_model")
    fitness_profile = [string](Get-PropertyValue -Object $bestEvalRecord -Name "fitness_profile")
    mean_gold_delta = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "mean_gold_delta")
    p10_gold_delta = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "p10_gold_delta")
    p50_gold_delta = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "p50_gold_delta")
    gold_core = Get-OptionalDouble -Value (Get-PropertyValue -Object $fitnessComponents -Name "gold_core")
    expected_result = Get-OptionalDouble -Value (Get-PropertyValue -Object $fitnessComponents -Name "expected_result")
    bankrupt_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $fitnessComponents -Name "bankrupt_rate")
    bankrupt_penalty = Get-OptionalDouble -Value (Get-PropertyValue -Object $fitnessComponents -Name "bankrupt_penalty")
    go_count = [int](Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "go_count") -DefaultValue 0.0)
    go_games = [int](Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "go_games") -DefaultValue 0.0)
    go_fail_count = [int](Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "go_fail_count") -DefaultValue 0.0)
    go_fail_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "go_fail_rate")
    go_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "go_rate")
    luck_proxy = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "luck_proxy")
    luck_seat_win_rate_gap = Get-OptionalDouble -Value (Get-PropertyValue -Object $luckComponents -Name "seat_win_rate_gap")
    luck_gold_volatility_norm = Get-OptionalDouble -Value (Get-PropertyValue -Object $luckComponents -Name "gold_volatility_norm")
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

  $generationMetricsPathRaw = [string](Get-PropertyValue -Object $Summary -Name "generation_metrics_log")
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
  $meanMeanGoldDelta = Get-PropertyValue -Object $generationRecord -Name "mean_mean_gold_delta"
  $meanP10GoldDelta = Get-PropertyValue -Object $generationRecord -Name "mean_p10_gold_delta"
  $meanP50GoldDelta = Get-PropertyValue -Object $generationRecord -Name "mean_p50_gold_delta"
  $meanGoldCore = Get-PropertyValue -Object $generationRecord -Name "mean_gold_core"
  $meanExpectedResult = Get-PropertyValue -Object $generationRecord -Name "mean_expected_result"
  $meanBankruptRate = Get-PropertyValue -Object $generationRecord -Name "mean_bankrupt_rate"
  $meanBankruptPenalty = Get-PropertyValue -Object $generationRecord -Name "mean_bankrupt_penalty"
  $meanGoGames = Get-PropertyValue -Object $generationRecord -Name "mean_go_games"
  $meanGoRate = Get-PropertyValue -Object $generationRecord -Name "mean_go_rate"
  $meanGoFailRate = Get-PropertyValue -Object $generationRecord -Name "mean_go_fail_rate"
  return [ordered]@{
    available = $true
    mean_gold_delta = Get-OptionalDouble -Value $meanMeanGoldDelta
    mean_p10_gold_delta = Get-OptionalDouble -Value $meanP10GoldDelta
    mean_p50_gold_delta = Get-OptionalDouble -Value $meanP50GoldDelta
    mean_gold_core = Get-OptionalDouble -Value $meanGoldCore
    mean_expected_result = Get-OptionalDouble -Value $meanExpectedResult
    mean_bankrupt_rate = Get-OptionalDouble -Value $meanBankruptRate
    mean_bankrupt_penalty = Get-OptionalDouble -Value $meanBankruptPenalty
    mean_go_games = Get-OptionalDouble -Value $meanGoGames
    mean_go_rate = Get-OptionalDouble -Value $meanGoRate
    mean_go_fail_rate = Get-OptionalDouble -Value $meanGoFailRate
  }
}

# ---------------------------------------------------------------------------
# Section 2) Runtime paths and validation
# ---------------------------------------------------------------------------
$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "python not found: $python"
}

$configFeedforward = "neat_by_GPT/configs/neat_feedforward.ini"
$runtimeConfig = "neat_by_GPT/configs/runtime_phase${Phase}.json"
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
$baseCmd = @(
  "neat_by_GPT/scripts/neat_train_worker.py",
  "--config-feedforward", $configFeedforward,
  "--runtime-config", $runtimeConfig,
  "--output-dir", $outputDir,
  "--seed", "$Seed",
  "--profile-name", "gpt_phase${Phase}_seed$Seed"
)

$phaseRunStartedAt = Get-Date
$summary = $null
$phase1StageASummary = $null

if ($Phase -eq "1") {
  $phase1Runtime = Read-JsonFile -Path $runtimeConfig
  $phase1Curriculum = Get-PropertyValue -Object $phase1Runtime -Name "phase1_curriculum"
  $phase1CurriculumEnabled = To-BoolOrDefault -Value (Get-PropertyValue -Object $phase1Curriculum -Name "enabled") -DefaultValue $false

  if ($phase1CurriculumEnabled) {
    $phase1Generations = To-PositiveIntOrDefault -Value (Get-PropertyValue -Object $phase1Runtime -Name "generations") -DefaultValue 20
    $phase1StageAGenerations = To-PositiveIntOrDefault -Value (Get-PropertyValue -Object $phase1Curriculum -Name "stage_a_generations") -DefaultValue 8
    if ($phase1StageAGenerations -ge $phase1Generations) {
      throw "invalid phase1_curriculum: stage_a_generations ($phase1StageAGenerations) must be less than generations ($phase1Generations)"
    }
    $phase1StageAOpponentPolicy = [string](Get-PropertyValue -Object $phase1Curriculum -Name "stage_a_opponent_policy")
    if ([string]::IsNullOrWhiteSpace($phase1StageAOpponentPolicy)) {
      $phase1StageAOpponentPolicy = "H-Gemini"
    }
    $phase1StageBGenerations = $phase1Generations - $phase1StageAGenerations

    Write-Host "Phase1 curriculum enabled: stageA=$phase1StageAGenerations (opponent=$phase1StageAOpponentPolicy), stageB=$phase1StageBGenerations (runtime mix)"

    $phase1StageACmd = $baseCmd + @(
      "--generations", "$phase1StageAGenerations",
      "--opponent-policy", "$phase1StageAOpponentPolicy",
      "--profile-name", "gpt_phase1a_seed$Seed"
    )
    $phase1StageASummary = Invoke-TrainingRun -Python $python -Arguments $phase1StageACmd

    $phase1CheckpointDir = Join-Path $outputDir "checkpoints"
    $phase1Resume = Resolve-PreviousCheckpoint -CheckpointDir $phase1CheckpointDir -PreferredGeneration $phase1StageAGenerations -Label "phase1-stageA"

    $phase1StageBCmd = $baseCmd + @(
      "--generations", "$phase1StageBGenerations",
      "--resume", "$($phase1Resume.path)",
      "--base-generation", "$($phase1Resume.generation)",
      "--profile-name", "gpt_phase1b_seed$Seed"
    )
    $summary = Invoke-TrainingRun -Python $python -Arguments $phase1StageBCmd
  }
  else {
    $summary = Invoke-TrainingRun -Python $python -Arguments $baseCmd
  }
}
else {
  $previousPhase = [int]$Phase - 1
  $previousLabel = "phase$previousPhase"
  $previousRuntimePath = "neat_by_GPT/configs/runtime_phase${previousPhase}.json"
  $previousRuntime = Read-JsonFile -Path $previousRuntimePath
  $previousGenerations = To-PositiveIntOrDefault -Value $previousRuntime.generations -DefaultValue 20
  $previousCheckpointDir = "logs/NEAT_GPT/neat_phase${previousPhase}_seed$Seed/checkpoints"
  $resume = Resolve-PreviousCheckpoint -CheckpointDir $previousCheckpointDir -PreferredGeneration $previousGenerations -Label $previousLabel
  $cmd = $baseCmd + @(
    "--resume", "$($resume.path)",
    "--base-generation", "$($resume.generation)"
  )
  $summary = Invoke-TrainingRun -Python $python -Arguments $cmd
}

# ---------------------------------------------------------------------------
# Section 4) Execute training and parse summary
# ---------------------------------------------------------------------------
$phaseRunElapsedSec = [math]::Round(((Get-Date) - $phaseRunStartedAt).TotalSeconds, 3)
if ($null -eq $summary) {
  throw "phase run summary is null"
}

# ---------------------------------------------------------------------------
# Section 5) Human-readable summary output
# ---------------------------------------------------------------------------
$goMetrics = Get-BestGenomeGoMetrics -Summary $summary
$goAverages = Get-LatestGenerationGoAverages -Summary $summary
$summaryElapsedSec = Get-OptionalDouble -Value $summary.run_elapsed_sec -DefaultValue $phaseRunElapsedSec
if ($null -ne $phase1StageASummary) {
  $summaryElapsedSec = $phaseRunElapsedSec
}

Write-Host "=== Phase$Phase GPT Summary (Seed=$Seed) ==="
if ($null -ne $phase1StageASummary) {
  Write-Host "StageA latest win: $($phase1StageASummary.gate_state.latest_win_rate)"
  Write-Host "StageA best fit:   $($phase1StageASummary.best_fitness)"
  }
Write-Host "EMA win rate:     $($summary.gate_state.ema_win_rate)"
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
  Write-Host "Mean bankrupt:    $($goAverages.mean_bankrupt_rate)"
  Write-Host "GO mean games:    $($goAverages.mean_go_games)"
  Write-Host "GO mean fail:     $($goAverages.mean_go_fail_rate)"
  Write-Host "GO mean rate:     $($goAverages.mean_go_rate)"
}
else {
  Write-Host "Mean gold delta:  N/A"
  Write-Host "Mean p10 delta:   N/A"
  Write-Host "Mean p50 delta:   N/A"
  Write-Host "Mean gold core:   N/A"
  Write-Host "Mean bankrupt:    N/A"
  Write-Host "GO mean games:    N/A"
  Write-Host "GO mean fail:     N/A"
  Write-Host "GO mean rate:     N/A"
}
if ([bool]$goMetrics.available) {
  Write-Host "Fitness model:    $($goMetrics.fitness_model)"
  Write-Host "Best gold delta:  $($goMetrics.mean_gold_delta)"
  Write-Host "Best p10 delta:   $($goMetrics.p10_gold_delta)"
  Write-Host "Best p50 delta:   $($goMetrics.p50_gold_delta)"
  Write-Host "Best gold core:   $($goMetrics.gold_core)"
  Write-Host "Best bankrupt:    $($goMetrics.bankrupt_rate)"
  Write-Host "GO count:         $($goMetrics.go_count)"
  Write-Host "GO games:         $($goMetrics.go_games)"
  Write-Host "GO fail count:    $($goMetrics.go_fail_count)"
  Write-Host "GO fail rate:     $($goMetrics.go_fail_rate)"
  Write-Host "GO rate:          $($goMetrics.go_rate)"
}
else {
  Write-Host "GO/Luck metrics:  unavailable"
}
Write-Host "================================"

exit 0
