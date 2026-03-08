param(
  [Parameter(Mandatory = $true)][int]$Seed,
  [string]$Phase = "",
  [string]$RuntimeConfig = "",
  [string]$OutputDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$CommonScript = Join-Path $PSScriptRoot "common.ps1"
if (-not (Test-Path $CommonScript)) {
  throw "common helper not found: $CommonScript"
}
. $CommonScript

$RepoRoot = Get-NeatGptRepoRoot -ScriptRoot $PSScriptRoot
$ConfigFeedforward = Join-Path $PSScriptRoot "configs\neat_feedforward.ini"
$TrainWorker = Join-Path $PSScriptRoot "scripts\neat_train_worker.py"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

function Read-JsonFile {
  param([Parameter(Mandatory = $true)][string]$Path)
  return Read-NeatGptRuntimeJson -Path $Path
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

function To-PositiveIntOrDefault {
  param(
    [Parameter(Mandatory = $false)]$Value,
    [Parameter(Mandatory = $true)][int]$DefaultValue
  )
  try {
    $n = [int]$Value
    if ($n -gt 0) {
      return $n
    }
  }
  catch {
  }
  return $DefaultValue
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

function Get-LastNdjsonRecord {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path $Path)) {
    return $null
  }
  $line = Get-Content $Path | Select-Object -Last 1
  if ([string]::IsNullOrWhiteSpace($line)) {
    return $null
  }
  return ($line | ConvertFrom-Json)
}

function Get-BestEvalRecord {
  param([Parameter(Mandatory = $true)]$Summary)

  $generationMetricsPath = [string](Get-PropertyValue -Object $Summary -Name "generation_metrics_log")
  $evalMetricsPath = [string](Get-PropertyValue -Object $Summary -Name "eval_metrics_log")
  if ([string]::IsNullOrWhiteSpace($generationMetricsPath) -or [string]::IsNullOrWhiteSpace($evalMetricsPath)) {
    return $null
  }
  if (-not (Test-Path $generationMetricsPath) -or -not (Test-Path $evalMetricsPath)) {
    return $null
  }

  $generationRecord = Get-LastNdjsonRecord -Path $generationMetricsPath
  if ($null -eq $generationRecord) {
    return $null
  }

  $targetGeneration = [int]$generationRecord.generation
  $targetGenomeKey = [int]$generationRecord.best_genome_key
  return (
    Get-Content $evalMetricsPath |
      ForEach-Object { $_ | ConvertFrom-Json } |
      Where-Object { [int]$_.generation -eq $targetGeneration -and [int]$_.genome_key -eq $targetGenomeKey } |
      Select-Object -First 1
  )
}

function Assert-RuntimeKeys {
  param([Parameter(Mandatory = $true)]$Runtime)
  $requiredKeys = @(
    "format_version",
    "seed",
    "eval_script",
    "teacher_policy",
    "fitness_profile",
    "generations",
    "eval_workers",
    "games_per_genome",
    "eval_timeout_sec",
    "max_eval_steps",
    "opponent_policy",
    "opponent_policy_mix",
    "opponent_genome",
    "switch_seats",
    "checkpoint_every",
    "selection_eval_games",
    "selection_top_k",
    "selection_opponent_policy",
    "selection_opponent_policy_mix",
    "selection_opponent_genome",
    "gate_mode",
    "gate_ema_window",
    "transition_ema_win_rate",
    "transition_mean_gold_delta_min",
    "transition_cvar10_gold_delta_min",
    "transition_catastrophic_loss_rate_max",
    "transition_best_fitness_min",
    "transition_streak",
    "failure_generation_min",
    "failure_ema_win_rate_max",
    "failure_mean_gold_delta_max",
    "failure_cvar10_gold_delta_max",
    "failure_catastrophic_loss_rate_min",
    "failure_slope_5_max",
    "failure_slope_metric"
  )
  foreach ($key in $requiredKeys) {
    if (-not ($Runtime.PSObject.Properties.Name -contains $key)) {
      throw "runtime missing required key: $key"
    }
  }
}

$RuntimeConfig = Resolve-NeatGptRuntimeConfigPath -ScriptRoot $PSScriptRoot -RepoRoot $RepoRoot -RuntimeConfig $RuntimeConfig -Phase $Phase
$resolvedOutputDir = Resolve-NeatGptOutputDirPath -RepoRoot $RepoRoot -RuntimeConfigPath $RuntimeConfig -SeedValue $Seed -ExplicitOutputDir $OutputDir

if (-not (Test-Path $Python)) {
  throw "python not found: $Python"
}
if (-not (Test-Path $ConfigFeedforward)) {
  throw "config not found: $ConfigFeedforward"
}
if (-not (Test-Path $TrainWorker)) {
  throw "train worker not found: $TrainWorker"
}
if (-not (Test-Path $RuntimeConfig)) {
  throw "runtime config not found: $RuntimeConfig"
}

$runtime = Read-JsonFile -Path $RuntimeConfig
Assert-RuntimeKeys -Runtime $runtime
$profileName = [System.IO.Path]::GetFileNameWithoutExtension($RuntimeConfig)
$phaseSeedState = Resolve-NeatGptPhaseSeedState -ScriptRoot $PSScriptRoot -RepoRoot $RepoRoot -Phase $Phase -SeedValue $Seed
if ($null -ne $phaseSeedState) {
  $summaryPath = [string]$phaseSeedState.summary_path
  if (-not (Test-Path $summaryPath) -and (Test-Path $phaseSeedState.legacy_summary_path)) {
    $summaryPath = [string]$phaseSeedState.legacy_summary_path
  }
  if (-not (Test-Path $summaryPath)) {
    throw "$($phaseSeedState.previous_label) run summary not found: $($phaseSeedState.summary_path)"
  }
  $previousSummary = Read-JsonFile -Path $summaryPath
  $previousRuntime = Read-JsonFile -Path $phaseSeedState.runtime_config
  $previousAppliedOverrides = Get-PropertyValue -Object $previousSummary -Name "applied_overrides"
  $previousBaseGeneration = 0
  if ($null -ne $previousAppliedOverrides) {
    $previousBaseGeneration = To-PositiveIntOrDefault -Value (Get-PropertyValue -Object $previousAppliedOverrides -Name "base_generation") -DefaultValue 0
  }
  $previousGenerations = To-PositiveIntOrDefault -Value $previousSummary.generations -DefaultValue (
    To-PositiveIntOrDefault -Value $previousRuntime.generations -DefaultValue 20
  )
  $cumulativeBaseGeneration = $previousBaseGeneration + $previousGenerations
  $previousWinnerRaw = [string](Get-PropertyValue -Object $previousSummary -Name "winner_pickle")
  if ([string]::IsNullOrWhiteSpace($previousWinnerRaw)) {
    if ($summaryPath -eq [string]$phaseSeedState.legacy_summary_path) {
      $previousWinnerRaw = [string]$phaseSeedState.legacy_fallback_winner_path
    }
    else {
      $previousWinnerRaw = [string]$phaseSeedState.fallback_winner_path
    }
  }
  $previousWinnerPath = Resolve-PathFromBase -Path $previousWinnerRaw -BasePath $RepoRoot
  if (-not (Test-Path $previousWinnerPath)) {
    throw "$($phaseSeedState.previous_label) winner genome not found: $previousWinnerPath"
  }
  $phaseSeedState | Add-Member -NotePropertyName summary_path_resolved -NotePropertyValue $summaryPath -Force
  $phaseSeedState | Add-Member -NotePropertyName winner_path -NotePropertyValue $previousWinnerPath -Force
  $phaseSeedState | Add-Member -NotePropertyName base_generation -NotePropertyValue $cumulativeBaseGeneration -Force
}
$cmd = @(
  $TrainWorker,
  "--config-feedforward", $ConfigFeedforward,
  "--runtime-config", $RuntimeConfig,
  "--output-dir", $resolvedOutputDir,
  "--seed", "$Seed",
  "--profile-name", "${profileName}_seed$Seed"
)
if ($null -ne $phaseSeedState) {
  $cmd += @(
    "--seed-genome", "$($phaseSeedState.winner_path)",
    "--base-generation", "$($phaseSeedState.base_generation)"
  )
}

$resultText = & $Python @cmd | Out-String
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
  if (-not [string]::IsNullOrWhiteSpace($resultText)) {
    Write-Host $resultText.Trim()
  }
  exit $exitCode
}

try {
  $summary = $resultText | ConvertFrom-Json
}
catch {
  if (-not [string]::IsNullOrWhiteSpace($resultText)) {
    Write-Host $resultText.Trim()
  }
  throw "failed to parse neat_train_worker output as JSON"
}

$generationRecord = Get-LastNdjsonRecord -Path ([string]$summary.generation_metrics_log)
$latestTrain = Get-PropertyValue -Object $summary -Name "latest_train_best"
$latestTarget = Get-PropertyValue -Object $summary -Name "latest_target_best"
$bestTarget = Get-PropertyValue -Object $summary -Name "best_target"

Write-Host "=== NEAT GPT Run Summary ==="
Write-Host "Runtime:          $RuntimeConfig"
Write-Host "Output:           $resolvedOutputDir"
if ($null -ne $phaseSeedState) {
  Write-Host "Seed genome:      $($phaseSeedState.winner_path)"
  Write-Host "Base generation:  $($phaseSeedState.base_generation)"
  Write-Host "Seed summary:     $($phaseSeedState.summary_path_resolved)"
}
Write-Host "Winner basis:     $($summary.winner_model_basis)"
Write-Host "Best fitness:     $($summary.best_fitness)"
Write-Host "Train fitness:    $($summary.train_best_fitness)"
Write-Host "Elapsed time:     $($summary.run_elapsed_sec) s"
Write-Host "EMA win rate:     $($summary.gate_state.ema_win_rate)"
Write-Host "Transition ready: $($summary.gate_state.transition_ready)"
Write-Host "Transition gen:   $($summary.gate_state.transition_generation)"
if ($null -ne $generationRecord) {
  Write-Host "Mean gold delta:  $($generationRecord.mean_mean_gold_delta)"
  Write-Host "Mean p10 delta:   $($generationRecord.mean_p10_gold_delta)"
  Write-Host "Mean CVaR10:      $($generationRecord.mean_cvar10_gold_delta)"
  Write-Host "Mean cat loss:    $($generationRecord.mean_catastrophic_loss_rate)"
  Write-Host "Mean GO games:    $($generationRecord.mean_go_games)"
  Write-Host "Mean GO fail:     $($generationRecord.mean_go_fail_rate)"
}
if ($null -ne $latestTrain) {
  Write-Host "Latest train WR:  $($latestTrain.win_rate)"
  Write-Host "Latest train GD:  $($latestTrain.mean_gold_delta)"
  Write-Host "Latest train CVaR:$($latestTrain.cvar10_gold_delta)"
}
if ($null -ne $latestTarget) {
  Write-Host "Latest target WR: $($latestTarget.win_rate)"
  Write-Host "Latest target GD: $($latestTarget.mean_gold_delta)"
  Write-Host "Latest target CVaR: $($latestTarget.cvar10_gold_delta)"
  Write-Host "Latest target CL: $($latestTarget.catastrophic_loss_rate)"
}
if ($null -ne $bestTarget) {
  Write-Host "Best target WR:   $($bestTarget.win_rate)"
  Write-Host "Best target GD:   $($bestTarget.mean_gold_delta)"
  Write-Host "Best target CVaR: $($bestTarget.cvar10_gold_delta)"
  Write-Host "Best target CL:   $($bestTarget.catastrophic_loss_rate)"
}
Write-Host "============================"
