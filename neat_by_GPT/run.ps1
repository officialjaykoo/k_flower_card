param(
  [Parameter(Mandatory = $true)][int]$Seed,
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
$DefaultRuntimeConfig = Join-Path $PSScriptRoot "configs\runtime_focus_cl_v1.json"
$ConfigFeedforward = Join-Path $PSScriptRoot "configs\neat_feedforward.ini"
$TrainWorker = Join-Path $PSScriptRoot "scripts\neat_train_worker.py"
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"

function Read-JsonFile {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path $Path)) {
    throw "json file not found: $Path"
  }
  return Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json
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

$RuntimeConfig = Resolve-PathFromBase -Path $RuntimeConfig -BasePath $RepoRoot -DefaultPath $DefaultRuntimeConfig
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
$cmd = @(
  $TrainWorker,
  "--config-feedforward", $ConfigFeedforward,
  "--runtime-config", $RuntimeConfig,
  "--output-dir", $resolvedOutputDir,
  "--seed", "$Seed",
  "--profile-name", "${profileName}_seed$Seed"
)

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
