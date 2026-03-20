param(
  [Parameter(Mandatory = $true)][ValidateSet("1", "2", "3")][string]$Phase,
  [Parameter(Mandatory = $true)][int]$Seed,
  [Parameter(Mandatory = $false)][ValidateSet("classic")][string]$LineageProfile = "classic",
  [Parameter(Mandatory = $false)][ValidateSet("race2", "race5", "race6", "race7", "race8", "race20", "hand7", "hand10", "material10", "race10", "oracle10", "oracle10v2")][string]$FeatureProfile = "",
  [Parameter(Mandatory = $false)][string[]]$BootstrapSeedSpec = @()
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

function To-BoolOrDefault {
  param(
    [Parameter(Mandatory = $false)]$Value,
    [Parameter(Mandatory = $true)][bool]$DefaultValue
  )
  if ($null -eq $Value) {
    return $DefaultValue
  }
  if ($Value -is [bool]) {
    return [bool]$Value
  }
  $text = [string]$Value
  if ([string]::IsNullOrWhiteSpace($text)) {
    return $DefaultValue
  }
  switch ($text.Trim().ToLowerInvariant()) {
    "1" { return $true }
    "true" { return $true }
    "yes" { return $true }
    "y" { return $true }
    "on" { return $true }
    "0" { return $false }
    "false" { return $false }
    "no" { return $false }
    "n" { return $false }
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

function Get-OverallBestMetrics {
  param([Parameter(Mandatory = $true)]$Summary)

  $empty = [ordered]@{
    available = $false
    generation = $null
    genome_key = $null
    fitness = $null
    win_rate = $null
    imitation_weighted_score = $null
    mean_gold_delta = $null
    go_opportunity_count = $null
    go_opportunity_games = $null
    go_opportunity_rate = $null
    go_take_rate = $null
    go_count = $null
    go_games = $null
    go_fail_count = $null
    go_fail_rate = $null
    go_rate = $null
    generation_mean_go_games = $null
    generation_mean_go_rate = $null
    generation_mean_go_fail_rate = $null
  }

  $bestRecord = Get-PropertyValue -Object $Summary -Name "best_record"
  if ($null -ne $bestRecord) {
    $bestRecordFitness = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestRecord -Name "fitness")
    if (-not [double]::IsNaN($bestRecordFitness)) {
      return [ordered]@{
        available = $true
        generation = Get-PropertyValue -Object $bestRecord -Name "generation"
        genome_key = Get-PropertyValue -Object $bestRecord -Name "genome_key"
        fitness = $bestRecordFitness
        win_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestRecord -Name "win_rate")
        imitation_weighted_score = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestRecord -Name "imitation_weighted_score")
        mean_gold_delta = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestRecord -Name "mean_gold_delta")
        go_opportunity_count = Get-PropertyValue -Object $bestRecord -Name "go_opportunity_count"
        go_opportunity_games = Get-PropertyValue -Object $bestRecord -Name "go_opportunity_games"
        go_opportunity_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestRecord -Name "go_opportunity_rate")
        go_take_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestRecord -Name "go_take_rate")
        go_count = Get-PropertyValue -Object $bestRecord -Name "go_count"
        go_games = Get-PropertyValue -Object $bestRecord -Name "go_games"
        go_fail_count = Get-PropertyValue -Object $bestRecord -Name "go_fail_count"
        go_fail_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestRecord -Name "go_fail_rate")
        go_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestRecord -Name "go_rate")
        generation_mean_go_games = $null
        generation_mean_go_rate = $null
        generation_mean_go_fail_rate = $null
      }
    }
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

  $generationRecord = $null
  $targetGeneration = $null
  $targetGenomeKey = $null

  if ($null -ne $bestRecord) {
    $bestRecordGeneration = Get-PropertyValue -Object $bestRecord -Name "generation"
    $bestRecordGenomeKey = Get-PropertyValue -Object $bestRecord -Name "genome_key"
    if ($null -ne $bestRecordGeneration -and $null -ne $bestRecordGenomeKey) {
      $targetGeneration = [int]$bestRecordGeneration
      $targetGenomeKey = [int]$bestRecordGenomeKey
      $generationRecord =
        Get-Content $generationMetricsPath |
          ForEach-Object { $_ | ConvertFrom-Json } |
          Where-Object { [int]$_.generation -eq $targetGeneration } |
          Select-Object -First 1
    }
  }

  if ($null -eq $generationRecord) {
    $generationRecord =
      Get-Content $generationMetricsPath |
        ForEach-Object { $_ | ConvertFrom-Json } |
        Sort-Object { [double](Get-PropertyValue -Object $_ -Name "best_fitness") } -Descending |
        Select-Object -First 1
    if ($null -eq $generationRecord) {
      return $empty
    }
    $targetGeneration = [int](Get-PropertyValue -Object $generationRecord -Name "generation")
    $targetGenomeKey = [int](Get-PropertyValue -Object $generationRecord -Name "best_genome_key")
  }

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
    generation = [int]$targetGeneration
    genome_key = [int]$targetGenomeKey
    fitness = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "fitness")
    win_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "win_rate")
    imitation_weighted_score = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "imitation_weighted_score")
    mean_gold_delta = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "mean_gold_delta")
    go_opportunity_count = [int](Get-PropertyValue -Object $bestEvalRecord -Name "go_opportunity_count")
    go_opportunity_games = [int](Get-PropertyValue -Object $bestEvalRecord -Name "go_opportunity_games")
    go_opportunity_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "go_opportunity_rate")
    go_take_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "go_take_rate")
    go_count = [int](Get-PropertyValue -Object $bestEvalRecord -Name "go_count")
    go_games = [int](Get-PropertyValue -Object $bestEvalRecord -Name "go_games")
    go_fail_count = [int](Get-PropertyValue -Object $bestEvalRecord -Name "go_fail_count")
    go_fail_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "go_fail_rate")
    go_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $bestEvalRecord -Name "go_rate")
    generation_mean_go_games = Get-OptionalDouble -Value (Get-PropertyValue -Object $generationRecord -Name "mean_go_games")
    generation_mean_go_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $generationRecord -Name "mean_go_rate")
    generation_mean_go_fail_rate = Get-OptionalDouble -Value (Get-PropertyValue -Object $generationRecord -Name "mean_go_fail_rate")
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

function Get-NestedPropertyValue {
  param(
    [Parameter(Mandatory = $false)]$Object,
    [Parameter(Mandatory = $true)][string[]]$Path
  )
  $current = $Object
  foreach ($segment in $Path) {
    $current = Get-PropertyValue -Object $current -Name $segment
    if ($null -eq $current) {
      return $null
    }
  }
  return $current
}

function Resolve-RunSummaryObject {
  param([Parameter(Mandatory = $true)]$Summary)

  $runSummaryPathRaw = [string](Get-PropertyValue -Object $Summary -Name "run_summary")
  if ([string]::IsNullOrWhiteSpace($runSummaryPathRaw)) {
    return $Summary
  }

  $runSummaryPath = [System.IO.Path]::GetFullPath($runSummaryPathRaw)
  if (-not (Test-Path $runSummaryPath)) {
    return $Summary
  }

  try {
    return Read-JsonFile -Path $runSummaryPath
  }
  catch {
    return $Summary
  }
}

function Get-LineageLayout {
  param([Parameter(Mandatory = $true)][string]$Profile)

  switch ($Profile) {
    "classic" {
      return [ordered]@{
        profile = "classic"
        token_prefix = ""
        output_prefix = "neat"
        config_feedforward = "scripts/configs/neat_feedforward.ini"
        profile_name_prefix = ""
      }
    }
    default {
      throw "unsupported lineage profile: $Profile"
    }
  }
}

function Resolve-BootstrapWinnerPath {
  param([Parameter(Mandatory = $true)][string]$Spec)

  $raw = [string]$Spec
  if ([string]::IsNullOrWhiteSpace($raw)) {
    throw "bootstrap seed spec is empty"
  }

  $m = [regex]::Match($raw.Trim(), "^phase([123])_seed(\d+)$", [System.Text.RegularExpressions.RegexOptions]::IgnoreCase)
  if (-not $m.Success) {
    throw "invalid bootstrap seed spec: $Spec (use phase1_seed203)"
  }
  $bootstrapLayout = Get-LineageLayout -Profile "classic"
  $bootstrapPhase = [int]$m.Groups[1].Value
  $bootstrapSeed = [int]$m.Groups[2].Value
  $summaryPath = "logs/NEAT/$($bootstrapLayout.output_prefix)_phase${bootstrapPhase}_seed$bootstrapSeed/run_summary.json"
  $winnerPath = ""
  if (Test-Path $summaryPath) {
    $summary = Read-JsonFile -Path $summaryPath
    $repairStatus = [string](Get-PropertyValue -Object $summary -Name "winner_repair_status")
    if ($repairStatus -eq "summary_repaired_winner_unrecoverable") {
      throw "bootstrap winner is unrecoverable for $Spec; run_summary was repaired but exact winner genome could not be restored"
    }
    $winnerPath = [string](Get-PropertyValue -Object $summary -Name "winner_pickle")
  }
  if ([string]::IsNullOrWhiteSpace($winnerPath)) {
    $winnerPath = "logs/NEAT/$($bootstrapLayout.output_prefix)_phase${bootstrapPhase}_seed$bootstrapSeed/models/winner_genome.pkl"
  }
  $fullPath = [System.IO.Path]::GetFullPath($winnerPath)
  if (-not (Test-Path $fullPath)) {
    throw "bootstrap winner genome not found: $fullPath"
  }
  return $fullPath
}

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "python not found: $python"
}

$lineageLayout = Get-LineageLayout -Profile $LineageProfile
$runtimeConfig = "scripts/configs/runtime_phase1.json"
$outputDir = "logs/NEAT/$($lineageLayout.output_prefix)_phase${Phase}_seed$Seed"

if (-not (Test-Path $runtimeConfig)) {
  throw "runtime config not found: $runtimeConfig"
}
$phaseRuntime = Read-JsonFile -Path $runtimeConfig
$effectiveFeatureProfile = [string]$FeatureProfile
if ([string]::IsNullOrWhiteSpace($effectiveFeatureProfile)) {
  $effectiveFeatureProfile = [string](Get-PropertyValue -Object $phaseRuntime -Name "feature_profile")
}
if ([string]::IsNullOrWhiteSpace($effectiveFeatureProfile)) {
  $effectiveFeatureProfile = "race8"
}
$effectiveFeatureProfile = $effectiveFeatureProfile.Trim().ToLowerInvariant()

switch ($effectiveFeatureProfile) {
  "race2" { $configFeedforward = "scripts/configs/neat_feedforward_race2.ini" }
  "race5" { $configFeedforward = "scripts/configs/neat_feedforward_race5.ini" }
  "race6" { $configFeedforward = "scripts/configs/neat_feedforward_race6.ini" }
  "race7" { $configFeedforward = "scripts/configs/neat_feedforward_race7.ini" }
  "race8" { $configFeedforward = "scripts/configs/neat_feedforward_race8.ini" }
  "race20" { $configFeedforward = "scripts/configs/neat_feedforward_race20.ini" }
  "hand7" { $configFeedforward = "scripts/configs/neat_feedforward_hand7.ini" }
  "hand10" { $configFeedforward = [string]$lineageLayout.config_feedforward }
  "material10" { $configFeedforward = [string]$lineageLayout.config_feedforward }
  "race10" { $configFeedforward = [string]$lineageLayout.config_feedforward }
  "oracle10" { $configFeedforward = [string]$lineageLayout.config_feedforward }
  "oracle10v2" { $configFeedforward = [string]$lineageLayout.config_feedforward }
  default { throw "unsupported feature profile: $effectiveFeatureProfile" }
}
if (-not (Test-Path $configFeedforward)) {
  throw "config not found: $configFeedforward"
}

$cmd = @(
  "scripts/neat_train.py",
  "--config-feedforward", $configFeedforward,
  "--runtime-config", $runtimeConfig,
  "--output-dir", $outputDir,
  "--seed", "$Seed",
  "--profile-name", "$($lineageLayout.profile_name_prefix)phase${Phase}_seed$Seed"
)

if (-not [string]::IsNullOrWhiteSpace($FeatureProfile)) {
  $cmd += @(
    "--feature-profile", $FeatureProfile
  )
}

if ($BootstrapSeedSpec.Count -gt 0) {
  $bootstrapSpecs = @($BootstrapSeedSpec | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })
  if ($bootstrapSpecs.Count -eq 0) {
  }
  elseif ($bootstrapSpecs.Count -eq 1) {
    $bootstrapWinnerPath = Resolve-BootstrapWinnerPath -Spec $bootstrapSpecs[0]
    $cmd += @(
      "--seed-genome", "$bootstrapWinnerPath",
      "--seed-genome-count", "48"
    )
  }
  elseif ($bootstrapSpecs.Count -eq 2) {
    $bootstrapWinnerPath1 = Resolve-BootstrapWinnerPath -Spec $bootstrapSpecs[0]
    $bootstrapWinnerPath2 = Resolve-BootstrapWinnerPath -Spec $bootstrapSpecs[1]
    $cmd += @(
      "--seed-genome-spec", "$bootstrapWinnerPath1", "24",
      "--seed-genome-spec", "$bootstrapWinnerPath2", "24"
    )
  }
  else {
    throw "BootstrapSeedSpec supports at most 2 specs for now"
  }
}
elseif ($Phase -ne "1") {
  $continueFromPreviousPhase = To-BoolOrDefault -Value (Get-PropertyValue -Object $phaseRuntime -Name "continue_from_previous_phase") -DefaultValue $true
  if ($continueFromPreviousPhase) {
    $previousPhase = [int]$Phase - 1
    $previousLabel = "phase$previousPhase"
    $previousRuntimePath = $runtimeConfig
    $previousRuntime = Read-JsonFile -Path $previousRuntimePath
    $previousSummaryPath = "logs/NEAT/$($lineageLayout.output_prefix)_phase${previousPhase}_seed$Seed/run_summary.json"
    if (-not (Test-Path $previousSummaryPath)) {
      throw "$previousLabel run summary not found: $previousSummaryPath"
    }
    $previousSummary = Read-JsonFile -Path $previousSummaryPath
    $previousAppliedOverrides = Get-PropertyValue -Object $previousSummary -Name "applied_overrides"
    $previousBaseGeneration = 0
    if ($null -ne $previousAppliedOverrides) {
      $previousBaseGeneration = To-PositiveIntOrDefault -Value (Get-PropertyValue -Object $previousAppliedOverrides -Name "base_generation") -DefaultValue 0
    }
    $previousRuntimeGenerations = To-PositiveIntOrDefault -Value (Get-PropertyValue -Object $previousRuntime -Name "generations") -DefaultValue 20
    $previousGenerations = To-PositiveIntOrDefault -Value (Get-PropertyValue -Object $previousSummary -Name "generations") -DefaultValue $previousRuntimeGenerations
    $previousOverrideGenerations = To-PositiveIntOrDefault -Value (Get-PropertyValue -Object $previousAppliedOverrides -Name "generations") -DefaultValue 0
    if ($previousBaseGeneration -eq 0 -and $previousOverrideGenerations -gt 0 -and $previousOverrideGenerations -lt $previousRuntimeGenerations) {
      $previousGenerations = $previousRuntimeGenerations
    }
    $cumulativeBaseGeneration = $previousBaseGeneration + $previousGenerations
    $previousWinnerRaw = [string](Get-PropertyValue -Object $previousSummary -Name "winner_pickle")
    if ([string]::IsNullOrWhiteSpace($previousWinnerRaw)) {
      $previousWinnerRaw = "logs/NEAT/$($lineageLayout.output_prefix)_phase${previousPhase}_seed$Seed/models/winner_genome.pkl"
    }
    $previousWinnerPath = [System.IO.Path]::GetFullPath($previousWinnerRaw)
    if (-not (Test-Path $previousWinnerPath)) {
      throw "$previousLabel winner genome not found: $previousWinnerPath"
    }

    $cmd += @(
      "--seed-genome", "$previousWinnerPath",
      "--base-generation", "$cumulativeBaseGeneration"
    )
  }
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

$summary = Resolve-RunSummaryObject -Summary $summary

$bestMetrics = Get-OverallBestMetrics -Summary $summary
$summaryElapsedSec = Get-OptionalDouble -Value (Get-PropertyValue -Object $summary -Name "run_elapsed_sec") -DefaultValue $phaseRunElapsedSec

Write-Host "=== Phase$Phase Summary (Seed=$Seed, Profile=$LineageProfile) ==="
if ([bool]$bestMetrics.available) {
  Write-Host "Best gen:         $($bestMetrics.generation)"
  Write-Host "Best genome key:  $($bestMetrics.genome_key)"
  Write-Host "Best win rate:    $($bestMetrics.win_rate)"
  Write-Host "Best fitness:     $($bestMetrics.fitness)"
  Write-Host "Best gold delta:  $($bestMetrics.mean_gold_delta)"
  Write-Host "GO opp games:     $($bestMetrics.go_opportunity_games)"
  Write-Host "GO opp count:     $($bestMetrics.go_opportunity_count)"
  Write-Host "GO count:         $($bestMetrics.go_count)"
  Write-Host "GO games:         $($bestMetrics.go_games)"
  Write-Host "GO fail count:    $($bestMetrics.go_fail_count)"
  Write-Host "GO fail rate:     $($bestMetrics.go_fail_rate)"
  Write-Host "GO rate:          $($bestMetrics.go_rate)"
}
else {
  Write-Host "Best gen:         N/A"
  Write-Host "Best genome key:  N/A"
  Write-Host "Best win rate:    N/A"
  Write-Host "Best fitness:     $(Get-PropertyValue -Object $summary -Name 'best_fitness')"
  Write-Host "Best gold delta:  N/A"
  Write-Host "GO opp games:     "
  Write-Host "GO opp count:     "
  Write-Host "GO count:         "
  Write-Host "GO games:         "
  Write-Host "GO fail count:    "
  Write-Host "GO fail rate:     "
  Write-Host "GO rate:          "
  Write-Host "Danger samples:   "
  Write-Host "Mean self danger: "
  Write-Host "Max self danger:  "
  Write-Host "Mean opp danger:  "
  Write-Host "Terminal self G:  "
  Write-Host "Terminal opp G:   "
}
Write-Host "Current EMA win:  $(Get-NestedPropertyValue -Object $summary -Path @('gate_state', 'ema_win_rate'))"
Write-Host "Current EMA imit: $(Get-NestedPropertyValue -Object $summary -Path @('gate_state', 'ema_imitation'))"
Write-Host "Current win rate: $(Get-NestedPropertyValue -Object $summary -Path @('gate_state', 'latest_win_rate'))"
Write-Host "Win slope(5):     $(Get-NestedPropertyValue -Object $summary -Path @('gate_state', 'latest_win_rate_slope_5'))"
Write-Host "Transition ready: $(Get-NestedPropertyValue -Object $summary -Path @('gate_state', 'transition_ready'))"
Write-Host "Transition gen:   $(Get-NestedPropertyValue -Object $summary -Path @('gate_state', 'transition_generation'))"
Write-Host "Elapsed time:     $summaryElapsedSec s"
Write-Host "================================"

exit 0
