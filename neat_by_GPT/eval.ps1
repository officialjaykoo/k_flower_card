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
$EvalWorker = Join-Path $PSScriptRoot "scripts\neat_eval_worker.mjs"

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

function Assert-RuntimeEvalKeys {
  param([Parameter(Mandatory = $true)]$Runtime)
  $requiredKeys = @(
    "max_eval_steps",
    "teacher_policy",
    "fitness_profile",
    "eval_games",
    "eval_opponent_policy",
    "eval_opponent_policy_mix",
    "eval_pass_win_rate_min",
    "eval_pass_mean_gold_delta_min",
    "eval_pass_cvar10_gold_delta_min",
    "eval_pass_catastrophic_loss_rate_max"
  )
  foreach ($key in $requiredKeys) {
    if (-not ($Runtime.PSObject.Properties.Name -contains $key)) {
      throw "runtime missing required key: $key"
    }
  }
}

function Get-PolicyLabel {
  param(
    [Parameter(Mandatory = $false)][string]$Policy,
    [Parameter(Mandatory = $false)]$PolicyMix
  )
  if (-not [string]::IsNullOrWhiteSpace($Policy)) {
    return ($Policy.ToLowerInvariant() -replace "[^a-z0-9]+", "_").Trim("_")
  }
  if ($null -ne $PolicyMix -and $PolicyMix.Count -gt 0) {
    return "mix"
  }
  throw "eval_opponent_policy or eval_opponent_policy_mix is required"
}

$RuntimeConfig = Resolve-PathFromBase -Path $RuntimeConfig -BasePath $RepoRoot -DefaultPath $DefaultRuntimeConfig
if (-not (Test-Path $RuntimeConfig)) {
  throw "runtime config not found: $RuntimeConfig"
}
if (-not (Test-Path $EvalWorker)) {
  throw "eval worker not found: $EvalWorker"
}

$runtime = Read-JsonFile -Path $RuntimeConfig
Assert-RuntimeEvalKeys -Runtime $runtime
$outputDirResolved = Resolve-NeatGptOutputDirPath -RepoRoot $RepoRoot -RuntimeConfigPath $RuntimeConfig -SeedValue $Seed -ExplicitOutputDir $OutputDir
$genomePath = Join-Path $outputDirResolved "models/winner_genome.json"
if (-not (Test-Path $genomePath)) {
  throw "winner genome not found: $genomePath"
}

$games = [int]$runtime.eval_games
$policyValue = [string]$runtime.eval_opponent_policy
$policyMix = $runtime.eval_opponent_policy_mix
$teacherPolicy = [string]$runtime.teacher_policy
$fitnessProfile = [string]$runtime.fitness_profile
if ([string]::IsNullOrWhiteSpace($fitnessProfile)) {
  throw "runtime fitness_profile is empty"
}
$policyLabel = Get-PolicyLabel -Policy $policyValue -PolicyMix $policyMix
$seedTag = "$([System.IO.Path]::GetFileNameWithoutExtension($RuntimeConfig))_eval_seed$Seed"
$cmd = @(
  $EvalWorker,
  "--genome", $genomePath,
  "--games", "$games",
  "--seed", $seedTag,
  "--max-steps", "$($runtime.max_eval_steps)",
  "--fitness-profile", $fitnessProfile,
  "--first-turn-policy", "alternate"
)
if (-not [string]::IsNullOrWhiteSpace($teacherPolicy)) {
  $cmd += @("--teacher-policy", $teacherPolicy)
}
if (-not [string]::IsNullOrWhiteSpace($policyValue)) {
  $cmd += @("--opponent-policy", $policyValue)
}
if ($null -ne $policyMix -and $policyMix.Count -gt 0) {
  $mixJson = $policyMix | ConvertTo-Json -Depth 8 -Compress
  $cmd += @("--opponent-policy-mix", $mixJson)
}

$resultLines = & node @cmd
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
  exit $exitCode
}

$resultJson = $resultLines | Select-Object -Last 1
if ([string]::IsNullOrWhiteSpace($resultJson)) {
  throw "empty eval output"
}

$r = $resultJson | ConvertFrom-Json
$savePath = Join-Path $outputDirResolved "eval_${policyLabel}_${games}.json"
$passStatePath = Join-Path $outputDirResolved "eval_${policyLabel}_${games}_pass.json"
$enc = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($savePath), $resultJson, $enc)

$cvar10GoldDelta = [double](Get-OptionalDouble -Value $r.cvar10_gold_delta -DefaultValue 0.0)
$catastrophicLossRate = [double](Get-OptionalDouble -Value $r.catastrophic_loss_rate -DefaultValue 0.0)
$goGames = [int](Get-OptionalDouble -Value $r.go_games -DefaultValue 0.0)
$goCount = [int](Get-OptionalDouble -Value $r.go_count -DefaultValue 0.0)
$goFailCount = [int](Get-OptionalDouble -Value $r.go_fail_count -DefaultValue 0.0)
$goFailRate = [double](Get-OptionalDouble -Value $r.go_fail_rate -DefaultValue 0.0)
$goRate = [double](Get-OptionalDouble -Value $r.go_rate -DefaultValue 0.0)

$passMeanGold = ([double]$r.mean_gold_delta -ge [double]$runtime.eval_pass_mean_gold_delta_min)
$passWinRate = ([double]$r.win_rate -ge [double]$runtime.eval_pass_win_rate_min)
$passCvar10Gold = ($cvar10GoldDelta -ge [double]$runtime.eval_pass_cvar10_gold_delta_min)
$passCatastrophicLossRate = ($catastrophicLossRate -le [double]$runtime.eval_pass_catastrophic_loss_rate_max)
$passed = $passMeanGold -and $passWinRate -and $passCvar10Gold -and $passCatastrophicLossRate

$failReasons = @()
if (-not $passMeanGold) { $failReasons += "mean_gold_delta" }
if (-not $passWinRate) { $failReasons += "win_rate" }
if (-not $passCvar10Gold) { $failReasons += "cvar10_gold_delta" }
if (-not $passCatastrophicLossRate) { $failReasons += "catastrophic_loss_rate" }
$reasonText = if ($passed) { "eval_gate_passed" } else { "eval_gate_not_passed:" + ($failReasons -join ",") }

$passState = [ordered]@{
  passed = $passed
  reason = $reasonText
  seed = "$Seed"
  runtime = [System.IO.Path]::GetFileName($RuntimeConfig)
  opponent = if (-not [string]::IsNullOrWhiteSpace($policyValue)) { $policyValue } else { "mix" }
  pass_rule = [ordered]@{
    mean_gold_delta_min = [double]$runtime.eval_pass_mean_gold_delta_min
    win_rate_min = [double]$runtime.eval_pass_win_rate_min
    cvar10_gold_delta_min = [double]$runtime.eval_pass_cvar10_gold_delta_min
    catastrophic_loss_rate_max = [double]$runtime.eval_pass_catastrophic_loss_rate_max
  }
  win_rate = [double]$r.win_rate
  mean_gold_delta = [double]$r.mean_gold_delta
  cvar10_gold_delta = $cvar10GoldDelta
  catastrophic_loss_rate = $catastrophicLossRate
  fitness = [double]$r.fitness
  go_count = $goCount
  go_games = $goGames
  go_fail_count = $goFailCount
  go_fail_rate = $goFailRate
  go_rate = $goRate
  eval_result_path = $savePath
}

$passStateJson = $passState | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($passStatePath), $passStateJson, $enc)

Write-Host "=== NEAT GPT Evaluation ==="
Write-Host "Runtime:          $RuntimeConfig"
Write-Host "Opponent:         $(if (-not [string]::IsNullOrWhiteSpace($policyValue)) { $policyValue } else { 'mix' })"
Write-Host "Games:            $games"
Write-Host "Win rate:         $($r.win_rate)"
Write-Host "Mean gold delta:  $($r.mean_gold_delta)"
Write-Host "P10/P50/P90:      $($r.p10_gold_delta) / $($r.p50_gold_delta) / $($r.p90_gold_delta)"
Write-Host "CVaR10 delta:     $cvar10GoldDelta"
Write-Host "Cat loss rate:    $catastrophicLossRate"
Write-Host "Fitness:          $($r.fitness)"
Write-Host "GO games:         $goGames"
Write-Host "GO count:         $goCount"
Write-Host "GO fail count:    $goFailCount"
Write-Host "GO fail rate:     $goFailRate"
Write-Host "GO rate:          $goRate"
Write-Host "Passed:           $passed"
Write-Host "==========================="

Write-Output $passStateJson
if ($passed) {
  exit 0
}
exit 2
