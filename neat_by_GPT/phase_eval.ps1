#
# phase_eval.ps1
# - Evaluate winner genome for selected phase/seed.
# - Keep runtime logic unchanged; this file is comment/structure organized.
#

param(
  [Parameter(Mandatory = $true)][ValidateSet("1", "2", "3")][string]$Phase,
  [Parameter(Mandatory = $true)][int]$Seed
)

# ---------------------------------------------------------------------------
# Section 1) Strict mode and common helpers
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

function Resolve-EvalGateRule {
  param([Parameter(Mandatory = $true)]$Runtime)

  $defaultMeanGold = 0.0
  $defaultWinRate = 0.45
  $defaultCvar10Gold = -1000000000.0
  $defaultCatastrophicLossRate = 1.0

  $meanGold = Get-OptionalDouble -Value $Runtime.eval_pass_mean_gold_delta_min
  if ([double]::IsNaN($meanGold)) {
    $meanGold = Get-OptionalDouble -Value $Runtime.transition_mean_gold_delta_min
  }
  if ([double]::IsNaN($meanGold)) {
    $meanGold = $defaultMeanGold
  }

  $winRate = Get-OptionalDouble -Value $Runtime.eval_pass_win_rate_min
  if ([double]::IsNaN($winRate)) {
    $winRate = $defaultWinRate
  }

  $cvar10Gold = Get-OptionalDouble -Value $Runtime.eval_pass_cvar10_gold_delta_min
  if ([double]::IsNaN($cvar10Gold)) {
    $cvar10Gold = $defaultCvar10Gold
  }

  $catastrophicLossRate = Get-OptionalDouble -Value $Runtime.eval_pass_catastrophic_loss_rate_max
  if ([double]::IsNaN($catastrophicLossRate)) {
    $catastrophicLossRate = $defaultCatastrophicLossRate
  }

  return [ordered]@{
    mean_gold_delta_min = [double]$meanGold
    win_rate_min = [double]$winRate
    cvar10_gold_delta_min = [double]$cvar10Gold
    catastrophic_loss_rate_max = [double]$catastrophicLossRate
  }
}

# ---------------------------------------------------------------------------
# Section 2) Paths and runtime loading
# ---------------------------------------------------------------------------
$runtimeConfigPath = "neat_by_GPT/configs/runtime_phase${Phase}.json"
$outputDir = "logs/NEAT_GPT/neat_phase${Phase}_seed$Seed"
$gateStatePath = Join-Path $outputDir "gate_state.json"
$genomePath = Join-Path $outputDir "models/winner_genome.json"

if (-not (Test-Path $runtimeConfigPath)) {
  throw "runtime config not found: $runtimeConfigPath"
}
if (-not (Test-Path $gateStatePath)) {
  throw "gate_state not found: $gateStatePath"
}
if (-not (Test-Path $genomePath)) {
  throw "winner genome not found: $genomePath"
}

$runtime = Read-JsonFile -Path $runtimeConfigPath
$gate = Read-JsonFile -Path $gateStatePath
$passRule = Resolve-EvalGateRule -Runtime $runtime

# ---------------------------------------------------------------------------
# Section 3) Build node evaluator command
# ---------------------------------------------------------------------------
$games = 1000
$seedTag = "gpt_phase${Phase}_eval_$Seed"
$policyValue = ""
if ($runtime.PSObject.Properties.Name -contains "opponent_policy") {
  $policyValue = [string]$runtime.opponent_policy
}
$hasPolicy = -not [string]::IsNullOrWhiteSpace($policyValue)
$hasPolicyMix = $false
$mixValue = $null
if ($runtime.PSObject.Properties.Name -contains "opponent_policy_mix") {
  $mixValue = $runtime.opponent_policy_mix
  $hasPolicyMix = ($null -ne $mixValue) -and ($mixValue.Count -gt 0)
}
if (-not $hasPolicy -and -not $hasPolicyMix) {
  throw "runtime must contain opponent_policy or opponent_policy_mix"
}
if (-not ($runtime.PSObject.Properties.Name -contains "fitness_profile")) {
  throw "runtime must contain fitness_profile"
}
$fitnessProfile = [string]$runtime.fitness_profile
if ([string]::IsNullOrWhiteSpace($fitnessProfile)) {
  throw "runtime fitness_profile is empty"
}

$cmd = @(
  "neat_by_GPT/scripts/neat_eval_worker.mjs",
  "--genome", $genomePath,
  "--games", "$games",
  "--seed", $seedTag,
  "--max-steps", "$($runtime.max_eval_steps)",
  "--fitness-profile", "$fitnessProfile",
  "--first-turn-policy", "alternate"
)

if ($hasPolicy) {
  $cmd += @("--opponent-policy", "$policyValue")
}
if ($hasPolicyMix) {
  $mixJson = $mixValue | ConvertTo-Json -Depth 8 -Compress
  $cmd += @("--opponent-policy-mix", "$mixJson")
}

# ---------------------------------------------------------------------------
# Section 4) Execute and parse
# ---------------------------------------------------------------------------
$resultLines = & node @cmd
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
  exit $exitCode
}

$resultJson = $resultLines | Select-Object -Last 1
if ([string]::IsNullOrWhiteSpace($resultJson)) {
  throw "empty eval output"
}

# ---------------------------------------------------------------------------
# Section 5) Evaluation report output
# ---------------------------------------------------------------------------
$savePath = Join-Path $outputDir "phase${Phase}_eval_1000.json"
$enc = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($savePath), $resultJson, $enc)

$r = $resultJson | ConvertFrom-Json
$goCount = [int](Get-OptionalDouble -Value $r.go_count -DefaultValue 0.0)
$goGames = [int](Get-OptionalDouble -Value $r.go_games -DefaultValue 0.0)
$goFailCount = [int](Get-OptionalDouble -Value $r.go_fail_count -DefaultValue 0.0)
$goFailRate = [double](Get-OptionalDouble -Value $r.go_fail_rate -DefaultValue 0.0)
$goRate = [double](Get-OptionalDouble -Value $r.go_rate -DefaultValue 0.0)
$cvar10GoldDelta = [double](Get-OptionalDouble -Value $r.cvar10_gold_delta -DefaultValue 0.0)
$catastrophicLossRate = [double](Get-OptionalDouble -Value $r.catastrophic_loss_rate -DefaultValue 0.0)
$fitnessModel = [string]$r.fitness_model
$fitnessProfile = [string]$r.fitness_profile
$goldCore = [double](Get-OptionalDouble -Value $r.fitness_components.gold_core -DefaultValue 0.0)
$expectedResult = [double](Get-OptionalDouble -Value $r.fitness_components.expected_result -DefaultValue 0.0)
$tieBreak = [double](Get-OptionalDouble -Value $r.fitness_components.tie_break -DefaultValue 0.0)
$bankruptRate = [double](Get-OptionalDouble -Value $r.fitness_components.bankrupt_rate -DefaultValue 0.0)
$bankruptPenalty = [double](Get-OptionalDouble -Value $r.fitness_components.bankrupt_penalty -DefaultValue 0.0)
$goZeroHardFail = [bool]$r.fitness_components.go_zero_hard_fail

Write-Host ""
Write-Host "=== Phase$Phase Evaluation (Seed=$Seed) ==="
Write-Host "Win rate:        $($r.win_rate)"
Write-Host "Mean gold delta: $($r.mean_gold_delta)"
Write-Host "P10/P50/P90:     $($r.p10_gold_delta) / $($r.p50_gold_delta) / $($r.p90_gold_delta)"
Write-Host "CVaR10 delta:    $cvar10GoldDelta"
Write-Host "Cat loss rate:   $catastrophicLossRate"
Write-Host "Fitness:         $($r.fitness)"
Write-Host "Fitness model:   $fitnessModel"
Write-Host "Fitness profile: $fitnessProfile"
Write-Host "Gold core:       $goldCore"
Write-Host "Expected result: $expectedResult"
Write-Host "Tie break:       $tieBreak"
Write-Host "Bankrupt rate:   $bankruptRate"
Write-Host "Bankrupt penalty:$bankruptPenalty"
Write-Host "GO hard-fail:    $goZeroHardFail"
Write-Host "GO count:        $goCount"
Write-Host "GO games:        $goGames"
Write-Host "GO fail count:   $goFailCount"
Write-Host "GO fail rate:    $goFailRate"
Write-Host "GO rate:         $goRate"
Write-Host "Bankrupt count:  $($r.bankrupt.my_bankrupt_count)"
Write-Host "================================"

# ---------------------------------------------------------------------------
# Section 6) Gate pass/fail and pass-state export
# ---------------------------------------------------------------------------
$passMeanGold = ([double]$r.mean_gold_delta -ge [double]$passRule.mean_gold_delta_min)
$passWinRate = ([double]$r.win_rate -ge [double]$passRule.win_rate_min)
$passCvar10Gold = ($cvar10GoldDelta -ge [double]$passRule.cvar10_gold_delta_min)
$passCatastrophicLossRate = ($catastrophicLossRate -le [double]$passRule.catastrophic_loss_rate_max)
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
  phase = "phase$Phase"
  pass_rule = [ordered]@{
    mean_gold_delta_min = [double]$passRule.mean_gold_delta_min
    win_rate_min = [double]$passRule.win_rate_min
    cvar10_gold_delta_min = [double]$passRule.cvar10_gold_delta_min
    catastrophic_loss_rate_max = [double]$passRule.catastrophic_loss_rate_max
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
  fitness_model = $fitnessModel
  fitness_profile = $fitnessProfile
  gold_core = $goldCore
  expected_result = $expectedResult
  tie_break = $tieBreak
  bankrupt_rate = $bankruptRate
  bankrupt_penalty = $bankruptPenalty
  go_zero_hard_fail = $goZeroHardFail
  eval_result_path = $savePath
  gate_state_path = $gateStatePath
  transition_ready = [bool]$gate.transition_ready
  transition_generation = $gate.transition_generation
}

$passStatePath = Join-Path $outputDir "phase${Phase}_pass_state.json"
$passStateJson = $passState | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($passStatePath), $passStateJson, $enc)
Write-Output $passStateJson

if ($passed) {
  exit 0
}

exit 2
