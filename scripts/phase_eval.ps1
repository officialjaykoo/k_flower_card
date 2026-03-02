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

  $defaultMeanGold = 100.0
  $defaultWinRate = 0.48
  $defaultGoGamesMin = 0

  $meanGold = Get-OptionalDouble -Value $Runtime.eval_pass_mean_gold_delta_min
  if ([double]::IsNaN($meanGold)) {
    $meanGold = Get-OptionalDouble -Value $Runtime.transition_mean_gold_delta_min
  }
  if ([double]::IsNaN($meanGold)) {
    $meanGold = $defaultMeanGold
  }

  $winRate = Get-OptionalDouble -Value $Runtime.eval_pass_win_rate_min
  if ([double]::IsNaN($winRate)) {
    $winRate = Get-OptionalDouble -Value $Runtime.transition_ema_win_rate
  }
  if ([double]::IsNaN($winRate)) {
    $winRate = $defaultWinRate
  }

  $goGamesMin = [int](Get-OptionalDouble -Value $Runtime.eval_pass_go_games_min -DefaultValue $defaultGoGamesMin)
  if ($goGamesMin -lt 0) {
    $goGamesMin = 0
  }

  return [ordered]@{
    mean_gold_delta_min = [double]$meanGold
    win_rate_min = [double]$winRate
    go_games_min = [int]$goGamesMin
  }
}

$runtimeConfigPath = "scripts/configs/runtime_phase$Phase.json"
$outputDir = "logs/NEAT/neat_phase${Phase}_seed$Seed"
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

$games = 1000
$seedTag = "phase${Phase}_eval_$Seed"
$cmd = @(
  "scripts/neat_eval_worker.mjs",
  "--genome", $genomePath,
  "--games", "$games",
  "--seed", $seedTag,
  "--max-steps", "$($runtime.max_eval_steps)",
  "--opponent-policy", "$($runtime.opponent_policy)",
  "--first-turn-policy", "alternate",
  "--fitness-gold-scale", "$($runtime.fitness_gold_scale)",
  "--fitness-win-weight", "$($runtime.fitness_win_weight)",
  "--fitness-gold-weight", "$($runtime.fitness_gold_weight)",
  "--fitness-go-min-games", "$($runtime.fitness_go_min_games)"
)

if ($null -ne $runtime.fitness_go_low_games_penalty) {
  $cmd += @("--fitness-go-low-games-penalty", "$($runtime.fitness_go_low_games_penalty)")
}
if ($null -ne $runtime.fitness_go_max_games) {
  $cmd += @("--fitness-go-max-games", "$($runtime.fitness_go_max_games)")
}
if ($null -ne $runtime.fitness_go_max_games_penalty) {
  $cmd += @("--fitness-go-max-games-penalty", "$($runtime.fitness_go_max_games_penalty)")
}
if ($null -ne $runtime.fitness_go_fail_penalty_trigger) {
  $cmd += @("--fitness-go-fail-penalty-trigger", "$($runtime.fitness_go_fail_penalty_trigger)")
}
if ($null -ne $runtime.fitness_go_fail_penalty_amount) {
  $cmd += @("--fitness-go-fail-penalty-amount", "$($runtime.fitness_go_fail_penalty_amount)")
}
if ($null -ne $runtime.fitness_go_fail_bonus_trigger) {
  $cmd += @("--fitness-go-fail-bonus-trigger", "$($runtime.fitness_go_fail_bonus_trigger)")
}
if ($null -ne $runtime.fitness_go_fail_bonus_amount) {
  $cmd += @("--fitness-go-fail-bonus-amount", "$($runtime.fitness_go_fail_bonus_amount)")
}
if ($null -ne $runtime.fitness_go_fail_bonus_trigger_2) {
  $cmd += @("--fitness-go-fail-bonus-trigger-2", "$($runtime.fitness_go_fail_bonus_trigger_2)")
}
if ($null -ne $runtime.fitness_go_fail_bonus_amount_2) {
  $cmd += @("--fitness-go-fail-bonus-amount-2", "$($runtime.fitness_go_fail_bonus_amount_2)")
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

$savePath = Join-Path $outputDir "phase${Phase}_eval_1000.json"
$enc = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($savePath), $resultJson, $enc)

$r = $resultJson | ConvertFrom-Json
$goCount = [int](Get-OptionalDouble -Value $r.go_count -DefaultValue 0.0)
$goGames = [int](Get-OptionalDouble -Value $r.go_games -DefaultValue 0.0)
$goFailCount = [int](Get-OptionalDouble -Value $r.go_fail_count -DefaultValue 0.0)
$goFailRate = [double](Get-OptionalDouble -Value $r.go_fail_rate -DefaultValue 0.0)
$goRate = [double](Get-OptionalDouble -Value $r.go_rate -DefaultValue 0.0)
$luckProxy = [double](Get-OptionalDouble -Value $r.luck_proxy -DefaultValue 0.0)
$luckSeatWinRateGap = [double](Get-OptionalDouble -Value $r.luck_components.seat_win_rate_gap -DefaultValue 0.0)
$luckGoldVolatilityNorm = [double](Get-OptionalDouble -Value $r.luck_components.gold_volatility_norm -DefaultValue 0.0)

Write-Host ""
Write-Host "=== Phase$Phase Evaluation (Seed=$Seed) ==="
Write-Host "Win rate:        $($r.win_rate)"
Write-Host "Mean gold delta: $($r.mean_gold_delta)"
Write-Host "Fitness:         $($r.fitness)"
Write-Host "GO count:        $goCount"
Write-Host "GO games:        $goGames"
Write-Host "GO fail count:   $goFailCount"
Write-Host "GO fail rate:    $goFailRate"
Write-Host "GO rate:         $goRate"
Write-Host "Luck proxy:      $luckProxy"
Write-Host "Luck seat gap:   $luckSeatWinRateGap"
Write-Host "Luck volatility: $luckGoldVolatilityNorm"
Write-Host "Imit play ratio: $($r.imitation_play_ratio)"
Write-Host "Imit match ratio:$($r.imitation_match_ratio)"
Write-Host "Imit opt ratio:  $($r.imitation_option_ratio)"
Write-Host "Bankrupt count:  $($r.bankrupt.my_bankrupt_count)"
Write-Host "================================"

$passMeanGold = ([double]$r.mean_gold_delta -ge [double]$passRule.mean_gold_delta_min)
$passWinRate = ([double]$r.win_rate -ge [double]$passRule.win_rate_min)
$passGoGames = ($goGames -ge [int]$passRule.go_games_min)
$passed = $passMeanGold -and $passWinRate -and $passGoGames

$failReasons = @()
if (-not $passMeanGold) { $failReasons += "mean_gold_delta" }
if (-not $passWinRate) { $failReasons += "win_rate" }
if (-not $passGoGames) { $failReasons += "go_games" }
$reasonText = if ($passed) { "eval_gate_passed" } else { "eval_gate_not_passed:" + ($failReasons -join ",") }
$passState = [ordered]@{
  passed = $passed
  reason = $reasonText
  seed = "$Seed"
  phase = "phase$Phase"
  pass_rule = [ordered]@{
    mean_gold_delta_min = [double]$passRule.mean_gold_delta_min
    win_rate_min = [double]$passRule.win_rate_min
    go_games_min = [int]$passRule.go_games_min
  }
  win_rate = [double]$r.win_rate
  mean_gold_delta = [double]$r.mean_gold_delta
  fitness = [double]$r.fitness
  go_count = $goCount
  go_games = $goGames
  go_fail_count = $goFailCount
  go_fail_rate = $goFailRate
  go_rate = $goRate
  luck_proxy = $luckProxy
  luck_seat_win_rate_gap = $luckSeatWinRateGap
  luck_gold_volatility_norm = $luckGoldVolatilityNorm
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
