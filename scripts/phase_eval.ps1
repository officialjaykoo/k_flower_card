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

  return [ordered]@{
    mean_gold_delta_min = [double]$meanGold
    win_rate_min = [double]$winRate
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
  "--fitness-loss-weight", "$($runtime.fitness_loss_weight)",
  "--fitness-draw-weight", "$($runtime.fitness_draw_weight)"
)

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

Write-Host ""
Write-Host "=== Phase$Phase Evaluation (Seed=$Seed) ==="
Write-Host "Win rate:        $($r.win_rate)"
Write-Host "Mean gold delta: $($r.mean_gold_delta)"
Write-Host "Fitness:         $($r.fitness)"
Write-Host "Imit play ratio: $($r.imitation_play_ratio)"
Write-Host "Imit match ratio:$($r.imitation_match_ratio)"
Write-Host "Imit opt ratio:  $($r.imitation_option_ratio)"
Write-Host "Bankrupt count:  $($r.bankrupt.my_bankrupt_count)"
Write-Host "================================"

$passed = ([double]$r.mean_gold_delta -ge [double]$passRule.mean_gold_delta_min) -and ([double]$r.win_rate -ge [double]$passRule.win_rate_min)
$passState = [ordered]@{
  passed = $passed
  reason = if ($passed) { "eval_gate_passed" } else { "eval_gate_not_passed" }
  seed = "$Seed"
  phase = "phase$Phase"
  pass_rule = [ordered]@{
    mean_gold_delta_min = [double]$passRule.mean_gold_delta_min
    win_rate_min = [double]$passRule.win_rate_min
  }
  win_rate = [double]$r.win_rate
  mean_gold_delta = [double]$r.mean_gold_delta
  fitness = [double]$r.fitness
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
