# Pipeline Stage: Phase 3 Eval Wrapper
# Quick Read Map:
# 1) Load runtime + gate + genomes
# 2) Validate thresholds from gate_state
# 3) Build neat_eval_worker command
# 4) Save and print pass/fail summary

param(
  [Parameter(Mandatory = $true)][int]$Seed
)

$runtimeConfigPath = "scripts/configs/runtime_phase3.json"
$outputDir = "logs/NEAT/neat_phase3_seed$Seed"
$gateStatePath = Join-Path $outputDir "gate_state.json"
$genomePath = Join-Path $outputDir "models/winner_genome.json"
$opponentGenomePath = "logs/NEAT/neat_phase2_seed$Seed/models/winner_genome.json"

if (-not (Test-Path $runtimeConfigPath)) {
  throw "runtime config not found: $runtimeConfigPath"
}
if (-not (Test-Path $gateStatePath)) {
  throw "gate_state not found: $gateStatePath"
}
if (-not (Test-Path $genomePath)) {
  throw "winner genome not found: $genomePath"
}
if (-not (Test-Path $opponentGenomePath)) {
  throw "phase2 winner genome not found: $opponentGenomePath"
}

$runtime = Get-Content $runtimeConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$gate = Get-Content $gateStatePath -Raw -Encoding UTF8 | ConvertFrom-Json

function Test-ApproxEq {
  param(
    [double]$A,
    [double]$B,
    [double]$Eps = 1e-9
  )
  return [math]::Abs($A - $B) -le $Eps
}

$thresholds = $gate.thresholds
if ($null -eq $thresholds) {
  throw "thresholds missing in gate_state: $gateStatePath"
}

$thresholdsOk = $true
$thresholdsOk = $thresholdsOk -and ($thresholds.gate_mode -eq "win_rate_only")
$thresholdsOk = $thresholdsOk -and (Test-ApproxEq ([double]$thresholds.transition_ema_win_rate) 0.45)
$thresholdsOk = $thresholdsOk -and (Test-ApproxEq ([double]$thresholds.transition_mean_gold_delta_min) 0.0)
$thresholdsOk = $thresholdsOk -and ([int]$thresholds.transition_streak -eq 1)

$gatePassed = [bool]$gate.transition_ready -and ($null -ne $gate.transition_generation) -and (-not [bool]$gate.failure_triggered)

if (-not ($thresholdsOk -and $gatePassed)) {
  $fail = [ordered]@{
    passed = $false
    reason = "gate_not_passed"
    seed = "$Seed"
    gate_state_path = $gateStatePath
    transition_ready = [bool]$gate.transition_ready
    transition_generation = $gate.transition_generation
    failure_triggered = [bool]$gate.failure_triggered
    ema_win_rate = $gate.ema_win_rate
    latest_mean_gold_delta = $gate.latest_mean_gold_delta
    latest_best_fitness = $gate.latest_best_fitness
  }
  Write-Output ($fail | ConvertTo-Json -Depth 8)
  exit 2
}

$games = 1000
$seedTag = "phase3_eval_$Seed"
$cmd = @(
  "scripts/neat_eval_worker.mjs",
  "--genome", $genomePath,
  "--opponent-policy", "genome",
  "--opponent-genome", $opponentGenomePath,
  "--games", "$games",
  "--seed", $seedTag,
  "--max-steps", "$($runtime.max_eval_steps)",
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

$savePath = Join-Path $outputDir "phase3_eval_1000.json"
$enc = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($savePath), $resultJson, $enc)

$r = $resultJson | ConvertFrom-Json

Write-Host ""
Write-Host "=== Phase3 실전 결과 (Seed=$Seed) ==="
Write-Host "승률:          $($r.win_rate)"
Write-Host "골드 평균:     $($r.mean_gold_delta)"
Write-Host "fitness:       $($r.fitness)"
Write-Host "play 모방률:   $($r.imitation_play_ratio)"
Write-Host "match 모방률:  $($r.imitation_match_ratio)"
Write-Host "option 모방률: $($r.imitation_option_ratio)"
Write-Host "파산 당한 수:  $($r.bankrupt.my_bankrupt_count)"
Write-Host "================================"

exit 0
