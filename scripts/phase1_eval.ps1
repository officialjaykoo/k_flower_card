param(
  [Parameter(Mandatory = $true)][int]$Seed
)

$runtimeConfigPath = "scripts/configs/runtime_phase1.json"
$outputDir = "logs/neat_phase1_seed$Seed"
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

$runtime = Get-Content $runtimeConfigPath -Raw -Encoding UTF8 | ConvertFrom-Json
$gate = Get-Content $gateStatePath -Raw -Encoding UTF8 | ConvertFrom-Json

$passMeanGoldMin = 100
$passWinRateMin = 0.48

$games = 1000
$seedTag = "phase1_eval_$Seed"
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

$savePath = Join-Path $outputDir "phase1_eval_1000.json"
$enc = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($savePath), $resultJson, $enc)

$r = $resultJson | ConvertFrom-Json

Write-Host ""
Write-Host "=== Phase1 실전 결과 (Seed=$Seed) ==="
Write-Host "승률:          $($r.win_rate)"
Write-Host "골드 평균:     $($r.mean_gold_delta)"
Write-Host "fitness:       $($r.fitness)"
Write-Host "play 모방률:   $($r.imitation_play_ratio)"
Write-Host "match 모방률:  $($r.imitation_match_ratio)"
Write-Host "option 모방률: $($r.imitation_option_ratio)"
Write-Host "파산 당한 수:  $($r.bankrupt.my_bankrupt_count)"
Write-Host "================================"

$passed = ([double]$r.mean_gold_delta -ge $passMeanGoldMin) -and ([double]$r.win_rate -ge $passWinRateMin)
$passState = [ordered]@{
  passed = $passed
  reason = if ($passed) { "eval_gate_passed" } else { "eval_gate_not_passed" }
  seed = "$Seed"
  pass_rule = [ordered]@{
    mean_gold_delta_min = $passMeanGoldMin
    win_rate_min = $passWinRateMin
  }
  win_rate = [double]$r.win_rate
  mean_gold_delta = [double]$r.mean_gold_delta
  fitness = [double]$r.fitness
  eval_result_path = $savePath
  gate_state_path = $gateStatePath
  transition_ready = [bool]$gate.transition_ready
  transition_generation = $gate.transition_generation
}

$passStatePath = Join-Path $outputDir "phase1_pass_state.json"
$passStateJson = $passState | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($passStatePath), $passStateJson, $enc)
Write-Output $passStateJson

if ($passed) {
  exit 0
}

exit 2
