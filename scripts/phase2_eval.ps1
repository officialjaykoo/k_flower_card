param(
  [Parameter(Mandatory = $true)][int]$Seed
)

$runtimeConfigPath = "scripts/configs/runtime_phase2.json"
$outputDir = "logs/neat_phase2_seed$Seed"
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
$thresholdsOk = $thresholdsOk -and ([int]$thresholds.transition_streak -eq 3)

$gatePassed = [bool]$gate.transition_ready -and ($null -ne $gate.transition_generation)

if (-not ($thresholdsOk -and $gatePassed)) {
  $fail = [ordered]@{
    passed = $false
    reason = "gate_not_passed"
    seed = "$Seed"
    gate_state_path = $gateStatePath
    transition_ready = [bool]$gate.transition_ready
    transition_generation = $gate.transition_generation
    ema_win_rate = $gate.ema_win_rate
    gate_streak = $gate.gate_streak
  }
  Write-Output ($fail | ConvertTo-Json -Depth 8)
  exit 2
}

$games = 1000
$seedTag = "phase2_eval_$Seed"
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

$savePath = Join-Path $outputDir "phase2_eval_1000.json"
$enc = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($savePath), $resultJson, $enc)

Write-Output $resultJson
exit 0
