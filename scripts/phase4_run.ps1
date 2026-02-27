# Pipeline Stage: Phase 4 Runner Wrapper
# Quick Read Map:
# 1) Validate resume checkpoint/runtime
# 2) Build neat_train resume command
# 3) Execute and parse summary JSON
# 4) Print gate/eval highlights

param(
  [Parameter(Mandatory = $true)][int]$Seed
)

$python = ".venv\Scripts\python.exe"
if (-not (Test-Path $python)) {
  throw "python not found: $python"
}

$configFeedforward = "scripts/configs/neat_feedforward.ini"
$runtimeConfig = "scripts/configs/runtime_phase4.json"
$resumeCheckpoint = "logs/NEAT/neat_phase3_seed$Seed/checkpoints/neat-checkpoint-gen199"
$outputDir = "logs/NEAT/neat_phase4_seed$Seed"

if (-not (Test-Path $resumeCheckpoint)) {
  throw "phase3 checkpoint not found: $resumeCheckpoint"
}

$cmd = @(
  "scripts/neat_train.py",
  "--config-feedforward", $configFeedforward,
  "--runtime-config", $runtimeConfig,
  "--output-dir", $outputDir,
  "--resume", $resumeCheckpoint,
  "--base-generation", "200",
  "--seed", "$Seed",
  "--profile-name", "phase4_seed$Seed"
)

$result = & $python @cmd | Out-String
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

Write-Host "=== Phase4 결과 (Seed=$Seed) ==="
Write-Host "EMA 승률:       $($summary.gate_state.ema_win_rate)"
Write-Host "EMA 모방:       $($summary.gate_state.ema_imitation)"
Write-Host "최신 승률:      $($summary.gate_state.latest_win_rate)"
Write-Host "승률 기울기:    $($summary.gate_state.latest_win_rate_slope_5)"
Write-Host "전환 준비:      $($summary.gate_state.transition_ready)"
Write-Host "전환 세대:      $($summary.gate_state.transition_generation)"
Write-Host "best_fitness:   $($summary.best_fitness)"
Write-Host "================================"

exit 0
