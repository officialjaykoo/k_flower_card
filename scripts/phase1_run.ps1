# Pipeline Stage: Phase 1 Runner Wrapper
# Quick Read Map:
# 1) Resolve python/runtime paths
# 2) Build neat_train command
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
$runtimeConfig = "scripts/configs/runtime_phase1.json"
$outputDir = "logs/neat_phase1_seed$Seed"

$cmd = @(
  "scripts/neat_train.py",
  "--config-feedforward", $configFeedforward,
  "--runtime-config", $runtimeConfig,
  "--output-dir", $outputDir,
  "--seed", "$Seed",
  "--profile-name", "phase1_seed$Seed"
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

Write-Host "=== Phase1 결과 (Seed=$Seed) ==="
Write-Host "EMA 승률:       $($summary.gate_state.ema_win_rate)"
Write-Host "EMA 모방:       $($summary.gate_state.ema_imitation)"
Write-Host "최신 승률:      $($summary.gate_state.latest_win_rate)"
Write-Host "승률 기울기:    $($summary.gate_state.latest_win_rate_slope_5)"
Write-Host "전환 준비:      $($summary.gate_state.transition_ready)"
Write-Host "전환 세대:      $($summary.gate_state.transition_generation)"
Write-Host "best_fitness:   $($summary.best_fitness)"
Write-Host "================================"

exit 0
