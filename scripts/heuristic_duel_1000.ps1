# Pipeline Stage: Heuristic Benchmark Wrapper
# Quick Read Map:
# 1) Parse params and normalize defaults
# 2) Build heuristic_duel_worker command
# 3) Execute worker and parse JSON
# 4) Save summary artifact
# 5) Print concise report

param(
  [Parameter(Mandatory = $true)][string]$PolicyA,
  [Parameter(Mandatory = $true)][string]$PolicyB,
  [string]$Seed = "",
  [int]$MaxSteps = 600,
  [ValidateSet("lean", "full")][string]$KiboDetail = "lean",
  [string]$KiboOut = "",
  [string]$DatasetOut = "",
  [ValidateSet("all", "human", "ai")][string]$DatasetActor = "all",
  [string]$UnresolvedOut = "",
  [int]$UnresolvedLimit = 0
)

if ([string]::IsNullOrWhiteSpace($Seed)) {
  $Seed = "heuristic_duel_{0}_{1}" -f (Get-Date -Format "yyyyMMdd_HHmmss_fff"), (Get-Random -Minimum 100000 -Maximum 999999)
}

$games = 1000
$cmd = @(
  "scripts/heuristic_duel_worker.mjs",
  "--policy-a", "$PolicyA",
  "--policy-b", "$PolicyB",
  "--games", "$games",
  "--seed", "$Seed",
  "--max-steps", "$MaxSteps",
  "--first-turn-policy", "alternate",
  "--continuous-series", "1",
  "--kibo-detail", "$KiboDetail"
)
if (-not [string]::IsNullOrWhiteSpace($KiboOut)) {
  $cmd += @("--kibo-out", "$KiboOut")
}
if (-not [string]::IsNullOrWhiteSpace($DatasetOut)) {
  $cmd += @("--dataset-out", "$DatasetOut", "--dataset-actor", "$DatasetActor")
}
if (-not [string]::IsNullOrWhiteSpace($UnresolvedOut)) {
  $cmd += @("--unresolved-out", "$UnresolvedOut", "--unresolved-limit", "$UnresolvedLimit")
}

$resultLines = & node @cmd
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
  exit $exitCode
}

$resultJson = $resultLines | Select-Object -Last 1
if ([string]::IsNullOrWhiteSpace($resultJson)) {
  throw "empty duel output"
}

$outputDir = "logs/heuristic_duel"
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

function Sanitize-FilePart {
  param([string]$Text)
  return (($Text -replace "[^A-Za-z0-9._-]", "_").Trim("_"))
}

$a = Sanitize-FilePart $PolicyA
$b = Sanitize-FilePart $PolicyB
if ([string]::IsNullOrWhiteSpace($a)) { $a = "policyA" }
if ([string]::IsNullOrWhiteSpace($b)) { $b = "policyB" }

$saveName = "{0}_vs_{1}_1000.json" -f $a, $b
$savePath = Join-Path $outputDir $saveName
$enc = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($savePath), $resultJson, $enc)

$r = $resultJson | ConvertFrom-Json

Write-Host ""
Write-Host "=== Heuristic Duel (1000 games) ==="
Write-Host "Policy A (human): $($r.policy_a)"
Write-Host "Policy B (ai):    $($r.policy_b)"
Write-Host "Seed:             $Seed"
if ($r.kibo_out) { Write-Host "Kibo Out:         $($r.kibo_out)" }
if ($r.dataset_out) {
  Write-Host "Dataset Out:      $($r.dataset_out)"
  Write-Host "Dataset Actor:    $($r.dataset_actor)"
  Write-Host "Dataset Rows:     $($r.dataset_rows) (pos=$($r.dataset_positive_rows), decisions=$($r.dataset_decisions), unresolved=$($r.dataset_unresolved_decisions))"
}
if ($r.unresolved_out) {
  Write-Host "Unresolved Out:   $($r.unresolved_out)"
  Write-Host "Unresolved Rows:  $($r.unresolved_rows) (decisions=$($r.unresolved_decisions), rate=$($r.unresolved_decision_rate))"
}
Write-Host "Wins A/B/Draw:    $($r.wins_a) / $($r.wins_b) / $($r.draws)"
Write-Host "WinRate A/B:      $($r.win_rate_a) / $($r.win_rate_b)"
Write-Host "GO count A/B:     $($r.go_count_a) / $($r.go_count_b)"
Write-Host "GO fail A/B:      $($r.go_fail_count_a) / $($r.go_fail_count_b)"
Write-Host "GO failRate A/B:  $($r.go_fail_rate_a) / $($r.go_fail_rate_b)"
Write-Host "MeanGoldDelta A:  $($r.mean_gold_delta_a)"
Write-Host ""
Write-Host "[A split]"
Write-Host "A 선공 W/L/D:      $($r.seat_split_a.when_first.wins) / $($r.seat_split_a.when_first.losses) / $($r.seat_split_a.when_first.draws)"
Write-Host "A 선공 WinRate:    $($r.seat_split_a.when_first.win_rate)"
Write-Host "A 선공 GO/실패:    $($r.seat_split_a.when_first.go_count_total) / $($r.seat_split_a.when_first.go_fail_count)"
Write-Host "A 선공 GoldDelta:  $($r.seat_split_a.when_first.mean_gold_delta)"
Write-Host "A 후공 W/L/D:      $($r.seat_split_a.when_second.wins) / $($r.seat_split_a.when_second.losses) / $($r.seat_split_a.when_second.draws)"
Write-Host "A 후공 WinRate:    $($r.seat_split_a.when_second.win_rate)"
Write-Host "A 후공 GO/실패:    $($r.seat_split_a.when_second.go_count_total) / $($r.seat_split_a.when_second.go_fail_count)"
Write-Host "A 후공 GoldDelta:  $($r.seat_split_a.when_second.mean_gold_delta)"
Write-Host "A 합산 W/L/D:      $($r.seat_split_a.combined.wins) / $($r.seat_split_a.combined.losses) / $($r.seat_split_a.combined.draws)"
Write-Host "A 합산 WinRate:    $($r.seat_split_a.combined.win_rate)"
Write-Host "A 합산 GO/실패:    $($r.seat_split_a.combined.go_count_total) / $($r.seat_split_a.combined.go_fail_count)"
Write-Host "A 합산 GoldDelta:  $($r.seat_split_a.combined.mean_gold_delta)"
Write-Host ""
Write-Host "[B split]"
Write-Host "B 선공 W/L/D:      $($r.seat_split_b.when_first.wins) / $($r.seat_split_b.when_first.losses) / $($r.seat_split_b.when_first.draws)"
Write-Host "B 선공 WinRate:    $($r.seat_split_b.when_first.win_rate)"
Write-Host "B 선공 GO/실패:    $($r.seat_split_b.when_first.go_count_total) / $($r.seat_split_b.when_first.go_fail_count)"
Write-Host "B 선공 GoldDelta:  $($r.seat_split_b.when_first.mean_gold_delta)"
Write-Host "B 후공 W/L/D:      $($r.seat_split_b.when_second.wins) / $($r.seat_split_b.when_second.losses) / $($r.seat_split_b.when_second.draws)"
Write-Host "B 후공 WinRate:    $($r.seat_split_b.when_second.win_rate)"
Write-Host "B 후공 GO/실패:    $($r.seat_split_b.when_second.go_count_total) / $($r.seat_split_b.when_second.go_fail_count)"
Write-Host "B 후공 GoldDelta:  $($r.seat_split_b.when_second.mean_gold_delta)"
Write-Host "B 합산 W/L/D:      $($r.seat_split_b.combined.wins) / $($r.seat_split_b.combined.losses) / $($r.seat_split_b.combined.draws)"
Write-Host "B 합산 WinRate:    $($r.seat_split_b.combined.win_rate)"
Write-Host "B 합산 GO/실패:    $($r.seat_split_b.combined.go_count_total) / $($r.seat_split_b.combined.go_fail_count)"
Write-Host "B 합산 GoldDelta:  $($r.seat_split_b.combined.mean_gold_delta)"
Write-Host "Saved:            $([System.IO.Path]::GetFullPath($savePath))"
Write-Host "==================================="

exit 0
