param(
  [int]$TrainGames = 200000,
  [int]$Workers = 4,
  [string]$Tag = "",
  [string]$LeagueConfig = "",
  [int]$LeagueChunkGames = 20000,
  [string]$PolicyMySide = "heuristic_v3",
  [string]$PolicyYourSide = "heuristic_v3",
  [string]$PolicyModelMySide = "",
  [string]$ValueModelMySide = "",
  [string]$PolicyModelYourSide = "",
  [string]$ValueModelYourSide = "",
  [switch]$ModelOnly,
  [string]$ValueDevice = "cpu"
)

if ($TrainGames -le 0) {
  Write-Error "TrainGames must be > 0."
  exit 2
}
if ($TrainGames % 2 -ne 0) {
  Write-Error "TrainGames must be even."
  exit 2
}
if ($Workers -le 0) {
  Write-Error "Workers must be > 0."
  exit 2
}

$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$baseTag = if ([string]::IsNullOrWhiteSpace($Tag)) { "fast-$stamp" } else { "$Tag-$stamp" }
$trainLog = "logs/train-$baseTag.jsonl"

if (-not [string]::IsNullOrWhiteSpace($LeagueConfig)) {
  $genArgs = @(
    "-3", "scripts/run_league_selfplay.py",
    "--config", $LeagueConfig,
    "--total-games", "$TrainGames",
    "--workers", "$Workers",
    "--chunk-games", "$LeagueChunkGames",
    "--output", $trainLog
  )
} else {
  $genArgs = @(
    "-3", "scripts/run_parallel_selfplay.py",
    "$TrainGames",
    "--workers", "$Workers",
    "--output", $trainLog,
    "--",
    "--log-mode=train",
    "--policy-my-side=$PolicyMySide",
    "--policy-your-side=$PolicyYourSide"
  )

  if (-not [string]::IsNullOrWhiteSpace($PolicyModelMySide)) {
    $genArgs += "--policy-model-my-side=$PolicyModelMySide"
  }
  if (-not [string]::IsNullOrWhiteSpace($ValueModelMySide)) {
    $genArgs += "--value-model-my-side=$ValueModelMySide"
  }
  if (-not [string]::IsNullOrWhiteSpace($PolicyModelYourSide)) {
    $genArgs += "--policy-model-your-side=$PolicyModelYourSide"
  }
  if (-not [string]::IsNullOrWhiteSpace($ValueModelYourSide)) {
    $genArgs += "--value-model-your-side=$ValueModelYourSide"
  }
  if ($ModelOnly) {
    $genArgs += "--model-only"
  }
}

Write-Host "> py $($genArgs -join ' ')"
& py @genArgs
if ($LASTEXITCODE -ne 0) {
  Write-Error "Parallel self-play failed."
  exit $LASTEXITCODE
}

$policyOut = "models/policy-$baseTag.json"
$valueOut = "models/value-$baseTag.json"

if (-not (Test-Path "models")) {
  New-Item -Path "models" -ItemType Directory | Out-Null
}

$policyArgs = @(
  "-3", "scripts/01_train_policy.py",
  "--input", $trainLog,
  "--output", $policyOut,
  "--max-samples", "100000",
  "--skip-train-metrics"
)
Write-Host "> py $($policyArgs -join ' ')"
& py @policyArgs
if ($LASTEXITCODE -ne 0) {
  Write-Error "Policy training failed."
  exit $LASTEXITCODE
}

$valueArgs = @(
  "-3", "scripts/02_train_value.py",
  "--input", $trainLog,
  "--output", $valueOut,
  "--max-samples", "200000",
  "--device", $ValueDevice,
  "--dim", "128",
  "--epochs", "1",
  "--batch-size", "8192",
  "--progress-every", "0"
)
Write-Host "> py $($valueArgs -join ' ')"
& py @valueArgs
if ($LASTEXITCODE -ne 0) {
  Write-Error "Value training failed."
  exit $LASTEXITCODE
}

Write-Host "Fast loop completed."
Write-Host "Train log: $trainLog"
Write-Host "Tag: $baseTag"
Write-Host "Policy model: $policyOut"
Write-Host "Value model: $valueOut"

