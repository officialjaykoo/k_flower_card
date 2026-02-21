param(
  [Parameter(Mandatory = $true)]
  [string]$CandidateTag,

  [Parameter(Mandatory = $true)]
  [string]$CandidateAttackModel,

  [Parameter(Mandatory = $true)]
  [string]$CandidateDefenseModel,

  [string]$CandidateValueModel = "",

  [string]$V16AttackModel = "models/policy-danmokv16-attack.json",
  [string]$V16DefenseModel = "models/policy-danmokv16-defense.json",
  [string]$V16ValueModel = "models/value-danmokv16-gold.json",

  [int]$Workers = 4,
  [int]$Games = 1000,

  [switch]$Execute
)

if ($Games -ne 1000) {
  Write-Error "Baseline Lock protocol requires exactly 1000 games per run."
  exit 2
}
if ($Workers -le 0) {
  Write-Error "Workers must be > 0."
  exit 2
}

$required = @($CandidateAttackModel, $CandidateDefenseModel, $V16AttackModel, $V16DefenseModel)
foreach ($p in $required) {
  if (-not (Test-Path $p)) {
    Write-Error "Missing required model file: $p"
    exit 2
  }
}
if (-not [string]::IsNullOrWhiteSpace($CandidateValueModel) -and -not (Test-Path $CandidateValueModel)) {
  Write-Error "Missing candidate value model file: $CandidateValueModel"
  exit 2
}
if (-not [string]::IsNullOrWhiteSpace($V16ValueModel) -and -not (Test-Path $V16ValueModel)) {
  Write-Error "Missing V16 value model file: $V16ValueModel"
  exit 2
}

if (-not (Test-Path logs)) {
  New-Item -Path logs -ItemType Directory | Out-Null
}

function Build-Command {
  param(
    [string]$OutputLog,
    [string]$PolicyMy,
    [string]$PolicyYour,
    [string]$AttackMy = "",
    [string]$DefenseMy = "",
    [string]$ValueMy = "",
    [string]$AttackYour = "",
    [string]$DefenseYour = "",
    [string]$ValueYour = "",
    [bool]$ModelOnly = $false
  )

  $cmd = @(
    "py", "-3", "scripts/run_parallel_selfplay.py", "$Games",
    "--workers", "$Workers",
    "--output", $OutputLog,
    "--",
    "--log-mode=train",
    "--policy-my-side=$PolicyMy",
    "--policy-your-side=$PolicyYour"
  )
  if (-not [string]::IsNullOrWhiteSpace($AttackMy)) {
    $cmd += "--policy-model-attack-my-side=$AttackMy"
  }
  if (-not [string]::IsNullOrWhiteSpace($DefenseMy)) {
    $cmd += "--policy-model-defense-my-side=$DefenseMy"
  }
  if (-not [string]::IsNullOrWhiteSpace($ValueMy)) {
    $cmd += "--value-model-my-side=$ValueMy"
  }
  if (-not [string]::IsNullOrWhiteSpace($AttackYour)) {
    $cmd += "--policy-model-attack-your-side=$AttackYour"
  }
  if (-not [string]::IsNullOrWhiteSpace($DefenseYour)) {
    $cmd += "--policy-model-defense-your-side=$DefenseYour"
  }
  if (-not [string]::IsNullOrWhiteSpace($ValueYour)) {
    $cmd += "--value-model-your-side=$ValueYour"
  }
  if ($ModelOnly) {
    $cmd += "--model-only"
  }
  return ,$cmd
}

$tag = $CandidateTag.Trim()
$run1 = "logs/$tag" + "_vs_v16_1000.jsonl"
$run2 = "logs/v16_vs_$tag" + "_1000.jsonl"
$run3 = "logs/$tag" + "_vs_v4_1000.jsonl"
$run4 = "logs/v4_vs_$tag" + "_1000.jsonl"

$commands = @()

# 1) Candidate vs V16 (both model sides, strict model-only)
$commands += ,(Build-Command `
  -OutputLog $run1 `
  -PolicyMy "heuristic_v3" `
  -PolicyYour "heuristic_v3" `
  -AttackMy $CandidateAttackModel `
  -DefenseMy $CandidateDefenseModel `
  -ValueMy $CandidateValueModel `
  -AttackYour $V16AttackModel `
  -DefenseYour $V16DefenseModel `
  -ValueYour $V16ValueModel `
  -ModelOnly $true)

# 2) V16 vs Candidate (both model sides, strict model-only)
$commands += ,(Build-Command `
  -OutputLog $run2 `
  -PolicyMy "heuristic_v3" `
  -PolicyYour "heuristic_v3" `
  -AttackMy $V16AttackModel `
  -DefenseMy $V16DefenseModel `
  -ValueMy $V16ValueModel `
  -AttackYour $CandidateAttackModel `
  -DefenseYour $CandidateDefenseModel `
  -ValueYour $CandidateValueModel `
  -ModelOnly $true)

# 3) Candidate vs V4 (heuristic anchor side present, no global model-only)
$commands += ,(Build-Command `
  -OutputLog $run3 `
  -PolicyMy "heuristic_v3" `
  -PolicyYour "heuristic_v4" `
  -AttackMy $CandidateAttackModel `
  -DefenseMy $CandidateDefenseModel `
  -ValueMy $CandidateValueModel `
  -ModelOnly $false)

# 4) V4 vs Candidate (heuristic anchor side present, no global model-only)
$commands += ,(Build-Command `
  -OutputLog $run4 `
  -PolicyMy "heuristic_v4" `
  -PolicyYour "heuristic_v3" `
  -AttackYour $CandidateAttackModel `
  -DefenseYour $CandidateDefenseModel `
  -ValueYour $CandidateValueModel `
  -ModelOnly $false)

$plan = [ordered]@{
  baselineLock = "v1"
  candidateTag = $tag
  gamesPerRun = $Games
  workers = $Workers
  runs = @(
    [ordered]@{ name = "$tag vs v16"; output = $run1 },
    [ordered]@{ name = "v16 vs $tag"; output = $run2 },
    [ordered]@{ name = "$tag vs v4"; output = $run3 },
    [ordered]@{ name = "v4 vs $tag"; output = $run4 }
  )
}
$planPath = "logs/$tag" + "_baseline_lock_plan.json"
$plan | ConvertTo-Json -Depth 8 | Set-Content -Encoding utf8 $planPath

Write-Host "Baseline lock plan saved: $planPath"
Write-Host ""
Write-Host "Commands:"
for ($i = 0; $i -lt $commands.Count; $i++) {
  $line = ($commands[$i] -join " ")
  Write-Host "[$($i+1)] $line"
}

if (-not $Execute) {
  Write-Host ""
  Write-Host "Dry-run mode. Add -Execute to run."
  exit 0
}

for ($i = 0; $i -lt $commands.Count; $i++) {
  Write-Host ""
  Write-Host "Running [$($i+1)/$($commands.Count)] ..."
  & $commands[$i][0] $commands[$i][1..($commands[$i].Count-1)]
  if ($LASTEXITCODE -ne 0) {
    Write-Error "Failed at step $($i+1)."
    exit $LASTEXITCODE
  }
}

Write-Host ""
Write-Host "Baseline lock run completed."
Write-Host "Expected reports:"
Write-Host " - $($run1 -replace '\.jsonl$','-report.json')"
Write-Host " - $($run2 -replace '\.jsonl$','-report.json')"
Write-Host " - $($run3 -replace '\.jsonl$','-report.json')"
Write-Host " - $($run4 -replace '\.jsonl$','-report.json')"

