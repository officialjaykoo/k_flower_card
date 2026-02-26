# Pipeline Stage: Full-League Orchestrator
# Quick Read Map:
# 1) Validate policy set / game count
# 2) Build round-robin matchup list
# 3) Run each 1000-game matchup
# 4) Aggregate standings and metrics
# 5) Save league outputs

param(
  [int]$GamesPerMatch = 1000,
  [int]$MaxSteps = 600,
  [string]$OutputTag = "",
  [string[]]$Policies = @("heuristic_v3", "heuristic_v4", "heuristic_v5", "heuristic_v5plus", "heuristic_v6", "heuristic_v7_gold_digger")
)

if ($GamesPerMatch -ne 1000) {
  throw "Full-league benchmark is locked to 1000 games per matchup."
}

function Sanitize-FilePart {
  param([string]$Text)
  return (($Text -replace "[^A-Za-z0-9._-]", "_").Trim("_"))
}

function Ensure-UniqueLowerPolicies {
  param([string[]]$InputPolicies)
  $out = @()
  $seen = New-Object System.Collections.Generic.HashSet[string]
  foreach ($p in ($InputPolicies | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
    $v = $p.Trim().ToLowerInvariant()
    if ($seen.Add($v)) { $out += $v }
  }
  return $out
}

$policyList = Ensure-UniqueLowerPolicies -InputPolicies $Policies
if ($policyList.Count -lt 2) {
  throw "At least two distinct policies are required."
}

$ts = Get-Date -Format "yyyyMMdd_HHmmss"
$tag = if ([string]::IsNullOrWhiteSpace($OutputTag)) {
  "full_league_$ts"
} else {
  "full_league_${ts}_$(Sanitize-FilePart $OutputTag)"
}

$outputDir = Join-Path "logs/heuristic_duel/full_league" $tag
New-Item -ItemType Directory -Path $outputDir -Force | Out-Null

$enc = New-Object System.Text.UTF8Encoding($true)
$matchSummaries = @()

for ($i = 0; $i -lt $policyList.Count - 1; $i += 1) {
  for ($j = $i + 1; $j -lt $policyList.Count; $j += 1) {
    $policyA = $policyList[$i]
    $policyB = $policyList[$j]
    $seed = "full_league_${ts}_${policyA}_vs_${policyB}_$((Get-Random -Minimum 100000 -Maximum 999999))"

    $cmd = @(
      "scripts/heuristic_duel_worker.mjs",
      "--policy-a", $policyA,
      "--policy-b", $policyB,
      "--games", "$GamesPerMatch",
      "--seed", "$seed",
      "--max-steps", "$MaxSteps",
      "--first-turn-policy", "alternate",
      "--continuous-series", "1"
    )

    $resultLines = & node @cmd
    if ($LASTEXITCODE -ne 0) {
      throw "duel worker failed: $policyA vs $policyB"
    }

    $resultJson = $resultLines | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace($resultJson)) {
      throw "empty duel output: $policyA vs $policyB"
    }

    $saveName = "{0}_vs_{1}_{2}.json" -f (Sanitize-FilePart $policyA), (Sanitize-FilePart $policyB), $GamesPerMatch
    $savePath = Join-Path $outputDir $saveName
    [System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($savePath), $resultJson, $enc)

    $obj = $resultJson | ConvertFrom-Json
    $matchSummaries += [PSCustomObject]@{
      policy_a         = [string]$obj.policy_a
      policy_b         = [string]$obj.policy_b
      games            = [int]$obj.games
      wins_a           = [int]$obj.wins_a
      wins_b           = [int]$obj.wins_b
      draws            = [int]$obj.draws
      bankrupt_a       = [int]$obj.bankrupt.a_bankrupt_count
      bankrupt_b       = [int]$obj.bankrupt.b_bankrupt_count
      go_count_a       = [int]$obj.go_count_a
      go_count_b       = [int]$obj.go_count_b
      go_games_a       = [int]$obj.go_games_a
      go_games_b       = [int]$obj.go_games_b
      go_fail_count_a  = [int]$obj.go_fail_count_a
      go_fail_count_b  = [int]$obj.go_fail_count_b
      mean_gold_delta_a = [double]$obj.mean_gold_delta_a
      file             = [System.IO.Path]::GetFullPath($savePath)
    }

    Write-Host ("Saved matchup: {0}" -f [System.IO.Path]::GetFullPath($savePath))
  }
}

$board = @{}
foreach ($p in $policyList) {
  $board[$p] = [PSCustomObject]@{
    policy = $p
    games = 0
    wins = 0
    losses = 0
    draws = 0
    go_count = 0
    go_games = 0
    go_fail_count = 0
    bankrupt_inflicted = 0
    bankrupt_suffered = 0
    gold_delta_total = 0.0
  }
}

foreach ($m in $matchSummaries) {
  $a = $board[$m.policy_a]
  $b = $board[$m.policy_b]
  if (-not $a -or -not $b) { continue }

  $a.games += $m.games
  $a.wins += $m.wins_a
  $a.losses += $m.wins_b
  $a.draws += $m.draws
  $a.go_count += $m.go_count_a
  $a.go_games += $m.go_games_a
  $a.go_fail_count += $m.go_fail_count_a
  $a.bankrupt_suffered += $m.bankrupt_a
  $a.bankrupt_inflicted += $m.bankrupt_b
  $a.gold_delta_total += ($m.mean_gold_delta_a * $m.games)

  $b.games += $m.games
  $b.wins += $m.wins_b
  $b.losses += $m.wins_a
  $b.draws += $m.draws
  $b.go_count += $m.go_count_b
  $b.go_games += $m.go_games_b
  $b.go_fail_count += $m.go_fail_count_b
  $b.bankrupt_suffered += $m.bankrupt_b
  $b.bankrupt_inflicted += $m.bankrupt_a
  $b.gold_delta_total += ((-$m.mean_gold_delta_a) * $m.games)
}

$standings = @()
foreach ($p in $policyList) {
  $x = $board[$p]
  $standings += [PSCustomObject]@{
    policy = $x.policy
    games = $x.games
    wins = $x.wins
    losses = $x.losses
    draws = $x.draws
    win_rate = if ($x.games -gt 0) { $x.wins / $x.games } else { 0 }
    mean_gold_delta = if ($x.games -gt 0) { [Math]::Round(($x.gold_delta_total / $x.games), 0) } else { 0 }
    go_count = $x.go_count
    go_fail_count = $x.go_fail_count
    bankrupt_inflicted = $x.bankrupt_inflicted
    bankrupt_suffered = $x.bankrupt_suffered
    go_fail_rate = if ($x.go_games -gt 0) { $x.go_fail_count / $x.go_games } else { 0 }
    win_rate_pct = [Math]::Round($(if ($x.games -gt 0) { ($x.wins * 100.0) / $x.games } else { 0 }), 1)
    go_fail_rate_pct = [Math]::Round($(if ($x.go_games -gt 0) { ($x.go_fail_count * 100.0) / $x.go_games } else { 0 }), 1)
  }
}

$standings = $standings | Sort-Object -Property @{ Expression = "wins"; Descending = $true }, @{ Expression = "mean_gold_delta"; Descending = $true }

$summary = [ordered]@{
  generated_at = (Get-Date).ToString("o")
  games_per_match = $GamesPerMatch
  policies = $policyList
  matchup_count = $matchSummaries.Count
  matchups = $matchSummaries
  standings = $standings
}

$summaryPath = Join-Path $outputDir "full_league_summary.json"
$summaryJson = $summary | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($summaryPath), $summaryJson, $enc)

Write-Host ""
Write-Host "=== Full League Completed ==="
Write-Host "Output:  $([System.IO.Path]::GetFullPath($outputDir))"
Write-Host "Summary: $([System.IO.Path]::GetFullPath($summaryPath))"
Write-Host ""
$displayStandings = $standings | Select-Object `
  policy,
  @{ Name = "G"; Expression = { $_.games } },
  @{ Name = "W"; Expression = { $_.wins } },
  @{ Name = "L"; Expression = { $_.losses } },
  @{ Name = "DRAWS"; Expression = { $_.draws } },
  @{ Name = "WIN"; Expression = { "{0:N1}%" -f $_.win_rate_pct } },
  @{ Name = "GD_DELTA"; Expression = { "{0:N0}" -f ([Math]::Round($_.mean_gold_delta, 0)) } },
  @{ Name = "GO"; Expression = { $_.go_count } },
  @{ Name = "GO_FAIL"; Expression = { $_.go_fail_count } },
  @{ Name = "GO_FAIL_RATE"; Expression = { "{0:N1}%" -f $_.go_fail_rate_pct } },
  @{ Name = "BNK_INF"; Expression = { $_.bankrupt_inflicted } },
  @{ Name = "BNK_SUFF"; Expression = { $_.bankrupt_suffered } }
Write-Host ($displayStandings | Format-Table -AutoSize | Out-String -Width 4096)
