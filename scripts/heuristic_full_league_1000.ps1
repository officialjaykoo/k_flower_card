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
  [string]$ResumeFrom = "",
  [string]$Policies = "H-V4,H-V5,H-V5P,H-V6,H-V7"
)

if ($GamesPerMatch -ne 1000) {
  throw "Full-league benchmark is locked to 1000 games per matchup."
}

function ConvertTo-SafeFilePart {
  param([string]$Text)
  return (($Text -replace "[^A-Za-z0-9._-]", "_").Trim("_"))
}

function ConvertTo-UniqueLowerPolicies {
  param([string[]]$InputPolicies)
  $out = @()
  $seen = New-Object System.Collections.Generic.HashSet[string]
  foreach ($p in ($InputPolicies | Where-Object { -not [string]::IsNullOrWhiteSpace($_) })) {
    $v = ConvertTo-PolicyKey $p
    if ($seen.Add($v)) { $out += $v }
  }
  return $out
}

function ConvertFrom-PoliciesCsv {
  param([string]$CsvText)
  if ([string]::IsNullOrWhiteSpace($CsvText)) {
    return @()
  }
  $out = @()
  foreach ($part in $CsvText.Split(",")) {
    $token = ([string]$part).Trim()
    if (-not [string]::IsNullOrWhiteSpace($token)) {
      $out += $token
    }
  }
  return $out
}

function Convert-PolicyAliasToCanonical {
  param([string]$Value)
  $k = ([string]$Value).Trim().ToLowerInvariant()
  if ([string]::IsNullOrWhiteSpace($k)) { return "" }

  $known = @{
    "heuristic_v3" = "h-v3"
    "heuristic_v4" = "h-v4"
    "heuristic_v5" = "h-v5"
    "heuristic_v6" = "h-v6"
    "heuristic_v7" = "h-v7"
    "heuristic_v5plus" = "h-v5p"
    "heuristic_v5p" = "h-v5p"
    "heuristic_v7_gold_digger" = "h-v7"
    "h-v5plus" = "h-v5p"
    "hv3" = "h-v3"
    "hv4" = "h-v4"
    "hv5" = "h-v5"
    "hv6" = "h-v6"
    "hv7" = "h-v7"
    "hv5p" = "h-v5p"
  }
  if ($known.ContainsKey($k)) {
    return $known[$k]
  }

  $mHv = [System.Text.RegularExpressions.Regex]::Match($k, "^h[\s_-]?v([0-9]+)$")
  if ($mHv.Success) {
    return ("h-v{0}" -f $mHv.Groups[1].Value)
  }

  $mHeuristic = [System.Text.RegularExpressions.Regex]::Match($k, "^heuristic[\s_-]?v([0-9]+)$")
  if ($mHeuristic.Success) {
    return ("h-v{0}" -f $mHeuristic.Groups[1].Value)
  }

  return $k
}

function ConvertTo-PolicyKey {
  param([AllowNull()][string]$Text)
  return (Convert-PolicyAliasToCanonical ([string]$Text))
}

function Get-MatchupKey {
  param(
    [string]$PolicyA,
    [string]$PolicyB
  )
  $a = ConvertTo-PolicyKey $PolicyA
  $b = ConvertTo-PolicyKey $PolicyB
  if ([string]::IsNullOrWhiteSpace($a) -or [string]::IsNullOrWhiteSpace($b)) {
    return ""
  }
  if ($a -le $b) {
    return "$a|$b"
  }
  return "$b|$a"
}

function Convert-DuelResultToMatchSummary {
  param(
    [psobject]$ResultObject,
    [string]$SourceFile = ""
  )

  if ($null -eq $ResultObject) { return $null }

  $humanKey = if ($null -ne $ResultObject.human -and -not [string]::IsNullOrWhiteSpace([string]$ResultObject.human)) {
    [string]$ResultObject.human
  } else {
    [string]$ResultObject.policy_a
  }
  $aiKey = if ($null -ne $ResultObject.ai -and -not [string]::IsNullOrWhiteSpace([string]$ResultObject.ai)) {
    [string]$ResultObject.ai
  } else {
    [string]$ResultObject.policy_b
  }

  $humanKey = ConvertTo-PolicyKey $humanKey
  $aiKey = ConvertTo-PolicyKey $aiKey
  if ([string]::IsNullOrWhiteSpace($humanKey) -or [string]::IsNullOrWhiteSpace($aiKey)) {
    return $null
  }

  $resolvedFile = if ([string]::IsNullOrWhiteSpace($SourceFile)) { "" } else { [System.IO.Path]::GetFullPath($SourceFile) }
  return [PSCustomObject]@{
    policy_a          = $humanKey
    policy_b          = $aiKey
    games             = [int]$ResultObject.games
    wins_a            = [int]$ResultObject.wins_a
    wins_b            = [int]$ResultObject.wins_b
    draws             = [int]$ResultObject.draws
    bankrupt_a        = [int]$ResultObject.bankrupt.a_bankrupt_count
    bankrupt_b        = [int]$ResultObject.bankrupt.b_bankrupt_count
    go_count_a        = [int]$ResultObject.go_count_a
    go_count_b        = [int]$ResultObject.go_count_b
    go_games_a        = [int]$ResultObject.go_games_a
    go_games_b        = [int]$ResultObject.go_games_b
    go_fail_count_a   = [int]$ResultObject.go_fail_count_a
    go_fail_count_b   = [int]$ResultObject.go_fail_count_b
    mean_gold_delta_a = [double]$ResultObject.mean_gold_delta_a
    file              = $resolvedFile
  }
}

function Get-FileLastWriteUtcTicks {
  param([string]$Path)
  if ([string]::IsNullOrWhiteSpace($Path)) { return [int64]0 }
  try {
    return [int64](Get-Item -LiteralPath $Path -ErrorAction Stop).LastWriteTimeUtc.Ticks
  } catch {
    return [int64]0
  }
}

function Set-LatestMatchSummary {
  param(
    [hashtable]$Map,
    [string]$MatchKey,
    [psobject]$Summary,
    [int64]$SourceTicks
  )
  if ([string]::IsNullOrWhiteSpace($MatchKey) -or $null -eq $Summary) { return }
  if (-not $Map.ContainsKey($MatchKey)) {
    $Map[$MatchKey] = [PSCustomObject]@{
      ticks = $SourceTicks
      summary = $Summary
    }
    return
  }

  $current = $Map[$MatchKey]
  $currentTicks = if ($null -ne $current) { [int64]$current.ticks } else { [int64]0 }
  if ($SourceTicks -ge $currentTicks) {
    $Map[$MatchKey] = [PSCustomObject]@{
      ticks = $SourceTicks
      summary = $Summary
    }
  }
}

$policyInputs = ConvertFrom-PoliciesCsv -CsvText $Policies
if ($policyInputs.Count -lt 2) {
  throw "Policies must be one comma-separated string with at least two policies. Example: -Policies ""H-V4,H-V5,H-V5P,H-V6"""
}

$policyList = ConvertTo-UniqueLowerPolicies -InputPolicies $policyInputs
if ($policyList.Count -lt 2) {
  throw "At least two distinct policies are required."
}
$policySet = New-Object System.Collections.Generic.HashSet[string]
foreach ($p in $policyList) {
  [void]$policySet.Add($p)
}

$enc = New-Object System.Text.UTF8Encoding($true)
$allMatchSummaries = @()
$latestMatchSummaries = @{}
$existingMatchupKeys = New-Object System.Collections.Generic.HashSet[string]

if ([string]::IsNullOrWhiteSpace($ResumeFrom)) {
  $ts = Get-Date -Format "yyyyMMdd_HHmmss"
  $tag = if ([string]::IsNullOrWhiteSpace($OutputTag)) {
    "full_league_$ts"
  } else {
    "full_league_${ts}_$(ConvertTo-SafeFilePart $OutputTag)"
  }
  $outputDir = Join-Path "logs/full_league" $tag
  New-Item -ItemType Directory -Path $outputDir -Force | Out-Null
} else {
  $resolvedResume = Resolve-Path -LiteralPath $ResumeFrom -ErrorAction Stop
  if ($resolvedResume.Count -ne 1) {
    throw "ResumeFrom must resolve to exactly one directory: $ResumeFrom"
  }
  $resumeDir = $resolvedResume.Path
  if (-not (Test-Path -LiteralPath $resumeDir -PathType Container)) {
    throw "ResumeFrom must point to an existing directory: $ResumeFrom"
  }
  $outputDir = $resumeDir

  $existingFiles = Get-ChildItem -Path $outputDir -File -Filter "*.json" |
    Where-Object { $_.Name -ne "full_league_summary.json" }

  foreach ($f in $existingFiles) {
    try {
      $txt = Get-Content -Raw -Encoding UTF8 $f.FullName
      if ([string]::IsNullOrWhiteSpace($txt)) { continue }
      $obj = $txt | ConvertFrom-Json
      $summary = Convert-DuelResultToMatchSummary -ResultObject $obj -SourceFile $f.FullName
      if ($null -eq $summary) { continue }
      if ([int]$summary.games -ne $GamesPerMatch) { continue }
      if (-not $policySet.Contains([string]$summary.policy_a) -or -not $policySet.Contains([string]$summary.policy_b)) {
        continue
      }
      $matchKey = Get-MatchupKey $summary.policy_a $summary.policy_b
      if ([string]::IsNullOrWhiteSpace($matchKey)) { continue }
      $allMatchSummaries += $summary
      Set-LatestMatchSummary -Map $latestMatchSummaries -MatchKey $matchKey -Summary $summary -SourceTicks ([int64]$f.LastWriteTimeUtc.Ticks)
      [void]$existingMatchupKeys.Add($matchKey)
    } catch {
      Write-Warning ("Skip invalid resume file: {0} ({1})" -f $f.FullName, $_.Exception.Message)
    }
  }

  Write-Host ("Resume mode: loaded {0} matchup files ({1} unique) from {2}" -f $allMatchSummaries.Count, $existingMatchupKeys.Count, [System.IO.Path]::GetFullPath($outputDir))
}

for ($i = 0; $i -lt $policyList.Count - 1; $i += 1) {
  for ($j = $i + 1; $j -lt $policyList.Count; $j += 1) {
    $policyA = $policyList[$i]
    $policyB = $policyList[$j]
    $pairKey = Get-MatchupKey $policyA $policyB
    if ($existingMatchupKeys.Contains($pairKey)) {
      Write-Host ("Skip existing matchup: {0} vs {1}" -f $policyA, $policyB)
      continue
    }
    $ts = Get-Date -Format "yyyyMMdd_HHmmss"
    $seed = "full_league_${ts}_${policyA}_vs_${policyB}_$((Get-Random -Minimum 100000 -Maximum 999999))"
    $saveName = "{0}_vs_{1}_{2}.json" -f (ConvertTo-SafeFilePart $policyA), (ConvertTo-SafeFilePart $policyB), $GamesPerMatch
    $savePath = Join-Path $outputDir $saveName

    $cmd = @(
      "scripts/model_duel_worker.mjs",
      "--human", $policyA,
      "--ai", $policyB,
      "--games", "$GamesPerMatch",
      "--seed", "$seed",
      "--result-out", "$savePath",
      "--max-steps", "$MaxSteps",
      "--first-turn-policy", "alternate",
      "--continuous-series", "1",
      "--stdout-format", "json"
    )

    $resultLines = & node @cmd
    if ($LASTEXITCODE -ne 0) {
      throw "duel worker failed: $policyA vs $policyB"
    }

    $resultJson = $resultLines | Where-Object { -not [string]::IsNullOrWhiteSpace($_) } | Select-Object -Last 1
    if ([string]::IsNullOrWhiteSpace($resultJson)) {
      throw "empty duel output: $policyA vs $policyB"
    }

    $obj = $resultJson | ConvertFrom-Json
    $summary = Convert-DuelResultToMatchSummary -ResultObject $obj -SourceFile $savePath
    if ($null -eq $summary) {
      throw "invalid duel output schema: $policyA vs $policyB"
    }
    $allMatchSummaries += $summary
    $summaryTicks = Get-FileLastWriteUtcTicks -Path $savePath
    Set-LatestMatchSummary -Map $latestMatchSummaries -MatchKey $pairKey -Summary $summary -SourceTicks $summaryTicks
    [void]$existingMatchupKeys.Add($pairKey)

    Write-Host ("Saved matchup: {0}" -f [System.IO.Path]::GetFullPath($savePath))
  }
}

$matchSummaries = @()
foreach ($key in ($latestMatchSummaries.Keys | Sort-Object)) {
  $entry = $latestMatchSummaries[$key]
  if ($null -ne $entry -and $null -ne $entry.summary) {
    $matchSummaries += $entry.summary
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

$expectedUniqueMatchupCount = [int](($policyList.Count * ($policyList.Count - 1)) / 2)
$uniqueMatchupCount = [int]$existingMatchupKeys.Count
$historyMatchupCount = [int]$allMatchSummaries.Count
$duplicateMatchupRuns = [Math]::Max(0, $historyMatchupCount - $uniqueMatchupCount)
$missingUniqueMatchups = [Math]::Max(0, $expectedUniqueMatchupCount - $uniqueMatchupCount)
$isLeagueComplete = ($missingUniqueMatchups -eq 0)

$summary = [ordered]@{
  generated_at = (Get-Date).ToString("o")
  games_per_match = $GamesPerMatch
  policies = $policyList
  expected_unique_matchup_count = $expectedUniqueMatchupCount
  unique_matchup_count = $uniqueMatchupCount
  history_matchup_count = $historyMatchupCount
  duplicate_matchup_runs = $duplicateMatchupRuns
  missing_unique_matchups = $missingUniqueMatchups
  is_complete = $isLeagueComplete
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
  @{ Name = "D"; Expression = { $_.draws } },
  @{ Name = "W%"; Expression = { "{0:N1}%" -f $_.win_rate_pct } },
  @{ Name = "BI"; Expression = { $_.bankrupt_inflicted } },
  @{ Name = "BS"; Expression = { $_.bankrupt_suffered } },
  @{ Name = "G_DT"; Expression = { "{0:N0}" -f ([Math]::Round($_.mean_gold_delta, 0)) } },
  @{ Name = "GO"; Expression = { $_.go_count } },
  @{ Name = "GO_F"; Expression = { $_.go_fail_count } },
  @{ Name = "GO_F%"; Expression = { "{0:N1}%" -f $_.go_fail_rate_pct } }
Write-Host ($displayStandings | Format-Table -AutoSize | Out-String -Width 4096)
