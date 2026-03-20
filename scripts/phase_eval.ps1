param(
  [Parameter(Mandatory = $true)][ValidateSet("1", "2", "3")][string]$Phase,
  [Parameter(Mandatory = $true)][int]$Seed,
  [Parameter(Mandatory = $false)][ValidateSet("classic")][string]$LineageProfile = "classic"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Read-JsonFile {
  param([Parameter(Mandatory = $true)][string]$Path)
  if (-not (Test-Path $Path)) {
    throw "json file not found: $Path"
  }
  return Get-Content $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}

function Get-OptionalDouble {
  param(
    [Parameter(Mandatory = $false)]$Value,
    [Parameter(Mandatory = $false)][double]$DefaultValue = [double]::NaN
  )
  if ($null -eq $Value) {
    return $DefaultValue
  }
  try {
    return [double]$Value
  }
  catch {
    return $DefaultValue
  }
}

function Clamp01 {
  param([Parameter(Mandatory = $true)][double]$Value)
  if ($Value -le 0.0) { return 0.0 }
  if ($Value -ge 1.0) { return 1.0 }
  return [double]$Value
}

function Get-PropertyValue {
  param(
    [Parameter(Mandatory = $false)]$Object,
    [Parameter(Mandatory = $true)][string]$Name
  )
  if ($null -eq $Object) {
    return $null
  }
  if ($Object.PSObject.Properties.Name -contains $Name) {
    return $Object.$Name
  }
  return $null
}

function Get-NestedPropertyValue {
  param(
    [Parameter(Mandatory = $false)]$Object,
    [Parameter(Mandatory = $true)][string[]]$Path
  )
  $current = $Object
  foreach ($segment in $Path) {
    $current = Get-PropertyValue -Object $current -Name $segment
    if ($null -eq $current) {
      return $null
    }
  }
  return $current
}

function Get-RequiredDouble {
  param(
    [Parameter(Mandatory = $true)]$Object,
    [Parameter(Mandatory = $true)][string]$Key
  )
  if (-not ((Get-PropertyValue -Object $Object -Name $Key) -ne $null -or $Object.PSObject.Properties.Name -contains $Key)) {
    throw "runtime missing required key: $Key"
  }
  $v = Get-OptionalDouble -Value (Get-PropertyValue -Object $Object -Name $Key) -DefaultValue ([double]::NaN)
  if ([double]::IsNaN($v)) {
    throw "runtime key '$Key' must be finite number"
  }
  return [double]$v
}

function Assert-RequiredRuntimeKeys {
  param(
    [Parameter(Mandatory = $true)]$Runtime,
    [Parameter(Mandatory = $true)][string[]]$Keys
  )
  foreach ($k in $Keys) {
    if (-not ($Runtime.PSObject.Properties.Name -contains $k)) {
      throw "runtime missing required key: $k"
    }
    if ($null -eq (Get-PropertyValue -Object $Runtime -Name $k)) {
      throw "runtime key '$k' must not be null"
    }
  }
}

function Resolve-EvalGateRule {
  param([Parameter(Mandatory = $true)]$Runtime)
  $meanGold = Get-RequiredDouble -Object $Runtime -Key "eval_pass_mean_gold_delta_min"
  $winRate = Get-RequiredDouble -Object $Runtime -Key "eval_pass_win_rate_min"

  return [ordered]@{
    mean_gold_delta_min = [double]$meanGold
    win_rate_min = [double]$winRate
  }
}

function Compute-EvalFitness {
  param(
    [Parameter(Mandatory = $true)]$Summary,
    [Parameter(Mandatory = $true)]$Runtime
  )

  $weightedWinRate = Get-RequiredDouble -Object $Summary -Key "win_rate_a"
  $weightedDrawRate = Get-RequiredDouble -Object $Summary -Key "draw_rate"
  $weightedLossRate = Get-RequiredDouble -Object $Summary -Key "win_rate_b"
  $weightedMeanGoldDelta = Get-RequiredDouble -Object $Summary -Key "mean_gold_delta_a"

  $fitnessGoldScale = Get-RequiredDouble -Object $Runtime -Key "fitness_gold_scale"
  $fitnessGoldNeutralDelta = Get-RequiredDouble -Object $Runtime -Key "fitness_gold_neutral_delta"
  $fitnessWinWeight = Get-RequiredDouble -Object $Runtime -Key "fitness_win_weight"
  $fitnessGoldWeight = Get-RequiredDouble -Object $Runtime -Key "fitness_gold_weight"
  $fitnessWinNeutralRate = Get-RequiredDouble -Object $Runtime -Key "fitness_win_neutral_rate"

  $goldNorm = [Math]::Tanh(($weightedMeanGoldDelta - $fitnessGoldNeutralDelta) / [Math]::Max(1e-9, $fitnessGoldScale))
  $expectedResultRaw = (Clamp01 $weightedWinRate) + (0.5 * (Clamp01 $weightedDrawRate)) - (Clamp01 $weightedLossRate)
  $expectedResult = [Math]::Max(-1.0, [Math]::Min(1.0, $expectedResultRaw))
  $neutralExpectedResult = (2.0 * $fitnessWinNeutralRate) - 1.0

  if ($expectedResult -ge $neutralExpectedResult) {
    $resultUpperSpan = [Math]::Max(1e-9, 1.0 - $neutralExpectedResult)
    $resultNorm = Clamp01 (($expectedResult - $neutralExpectedResult) / $resultUpperSpan)
  }
  else {
    $resultLowerSpan = [Math]::Max(1e-9, $neutralExpectedResult + 1.0)
    $resultNorm = -(Clamp01 (($neutralExpectedResult - $expectedResult) / $resultLowerSpan))
  }

  return ($fitnessGoldWeight * $goldNorm) + ($fitnessWinWeight * $resultNorm)
}

function ConvertTo-NativeJsonArg {
  param([Parameter(Mandatory = $true)][string]$JsonText)
  # PowerShell native command invocation can strip raw double-quotes from JSON.
  # Escape quotes before passing to Node so JSON.parse receives valid text.
  return $JsonText.Replace('"', '\"')
}

function Allocate-GamesByWeight {
  param(
    [Parameter(Mandatory = $true)][int]$TotalGames,
    [Parameter(Mandatory = $true)]$Entries
  )

  $safeTotalGames = [Math]::Max(1, [int]$TotalGames)
  if ($null -eq $Entries -or $Entries.Count -eq 0) {
    return @()
  }

  $weights = @()
  foreach ($entry in $Entries) {
    $weights += [double](Get-PropertyValue -Object $entry -Name "weight")
  }
  $totalWeight = 0.0
  foreach ($weight in $weights) {
    $totalWeight += $weight
  }
  if ($totalWeight -le 0.0) {
    throw "opponent_policy_mix weights must be positive"
  }

  $raw = @()
  $counts = @()
  foreach ($weight in $weights) {
    $value = $weight * $safeTotalGames / $totalWeight
    $raw += $value
    $counts += [Math]::Max(1, [int][Math]::Floor($value))
  }

  while (($counts | Measure-Object -Sum).Sum -gt $safeTotalGames) {
    $maxIndex = 0
    for ($i = 1; $i -lt $counts.Count; $i++) {
      if ($counts[$i] -gt $counts[$maxIndex]) {
        $maxIndex = $i
      }
    }
    if ($counts[$maxIndex] -le 1) {
      break
    }
    $counts[$maxIndex] -= 1
  }

  while (($counts | Measure-Object -Sum).Sum -lt $safeTotalGames) {
    $maxIndex = 0
    $maxGap = ($raw[0] - $counts[0])
    for ($i = 1; $i -lt $counts.Count; $i++) {
      $gap = $raw[$i] - $counts[$i]
      if ($gap -gt $maxGap) {
        $maxGap = $gap
        $maxIndex = $i
      }
    }
    $counts[$maxIndex] += 1
  }

  return ,$counts
}

function Get-LineageLayout {
  param([Parameter(Mandatory = $true)][string]$Profile)

  switch ($Profile) {
    "classic" {
      return [ordered]@{
        output_prefix = "neat"
      }
    }
    default {
      throw "unsupported lineage profile: $Profile"
    }
  }
}

$lineageLayout = Get-LineageLayout -Profile $LineageProfile
<#

  $savePath = Join-Path $outputDir "phase${Phase}_eval_1000.json"
  $firstTurnPolicy = [string](Get-PropertyValue -Object $runtime -Name "first_turn_policy")
  if ([string]::IsNullOrWhiteSpace($firstTurnPolicy)) {
    $firstTurnPolicy = "alternate"
  }
  $continuousSeries = $true
  if ($runtime.PSObject.Properties.Name -contains "continuous_series") {
    $continuousSeries = [bool](Get-PropertyValue -Object $runtime -Name "continuous_series")
  }
  if ($hasPolicy) {
    $singleResultPath = $savePath
    $cmd = @(
      "scripts/model_duel_worker.mjs",
      "--human", $winnerRuntimePath,
      "--ai", "$policyValue",
      "--games", "$games",
      "--seed", $seedTag,
      "--max-steps", "$(Get-PropertyValue -Object $runtime -Name 'max_eval_steps')",
      "--first-turn-policy", $firstTurnPolicy,
      "--continuous-series", "$(if ($continuousSeries) { '1' } else { '2' })",
      "--stdout-format", "json",
      "--result-out", $singleResultPath
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
    $r = $resultJson | ConvertFrom-Json
  }
  else {
    $counts = Allocate-GamesByWeight -TotalGames $games -Entries $mixValue
    $aggregateGames = 0.0
    $aggregateWinsA = 0.0
    $aggregateWinsB = 0.0
    $aggregateDraws = 0.0
    $aggregateGoldSumA = 0.0
    $aggregateGoCountA = 0.0
    $aggregateGoGamesA = 0.0
    $aggregateGoFailCountA = 0.0
    $aggregateGoOpportunityCountA = 0.0

    for ($i = 0; $i -lt $mixValue.Count; $i++) {
      $entry = $mixValue[$i]
      $policy = [string](Get-PropertyValue -Object $entry -Name "policy")
      $gamesForPolicy = [int]$counts[$i]
      $resultPathPart = Join-Path $outputDir ("phase{0}_eval_1000_part{1}.json" -f $Phase, $i)
      $cmd = @(
        "scripts/model_duel_worker.mjs",
        "--human", $winnerRuntimePath,
        "--ai", "$policy",
        "--games", "$gamesForPolicy",
        "--seed", "${seedTag}_$i",
        "--max-steps", "$(Get-PropertyValue -Object $runtime -Name 'max_eval_steps')",
        "--first-turn-policy", $firstTurnPolicy,
        "--continuous-series", "$(if ($continuousSeries) { '1' } else { '2' })",
        "--stdout-format", "json",
        "--result-out", $resultPathPart
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
      $part = $resultJson | ConvertFrom-Json
      try {
        [System.IO.File]::Delete([System.IO.Path]::GetFullPath($resultPathPart))
      }
      catch {
      }
      $partGames = [double](Get-PropertyValue -Object $part -Name "games")
      $aggregateGames += $partGames
      $aggregateWinsA += [double](Get-PropertyValue -Object $part -Name "wins_a")
      $aggregateWinsB += [double](Get-PropertyValue -Object $part -Name "wins_b")
      $aggregateDraws += [double](Get-PropertyValue -Object $part -Name "draws")
      $aggregateGoldSumA += ([double](Get-PropertyValue -Object $part -Name "mean_gold_delta_a")) * $partGames
      $aggregateGoCountA += [double](Get-PropertyValue -Object $part -Name "go_count_a")
      $aggregateGoGamesA += [double](Get-PropertyValue -Object $part -Name "go_games_a")
      $aggregateGoFailCountA += [double](Get-PropertyValue -Object $part -Name "go_fail_count_a")
      $aggregateGoOpportunityCountA += [double](Get-PropertyValue -Object $part -Name "go_opportunity_count_a")
    }

    $gamesTotal = [Math]::Max(1.0, $aggregateGames)
    $goTakeRateA = if ($aggregateGoOpportunityCountA -gt 0.0) { $aggregateGoCountA / $aggregateGoOpportunityCountA } else { 0.0 }
    $goFailRateA = if ($aggregateGoCountA -gt 0.0) { $aggregateGoFailCountA / $aggregateGoCountA } else { 0.0 }
    $r = [pscustomobject]@{
      games = [int]$aggregateGames
      wins_a = $aggregateWinsA
      wins_b = $aggregateWinsB
      draws = $aggregateDraws
      win_rate_a = $aggregateWinsA / $gamesTotal
      win_rate_b = $aggregateWinsB / $gamesTotal
      draw_rate = $aggregateDraws / $gamesTotal
      mean_gold_delta_a = $aggregateGoldSumA / $gamesTotal
      go_count_a = [int]$aggregateGoCountA
      go_games_a = [int]$aggregateGoGamesA
      go_fail_count_a = [int]$aggregateGoFailCountA
      go_fail_rate_a = $goFailRateA
      go_opportunity_count_a = [int]$aggregateGoOpportunityCountA
      go_take_rate_a = $goTakeRateA
      result_out = $savePath
    }
    $enc = New-Object System.Text.UTF8Encoding($true)
    [System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($savePath), ($r | ConvertTo-Json -Depth 8), $enc)
  }

  $fitness = Compute-EvalFitness -Summary $r -Runtime $runtime
  $goCount = [int](Get-OptionalDouble -Value (Get-PropertyValue -Object $r -Name "go_count_a") -DefaultValue 0.0)
  $goGames = [int](Get-OptionalDouble -Value (Get-PropertyValue -Object $r -Name "go_games_a") -DefaultValue 0.0)
  $goFailCount = [int](Get-OptionalDouble -Value (Get-PropertyValue -Object $r -Name "go_fail_count_a") -DefaultValue 0.0)
  $goFailRate = [double](Get-OptionalDouble -Value (Get-PropertyValue -Object $r -Name "go_fail_rate_a") -DefaultValue 0.0)
  $goRate = [double](Get-OptionalDouble -Value (Get-PropertyValue -Object $r -Name "go_take_rate_a") -DefaultValue 0.0)

  Write-Host ""
  Write-Host "=== Phase$Phase Evaluation (Seed=$Seed, Profile=$LineageProfile) ==="
  Write-Host "Win rate:        $(Get-PropertyValue -Object $r -Name 'win_rate_a')"
  Write-Host "Mean gold delta: $(Get-PropertyValue -Object $r -Name 'mean_gold_delta_a')"
  Write-Host "Fitness:         $fitness"
  Write-Host "GO count:        $goCount"
  Write-Host "GO games:        $goGames"
  Write-Host "GO fail count:   $goFailCount"
  Write-Host "GO fail rate:    $goFailRate"
  Write-Host "GO rate:         $goRate"
  Write-Host "================================"

  $passMeanGold = ([double](Get-PropertyValue -Object $r -Name "mean_gold_delta_a") -ge [double]$passRule.mean_gold_delta_min)
  $passWinRate = ([double](Get-PropertyValue -Object $r -Name "win_rate_a") -ge [double]$passRule.win_rate_min)
  $passed = $passMeanGold -and $passWinRate

  $failReasons = @()
  if (-not $passMeanGold) { $failReasons += "mean_gold_delta" }
  if (-not $passWinRate) { $failReasons += "win_rate" }
  $reasonText = if ($passed) { "eval_gate_passed" } else { "eval_gate_not_passed:" + ($failReasons -join ",") }
  $passState = [ordered]@{
    passed = $passed
    reason = $reasonText
    seed = "$Seed"
    phase = "phase$Phase"
    pass_rule = [ordered]@{
      mean_gold_delta_min = [double]$passRule.mean_gold_delta_min
      win_rate_min = [double]$passRule.win_rate_min
    }
    win_rate = [double](Get-PropertyValue -Object $r -Name "win_rate_a")
    mean_gold_delta = [double](Get-PropertyValue -Object $r -Name "mean_gold_delta_a")
    fitness = [double]$fitness
    go_count = $goCount
    go_games = $goGames
    go_fail_count = $goFailCount
    go_fail_rate = $goFailRate
    go_rate = $goRate
    eval_result_path = $savePath
    run_summary_path = $runSummaryPath
    transition_ready = $null
    transition_generation = $null
  }

  $passStatePath = Join-Path $outputDir "phase${Phase}_pass_state.json"
  $passStateJson = $passState | ConvertTo-Json -Depth 8
  $enc = New-Object System.Text.UTF8Encoding($true)
  [System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($passStatePath), $passStateJson, $enc)
  Write-Output $passStateJson
#>

$runtimeConfigPath = "scripts/configs/runtime_phase1.json"
$outputDir = "logs/NEAT/$($lineageLayout.output_prefix)_phase${Phase}_seed$Seed"
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

$runtime = Read-JsonFile -Path $runtimeConfigPath
$gate = Read-JsonFile -Path $gateStatePath
$passRule = Resolve-EvalGateRule -Runtime $runtime
Assert-RequiredRuntimeKeys -Runtime $runtime -Keys @(
  "max_eval_steps",
  "fitness_gold_scale",
  "fitness_gold_neutral_delta",
  "fitness_win_weight",
  "fitness_gold_weight",
  "fitness_win_neutral_rate"
)

$games = 1000
$seedTag = "phase${Phase}_eval_$Seed"
$policyValue = ""
if ($runtime.PSObject.Properties.Name -contains "opponent_policy") {
  $policyValue = [string](Get-PropertyValue -Object $runtime -Name "opponent_policy")
}
$hasPolicy = -not [string]::IsNullOrWhiteSpace($policyValue)
$hasPolicyMix = $false
$mixValue = $null
if ($runtime.PSObject.Properties.Name -contains "opponent_policy_mix") {
  $mixValue = Get-PropertyValue -Object $runtime -Name "opponent_policy_mix"
  $hasPolicyMix = ($null -ne $mixValue) -and ($mixValue.Count -gt 0)
}
if (-not $hasPolicy -and -not $hasPolicyMix) {
  throw "runtime must contain opponent_policy or opponent_policy_mix"
}

$cmd = @(
  "scripts/neat_eval_worker.mjs",
  "--genome", $genomePath,
  "--games", "$games",
  "--seed", $seedTag,
  "--max-steps", "$(Get-PropertyValue -Object $runtime -Name 'max_eval_steps')",
  "--first-turn-policy", "alternate",
  "--fitness-gold-scale", "$(Get-PropertyValue -Object $runtime -Name 'fitness_gold_scale')",
  "--fitness-gold-neutral-delta", "$(Get-PropertyValue -Object $runtime -Name 'fitness_gold_neutral_delta')",
  "--fitness-win-weight", "$(Get-PropertyValue -Object $runtime -Name 'fitness_win_weight')",
  "--fitness-gold-weight", "$(Get-PropertyValue -Object $runtime -Name 'fitness_gold_weight')",
  "--fitness-win-neutral-rate", "$(Get-PropertyValue -Object $runtime -Name 'fitness_win_neutral_rate')"
)
if (-not $hasPolicy -and $hasPolicyMix) {
  $mixJson = $mixValue | ConvertTo-Json -Depth 8 -Compress
  $mixArg = ConvertTo-NativeJsonArg -JsonText $mixJson
  # Use key=value single token to avoid PowerShell quote-stripping on native args.
  $cmd += @("--opponent-policy-mix=$mixArg")
}
if ($hasPolicy) {
  # opponent_policy 우선. 둘 다 있으면 opponent_policy를 사용한다.
  $cmd += @("--opponent-policy", "$policyValue")
}

$resultLines = & node @cmd
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
  exit $exitCode
}

$resultJson = $resultLines | Select-Object -Last 1
if ([string]::IsNullOrWhiteSpace($resultJson)) {
  throw "empty eval output"
}

$savePath = Join-Path $outputDir "phase${Phase}_eval_1000.json"
$enc = New-Object System.Text.UTF8Encoding($true)
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($savePath), $resultJson, $enc)

$r = $resultJson | ConvertFrom-Json
$goCount = [int](Get-OptionalDouble -Value (Get-PropertyValue -Object $r -Name "go_count") -DefaultValue 0.0)
$goGames = [int](Get-OptionalDouble -Value (Get-PropertyValue -Object $r -Name "go_games") -DefaultValue 0.0)
$goFailCount = [int](Get-OptionalDouble -Value (Get-PropertyValue -Object $r -Name "go_fail_count") -DefaultValue 0.0)
$goFailRate = [double](Get-OptionalDouble -Value (Get-PropertyValue -Object $r -Name "go_fail_rate") -DefaultValue 0.0)
$goRate = [double](Get-OptionalDouble -Value (Get-PropertyValue -Object $r -Name "go_rate") -DefaultValue 0.0)

Write-Host ""
Write-Host "=== Phase$Phase Evaluation (Seed=$Seed, Profile=$LineageProfile) ==="
Write-Host "Win rate:        $(Get-PropertyValue -Object $r -Name 'win_rate')"
Write-Host "Mean gold delta: $(Get-PropertyValue -Object $r -Name 'mean_gold_delta')"
Write-Host "Fitness:         $(Get-PropertyValue -Object $r -Name 'fitness')"
Write-Host "GO count:        $goCount"
Write-Host "GO games:        $goGames"
Write-Host "GO fail count:   $goFailCount"
Write-Host "GO fail rate:    $goFailRate"
Write-Host "GO rate:         $goRate"
Write-Host "Imit play ratio: $(Get-PropertyValue -Object $r -Name 'imitation_play_ratio')"
Write-Host "Imit match ratio:$(Get-PropertyValue -Object $r -Name 'imitation_match_ratio')"
Write-Host "Imit opt ratio:  $(Get-PropertyValue -Object $r -Name 'imitation_option_ratio')"
Write-Host "Bankrupt count:  $(Get-NestedPropertyValue -Object $r -Path @('bankrupt', 'my_bankrupt_count'))"
Write-Host "================================"

$passMeanGold = ([double](Get-PropertyValue -Object $r -Name "mean_gold_delta") -ge [double]$passRule.mean_gold_delta_min)
$passWinRate = ([double](Get-PropertyValue -Object $r -Name "win_rate") -ge [double]$passRule.win_rate_min)
$passed = $passMeanGold -and $passWinRate

$failReasons = @()
if (-not $passMeanGold) { $failReasons += "mean_gold_delta" }
if (-not $passWinRate) { $failReasons += "win_rate" }
$reasonText = if ($passed) { "eval_gate_passed" } else { "eval_gate_not_passed:" + ($failReasons -join ",") }
$passState = [ordered]@{
  passed = $passed
  reason = $reasonText
  seed = "$Seed"
  phase = "phase$Phase"
  pass_rule = [ordered]@{
    mean_gold_delta_min = [double]$passRule.mean_gold_delta_min
    win_rate_min = [double]$passRule.win_rate_min
  }
  win_rate = [double](Get-PropertyValue -Object $r -Name "win_rate")
  mean_gold_delta = [double](Get-PropertyValue -Object $r -Name "mean_gold_delta")
  fitness = [double](Get-PropertyValue -Object $r -Name "fitness")
  go_count = $goCount
  go_games = $goGames
  go_fail_count = $goFailCount
  go_fail_rate = $goFailRate
  go_rate = $goRate
  eval_result_path = $savePath
  gate_state_path = $gateStatePath
  transition_ready = [bool](Get-PropertyValue -Object $gate -Name "transition_ready")
  transition_generation = Get-PropertyValue -Object $gate -Name "transition_generation"
}

$passStatePath = Join-Path $outputDir "phase${Phase}_pass_state.json"
$passStateJson = $passState | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($passStatePath), $passStateJson, $enc)
Write-Output $passStateJson

if ($passed) {
  exit 0
}

exit 2
