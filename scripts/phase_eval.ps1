param(
  [Parameter(Mandatory = $true)][ValidateSet("1", "2", "3")][string]$Phase,
  [Parameter(Mandatory = $true)][int]$Seed
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

function Get-RequiredDouble {
  param(
    [Parameter(Mandatory = $true)]$Object,
    [Parameter(Mandatory = $true)][string]$Key
  )
  if (-not ($Object.PSObject.Properties.Name -contains $Key)) {
    throw "runtime missing required key: $Key"
  }
  $v = Get-OptionalDouble -Value $Object.$Key -DefaultValue ([double]::NaN)
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
    if ($null -eq $Runtime.$k) {
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

function ConvertTo-NativeJsonArg {
  param([Parameter(Mandatory = $true)][string]$JsonText)
  # PowerShell native command invocation can strip raw double-quotes from JSON.
  # Escape quotes before passing to Node so JSON.parse receives valid text.
  return $JsonText.Replace('"', '\"')
}

$runtimeConfigPath = "scripts/configs/runtime_phase$Phase.json"
$outputDir = "logs/NEAT/neat_phase${Phase}_seed$Seed"
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
  $policyValue = [string]$runtime.opponent_policy
}
$hasPolicy = -not [string]::IsNullOrWhiteSpace($policyValue)
$hasPolicyMix = $false
$mixValue = $null
if ($runtime.PSObject.Properties.Name -contains "opponent_policy_mix") {
  $mixValue = $runtime.opponent_policy_mix
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
  "--max-steps", "$($runtime.max_eval_steps)",
  "--first-turn-policy", "alternate",
  "--fitness-gold-scale", "$($runtime.fitness_gold_scale)",
  "--fitness-gold-neutral-delta", "$($runtime.fitness_gold_neutral_delta)",
  "--fitness-win-weight", "$($runtime.fitness_win_weight)",
  "--fitness-gold-weight", "$($runtime.fitness_gold_weight)",
  "--fitness-win-neutral-rate", "$($runtime.fitness_win_neutral_rate)"
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
$goCount = [int](Get-OptionalDouble -Value $r.go_count -DefaultValue 0.0)
$goGames = [int](Get-OptionalDouble -Value $r.go_games -DefaultValue 0.0)
$goFailCount = [int](Get-OptionalDouble -Value $r.go_fail_count -DefaultValue 0.0)
$goFailRate = [double](Get-OptionalDouble -Value $r.go_fail_rate -DefaultValue 0.0)
$goRate = [double](Get-OptionalDouble -Value $r.go_rate -DefaultValue 0.0)

Write-Host ""
Write-Host "=== Phase$Phase Evaluation (Seed=$Seed) ==="
Write-Host "Win rate:        $($r.win_rate)"
Write-Host "Mean gold delta: $($r.mean_gold_delta)"
Write-Host "Fitness:         $($r.fitness)"
Write-Host "GO count:        $goCount"
Write-Host "GO games:        $goGames"
Write-Host "GO fail count:   $goFailCount"
Write-Host "GO fail rate:    $goFailRate"
Write-Host "GO rate:         $goRate"
Write-Host "Imit play ratio: $($r.imitation_play_ratio)"
Write-Host "Imit match ratio:$($r.imitation_match_ratio)"
Write-Host "Imit opt ratio:  $($r.imitation_option_ratio)"
Write-Host "Bankrupt count:  $($r.bankrupt.my_bankrupt_count)"
Write-Host "================================"

$passMeanGold = ([double]$r.mean_gold_delta -ge [double]$passRule.mean_gold_delta_min)
$passWinRate = ([double]$r.win_rate -ge [double]$passRule.win_rate_min)
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
  win_rate = [double]$r.win_rate
  mean_gold_delta = [double]$r.mean_gold_delta
  fitness = [double]$r.fitness
  go_count = $goCount
  go_games = $goGames
  go_fail_count = $goFailCount
  go_fail_rate = $goFailRate
  go_rate = $goRate
  eval_result_path = $savePath
  gate_state_path = $gateStatePath
  transition_ready = [bool]$gate.transition_ready
  transition_generation = $gate.transition_generation
}

$passStatePath = Join-Path $outputDir "phase${Phase}_pass_state.json"
$passStateJson = $passState | ConvertTo-Json -Depth 8
[System.IO.File]::WriteAllText([System.IO.Path]::GetFullPath($passStatePath), $passStateJson, $enc)
Write-Output $passStateJson

if ($passed) {
  exit 0
}

exit 2
