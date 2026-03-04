param(
  [Parameter(Mandatory = $true)][string]$RuntimeConfig,
  [Parameter(Mandatory = $false)][string]$Python = "python",
  [Parameter(Mandatory = $false)][string]$Seed = "",
  [Parameter(Mandatory = $false)][string]$OutputDir = "",
  [Parameter(Mandatory = $false)][string]$ResumeCheckpoint = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $RuntimeConfig)) {
  throw "runtime config not found: $RuntimeConfig"
}

function Resolve-SeedPath {
  param(
    [Parameter(Mandatory = $false)][string]$Template,
    [Parameter(Mandatory = $false)][string]$Fallback,
    [Parameter(Mandatory = $true)][string]$SeedValue
  )

  $tpl = [string]$Template
  if (-not [string]::IsNullOrWhiteSpace($tpl)) {
    return $tpl.Replace("{seed}", $SeedValue)
  }

  $base = [string]$Fallback
  if ([string]::IsNullOrWhiteSpace($base)) {
    return ""
  }
  $replaced = [regex]::Replace($base, "seed\d+", "seed$SeedValue", "IgnoreCase")
  if ($replaced -ne $base) {
    return $replaced
  }
  return "${base}_seed$SeedValue"
}

$runtimeObj = Get-Content -Path $RuntimeConfig -Raw -Encoding UTF8 | ConvertFrom-Json
$autoOutputDir = ""
$autoResumeCheckpoint = ""
if (-not [string]::IsNullOrWhiteSpace($Seed)) {
  if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $autoOutputDir = Resolve-SeedPath `
      -Template ([string]($runtimeObj.output_dir_template)) `
      -Fallback ([string]($runtimeObj.output_dir)) `
      -SeedValue $Seed
  }
  if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    $autoResumeCheckpoint = Resolve-SeedPath `
      -Template ([string]($runtimeObj.resume_checkpoint_template)) `
      -Fallback ([string]($runtimeObj.resume_checkpoint)) `
      -SeedValue $Seed
  }
}

$args = @(
  "ppo/scripts/train_ppo.py",
  "--runtime-config", "$RuntimeConfig"
)

if (-not [string]::IsNullOrWhiteSpace($Seed)) {
  $args += @("--seed", "$Seed")
}
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
  $args += @("--output-dir", "$OutputDir")
}
elseif (-not [string]::IsNullOrWhiteSpace($autoOutputDir)) {
  $args += @("--output-dir", "$autoOutputDir")
}
if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
  $args += @("--resume-checkpoint", "$ResumeCheckpoint")
}
elseif (-not [string]::IsNullOrWhiteSpace($autoResumeCheckpoint)) {
  $args += @("--resume-checkpoint", "$autoResumeCheckpoint")
}

Write-Host "=== PPO Train Start ==="
Write-Host "Python: $Python"
Write-Host "Runtime: $RuntimeConfig"
if (-not [string]::IsNullOrWhiteSpace($Seed)) { Write-Host "Seed override: $Seed" }
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) { Write-Host "Output override: $OutputDir" }
elseif (-not [string]::IsNullOrWhiteSpace($autoOutputDir)) { Write-Host "Output auto: $autoOutputDir" }
if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) { Write-Host "Resume: $ResumeCheckpoint" }
elseif (-not [string]::IsNullOrWhiteSpace($autoResumeCheckpoint)) { Write-Host "Resume auto: $autoResumeCheckpoint" }

& $Python @args
if ($LASTEXITCODE -ne 0) {
  throw "PPO training failed with exit code $LASTEXITCODE"
}
