param(
  [Parameter(Mandatory = $true)][string]$RuntimeConfig,
  [Parameter(Mandatory = $false)][string]$Python = ".\.venv\Scripts\python",
  [Parameter(Mandatory = $false)][string]$Seed = "",
  [Parameter(Mandatory = $false)][string]$OutputDir = "",
  [Parameter(Mandatory = $false)][string]$ResumeCheckpoint = "",
  [Parameter(Mandatory = $false)][Nullable[int]]$TotalUpdates = $null,
  [Parameter(Mandatory = $false)][Nullable[int]]$LogEveryUpdates = $null,
  [Parameter(Mandatory = $false)][Nullable[int]]$SaveEveryUpdates = $null,
  [Parameter(Mandatory = $false)][switch]$DisableTorchCompile = $true
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if (-not (Test-Path $RuntimeConfig)) {
  throw "runtime config not found: $RuntimeConfig"
}
if (-not (Get-Command $Python -ErrorAction SilentlyContinue)) {
  throw "python executable not found: $Python"
}

function Get-OptionalPropertyString {
  param(
    [Parameter(Mandatory = $false)]$Object,
    [Parameter(Mandatory = $true)][string]$Name
  )
  if ($null -eq $Object) {
    return ""
  }
  $prop = $Object.PSObject.Properties[$Name]
  if ($null -eq $prop) {
    return ""
  }
  return [string]$prop.Value
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
$runtimeOutputDirTemplate = Get-OptionalPropertyString -Object $runtimeObj -Name "output_dir_template"
$runtimeOutputDir = Get-OptionalPropertyString -Object $runtimeObj -Name "output_dir"
$runtimeResumeTemplate = Get-OptionalPropertyString -Object $runtimeObj -Name "resume_checkpoint_template"
$runtimeResume = Get-OptionalPropertyString -Object $runtimeObj -Name "resume_checkpoint"
if (-not [string]::IsNullOrWhiteSpace($Seed)) {
  if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $autoOutputDir = Resolve-SeedPath `
      -Template $runtimeOutputDirTemplate `
      -Fallback $runtimeOutputDir `
      -SeedValue $Seed
  }
  if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    $autoResumeCheckpoint = Resolve-SeedPath `
      -Template $runtimeResumeTemplate `
      -Fallback $runtimeResume `
      -SeedValue $Seed
  }
}

$args = @(
  "ppo_by_GPT/scripts/train_ppo.py",
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
if ($null -ne $TotalUpdates) {
  if ($TotalUpdates -le 0) {
    throw "TotalUpdates must be > 0 when provided, got: $TotalUpdates"
  }
  $args += @("--total-updates", "$TotalUpdates")
}
if ($null -ne $LogEveryUpdates) {
  if ($LogEveryUpdates -le 0) {
    throw "LogEveryUpdates must be > 0 when provided, got: $LogEveryUpdates"
  }
  $args += @("--log-every-updates", "$LogEveryUpdates")
}
if ($null -ne $SaveEveryUpdates) {
  if ($SaveEveryUpdates -le 0) {
    throw "SaveEveryUpdates must be > 0 when provided, got: $SaveEveryUpdates"
  }
  $args += @("--save-every-updates", "$SaveEveryUpdates")
}

Write-Host "=== PPO Train Start ==="
Write-Host "Python: $Python"
Write-Host "Runtime: $RuntimeConfig"
if (-not [string]::IsNullOrWhiteSpace($Seed)) { Write-Host "Seed override: $Seed" }
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) { Write-Host "Output override: $OutputDir" }
elseif (-not [string]::IsNullOrWhiteSpace($autoOutputDir)) { Write-Host "Output auto: $autoOutputDir" }
if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) { Write-Host "Resume: $ResumeCheckpoint" }
elseif (-not [string]::IsNullOrWhiteSpace($autoResumeCheckpoint)) { Write-Host "Resume auto: $autoResumeCheckpoint" }
if ($null -ne $TotalUpdates) { Write-Host "Total updates override: $TotalUpdates" }
if ($null -ne $LogEveryUpdates) { Write-Host "Log every updates override: $LogEveryUpdates" }
if ($null -ne $SaveEveryUpdates) { Write-Host "Save every updates override: $SaveEveryUpdates" }

if ($DisableTorchCompile) {
  $env:TORCH_COMPILE_DISABLE = "1"
  $env:TORCHDYNAMO_DISABLE = "1"
  Write-Host "Torch compile: disabled (TORCH_COMPILE_DISABLE=1, TORCHDYNAMO_DISABLE=1)"
}

& $Python @args
if ($LASTEXITCODE -ne 0) {
  throw "PPO training failed with exit code $LASTEXITCODE"
}
