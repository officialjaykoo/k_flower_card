function Get-NeatGptRepoRoot {
  param([Parameter(Mandatory = $true)][string]$ScriptRoot)
  return [System.IO.Path]::GetFullPath((Join-Path $ScriptRoot ".."))
}

function Resolve-PathFromBase {
  param(
    [Parameter(Mandatory = $false)][string]$Path,
    [Parameter(Mandatory = $true)][string]$BasePath,
    [Parameter(Mandatory = $false)][string]$DefaultPath = ""
  )
  $candidate = $Path
  if ([string]::IsNullOrWhiteSpace($candidate)) {
    $candidate = $DefaultPath
  }
  if ([string]::IsNullOrWhiteSpace($candidate)) {
    return ""
  }
  if ([System.IO.Path]::IsPathRooted($candidate)) {
    return [System.IO.Path]::GetFullPath($candidate)
  }
  return [System.IO.Path]::GetFullPath((Join-Path $BasePath $candidate))
}

function Resolve-NeatGptOutputDirPath {
  param(
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$RuntimeConfigPath,
    [Parameter(Mandatory = $true)][int]$SeedValue,
    [Parameter(Mandatory = $false)][string]$ExplicitOutputDir
  )
  if (-not [string]::IsNullOrWhiteSpace($ExplicitOutputDir)) {
    return Resolve-PathFromBase -Path $ExplicitOutputDir -BasePath $RepoRoot
  }
  $runtimeName = [System.IO.Path]::GetFileNameWithoutExtension($RuntimeConfigPath)
  return (Join-Path $RepoRoot "logs/NEAT_GPT/${runtimeName}_seed$SeedValue")
}
