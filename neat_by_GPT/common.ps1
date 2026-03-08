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

function Resolve-NeatGptRuntimeConfigPath {
  param(
    [Parameter(Mandatory = $true)][string]$ScriptRoot,
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $false)][string]$RuntimeConfig,
    [Parameter(Mandatory = $false)][string]$Phase = ""
  )

  if (-not [string]::IsNullOrWhiteSpace($RuntimeConfig)) {
    return Resolve-PathFromBase -Path $RuntimeConfig -BasePath $RepoRoot
  }

  $phaseKey = [string]$Phase
  if ([string]::IsNullOrWhiteSpace($phaseKey)) {
    return (Join-Path $ScriptRoot "configs\runtime_focus_cl_v1.json")
  }

  switch ($phaseKey.Trim().ToLowerInvariant()) {
    "1" { return (Join-Path $ScriptRoot "configs\runtime_phase1.json") }
    "2" { return (Join-Path $ScriptRoot "configs\runtime_phase2.json") }
    "3" { return (Join-Path $ScriptRoot "configs\runtime_phase3.json") }
    "focus" { return (Join-Path $ScriptRoot "configs\runtime_focus_cl_v1.json") }
    default { throw "invalid phase: $Phase (allowed: 1, 2, 3, focus)" }
  }
}

function Resolve-NeatGptPhaseSeedState {
  param(
    [Parameter(Mandatory = $true)][string]$ScriptRoot,
    [Parameter(Mandatory = $true)][string]$RepoRoot,
    [Parameter(Mandatory = $true)][string]$Phase,
    [Parameter(Mandatory = $true)][int]$SeedValue
  )

  $phaseKey = [string]$Phase
  if ([string]::IsNullOrWhiteSpace($phaseKey)) {
    return $null
  }

  $normalized = $phaseKey.Trim().ToLowerInvariant()
  if ($normalized -eq "1" -or $normalized -eq "focus") {
    return $null
  }
  if ($normalized -ne "2" -and $normalized -ne "3") {
    throw "phase auto seed chaining only supports phase 2 or 3: $Phase"
  }

  $previousPhase = [int]$normalized - 1
  $previousRuntimeConfig = Resolve-NeatGptRuntimeConfigPath -ScriptRoot $ScriptRoot -RepoRoot $RepoRoot -Phase "$previousPhase"
  $previousOutputDir = Resolve-NeatGptOutputDirPath -RepoRoot $RepoRoot -RuntimeConfigPath $previousRuntimeConfig -SeedValue $SeedValue
  $legacyOutputDir = Join-Path $RepoRoot "logs/NEAT_GPT/neat_phase${previousPhase}_seed$SeedValue"

  return [pscustomobject]@{
    previous_phase = $previousPhase
    previous_label = "phase$previousPhase"
    runtime_config = $previousRuntimeConfig
    output_dir = $previousOutputDir
    summary_path = (Join-Path $previousOutputDir "run_summary.json")
    fallback_winner_path = (Join-Path $previousOutputDir "models\winner_genome.pkl")
    legacy_output_dir = $legacyOutputDir
    legacy_summary_path = (Join-Path $legacyOutputDir "run_summary.json")
    legacy_fallback_winner_path = (Join-Path $legacyOutputDir "models\winner_genome.pkl")
  }
}

function ConvertTo-NeatGptHashtable {
  param([Parameter(Mandatory = $false)]$InputObject)

  if ($null -eq $InputObject) {
    return $null
  }
  if ($InputObject -is [System.Collections.IDictionary]) {
    $out = @{}
    foreach ($key in $InputObject.Keys) {
      $out[[string]$key] = ConvertTo-NeatGptHashtable -InputObject $InputObject[$key]
    }
    return $out
  }
  if ($InputObject -is [string]) {
    return $InputObject
  }
  if ($InputObject -is [System.Collections.IEnumerable]) {
    $items = @()
    foreach ($item in $InputObject) {
      $items += ,(ConvertTo-NeatGptHashtable -InputObject $item)
    }
    return $items
  }
  if ($InputObject -is [pscustomobject]) {
    $out = @{}
    foreach ($prop in $InputObject.PSObject.Properties) {
      $out[$prop.Name] = ConvertTo-NeatGptHashtable -InputObject $prop.Value
    }
    return $out
  }
  return $InputObject
}

function Read-NeatGptRuntimeJson {
  param([Parameter(Mandatory = $true)][string]$Path)

  $merged = @{}
  $seen = New-Object 'System.Collections.Generic.HashSet[string]'

  function Merge-NeatGptRuntimeFile {
    param([Parameter(Mandatory = $true)][string]$ConfigPath)

    $fullPath = [System.IO.Path]::GetFullPath($ConfigPath)
    if ($seen.Contains($fullPath)) {
      return
    }
    $null = $seen.Add($fullPath)
    if (-not (Test-Path $fullPath)) {
      throw "runtime config not found: $fullPath"
    }

    $rawObject = Get-Content $fullPath -Raw -Encoding UTF8 | ConvertFrom-Json
    $rawMap = ConvertTo-NeatGptHashtable -InputObject $rawObject
    if ($null -eq $rawMap) {
      throw "runtime config root must be an object: $fullPath"
    }

    $extendsList = @()
    if ($rawMap.ContainsKey("extends")) {
      $extendsRaw = $rawMap["extends"]
      if ($extendsRaw -is [string]) {
        if (-not [string]::IsNullOrWhiteSpace($extendsRaw)) {
          $extendsList += $extendsRaw.Trim()
        }
      }
      elseif ($extendsRaw -is [System.Collections.IEnumerable]) {
        foreach ($item in $extendsRaw) {
          $text = [string]$item
          if (-not [string]::IsNullOrWhiteSpace($text)) {
            $extendsList += $text.Trim()
          }
        }
      }
      $null = $rawMap.Remove("extends")
    }

    $baseDir = [System.IO.Path]::GetDirectoryName($fullPath)
    foreach ($child in $extendsList) {
      $childPath = Resolve-PathFromBase -Path $child -BasePath $baseDir
      Merge-NeatGptRuntimeFile -ConfigPath $childPath
    }

    foreach ($key in $rawMap.Keys) {
      $merged[$key] = $rawMap[$key]
    }
  }

  Merge-NeatGptRuntimeFile -ConfigPath $Path
  return [pscustomobject]$merged
}
