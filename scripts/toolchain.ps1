function Get-RepoRoot {
  return (Split-Path -Parent $PSScriptRoot)
}

function Get-LatestPortableToolDir {
  param(
    [Parameter(Mandatory = $true)][string]$Prefix
  )

  $toolsRoot = Join-Path (Get-RepoRoot) ".tools"
  if (-not (Test-Path $toolsRoot)) {
    return $null
  }

  return Get-ChildItem -LiteralPath $toolsRoot -Directory |
    Where-Object { $_.Name -like "$Prefix*" } |
    Sort-Object Name -Descending |
    Select-Object -First 1
}

function Resolve-PythonCommand {
  $repoRoot = Get-RepoRoot
  $venvPython = Join-Path $repoRoot ".venv\Scripts\python.exe"
  if (Test-Path $venvPython) {
    return [pscustomobject]@{
      Path = (Resolve-Path $venvPython).Path
      Directory = Split-Path -Parent (Resolve-Path $venvPython).Path
      ScriptsDirectory = Split-Path -Parent (Resolve-Path $venvPython).Path
      Source = ".venv"
    }
  }

  $portableDir = Get-LatestPortableToolDir -Prefix "python-"
  if ($null -ne $portableDir) {
    $portablePython = Join-Path $portableDir.FullName "python.exe"
    if (Test-Path $portablePython) {
      return [pscustomobject]@{
        Path = (Resolve-Path $portablePython).Path
        Directory = $portableDir.FullName
        ScriptsDirectory = Join-Path $portableDir.FullName "Scripts"
        Source = ".tools"
      }
    }
  }

  $systemPython = Get-Command python -ErrorAction SilentlyContinue
  if ($null -ne $systemPython -and (Test-Path $systemPython.Source) -and ($systemPython.Source -notmatch "WindowsApps")) {
    return [pscustomobject]@{
      Path = $systemPython.Source
      Directory = Split-Path -Parent $systemPython.Source
      ScriptsDirectory = $null
      Source = "system"
    }
  }

  throw "python not found: expected .venv\Scripts\python.exe or .tools\python-*\python.exe"
}

function Resolve-NodeCommand {
  $systemNode = Get-Command node -ErrorAction SilentlyContinue
  if ($null -ne $systemNode -and (Test-Path $systemNode.Source)) {
    return [pscustomobject]@{
      Path = $systemNode.Source
      Directory = Split-Path -Parent $systemNode.Source
      Source = "system"
    }
  }

  $portableDir = Get-LatestPortableToolDir -Prefix "node-"
  if ($null -ne $portableDir) {
    $portableNode = Join-Path $portableDir.FullName "node.exe"
    if (Test-Path $portableNode) {
      return [pscustomobject]@{
        Path = (Resolve-Path $portableNode).Path
        Directory = $portableDir.FullName
        Source = ".tools"
      }
    }
  }

  throw "node not found: expected system node or .tools\node-*\node.exe"
}

function Enable-RepoToolchainPath {
  param(
    [Parameter(Mandatory = $false)]$PythonInfo,
    [Parameter(Mandatory = $false)]$NodeInfo
  )

  $prepend = @()
  if ($null -ne $PythonInfo) {
    if ($PythonInfo.PSObject.Properties.Name -contains "ScriptsDirectory" -and -not [string]::IsNullOrWhiteSpace([string]$PythonInfo.ScriptsDirectory)) {
      if (Test-Path $PythonInfo.ScriptsDirectory) {
        $prepend += [string]$PythonInfo.ScriptsDirectory
      }
    }
    if ($PythonInfo.PSObject.Properties.Name -contains "Directory" -and -not [string]::IsNullOrWhiteSpace([string]$PythonInfo.Directory)) {
      if (Test-Path $PythonInfo.Directory) {
        $prepend += [string]$PythonInfo.Directory
      }
    }
  }
  if ($null -ne $NodeInfo) {
    if ($NodeInfo.PSObject.Properties.Name -contains "Directory" -and -not [string]::IsNullOrWhiteSpace([string]$NodeInfo.Directory)) {
      if (Test-Path $NodeInfo.Directory) {
        $prepend += [string]$NodeInfo.Directory
      }
    }
  }

  $uniquePrepend = @()
  foreach ($entry in $prepend) {
    if ([string]::IsNullOrWhiteSpace($entry)) {
      continue
    }
    if ($uniquePrepend -contains $entry) {
      continue
    }
    $uniquePrepend += $entry
  }

  if ($uniquePrepend.Count -gt 0) {
    $env:PATH = ($uniquePrepend -join ";") + ";" + $env:PATH
  }
}
