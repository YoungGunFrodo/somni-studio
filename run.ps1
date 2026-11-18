<#
.SYNOPSIS
  Start Somni Studio.
.DESCRIPTION
  Picks a Python environment, in this order:
    1. -Python <path to python.exe>
    2. $env:SOMNI_STUDIO_PYTHON
    3. the Somni checkout's environment (.venv-gpu, then .venv), found through somni.repo in studio.json
       (default ../somni), so the in-process model uses the same GPU build of PyTorch as Somni
    4. .venv in this folder
  Installs the web-server packages and three.js on first run, then opens the UI.
.EXAMPLE
  .\run.ps1
  .\run.ps1 -Port 9000 -NoBrowser
#>
param(
  [string]$Python = "",
  [string]$Config = "",
  [int]$Port = 0,
  [switch]$NoBrowser
)
$ErrorActionPreference = "Stop"
$root = $PSScriptRoot

function Resolve-SomniRepo {
  $settingsFile = if ($Config) { $Config } else { Join-Path $root "studio.json" }
  $repo = "../somni"
  if (Test-Path $settingsFile) {
    $json = Get-Content $settingsFile -Raw | ConvertFrom-Json
    if ($json.somni -and $json.somni.repo) { $repo = $json.somni.repo }
  }
  if ([System.IO.Path]::IsPathRooted($repo)) { return $repo }
  return Join-Path $root $repo
}

if (-not $Python -and $env:SOMNI_STUDIO_PYTHON) { $Python = $env:SOMNI_STUDIO_PYTHON }
if (-not $Python) {
  $repo = Resolve-SomniRepo
  $candidates = @(
    (Join-Path $repo ".venv-gpu\Scripts\python.exe"),
    (Join-Path $repo ".venv\Scripts\python.exe"),
    (Join-Path $root ".venv\Scripts\python.exe")
  )
  $Python = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
  if (-not $Python) {
    throw "No Python environment found. Set up Somni first (see README), or pass -Python <path to python.exe>."
  }
}
Write-Host "Python: $Python"

& $Python -c "import fastapi, uvicorn, websockets" 2>$null
if ($LASTEXITCODE -ne 0) {
  Write-Host "Installing the web-server packages"
  & $Python -m pip install --quiet "fastapi>=0.115" "uvicorn>=0.32" "websockets>=13"
  if ($LASTEXITCODE -ne 0) { throw "pip install failed" }
}

& $Python (Join-Path $root "scripts\vendor.py")
if ($LASTEXITCODE -ne 0) { throw "could not vendor three.js" }

$studioArgs = @("-m", "somni_studio")
if ($Config) { $studioArgs += @("--config", $Config) }
if ($Port) { $studioArgs += @("--port", "$Port") }
if (-not $NoBrowser) { $studioArgs += "--open" }

Push-Location $root
try { & $Python @studioArgs } finally { Pop-Location }
