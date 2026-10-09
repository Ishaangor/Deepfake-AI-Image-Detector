<#
.SYNOPSIS
  Regenerates the SBOM (sbom/sbom.cdx.json, CycloneDX JSON) and the vulnerability audit (sbom/pip-audit.json)
  from the project's virtual environment (.venv, created by scripts\setup.ps1).
.DESCRIPTION
  The SBOM and audit tools are installed into a separate throw-away venv (.sbom-tools, git-ignored) so that they do not
  appear in the SBOM themselves. Needs internet access (PyPI, and the OSV/PyPI advisory database for pip-audit).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\generate_sbom.ps1
#>
param(
    [string]$TargetVenv = ".venv"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

if (-not [System.IO.Path]::IsPathRooted($TargetVenv)) { $TargetVenv = Join-Path $Root $TargetVenv }
$Target = Join-Path $TargetVenv "Scripts\python.exe"
if (-not (Test-Path $Target)) {
    Write-Host "ERROR: $TargetVenv not found. Run scripts\setup.ps1 first." -ForegroundColor Red
    exit 1
}

$ToolsDir = Join-Path $Root ".sbom-tools"
if (-not (Test-Path "$ToolsDir\Scripts\python.exe")) {
    & $Target -m venv $ToolsDir
}
$Tools = Join-Path $ToolsDir "Scripts\python.exe"
& $Tools -m pip install --quiet --upgrade pip
& $Tools -m pip install --quiet "cyclonedx-bom==7.*" "pip-audit==2.*"
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: could not install cyclonedx-bom / pip-audit." -ForegroundColor Red; exit 1 }

New-Item -ItemType Directory -Force (Join-Path $Root "sbom") | Out-Null

# 1. SBOM of every package installed in the target environment
& $Tools -m cyclonedx_py environment --pyproject pyproject.toml --mc-type application --output-reproducible `
    --output-format JSON --output-file sbom\sbom.cdx.json $Target
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: SBOM generation failed." -ForegroundColor Red; exit 1 }

# 2. Vulnerability audit of exactly those pinned versions (local tags such as +cu124 / +cpu are stripped for lookup)
$frozen = Join-Path $ToolsDir "frozen.txt"
& $Target -m pip freeze --exclude-editable | ForEach-Object { $_ -replace '\+[A-Za-z0-9.]+$', '' } |
    Set-Content -Encoding ascii $frozen
& $Tools -m pip_audit --requirement $frozen --no-deps --disable-pip --format json --output sbom\pip-audit.json
$auditExit = $LASTEXITCODE
& $Tools -m pip_audit --requirement $frozen --no-deps --disable-pip --format columns
Write-Host ""
Write-Host "Wrote sbom\sbom.cdx.json and sbom\pip-audit.json (pip-audit exit code $auditExit; 1 = vulnerabilities found)."
