<#
.SYNOPSIS
  Starts the AI Image Detector web UI (http://localhost:8501 by default).
.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\run.ps1
  powershell -ExecutionPolicy Bypass -File scripts\run.ps1 -Cpu -Port 8502
#>
param(
    [switch]$Cpu,
    [int]$Port = 8501
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$Py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $Py)) {
    Write-Host "ERROR: .venv not found." -ForegroundColor Red
    Write-Host "  fix: run scripts\setup.ps1 first" -ForegroundColor Yellow
    exit 1
}
if ($Cpu) { $env:DETECTOR_DEVICE = "cpu"; Write-Host "Forcing CPU inference (DETECTOR_DEVICE=cpu)." }
$busy = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction SilentlyContinue
if ($busy) {
    Write-Host "ERROR: port $Port is already in use." -ForegroundColor Red
    Write-Host "  fix: close the other app or run: scripts\run.ps1 -Port $($Port + 1)" -ForegroundColor Yellow
    exit 1
}
Write-Host "Starting the app on http://localhost:$Port  (Ctrl+C to stop)"
& $Py -m streamlit run app.py --server.port $Port --server.headless false
