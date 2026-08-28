# Build a self-contained Windows package of the HandHead GUI.
# Usage (from anywhere):  powershell -ExecutionPolicy Bypass -File packaging\build_exe.ps1
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot   # project root
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $venvPython)) { $venvPython = "python" }

& $venvPython -m pip install --disable-pip-version-check --quiet pyinstaller

Push-Location $root
try {
    & $venvPython -m PyInstaller --clean --noconfirm packaging\handhead_gui.spec
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }

    $dist = Join-Path $root "dist\AIComplianceCheck"

    # Runtime assets the app resolves via relative paths at run time.
    Copy-Item -Recurse -Force (Join-Path $root "config") "$dist\config"
    Copy-Item -Recurse -Force (Join-Path $root "models") "$dist\models"
    Get-ChildItem $root -Filter "yolo*.pt" | ForEach-Object {
        Copy-Item -Force $_.FullName $dist
    }
}
finally {
    Pop-Location
}

Write-Host ""
Write-Host "Build OK -> dist\HandHeadGUI\HandHeadGUI.exe"
Write-Host "Ship the ENTIRE dist\HandHeadGUI folder (zip it). No Python needed on the customer machine."
