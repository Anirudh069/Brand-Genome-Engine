# ─────────────────────────────────────────────────────────────
#  Brand Genome Engine — single-command startup (Windows)
#  Launches FastAPI backend (:8000) and Vite frontend (:5173)
#  Usage:  .\start.ps1          (start both)
#          .\start.ps1 -Stop    (kill both)
# ─────────────────────────────────────────────────────────────

param(
    [switch]$Stop
)

$RootDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

function Stop-Services {
    Write-Host "Stopping services..." -ForegroundColor Cyan

    # Kill uvicorn processes
    Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -eq 'python' -and ($_.CommandLine -match 'uvicorn' -or $_.CommandLine -match 'src.api.main')
    } | Stop-Process -Force -ErrorAction SilentlyContinue

    # Kill vite/node processes on port 5173
    Get-Process -ErrorAction SilentlyContinue | Where-Object {
        $_.ProcessName -eq 'node' -and $_.CommandLine -match 'vite'
    } | Stop-Process -Force -ErrorAction SilentlyContinue

    # Free ports as fallback
    foreach ($port in @(8000, 5173)) {
        $connections = Get-NetTCPConnection -LocalPort $port -ErrorAction SilentlyContinue
        foreach ($conn in $connections) {
            if ($conn.OwningProcess -and $conn.OwningProcess -ne 0) {
                Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
                Write-Host "  Freed port $port"
            }
        }
    }

    Write-Host "All services stopped." -ForegroundColor Green
}

if ($Stop) {
    Stop-Services
    exit 0
}

# ── Ensure common tool directories are in PATH ───────────────
$extraPaths = @(
    "$env:ProgramFiles\nodejs",
    "$env:APPDATA\npm"
)
foreach ($p in $extraPaths) {
    if ((Test-Path $p) -and ($env:PATH -notlike "*$p*")) {
        $env:PATH = "$p;$env:PATH"
    }
}

# ── Resolve Python ────────────────────────────────────────────
# Prefer the 'py' launcher (always reliable on Windows), then fall back
# to a direct python in Programs, then generic 'python'.
$Python = $null
if (Get-Command py -ErrorAction SilentlyContinue) {
    $Python = 'py'
} elseif (Test-Path "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe") {
    $Python = (Get-Item "$env:LOCALAPPDATA\Programs\Python\Python3*\python.exe" | Select-Object -First 1).FullName
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    $Python = 'python'
}
if (-not $Python) {
    Write-Host "Error: python not found. Install Python 3.9+." -ForegroundColor Red
    exit 1
}

# ── Pre-flight checks ────────────────────────────────────────
if (-not (Get-Command node -ErrorAction SilentlyContinue)) {
    Write-Host "Error: node not found. Install Node.js 18+." -ForegroundColor Red
    exit 1
}

# ── Install deps if needed ───────────────────────────────────
& $Python -c "import uvicorn" 2>&1 | Out-Null
if ($LASTEXITCODE -ne 0) {
    Write-Host "Installing Python dependencies..." -ForegroundColor Cyan
    & $Python -m pip install -r "$RootDir\requirements.txt" --quiet
}

if (-not (Test-Path "$RootDir\frontend\node_modules")) {
    Write-Host "Installing frontend dependencies..." -ForegroundColor Cyan
    Push-Location "$RootDir\frontend"
    npm install --silent
    Pop-Location
}

# ── Kill any previous run ────────────────────────────────────
Stop-Services 2>$null

# ── Start backend ─────────────────────────────────────────────
Write-Host ""
Write-Host "Starting FastAPI backend on http://localhost:8000 ..." -ForegroundColor Green
$pythonExe = (Get-Command $Python).Source
$backendJob = Start-Process -PassThru -NoNewWindow -FilePath $pythonExe -ArgumentList "-m", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--reload", "--log-level", "info" -WorkingDirectory $RootDir

# ── Start frontend ────────────────────────────────────────────
Write-Host "Starting Vite frontend on http://localhost:5173 ..." -ForegroundColor Green
$viteCmd = "$RootDir\frontend\node_modules\.bin\vite.cmd"
if (-not (Test-Path $viteCmd)) {
    Write-Host "Error: vite not found. Run 'npm install' in frontend/." -ForegroundColor Red
    Stop-Process -Id $backendJob.Id -Force -ErrorAction SilentlyContinue
    exit 1
}
$frontendJob = Start-Process -PassThru -NoNewWindow -FilePath cmd.exe -ArgumentList "/c", $viteCmd, "--host", "127.0.0.1", "--port", "5173" -WorkingDirectory "$RootDir\frontend"

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Brand Genome Engine is running" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "  Frontend  ->  http://localhost:5173" -ForegroundColor Cyan
Write-Host "  Backend   ->  http://localhost:8000/docs" -ForegroundColor Cyan
Write-Host "" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C to stop both services" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# ── Trap Ctrl+C to clean up ──────────────────────────────────
try {
    while ($true) {
        if ($backendJob.HasExited -and $frontendJob.HasExited) {
            Write-Host "Both services exited unexpectedly." -ForegroundColor Red
            break
        }
        Start-Sleep -Seconds 2
    }
}
finally {
    Write-Host ""
    Write-Host "Shutting down..." -ForegroundColor Cyan

    if (-not $backendJob.HasExited) {
        Stop-Process -Id $backendJob.Id -Force -ErrorAction SilentlyContinue
    }
    if (-not $frontendJob.HasExited) {
        Stop-Process -Id $frontendJob.Id -Force -ErrorAction SilentlyContinue
    }

    Stop-Services
    Write-Host "Done." -ForegroundColor Green
}
