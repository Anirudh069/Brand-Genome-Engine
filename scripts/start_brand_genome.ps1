[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$RuntimeDir = Join-Path $RepoRoot '.runtime'
$RuntimeFile = Join-Path $RuntimeDir 'brand-genome-processes.json'
$BackendPort = 8000
$FrontendPort = 5173
$BackendUrl = 'http://127.0.0.1:8000'
$FrontendUrl = 'http://127.0.0.1:5173'
$BackendStateFile = Join-Path $RuntimeDir 'brand-genome-backend.json'
$FrontendStateFile = Join-Path $RuntimeDir 'brand-genome-frontend.json'

function Write-Section {
    param([string]$Text)
    Write-Host "`n$Text" -ForegroundColor Cyan
}

function Fail {
    param([string]$Message)
    Write-Host "Error: $Message" -ForegroundColor Red
    exit 1
}

function Test-PortServingUrl {
    param(
        [int]$Port,
        [string]$ExpectedPrefix
    )

    try {
        $response = Invoke-WebRequest -Uri "http://127.0.0.1:$Port" -UseBasicParsing -TimeoutSec 3
        return $response.StatusCode -ge 200 -and $response.StatusCode -lt 500 -and $response.BaseResponse.ResponseUri.AbsoluteUri.StartsWith($ExpectedPrefix)
    } catch {
        return $false
    }
}

function Wait-ForHttp {
    param(
        [string]$Uri,
        [int]$TimeoutSeconds = 45
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $response = Invoke-WebRequest -Uri $Uri -UseBasicParsing -TimeoutSec 3
            if ($response.StatusCode -ge 200 -and $response.StatusCode -lt 500) {
                return $true
            }
        } catch {
        }
        Start-Sleep -Seconds 2
    }

    return $false
}

function Get-ListenerPid {
    param([int]$Port)
    try {
        $connection = Get-NetTCPConnection -LocalPort $Port -State Listen -ErrorAction Stop | Select-Object -First 1
        return $connection.OwningProcess
    } catch {
        return $null
    }
}

function Test-PortOwnedByKnownLauncher {
    param(
        [int]$Port,
        [int[]]$KnownPids,
        [string]$ExpectedPrefix
    )

    $listenerPid = Get-ListenerPid -Port $Port
    if (-not $listenerPid) {
        return [pscustomobject]@{ Status = 'free'; Pid = $null }
    }

    $commandLine = $null
    try {
        $commandLine = (Get-CimInstance Win32_Process -Filter "ProcessId = $listenerPid" -ErrorAction Stop).CommandLine
    } catch {
    }

    if ($KnownPids -contains $listenerPid) {
        return [pscustomobject]@{ Status = 'known'; Pid = $listenerPid }
    }

    if ($commandLine -and $commandLine -match $ExpectedPrefix) {
        return [pscustomobject]@{ Status = 'known'; Pid = $listenerPid }
    }

    return [pscustomobject]@{ Status = 'foreign'; Pid = $listenerPid }
}

function Test-RequiredCommand {
    param([string]$Name)
    try {
        $command = Get-Command $Name -ErrorAction Stop
        if ($command -and $command.Path) {
            return $command.Path
        }
        if ($command -and $command.Source) {
            return $command.Source
        }
        return $Name
    } catch {
        Fail "$Name is not available on PATH. Install it and try again."
    }
}

function Test-ProjectFile {
    param([string]$RelativePath)
    $fullPath = Join-Path $RepoRoot $RelativePath
    if (-not (Test-Path $fullPath)) {
        Fail "Missing required file: $RelativePath"
    }
}

function Wait-ForChildProcess {
    param(
        [int]$ParentPid,
        [string]$NamePattern,
        [string]$CommandPattern,
        [int]$TimeoutSeconds = 20
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        try {
            $children = Get-CimInstance Win32_Process -Filter "ParentProcessId = $ParentPid" -ErrorAction Stop
            foreach ($child in $children) {
                if ($NamePattern -and ($child.Name -notlike $NamePattern)) {
                    continue
                }
                if ($CommandPattern -and ($child.CommandLine -notmatch $CommandPattern)) {
                    continue
                }
                return [int]$child.ProcessId
            }
        } catch {
        }
        Start-Sleep -Seconds 1
    }

    return 0
}

function Write-LauncherState {
    param(
        [string]$StateFile,
        [string]$Role,
        [int]$ShellPid,
        [int]$ChildPid,
        [string]$Command
    )

    if (-not (Test-Path $RuntimeDir)) {
        New-Item -ItemType Directory -Path $RuntimeDir | Out-Null
    }

    $state = [pscustomobject]@{
        role = $Role
        shell_pid = $ShellPid
        child_pid = $ChildPid
        command = $Command
        started_at = (Get-Date).ToString('o')
    }

    $state | ConvertTo-Json -Depth 4 | Set-Content -Path $StateFile -Encoding UTF8
}

function Test-PythonDependencies {
    param([string]$PythonExe)
    Write-Section 'Checking Python runtime packages'
    $check = @'
modules = [
    'fastapi',
    'uvicorn',
    'numpy',
    'pandas',
    'sklearn',
    'sentence_transformers',
    'faiss',
    'openai',
]
missing = []
for module in modules:
    try:
        __import__(module)
    except Exception:
        missing.append(module)
if missing:
    raise SystemExit('MISSING:' + ','.join(missing))
print('OK')
'@
    $output = & $PythonExe -c $check 2>&1
    if ($LASTEXITCODE -ne 0) {
        $message = ($output -join [Environment]::NewLine)
        if ($message -match 'MISSING:') {
            Fail "Python dependencies are missing. Run:`n`npip install -r requirements.txt"
        }
        Fail "Python dependency check failed:`n$message"
    }
}

function Ensure-AnalyticsArtifact {
    param([string]$PythonExe)
    $cachePath = Join-Path $RepoRoot 'data/processed/analytics_cache.json'
    if (Test-Path $cachePath) {
        return
    }
    Write-Section 'Building analytics cache'
    & $PythonExe -m scripts.build_analytics_cache
    if ($LASTEXITCODE -ne 0) {
        Fail 'Analytics cache build failed. Inspect the output above.'
    }
}

function Ensure-RagArtifact {
    param([string]$PythonExe)
    $ragDir = Join-Path $RepoRoot 'data/processed/rag'
    $manifest = Join-Path $ragDir 'manifest.json'
    if (Test-Path $manifest) {
        return
    }
    Write-Section 'Building RAG index'
    & $PythonExe -m scripts.build_rag_index
    if ($LASTEXITCODE -ne 0) {
        Fail 'RAG index build failed. Inspect the output above.'
    }
}

Write-Section 'Brand Genome Engine startup'
Write-Host "Repository root: $RepoRoot"

$PythonCmd = Test-RequiredCommand 'python'
$NodeCmd = Test-RequiredCommand 'node'
$NpmCmd = Test-RequiredCommand 'npm'

Test-ProjectFile 'requirements.txt'
Test-ProjectFile 'frontend/package.json'
Test-ProjectFile 'frontend/package-lock.json'
Test-ProjectFile 'data/brand_data.db'
Test-ProjectFile 'src/api/main.py'

Test-PythonDependencies -PythonExe $PythonCmd

Write-Section 'Checking derived artifacts'
& $PythonCmd -m scripts.verify_phase4
if ($LASTEXITCODE -ne 0) {
    Fail 'verify_phase4 reported a hard failure. Inspect the output above.'
}

Ensure-AnalyticsArtifact -PythonExe $PythonCmd
Ensure-RagArtifact -PythonExe $PythonCmd

Write-Section 'Checking ports'
$backendCheck = Test-PortOwnedByKnownLauncher -Port $BackendPort -KnownPids @() -ExpectedPrefix 'uvicorn src\.api\.main:app'
if ($backendCheck.Status -eq 'foreign') {
    Fail "Port $BackendPort is already in use by another process (PID $($backendCheck.Pid)). Stop that process or free the port and try again."
}
$frontendCheck = Test-PortOwnedByKnownLauncher -Port $FrontendPort -KnownPids @() -ExpectedPrefix 'vite'
if ($frontendCheck.Status -eq 'foreign') {
    Fail "Port $FrontendPort is already in use by another process (PID $($frontendCheck.Pid)). Stop that process or free the port and try again."
}

Write-Section 'Checking frontend dependencies'
$nodeModules = Join-Path $RepoRoot 'frontend/node_modules'
if (-not (Test-Path $nodeModules)) {
    Write-Host 'frontend/node_modules missing; running npm ci ...'
    Push-Location (Join-Path $RepoRoot 'frontend')
    try {
        & $NpmCmd ci
        if ($LASTEXITCODE -ne 0) {
            Fail 'npm ci failed. Inspect the output above.'
        }
    } finally {
        Pop-Location
    }
}

$backendPortCheck = Test-PortOwnedByKnownLauncher -Port $BackendPort -KnownPids @() -ExpectedPrefix 'uvicorn src\.api\.main:app'
$frontendPortCheck = Test-PortOwnedByKnownLauncher -Port $FrontendPort -KnownPids @() -ExpectedPrefix 'vite'

$backendRunning = $false
if ($backendPortCheck.Status -eq 'known') {
    Write-Host "Backend already appears to be running on $BackendUrl; reusing it."
    $backendRunning = $true
}

$frontendRunning = $false
if ($frontendPortCheck.Status -eq 'known') {
    Write-Host "Frontend already appears to be running on $FrontendUrl; reusing it."
    $frontendRunning = $true
}

$startedBackendPid = $null
$startedFrontendPid = $null

if (-not $backendRunning) {
    Write-Section 'Starting backend terminal'
    $backendArgs = @(
        '-NoProfile',
        '-NoExit',
        '-Command',
        "`$Host.UI.RawUI.WindowTitle = 'Brand Genome Backend'; Set-Location '$RepoRoot'; python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000"
    )
    $backendProcess = Start-Process -PassThru -WindowStyle Normal -FilePath 'powershell.exe' -ArgumentList $backendArgs -WorkingDirectory $RepoRoot
    $startedBackendPid = $backendProcess.Id
    $backendChildPid = Wait-ForChildProcess -ParentPid $startedBackendPid -NamePattern 'python*' -CommandPattern 'uvicorn\s+src\.api\.main:app'
    if (-not $backendChildPid) {
        Fail 'Could not confirm the backend Python process started.'
    }
    Write-LauncherState -StateFile $BackendStateFile -Role 'backend' -ShellPid $startedBackendPid -ChildPid $backendChildPid -Command "python -m uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8000"
}

if (-not $frontendRunning) {
    Write-Section 'Starting frontend terminal'
    $frontendArgs = @(
        '-NoProfile',
        '-NoExit',
        '-Command',
        "`$Host.UI.RawUI.WindowTitle = 'Brand Genome Frontend'; Set-Location '$RepoRoot\frontend'; npm run dev -- --host 127.0.0.1 --port 5173"
    )
    $frontendProcess = Start-Process -PassThru -WindowStyle Normal -FilePath 'powershell.exe' -ArgumentList $frontendArgs -WorkingDirectory (Join-Path $RepoRoot 'frontend')
    $startedFrontendPid = $frontendProcess.Id
    $frontendChildPid = Wait-ForChildProcess -ParentPid $startedFrontendPid -NamePattern 'node*' -CommandPattern 'npm-cli\.js|npm run dev'
    if (-not $frontendChildPid) {
        Fail 'Could not confirm the frontend Node process started.'
    }
    Write-LauncherState -StateFile $FrontendStateFile -Role 'frontend' -ShellPid $startedFrontendPid -ChildPid $frontendChildPid -Command 'npm run dev -- --host 127.0.0.1 --port 5173'
}

Write-Section 'Waiting for backend readiness'
if (-not (Wait-ForHttp -Uri "$BackendUrl/api/health" -TimeoutSeconds 60)) {
    Fail 'Backend failed to become ready.'
}

Write-Section 'Waiting for frontend readiness'
if (-not (Wait-ForHttp -Uri $FrontendUrl -TimeoutSeconds 60)) {
    Fail 'Frontend failed to become ready.'
}

Start-Process $FrontendUrl

Write-Host "`nBrand Genome Engine is running." -ForegroundColor Green
Write-Host "Frontend: $FrontendUrl" -ForegroundColor Green
Write-Host "Backend API: $BackendUrl" -ForegroundColor Green
Write-Host "API docs: $BackendUrl/docs" -ForegroundColor Green
exit 0
