[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$RepoRoot = (Resolve-Path (Join-Path $ScriptDir '..')).Path
$RuntimeDir = Join-Path $RepoRoot '.runtime'
$BackendStateFile = Join-Path $RuntimeDir 'brand-genome-backend.json'
$FrontendStateFile = Join-Path $RuntimeDir 'brand-genome-frontend.json'

function Write-Info {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Cyan
}

function Write-Warn {
    param([string]$Message)
    Write-Host $Message -ForegroundColor Yellow
}

function Fail {
    param([string]$Message)
    Write-Host "Error: $Message" -ForegroundColor Red
    exit 1
}

function Get-ProcessRecord {
    param([int]$ProcessId)
    try {
        return Get-CimInstance Win32_Process -Filter "ProcessId = $ProcessId" -ErrorAction Stop
    } catch {
        return $null
    }
}

function Stop-RecordedProcess {
    param(
        [int]$ProcessId,
        [string]$ExpectedName,
        [string]$ExpectedCommandPattern,
        [string]$Label
    )

    if ($ProcessId -le 0) {
        return
    }

    $record = Get-ProcessRecord -ProcessId $ProcessId
    if (-not $record) {
        Write-Info "$Label already exited."
        return
    }

    if ($ExpectedName -and ($record.Name -notlike $ExpectedName)) {
        Fail "Refusing to stop $Label PID $ProcessId because it no longer looks like the recorded launcher process ($($record.Name))."
    }

    if ($ExpectedCommandPattern -and ($record.CommandLine -notmatch $ExpectedCommandPattern)) {
        Fail "Refusing to stop $Label PID $ProcessId because its command line no longer matches the launcher record."
    }

    Stop-Process -Id $ProcessId -Force -ErrorAction Stop
    Write-Info "Stopped $Label PID $ProcessId."
}

Write-Info 'Stopping Brand Genome Engine processes...'

if (-not (Test-Path $RuntimeDir)) {
    Write-Warn 'No runtime state directory found. Nothing to stop.'
    exit 0
}

$didStopAny = $false

if (Test-Path $BackendStateFile) {
    $backendState = Get-Content $BackendStateFile -Raw | ConvertFrom-Json
    Stop-RecordedProcess -ProcessId ([int]$backendState.child_pid) -ExpectedName 'python*' -ExpectedCommandPattern 'uvicorn\s+src\.api\.main:app' -Label 'backend child'
    Stop-RecordedProcess -ProcessId ([int]$backendState.shell_pid) -ExpectedName 'powershell*' -ExpectedCommandPattern 'Brand Genome Backend' -Label 'backend terminal'
    Remove-Item $BackendStateFile -Force -ErrorAction SilentlyContinue
    $didStopAny = $true
}

if (Test-Path $FrontendStateFile) {
    $frontendState = Get-Content $FrontendStateFile -Raw | ConvertFrom-Json
    Stop-RecordedProcess -ProcessId ([int]$frontendState.child_pid) -ExpectedName 'node*' -ExpectedCommandPattern 'npm-cli\.js|npm run dev' -Label 'frontend child'
    Stop-RecordedProcess -ProcessId ([int]$frontendState.shell_pid) -ExpectedName 'powershell*' -ExpectedCommandPattern 'Brand Genome Frontend' -Label 'frontend terminal'
    Remove-Item $FrontendStateFile -Force -ErrorAction SilentlyContinue
    $didStopAny = $true
}

if ($didStopAny) {
    if ((Get-ChildItem $RuntimeDir -Force -ErrorAction SilentlyContinue | Measure-Object).Count -eq 0) {
        Remove-Item $RuntimeDir -Force -ErrorAction SilentlyContinue
    }
    Write-Info 'Brand Genome Engine stop complete.'
    exit 0
}

Write-Warn 'No launcher-owned processes were found to stop.'
exit 0
