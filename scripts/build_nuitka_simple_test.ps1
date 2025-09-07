[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Console
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Paths
$ScriptDir   = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptDir
$ReleaseRoot = Join-Path $ProjectRoot 'release'
$DistDir     = Join-Path $ReleaseRoot 'dist_simple_test'
$LogPath     = Join-Path $ReleaseRoot 'build_simple_test.log'

if (-not (Test-Path (Join-Path $ProjectRoot 'main.py'))) { 
    Write-Err "main.py not found in project root: $ProjectRoot"; exit 1 
}

# Transcript
$TranscriptStarted = $false
try { 
    if (!(Test-Path $ReleaseRoot)) { New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null }
    try { Stop-Transcript | Out-Null } catch {}
    Start-Transcript -Path $LogPath -Force | Out-Null
    $TranscriptStarted = $true
    Write-Info "Build log: $LogPath" 
} catch { 
    Write-Err "Could not start transcript: $($_.Exception.Message)" 
}

Write-Info "Project root: $ProjectRoot"
Write-Info "Dist dir    : $DistDir"

if ($Clean) { 
    Write-Info 'Cleaning previous build artifacts...'
    if (Test-Path $DistDir) { Remove-Item $DistDir -Recurse -Force }
    Write-Info 'Cleanup complete.' 
}

# Python probe
try {
    $PythonExe = & python -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($PythonExe)) { 
        throw "Python command failed" 
    }
    $PythonExe = $PythonExe.Trim()
    if (-not (Test-Path $PythonExe)) { 
        throw "Python executable not found at: $PythonExe" 
    }
    $PythonVersion = & "$PythonExe" -c "import sys; print('{}.{}.{}'.format(*sys.version_info[:3]))" 2>$null
    Write-Info "Python: $PythonVersion at $PythonExe"
} catch { 
    Write-Err "Python not found or invalid. Error: $($_.Exception.Message)"
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} } 
    exit 1 
}

# Verify Nuitka
try { 
    & "$PythonExe" -c "import nuitka" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "Nuitka import failed" }
    $NuitkaVersion = (& "$PythonExe" -m nuitka --version 2>$null)
    if ($NuitkaVersion){ 
        Write-Info ("Nuitka: " + $NuitkaVersion.Trim()) 
    } else { 
        Write-Info 'Nuitka validated' 
    } 
} catch { 
    Write-Err 'Nuitka not available. Install with: pip install nuitka'
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1 
}

# Prepare output
if (!(Test-Path $DistDir)) { New-Item -ItemType Directory -Path $DistDir -Force | Out-Null }

$IconPath = Join-Path $ProjectRoot 'resources/ShittyPIP.ico'

# Simple Nuitka build with minimal flags
Write-Info 'Starting simple Nuitka build...'
$nuArgs = @(
    '-m', 'nuitka',
    '--onefile',
    '--assume-yes-for-downloads',
    '--enable-plugin=pyside6',
    "--output-dir=$DistDir",
    '--output-filename=SPQ.exe'
)

if ($Console) { 
    $nuArgs += '--windows-console-mode=attach'
    Write-Info 'Console enabled' 
} else { 
    $nuArgs += '--windows-console-mode=disable'
    Write-Info 'Console disabled' 
}

if (Test-Path $IconPath) { 
    $nuArgs += "--windows-icon-from-ico=$IconPath"
    Write-Info "Using icon: $IconPath" 
}

$nuArgs += (Join-Path $ProjectRoot 'main.py')

Write-Info "Nuitka command: $PythonExe $($nuArgs -join ' ')"

$nuOut = Join-Path $ReleaseRoot 'nuitka_simple_stdout.log'
$nuErr = Join-Path $ReleaseRoot 'nuitka_simple_stderr.log'
foreach ($f in @($nuOut,$nuErr)) { if (Test-Path $f) { Remove-Item $f -Force } }

$sw = [Diagnostics.Stopwatch]::StartNew()
$proc = Start-Process -FilePath "$PythonExe" -ArgumentList $nuArgs -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru -RedirectStandardOutput $nuOut -RedirectStandardError $nuErr -Wait
$sw.Stop()
Write-Info ("Nuitka finished in {0:N1}s with exit code {1}" -f $sw.Elapsed.TotalSeconds, $proc.ExitCode)

if ($proc.ExitCode -ne 0) { 
    Write-Err "Nuitka build failed. See logs: $nuOut, $nuErr"
    if (Test-Path $nuErr) {
        Write-Host "--- Error Log ---" -ForegroundColor Red
        Get-Content $nuErr | Write-Host -ForegroundColor Red
    }
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit $proc.ExitCode 
}

$exePath = Join-Path $DistDir 'SPQ.exe'
if (-not (Test-Path $exePath)) { 
    Write-Err "Missing output: $exePath"
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1 
}

$sizeMB = [math]::Round((Get-Item $exePath).Length / 1MB, 1)
Write-Info "Simple build completed: $exePath (${sizeMB} MB)"

if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
exit 0
