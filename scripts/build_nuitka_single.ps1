[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Console,
    [switch]$BuildDebug
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }

function Remove-TreeRobust {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    try { Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop; return } catch {}
    try {
        Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
            try { $_.Attributes = 'Normal'; if (-not $_.PSIsContainer) { [GC]::Collect(); [GC]::WaitForPendingFinalizers() } } catch {}
        }
        Start-Sleep -Milliseconds 300;
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop; return
    } catch {}
    try {
        $parent = Split-Path -Parent $Path; $leaf = Split-Path -Leaf $Path;
        $alt = Join-Path $parent ($leaf + '._old_' + (Get-Date -Format 'yyyyMMdd_HHmmss'));
        Rename-Item -LiteralPath $Path -NewName (Split-Path -Leaf $alt) -ErrorAction Stop;
        Start-Job -ScriptBlock { param($p) Start-Sleep -Seconds 3; for($i=0;$i -lt 5;$i++){ try{ Remove-Item -LiteralPath $p -Recurse -Force -ErrorAction Stop; break } catch { Start-Sleep -Seconds 2 } } } -ArgumentList $alt | Out-Null;
        Write-Info "Renamed locked path to: $alt (scheduled for deletion)"; return
    } catch {}
    Write-Err "Failed to clean path: $Path"
}

# Paths
$ScriptDir   = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptDir
$ReleaseRoot = Join-Path $ProjectRoot 'release'
$VenvPath    = Join-Path $ReleaseRoot '.venv_nuitka'
$DistDir     = Join-Path $ReleaseRoot 'dist_single'
$LogPath     = Join-Path $ReleaseRoot 'build_single_nuitka.log'

if (-not (Test-Path (Join-Path $ProjectRoot 'main.py'))) { Write-Err "main.py not found in project root: $ProjectRoot"; exit 1 }

# Transcript
$TranscriptStarted = $false
try { if (!(Test-Path $ReleaseRoot)) { New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null }
      try { Stop-Transcript | Out-Null } catch {}
      Start-Transcript -Path $LogPath -Force | Out-Null; $TranscriptStarted = $true; Write-Info "Build log: $LogPath" } catch { Write-Err "Could not start transcript: $($_.Exception.Message)" }

Write-Info "PowerShell: $($PSVersionTable.PSVersion)"
Write-Info "Project root: $ProjectRoot"
Write-Info "Dist dir    : $DistDir"

if ($Clean) { Write-Info 'Cleaning previous build artifacts...'; if (Test-Path $DistDir) { Remove-TreeRobust -Path $DistDir }; Write-Info 'Cleanup complete.' }

# Python probe (base interpreter)
try {
    $BasePython = & python -c "import sys; print(sys.executable)" 2>$null; if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($BasePython)) { throw "Python command failed" }
    $BasePython = $BasePython.Trim(); if (-not (Test-Path $BasePython)) { throw "Python executable not found at: $BasePython" }
    $BasePyVer = & "$BasePython" -c "import sys; print('{}.{}.{}'.format(*sys.version_info[:3]))" 2>$null
    Write-Info "Base Python: $BasePyVer at $BasePython"
} catch { Write-Err "Python not found or invalid. Error: $($_.Exception.Message)"; if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} } ; exit 1 }

# Create isolated venv and install deps (no global requirements)
if (-not (Test-Path $VenvPath)) {
    Write-Info "Creating virtual environment at $VenvPath"
    & "$BasePython" -m venv "$VenvPath"
}
$PythonExe = Join-Path $VenvPath 'Scripts/python.exe'
if (-not (Test-Path $PythonExe)) { Write-Err "Virtual environment python not found: $PythonExe"; if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }; exit 1 }

Write-Info 'Upgrading pip/setuptools/wheel and installing requirements + Nuitka (ordered-set, zstandard)'
& "$PythonExe" -m pip install --upgrade pip setuptools wheel *> $null
& "$PythonExe" -m pip install -r (Join-Path $ProjectRoot 'requirements.txt') *> $null
& "$PythonExe" -m pip install nuitka ordered-set zstandard *> $null

# Nuitka probe (venv)
try { & "$PythonExe" -c "import nuitka" 2>$null; if ($LASTEXITCODE -ne 0) { throw "Nuitka import failed" }
     $NuitkaVersion = (& "$PythonExe" -m nuitka --version 2>$null); if ($NuitkaVersion){ Write-Info ("Nuitka: " + $NuitkaVersion.Trim()) } else { Write-Info 'Nuitka validated' } } 
catch { Write-Err 'Nuitka not installed in venv'; if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }; exit 1 }

# Verify precompiled Qt resources
$RcPyPath = Join-Path $ProjectRoot 'ui/resources_rc.py'
if (-not (Test-Path $RcPyPath)) { Write-Err 'Missing ui/resources_rc.py. Generate with: pyside6-rcc resources.qrc -o ui/resources_rc.py'; if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }; exit 1 }

# Prepare output
if (!(Test-Path $DistDir)) { New-Item -ItemType Directory -Path $DistDir -Force | Out-Null }

$IconPath = Join-Path $ProjectRoot 'resources/ShittyPIP.ico'

# Build Nuitka onefile
Write-Info 'Starting Nuitka build (single-file)...'
$nuArgs = @(
    '-m','nuitka',
    '--onefile',
    '--assume-yes-for-downloads',
    '--enable-plugin=pyside6',
    '--nofollow-import-to=pytest',
    '--nofollow-import-to=tests',
    "--output-dir=$DistDir",
    '--output-filename=SPQ.exe'
)
if ($BuildDebug) {
    $nuArgs += @('--show-scons','--verbose')
    Write-Info 'Debug build: verbose flags enabled'
}
if ($Console) { $nuArgs += '--windows-console-mode=attach'; Write-Info 'Console enabled' } else { $nuArgs += '--windows-console-mode=disable'; Write-Info 'Console disabled' }
if (Test-Path $IconPath) { $nuArgs += "--windows-icon-from-ico=$IconPath"; Write-Info "Using icon: $IconPath" }
$nuArgs += (Join-Path $ProjectRoot 'main.py')

Write-Info "Nuitka command: $PythonExe $($nuArgs -join ' ')"

$nuOut = Join-Path $ReleaseRoot 'nuitka_single_stdout.log'
$nuErr = Join-Path $ReleaseRoot 'nuitka_single_stderr.log'
foreach ($f in @($nuOut,$nuErr)) { if (Test-Path $f) { Remove-Item $f -Force } }

$sw = [Diagnostics.Stopwatch]::StartNew()
$proc = Start-Process -FilePath "$PythonExe" -ArgumentList $nuArgs -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru -RedirectStandardOutput $nuOut -RedirectStandardError $nuErr -Wait
$sw.Stop()
Write-Info ("Nuitka finished in {0:N1}s with exit code {1}" -f $sw.Elapsed.TotalSeconds, $proc.ExitCode)

if ($proc.ExitCode -ne 0) { Write-Err "Nuitka onefile build failed. See logs: $nuOut, $nuErr"; if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }; exit $proc.ExitCode }

$exePath = Join-Path $DistDir 'SPQ.exe'
if (-not (Test-Path $exePath)) { Write-Err "Missing output: $exePath"; if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }; exit 1 }

$sizeMB = [math]::Round((Get-Item $exePath).Length / 1MB, 1)
Write-Info "Single-file executable built: $exePath (${sizeMB} MB)"

if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
exit 0
