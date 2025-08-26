[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Console,
    [switch]$BuildDebug
)

Set-StrictMode -Version Latest;
$ErrorActionPreference = 'Stop';

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan };
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red };

function Remove-TreeRobust {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    try { Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop; return } catch {}
    try {
        Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
            try { $_.Attributes = 'Normal'; if (-not $_.PSIsContainer) { [GC]::Collect(); [GC]::WaitForPendingFinalizers() } } catch {}
        };
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
$ScriptDir   = Split-Path -Parent $PSCommandPath;
$ProjectRoot = Split-Path -Parent $ScriptDir;
$ReleaseRoot = Join-Path $ProjectRoot 'release';
$WorkRoot    = Join-Path $ReleaseRoot 'work_pyinstaller';
$VenvPath    = Join-Path $ReleaseRoot '.venv_pyi';
$DistDir     = Join-Path $ReleaseRoot 'dist_py_single';
$SpecDir     = Join-Path $WorkRoot 'spec';
$BuildDir    = Join-Path $WorkRoot 'build';
$LogPath     = Join-Path $ReleaseRoot 'build_pyinstaller_single.log';

# Basic validation
if (-not (Test-Path (Join-Path $ProjectRoot 'main.py'))) { Write-Err "main.py not found in project root: $ProjectRoot"; exit 1 };
$RcPyPath = Join-Path $ProjectRoot 'ui/resources_rc.py';
if (-not (Test-Path $RcPyPath)) { Write-Err 'Missing ui/resources_rc.py. Generate with: pyside6-rcc resources.qrc -o ui/resources_rc.py'; exit 1 };

# Transcript
$TranscriptStarted = $false;
try { if (!(Test-Path $ReleaseRoot)) { New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null };
      if (!(Test-Path $WorkRoot))    { New-Item -ItemType Directory -Path $WorkRoot    -Force | Out-Null };
      try { Stop-Transcript | Out-Null } catch {};
      Start-Transcript -Path $LogPath -Force | Out-Null; $TranscriptStarted = $true; Write-Info "Build log: $LogPath" } catch { Write-Err "Could not start transcript: $($_.Exception.Message)" };

Write-Info "PowerShell: $($PSVersionTable.PSVersion)";
Write-Info "Project root: $ProjectRoot";
Write-Info "Dist dir    : $DistDir";

if ($Clean) {
    Write-Info 'Cleaning previous build artifacts...';
    foreach ($p in @($DistDir, $BuildDir, $SpecDir)) { if (Test-Path $p) { Remove-TreeRobust -Path $p } };
    Write-Info 'Cleanup complete.'
}

# Locate base Python
try {
    $BasePython = & python -c "import sys; print(sys.executable)" 2>$null;
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($BasePython)) { throw "Python command failed" };
    $BasePython = $BasePython.Trim(); if (-not (Test-Path $BasePython)) { throw "Python not found at: $BasePython" };
    $PyVer = & "$BasePython" -c "import sys; print('{}.{}.{}'.format(*sys.version_info[:3]))" 2>$null;
    Write-Info "Base Python: $PyVer -> $BasePython";
} catch { Write-Err "Python not found. Install Python 3.11+ and retry. Error: $($_.Exception.Message)"; if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }; exit 1 };

# Self-contained venv
if (-not (Test-Path $VenvPath)) {
    Write-Info "Creating virtual environment at $VenvPath";
    & "$BasePython" -m venv "$VenvPath";
}
$VenvPy = Join-Path $VenvPath 'Scripts/python.exe';
if (-not (Test-Path $VenvPy)) { Write-Err "Virtual environment python not found: $VenvPy"; if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }; exit 1 };

Write-Info 'Upgrading pip/setuptools/wheel and installing requirements + PyInstaller';
& "$VenvPy" -m pip install --upgrade pip setuptools wheel *> $null;
& "$VenvPy" -m pip install -r (Join-Path $ProjectRoot 'requirements.txt') *> $null;
& "$VenvPy" -m pip install pyinstaller *> $null;

# Ensure output dirs
foreach ($d in @($DistDir, $SpecDir, $BuildDir)) { if (-not (Test-Path $d)) { New-Item -ItemType Directory -Path $d -Force | Out-Null } };

# Build args
$IconPath = Join-Path $ProjectRoot 'resources/ShittyPIP.ico';
$commonArgs = @(
    '--noconfirm',
    '--clean',
    '--onefile',
    "--distpath=$DistDir",
    "--workpath=$BuildDir",
    "--specpath=$SpecDir",
    '--name','SPQ'
);
if ($BuildDebug) { $commonArgs += @('--log-level','DEBUG','--debug','all'); Write-Info 'Debug build: PyInstaller debug logging enabled' };
if (-not $Console) { $commonArgs += '--noconsole' } else { Write-Info 'Console enabled' };
if (Test-Path $IconPath) { $commonArgs += @('--icon', $IconPath) };
# Avoid packing tests
$commonArgs += @('--exclude-module','pytest','--exclude-module','tests');
# Collect common libs used by the app
$commonArgs += @('--collect-all','PySide6','--collect-submodules','numpy','--collect-submodules','dxcam','--collect-submodules','comtypes');
# Bring along runtime theme files and settings by default
if (Test-Path (Join-Path $ProjectRoot 'themes'))   { $commonArgs += @('--add-data', ("{0};themes" -f (Join-Path $ProjectRoot 'themes'))) };
if (Test-Path (Join-Path $ProjectRoot 'settings')) { $commonArgs += @('--add-data', ("{0};settings" -f (Join-Path $ProjectRoot 'settings'))) };
if (Test-Path (Join-Path $ProjectRoot 'resources')){ $commonArgs += @('--add-data', ("{0};resources" -f (Join-Path $ProjectRoot 'resources'))) };

$entry = Join-Path $ProjectRoot 'main.py';
$pyiArgs = $commonArgs + @($entry);

Write-Info "PyInstaller command: $VenvPy -m PyInstaller $($pyiArgs -join ' ')";
$sw = [Diagnostics.Stopwatch]::StartNew();
$proc = Start-Process -FilePath $VenvPy -ArgumentList @('-m','PyInstaller') + $pyiArgs -NoNewWindow -PassThru -Wait -WorkingDirectory $ProjectRoot;
$sw.Stop();
Write-Info ("PyInstaller (onefile) finished in {0:N1}s with exit code {1}" -f $sw.Elapsed.TotalSeconds, $proc.ExitCode);

if ($proc.ExitCode -ne 0) { Write-Err "PyInstaller onefile build failed."; if ($TranscriptStarted){ try { Stop-Transcript | Out-Null } catch {} }; exit $proc.ExitCode };

# Validate output
$exePath = Join-Path $DistDir 'SPQ.exe';
if (-not (Test-Path $exePath)) { Write-Err "Missing output EXE: $exePath"; if ($TranscriptStarted){ try { Stop-Transcript | Out-Null } catch {} }; exit 1 };

$sizeMB = [math]::Round((Get-Item $exePath).Length / 1MB, 1);
Write-Info "One-file executable built: $exePath (${sizeMB} MB)";
if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} };
exit 0;
