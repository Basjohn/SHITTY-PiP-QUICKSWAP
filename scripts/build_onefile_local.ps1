[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Console
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }

function Get-AppVersion {
    param([string]$VersionFile = "version.py")
    $versionPath = Join-Path $PSScriptRoot ".." $VersionFile
    if (-not (Test-Path $versionPath)) {
        Write-Warning "version.py not found, using fallback 2.1.0a"
        return @{ Win32='2.1.0.0'; String='2.1.0a'; Company='Faecal Failures'; DisplayName='Shitty PiP QuickSwap'; Description='Overlays, too many fucking overlays.' }
    }
    try {
        $content = Get-Content $versionPath -Raw
        $major = if ($content -match 'VERSION_MAJOR\s*=\s*(\d+)') { [int]$Matches[1] } else { 2 }
        $minor = if ($content -match 'VERSION_MINOR\s*=\s*(\d+)') { [int]$Matches[1] } else { 1 }
        $patch = if ($content -match 'VERSION_PATCH\s*=\s*(\d+)') { [int]$Matches[1] } else { 0 }
        $suffix = if ($content -match 'VERSION_SUFFIX\s*=\s*"([^"]*)"') { $Matches[1] } else { '' }
        $company = if ($content -match 'APP_COMPANY\s*=\s*"([^"]*)"') { $Matches[1] } else { 'Faecal Failures' }
        $displayName = if ($content -match 'APP_DISPLAY_NAME\s*=\s*"([^"]*)"') { $Matches[1] } else { 'Shitty PiP QuickSwap' }
        $description = if ($content -match 'APP_DESCRIPTION\s*=\s*"([^"]*)"') { $Matches[1] } else { 'Overlays, too many fucking overlays.' }
        return @{ Win32="$major.$minor.$patch.0"; String="$major.$minor.$patch$suffix"; Company=$company; DisplayName=$displayName; Description=$description }
    } catch {
        return @{ Win32='2.1.0.0'; String='2.1.0a'; Company='Faecal Failures'; DisplayName='Shitty PiP QuickSwap'; Description='Overlays, too many fucking overlays.' }
    }
}

# Paths
$ScriptDir   = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptDir
$ReleaseRoot = Join-Path $ProjectRoot 'release'
$DistDir     = Join-Path $ReleaseRoot 'onefile_local'
$LogPath     = Join-Path $ReleaseRoot 'build_onefile_local.log'

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
Write-Info "Output dir  : $DistDir"

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
    $PythonVer = & python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
    Write-Info "Python: $PythonVer at $PythonExe"
} catch {
    Write-Err 'Python not available. Ensure python is in PATH.'
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1
}

# Nuitka check
try {
    $NuitkaVer = & python -m nuitka --version 2>$null
    if ($LASTEXITCODE -ne 0) { throw "Nuitka command failed" }
    Write-Info "Nuitka: $NuitkaVer"
} catch {
    Write-Err 'Nuitka not available. Install with: pip install nuitka'
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1
}

# Read version from version.py
$version = Get-AppVersion
Write-Info "Building version: $($version.String)"

# Prepare output
if (!(Test-Path $DistDir)) { New-Item -ItemType Directory -Path $DistDir -Force | Out-Null }

$IconPath = Join-Path $ProjectRoot 'resources/ShittyPIP.ico'

# Onefile build with persistent cache extraction (user cache directory)
Write-Info 'Starting onefile build with local extraction (AV-safe)...'
$nuArgs = @(
    '-m', 'nuitka',
    '--onefile',
    '--assume-yes-for-downloads',
    '--enable-plugin=pyside6',
    "--output-dir=$DistDir",
    '--output-filename=SPQ.exe',
    '--onefile-no-compression',
    '--onefile-tempdir-spec={CACHE_DIR}/SPQ/runtime',
    "--windows-company-name=`"$($version.Company)`"",
    "--windows-product-name=`"$($version.DisplayName)`"",
    "--windows-file-version=$($version.Win32)",
    "--windows-product-version=$($version.Win32)",
    "--windows-file-description=`"$($version.Description)`""
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

# Run Nuitka
$nuOut = Join-Path $ReleaseRoot 'nuitka_onefile_local_stdout.log'
$nuErr = Join-Path $ReleaseRoot 'nuitka_onefile_local_stderr.log'

$proc = Start-Process -FilePath $PythonExe -ArgumentList $nuArgs -NoNewWindow -PassThru -RedirectStandardOutput $nuOut -RedirectStandardError $nuErr -Wait

$elapsed = $proc.ExitTime - $proc.StartTime
Write-Info "Nuitka finished in $([math]::Round($elapsed.TotalSeconds, 1))s with exit code $($proc.ExitCode)"

if ($proc.ExitCode -ne 0) { 
    Write-Err "Nuitka build failed. See logs: $nuOut, $nuErr"
    if (Test-Path $nuErr) {
        Write-Host "--- Error Log ---" -ForegroundColor Red
        Get-Content $nuErr | Write-Host -ForegroundColor Red
    }
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit $proc.ExitCode 
}

# Verify output
$exePath = Join-Path $DistDir 'SPQ.exe'
if (-not (Test-Path $exePath)) {
    Write-Err "Expected output not found: $exePath"
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1
}

# Create accompanying subdirectories for cleaner structure
Write-Info "Creating application folder structure..."
New-Item -ItemType Directory -Path (Join-Path $DistDir 'data') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $DistDir 'logs') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $DistDir 'settings') -Force | Out-Null

# Calculate size
$exeSizeMB = [math]::Round((Get-Item $exePath).Length / 1MB, 1)

Write-Info ''
Write-Info '==================================='
Write-Info 'Onefile (persistent cache) build completed!'
Write-Info '==================================='
Write-Info "Output directory: $DistDir"
Write-Info "Executable: SPQ.exe (${exeSizeMB} MB)"
Write-Info ''
Write-Info 'Structure:'
Write-Info '  SPQ.exe          -> Main application (run this)'
Write-Info '  data/            -> Application resources'
Write-Info '  settings/        -> User configuration'
Write-Info '  logs/            -> Runtime logs'
Write-Info ''
Write-Info 'Extraction behavior:'
Write-Info '  - First launch: Extracts to %LOCALAPPDATA%\SPQ\runtime (~2-3s delay)'
Write-Info '  - Subsequent launches: Uses cached extraction (instant)'
Write-Info '  - Cache persists between sessions'
Write-Info '  - Portable: Cache location moves with exe location'

if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
exit 0
