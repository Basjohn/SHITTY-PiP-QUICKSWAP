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
$WorkDir     = Join-Path $ReleaseRoot 'work_standalone'
$FinalDistDir = Join-Path $ReleaseRoot 'standalone'
$LogPath     = Join-Path $ReleaseRoot 'build_standalone.log'

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
Write-Info "Work dir    : $WorkDir"
Write-Info "Output dir  : $FinalDistDir"

if ($Clean) { 
    Write-Info 'Cleaning previous build artifacts...'
    if (Test-Path $WorkDir) { Remove-Item $WorkDir -Recurse -Force }
    if (Test-Path $FinalDistDir) { Remove-Item $FinalDistDir -Recurse -Force }
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

# Read version from version.py
$version = Get-AppVersion
Write-Info "Building version: $($version.String)"

# Prepare output
if (!(Test-Path $WorkDir)) { New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null }

$IconPath = Join-Path $ProjectRoot 'resources/ShittyPIP.ico'

# Standalone directory build (AV-safe baseline - MINIMAL FLAGS ONLY)
Write-Info 'Starting standalone build (directory with dependencies)...'
$nuArgs = @(
    '-m', 'nuitka',
    '--standalone',
    '--assume-yes-for-downloads',
    '--enable-plugin=pyside6',
    '--include-module=ctypes',
    '--follow-import-to=ctypes',
    "--output-dir=$WorkDir",
    '--output-filename=SPQCoreApp.exe',
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

$nuOut = Join-Path $ReleaseRoot 'nuitka_standalone_stdout.log'
$nuErr = Join-Path $ReleaseRoot 'nuitka_standalone_stderr.log'
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

# Nuitka creates a .dist folder for standalone builds - find it
Write-Info "Looking for Nuitka output in: $WorkDir"
$distFolders = @(Get-ChildItem -Path $WorkDir -Directory -Filter '*.dist' -ErrorAction SilentlyContinue)

if ($distFolders.Count -eq 0) { 
    Write-Err "No .dist folder found in $WorkDir"
    Write-Err "Nuitka may have failed or created output elsewhere."
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1 
}

$CoreDistPath = $distFolders[0].FullName
Write-Info "Found Nuitka output: $CoreDistPath"

$coreExe = Join-Path $CoreDistPath 'SPQCoreApp.exe'
if (-not (Test-Path $coreExe)) { 
    Write-Err "Missing executable: $coreExe"
    Write-Info "Contents of ${CoreDistPath}:"
    Get-ChildItem $CoreDistPath | ForEach-Object { Write-Info "  $($_.Name)" }
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1 
}

# Reorganize into clean structure: SPQ.exe in root, DLLs in data/
Write-Info 'Reorganizing build output...'

# Clean and create structure
if (Test-Path $FinalDistDir) { Remove-Item $FinalDistDir -Recurse -Force }
$DataDir = Join-Path $FinalDistDir 'data'
New-Item -ItemType Directory -Path $DataDir -Force | Out-Null

# Copy main exe to root and rename
$coreExeOld = Join-Path $CoreDistPath 'SPQCoreApp.exe'
$coreExeNew = Join-Path $FinalDistDir 'SPQ.exe'
Copy-Item -Path $coreExeOld -Destination $coreExeNew -Force
Write-Info "Copied and renamed SPQCoreApp.exe -> SPQ.exe (root)"

# Copy all DLLs and dependencies to data/
Write-Info "Copying DLLs and dependencies to data/..."
Get-ChildItem -Path $CoreDistPath | Where-Object { $_.Name -ne 'SPQCoreApp.exe' } | Copy-Item -Destination $DataDir -Recurse -Force

# Create subdirectories
New-Item -ItemType Directory -Path (Join-Path $FinalDistDir 'logs') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $FinalDistDir 'settings') -Force | Out-Null
Write-Info "Created clean structure: SPQ.exe in root, dependencies in data/"

# Calculate total size
$totalSizeMB = [math]::Round((Get-ChildItem $FinalDistDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)
$dataFileCount = (Get-ChildItem $DataDir -File -Recurse | Measure-Object).Count
$dllCount = (Get-ChildItem $DataDir -Filter '*.dll' -File -Recurse | Measure-Object).Count

Write-Info ''
Write-Info '==================================='
Write-Info 'Standalone build completed!'
Write-Info '==================================='
Write-Info "Output directory: $FinalDistDir"
Write-Info "Total size: ${totalSizeMB} MB"
Write-Info "Main executable: SPQ.exe (root)"
Write-Info "Dependencies: $dataFileCount files in data/ ($dllCount DLLs)"
Write-Info ''
Write-Info 'Structure (clean layout):'
Write-Info '  SPQ.exe          -> Main application (run this)'
Write-Info '  data/            -> All DLLs and dependencies'
Write-Info '  settings/        -> User configuration'
Write-Info '  logs/            -> Runtime logs'
Write-Info ''
Write-Info 'NOTE: All DLLs are in data/ subdirectory for clean root folder.'

# Clean up temp files
if (Test-Path $WorkDir) { Remove-Item $WorkDir -Recurse -Force -ErrorAction SilentlyContinue }

if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
exit 0
