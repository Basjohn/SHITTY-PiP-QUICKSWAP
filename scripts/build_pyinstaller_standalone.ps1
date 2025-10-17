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
$WorkDir     = Join-Path $ReleaseRoot 'work_pyinstaller'
$FinalDistDir = Join-Path $ReleaseRoot 'pyinstaller_standalone'
$LogPath     = Join-Path $ReleaseRoot 'build_pyinstaller_standalone.log'

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
    $PythonVer = & python -c "import sys; print('.'.join(map(str, sys.version_info[:3])))" 2>$null
    Write-Info "Python: $PythonVer at $PythonExe"
} catch {
    Write-Err 'Python not available. Ensure python is in PATH.'
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1
}

# PyInstaller check
try {
    & python -c "import PyInstaller" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "PyInstaller import failed" }
    $PyInstallerVer = & python -c "import PyInstaller; print(PyInstaller.__version__)" 2>$null
    Write-Info "PyInstaller: $PyInstallerVer"
} catch {
    Write-Err 'PyInstaller not available. Install with: pip install pyinstaller'
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1
}

# Read version from version.py
$version = Get-AppVersion
Write-Info "Building version: $($version.String)"

# Prepare output
if (!(Test-Path $WorkDir)) { New-Item -ItemType Directory -Path $WorkDir -Force | Out-Null }

$IconPath = Join-Path $ProjectRoot 'resources/ShittyPIP.ico'
$ResourcesPath = Join-Path $ProjectRoot 'resources'
$ThemesPath = Join-Path $ProjectRoot 'themes'

# PyInstaller standalone build with clean structure
Write-Info 'Starting PyInstaller standalone build...'
$pyArgs = @(
    '-m', 'PyInstaller',
    '--distpath', $WorkDir,
    '--workpath', (Join-Path $WorkDir 'build'),
    '--specpath', $WorkDir,
    '--name', 'SPQCoreApp',
    '--onedir'
)

# Console mode
if ($Console) { 
    $pyArgs += '--console'
    Write-Info 'Console enabled' 
} else { 
    $pyArgs += '--windowed'
    Write-Info 'Console disabled' 
}

# Icon
if (Test-Path $IconPath) { 
    $pyArgs += '--icon', $IconPath
    Write-Info "Using icon: $IconPath" 
}

# Add data files
if (Test-Path $ResourcesPath) {
    $pyArgs += '--add-data', "$ResourcesPath;resources"
    Write-Info "Including resources: $ResourcesPath"
}
if (Test-Path $ThemesPath) {
    $pyArgs += '--add-data', "$ThemesPath;themes"
    Write-Info "Including themes: $ThemesPath"
}

# Hidden imports for PySide6
$pyArgs += '--hidden-import', 'PySide6.QtCore'
$pyArgs += '--hidden-import', 'PySide6.QtGui'
$pyArgs += '--hidden-import', 'PySide6.QtWidgets'

# Hidden imports for lazy-loaded modules (function-scope imports)
$pyArgs += '--hidden-import', 'core.opacity.manager'

# Clean build
$pyArgs += '--clean'
$pyArgs += '--noconfirm'

# Version info (Windows only)
$pyArgs += '--version-file', (Join-Path $WorkDir 'version_info.txt')

# Main script
$MainPy = Join-Path $ProjectRoot 'main.py'
$pyArgs += $MainPy

# Create version info file for PyInstaller
$versionInfoContent = @"
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=($($version.Win32.Replace('.', ', '))),
    prodvers=($($version.Win32.Replace('.', ', '))),
    mask=0x3f,
    flags=0x0,
    OS=0x40004,
    fileType=0x1,
    subtype=0x0,
    date=(0, 0)
  ),
  kids=[
    StringFileInfo(
      [
      StringTable(
        u'040904B0',
        [StringStruct(u'CompanyName', u'$($version.Company)'),
        StringStruct(u'FileDescription', u'$($version.Description)'),
        StringStruct(u'FileVersion', u'$($version.Win32)'),
        StringStruct(u'InternalName', u'SPQ'),
        StringStruct(u'OriginalFilename', u'SPQ.exe'),
        StringStruct(u'ProductName', u'$($version.DisplayName)'),
        StringStruct(u'ProductVersion', u'$($version.String)')])
      ]),
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)
"@

$versionInfoPath = Join-Path $WorkDir 'version_info.txt'
Set-Content -Path $versionInfoPath -Value $versionInfoContent -Encoding UTF8
Write-Info "Created version info: $versionInfoPath"

Write-Info "PyInstaller command: $PythonExe $($pyArgs -join ' ')"

# Run PyInstaller
$pyOut = Join-Path $ReleaseRoot 'pyinstaller_standalone_stdout.log'
$pyErr = Join-Path $ReleaseRoot 'pyinstaller_standalone_stderr.log'

$sw = [Diagnostics.Stopwatch]::StartNew()
$proc = Start-Process -FilePath $PythonExe -ArgumentList $pyArgs -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru -RedirectStandardOutput $pyOut -RedirectStandardError $pyErr -Wait
$sw.Stop()
Write-Info "PyInstaller finished in $([math]::Round($sw.Elapsed.TotalSeconds, 1))s with exit code $($proc.ExitCode)"

if ($proc.ExitCode -ne 0) { 
    Write-Err "PyInstaller build failed. See logs: $pyOut, $pyErr"
    if (Test-Path $pyErr) {
        Write-Host "--- Error Log ---" -ForegroundColor Red
        Get-Content $pyErr | Write-Host -ForegroundColor Red
    }
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit $proc.ExitCode 
}

# Verify PyInstaller output
$pyInstallerDist = Join-Path $WorkDir 'SPQCoreApp'
if (-not (Test-Path $pyInstallerDist)) {
    Write-Err "PyInstaller dist directory not found: $pyInstallerDist"
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1
}

# Create clean directory structure (keep PyInstaller structure intact)
Write-Info "Setting up final distribution..."
if (!(Test-Path $FinalDistDir)) { New-Item -ItemType Directory -Path $FinalDistDir -Force | Out-Null }

# Copy entire PyInstaller dist to final location
Copy-Item -Path (Join-Path $pyInstallerDist '*') -Destination $FinalDistDir -Recurse -Force
Write-Info "Copied PyInstaller output to $FinalDistDir"

# Rename main exe
$coreExe = Join-Path $FinalDistDir 'SPQCoreApp.exe'
$finalExe = Join-Path $FinalDistDir 'SPQ.exe'
if (Test-Path $coreExe) {
    Move-Item $coreExe $finalExe -Force
    Write-Info "Renamed: SPQCoreApp.exe -> SPQ.exe"
} else {
    Write-Err "Main exe not found: $coreExe"
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1
}

# Create empty settings and logs directories at same level
New-Item -ItemType Directory -Path (Join-Path $FinalDistDir 'settings') -Force | Out-Null
New-Item -ItemType Directory -Path (Join-Path $FinalDistDir 'logs') -Force | Out-Null

# Calculate size
$exeSizeMB = [math]::Round((Get-Item $finalExe).Length / 1MB, 1)
$totalSizeMB = [math]::Round((Get-ChildItem $FinalDistDir -Recurse | Measure-Object -Property Length -Sum).Sum / 1MB, 1)

Write-Info ''
Write-Info '==================================='
Write-Info 'PyInstaller standalone build completed!'
Write-Info '==================================='
Write-Info "Output directory: $FinalDistDir"
Write-Info "Main exe: SPQ.exe (${exeSizeMB} MB)"
Write-Info "Total size: ${totalSizeMB} MB"
Write-Info ''
Write-Info 'Structure:'
Write-Info '  SPQ.exe          -> Main application (run this)'
Write-Info '  _internal/       -> Python runtime & dependencies (PyInstaller structure)'
Write-Info '  settings/        -> User configuration'
Write-Info '  logs/            -> Runtime logs'

if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
exit 0
