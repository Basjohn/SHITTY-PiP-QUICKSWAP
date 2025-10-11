[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Console,
    [switch]$BuildDebug,
    [string]$CertPath = '',
    [string]$CertPassword = '',
    [string]$TimestampUrl = 'http://timestamp.digicert.com'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }

function Get-AppVersion {
    param([string]$VersionFile = "version.py")
    $versionPath = Join-Path $PSScriptRoot ".." $VersionFile
    if (-not (Test-Path $versionPath)) { return @{ Win32='2.1.0.0'; String='2.1.0a'; Company='Faecal Failures'; DisplayName='Shitty PiP QuickSwap'; Description='Overlays, too many fucking overlays.' } }
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
    } catch { return @{ Win32='2.1.0.0'; String='2.1.0a'; Company='Faecal Failures'; DisplayName='Shitty PiP QuickSwap'; Description='Overlays, too many fucking overlays.' } }
}

# Paths
$ScriptDir   = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptDir
$ReleaseRoot = Join-Path $ProjectRoot 'release'
$DistDir     = Join-Path $ReleaseRoot 'dist_single_av_safe'
$LogPath     = Join-Path $ReleaseRoot 'build_single_nuitka_av_safe.log'

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

Write-Info "PowerShell: $($PSVersionTable.PSVersion)"
Write-Info "Project root: $ProjectRoot"
Write-Info "Dist dir    : $DistDir"

if ($Clean) { 
    Write-Info 'Cleaning previous build artifacts...'
    if (Test-Path $DistDir) { Remove-Item $DistDir -Recurse -Force }
    Write-Info 'Cleanup complete.' 
}

# Code signing setup
$SigningEnabled = $false
if ($CertPath -and (Test-Path $CertPath)) {
    # Check for signtool.exe
    $SignTool = $null
    $WindowsKitsPath = "${env:ProgramFiles(x86)}\Windows Kits\10\bin"
    if (Test-Path $WindowsKitsPath) {
        $SignToolCandidates = Get-ChildItem -Path $WindowsKitsPath -Recurse -Filter "signtool.exe" -ErrorAction SilentlyContinue | Sort-Object FullName -Descending
        if ($SignToolCandidates) {
            $SignTool = $SignToolCandidates[0].FullName
        }
    }
    
    if ($SignTool -and (Test-Path $SignTool)) {
        $SigningEnabled = $true
        Write-Info "Code signing enabled with certificate: $CertPath"
        Write-Info "Using signtool: $SignTool"
    } else {
        Write-Info "Certificate provided but signtool.exe not found - code signing will be skipped"
    }
} else {
    Write-Info "signtool.exe not found - code signing will be skipped"
}

# Python setup with virtual environment for AV-safe build
$VenvPath = Join-Path $ReleaseRoot '.venv_nuitka_safe'
$PythonExe = Join-Path $VenvPath 'Scripts/python.exe'

# Check base Python
try {
    $BasePython = & python -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($BasePython)) { 
        throw "Python command failed" 
    }
    $BasePython = $BasePython.Trim()
    if (-not (Test-Path $BasePython)) { 
        throw "Python executable not found at: $BasePython" 
    }
    $PythonVersion = & "$BasePython" -c "import sys; print('{}.{}.{}'.format(*sys.version_info[:3]))" 2>$null
    Write-Info "Base Python: $PythonVersion at $BasePython"
} catch { 
    Write-Err "Python not found or invalid. Error: $($_.Exception.Message)"
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} } 
    exit 1 
}

# Create/update virtual environment
if (-not (Test-Path $PythonExe)) {
    Write-Info "Creating virtual environment at $VenvPath"
    & "$BasePython" -m venv "$VenvPath" --clear
    if ($LASTEXITCODE -ne 0) {
        Write-Err "Failed to create virtual environment"
        if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
        exit 1
    }
}

# Upgrade pip and install requirements
Write-Info "Upgrading pip/setuptools/wheel and installing requirements + Nuitka"
$RequirementsPath = Join-Path $ProjectRoot 'requirements.txt'
if (Test-Path $RequirementsPath) {
    & "$PythonExe" -m pip install --upgrade pip setuptools wheel
    & "$PythonExe" -m pip install -r "$RequirementsPath"
    & "$PythonExe" -m pip install nuitka
} else {
    Write-Err "requirements.txt not found at $RequirementsPath"
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1
}

# Verify Nuitka installation
try { 
    & "$PythonExe" -c "import nuitka" 2>$null
    if ($LASTEXITCODE -ne 0) { throw "Nuitka import failed" }
    $NuitkaVersion = (& "$PythonExe" -m nuitka --version 2>$null)
    if ($NuitkaVersion){ 
        Write-Info $NuitkaVersion.Trim()
    } else { 
        Write-Info 'Nuitka validated' 
    } 
} catch { 
    Write-Err 'Nuitka not available in virtual environment'
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1 
}

# Verify resources_rc.py exists
$RcPyPath = Join-Path $ProjectRoot 'ui/resources_rc.py'
if (-not (Test-Path $RcPyPath)) { 
    Write-Err 'Missing ui/resources_rc.py. Generate with: pyside6-rcc resources.qrc -o ui/resources_rc.py'
    if ($TranscriptStarted){ try{ Stop-Transcript|Out-Null }catch{} }
    exit 1 
}

# Prepare output
if (!(Test-Path $DistDir)) { New-Item -ItemType Directory -Path $DistDir -Force | Out-Null }

$IconPath = Join-Path $ProjectRoot 'resources/ShittyPIP.ico'

# Read version from version.py
$version = Get-AppVersion
Write-Info "Building version: $($version.String)"

# Build Nuitka onefile with AV-safe configuration
Write-Info 'Starting Nuitka build (AV-safe single-file)...'
$nuArgs = @(
    '-m', 'nuitka',
    '--onefile',
    '--assume-yes-for-downloads',
    '--enable-plugin=pyside6',
    "--output-dir=$DistDir",
    '--output-filename=SPQ.exe',
    "--windows-company-name=`"$($version.Company)`"",
    "--windows-product-name=`"$($version.DisplayName)`"",
    "--windows-file-version=$($version.Win32)",
    "--windows-product-version=$($version.Win32)",
    "--windows-file-description=`"$($version.Description)`"",
    '--onefile-no-compression',
    '--onefile-tempdir-spec={TEMP}/SPQ_{PID}_{TIME}'
)

if ($BuildDebug) {
    $nuArgs += '--show-scons'
    $nuArgs += '--verbose'
    Write-Info 'Debug build: verbose flags enabled'
}

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

$nuOut = Join-Path $ReleaseRoot 'nuitka_single_av_safe_stdout.log'
$nuErr = Join-Path $ReleaseRoot 'nuitka_single_av_safe_stderr.log'
foreach ($f in @($nuOut,$nuErr)) { if (Test-Path $f) { Remove-Item $f -Force } }

$sw = [Diagnostics.Stopwatch]::StartNew()
$proc = Start-Process -FilePath "$PythonExe" -ArgumentList $nuArgs -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru -RedirectStandardOutput $nuOut -RedirectStandardError $nuErr -Wait
$sw.Stop()
Write-Info ("Nuitka finished in {0:N1}s with exit code {1}" -f $sw.Elapsed.TotalSeconds, $proc.ExitCode)

if ($proc.ExitCode -ne 0) { 
    Write-Err "Nuitka onefile build failed. See logs: $nuOut, $nuErr"
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

# Code signing (if enabled)
if ($SigningEnabled -and $SignTool -and $CertPath -and (Test-Path $exePath)) {
    Write-Info "Signing executable: $exePath"
    $signArgs = @(
        'sign',
        '/f', "$CertPath",
        '/fd', 'SHA256',
        '/tr', "$TimestampUrl",
        '/td', 'SHA256',
        "$exePath"
    )
    
    if ($CertPassword) {
        $signArgs = $signArgs[0..2] + @('/p', "$CertPassword") + $signArgs[3..($signArgs.Length-1)]
    }
    
    try {
        & "$SignTool" @signArgs
        if ($LASTEXITCODE -eq 0) {
            Write-Info "Code signing completed successfully"
        } else {
            Write-Err "Code signing failed with exit code $LASTEXITCODE"
        }
    } catch {
        Write-Err "Code signing error: $($_.Exception.Message)"
    }
}

$sizeMB = [math]::Round((Get-Item $exePath).Length / 1MB, 1)
Write-Info "AV-safe build completed: $exePath (${sizeMB} MB)"

# Copy additional files if they exist
$ReadmePath = Join-Path $ProjectRoot 'README.txt'
if (Test-Path $ReadmePath) {
    Copy-Item $ReadmePath $DistDir -Force
    Write-Info "Copied README.txt to distribution"
}

if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
exit 0
