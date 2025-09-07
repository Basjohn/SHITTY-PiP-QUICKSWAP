[CmdletBinding()]
param(
    [switch]$Clean,
    [switch]$Console
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Write-Info($msg) { Write-Host "[INFO] $msg" -ForegroundColor Cyan }
function Write-Err($msg)  { Write-Host "[ERROR] $msg" -ForegroundColor Red }

# Robust directory removal to handle locked files
function Remove-TreeRobust {
    param([Parameter(Mandatory=$true)][string]$Path)
    if (-not (Test-Path -LiteralPath $Path)) { return }
    
    # First attempt: simple removal
    try {
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
        return
    } catch {
        Write-Info "Standard removal failed for ${Path}, trying advanced cleanup..."
    }
    
    # Second attempt: clear attributes and retry
    try {
        Get-ChildItem -LiteralPath $Path -Recurse -Force -ErrorAction SilentlyContinue | ForEach-Object {
            try { 
                $_.Attributes = 'Normal' 
                if ($_.PSIsContainer -eq $false) {
                    # Try to unlock file handles
                    [System.GC]::Collect()
                    [System.GC]::WaitForPendingFinalizers()
                }
            } catch {}
        }
        Start-Sleep -Milliseconds 500
        Remove-Item -LiteralPath $Path -Recurse -Force -ErrorAction Stop
        return
    } catch {
        Write-Info "Attribute clearing failed, trying rename fallback..."
    }
    
    # Third attempt: rename and schedule background deletion
    try {
        $parent = Split-Path -Parent $Path
        $leaf   = Split-Path -Leaf   $Path
        $alt    = Join-Path $parent ($leaf + '._old_' + (Get-Date -Format 'yyyyMMdd_HHmmss'))
        Rename-Item -LiteralPath $Path -NewName (Split-Path -Leaf $alt) -ErrorAction Stop
        
        # Schedule background cleanup
        Start-Job -ScriptBlock {
            param($targetPath)
            Start-Sleep -Seconds 3
            for ($i = 0; $i -lt 5; $i++) {
                try { 
                    Remove-Item -LiteralPath $targetPath -Recurse -Force -ErrorAction Stop
                    break
                } catch {
                    Start-Sleep -Seconds 2
                }
            }
        } -ArgumentList $alt | Out-Null
        
        Write-Info "Renamed locked path to: ${alt} (scheduled for background deletion)"
        return
    } catch {}
    
    Write-Err "Failed to clean path: ${Path} (locked files detected). Build may fail."
}

# Paths
$ScriptDir   = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptDir
$ReleaseRoot = Join-Path $ProjectRoot 'release'
$WorkRoot    = Join-Path $ReleaseRoot 'work'
$DistDir     = Join-Path $ReleaseRoot 'dist'
$BinDir      = Join-Path $DistDir 'data/bin'
$DataDir     = Join-Path $DistDir 'data'
$LogPath     = Join-Path $ReleaseRoot 'build.log'

# Validate project structure
if (-not (Test-Path (Join-Path $ProjectRoot 'main.py'))) {
    Write-Err "main.py not found in project root: ${ProjectRoot}"
    exit 1
}

# Start transcript with better error handling
$TranscriptStarted = $false
try {
    if (!(Test-Path $ReleaseRoot)) { 
        New-Item -ItemType Directory -Path $ReleaseRoot -Force | Out-Null 
    }
    
    # Stop any existing transcript first
    try { Stop-Transcript | Out-Null } catch {}
    
    Start-Transcript -Path $LogPath -Force | Out-Null
    $TranscriptStarted = $true
    Write-Info "Build log: ${LogPath}"
} catch {
    Write-Err "Could not start transcript: $($_.Exception.Message)"
    Write-Info "Continuing without logging..."
}

Write-Info "Project root: ${ProjectRoot}"
Write-Info "Dist dir    : ${DistDir}"
Write-Info "PowerShell version: $($PSVersionTable.PSVersion)"

# Clean with better feedback
if ($Clean) {
    Write-Info 'Cleaning previous build artifacts...'
    $cleanPaths = @($WorkRoot, $DistDir)
    foreach ($p in $cleanPaths) { 
        if (Test-Path $p) { 
            Write-Info "Removing: ${p}"
            Remove-TreeRobust -Path $p 
        }
    }
    Write-Info 'Cleanup complete.'
}

# Locate and validate Python
try {
    $PythonExe = & python -c "import sys; print(sys.executable)" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($PythonExe)) {
        throw "Python command failed or returned empty path"
    }
    $PythonExe = $PythonExe.Trim()
    
    if (-not (Test-Path $PythonExe)) {
        throw "Python executable not found at: ${PythonExe}"
    }
    
    $PythonVersion = & "$PythonExe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
    Write-Info "Python: ${PythonVersion} at ${PythonExe}"
} catch {
    Write-Err "Python not found or invalid. Error: $($_.Exception.Message)"
    Write-Err "Please ensure Python is installed and available on PATH."
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

# Validate Nuitka with better error reporting
try {
    & "$PythonExe" -c "import nuitka" 2>$null
    if ($LASTEXITCODE -ne 0) {
        throw "Nuitka import failed"
    }
    $NuitkaVersion = (& "$PythonExe" -m nuitka --version 2>$null)
    if (-not [string]::IsNullOrWhiteSpace($NuitkaVersion)) {
        $NuitkaVersion = $NuitkaVersion.Trim()
        Write-Info "Nuitka: $NuitkaVersion (validated import)"
    } else {
        Write-Info "Nuitka: (validated import; version query unavailable)"
    }
} catch {
    Write-Err "Nuitka not found or invalid installation."
    Write-Err "Please install: python -m pip install nuitka ordered-set zstandard"
    Write-Err "Error details: $($_.Exception.Message)"
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

# Validate required dependencies (runtime + build)
function Test-Imports {
    param(
        [Parameter(Mandatory=$true)][string]$PythonExe,
        [Parameter(Mandatory=$true)][hashtable]$Map,
        [Parameter(Mandatory=$true)][string]$Title
    )
    $missing = @()
    foreach ($pkg in $Map.Keys) {
        $mods = $Map[$pkg]
        $ok = $false
        foreach ($m in $mods) {
            try {
                & "$PythonExe" -c "import $m" 2>$null
                if ($LASTEXITCODE -eq 0) { $ok = $true; break }
            } catch { }
        }
        if (-not $ok) { $missing += $pkg }
    }
    if ($missing.Count -gt 0) {
        Write-Err "$Title missing: $($missing -join ', ')"
        Write-Err "Install with: python -m pip install -r `"$ProjectRoot\requirements.txt`""
        if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
        exit 1
    } else {
        Write-Info "$Title validated: $(( $Map.Keys ) -join ', ')"
    }
}

$RuntimeImportMap = @{
    'PySide6' = @('PySide6')
    'psutil'  = @('psutil')
    'pywin32' = @('win32api','win32gui')
    'pycaw'   = @('pycaw')
    'comtypes'= @('comtypes')
    'dxcam'   = @('dxcam')
    'numpy'   = @('numpy')
}

$BuildImportMap = @{
    'ordered-set' = @('ordered_set')
    'zstandard'   = @('zstandard')
    'pyinstaller' = @('PyInstaller')
}

Test-Imports -PythonExe $PythonExe -Map $RuntimeImportMap -Title 'Runtime dependencies'
Test-Imports -PythonExe $PythonExe -Map $BuildImportMap   -Title 'Build-time dependencies'

# Verify precompiled Qt resources
$RcPyPath = Join-Path $ProjectRoot 'ui/resources_rc.py'
if (Test-Path $RcPyPath) {
    Write-Info "Using precompiled Qt resources: ${RcPyPath}"
} else {
    Write-Err "Missing ui/resources_rc.py"
    Write-Err "Generate with: pyside6-rcc resources.qrc -o ui/resources_rc.py"
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

# Build with Nuitka
$IconPath = Join-Path $ProjectRoot 'resources/ShittyPIP.ico'
$WorkOut  = $WorkRoot

Write-Info 'Starting Nuitka build (standalone one-directory)...'

# Create work directory
if (!(Test-Path $WorkOut)) {
    New-Item -ItemType Directory -Path $WorkOut -Force | Out-Null
}

$nuArgs = @(
    '-m', 'nuitka',
    '--standalone',
    '--assume-yes-for-downloads',
    '--enable-plugin=pyside6',
    '--nofollow-import-to=pytest',
    '--nofollow-import-to=tests',
    "--output-dir=$WorkOut",
    '--output-filename=SPQ_core.exe'
)

# Console mode selection
if ($Console) {
    $nuArgs += '--windows-console-mode=attach'
    Write-Info "Building with console output enabled"
} else {
    $nuArgs += '--windows-console-mode=disable'
    Write-Info "Building without console window"
}

# Add icon if available
if (Test-Path $IconPath) { 
    $nuArgs += "--windows-icon-from-ico=$IconPath" 
    Write-Info "Using icon: ${IconPath}"
}

$nuArgs += (Join-Path $ProjectRoot 'main.py')

Write-Info "Nuitka command: ${PythonExe} $($nuArgs -join ' ')"

# Run Nuitka with better output handling
$nuOut = Join-Path $ReleaseRoot 'nuitka_stdout.log'
$nuErr = Join-Path $ReleaseRoot 'nuitka_stderr.log'

# Clean old logs
foreach ($f in @($nuOut, $nuErr)) { 
    if (Test-Path $f) { Remove-Item $f -Force } 
}

Write-Info "Starting Nuitka compilation (this may take several minutes)..."
$stopwatch = [System.Diagnostics.Stopwatch]::StartNew()

$proc = Start-Process -FilePath "$PythonExe" -ArgumentList $nuArgs -WorkingDirectory $ProjectRoot -NoNewWindow -PassThru -RedirectStandardOutput $nuOut -RedirectStandardError $nuErr -Wait

$stopwatch.Stop()
$buildTime = $stopwatch.Elapsed.TotalSeconds

Write-Info "Nuitka completed in $([math]::Round($buildTime, 1)) seconds with exit code: $($proc.ExitCode)"

# Show build output
if (Test-Path $nuOut) { 
    $outContent = Get-Content $nuOut -Raw
    if (-not [string]::IsNullOrWhiteSpace($outContent)) {
        Write-Host '--- Nuitka stdout ---' -ForegroundColor Green
        Write-Host $outContent
    }
}

if (Test-Path $nuErr) { 
    $errContent = Get-Content $nuErr -Raw
    if (-not [string]::IsNullOrWhiteSpace($errContent)) {
        Write-Host '--- Nuitka stderr ---' -ForegroundColor Yellow
        Write-Host $errContent -ForegroundColor Yellow
    }
}

if ($proc.ExitCode -ne 0) {
    Write-Err "Nuitka build failed with exit code $($proc.ExitCode)"
    Write-Err "Check logs: ${nuOut} and ${nuErr}"
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit $proc.ExitCode
}

# Locate and validate Nuitka output
Write-Info "Locating Nuitka output in ${WorkOut}..."

$distCandidates = @()
if (Test-Path $WorkOut) {
    $distCandidates = @(Get-ChildItem -Directory -Path $WorkOut -Filter '*.dist' -ErrorAction SilentlyContinue | Where-Object { 
        Test-Path (Join-Path $_.FullName 'SPQ_core.exe') 
    })
    Write-Info "Found $($distCandidates.Count) dist directories: $(( $distCandidates | ForEach-Object Name ) -join ', ')"
}

# Fallback to common default
if ($distCandidates.Count -eq 0) {
    $fallback = Join-Path $WorkOut 'main.dist'
    if (Test-Path $fallback -and (Test-Path (Join-Path $fallback 'SPQ_core.exe'))) {
        $distCandidates = @(Get-Item $fallback)
        Write-Info "Using fallback directory: ${fallback}"
    }
}

if ($distCandidates.Count -eq 0) {
    Write-Err "Nuitka output not found!"
    Write-Err "Expected: *.dist directory containing SPQ_core.exe under ${WorkOut}"
    
    # Debug info
    if (Test-Path $WorkOut) {
        $items = Get-ChildItem $WorkOut
        Write-Info "Contents of ${WorkOut}:"
        $items | ForEach-Object { Write-Info "  $($_.Name) ($($_.GetType().Name))" }
    }
    
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

$CoreDistPath = $distCandidates[0].FullName
Write-Info "Found Nuitka output: ${CoreDistPath}"

# Stage the portable layout
Write-Info "Staging portable layout to ${DistDir}..."

# Create directory structure
foreach ($dir in @($DistDir, $BinDir, $DataDir)) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Copy Nuitka payload to data/bin
Write-Info "Copying Nuitka payload from ${CoreDistPath} to ${BinDir}..."
try {
    # Use robocopy for better large file handling
    $robocopyArgs = @($CoreDistPath, $BinDir, '/E', '/NP', '/R:3', '/W:1')
    $robocopyResult = Start-Process -FilePath 'robocopy' -ArgumentList $robocopyArgs -NoNewWindow -PassThru -Wait
    
    # Robocopy exit codes: 0-7 are success, 8+ are errors
    if ($robocopyResult.ExitCode -gt 7) {
        throw "Robocopy failed with exit code $($robocopyResult.ExitCode)"
    }
} catch {
    Write-Info "Robocopy failed, falling back to PowerShell copy..."
    Copy-Item -Recurse -Force -Path (Join-Path $CoreDistPath '*') -Destination $BinDir
}

# Validate staging
$StagedCore = Join-Path $BinDir 'SPQ_core.exe'
if (-not (Test-Path $StagedCore)) {
    Write-Err "Staging failed: SPQ_core.exe not found at ${StagedCore}"
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

$coreSize = [math]::Round((Get-Item $StagedCore).Length / 1MB, 1)
Write-Info "Staged SPQ_core.exe (${coreSize} MB)"

# Clean up empty directories
Write-Info 'Cleaning up empty directories...'
try {
    $cleanupDirs = @(
        (Join-Path $DataDir 'themes'),
        (Join-Path $DataDir 'resources')
    )
    
    foreach ($dir in $cleanupDirs) {
        if (Test-Path $dir) {
            $hasFiles = @(Get-ChildItem $dir -Recurse -File -ErrorAction SilentlyContinue).Count
            if ($hasFiles -eq 0) {
                Remove-TreeRobust -Path $dir
                Write-Info "Removed empty directory: ${dir}"
            }
        }
    }
} catch {
    Write-Info "Directory cleanup failed (non-critical): $($_.Exception.Message)"
}

# Create documentation
$Readme = @(
    'SPQ Portable Application',
    '========================',
    '',
    'Structure:',
    '  SPQ.exe              -> Application launcher',
    '  data/',
    '    bin/               -> Nuitka compiled application + dependencies',
    '  settings/            -> Portable configuration (auto-created)',
    '  logs/                -> Runtime logs (auto-created)',
    '',
    'Usage:',
    '  SPQ.exe              -> Run normally (no console)',
    '  SPQ.exe --debug      -> Run with debug console',
    '',
    'Assets (icons, themes) are embedded via Qt resources.',
    "Built on: $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss')",
    "Python: ${PythonVersion}",
    ("Nuitka: " + ($NuitkaVersion ? $NuitkaVersion : '(unknown)'))
) -join "`r`n"

Set-Content -Path (Join-Path $DistDir 'README.txt') -Value $Readme -Encoding UTF8

# Build launcher with improved compilation
Write-Info "Building launcher executable..."

$LauncherSrc = @'
using System;
using System.Diagnostics;
using System.IO;
using System.Runtime.InteropServices;
using System.Text;
using System.Windows.Forms;

internal static class Native
{
    private const uint LOAD_LIBRARY_SEARCH_DEFAULT_DIRS = 0x00001000;
    
    [DllImport("kernel32", SetLastError = true)]
    private static extern bool SetDefaultDllDirectories(uint DirectoryFlags);
    
    [DllImport("kernel32", CharSet = CharSet.Unicode, SetLastError = true)]
    private static extern IntPtr AddDllDirectory(string NewDirectory);

    public static void SetupDllDirectories(string directory)
    {
        try { SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS); } catch { }
        try { AddDllDirectory(directory); } catch { }
    }
}

public static class Program
{
    [STAThread]
    public static int Main(string[] args)
    {
        try
        {
            string baseDir = AppDomain.CurrentDomain.BaseDirectory;
            string binDir = Path.Combine(baseDir, "data", "bin");
            string coreExe = Path.Combine(binDir, "SPQ_core.exe");

            // Validate core application
            if (!Directory.Exists(binDir) || !File.Exists(coreExe))
            {
                string msg = string.Format("SPQ core application not found.\n\nExpected: {0}\n\nEnsure the complete application folder structure is present.", coreExe);
                MessageBox.Show(msg, "SPQ - Missing Core", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 2;
            }

            // Setup DLL resolution
            Native.SetupDllDirectories(binDir);
            
            // Ensure required directories exist
            string logsDir = Path.Combine(baseDir, "logs");
            string settingsDir = Path.Combine(baseDir, "settings");
            
            try { Directory.CreateDirectory(logsDir); } catch { }
            try { Directory.CreateDirectory(settingsDir); } catch { }

            // Configure process startup
            var psi = new ProcessStartInfo(coreExe)
            {
                UseShellExecute = false,
                WorkingDirectory = baseDir,
                Arguments = string.Join(" ", args),
                RedirectStandardOutput = true,
                RedirectStandardError = true,
                CreateNoWindow = true
            };

            // Setup environment
            try 
            { 
                psi.EnvironmentVariables["PATH"] = binDir + Path.PathSeparator + Environment.GetEnvironmentVariable("PATH");
                psi.EnvironmentVariables["SPQ_PORTABLE"] = "1";
                psi.EnvironmentVariables["SPQ_RUNTIME_ROOT"] = baseDir;
            } 
            catch { }

            // Check for debug mode
            bool isDebug = false;
            foreach (var arg in args)
            {
                if (string.Equals(arg, "--debug", StringComparison.OrdinalIgnoreCase))
                {
                    isDebug = true;
                    psi.CreateNoWindow = false;
                    break;
                }
            }

            // Start core process
            var process = Process.Start(psi);
            if (process == null)
            {
                MessageBox.Show("Failed to start SPQ core process.", "SPQ - Launch Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
                return 1;
            }

            // Handle output logging
            try
            {
                string timestamp = DateTime.Now.ToString("yyyyMMdd_HHmmss");
                string stdoutLog = Path.Combine(logsDir, string.Format("stdout_{0}.log", timestamp));
                string stderrLog = Path.Combine(logsDir, string.Format("stderr_{0}.log", timestamp));

                using (var stdout = new StreamWriter(stdoutLog, false) { AutoFlush = true })
                using (var stderr = new StreamWriter(stderrLog, false) { AutoFlush = true })
                {
                    process.OutputDataReceived += (s, e) => {
                        if (e.Data != null) {
                            try { stdout.WriteLine(e.Data); } catch { }
                        }
                    };
                    
                    process.ErrorDataReceived += (s, e) => {
                        if (e.Data != null) {
                            try { stderr.WriteLine(e.Data); } catch { }
                        }
                    };

                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();

                    if (isDebug)
                    {
                        var startTime = DateTime.UtcNow;
                        process.WaitForExit();
                        var endTime = DateTime.UtcNow;
                        var runtime = endTime - startTime;
                        
                        int exitCode = process.ExitCode;
                        string launcherLog = Path.Combine(logsDir, "launcher.log");
                        
                        try
                        {
                            string logEntry = string.Format("[{0:yyyy-MM-dd HH:mm:ss}] Process exited: code={1}, runtime={2:F1}s\n", DateTime.UtcNow, exitCode, runtime.TotalSeconds);
                            File.AppendAllText(launcherLog, logEntry);
                        }
                        catch { }

                        // Show warning for quick exits
                        if (runtime.TotalSeconds < 3 && exitCode != 0)
                        {
                            string msg = string.Format("SPQ core exited quickly with error code {0}.\n\nRuntime: {1:F1} seconds\n\nCheck the log files in the logs/ directory for details.", exitCode, runtime.TotalSeconds);
                            MessageBox.Show(msg, "SPQ - Quick Exit Detected", MessageBoxButtons.OK, MessageBoxIcon.Warning);
                        }

                        return exitCode;
                    }
                }
            }
            catch (Exception ex)
            {
                if (isDebug)
                {
                    MessageBox.Show(string.Format("Logging error: {0}", ex.Message), "SPQ - Debug", MessageBoxButtons.OK, MessageBoxIcon.Information);
                }
            }

            // Non-debug mode: don't wait
            if (!isDebug)
            {
                return 0;
            }

            process.WaitForExit();
            return process.ExitCode;
        }
        catch (Exception ex)
        {
            string msg = string.Format("SPQ Launcher Error:\n\n{0}\n\nStack Trace:\n{1}", ex.Message, ex.StackTrace);
            MessageBox.Show(msg, "SPQ - Critical Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            return 1;
        }
    }
}
'@

$LauncherSrcPath = Join-Path $WorkRoot 'SpqLauncher.cs'
Set-Content -Path $LauncherSrcPath -Value $LauncherSrc -Encoding UTF8
$LauncherOut = Join-Path $DistDir 'SPQ.exe'

function Get-CSC-Path {
    # Try to find csc.exe in PATH first
    try {
        $cmd = Get-Command csc.exe -ErrorAction SilentlyContinue
        if ($cmd -and (Test-Path $cmd.Source)) { 
            return $cmd.Source 
        }
    } catch { }
    
    # Try standard .NET Framework locations
    $candidates = @(
        (Join-Path $env:WINDIR 'Microsoft.NET\Framework64\v4.0.30319\csc.exe'),
        (Join-Path $env:WINDIR 'Microsoft.NET\Framework\v4.0.30319\csc.exe')
    )
    
    foreach ($candidate in $candidates) {
        if (Test-Path $candidate) { 
            return $candidate 
        }
    }
    
    return $null
}

function Invoke-CompileLauncherWithCSC {
    try {
        $cscPath = Get-CSC-Path
        if (-not $cscPath) { 
            Write-Info "csc.exe not found in standard locations"
            return $false 
        }
        
        Write-Info "Compiling launcher with csc.exe: ${cscPath}"
        
        # Build arguments array
        $cscArgs = @(
            '/nologo',
            '/optimize+',
            '/target:winexe',
            "/out:$LauncherOut",
            '/r:System.Windows.Forms.dll',
            '/r:System.Runtime.dll', 
            '/r:System.dll',
            "$LauncherSrcPath"
        )
        
        # Add icon if available
        if (Test-Path $IconPath) {
            $cscArgs += "/win32icon:$IconPath"
            Write-Info "Including icon: ${IconPath}"
        }
        
        Write-Info "CSC Arguments: $($cscArgs -join ' ')"
        
        # Compile with timeout
        # Capture CSC stdout/stderr for diagnostics
        $cscOut = Join-Path $ReleaseRoot 'csc_stdout.log'
        $cscErr = Join-Path $ReleaseRoot 'csc_stderr.log'
        foreach ($f in @($cscOut, $cscErr)) { if (Test-Path $f) { Remove-Item $f -Force } }

        $proc = Start-Process -FilePath $cscPath -ArgumentList $cscArgs -NoNewWindow -PassThru -RedirectStandardOutput $cscOut -RedirectStandardError $cscErr -Wait
        
        if ($proc.ExitCode -eq 0 -and (Test-Path $LauncherOut)) {
            $launcherSize = [math]::Round((Get-Item $LauncherOut).Length / 1KB, 1)
            Write-Info "Launcher compiled successfully (${launcherSize} KB)"
            return $true
        } else {
            Write-Err "CSC compilation failed with exit code: $($proc.ExitCode)"
            return $false
        }
    } catch {
        Write-Err "CSC compilation exception: $($_.Exception.Message)"
        return $false
    }
}

function Invoke-CompileLauncherWithAddType {
    try {
        Write-Info 'Compiling launcher with Add-Type (PowerShell/Roslyn)'
        
        $compilerOptions = '/optimize+'
        if (Test-Path $IconPath) { 
            $compilerOptions += " /win32icon:$IconPath"
        }
        
        $referencedAssemblies = @(
            'System.Windows.Forms',
            'System.Runtime',
            'System'
        )
        
        Add-Type -OutputType WindowsApplication -CompilerOptions $compilerOptions -OutputAssembly $LauncherOut -TypeDefinition (Get-Content $LauncherSrcPath -Raw) -ReferencedAssemblies $referencedAssemblies
        
        if (Test-Path $LauncherOut) {
            $launcherSize = [math]::Round((Get-Item $LauncherOut).Length / 1KB, 1)
            Write-Info "Launcher compiled with Add-Type (${launcherSize} KB)"
            return $true
        }
        
        return $false
    } catch {
        Write-Err "Add-Type compilation failed: $($_.Exception.Message)"
        return $false
    }
}

# Optional: Compile with Roslyn csc.dll via dotnet exec
function Invoke-CompileLauncherWithDotnetRoslyn {
    try {
        $dotnet = Get-Command dotnet -ErrorAction SilentlyContinue
        if (-not $dotnet) {
            Write-Info 'dotnet not found; skipping Roslyn csc.dll probe'
            return $false
        }

        Write-Info 'Probing dotnet SDK for Roslyn csc.dll'
        $info = & dotnet --info 2>$null
        $basePath = $null
        if ($info) {
            $m = ($info | Select-String -Pattern '^\s*Base Path:\s*(.+)$' -AllMatches).Matches | Select-Object -First 1
            if ($m) { $basePath = $m.Groups[1].Value.Trim() }
        }
        if (-not $basePath -or -not (Test-Path $basePath)) {
            # Try common location
            $sdkRoot = Join-Path $env:ProgramFiles 'dotnet\sdk'
            if (Test-Path $sdkRoot) {
                $latest = Get-ChildItem -Directory $sdkRoot | Sort-Object Name -Descending | Select-Object -First 1
                if ($latest) { $basePath = $latest.FullName }
            }
        }

        if (-not $basePath) {
            Write-Info 'Could not determine dotnet SDK base path; skipping Roslyn compile'
            return $false
        }

        # Support both legacy and new Roslyn locations (bincore)
        $cscDll = $null
        $cscCandidates = @(
            (Join-Path $basePath 'Roslyn\csc.dll'),
            (Join-Path $basePath 'Roslyn\bincore\csc.dll')
        )
        foreach ($cand in $cscCandidates) {
            if (Test-Path $cand) { $cscDll = $cand; break }
        }

        # If not found at the reported base path, enumerate SDKs under Program Files
        if (-not $cscDll) {
            $sdkRoot = Join-Path $env:ProgramFiles 'dotnet\sdk'
            if (Test-Path $sdkRoot) {
                $sdkDirs = Get-ChildItem -Directory $sdkRoot | Sort-Object Name -Descending
                foreach ($sdk in $sdkDirs) {
                    foreach ($rel in @('Roslyn\csc.dll','Roslyn\bincore\csc.dll')) {
                        $probe = Join-Path $sdk.FullName $rel
                        if (Test-Path $probe) { $cscDll = $probe; break }
                    }
                    if ($cscDll) { break }
                }
            }
        }

        if (-not $cscDll) {
            Write-Info 'Roslyn csc.dll not found in SDK base paths'
            return $false
        }

        Write-Info "Compiling launcher with dotnet exec: $cscDll"
        $cscArgs = @(
            '/nologo',
            '/optimize+',
            '/target:winexe',
            "/out:$LauncherOut",
            '/r:System.Windows.Forms.dll',
            '/r:System.Runtime.dll',
            '/r:System.dll',
            "$LauncherSrcPath"
        )

        if (Test-Path $IconPath) {
            $cscArgs += "/win32icon:$IconPath"
            Write-Info "Including icon: ${IconPath}"
        }

        Write-Info "Roslyn CSC Arguments: $($cscArgs -join ' ')"
        # Capture Roslyn stdout/stderr for diagnostics
        $rosOut = Join-Path $ReleaseRoot 'roslyn_stdout.log'
        $rosErr = Join-Path $ReleaseRoot 'roslyn_stderr.log'
        foreach ($f in @($rosOut, $rosErr)) { if (Test-Path $f) { Remove-Item $f -Force } }

        $proc = Start-Process -FilePath $dotnet.Source -ArgumentList @('exec', $cscDll) + $cscArgs -NoNewWindow -PassThru -RedirectStandardOutput $rosOut -RedirectStandardError $rosErr -Wait

        if ($proc.ExitCode -eq 0 -and (Test-Path $LauncherOut)) {
            $launcherSize = [math]::Round((Get-Item $LauncherOut).Length / 1KB, 1)
            Write-Info "Launcher compiled successfully via dotnet Roslyn (${launcherSize} KB)"
            return $true
        } else {
            Write-Err "Roslyn compilation failed with exit code: $($proc.ExitCode)"
            return $false
        }
    } catch {
        Write-Err "Roslyn compilation exception: $($_.Exception.Message)"
        return $false
    }
}

# Try CSC first, fallback to Add-Type
$launcherCompiled = $false

if (Invoke-CompileLauncherWithCSC) {
    $launcherCompiled = $true
} elseif (Invoke-CompileLauncherWithDotnetRoslyn) {
    $launcherCompiled = $true
} elseif (Invoke-CompileLauncherWithAddType) {
    $launcherCompiled = $true
} else {
    Write-Err 'Failed to compile launcher with both CSC and Add-Type methods.'
    Write-Err 'Please install .NET SDK or Visual Studio Build Tools.'
    Write-Err 'Alternative: Install via chocolatey -> choco install dotnet-sdk -y'
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

# Final validation
if (-not (Test-Path $StagedCore)) {
    Write-Err "Build validation failed: SPQ_core.exe missing"
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

if (-not (Test-Path $LauncherOut)) {
    Write-Err "Build validation failed: SPQ.exe launcher missing"
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

# Success summary
$totalSize = [math]::Round(((Get-ChildItem $DistDir -Recurse -File | Measure-Object -Property Length -Sum).Sum / 1MB), 1)

Write-Info '========================================='
Write-Info 'BUILD COMPLETED SUCCESSFULLY'
Write-Info '========================================='
Write-Info "Output directory: ${DistDir}"
Write-Info "Total size: ${totalSize} MB"
Write-Info "Main executable: SPQ.exe"
Write-Info "Core application: data/bin/SPQ_core.exe"
Write-Info ''
Write-Info 'Usage:'
Write-Info '  SPQ.exe          -> Run application'
Write-Info '  SPQ.exe --debug  -> Run with debug console'
Write-Info '========================================='

if ($TranscriptStarted) { 
    try { Stop-Transcript | Out-Null } catch {} 
}

exit 0