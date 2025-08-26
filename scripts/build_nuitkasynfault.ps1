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
        Write-Info "Standard removal failed for $Path, trying advanced cleanup..."
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
        
        Write-Info "Renamed locked path to: $alt (scheduled for background deletion)"
        return
    } catch {}
    
    Write-Err "Failed to clean path: $Path (locked files detected). Build may fail."
}

# Paths
$ScriptDir   = Split-Path -Parent $PSCommandPath
$ProjectRoot = Split-Path -Parent $ScriptDir
$ReleaseRoot = Join-Path $ProjectRoot 'release'
$WorkRoot    = Join-Path $ReleaseRoot 'work'
$DistRoot    = Join-Path $ReleaseRoot 'dist'
$DistDir     = $DistRoot
$BinDir      = Join-Path $DistDir 'data/bin'
$DataDir     = Join-Path $DistDir 'data'
$LogsDir     = Join-Path $DistDir 'logs'
$LogPath     = Join-Path $ReleaseRoot 'build.log'

# Validate project structure
if (-not (Test-Path (Join-Path $ProjectRoot 'main.py'))) {
    Write-Err "main.py not found in project root: $ProjectRoot"
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
    Write-Info "Build log: $LogPath"
} catch {
    Write-Err "Could not start transcript: $($_.Exception.Message)"
    Write-Info "Continuing without logging..."
}

Write-Info "Project root: $ProjectRoot"
Write-Info "Dist dir    : $DistDir"
Write-Info "PowerShell version: $($PSVersionTable.PSVersion)"

# Clean with better feedback
if ($Clean) {
    Write-Info 'Cleaning previous build artifacts...'
    $cleanPaths = @($WorkRoot, $DistDir)
    foreach ($p in $cleanPaths) { 
        if (Test-Path $p) { 
            Write-Info "Removing: $p"
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
        throw "Python executable not found at: $PythonExe"
    }
    
    $PythonVersion = & "$PythonExe" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')" 2>$null
    Write-Info "Python: $PythonVersion at $PythonExe"
} catch {
    Write-Err "Python not found or invalid. Error: $($_.Exception.Message)"
    Write-Err "Please ensure Python is installed and available on PATH."
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

# Validate Nuitka with better error reporting
try {
    $NuitkaTest = & "$PythonExe" -c "import nuitka; print(nuitka.__version__)" 2>$null
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($NuitkaTest)) {
        throw "Nuitka import failed or version not accessible"
    }
    Write-Info "Nuitka: $($NuitkaTest.Trim())"
} catch {
    Write-Err "Nuitka not found or invalid installation."
    Write-Err "Please install: python -m pip install nuitka ordered-set zstandard"
    Write-Err "Error details: $($_.Exception.Message)"
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

# Validate required dependencies
$RequiredModules = @('PySide6', 'psutil', 'pywin32')
foreach ($module in $RequiredModules) {
    try {
        $null = & "$PythonExe" -c "import $module" 2>$null
        if ($LASTEXITCODE -ne 0) {
            throw "Module import failed"
        }
    } catch {
        Write-Err "Required module '$module' not found. Please install your requirements.txt"
        if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
        exit 1
    }
}
Write-Info "Required Python modules validated"

# Verify precompiled Qt resources
$RcPyPath = Join-Path $ProjectRoot 'ui/resources_rc.py'
if (Test-Path $RcPyPath) {
    Write-Info "Using precompiled Qt resources: $RcPyPath"
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
    Write-Info "Using icon: $IconPath"
}

$nuArgs += (Join-Path $ProjectRoot 'main.py')

Write-Info "Nuitka command: $PythonExe $($nuArgs -join ' ')"

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
    Write-Err "Check logs: $nuOut and $nuErr"
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit $proc.ExitCode
}

# Locate and validate Nuitka output
Write-Info "Locating Nuitka output in $WorkOut..."

$distCandidates = @()
if (Test-Path $WorkOut) {
    $distCandidates = Get-ChildItem -Directory -Path $WorkOut -Filter '*.dist' | Where-Object { 
        Test-Path (Join-Path $_.FullName 'SPQ_core.exe') 
    }
}

# Fallback to common default
if ($distCandidates.Count -eq 0) {
    $fallback = Join-Path $WorkOut 'main.dist'
    if (Test-Path $fallback -and (Test-Path (Join-Path $fallback 'SPQ_core.exe'))) {
        $distCandidates = @(Get-Item $fallback)
    }
}

if ($distCandidates.Count -eq 0) {
    Write-Err "Nuitka output not found!"
    Write-Err "Expected: *.dist directory containing SPQ_core.exe under $WorkOut"
    
    # Debug info
    if (Test-Path $WorkOut) {
        $items = Get-ChildItem $WorkOut
        Write-Info "Contents of $WorkOut:"
        $items | ForEach-Object { Write-Info "  $($_.Name) ($($_.GetType().Name))" }
    }
    
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

$CoreDistPath = $distCandidates[0].FullName
Write-Info "Found Nuitka output: $CoreDistPath"

# Stage the portable layout
Write-Info "Staging portable layout to $DistDir..."

# Create directory structure
foreach ($dir in @($DistDir, $BinDir, $DataDir)) {
    if (!(Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# Copy Nuitka payload to data/bin
Write-Info "Copying Nuitka payload from $CoreDistPath to $BinDir..."
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
    Write-Err "Staging failed: SPQ_core.exe not found at $StagedCore"
    if ($TranscriptStarted) { try { Stop-Transcript | Out-Null } catch {} }
    exit 1
}

$coreSize = [math]::Round((Get-Item $StagedCore).Length / 1MB, 1)
Write-Info "Staged SPQ_core.exe ($coreSize MB)"

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
                Write-Info "Removed empty directory: $dir"
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
    "Python: $PythonVersion",
    "Nuitka: $($NuitkaTest.Trim())"
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
            string baseDir = AppContext.BaseDirectory;
            string binDir = Path.Combine(baseDir, "data", "bin");
            string coreExe = Path.Combine(binDir, "SPQ_core.exe");

            // Validate core application
            if (!Directory.Exists(binDir) || !File.Exists(coreExe))
            {
                string msg = $"SPQ core application not found.\n\nExpected: {coreExe}\n\nEnsure the complete application folder structure is present.";
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
                string stdoutLog = Path.Combine(logsDir, $"stdout_{timestamp}.log");
                string stderrLog = Path.Combine(logsDir, $"stderr_{timestamp}.log");

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
                            string logEntry = $"[{DateTime.UtcNow:yyyy-MM-dd HH:mm:ss}] Process exited: code={exitCode}, runtime={runtime.TotalSeconds:F1}s\n";
                            File.AppendAllText(launcherLog, logEntry);
                        }
                        catch { }

                        // Show warning for quick exits
                        if (runtime.TotalSeconds < 3 && exitCode != 0)
                        {
                            string msg = $"SPQ core exited quickly with error code {exitCode}.\n\nRuntime: {runtime.TotalSeconds:F1} seconds\n\nCheck the log files in the logs/ directory for details.";
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
                    MessageBox.Show($"Logging error: {ex.Message}", "SPQ - Debug", MessageBoxButtons.OK, MessageBoxIcon.Information);
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
            string msg = $"SPQ Launcher Error:\n\n{ex.Message}\n\nStack Trace:\n{ex.StackTrace}";
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
        
        Write-Info "Compiling launcher with csc.exe: $cscPath"
        
        # Build arguments array
        $args = @(
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
            $args += "/win32icon:$IconPath"
            Write-Info "Including icon: $IconPath"
        }
        
        Write-Info "CSC Arguments: $($args -join ' ')"
        
        # Compile with timeout
        $proc = Start-Process -FilePath $cscPath -ArgumentList $args -NoNewWindow -PassThru -Wait
        
        if ($proc.ExitCode -eq 0 -and (Test-Path $LauncherOut)) {
            $launcherSize = [math]::Round((Get-Item $LauncherOut).Length / 1KB, 1)
            Write-Info "Launcher compiled successfully ($launcherSize KB)"
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
            'System.Windows.Forms.dll',
            'System.Runtime.dll',
            'System.dll'
        )
        
        Add-Type -OutputType WindowsApplication -CompilerOptions $compilerOptions -OutputAssembly $LauncherOut -TypeDefinition (Get-Content $LauncherSrcPath -Raw) -ReferencedAssemblies $referencedAssemblies
        
        if (Test-Path $LauncherOut) {
            $launcherSize = [math]::Round((Get-Item $LauncherOut).Length / 1KB, 1)
            Write-Info "Launcher compiled with Add-Type ($launcherSize KB)"
            return $true
        }
        
        return $false
    } catch {
        Write-Err "Add-Type compilation failed: $($_.Exception.Message)"
        return $false
    }
}

# Try CSC first, fallback to Add-Type
$launcherCompiled = $false

if (Invoke-CompileLauncherWithCSC) {
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
Write-Info "Output directory: $DistDir"
Write-Info "Total size: $totalSize MB"
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