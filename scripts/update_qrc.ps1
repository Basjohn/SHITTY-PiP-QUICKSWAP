param(
    [switch]$DryRun,
    [switch]$NoCompile,
    [string]$QrcPath,
    [string]$ResourcesDir,
    [string]$ThemesDir,
    [string]$PyOutPath
)

$ErrorActionPreference = 'Stop'
Set-StrictMode -Version Latest

function Resolve-ProjectPaths {
    param()
    $root = Resolve-Path (Join-Path $PSScriptRoot '..')

    $global:ProjectRoot   = $root
    $global:ResolvedQrc   = if ($QrcPath) { Resolve-Path $QrcPath } else { Join-Path $ProjectRoot 'resources.qrc' }
    $global:ResolvedRes   = if ($ResourcesDir) { Resolve-Path $ResourcesDir } else { Join-Path $ProjectRoot 'resources' }
    $global:ResolvedThemes= if ($ThemesDir) { Resolve-Path $ThemesDir } else { Join-Path $ProjectRoot 'themes' }
    $global:ResolvedPyOut = if ($PyOutPath) { $PyOutPath } else { Join-Path $ProjectRoot 'ui\resources_rc.py' }
}

function Get-RelativePathUnix {
    param(
        [Parameter(Mandatory)] [string]$Base,
        [Parameter(Mandatory)] [string]$FullPath
    )
    $rel = [System.IO.Path]::GetRelativePath($Base, $FullPath)
    return ($rel -replace '\\','/')
}

function Get-QrcResources {
    param()
    if (-not (Test-Path $ResolvedRes)) { throw "Resources directory not found: $ResolvedRes" }
    if (-not (Test-Path $ResolvedThemes)) { throw "Themes directory not found: $ResolvedThemes" }

    $iconExts = @('.png','.ico','.jpg','.jpeg','.svg','.webp')
    $iconFiles = Get-ChildItem -Path $ResolvedRes -Recurse -File | Where-Object { $iconExts -contains $_.Extension.ToLowerInvariant() }

    # Badges: BadgeN.png at any depth under resources
    $badgeRegex = '^(?i)Badge\d+$'
    $badges = $iconFiles | Where-Object { $_.BaseName -match $badgeRegex -and ($_.Extension -ieq '.png') }

    # Other icons/images excluding badges
    $badgeSet = [System.Collections.Generic.HashSet[string]]::new([StringComparer]::OrdinalIgnoreCase)
    $badges | ForEach-Object { [void]$badgeSet.Add($_.FullName) }
    $icons = $iconFiles | Where-Object { -not $badgeSet.Contains($_.FullName) }

    $themes = @()
    if (Test-Path $ResolvedThemes) {
        $themes = Get-ChildItem -Path $ResolvedThemes -File -Filter '*.qss'
    }

    return [pscustomobject]@{
        Icons  = $icons  | Sort-Object FullName
        Badges = $badges | Sort-Object FullName
        Themes = $themes | Sort-Object FullName
    }
}

function New-QrcContent {
    param(
        [Parameter(Mandatory)] $Collected
    )

    $nl = "`n"
    $sb = [System.Text.StringBuilder]::new()
    [void]$sb.Append('<RCC>')
    [void]$sb.Append($nl)

    # /icons
    [void]$sb.Append('  <qresource prefix="/icons">')
    [void]$sb.Append($nl)
    foreach ($f in $Collected.Icons) {
        $rel = Get-RelativePathUnix -Base $ProjectRoot -FullPath $f.FullName
        # we want paths like resources/...
        if (-not $rel.StartsWith('resources/')) { $rel = 'resources/' + (Get-RelativePathUnix -Base $ResolvedRes -FullPath $f.FullName) }
        $alias = $f.Name
        [void]$sb.AppendFormat('    <file alias="{0}">{1}</file>{2}', $alias, $rel, $nl)
    }
    [void]$sb.Append('  </qresource>')
    [void]$sb.Append($nl)

    # /badges
    if ($Collected.Badges.Count -gt 0) {
        [void]$sb.Append('  <qresource prefix="/badges">')
        [void]$sb.Append($nl)
        foreach ($f in $Collected.Badges) {
            $rel = Get-RelativePathUnix -Base $ProjectRoot -FullPath $f.FullName
            if (-not $rel.StartsWith('resources/')) { $rel = 'resources/' + (Get-RelativePathUnix -Base $ResolvedRes -FullPath $f.FullName) }
            $alias = $f.Name
            [void]$sb.AppendFormat('    <file alias="{0}">{1}</file>{2}', $alias, $rel, $nl)
        }
        [void]$sb.Append('  </qresource>')
        [void]$sb.Append($nl)
    }

    # /themes
    if ($Collected.Themes.Count -gt 0) {
        [void]$sb.Append('  <qresource prefix="/themes">')
        [void]$sb.Append($nl)
        foreach ($f in $Collected.Themes) {
            $rel = Get-RelativePathUnix -Base $ProjectRoot -FullPath $f.FullName
            if (-not $rel.StartsWith('themes/')) { $rel = 'themes/' + $f.Name }
            $alias = $f.Name
            [void]$sb.AppendFormat('    <file alias="{0}">{1}</file>{2}', $alias, $rel, $nl)
        }
        [void]$sb.Append('  </qresource>')
        [void]$sb.Append($nl)
    }

    [void]$sb.Append('</RCC>')
    [void]$sb.Append($nl)

    return $sb.ToString()
}

function Get-RccCompiler {
    # Align with build_nuitka.ps1: require precompiled resources via PySide6
    try {
        $cmd = Get-Command 'pyside6-rcc' -ErrorAction Stop
        if ($null -ne $cmd) { return $cmd.Name }
    } catch { }
    return $null
}

function Write-QrcAndMaybeCompile {
    Resolve-ProjectPaths
    Write-Host "[QRC] ProjectRoot: $ProjectRoot"
    Write-Host "[QRC] Resources:   $ResolvedRes"
    Write-Host "[QRC] Themes:      $ResolvedThemes"
    Write-Host "[QRC] QRC Path:    $ResolvedQrc"
    Write-Host "[QRC] Py Out:      $ResolvedPyOut"

    $collected = Get-QrcResources
    Write-Host ("[QRC] Files → Icons={0}, Badges={1}, Themes={2}" -f $collected.Icons.Count, $collected.Badges.Count, $collected.Themes.Count)

    $content = New-QrcContent -Collected $collected

    if ($DryRun) {
        Write-Host "[QRC] DryRun enabled — not writing files. Showing preview of first 40 lines:" -ForegroundColor Yellow
        $content -split "`n" | Select-Object -First 40 | ForEach-Object { Write-Host $_ }
        return
    }

    # Backup existing QRC if present
    if (Test-Path $ResolvedQrc) {
        $bak = "$ResolvedQrc.bak"
        Copy-Item -Path $ResolvedQrc -Destination $bak -Force
        Write-Host "[QRC] Backed up existing QRC → $bak"
    }

    # Write QRC
    $utf8NoBom = New-Object System.Text.UTF8Encoding($false)
    [System.IO.File]::WriteAllText($ResolvedQrc, $content, $utf8NoBom)
    Write-Host "[QRC] Wrote: $ResolvedQrc"

    if ($NoCompile) {
        Write-Host "[QRC] Skipping Python resource compilation (NoCompile)."
        return
    }

    $compiler = Get-RccCompiler
    if (-not $compiler) {
        Write-Warning "[QRC] pyside6-rcc not found. Install PySide6 and ensure 'pyside6-rcc' is on PATH. Skipping compilation."
        Write-Host   "[QRC] Expected command: pyside6-rcc resources.qrc -o ui/resources_rc.py"
        return
    }

    Write-Host "[QRC] Using compiler: $compiler"

    # Ensure output directory exists
    $pyDir = Split-Path -Parent $ResolvedPyOut
    if (-not (Test-Path $pyDir)) { New-Item -ItemType Directory -Force -Path $pyDir | Out-Null }

    $rcArgs = @('-o', $ResolvedPyOut, $ResolvedQrc)
    & $compiler @rcArgs
    if ($LASTEXITCODE -ne 0) { throw "Resource compilation failed with exit code $LASTEXITCODE" }
    Write-Host "[QRC] Compiled → $ResolvedPyOut"
}

Write-QrcAndMaybeCompile
