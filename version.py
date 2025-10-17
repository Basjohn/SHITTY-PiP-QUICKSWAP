"""
Application version information.

This file is the SINGLE SOURCE OF TRUTH for version numbers.
It is read by:
- Python runtime (import version)
- Build scripts (PowerShell parsing)
- Documentation generators

Format: Semantic Versioning (Major.Minor.Patch)
Alpha/Beta/RC suffixes supported (e.g., "2.1.0a" for alpha)
"""

# Version components
VERSION_MAJOR = 2
VERSION_MINOR = 2
VERSION_PATCH = 0
VERSION_SUFFIX = ""  # Empty string for release, "a" for alpha, "b" for beta, "rc" for release candidate

# Full version string (for Python)
__version__ = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}{VERSION_SUFFIX}"

# Windows version (4-part for PE metadata - suffix not supported in Win32 format)
VERSION_WIN32 = f"{VERSION_MAJOR}.{VERSION_MINOR}.{VERSION_PATCH}.0"

# Build metadata (optional, set by build scripts)
BUILD_DATE = None  # Format: "2025-10-10"
BUILD_TIME = None  # Format: "17:31:45"
BUILD_COMMIT = None  # Git commit hash (if available)

# Application metadata
APP_NAME = "SPQ"
APP_DISPLAY_NAME = "Shitty PiP QUICKSWAP"
APP_COMPANY = "Faecal Failures"
APP_DESCRIPTION = "SPQ"

def get_version_string(include_build=False):
    """Get formatted version string.
    
    Args:
        include_build: Include build metadata if available
        
    Returns:
        Version string (e.g., "2.1.0a" or "2.1.0a+build.20251010")
    """
    version = __version__
    if include_build and BUILD_DATE:
        build_info = BUILD_DATE.replace("-", "")
        if BUILD_TIME:
            build_info += "." + BUILD_TIME.replace(":", "")
        version += f"+build.{build_info}"
    return version

def get_full_version_info():
    """Get complete version information dictionary."""
    return {
        "version": __version__,
        "version_win32": VERSION_WIN32,
        "major": VERSION_MAJOR,
        "minor": VERSION_MINOR,
        "patch": VERSION_PATCH,
        "suffix": VERSION_SUFFIX,
        "build_date": BUILD_DATE,
        "build_time": BUILD_TIME,
        "build_commit": BUILD_COMMIT,
        "app_name": APP_NAME,
        "display_name": APP_DISPLAY_NAME,
        "company": APP_COMPANY,
        "description": APP_DESCRIPTION,
    }
