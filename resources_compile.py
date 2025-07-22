#!/usr/bin/env python3
"""
Resource Compilation Script for SPQStruggle

This script compiles Qt resource files (.qrc) into Python modules.
It automatically detects and uses the appropriate RCC tool from available Qt/PySide installations.
"""

import subprocess
import sys
import os
import shutil
import glob
from pathlib import Path

# Path configuration
BASE_DIR = Path(__file__).parent.absolute()
QRC_FILE = BASE_DIR / 'resources.qrc'
OUTPUT_PY = BASE_DIR / 'py' / 'resources_rc.py'
RESOURCES_DIR = BASE_DIR / 'Resources'

# Ensure the py directory exists
try:
    (BASE_DIR / 'py').mkdir(exist_ok=True)
except OSError as e:
    print(f"Error creating directory: {e}")
    sys.exit(1)

def find_rcc_tool():
    """
    Find a working RCC tool.
    Tries multiple approaches in order of preference.
    
    Returns:
        list: Command to run the RCC tool, or None if not found
    """
    # List of methods to try, in order of preference
    methods = [
        # Method 1: pyside6-rcc (new standard)
        lambda: ['pyside6-rcc'] if shutil.which('pyside6-rcc') else None,
        
        # Method 2: python -m PySide6.tools.rcc
        lambda: (
            [sys.executable, '-m', 'PySide6.tools.rcc']
            if _check_module_available('PySide6.tools.rcc')
            else None
        ),
        
        # Method 3: pyqt6rc (alternative)
        lambda: ['pyqt6rc'] if shutil.which('pyqt6rc') else None,
        
        # Method 4: Direct pyside6-rcc.exe path (Windows)
        lambda: (
            [str(Path(sys.executable).parent / 'Scripts' / 'pyside6-rcc.exe')]
            if sys.platform == 'win32' and 
               (Path(sys.executable).parent / 'Scripts' / 'pyside6-rcc.exe').exists()
            else None
        ),
        
        # Method 5: pyrcc6 (legacy)
        lambda: ['pyrcc6'] if shutil.which('pyrcc6') else None,
        
        # Method 6: python -m PySide6.scripts.pyrcc6 (very old)
        lambda: (
            [sys.executable, '-m', 'PySide6.scripts.pyrcc6']
            if _check_module_available('PySide6.scripts.pyrcc6')
            else None
        ),
    ]
    
    # Try each method until we find a working one
    for method in methods:
        try:
            cmd = method()
            if cmd and _test_rcc_tool(cmd):
                return cmd
        except Exception as e:
            if '--verbose' in sys.argv:
                print(f"RCC tool check failed: {e}")
    
    # If we get here, no working tool was found
    print("Error: No working RCC tool found.")
    print("Please ensure PySide6 is installed: pip install PySide6")
    return None


def _check_module_available(module_name):
    """Check if a Python module is available."""
    try:
        __import__(module_name)
        return True
    except ImportError:
        return False


def _test_rcc_tool(cmd):
    """Test if an RCC tool works by running it with --help."""
    try:
        result = subprocess.run(
            cmd + ['--help'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError, TimeoutError):
        return False

def verify_resources():
    """Verify that all resources in the .qrc file exist on disk."""
    if not QRC_FILE.exists():
        print(f"Error: {QRC_FILE} not found.")
        return False
    
    # Read the .qrc file
    try:
        with open(QRC_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"Error reading {QRC_FILE}: {e}")
        return False
    
    # Extract file paths from the .qrc file
    import re
    pattern = r'<file[^>]*>([^<]+)</file>'
    files_in_qrc = re.findall(pattern, content)
    
    # Check if all files exist
    missing_files = []
    for file_path in files_in_qrc:
        full_path = BASE_DIR / file_path
        if not full_path.exists():
            missing_files.append(str(full_path))
    
    if missing_files:
        print("Error: The following resource files are missing:")
        for f in missing_files:
            print(f"  - {f}")
        return False
    
    return True


def compile_resources():
    """Compile resources.qrc to resources_rc.py using available RCC tool."""
    if not QRC_FILE.exists():
        print(f"Error: {QRC_FILE} not found.")
        return False
    
    # Verify all resources exist before compiling
    if not verify_resources():
        print("Resource verification failed. Please fix the missing files.")
        return False

    # Ensure output directory exists
    try:
        OUTPUT_PY.parent.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error creating directory: {e}")
        return False

    # Find and verify the RCC tool
    rcc_cmd = find_rcc_tool()
    if rcc_cmd is None:
        return False

    # Build the command
    cmd = rcc_cmd + [str(QRC_FILE), '-o', str(OUTPUT_PY)]
    
    if '--verbose' in sys.argv:
        print(f"Running: {' '.join(cmd)}")
    
    try:
        # Run the RCC tool
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=30  # 30 second timeout
        )
        
        # Print output if in verbose mode
        if '--verbose' in sys.argv:
            if result.stdout:
                print("RCC output:")
                print(result.stdout)
        
        print(f"✓ Successfully compiled {QRC_FILE} to {OUTPUT_PY}")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Error compiling resources (exit code {e.returncode}):")
        if e.stdout:
            print("--- STDOUT ---")
            print(e.stdout)
        if e.stderr:
            print("--- STDERR ---")
            print(e.stderr)
        return False
        
    except FileNotFoundError as e:
        print(f"❌ RCC tool not found: {e}")
        return False
        
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

def print_help():
    """Print help information."""
    print(f"""
Resource Compiler for SPQStruggle

Usage: python {sys.argv[0]} [options]

Options:
  --help, -h     Show this help message
  --verbose, -v  Enable verbose output
""")

def main():
    """Main entry point."""
    # Parse command line arguments
    if '--help' in sys.argv or '-h' in sys.argv:
        print_help()
        return 0
    
    print(f"Compiling resources from: {QRC_FILE}")
    print(f"Output will be written to: {OUTPUT_PY}")
    print()
    
    # Check if PySide6 is installed
    try:
        import PySide6
        success = compile_resources()
    except ImportError:
        print("❌ PySide6 is required but not installed.")
        print("Please install it using: pip install PySide6")
        return 1
    
    if success:
        print("\n✅ Resource compilation completed successfully!")
    else:
        print("\n❌ Resource compilation failed!")
        print("Check the output above for error messages.")
    
    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())