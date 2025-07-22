# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for SPQ (Shitty PiP QuickSwap)
This spec file handles numpy, tkinter, and other common dependencies properly.
"""

import os
import sys
from pathlib import Path

# Get the project root directory
project_root = Path(SPECPATH).parent if 'SPECPATH' in locals() else Path('.')

# Find the main script
main_script = None
possible_mains = [
    project_root / 'spq.py',
    project_root / 'main.py', 
    project_root / 'SPQ.py',
    project_root / 'py' / 'spq.py',
    project_root / 'py' / 'main.py'
]

for script in possible_mains:
    if script.exists():
        main_script = str(script)
        break

if not main_script:
    raise FileNotFoundError("Could not find main SPQ script. Please update the spec file with the correct path.")

print(f"Building with main script: {main_script}")

# Hidden imports - these are modules that PyInstaller often misses
hiddenimports = [
    # Numpy and scientific computing
    'numpy',
    'numpy.core',
    'numpy.core._methods',
    'numpy.core._type_aliases',
    'numpy.core.multiarray',
    'numpy.core.numeric',
    'numpy.core.umath',
    'numpy.lib.format',
    'numpy.linalg',
    'numpy.random',
    'numpy.random._pickle',
    'numpy.ma.core',
    
    # Scipy (if used)
    'scipy',
    'scipy.sparse',
    'scipy.sparse.csgraph',
    'scipy.sparse.csgraph._validation',
    'scipy.spatial.distance',
    'scipy.integrate',
    'scipy.interpolate',
    
    # Standard library modules that sometimes cause issues
    'pkg_resources.py2_warn',
    'pkg_resources.markers',
    
    # GUI frameworks (uncomment what you use)
    'tkinter',
    'tkinter.ttk',
    'tkinter.messagebox',
    'tkinter.filedialog',
    'tkinter.simpledialog',
    'tkinter.colorchooser',
    
    # Common utility modules
    'json',
    'csv',
    'sqlite3',
    'configparser',
    'urllib',
    'urllib.request',
    'urllib.parse',
    'requests',
    'requests.adapters',
    'requests.packages',
    'requests.packages.urllib3',
    'requests.packages.urllib3.util',
    'requests.packages.urllib3.util.retry',
    
    # Date/time handling
    'datetime',
    'time',
    'calendar',
    
    # File/path handling
    'pathlib',
    'glob',
    'shutil',
    
    # Encoding
    'encodings',
    'encodings.utf_8',
    'encodings.cp1252',
    'encodings.ascii',
]

# Data files to include (add your resource files here)
datas = []

# Look for common data files in the project
data_patterns = ['*.txt', '*.json', '*.ini', '*.cfg', '*.yaml', '*.yml', '*.xml']
for pattern in data_patterns:
    for data_file in project_root.glob(pattern):
        if data_file.is_file():
            datas.append((str(data_file), '.'))

# Look for data directories
data_dirs = ['data', 'resources', 'assets', 'config']
for dir_name in data_dirs:
    data_dir = project_root / dir_name
    if data_dir.exists() and data_dir.is_dir():
        datas.append((str(data_dir), dir_name))

# Binary files and libraries to exclude (reduces size)
excludes = [
    # Large unused modules
    'matplotlib',
    'PIL',
    'Pillow',
    'PyQt4', 'PyQt5', 'PyQt6',
    'PySide', 'PySide2', 'PySide6',
    'wx', 'wxPython',
    
    # Development and testing modules
    'pytest',
    'nose',
    'unittest2',
    'doctest',
    'pdb',
    'pydoc',
    
    # Notebook and IPython
    'IPython',
    'jupyter',
    'notebook',
    
    # Documentation generators
    'sphinx',
    'alabaster',
    
    # Unused standard library modules
    'turtle',
    'curses',
    'readline',
    'rlcompleter',
]

# Analysis step - finds all dependencies
a = Analysis(
    [main_script],
    pathex=[str(project_root)],
    binaries=[],
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=excludes,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

# Remove duplicate entries
pyz = PYZ(a.pure, a.zipped_data, cipher=None)

# Create the executable
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='SPQ',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,  # Compress executable (set to False if you have issues)
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # Set to True if you want a console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon=None,  # Add path to .ico file if you have one: 'icon.ico'
)

# Optional: Create a directory distribution instead of a single file
# Uncomment these lines if you prefer a folder distribution:
"""
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='SPQ',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='SPQ'
)
"""