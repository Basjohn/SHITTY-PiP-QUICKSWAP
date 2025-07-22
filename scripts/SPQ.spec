# -*- mode: python ; coding: utf-8 -*-

import os
import sys

# Get the project root directory (where this spec file is located)
# When running in PyInstaller, use _MEIPASS if available, otherwise use the current directory
if getattr(sys, 'frozen', False):
    # If the application is run as a bundle (PyInstaller)
    project_root = sys._MEIPASS
else:
    # If running in a normal Python environment
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Add the project root to the path
sys.path.insert(0, project_root)

block_cipher = None

a = Analysis(
    [os.path.join(project_root, 'py', 'main.py')],
    pathex=[project_root],
    binaries=[],
    datas=[
        (os.path.join(project_root, 'py', 'resources_rc.py'), 'py'),
        (os.path.join(project_root, 'Resources'), 'Resources'),
    ],
    hiddenimports=[
        'numpy',
        'numpy.core._dtype_ctypes',
        'numpy.core._multiarray_umath',
        'numpy.core._multiarray_tests',
        'numpy.core._string_helpers',
        'numpy.core._operand_flag_tests',
        'numpy.core._struct_ufunc_tests',
        'numpy.core._umath_tests',
        'numpy.fft._pocketfft_internal',
        'numpy.linalg.lapack_lite',
        'numpy.random.mtrand',
        'numpy.random._pickle',
        'numpy.random._common',
        'numpy.random._bounded_integers',
        'numpy.random.bit_generator',
        'numpy.random._mt19937',
        'numpy.random._philox',
        'numpy.random._pcg64',
        'numpy.random._sfc64',
        'numpy.random._generator',
        'PySide6.QtSvg',
        'keyboard._winkeyboard',
        'win32timezone',
        'win32clipboard',
        'win32ui',
        'pythoncom',
        'pywintypes',
        'win32api',
        'win32con',
        'win32gui',
        'win32process',
        'win32event',
        'win32security',
        'win32profile',
        'win32ts',
        'win32wnet',
        'winxpgui',
        'dxcam',
        'mss',
        'mss.windows'
    ],
    hookspath=['scripts'],
    runtime_hooks=[],
    excludes=[
        'tkinter',
        'matplotlib',
        'pandas',
        'scipy',
        'PIL',
        'PIL._tkinter_finder',
        'PyQt5',
        'PyQt6',
        'PySide2',
        'pytest',
        'unittest',
        'test',
        'tests',
        'setuptools',
        'distutils',
        'pip',
        'wheel',
        'notebook',
        'jupyter',
        'IPython',
        'ipykernel',
        'ipython_genutils',
        'ipywidgets',
        'jedi',
        'jsonschema',
        'jupyter_client',
        'jupyter_console',
        'jupyter_core',
        'nbconvert',
        'nbformat',
        'notebook',
        'qtpy',
        'sphinx',
        'spyder',
        'traitlets',
        'zmq',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Add any additional files or binaries
# a.binaries += TOC([...])

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

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
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='Resources/ShittyPIP.ico',
)
