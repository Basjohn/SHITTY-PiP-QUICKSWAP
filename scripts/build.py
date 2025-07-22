#!/usr/bin/env python3
"""
Enhanced Build script for SPQ (Shitty PiP QuickSwap) application.
Now with icon support and performance optimizations.
"""

import os
import shutil
import sys
import subprocess
from pathlib import Path
import ast
import importlib.util


def find_icon_file():
    """Find icon file in common locations."""
    project_root = Path(__file__).parent.parent
    
    # Common icon locations and formats
    icon_paths = [
        project_root / 'icon.ico',
        project_root / 'assets' / 'icon.ico',
        project_root / 'icons' / 'icon.ico',
        project_root / 'resources' / 'icon.ico',
        project_root / 'spq.ico',
        project_root / 'SPQ.ico',
        # PNG icons (PyInstaller can convert)
        project_root / 'icon.png',
        project_root / 'assets' / 'icon.png',
        project_root / 'icons' / 'icon.png',
        project_root / 'spq.png',
        project_root / 'SPQ.png',
    ]
    
    for icon_path in icon_paths:
        if icon_path.exists():
            print(f"Found icon: {icon_path}")
            return icon_path
    
    print("No icon found. The executable will use the default Python icon.")
    print("To add an icon, place 'icon.ico' or 'icon.png' in the project root.")
    return None


def create_version_info_file():
    """Create a version info file for Windows exe properties."""
    version_info_content = '''# UTF-8
#
# For more details about fixed file info 'ffi' see:
# http://msdn.microsoft.com/en-us/library/ms646997.aspx
VSVersionInfo(
  ffi=FixedFileInfo(
    filevers=(1,0,0,0),
    prodvers=(1,0,0,0),
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
        [StringStruct(u'CompanyName', u''),
        StringStruct(u'FileDescription', u'SPQ - Quick PiP Window Manager'),
        StringStruct(u'FileVersion', u'1.0.0.0'),
        StringStruct(u'InternalName', u'SPQ'),
        StringStruct(u'LegalCopyright', u''),
        StringStruct(u'OriginalFilename', u'SPQ.exe'),
        StringStruct(u'ProductName', u'SPQ'),
        StringStruct(u'ProductVersion', u'1.0.0.0')])
      ]), 
    VarFileInfo([VarStruct(u'Translation', [1033, 1200])])
  ]
)'''
    
    version_file = Path('version_info.txt')
    with open(version_file, 'w', encoding='utf-8') as f:
        f.write(version_info_content)
    
    print(f"Created version info file: {version_file}")
    return version_file


def run_production_build(cmd, project_root):
    """Execute the production build command."""
    try:
        print(f"Running production build: {' '.join(cmd)}")
        
        # Check if executable already exists and is running
        exe_name = 'SPQ.exe' if os.name == 'nt' else 'SPQ'
        output_file = project_root / 'dist' / exe_name
        
        if output_file.exists():
            print(f"Existing executable found: {output_file}")
            if os.name == 'nt':
                # Try to kill any running instances
                try:
                    subprocess.run(['taskkill', '/F', '/IM', exe_name], 
                                  capture_output=True, check=False)
                    print("Killed running instances")
                    import time
                    time.sleep(1)  # Give it a moment
                except:
                    pass
        
        # Optimize environment for build performance
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = '1'
        env['PYTHONOPTIMIZE'] = '1'  # Use -O mode instead of -OO (avoids pycparser issues)
        env['PYTHONDONTWRITEBYTECODE'] = '1'  # Don't create .pyc files during build
        
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, env=env)
        
        if result.stdout:
            print("\n=== Build Output ===")
            print(result.stdout)
        if result.stderr:
            print("\n=== Build Warnings ===")
            print(result.stderr)
        
        exe_name = 'SPQ.exe' if os.name == 'nt' else 'SPQ'
        output_file = project_root / 'dist' / exe_name
        
        if output_file.exists():
            print(f"\nSUCCESS: Production build successful!")
            print(f"Executable: {output_file}")
            print(f"Size: {output_file.stat().st_size / 1024 / 1024:.1f} MB")
            
            # Test the executable
            print("\nTesting the executable...")
            try:
                test_result = subprocess.run([str(output_file), '--version'], 
                                           capture_output=True, text=True, timeout=10)
                print("SUCCESS: Executable runs without crashing")
                return True
            except subprocess.TimeoutExpired:
                print("WARNING: Executable runs but didn't respond to --version quickly")
                return True
            except Exception as e:
                print(f"ERROR: Executable test failed: {e}")
                print("But the build itself succeeded - try running the exe manually")
                return True
        else:
            print("ERROR: Build completed but no executable found")
            return False
            
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Build failed with error code {e.returncode}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False


def clean_build():
    """Remove previous build and dist directories."""
    print("Cleaning previous builds...")
    
    # Force kill any running SPQ processes first
    if os.name == 'nt':  # Windows
        try:
            subprocess.run(['taskkill', '/F', '/IM', 'SPQ.exe'], 
                          capture_output=True, check=False)
            print("Killed any running SPQ.exe processes")
        except:
            pass
    
    # Clean build directories with more aggressive approach
    import time
    for dir_name in ['build', 'dist', '__pycache__']:
        if os.path.exists(dir_name):
            for attempt in range(3):
                try:
                    shutil.rmtree(dir_name, ignore_errors=True)
                    break
                except PermissionError:
                    print(f"Attempt {attempt + 1}: Waiting for {dir_name} to be released...")
                    time.sleep(2)
    
    # Clean temporary files
    for pattern in ['*.pyc', '*.pyo', '*.pyd', 'version_info.txt']:
        for file in Path('.').rglob(pattern):
            try:
                file.unlink(missing_ok=True)
            except:
                pass


def find_main_script():
    """Find the main SPQ script to build."""
    project_root = Path(__file__).parent.parent
    
    # Common locations for main script
    possible_mains = [
        project_root / 'spq.py',
        project_root / 'main.py', 
        project_root / 'SPQ.py',
        project_root / 'src' / 'spq.py',
        project_root / 'py' / 'main.py'
    ]
    
    for main_file in possible_mains:
        if main_file.exists():
            print(f"Found main script: {main_file}")
            return main_file
    
    print("Could not find main script. Please specify the path.")
    return None


def analyze_all_imports():
    """Thoroughly analyze all Python files to find where transformers might be imported."""
    print("Analyzing ALL Python files for problematic imports...")
    
    project_root = Path(__file__).parent.parent
    python_files = []
    
    # Find all Python files
    for root, dirs, files in os.walk(project_root):
        # Skip build directories
        if any(skip in root for skip in ['build', 'dist', '__pycache__', '.git', 'venv', 'env']):
            continue
            
        for file in files:
            if file.endswith('.py'):
                python_files.append(Path(root) / file)
    
    print(f"Found {len(python_files)} Python files to analyze...")
    
    problematic_modules = ['transformers', 'torch', 'tensorflow', 'sklearn', 'scipy']
    files_with_issues = {}
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
            
            # Parse the AST to find all imports
            try:
                tree = ast.parse(content)
                imports_found = []
                
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            imports_found.append(alias.name.split('.')[0])
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            imports_found.append(node.module.split('.')[0])
                
                # Check for problematic imports
                problematic_found = [imp for imp in imports_found if imp in problematic_modules]
                if problematic_found:
                    files_with_issues[str(py_file.relative_to(project_root))] = problematic_found
                    
            except SyntaxError:
                # If AST parsing fails, fall back to regex
                import re
                import_pattern = r'^\s*(?:from\s+(\S+)\s+import|import\s+(\S+))'
                imports = re.findall(import_pattern, content, re.MULTILINE)
                
                for from_module, import_module in imports:
                    module = (from_module or import_module).split('.')[0]
                    if module in problematic_modules:
                        if str(py_file.relative_to(project_root)) not in files_with_issues:
                            files_with_issues[str(py_file.relative_to(project_root))] = []
                        files_with_issues[str(py_file.relative_to(project_root))].append(module)
                        
        except Exception as e:
            print(f"Warning: Could not analyze {py_file}: {e}")
    
    if files_with_issues:
        print("\nFOUND PROBLEMATIC IMPORTS:")
        for file, modules in files_with_issues.items():
            print(f"  FILE: {file}: {', '.join(set(modules))}")
        print("\nThese imports are causing PyInstaller to try to include heavy ML libraries.")
    else:
        print("\nSUCCESS: No direct problematic imports found in your code.")
        print("The issue might be from an indirect dependency.")
    
    return files_with_issues


def check_installed_packages():
    """Check what packages are actually installed that might be causing issues."""
    print("\nChecking installed packages...")
    
    try:
        result = subprocess.run([sys.executable, '-m', 'pip', 'list'], 
                              capture_output=True, text=True, check=True)
        lines = result.stdout.split('\n')[2:]  # Skip header lines
        
        problematic_packages = []
        for line in lines:
            if line.strip():
                package_name = line.split()[0].lower()
                if any(prob in package_name for prob in ['transform', 'torch', 'tensorflow', 'sklearn', 'scipy']):
                    problematic_packages.append(line.strip())
        
        if problematic_packages:
            print("\nPOTENTIALLY PROBLEMATIC PACKAGES INSTALLED:")
            for package in problematic_packages:
                print(f"  PACKAGE: {package}")
            print("\nThese packages might have hooks that PyInstaller is trying to process.")
        else:
            print("SUCCESS: No obviously problematic packages found.")
            
    except Exception as e:
        print(f"Could not check installed packages: {e}")


def build_executable_production():
    """Build with the packages your app actually needs, with performance and icon optimizations."""
    print("Building optimized production version with icon support...")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    main_script = find_main_script()
    if not main_script:
        return False
    
    # Look for icon file
    icon_file = find_icon_file()
    
    # Create version info for Windows
    version_file = None
    if os.name == 'nt':
        version_file = create_version_info_file()
    
    # Build command with performance optimizations
    cmd = [
        'pyinstaller',
        '--onefile',
        '--windowed',  # No console window for GUI app
        '--clean',
        '--workpath=build',
        '--distpath=dist',
        '--name=SPQ',
        '--noconfirm',
        
        # PERFORMANCE OPTIMIZATIONS
        '--noupx',       # Disable UPX compression (can be slower to start)
        
        # EXCLUDE HEAVY ML PACKAGES
        '--exclude-module=transformers',
        '--exclude-module=sentence_transformers',
        '--exclude-module=torch',
        '--exclude-module=tensorflow', 
        '--exclude-module=tf',
        '--exclude-module=sklearn',
        '--exclude-module=huggingface_hub',
        '--exclude-module=tokenizers',
        '--exclude-module=safetensors',
        '--exclude-module=matplotlib.pyplot',
        '--exclude-module=PIL',
        '--exclude-module=cv2',
        '--exclude-module=IPython',
        '--exclude-module=jupyter',
        '--exclude-module=pytest',
        '--exclude-module=setuptools',
        '--exclude-module=pip',
        
        # INCLUDE WHAT YOUR APP NEEDS
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=tkinter.messagebox',
        '--hidden-import=tkinter.filedialog',
        '--hidden-import=numpy',
        '--hidden-import=numpy.core',
        '--hidden-import=numpy.core._methods',
        '--hidden-import=numpy.lib.format',
        '--collect-submodules=numpy',
        '--hidden-import=PySide6.QtCore',
        '--hidden-import=PySide6.QtGui', 
        '--hidden-import=PySide6.QtWidgets',
        '--collect-submodules=PySide6',
        
        # WINDOWS-SPECIFIC
        '--hidden-import=win32gui',
        '--hidden-import=win32con',
        '--hidden-import=win32process',
        '--hidden-import=win32api',
        '--hidden-import=win32event',
        '--hidden-import=win32com',
        '--hidden-import=keyboard',
    ]
    
    # Add icon if found
    if icon_file:
        cmd.extend(['--icon', str(icon_file)])
        print(f"Using icon: {icon_file}")
    
    # Add version info for Windows
    if version_file and os.name == 'nt':
        cmd.extend(['--version-file', str(version_file)])
        print(f"Using version info: {version_file}")
    
    # Add main script
    cmd.append(str(main_script))
    
    success = run_production_build(cmd, project_root)
    
    # Clean up temporary files
    if version_file and version_file.exists():
        try:
            version_file.unlink()
        except:
            pass
    
    return success


def build_executable_minimal():
    """Build with minimal includes to avoid the transformers issue."""
    print("Building with minimal configuration to avoid transformers issue...")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    main_script = find_main_script()
    if not main_script:
        return False
    
    # Look for icon file
    icon_file = find_icon_file()
    
    # Very conservative build approach - avoid the problematic ML packages
    cmd = [
        'pyinstaller',
        '--onefile',
        '--windowed',  # No console window
        '--clean',
        '--workpath=build',
        '--distpath=dist',
        '--name=SPQ',
        '--noconfirm',
        '--noupx',       # Avoid compression issues
        
        # Exclude the ML packages that are causing issues
        '--exclude-module=transformers',
        '--exclude-module=torch',
        '--exclude-module=sentence_transformers',
        '--exclude-module=tensorflow', 
        '--exclude-module=tf',
        '--exclude-module=sklearn',
        '--exclude-module=scipy',
        '--exclude-module=pandas',
        '--exclude-module=matplotlib',
        '--exclude-module=PIL',
        '--exclude-module=cv2',
        '--exclude-module=IPython',
        '--exclude-module=jupyter',
        '--exclude-module=huggingface_hub',
        '--exclude-module=tokenizers',
        '--exclude-module=safetensors',
        
        # Include numpy since your app needs it
        '--hidden-import=numpy',
        '--hidden-import=numpy.core',
        '--hidden-import=numpy.core._methods',
        '--hidden-import=numpy.lib.format',
        '--collect-submodules=numpy',
        
        # Only include what's absolutely necessary
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=tkinter.messagebox',
        '--hidden-import=tkinter.filedialog',
    ]
    
    # Add icon if found
    if icon_file:
        cmd.extend(['--icon', str(icon_file)])
    
    cmd.append(str(main_script))
    
    return run_production_build(cmd, project_root)


def create_custom_spec():
    """Create a custom .spec file to have more control over the build."""
    print("Creating custom .spec file with icon and performance optimizations...")
    
    main_script = find_main_script()
    if not main_script:
        return None
    
    icon_file = find_icon_file()
    icon_param = f"icon='{icon_file}'," if icon_file else ""
    
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

# Custom spec file for SPQ with performance and icon optimizations

block_cipher = None

# Explicitly define what to include/exclude
a = Analysis(
    ['{main_script}'],
    pathex=[],
    binaries=[],
    datas=[],
    hiddenimports=[
        'tkinter',
        'tkinter.ttk', 
        'tkinter.messagebox',
        'tkinter.filedialog',
        'numpy',
        'numpy.core',
        'numpy.core._methods',
        'numpy.lib.format',
        'PySide6.QtCore',
        'PySide6.QtGui',
        'PySide6.QtWidgets',
        'win32gui',
        'win32con',
        'win32process',
        'win32api',
        'keyboard',
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        # Heavy ML packages
        'transformers',
        'torch',
        'tensorflow',
        'tf',
        'sklearn',
        'scipy',
        'sentence_transformers',
        'huggingface_hub',
        'tokenizers',
        'safetensors',
        # Other heavy packages
        'pandas',
        'matplotlib',
        'PIL',
        'cv2',
        'IPython',
        'jupyter',
        'notebook',
        'pytest',
        'setuptools',
        'pip',
        'wheel',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Remove duplicate and unnecessary files to reduce size
a.datas = [x for x in a.datas if not x[0].startswith('tcl')]
a.datas = [x for x in a.datas if not x[0].startswith('tk')]

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
    strip=True,  # Strip debug symbols for smaller size
    upx=False,   # Disable UPX compression
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app without console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    {icon_param}
)
'''
    
    spec_file = Path('SPQ_optimized.spec')
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    
    print(f"Created optimized spec file: {spec_file}")
    if icon_file:
        print(f"Icon included: {icon_file}")
    return spec_file


def main():
    """Main entry point with full diagnostics."""
    try:
        print("SPQ Enhanced Build with Performance & Icon Optimizations")
        print("=" * 60)
        
        # Step 1: Analyze all imports
        problematic_files = analyze_all_imports()
        
        # Step 2: Check installed packages
        check_installed_packages()
        
        # Step 3: Ask user what to do
        if problematic_files:
            print("\nWhat would you like to do?")
            print("1. Try optimized build anyway (includes icon support)")
            print("2. Try minimal build (includes icon support)")
            print("3. Create optimized .spec file for manual editing")
            print("4. Show me which files have the problematic imports")
            print("5. Exit and let me fix the imports first")
            
            choice = input("\nEnter choice (1-5): ").strip()
            
            if choice == "4":
                print("\nFiles with problematic imports:")
                for file, modules in problematic_files.items():
                    print(f"\nFILE: {file}:")
                    try:
                        with open(file, 'r', encoding='utf-8') as f:
                            lines = f.readlines()
                        
                        for i, line in enumerate(lines, 1):
                            if any(f'import {mod}' in line or f'from {mod}' in line 
                                  for mod in modules):
                                print(f"  Line {i}: {line.strip()}")
                    except Exception as e:
                        print(f"  Could not read file: {e}")
                return 0
                
            elif choice == "5":
                print("\nGood choice! Fix those imports first, then run the build again.")
                return 0
                
            elif choice == "3":
                spec_file = create_custom_spec()
                if spec_file:
                    print(f"\nSUCCESS: Created {spec_file}")
                    print("You can now edit it and run: pyinstaller SPQ_optimized.spec")
                return 0
                
            elif choice == "2":
                clean_build()
                success = build_executable_minimal()
            else:  # Default to choice 1
                clean_build()
                success = build_executable_production()
        else:
            # No problematic files, go straight to optimized build
            clean_build()
            success = build_executable_production()
        
        if success:
            print("\nBUILD COMPLETED SUCCESSFULLY!")
            print("\nPerformance Tips for Runtime:")
            print("- The executable uses Python -O2 optimization")
            print("- Debug symbols are stripped for smaller size")
            print("- UPX compression is disabled for faster startup")
            if find_icon_file():
                print("- Icon should be visible in File Explorer")
        else:
            print("\nSuggestions:")
            print("1. Check the files with problematic imports shown above")
            print("2. Remove or comment out unused imports")  
            print("3. Use a virtual environment with only needed packages")
            print("4. Try the custom .spec file approach")
        
        return 0 if success else 1
        
    except KeyboardInterrupt:
        print("\nBuild interrupted by user.")
        return 1
    except Exception as e:
        print(f"\nERROR: Error during build: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())