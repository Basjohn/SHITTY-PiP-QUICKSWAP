#!/usr/bin/env python3
"""
Build script for SPQ (Shitty PiP QuickSwap) application.
Diagnostic version to identify the real source of the transformers import issue.
"""

import os
import shutil
import sys
import subprocess
from pathlib import Path
import ast
import importlib.util


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
        
        env = os.environ.copy()
        env['PYTHONHASHSEED'] = '1'
        
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
    
    # Also clean any .pyc files
    for root, dirs, files in os.walk('.'):
        for file in files:
            if file.endswith('.pyc'):
                try:
                    os.remove(os.path.join(root, file))
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


def build_executable_minimal():
    """Build with minimal includes to avoid the transformers issue."""
    print("Building with minimal configuration to avoid transformers issue...")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    main_script = find_main_script()
    if not main_script:
        return False
    
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
        # Exclude the ML packages that are causing issues
        '--exclude-module=transformers',
        '--exclude-module=torch',
        '--exclude-module=sentence_transformers',
        '--exclude-module=sentence-transformers',
        '--exclude-module=tensorflow', 
        '--exclude-module=tf',
        '--exclude-module=sklearn',
        '--exclude-module=scipy',
        '--exclude-module=pandas',
        # Include numpy since your app needs it
        '--hidden-import=numpy',
        '--hidden-import=numpy.core',
        '--hidden-import=numpy.core._methods',
        '--hidden-import=numpy.lib.format',
        '--collect-submodules=numpy',
        '--exclude-module=matplotlib',
        '--exclude-module=PIL',
        '--exclude-module=cv2',
        '--exclude-module=IPython',
        '--exclude-module=jupyter',
        '--exclude-module=huggingface_hub',
        '--exclude-module=tokenizers',
        '--exclude-module=safetensors',
        # Only include what's absolutely necessary
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=tkinter.messagebox',
        '--hidden-import=tkinter.filedialog',
        # Use noupx to avoid compression issues
        '--noupx',
        str(main_script)
    ]
    
    return run_production_build(cmd, project_root)


def build_executable_production():
    """Build with the packages your app actually needs."""
    print("Building with required packages (including numpy)...")
    
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    
    main_script = find_main_script()
    if not main_script:
        return False
    
    # Build with what you actually need, excluding problematic ML packages
    cmd = [
        'pyinstaller',
        '--onefile',
        '--windowed',  # No console window for GUI app
        '--clean',
        '--workpath=build',
        '--distpath=dist',
        '--name=SPQ',
        '--noconfirm',
        # Exclude the BIG ML packages that cause the dataclasses issue
        '--exclude-module=transformers',
        '--exclude-module=sentence_transformers',
        '--exclude-module=torch',
        '--exclude-module=tensorflow', 
        '--exclude-module=tf',
        '--exclude-module=sklearn',
        '--exclude-module=huggingface_hub',
        '--exclude-module=tokenizers',
        '--exclude-module=safetensors',
        # Exclude other heavy optional packages
        '--exclude-module=matplotlib.pyplot',  # Keep core matplotlib if needed
        '--exclude-module=PIL',
        '--exclude-module=cv2',
        '--exclude-module=IPython',
        '--exclude-module=jupyter',
        # Include what your app actually needs
        '--hidden-import=tkinter',
        '--hidden-import=tkinter.ttk',
        '--hidden-import=tkinter.messagebox',
        '--hidden-import=tkinter.filedialog',
        # Include numpy properly (your app needs it)
        '--hidden-import=numpy',
        '--hidden-import=numpy.core',
        '--hidden-import=numpy.core._methods',
        '--hidden-import=numpy.lib.format',
        '--collect-submodules=numpy',
        # Include PySide6 components
        '--hidden-import=PySide6.QtCore',
        '--hidden-import=PySide6.QtGui', 
        '--hidden-import=PySide6.QtWidgets',
        '--collect-submodules=PySide6',
        # Include Windows-specific modules
        '--hidden-import=win32gui',
        '--hidden-import=win32con',
        '--hidden-import=win32process',
        '--hidden-import=win32api',
        '--hidden-import=win32event',
        '--hidden-import=win32com',
        '--hidden-import=keyboard',
        # Use noupx to avoid compression issues
        '--noupx',
        str(main_script)
    ]
    
    return run_production_build(cmd, project_root)


def create_custom_spec():
    """Create a custom .spec file to have more control over the build."""
    print("Creating custom .spec file...")
    
    main_script = find_main_script()
    if not main_script:
        return None
    
    spec_content = f'''# -*- mode: python ; coding: utf-8 -*-

# Custom spec file for SPQ to avoid transformers issues

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
    ],
    hookspath=[],
    hooksconfig={{}},
    runtime_hooks=[],
    excludes=[
        'transformers',
        'torch',
        'tensorflow',
        'tf',
        'sklearn',
        'scipy',
        'pandas',
        'matplotlib',
        'PIL',
        'cv2',
        'IPython',
        'jupyter',
        'notebook',
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

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
    upx=False,  # Disable UPX compression to avoid issues
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,  # GUI app without console window
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
'''
    
    spec_file = Path('SPQ_custom.spec')
    with open(spec_file, 'w') as f:
        f.write(spec_content)
    
    print(f"Created spec file: {spec_file}")
    return spec_file


def main():
    """Main entry point with full diagnostics."""
    try:
        print("SPQ Build Diagnostics")
        print("=" * 50)
        
        # Step 1: Analyze all imports
        problematic_files = analyze_all_imports()
        
        # Step 2: Check installed packages
        check_installed_packages()
        
        # Step 3: Ask user what to do
        if problematic_files:
            print("\nWhat would you like to do?")
            print("1. Try to build anyway (minimal config)")
            print("2. Create custom .spec file for manual editing")
            print("3. Show me which files have the problematic imports")
            print("4. Exit and let me fix the imports first")
            
            choice = input("\nEnter choice (1-4): ").strip()
            
            if choice == "3":
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
                
            elif choice == "4":
                print("\nGood choice! Fix those imports first, then run the build again.")
                return 0
                
            elif choice == "2":
                spec_file = create_custom_spec()
                if spec_file:
                    print(f"\nSUCCESS: Created {spec_file}")
                    print("You can now edit it and run: pyinstaller SPQ_custom.spec")
                return 0
        
        # Default: try production build that includes what you need
        clean_build()
        success = build_executable_production()
        
        if success:
            print("\nBUILD COMPLETED SUCCESSFULLY!")
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