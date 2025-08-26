"""
Path utilities for resolving runtime directories in both source and frozen (packaged) runs.

- get_runtime_root(): directory containing the executable when frozen, else the repo root
- get_data_dir(): runtime_root / 'data'
"""
from __future__ import annotations

import sys
from pathlib import Path
import os


def is_frozen() -> bool:
    """Return True when running under a frozen/packaged executable."""
    return getattr(sys, "frozen", False) is True


def _normalize_portable_root(exe_dir: Path) -> Path:
    """Normalize a staged portable layout back to the portable root.

    If the executable lives under "<root>/data/bin" (common for Nuitka/PyInstaller
    staged one-directory payloads), return "<root>". If it's under "<root>/data",
    return "<root>" as well. Otherwise, return the input directory.
    """
    try:
        # Fast-path exact suffix checks
        name = exe_dir.name.lower()
        parent = exe_dir.parent if exe_dir is not None else None
        if name == 'bin' and parent and parent.name.lower() == 'data' and parent.parent:
            return parent.parent
        if name == 'data' and parent:
            return parent

        # General case: collapse any trailing .../data/bin back to root
        parts_lower = [p.lower() for p in exe_dir.parts]
        for i in range(len(parts_lower) - 2):
            j = i + 1
            # Look for the last occurrence of .../data/bin tail
            if parts_lower[-2] == 'data' and parts_lower[-1] == 'bin':
                # Root is everything before the 'data' segment
                return Path(*exe_dir.parts[: len(exe_dir.parts) - 2])
        return exe_dir
    except Exception:
        return exe_dir


def get_runtime_root() -> Path:
    """Resolve the runtime root directory.

    - Frozen: the directory containing the executable (sys.executable)
    - Source: the repository root (parent of the 'utils' directory containing this file)
    """
    # 1) Explicit override via environment (set by launcher)
    try:
        env_root = os.environ.get("SPQ_RUNTIME_ROOT")
        if env_root:
            return Path(env_root).resolve()
    except Exception:
        pass

    # 2) Frozen/compiled executable path
    if is_frozen() or getattr(sys, "compiled", False):  # Nuitka may set 'compiled'
        exe_dir = Path(sys.executable).resolve().parent
        return _normalize_portable_root(exe_dir)

    # 3) Source or fallback: derive from this file location and normalize if running under portable tree
    here = Path(__file__).resolve()
    utils_dir = here.parent
    # If the module path happens to be inside a staged portable tree (e.g., data/bin/utils), normalize to root
    try:
        parent = utils_dir.parent  # e.g., <root>/utils or <root>/data/bin
        grandparent = parent.parent if parent else None
        great = grandparent.parent if grandparent else None
        if parent and parent.name.lower() == 'bin' and grandparent and grandparent.name.lower() == 'data' and great:
            return great
        if parent and parent.name.lower() == 'data' and grandparent:
            return grandparent
    except Exception:
        pass
    # Normal source tree: <repo>/utils/paths.py -> repo root is parents[1]
    return here.parents[1]


def get_data_dir() -> Path:
    """Return the data directory for runtime assets."""
    return get_runtime_root() / "data"
