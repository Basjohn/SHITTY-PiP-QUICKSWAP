"""
Utility module for cleaning up Python cache files.
This module provides functionality to clean up __pycache__ directories.
"""
import os
import shutil
from core.logging import get_logger
from pathlib import Path

logger = get_logger(__name__)

def cleanup_pycache(root_dir=None, suppress_errors=True):
    """
    Recursively remove all __pycache__ directories under the given root directory.
    
    Args:
        root_dir (str or Path, optional): Root directory to search for __pycache__ folders.
                                         If not provided, uses the parent directory of this file.
        suppress_errors (bool): If True, log errors but don't raise exceptions.
    
    Returns:
        tuple: (success_count, failure_count) - Number of successfully removed and failed removals
    """
    if root_dir is None:
        # Default to the parent directory of this file
        root_dir = Path(__file__).parent.parent
    
    root_path = Path(root_dir).resolve()
    success = 0
    failures = 0
    
    if not root_path.exists():
        if not suppress_errors:
            raise FileNotFoundError(f"Root directory does not exist: {root_path}")
        logger.warning(f"Cannot clean __pycache__: directory does not exist: {root_path}")
        return 0, 0
    
    logger.debug(f"Searching for __pycache__ directories in: {root_path}")
    
    for dirpath, dirnames, _ in os.walk(root_path, topdown=True):
        if os.path.basename(dirpath) == '__pycache__':
            try:
                shutil.rmtree(dirpath)
                success += 1
                logger.debug(f"Removed: {dirpath}")
            except Exception as e:
                failures += 1
                if not suppress_errors:
                    raise
                logger.warning(f"Failed to remove {dirpath}: {e}")
    
    if success or failures:
        logger.info(f"Cache cleanup complete. Removed: {success} __pycache__ directories, Failed: {failures}")
    
    return success, failures


def register_cleanup_on_exit(root_dir=None, cleanup_on_start=True):
    """
    Register the cleanup_pycache function to run at interpreter shutdown.
    Optionally also cleans on registration (startup).
    
    Args:
        root_dir (str or Path, optional): Root directory to clean. If None, uses default.
        cleanup_on_start (bool): If True, also clean immediately on registration (default: True)
    """
    import atexit
    
    # Clean on startup to ensure fresh bytecode
    if cleanup_on_start:
        cleanup_pycache(root_dir=root_dir, suppress_errors=True)
    
    # Also clean on exit
    atexit.register(cleanup_pycache, root_dir=root_dir, suppress_errors=True)
    logger.debug("Registered __pycache__ cleanup for application exit")