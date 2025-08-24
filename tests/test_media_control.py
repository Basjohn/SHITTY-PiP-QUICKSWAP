#!/usr/bin/env python3
"""Test script for enhanced media control functionality.

Pytest note: this module is intentionally skipped during pytest runs to avoid
external media control side effects in CI/local test loops.
"""

import sys
import time
from pathlib import Path
import pytest

# Skip this module in pytest runs
pytestmark = pytest.mark.skip(reason="Media control test skipped by default in pytest")

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from core.media.media_controller import get_media_controller
from core.logging import get_logger

def test_media_control():
    """Test media control functionality."""
    logger = get_logger("MediaControlTest")
    controller = get_media_controller()
    
    print("=== Enhanced Media Control Test ===")
    print(f"Media control enabled: {controller.is_enabled()}")
    
    # List running media apps
    running_apps = controller.list_running_apps()
    print(f"Running media apps: {running_apps}")
    
    if not running_apps:
        print("No media applications found. Please start a media player and try again.")
        return
    
    # Test play/pause on any available app
    print("\n--- Testing play/pause (any app) ---")
    success, msg = controller.play_pause()
    print(f"Result: {msg} ({'✓' if success else '✗'})")
    
    # Test specific apps if available
    test_apps = ['spotify', 'chrome', 'firefox', 'vlc', 'mpc_hc', 'mpc_be']
    for app in test_apps:
        if app in running_apps:
            print(f"\n--- Testing {app} ---")
            
            # Test play/pause
            success, msg = controller.play_pause(app)
            print(f"Play/Pause: {msg} ({'✓' if success else '✗'})")
            
            time.sleep(0.5)  # Small delay between commands
            
            # Test next track
            success, msg = controller.next_track(app)
            print(f"Next: {msg} ({'✓' if success else '✗'})")
            
            time.sleep(0.5)
            
            # Test previous track
            success, msg = controller.previous_track(app)
            print(f"Previous: {msg} ({'✓' if success else '✗'})")
    
    print("\n=== Test Complete ===")

if __name__ == "__main__":
    test_media_control()
