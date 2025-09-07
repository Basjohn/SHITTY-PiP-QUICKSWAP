# Ensure the repository root is on sys.path so top-level packages (e.g., `core`, `utils`, `ui`) can be imported
# consistently whether running the full test suite or an individual test file.
from __future__ import annotations

import os
import sys
import threading
import time
import subprocess
from pathlib import Path
from typing import Optional

import pytest

# tests/ -> repo root
_REPO_ROOT = Path(__file__).resolve().parents[1]
repo_str = str(_REPO_ROOT)
if repo_str not in sys.path:
    sys.path.insert(0, repo_str)

# Centralized test logging setup (rotating file logs)
try:
    from core.logging import configure_logging, get_logger
except Exception:
    configure_logging = None  # type: ignore
    def get_logger(name: str):  # type: ignore
        import logging
        return logging.getLogger(name)

_TEST_LOG_DIR: Path = _REPO_ROOT / "logs" / "tests"
_LOGGER = get_logger("TESTS")

def pytest_configure(config: pytest.Config) -> None:
    """Configure centralized rotating logging for test runs."""
    # Ensure directory exists even if core.logging is unavailable
    try:
        _TEST_LOG_DIR.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

    if configure_logging is not None:
        # 2MB cap, 3 backups for test logs; verbose to file, concise on console
        app_logger = configure_logging(
            name="tests",
            log_dir=_TEST_LOG_DIR,
            console_level=20,  # INFO
            file_level=10,     # DEBUG
            max_bytes=2 * 1024 * 1024,
            backup_count=3,
            enable_exception_hook=True,
            resource_manager=None,
        )
        # Stash for potential plugins/fixtures
        setattr(config, "_app_logger", app_logger)

    os.environ["SPQ_TEST_LOG_DIR"] = str(_TEST_LOG_DIR)
    _LOGGER.info(f"Test logging initialized at: {_TEST_LOG_DIR}")

    # Optional timeout watchdog (opt-in): set SPQ_TEST_TIMEOUT=1
    if os.environ.get("SPQ_TEST_TIMEOUT", "0").lower() in {"1", "true", "yes", "on"}:
        try:
            seconds = float(os.environ.get("SPQ_TEST_TIMEOUT_SECONDS", "60"))
        except Exception:
            seconds = 60.0
        _start_timeout_watchdog(seconds)

@pytest.fixture(scope="session")
def test_log_dir() -> Path:
    """Path to the rotating test log directory."""
    return _TEST_LOG_DIR

def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    _LOGGER.info(f"Pytest session finished with status {exitstatus}. Logs at: {_TEST_LOG_DIR}")


# --- Internal helpers ---

def _start_timeout_watchdog(timeout_seconds: float) -> None:
    """Start a background watchdog that enforces a hard timeout on the test run.

    On timeout, attempts to:
      1) Log a timeout message
      2) Print the tail of the latest test log file to stderr
      3) Kill the current process (Windows: taskkill /F /PID <pid>), else os._exit(3)
    """
    pid = os.getpid()

    def _tail_file(path: Path, lines: int = 200) -> str:
        try:
            data = path.read_text(encoding="utf-8", errors="replace").splitlines()
            return "\n".join(data[-lines:])
        except Exception:
            return ""

    def _latest_log_file() -> Optional[Path]:
        try:
            files = sorted(_TEST_LOG_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime)
            return files[-1] if files else None
        except Exception:
            return None

    def _watchdog() -> None:
        _LOGGER.warning(f"Starting test timeout watchdog: {timeout_seconds:.1f}s")
        time.sleep(max(0.1, timeout_seconds))
        _LOGGER.error("Test runtime exceeded timeout. Attempting to dump latest logs and terminate.")
        latest = _latest_log_file()
        if latest:
            tail = _tail_file(latest)
            if tail:
                sys.stderr.write("\n===== BEGIN TEST LOG TAIL =====\n")
                sys.stderr.write(tail + "\n")
                sys.stderr.write("===== END TEST LOG TAIL =====\n")
                sys.stderr.flush()
        # Attempt taskkill on Windows for a clean termination
        if os.name == "nt":
            try:
                subprocess.run(["taskkill", "/F", "/PID", str(pid)], check=False, capture_output=True)
                # Give a moment for termination
                time.sleep(0.5)
            except Exception:
                pass
        # Fallback hard exit (process may already be terminating)
        try:
            os._exit(3)  # noqa: PLR1722
        except Exception:
            pass

    t = threading.Thread(target=_watchdog, name="pytest-timeout-watchdog", daemon=True)
    t.start()
