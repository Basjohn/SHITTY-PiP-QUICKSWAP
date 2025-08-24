#!/usr/bin/env python3
"""
Run pytest with the project's logging-first policy on Windows and produce deterministic output.

Behavior:
- Launches pytest with all output redirected to pytest.log in the repo root
- Waits up to --timeout seconds for pytest to finish
  - If still running after the timeout, terminates pytest
- Sleeps an additional --post-wait seconds to ensure logs are flushed
- Tails the last --tail lines of pytest.log to stdout

Defaults follow our policy: 30s guard + 45s wait before reading logs.

Examples:
  python scripts/run_pytest_guarded.py
  python scripts/run_pytest_guarded.py --maxfail 5 -k tests/test_events.py::TestEventSystem::test_subscription_priority
  python scripts/run_pytest_guarded.py --collect-only

Exit codes:
- If pytest completes: returns pytest's exit code
- If terminated due to timeout: returns 124 (conventional timeout code)
- If pytest failed to start or pytest.log missing: returns 2
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def tail_file(path: Path, lines: int) -> str:
    if not path.exists():
        return "pytest.log not found - test may have failed to start"
    try:
        with path.open("r", encoding="utf-8", errors="replace") as f:
            content = f.readlines()
        return "".join(content[-lines:])
    except Exception as e:
        return f"Failed to read log: {e}"


def run_pytest(args: argparse.Namespace) -> int:
    repo_root = Path(__file__).resolve().parents[1]
    log_path = repo_root / "pytest.log"

    pytest_cmd: List[str] = [sys.executable, "-m", "pytest", "-vv"]
    if args.collect_only:
        pytest_cmd = [sys.executable, "-m", "pytest", "--collect-only", "-q"]
    if args.maxfail is not None and not args.collect_only:
        pytest_cmd += [f"--maxfail={int(args.maxfail)}"]
    # Additional user-specified pytest args (e.g., -k, test paths)
    if args.pytest_args:
        pytest_cmd += args.pytest_args

    # Open the log file for both stdout and stderr
    try:
        log_f = log_path.open("w", encoding="utf-8", errors="replace")
    except Exception as e:
        print(f"ERROR: Unable to open {log_path} for writing: {e}", file=sys.stderr)
        return 2

    print(f"Running: {' '.join(pytest_cmd)}")
    print(f"Logging to: {log_path}")
    sys.stdout.flush()

    proc = None
    try:
        proc = subprocess.Popen(
            pytest_cmd,
            cwd=str(repo_root),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            env=os.environ.copy(),
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0),
        )
    except FileNotFoundError as e:
        print(f"ERROR: Failed to start pytest: {e}", file=sys.stderr)
        log_f.close()
        return 2

    # Guarded wait
    try:
        ret = proc.wait(timeout=args.timeout)
        finished = True
    except subprocess.TimeoutExpired:
        finished = False
        ret = 124  # conventional timeout exit code
        print(f"Pytest did not complete within {args.timeout}s; terminating...")
        try:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        except Exception:
            pass

    # Ensure file handlers are flushed and closed before reading
    try:
        log_f.flush()
    except Exception:
        pass
    finally:
        log_f.close()

    # Post-wait to allow log system to flush
    if args.post_wait > 0:
        print(f"Waiting {args.post_wait}s before reading logs...")
        sys.stdout.flush()
        time.sleep(args.post_wait)

    # Tail the log
    print("\n===== pytest.log (tail) =====")
    print(tail_file(log_path, args.tail))
    print("===== end tail =====\n")
    sys.stdout.flush()

    return ret if finished else 124


def main() -> int:
    parser = argparse.ArgumentParser(description="Run pytest with logging-first guard and deterministic tail output")
    parser.add_argument("pytest_args", nargs=argparse.REMAINDER, help="Additional pytest args (e.g., -k expr, test paths)")
    parser.add_argument("--maxfail", type=int, default=2, help="Max failures before pytest stops (ignored in --collect-only)")
    parser.add_argument("--timeout", type=int, default=30, help="Seconds to wait for pytest to finish before termination")
    parser.add_argument("--post-wait", type=int, default=45, help="Seconds to sleep before tailing the log")
    parser.add_argument("--tail", type=int, default=120, help="Number of lines to tail from pytest.log")
    parser.add_argument("--collect-only", action="store_true", help="Only collect tests and exit")

    args = parser.parse_args()

    try:
        return run_pytest(args)
    except KeyboardInterrupt:
        print("Interrupted.")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
