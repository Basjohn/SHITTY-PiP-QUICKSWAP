from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.logging import get_logger
from core.settings.settings_manager import SettingsManager

try:
    import win32gui
    import win32process
    import win32con
    _WIN_AVAILABLE = True
except Exception:
    _WIN_AVAILABLE = False

# Throttle interval for checking mtime changes (seconds)
_THROTTLE_SEC = 2.0


@dataclass
class BlockRule:
    kind: str  # 'exe' | 'title_exact' | 'title_contains'
    value: Any  # str or List[str]
    source: str  # 'plain' | 'json'
    raw: str    # original line for reference


class KeyPassthroughBlocklist:
    """Parser/loader for key passthrough blocklist with caching and mtime checks."""

    def __init__(self, blocklist_path: Optional[Path] = None) -> None:
        self._logger = get_logger("KEYPASS")
        self._settings = SettingsManager()
        if blocklist_path is None:
            try:
                settings_dir: Path = self._settings.get_settings_dir()
            except Exception:
                settings_dir = Path.home() / ".spqmodular"
            blocklist_path = settings_dir / "keypassthrough_blocklist.txt"
        self._path: Path = Path(blocklist_path)

        # Cache
        self._rules: List[BlockRule] = []
        self._mtime: float = 0.0
        self._last_check: float = 0.0

    # --- Public API -----------------------------------------------------
    def match(self, exe_name: str | None, window_title: str | None) -> Optional[Dict[str, Any]]:
        """Return match info if exe/title matches a rule, else None.

        Precedence: exe > title_exact > title_contains.
        """
        if not self._ensure_loaded():
            # Even if not loaded, proceed with empty rules
            pass
        exe = (exe_name or "").strip().lower()
        title = (window_title or "").strip()
        title_l = title.lower()

        # 1) exe
        for r in self._rules:
            if r.kind == "exe" and exe and exe == str(r.value).lower():
                return {
                    "type": "exe",
                    "value": r.value,
                    "source": r.source,
                }
        # 2) title_exact
        for r in self._rules:
            if r.kind == "title_exact" and title and title_l == str(r.value).lower():
                return {
                    "type": "title_exact",
                    "value": r.value,
                    "source": r.source,
                }
        # 3) title_contains (all terms)
        for r in self._rules:
            if r.kind == "title_contains" and title:
                terms = [str(t).lower() for t in (r.value if isinstance(r.value, list) else [r.value])]
                if all(t in title_l for t in terms):
                    return {
                        "type": "title_contains",
                        "value": r.value,
                        "source": r.source,
                    }
        return None

    def match_for_hwnd(self, hwnd: int) -> Optional[Dict[str, Any]]:
        """Fetch process exe and window title for hwnd and run match()."""
        if not _WIN_AVAILABLE or not hwnd:
            return None
        try:
            # Title
            title = win32gui.GetWindowText(hwnd)
            # Exe name
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            exe_path = None
            try:
                # PROCESS_QUERY_LIMITED_INFORMATION (0x1000) may be enough, but use standard flags
                import win32api
                process_handle = win32api.OpenProcess(
                    win32con.PROCESS_QUERY_INFORMATION | win32con.PROCESS_VM_READ,
                    False,
                    pid,
                )
                try:
                    exe_path = win32process.GetModuleFileNameEx(process_handle, 0)
                finally:
                    win32api.CloseHandle(process_handle)
            except Exception:
                exe_path = None
            exe_name = os.path.basename(exe_path) if exe_path else None
            return self.match(exe_name, title)
        except Exception as e:
            # Non-fatal
            self._logger.debug(f"Blocklist hwnd match failed: {e}")
            return None

    # --- Internal -------------------------------------------------------
    def _ensure_loaded(self) -> bool:
        now = time.monotonic()
        if now - self._last_check < _THROTTLE_SEC:
            return True
        self._last_check = now
        try:
            st = self._path.stat()
            mtime = st.st_mtime
        except FileNotFoundError:
            # No file yet; keep empty rules
            self._mtime = 0.0
            self._rules = []
            return True
        except Exception as e:
            self._logger.error(f"[KEYPASS] Failed to stat blocklist file {self._path}: {e}")
            return False
        if mtime == self._mtime:
            return True
        # Reload
        try:
            rules = self._parse_file(self._path)
            self._rules = rules
            self._mtime = mtime
            self._logger.debug(f"[KEYPASS] Reloaded blocklist: {len(self._rules)} rules")
            return True
        except Exception as e:
            self._logger.error(f"[KEYPASS] Failed to parse blocklist file {self._path}: {e}", exc_info=True)
            return False

    def _parse_file(self, path: Path) -> List[BlockRule]:
        rules: List[BlockRule] = []
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
        except Exception as e:
            raise e

        for raw in lines:
            line = raw.strip()
            if not line or line.startswith('#'):
                continue
            # allow trailing comma for convenience
            if line.endswith(','):
                line = line[:-1].rstrip()
            # JSON object line?
            if line.startswith('{') and line.endswith('}'):
                try:
                    obj = json.loads(line)
                    # normalize keys case-insensitive
                    norm: Dict[str, Any] = {str(k).strip().lower(): v for k, v in obj.items()}
                    if 'exe' in norm and isinstance(norm['exe'], str) and norm['exe'].strip():
                        rules.append(BlockRule('exe', norm['exe'].strip(), 'json', raw))
                        continue
                    if 'title_exact' in norm and isinstance(norm['title_exact'], str) and norm['title_exact'].strip():
                        rules.append(BlockRule('title_exact', norm['title_exact'].strip(), 'json', raw))
                        continue
                    if 'title_contains' in norm:
                        v = norm['title_contains']
                        if isinstance(v, str) and v.strip():
                            # split into words
                            terms = [t for t in v.strip().split() if t]
                            if terms:
                                rules.append(BlockRule('title_contains', terms, 'json', raw))
                                continue
                        elif isinstance(v, list):
                            terms = [str(t).strip() for t in v if str(t).strip()]
                            if terms:
                                rules.append(BlockRule('title_contains', terms, 'json', raw))
                                continue
                    # If object not recognized, skip silently with debug
                    self._logger.debug(f"[KEYPASS] Ignoring unrecognized JSON rule: {raw.strip()}")
                except Exception as e:
                    self._logger.warning(f"[KEYPASS] Failed to parse JSON rule: {raw.strip()} ({e})")
                continue
            # Plain line rule
            if line.lower().endswith('.exe'):
                rules.append(BlockRule('exe', line, 'plain', raw))
            else:
                # Split words; all must match in any order
                terms = [t for t in line.split() if t]
                if terms:
                    rules.append(BlockRule('title_contains', terms, 'plain', raw))
        return rules


_singleton: Optional[KeyPassthroughBlocklist] = None


def get_blocklist() -> KeyPassthroughBlocklist:
    global _singleton
    if _singleton is None:
        _singleton = KeyPassthroughBlocklist()
    return _singleton
