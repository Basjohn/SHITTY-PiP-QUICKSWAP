import os
from pathlib import Path

import pytest

from core.input.keypassthrough_blocklist import KeyPassthroughBlocklist


def _write(path: Path, content: str, mtime: float | None = None) -> None:
    path.write_text(content, encoding="utf-8")
    if mtime is not None:
        os.utime(path, (mtime, mtime))


class TestKeyPassthroughBlocklist:
    def test_parse_and_match_precedence(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        # Arrange: rules where exe and title would both match; exe must take precedence
        blk = tmp_path / "blk.txt"
        content = """
        # comment
        GameAntiCheat.exe
        title words here
        {"title_exact": "Exact Title"}
        {"exe": "Browser.exe"}
        {"title_contains": ["Foo", "Bar"]}
        {"title_contains": "multi words"}
        """.strip()
        _write(blk, content, mtime=1000.0)

        # Force initial load by advancing monotonic beyond throttle window
        t = {"now": 3.0}

        def fake_monotonic() -> float:
            return t["now"]

        monkeypatch.setattr(
            "core.input.keypassthrough_blocklist.time.monotonic", fake_monotonic
        )

        loader = KeyPassthroughBlocklist(blocklist_path=blk)

        # Act + Assert: exe precedence
        assert loader.match("GameAntiCheat.exe", "title words here something") == {
            "type": "exe",
            "value": "GameAntiCheat.exe",
            "source": "plain",
        }
        # Title exact
        assert loader.match("", "Exact Title") == {
            "type": "title_exact",
            "value": "Exact Title",
            "source": "json",
        }
        # Title contains ALL terms (list)
        assert loader.match("", "x foo y bar z") == {
            "type": "title_contains",
            "value": ["Foo", "Bar"],
            "source": "json",
        }
        # Title contains from plain line (split into words)
        assert loader.match("", "xx title yy words here zz") == {
            "type": "title_contains",
            "value": ["title", "words", "here"],
            "source": "plain",
        }

    def test_title_exact_case_insensitive(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        blk = tmp_path / "blk.txt"
        _write(blk, '{"title_exact": "Special Window"}\n', mtime=2000.0)

        t = {"now": 3.0}

        def fake_monotonic() -> float:
            return t["now"]

        monkeypatch.setattr(
            "core.input.keypassthrough_blocklist.time.monotonic", fake_monotonic
        )

        loader = KeyPassthroughBlocklist(blocklist_path=blk)
        assert loader.match(None, "SPECIAL WINDOW") == {
            "type": "title_exact",
            "value": "Special Window",
            "source": "json",
        }

    def test_reload_throttling_and_mtime(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
        blk = tmp_path / "blk.txt"
        # Initial rule: a.exe
        _write(blk, "a.exe\n", mtime=3000.0)

        # Start past throttle window so first call loads
        t = {"now": 3.0}

        def fake_monotonic() -> float:
            return t["now"]

        monkeypatch.setattr(
            "core.input.keypassthrough_blocklist.time.monotonic", fake_monotonic
        )

        loader = KeyPassthroughBlocklist(blocklist_path=blk)

        # Initial match works
        assert loader.match("a.exe", "") == {
            "type": "exe",
            "value": "a.exe",
            "source": "plain",
        }

        # Modify file to b.exe but keep within throttle window; should NOT reload yet
        _write(blk, "b.exe\n", mtime=3010.0)
        t["now"] = 4.0  # < throttle interval since last check
        assert loader.match("b.exe", "") is None

        # Advance beyond throttle; should reload and match new rule
        t["now"] = 6.5
        assert loader.match("b.exe", "") == {
            "type": "exe",
            "value": "b.exe",
            "source": "plain",
        }
