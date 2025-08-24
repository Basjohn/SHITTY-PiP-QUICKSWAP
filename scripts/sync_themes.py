#!/usr/bin/env python3
"""
Theme Synchronization Script

Synchronize styles between dark and light themes by mirroring the structure of
dark.qss into light.qss and inverting ALL color values across the entire file.

Rules:
 - Preserve structure, selectors, whitespace, ordering, and alpha values
 - Invert RGB components in rgba(), rgb(), and hex (#RGB/#RRGGBB)
 - Also invert color values that appear inside inline comments (e.g. 'color-picker: ...')
 - Do not change non-color tokens (numbers for sizes, radii, margins, etc.)

This script is idempotent relative to dark.qss: it always re-generates light.qss
from dark.qss by inversion, ensuring both stay in sync.
"""
import re
from pathlib import Path

def _clamp_byte(n: int) -> int:
    return max(0, min(255, n))

def _invert_byte(n: int) -> int:
    return 255 - _clamp_byte(n)

def invert_hex(hex_str: str) -> str:
    """Invert a hex color string. Supports #RGB and #RRGGBB (case preserved as lower)."""
    s = hex_str.strip()
    if not s.startswith('#'):
        return hex_str
    # Expand #RGB to #RRGGBB
    if len(s) == 4:
        s = '#' + ''.join(ch * 2 for ch in s[1:])
    if len(s) != 7:
        return hex_str
    try:
        r = int(s[1:3], 16)
        g = int(s[3:5], 16)
        b = int(s[5:7], 16)
        ri, gi, bi = _invert_byte(r), _invert_byte(g), _invert_byte(b)
        return f"#{ri:02x}{gi:02x}{bi:02x}"
    except ValueError:
        return hex_str

def invert_rgba(r: int, g: int, b: int, a: str) -> str:
    """Return inverted rgba() string preserving alpha token as-is."""
    return f"rgba({_invert_byte(r)}, {_invert_byte(g)}, {_invert_byte(b)}, {a})"

def invert_rgb(r: int, g: int, b: int) -> str:
    return f"rgb({_invert_byte(r)}, {_invert_byte(g)}, {_invert_byte(b)})"

def invert_all_colors(text: str) -> str:
    """Invert all color-like tokens in the provided QSS text.

    Order of replacement matters; do rgba -> rgb -> hex to avoid double-processing.
    """
    # rgba(R, G, B, A)
    def rgba_repl(m: re.Match) -> str:
        r, g, b, a = m.groups()
        return invert_rgba(int(r), int(g), int(b), a)

    text = re.sub(r"rgba\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*([\d.]+)\s*\)", rgba_repl, text)

    # rgb(R, G, B)
    def rgb_repl(m: re.Match) -> str:
        r, g, b = m.groups()
        return invert_rgb(int(r), int(g), int(b))

    text = re.sub(r"rgb\(\s*(\d{1,3})\s*,\s*(\d{1,3})\s*,\s*(\d{1,3})\s*\)", rgb_repl, text)

    # Hex colors (#RGB or #RRGGBB). Use negative lookbehind to avoid ## or identifiers.
    def hex_repl(m: re.Match) -> str:
        return invert_hex(m.group(0))

    text = re.sub(r"#[0-9a-fA-F]{3}(?:[0-9a-fA-F]{3})?\b", hex_repl, text)

    return text

def sync_themes():
    """Synchronize styles between dark and light themes (full-file inversion)."""
    project_root = Path(__file__).parent.parent
    dark_theme = project_root / 'themes' / 'dark.qss'
    light_theme = project_root / 'themes' / 'light.qss'
    
    if not dark_theme.exists() or not light_theme.exists():
        print("Error: Could not find theme files")
        return False
    
    # Read dark theme
    with open(dark_theme, 'r', encoding='utf-8') as f:
        dark_content = f.read()
    
    # Invert all color tokens across the entire dark theme content
    inverted_content = invert_all_colors(dark_content)

    # Write updated light theme (full overwrite)
    with open(light_theme, 'w', encoding='utf-8') as f:
        f.write(inverted_content)
    
    print("Successfully synchronized light.qss from dark.qss (full inversion)")
    return True

if __name__ == "__main__":
    sync_themes()
