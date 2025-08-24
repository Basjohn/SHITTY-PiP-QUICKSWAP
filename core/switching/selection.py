from __future__ import annotations
from typing import Callable, Dict, List, Optional, Tuple

# Centralized cyclic MRU selection logic for overlay window switching.
# No fallbacks: explicit, deterministic, with clear reasons in logs by callers.

OverlayKey = str


def _normalize_key(overlay_id: Optional[object]) -> OverlayKey:
    return str(overlay_id) if overlay_id is not None else "_global"


def compute_next_selection(
    *,
    candidates: List[int],
    filtered: List[int],
    cur_hwnd: Optional[int],
    current_src: Optional[int],
    cycle_last_by_overlay: Dict[OverlayKey, int],
    overlay_id: Optional[object],
    pick_from_zorder: Callable[[Optional[int], Optional[int]], Optional[int]],
    is_valid: Callable[[int], bool],
) -> Tuple[Optional[int], int, str, List[int], int, Optional[int], Optional[int]]:
    """
    Compute the next selection for overlay switching using cyclic MRU rules.

    Returns: (selected_hwnd, selected_display_idx, reason, ordered_list, start_idx, ref_last, ref_fore)
      - selected_hwnd: chosen hwnd or None if none valid
      - selected_display_idx: 1-based index in the filtered list for logging (relative to original filtered order)
      - reason: 'after_last' | 'after_foreground' | 'after_zorder' | 'first'
      - ordered_list: filtered list rotated to start position (for debugging)
      - start_idx: the rotation start index used on the filtered list
      - ref_last: last hwnd reference used (overlay/global)
      - ref_fore: the current foreground hwnd reference considered
    """
    if not filtered:
        return None, 0, "first", [], 0, None, cur_hwnd

    key_active = _normalize_key(overlay_id)

    def pick_start_idx(reference: Optional[int]) -> Optional[int]:
        if reference is None:
            return None
        if reference in filtered:
            return (filtered.index(reference) + 1) % len(filtered)
        if reference in candidates:
            start = (candidates.index(reference) + 1) % len(candidates)
            for k in range(len(candidates)):
                c = candidates[(start + k) % len(candidates)]
                if c in filtered:
                    return filtered.index(c)
        return None

    ref_last = cycle_last_by_overlay.get(key_active)
    if ref_last is None:
        ref_last = cycle_last_by_overlay.get("_global")

    idx_from_last = pick_start_idx(ref_last)
    idx_from_fore = pick_start_idx(cur_hwnd)

    if idx_from_last is not None:
        next_idx, reason = idx_from_last, "after_last"
    elif idx_from_fore is not None:
        next_idx, reason = idx_from_fore, "after_foreground"
    else:
        z_hint = None
        try:
            z_hint = pick_from_zorder(cur_hwnd, current_src)
        except Exception:
            z_hint = None
        if z_hint in filtered:
            next_idx, reason = filtered.index(z_hint), "after_zorder"
        else:
            next_idx, reason = 0, "first"

    ordered = filtered[next_idx:] + filtered[:next_idx]

    # Find first valid in ordered list
    for offset, hwnd in enumerate(ordered):
        if is_valid(hwnd):
            # Update cycle state
            cycle_last_by_overlay[key_active] = int(hwnd)
            cycle_last_by_overlay["_global"] = int(hwnd)
            # Display idx relative to original filtered ordering
            display_idx = ((next_idx + offset) % len(filtered)) + 1
            return hwnd, display_idx, reason, ordered, next_idx, ref_last, cur_hwnd

    # No valid hwnd found
    return None, 0, reason, ordered, next_idx, ref_last, cur_hwnd
