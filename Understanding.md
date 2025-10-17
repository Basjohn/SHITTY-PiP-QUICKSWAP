# SPQ Project Understanding Guide

This document complements `Index.md` (file map) and `Spec.md` (architecture and policies). It focuses on how the system works end-to-end so you (and I) can quickly reason about behavior, debug issues, and extend features with minimal friction.

Use this as a living doc; keep it high signal and grounded in actual code.

---

## High-level Overview

- **Purpose**
  - SPQ renders one or more window overlays using DWM thumbnails (or alternative backends) and provides rich interaction (quickswitch, foreground-autoswitch, closed-window switching, docking, opacity, context menus, etc.).
  - **DWM Capture Mode**: Uses full window capture (`source_client_area_only=False`) to properly render modern Windows apps (Task Manager, Notepad, Settings) that use DWM Extended Frame technique. This matches Windows taskbar preview behavior.
- **Core execution model**
  - Single UI thread (Qt). Centralized `ThreadManager` provides coalesced UI scheduling and pools for other work.
  - Most logic is lock-free and uses: "UI-thread only mutation + coalesced timers".
  - **Single instance enforcement**: `ApplicationInstanceManager` uses OS-level mutex to prevent multiple instances (prevents keyboard hook conflicts).
- **Centralized systems**
  - `utils/resource_manager.py` manages lifecycle registration and cleanup.
  - `core/graphics/overlay_manager.py` manages window-mode overlays (non-docking), MRU of overlays, auto-switch registration.
  - `core/graphics/docking/manager.py` orchestrates the multi-overlay docking system (positioning, binding, MRU-aware assignment, quickswitch mapping, cycle mode).
  - `core/switching/*` provides MRU tracking, quickswitch, and foreground-autoswitch controllers; `core/graphics/window_monitor.py` provides closed-window switching.
  - `utils/window_validation.py` is the single truth for filtering valid windows.

```mermaid
flowchart LR
  subgraph UI Thread
  A[User Input (mouse/keyboard)] --> B[OverlayHost / Widgets]
  B --> C[DockingOverlay / Overlay]
  C -->|show/update| D[IntegratedDWMOverlay]
  end

  D -->|register/cleanup| RM[ResourceManager]
  C <-->|geometry/opacity| DM[DockingOverlayManager]
  OM[OverlayManager] <-->|MRU sync| DM
  MRU[MRUManager] <-->|read/record| QSC[QuickSwitchController]
  MRU <-->|push/pull| DM
  FASC[ForegroundAutoswitchController] <-->|monitor focus| OM & DM
  CWSM[ClosedWindowSwitchManager] <-->|monitor closures| OM & DM
```

## Recent changes

### 2025-10-12: Aspect Ratio Canvas Initialization Fix

**ISSUE 1**: Windows with out-of-bounds AR (e.g., Spotify 237x39, AR 6.077) fell back to 16:9 but rendered as **tiny thumbnails with massive padding** on all sides in docking mode.

**ROOT CAUSE**: Canvas AR was set during initialization with arbitrary dimensions (`1280x720`) before the docking manager sized the overlay. Canvas calculated `content_rect` as tiny (e.g., `247x139`) based on those preset dimensions, causing severe size mismatch with actual overlay geometry (e.g., `862x492`).

**FIX**: `core/graphics/backends/dwm/integrated_dwm_backend.py` 
- **Initialization (lines 175-194)**: Skip canvas AR setup entirely for docking overlays
```python
# For docking overlays, skip AR init - let manager handle it
if not self._is_docking_overlay():
    # Standalone overlay - set AR immediately from window dimensions
    self._canvas.set_content_aspect(src_w, src_h)
# Docking overlay - AR will be set dynamically after manager sizes overlay
```

- **Dynamic AR Setup (lines 621-652)**: Set canvas AR in `_update_thumbnail_properties()` based on window validity
```python
# Check if window dimensions match the cached source aspect (within 5%)
if abs(window_ar - source_aspect) / source_aspect < 0.05:
    # Valid window AR - use actual window dimensions
    self._canvas.set_content_aspect(src_w, src_h)  # e.g., 2106x1071
else:
    # Fallback AR - scale from overlay's current size
    overlay_w = self._host.width()  # e.g., 862
    overlay_h = int(overlay_w / source_aspect)  # e.g., 485 for 16:9
    self._canvas.set_content_aspect(overlay_w, overlay_h)
```

**RESULT**: Canvas AR always matches overlay's actual size, eliminating tiny thumbnails. Overlays B/C/D/E (valid AR) use window dimensions, overlay A (fallback AR) scales from overlay size.

**ISSUE 2**: Wheel-resize triggered **endless resize loop** - kept oscillating between 120px and 365px height.

**ROOT CAUSE**: AR validation ran on EVERY `sync_overlay_properties()` call, including during user resizes:
1. User resizes to 648x120 (AR 6.077) 
2. Sync triggers → AR validation detects mismatch
3. Resizes to 648x371 (AR 1.778)
4. Triggers another resize event → another sync → loop continues

**FIX**: `core/graphics/docking/manager.py` Smart HWND-based validation:
```python
# Track which HWND was validated
self._last_validated_main_hwnd = None
self._user_resize_in_progress = False  # Set during wheel/resize events

# Only validate AR when HWND changes (swaps) OR first time
should_validate = (
    not self._user_resize_in_progress and
    (current_main_hwnd != self._last_validated_main_hwnd or ...)
)
```

**AR Validation Behavior Matrix**:
| Trigger | Validates? | Reason |
|---------|-----------|--------|
| Creation/initialization | ✅ | `_last_validated_main_hwnd == None` |
| Window swap (manual/autoswitch/cycle) | ✅ | HWND changed |
| User wheel-resize | ❌ | `_user_resize_in_progress == True` |
| User drag-resize | ❌ | `_user_resize_in_progress == True` |
| Secondary positioning sync | ❌ | No HWND change |

**AUTOSWITCH/CYCLE INTEGRATION**: Added sync triggers after display updates (lines 2876, 1118, 2726) to ensure AR validation runs after autoswitch/cycle swaps new windows.

**RESULT**: 
- Fallback AR windows render properly with no padding issues
- User resizing works smoothly without AR validation interference  
- Autoswitch/cycle swaps correctly validate and resize for new window ARs
- Eliminates endless resize loops

**Audit**: See `audits/ar_chain_reaction_debugging_report.md` for complete analysis

### 2025-10-09: Overlay E Bottom Alignment Fix

**ISSUE**: Overlay E (rightmost/smallest secondary) drifted downward at minimum sizes when using bottom alignment, but top alignment worked perfectly.

**ROOT CAUSE**: Asymmetric canvas inset handling - insets were added to width but NOT height.

**THE BUG**:
```python
# BEFORE (BROKEN):
sec_w = max(1, int(inner_w + 2 * ix))  # Width: insets added ✓
sec_sizes.append((sec_w, sec_h))       # Height: insets missing ✗
```

**WHY IT ONLY BROKE BOTTOM**:
- Top alignment: `y = main_top` (no height dependency) ✓
- Bottom alignment: `y = main_bottom - sec_h` (depends on accurate height) ✗
- At minimum size: Missing 12px of canvas insets caused 12px downward drift

**FIX**: `core/graphics/docking/manager.py` lines 1632, 1637
```python
sec_h = max(1, int(sec_h + 2 * iy))  # Add canvas insets to height (symmetric!)
```

**RESULT**: Top and bottom alignment now work identically at all sizes, all DPI scales.

**Audit**: See `audits/overlay_e_bottom_alignment_investigation.md`

### 2025-10-09: Wheel Resize Screen Boundary Performance Fix

**ISSUE**: Wheel resize was extremely slow at **screen edges** (especially bottom-left corner) - only 2px growth per wheel tick instead of the calculated 56px.

**ROOT CAUSE**: Boundary smoothing logic was triggering when pinned edges were **stable** (window growing away from boundary, not into it).

**THE BUG**: `utils/window/behavior.py` lines 1127-1144
```python
# BEFORE (BROKEN):
if clamped != new_rect and abs(dx) > 1:
    step = max(1, min(abs(dx), 2))  # Force 2px max when ANY boundary detected!
```

**WHY SCREEN EDGES MATTERED**:
- **Bottom-left corner (x=0)**: Left edge pinned at x=0
  - Window tries to grow → geometry calculation allows left < 0 → clamping brings back to 0
  - `clamped != new_rect` detected → **boundary smoothing reduces 56px to 2px** ✗
  - Result: Extremely slow resize
- **Bottom-right corner**: No boundary conflict
  - No clamping needed → full 56px applied → **fast resize** ✓

**FIX**: Lines 1129-1144
```python
# AFTER (FIXED):
pinned_edges_stable = False
if pinned_left and clamped.left() == new_rect.left() == cur_geo.left():
    pinned_edges_stable = True  # Left edge stable, growing rightward
# (same for right/top/bottom)

if clamped != new_rect and abs(dx) > 1 and not pinned_edges_stable:
    step = max(1, min(abs(dx), 2))  # Only smooth when actually hitting boundary
```

**RESULT**: Consistent wheel-resize speed at **all corners** and positions. Boundary smoothing still works when actually needed (window trying to move beyond screen limits), but doesn't trigger when pinned edges are stable and window is growing away from boundaries.

**Audit**: See `audits/docking_persistence_resize_conflicts.md`

### 2025-10-05: Z-Order Enforcement for Secondary Overlays

**ISSUE**: Secondary overlays (B/C/D/E) were frequently being covered by Windows taskbar, while main overlay (A) was always correct.

**ROOT CAUSE**: Z-order enforcement was registered but debounced/queued (16ms delay or 7ms coalescing window). Overlays became visible before `SetWindowPos(HWND_TOPMOST)` was called, allowing taskbar to appear above them.

**FIX**: Added explicit `ZOrderPriority.CRITICAL` enforcement immediately after showing overlays:
- Main overlay: Enforced after `show()` call
- Secondary overlays: Enforced in loop after each `show()` call  
- CRITICAL priority bypasses debouncing for immediate enforcement
- Uses Windows `SetWindowPos(HWND_TOPMOST)` via z_order_manager

**Location**: `core/graphics/docking/manager.py::_show_all_overlays()` lines 793-813

**Result**: All overlays guaranteed to be at `HWND_TOPMOST` z-level when shown, preventing taskbar coverage

**Audit**: See `audits/docking_z_order_investigation.md` for full analysis

### 2025-10-05: MRU Single Source of Truth Architecture (COMPLETE REFACTOR)

**CRITICAL ARCHITECTURAL CHANGE**: Eliminated all dual MRU lists - MRUManager is now the ONLY storage.

**OLD ARCHITECTURE (REMOVED)**:
- ❌ `DockingManager._mru_list` (local cache, caused stale data)
- ❌ `_on_mru_changed()` callback (listener/push system)
- ❌ Synchronization logic (race conditions, "sometimes works")
- ❌ Suppression flags (complex timing issues)

**NEW ARCHITECTURE (Single Source of Truth)**:
```
MRUManager._mru  ← SINGLE SOURCE OF TRUTH (window HWNDs)
     ↑         ↑
     │ (read)  │ (record)
     ↓         ↓
DockingManager + OverlayManager
  (no local caches, always fresh)
```

**How it works now**:
1. `FocusTracker` polls foreground window every 200ms
2. Records valid windows to `MRUManager` (filters out overlays via PID)
3. **DockingManager reads directly** via `_get_current_mru()` when needed
4. **OverlayManager reads directly** via `get_mru_window_list()`
5. All writes via `MRUManager.record()` or `_reorder_mru()`

**Key Changes**:
- ✅ Added `_get_current_mru()` helper - reads from MRUManager
- ✅ Added `_reorder_mru()` helper - writes to MRUManager  
- ✅ Updated 26+ locations to use helpers
- ✅ OverlayManager now reads/writes MRUManager for window HWND MRU
- ✅ Maintains separate overlay ID MRU for UI interaction tracking
- ❌ Removed 70+ lines of synchronization code

**Benefits**:
- No stale data (impossible by design)
- No synchronization bugs
- No race conditions
- Simpler, more reliable code
- Always works consistently

**See**: `audits/mru_single_source_of_truth_COMPLETE.md` for full details

### 2025-10-05: Window State Preservation Fix

**ISSUE**: Switching to fullscreen windows would cause them to exit fullscreen and shrink to normal window mode.

**ROOT CAUSE**: Aggressive focus method in `DockingManager._bring_hwnd_to_focus()` used `SW_SHOWNORMAL` which forces windows into normal mode, destroying fullscreen/maximized states.

**FIX**: Replaced with proven approach from `quickswitch_controller._focus_window()`:
- ✅ Only uses `SW_RESTORE` if window is minimized (preserves fullscreen/maximized)
- ❌ NEVER uses `SW_SHOWNORMAL` (removed entirely)
- ✅ Uses `SetWindowPos` with `SWP_SHOWWINDOW` flag instead (preserves state)
- ✅ Fallback strategies: simple → SetWindowPos sequence → thread attachment → Alt keystroke

**Result**: Fullscreen windows stay fullscreen, maximized windows stay maximized. Window state is never disrupted.

### 2025-10-05: Hide/Show All Overlays Hotkey

**FEATURE**: Added customizable global hotkey to toggle visibility of all overlays.

**Implementation**:
- New `HideShowController` in `core/hide_show_controller.py`
- Integrates with existing `OverlayStateManager` for state capture/restore
- No suppression (like other multipress hotkeys)
- Works with both single overlay and docking modes
- Registered automatically at application startup
- **Customizable**: Default Ctrl+Shift+H, adjustable in SubSettings dialog
- **Live updates**: Changes in SubSettings immediately update the hotkey

**Behavior**:
- First press: Hides all overlays and saves state
- Second press: Restores overlays from saved state
- Preserves window assignments, geometry, opacity, lock states
- **Docking mode**: Properly restores docking system (creates DockingOverlayManager if needed)

### 2025-10-05: Context Menu and Mode Switching Fixes

**ISSUE 1**: "Switch To Single Overlay" was greyed out in docking mode context menus.
- **ROOT CAUSE**: `switch_to_single_overlay` callback was never registered in docking overlay config.
- **FIX**: Added callback to actions dict in `overlay.py::_setup_context_menu_integration()`
- **IMPLEMENTATION**: New `_handle_switch_to_single_overlay()` method preserves current window when switching

**ISSUE 2**: Mode transitions from main dialog didn't destroy old overlays.
- **ROOT CAUSE**: Creation methods never destroyed existing overlays before creating new ones.
- **FIX**: Added `_destroy_all_existing_overlays()` method called before each creation
- **BEHAVIOR**: 
  - Destroys docking system if active
  - Closes all single overlays if any
  - Ensures clean transition between modes

**Result**: Mode switching now works correctly - old overlays are properly cleaned up before new ones are created.

### 2025-10-05: Mode-Aware Docking Implementation

- **MAJOR MODE-AWARE REFACTOR:** Completely separated Normal and Cycle mode behaviors in docking:
  - **Normal Mode**: Sticky window assignments - overlays maintain content unless explicitly changed
    - Added `_normal_mode_assignments` dict to track which window belongs in each overlay
    - `_update_normal_mode_displays()`: Only updates overlays with invalid/closed windows
    - **Double-click ANY overlay (A/B/C/D/E)**: Targeted swap - brings overlay's window to foreground
      - Overlay's current window → Goes to foreground (user wants to interact with it)
      - Next MRU[0] → Swaps into the overlay
      - This includes main overlay (A) - same targeted swap behavior
    - **Quickswitch hotkey**: Same as double-click on main overlay - rotates MRU and swaps
  - **Cycle Mode**: Dynamic MRU-based assignment (existing behavior preserved)
    - `_update_cycle_mode_displays()`: Updates all overlays based on MRU order (excluding foreground)
    - Double-click any overlay: Focus window, all overlays update
    - Quickswitch: Rotates MRU, all overlays update
  - **Mode Transitions**: Smooth switching via `_on_docking_mode_changed()` - captures current state as sticky baseline when switching Cycle→Normal
  - **Critical Quickswitch Fix (2025-10-06)**: Simplified `_rotate_mru_forward()` with targeted swap logic:
    - **Rotates MRU**: If current window == MRU[0], swap MRU[0] ↔ MRU[1]
    - **Focuses outgoing window**: The window being swapped OUT goes to foreground (targeted swap)
    - **Swaps in MRU[0]**: Main overlay displays new MRU[0] after rotation
    - Removed complex 3-case logic - simplified to single consistent behavior
  - **Helper Methods**: `_update_mru_order_only()`, `_assign_overlay()`, `_find_overlay_showing_window()`, `_normal_mode_swap_with_foreground()`

### 2025-09-18: Quickswitch Fixes

- **MAJOR QUICKSWITCH FIX:** Fixed broken `_rotate_mru_forward()` method in `DockingOverlayManager` that only worked once. Now properly implements Alt+Tab style MRU rotation:
  - If current focus == MRU[0]: swap MRU[0] ↔ MRU[1], then focus new MRU[0]
  - Otherwise: just focus MRU[0]
  - Enables consistent cycling behavior like classic Alt+Tab
- **Enhanced Secondary Focus:** Double-click on secondary overlays now properly focuses correct window with bounds checking and detailed logging
- **Improved Locked Overlay Handling:** `_bring_overlay_window_to_focus()` clarified for locked overlays only, simplified logic
- **Docking quickswitch handling (local):** `core/graphics/docking/manager.py::handle_overlay_interaction()` now maps `"quickswitch"` directly to `_handle_double_click("main")`, avoiding recursion with `QuickSwitchController` and stabilizing behavior when locks are enabled.
- **Validated MRU assignment:** `DockingOverlayManager._update_overlay_displays()` now filters candidates using `utils/window_validation.is_valid_window` to ensure only real top-level app HWNDs are assigned, preventing overlay host/thumbnail handles from being used.
- **ResourceManager UI single-writer:** `core/resources/manager.py::_enqueue_mutation_sync()` now dispatches off-UI mutations to the UI thread and synchronously awaits completion before publishing the snapshot. This enforces the single-writer policy for resource mutations.
- **HotkeyManager correctness fixes:** Initialized missing locks (`_hotkey_lock`, `_cmd_lock`), aligned SPSC unregister tuple and event signaling in the hotkey thread, and removed invalid `submit_to_pool` fallback. Message loop starts via `submit_task(ThreadPoolType.IO, ...)` only.
- **Z-order registration path:** Added `ResourceManager.register_overlay(overlay_id, main_widget, border_widget=None)` and guarded the deferred registration in `OverlayManager` with try/except to surface failures.

---

## Central Modules at a Glance

- **Threading**
  - `core/threading/manager.py`: `ThreadManager`, pools, and UI coalescers (`single_shot`, `run_on_ui_thread`).
- **Resource lifecycle**
  - `utils/resource_manager.py`: registers resources/overlays by type, deterministic cleanup, z-order registration helpers.
- **Settings**
  - `core/settings/settings_manager.py`: persistent settings (`docking.mode`, `docking.overlay_count`, `debug.docking_verbose`, etc.).
- **Window enumeration/validation**
  - `core/window/enumerator.py`: lists candidate windows for menus.
  - `utils/window_validation.py`: `is_valid_window(hwnd, our_pid)` gate used everywhere for filtering.
- **Overlays (window mode)**
  - `core/graphics/overlay_manager.py`: creates overlays, MRU of overlays (not windows), auto-switch registration.
- **Overlays (docking mode)**
  - `core/graphics/docking/manager.py`: coordinates main + N secondaries. Single positioning source. MRU-aware assignment. Cycle mode.
  - `core/graphics/docking/overlay.py`: wrapper per docking overlay (host configuration, interaction wiring, pass-through to DWM backend, focus indicator integration).
  - `core/graphics/docking/overlay_pool.py`: pooling of backends for reuse.
- **DWM backend**
  - `core/graphics/backends/dwm/integrated_dwm_backend.py` (and related): wraps DWM thumbnails, `update_source()`/`_handle_swap_window()`, host windows, canvas, focus indicator, and thumbnail attributes.
- **Switching**
  - `core/switching/mru_manager.py`: list of valid hwnds in most-recent-first order.
  - `core/switching/quickswitch_controller.py`: global hotkey registration and switching when NOT in docking mode. In docking mode it triggers docking’s handler.
  - `core/switching/autoswitch_controller.py`: suppressions/debouncing on source focus changes, auto-monitoring on close.

---

## Docking Mode: Lifecycle and Data Flow

Key orchestrator: `core/graphics/docking/manager.py` (`DockingOverlayManager`).

- **Creation** (`create_docking_system(target_hwnds)`).
  - Builds `DockingOverlay` instances: `main` and `secondary_i` overlays using the pool/backends.
  - Registers overlays with autoswitch monitoring and the `ResourceManager`.
  - Restores persisted main geometry if available (`docking.last_state`).
  - Binds hosts with event filters for group drag, coalesced sync, and interaction.
  - Marks active and schedules initial `sync_overlay_properties()`.

- **Positioning (Single Source of Truth)**
  - `sync_overlay_properties()` computes sizes and positions for secondaries relative to main.
  - Uses screen-aware inward/outward horizontal placement, bottom work-area clamp, and per-role size ratios.
  - Prevents feedback loops with a `_batch_applying` guard. UI-thread geometry set.

- **Persistence Architecture** (CRITICAL - 2025-10-09)
  - **On Create**: Loads `docking.last_state` with physical dimensions (e.g., 1102x630)
  - **Conversion**: Converts physical → logical using DPR from target monitor (e.g., ÷1.5 → 735x420)
  - **Application**: Applies geometry before showing overlays
  - **Delayed Sync**: 100ms delay after overlays shown to allow stabilization
  - **On Resize/Move**: Saves geometry as physical pixels with 2s suppression window
  - **Protection Mechanisms**:
    - `_batch_applying`: Prevents feedback during sync
    - `_is_initializing`: Blocks syncs during setup
    - Unidirectional sizing: Main → Secondaries only (secondaries NEVER resize main)
    - Single sync source: Delayed initial sync only (no immediate sync after restoration)
  - See `audits/docking_persistence_complete_call_chain_analysis.md` for complete fix details

- **Display Assignment (What each overlay shows) - MODE-AWARE**
  - `_update_overlay_displays()` dispatches to mode-specific methods:
    - **Normal Mode** (`_update_normal_mode_displays()`):
      - **Sticky assignments**: Overlays maintain their assigned windows
      - Tracks assignments in `_normal_mode_assignments` dict: `{"main": hwnd, "secondary_0": hwnd, ...}`
      - Only updates when:
        1. Window becomes invalid/closed
        2. Overlay has no assignment yet (initialization)
        3. Assignment is duplicated (conflict resolution)
        4. Desynced from tracking
      - Respects locks
      - **Does NOT reassign on every MRU change** (key difference from Cycle)
    - **Cycle Mode** (`_update_cycle_mode_displays()`):
      - **Dynamic assignments**: Overlays update based on MRU order
      - Excludes foreground window (shows next N from MRU)
      - Updates all overlays on every MRU change
      - Can fade changed overlays (~200ms)
      - Respects locks
  - Both modes filter MRU using `utils.window_validation.is_valid_window` to prevent overlay-host/thumbnail handles from being assigned.

- **Swapping/Quickswitch in Docking - MODE-AWARE**
  - `handle_overlay_interaction("quickswitch")` maps to `_handle_double_click("main")` locally (no recursion back to `QuickSwitchController`).
  - **`_rotate_mru_forward()` - Mode-Aware MRU Rotation**:
    - Implements proper Alt+Tab behavior: swaps MRU[0] ↔ MRU[1] when current focus is MRU[0]
    - **Normal Mode**: Enhanced 3-case logic handles foreground in any overlay:
      - **Case 1**: Foreground in Overlay A (or not in any overlay)
        - Simple: Update A with next MRU item
      - **Case 2**: Foreground in B/C/D/E (e.g., user working in Overlay C)
        - Move foreground to Overlay A
        - Focus window from Overlay A
        - Replace source overlay (C) with next available window from MRU
        - Prevents duplicates; allows duplicate as last resort if insufficient windows
      - **Case 3**: External foreground (not in any overlay)
        - Update A with next MRU item
    - **Cycle Mode**: Focus next MRU item, all overlays update dynamically
  - **Double-Click Behavior**:
    - **Normal Mode**:
      - **ALL overlays (A/B/C/D/E)**: Targeted swap - same consistent behavior
        - Overlay's current window → Goes to foreground (user wants to interact with it)
        - Next MRU[0] (after rotation) → Swaps into the overlay
        - Other overlays unchanged
        - Implementation: Main uses `_rotate_mru_forward()`, secondaries use `_normal_mode_swap_with_foreground()`
    - **Cycle Mode**:
      - Any overlay: Focus that overlay's window via `_focus_and_promote_to_mru()`
      - All overlays update based on new MRU order
  - **Helper Methods**:
    - `_find_overlay_showing_window(hwnd)`: Detects which overlay (if any) shows a given window
    - `_update_mru_order_only(hwnd)`: Updates MRU without triggering overlay reassignment (Normal mode)
    - `_assign_overlay(overlay_id, hwnd)`: Assigns window to overlay with tracking
    - `_normal_mode_swap_with_foreground(overlay_id)`: Swaps overlay with foreground
  - **Enhanced Debugging**: Comprehensive logging with QUICKSWITCH, SECONDARY_FOCUS, NORMAL_SWAP, LOCKED_FOCUS prefixes
  - Swap implementation for sources follows DWM pattern: `_swap_main_source_hwnd()` / `_swap_secondary_source_hwnd()` with validation, MRU sync, and coalesced re-layout.

- **Locking**
  - Lock on an overlay preserves its target during automatic updates (assignment skips locked overlays and marks the hwnd as assigned to prevent duplication).

- **Persistence**
  - Persists main overlay nearest-corner geometry on movement. Restores on launch.

---

## Window Mode (Non-docking)

- `core/graphics/overlay_manager.py`
  - Creates/Destroys overlays (DWM/Monitor/Software) and maintains an MRU list of overlay IDs.
  - Exposes `get_mru_window_list(limit)` for docking to consume (maps overlay MRU -> hwnd MRU).
  - Registers overlay windows with `AutoswitchController`.

---

## Switching Layer

- **MRU** (`core/switching/mru_manager.py`)
  - Ensures only valid hwnds are kept. Notifies listeners on change (used by docking cycle mode).
- **Quickswitch** (`core/switching/quickswitch_controller.py`)
  - Registers a global hotkey. If docking mode is active, it delegates to the docking manager’s quickswitch handler.
  - Otherwise, it does the overlay-mode swap (foreground vs MRU candidates) and focuses the appropriate window.
- **Foreground Autoswitch** (`core/switching/autoswitch_controller.py::ForegroundAutoswitchController`)
  - Monitors foreground window focus changes via FocusTracker (200ms polling)
  - When user focuses a window displayed in an overlay, triggers swap for that overlay
  - Docking: Checks all overlays A/B/C/D/E and swaps matching ones
  - Single: Standard overlay swap behavior
  - Suppresses noisy focus transitions and debounces rapid changes
- **Closed-Window Switching** (`core/graphics/window_monitor.py::ClosedWindowSwitchManager`)
  - Monitors registered overlay windows for closure (2s polling via WindowMonitor)
  - When a window closes, automatically switches that overlay to next MRU window
  - Respects lock states and prevents duplicates
  - Works for both single overlays and docking overlays

---

## Hotkeys Strategy (Quickswitch vs Global)

- **Ownership**
  - `QuickSwitchController` owns the quickswitch combo (keyboard library), gates cooldown and locking, and delegates to docking when active.
  - `HotkeyManager` owns all other global/system hotkeys (RegisterHotKey and keyboard fallback), with a dedicated message loop on a ThreadManager IO task.

- **Centralization decision (Option A retained)**
  - We keep quickswitch registration in `QuickSwitchController` to avoid duplicating docking/window-mode selection logic in two places.
  - In docking mode, the controller calls `DockingOverlayManager.handle_overlay_interaction("main", "quickswitch")` and returns (no recursion back to controller).

- **Option B (deferred)**
  - If future work centralizes quickswitch under `HotkeyManager`, extract a small pure orchestrator (no Qt) for selection/decision so the manager and controller don’t diverge. For now, Option B is not implemented due to anti-duplication risk.

---

## Validation Policy (critical to correctness)

- `utils/window_validation.is_valid_window(hwnd, our_pid)` filters out:
  - Our own process windows (overlay hosts, canvases, indicator windows, etc.).
  - System/special windows (taskbar, shell experiences, tool windows, disabled windows).
  - Invisible or zero-size windows.
- This function is the authoritative gate before any assignment or MRU record.

---

## Threading and Events

- **UI thread first**
  - All overlay mutation happens on UI thread; use `ThreadManager.run_on_ui_thread` or `single_shot`.
- **Coalescing**
  - Docking uses `_coalesced_sync(delay_ms)` to debounce resize/move bursts and avoid layout thrash.
- **Event filters**
  - Docking installs filters on hosts to translate group drags, scale via wheel, and sync after interactions without feedback loops.

---

## Policies and Practices (must-know)

- **Single positioning system in docking**
  - Only `DockingOverlayManager.sync_overlay_properties()` is allowed to position secondaries. No duplicate systems.
- **Bottom work-area clamp**
  - Overlays are aligned against `availableGeometry` bottom to avoid taskbar overlap.
- **Logging-first**
  - Prefer logs over terminal output; add concise logs around critical transitions (swap, assignment, coalesce, persistence).
- **Windows-specific behaviors**
  - Quits, focus changes, and DWM attribute support vary; errors are logged but not fatal.

---

## Typical Debugging Recipes

- **Docking shows wrong/duplicate windows**
  - **First**: Check which mode is active (Normal vs Cycle)
  - **Normal Mode**: Check `_update_normal_mode_displays()` logs. Look for "NORMAL:" prefixed messages. Verify `_normal_mode_assignments` dict has expected values.
  - **Cycle Mode**: Check `_update_cycle_mode_displays()` logs. Ensure `validated_mru` has enough candidates and foreground is properly excluded.
  - Confirm `is_valid_window` would accept the expected hwnd; verify the window is visible and not our own process.
  - Check for desync: overlay's actual content vs tracked assignment in Normal mode.

- **Normal mode overlays changing unexpectedly**
  - **Symptom**: Secondary overlays (B/C/D/E) change content when they shouldn't
  - **Cause**: Mode might be Cycle, or `_update_normal_mode_displays()` not being called
  - **Check**: `_update_overlay_displays()` should dispatch to `_update_normal_mode_displays()` in Normal mode
  - **Look for**: Log messages with "NORMAL:" prefix showing why overlay was updated
  - **Verify**: `_normal_mode_assignments` dict maintains expected values across operations

- **Quickswitch flaky in docking / foreground window disappears from overlays**
  - **CHECK FIRST**: Ensure `_rotate_mru_forward()` properly swaps MRU[0] ↔ MRU[1] when current focus equals MRU[0]
  - **Normal Mode Specific**: Check for "QUICKSWITCH (Normal):" logs showing 3-case logic:
    - Case 1: "Main overlay updated to X"
    - Case 2: "Moved fg X to A, replaced secondary_Y with Z"
    - Verify `_find_overlay_showing_window()` correctly detects which overlay has foreground
  - **Cycle Mode**: Should see "QUICKSWITCH: Rotating from X to Y" followed by all overlays updating
  - Verify that docking `handle_overlay_interaction("quickswitch")` maps to `_handle_double_click("main")` (no recursion with `QuickSwitchController`).
  - Look for cooldown suppressions from the centralized controller; they should no longer apply in docking.

- **Double-click on secondary overlay doesn't work as expected**
  - **Normal Mode**: Should see "NORMAL_SWAP:" log showing swap between overlay and foreground
  - **Cycle Mode**: Should see all overlays updating after focus change
  - Check which mode is active and verify mode-specific logic is executing

- **Overlays stack or misalign**
  - Ensure only `sync_overlay_properties()` sets secondary geometry and `_batch_applying` guards are respected.
  - Watch for inward/outward logs to confirm boundary detection.

- **Thumbnail invalidation / "Invalid source window: 0/…**"
  - Check that swaps always pass `is_valid_window` and docking reset paths don't change source hwnds unexpectedly.

---

## Glossary

- **Overlay**: A window created by the app to display content (DWM thumbnail or otherwise).
- **Docking**: Multi-overlay layout: main + secondaries bound together.
- **Normal Mode** (Docking): Sticky window assignments - overlays maintain their content unless explicitly changed by user action or window closure. Enables stable multi-window workflow.
- **Cycle Mode** (Docking): Dynamic MRU-based assignments - overlays automatically update to show top N most recently used windows (excluding foreground). Updates continuously as MRU changes.
- **MRU**: Most Recently Used window list (hwnds).
- **Swap**: Changing an overlay's source window (DWM thumbnail target).
- **Host**: The Qt window hosting the overlay's canvas/focus indicator.
- **Sticky Assignment**: In Normal mode, the persistent mapping between an overlay and its assigned window, tracked in `_normal_mode_assignments`.

---

## Pointers to Key Code

- Docking manager: `core/graphics/docking/manager.py`
- Docking overlay: `core/graphics/docking/overlay.py`
- Overlay manager (window mode): `core/graphics/overlay_manager.py`
- MRU manager: `core/switching/mru_manager.py`
- Quickswitch controller: `core/switching/quickswitch_controller.py`
- Window validation: `utils/window_validation.py`
- Thread manager: `core/threading/manager.py`
- Resource manager: `utils/resource_manager.py`

---

## Maintenance Tips

- Keep `Index.md` focused on file map. Keep `Spec.md` focused on policies and architecture contracts.
- Update this guide when:
  - A cross-cutting behavior changes (e.g., docking quickswitch mapping, MRU validation rules).
  - New centralized modules are introduced.
  - Positioning or switching algorithms materially change.
