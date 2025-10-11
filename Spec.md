# SPQModular – Project Specification

*This specification defines the architecture, policies, and implementation details for the SPQModular overlay application.*

---

## 🎯 Purpose & Overview

SPQModular is a hardware-accelerated overlay application that provides real-time display of screen content and system metrics on top of other windows. The application features a resizable, draggable overlay with support for multiple monitors, DPI awareness, and a comprehensive theming system. It's designed with a strict no-fallback architecture, ensuring either complete functionality or explicit failure with clear logging and optional deferred retries.

## ✨ Key Features

- **Hardware-Accelerated Rendering**: DXGI-only capture via dxcam (CPU-copy frames). No WGC. No swapchain presenter.
- **Multi-Monitor Support**: Full support for multiple monitor configurations with independent settings
- **DPI Awareness**: Proper handling of high-DPI displays and scaling
- **Docking Mode**: Multi-overlay system with three synchronized overlays (main 100%, secondary 70%/50% size ratios)
  - **CENTRALIZED ARCHITECTURE POLICY**: All MRU management, switching, and app instance injection must use centralized core modules
  - **MRU SINGLE SOURCE OF TRUTH** (CRITICAL - see `audits/mru_single_source_of_truth_COMPLETE.md`):
    - **POLICY**: `MRUManager` is the ONLY storage for window HWND MRU - no local caches permitted
    - **DockingManager**: Reads via `_get_current_mru()` helper, writes via `MRUManager.record()` or `_reorder_mru()`
    - **OverlayManager**: Reads via `get_mru_window_list()`, writes via `MRUManager.record()`
    - **NO LISTENERS**: Eliminated push/callback system - pull-based on demand only
    - **NO SYNCHRONIZATION**: No sync logic needed - impossible to have stale data
    - **NO LOCAL LISTS**: Removed `_mru_list`, `_on_mru_changed()`, suppression flags
    - **BENEFITS**: No stale data, no race conditions, no sync bugs, always consistent
    - **ARCHITECTURE**: `FocusTracker` → `MRUManager` (write) ← `DockingManager` + `OverlayManager` (read)
  - **Switching Integration**: Delegates all quickswitch operations to `core.switching.quickswitch_controller.QuickSwitchController`
  - **HIDE/SHOW INTEGRATION POLICY** (CRITICAL):
    - **SINGLE IMPLEMENTATION**: `OverlayStateManager` is the ONLY hide/show implementation
    - **UNIFIED USAGE**: Context menu, tray menu, and Ctrl+Shift+H hotkey ALL use `OverlayStateManager`
    - **NO SIMPLE .hide()**: Never use simple `.hide()` calls - always use state manager for proper capture/restore
    - **STATE CAPTURE**: Captures geometry, opacity, locks, window assignments, Normal mode assignments
    - **STATE RESTORATION**: Full restoration including window validation and MRU seeding
    - **COMPONENTS**:
      - `HideShowController`: Owns Ctrl+Shift+H hotkey, delegates to `OverlayStateManager`
      - `DockingManager.hide_all_overlays()`: Calls `OverlayStateManager.hide_all_overlays()`
      - `TrayManager._restore_hidden_overlays()`: Calls `OverlayStateManager.show_all_overlays()`
      - Context menu: Routes to manager which uses state manager
  - **MODE SWITCHING POLICY** (CRITICAL):
    - **CLEAN TRANSITIONS**: All mode transitions must destroy existing overlays before creating new ones
    - **IMPLEMENTATION**: `MainDialog._destroy_all_existing_overlays()` called before each creation
    - **DESTRUCTION ORDER**:
      1. Check and destroy active docking system via `DockingOverlayManager.destroy_docking_system()`
      2. Close all single overlays via `OverlayManager.get_all_overlays()` and `overlay.close()`
    - **MODES**: Window → Monitor, Window → Docking, Monitor → Docking, etc. all follow this pattern
    - **RESULT**: No orphaned overlays, clean resource management, proper state transitions
  - **CONTEXT MENU INTEGRATION POLICY**:
    - **SWITCH TO SINGLE OVERLAY**: Enabled in docking mode context menus
    - **IMPLEMENTATION**: `_handle_switch_to_single_overlay()` in `DockingOverlay`
    - **BEHAVIOR**: Preserves current main overlay's window when switching to single mode
    - **VALIDATION**: Validates preserved window is still valid before creating single overlay
    - **CALLBACK REGISTRATION**: Must be in docking config `actions` dict for context menu visibility
  - **App Instance Injection**: Uses centralized provider pattern from main overlay manager for context menu functionality
  - **Debug Log Suppression**: Uses `utils.debug.log_suppressor` to reduce excessive debug spam while preserving error/warning visibility
  - **Screen-Aware Positioning**: Automatic inward/outward positioning based on screen boundaries with proper bounds validation to prevent overlay C overlap/shifting during resize
  - **Vertical Alignment Policy**: Secondary overlays align to main overlay's top or bottom edge (adaptive based on nearest corner). All overlays (main and secondaries) can extend below taskbar boundary and rely on Z-order enforcement to stay visible above taskbar. Simple screen bounds check prevents extending beyond physical display edges.
  - **Z-Order Enforcement Policy** (CRITICAL - see `audits/docking_z_order_investigation.md`):
    - All overlays (main and secondary) use `Qt.WindowStaysOnTopHint` flag
    - All overlays register with unified `ZOrderManager` on creation
    - CRITICAL: Explicit `ZOrderPriority.CRITICAL` enforcement after `show()` calls
    - Immediate `SetWindowPos(HWND_TOPMOST)` bypasses debouncing to prevent taskbar coverage
    - Without explicit enforcement, debounced z-order allows overlays to appear below taskbar
    - Main overlay enforced once after show, secondaries enforced in loop
  - **Single Positioning Source (Policy)**: The only positioning logic lives in `core/graphics/docking/manager.py::sync_overlay_properties()`. All legacy positioners are removed/ignored. The method emits `FIT` and `POSITION` diagnostics and performs min-size-aware fit scaling using `utils/window/overlay_constants.py` floors
  - **Natural Bottom Alignment**: The `aligned_y()` helper within `sync_overlay_properties()` aligns secondaries directly to main's top or bottom edge without artificial taskbar clamping. All overlays (main + secondaries) benefit from identical Z-order enforcement that keeps them visible above the taskbar, allowing perfect bottom-alignment even when dragged low. Only physical screen bounds are enforced.
  - **Uniform Horizontal Offset**: In outward positioning mode, screen overflow offset is applied uniformly to all secondaries by tracking the nominal cursor separately from the offset-adjusted positions. This prevents cumulative gaps that would otherwise appear at minimum sizes (especially overlay E). The offset shifts the entire group as a unit.
  - **Deferred AR Correction Disabled**: Docking overlays skip the 15ms deferred `_handle_correct_aspect()` that runs after DWM initialization. The `DockingOverlayManager` has exclusive control over docking overlay geometry through `sync_overlay_properties()`, ensuring persisted bottom-corner geometry isn't overridden on restore.
  - **Tight Content Framing (Docking)**: OUTER geometry for docking overlays is computed to tightly hug INNER content plus border/inner-accent margins. Secondary widths include canvas content insets and cached AR so creation/recreation/"Correct AR" do not exhibit side gaps. Docking overlays skip the extra DWM-level thumbnail inset (the canvas `content_rect()` already accounts for borders/accents) to avoid double-insetting.
  - **Strict Secondary Size Hierarchy**: Overlay sizes strictly decrease A > B > C > D > E until min/max floors. Baseline: B≈70% of A; each further secondary decays by ~15% (floors enforced by `OVERLAY_MIN_WIDTH/HEIGHT`). Post-fit enforcement adjusts ties down by 1px and recomputes widths from AR+insets.
  - **Deferred AR Correction**: After DWM overlay initialization, a small deferred `_handle_correct_aspect()` (~15ms via `ThreadManager.single_shot`) eliminates minor initial gaps without user interaction.
  - Interactive overlay switching via centralized controllers (main = delegate to centralized MRU rotation; secondary = centralized focus management)
  - **MRU Minimized Window Support**: MRU validation skips visibility checks during quickswitch retrieval (`check_visible=False`), allowing minimized/hidden windows to remain valid quickswitch targets. Only destroyed/invalid windows are filtered out.
  - **Hide/Show Hotkey (Ctrl+Shift+H)**: Toggles visibility of all overlays via `HideShowController`. Gated behind `hotkeys.hide_show_enabled` setting (default: False) with CircleCheckBox toggle in subsettings. Uses `OverlayStateManager` for state persistence and restoration. Both controllers registered with ResourceManager for proper cleanup. All Qt operations (overlay creation/destruction) run via `ThreadManager.run_in_main_thread()` to prevent cross-thread violations. State manager uses `find_resource_by_description()` to locate overlay managers in ResourceManager.
  - **HotkeyManager Singleton Pattern**: All controllers (`HideShowController`, `OpacityManager`) use `HotkeyManager()` directly from `core/hotkeys/manager.py`. The class implements singleton via `__new__()`, ensuring all instances share the same hotkey registration state. This prevents duplicate Windows hotkey registration conflicts. The deprecated `core/hotkeys/compat.py` compatibility layer has been removed.
  - Lock-aware automatic switching via centralized switching controllers with proper delegation patterns
  - Comprehensive resource management with ResourceManager integration and proper cleanup handlers
  - Secondary overlays apply direct geometry with screen bounds validation to prevent overlap and positioning conflicts
  - Movement binding uses event filter with proper host availability checking and delayed retry mechanisms
  - **Batch Geometry Guard**: `_batch_applying` suppresses host Move/Resize reactions during group geometry application to eliminate feedback loops and teleporting; cleared after scheduled UI tasks complete
  - **Secondary Drag Delegation**: Secondary overlay hosts do not start local drags. `DockingOverlayManager.eventFilter()` translates the main overlay by global mouse deltas during a secondary-held drag, then coalesces a sync to reposition secondaries (smooth group dragging)
  - **Unified Wheel-Resize**: Wheel on secondary overlays is intercepted to scale the main overlay; prevents transient self-resize distortion on secondaries and keeps the dock cohesive
  - **Geometry Persistence Architecture** (CRITICAL - 2025-10-09):
    - Saves docking main overlay geometry in physical pixels to `docking.last_state`
    - Uses nearest-corner state persistence (see `utils/window/overlay_persistence.py`)
    - Converts logical ↔ physical pixels with proper DPR scaling
    - Suppresses saves for 2 seconds after restoration to prevent spurious updates
    - Single sync source: delayed initial sync (100ms after overlays shown)
    - **Anti-Cascade Architecture**:
      - Bidirectional sizing eliminated - secondaries NEVER resize main
      - Batch mode (`_batch_applying`) blocks resize events during sync
      - Initialization mode (`_is_initializing`) blocks all syncs until stable
      - Coalesced syncs check both flags before scheduling
    - **Result**: Main overlay geometry persists correctly across restarts without resize cascades
    - See `audits/docking_persistence_complete_call_chain_analysis.md` for implementation details
  - **MRU-Aware Autoswitch Logic**: Overlays A/B/C maintain MRU ordering with duplicate prevention and lock-aware cycling
    - **Overlay Assignment**: A=MRU[0], B=MRU[1], C=MRU[2] from centralized MRU manager
    - **Duplicate Prevention**: When autoswitch assigns window X to overlay A, if B or C already has X, they swap content to maintain uniqueness
    - **Priority Resolution**: If multiple overlays would receive the same window, A takes priority and B/C shift to next available MRU entries
    - **Cycling Logic**: When A returns to same window (e.g., 1001→1001), push A→B, B→C, and assign most recent valid MRU to A
    - **Lock Awareness**: Locked overlays skip content changes but system still prevents duplicates across unlocked overlays
    - **Fallback Population**: When MRU has <3 entries, enumerate visible windows via WindowEnumerator to ensure B/C have valid targets
- **Robust Aspect Ratio Management**: 
  - DWM overlay uses sophisticated client area and window rect fallback logic
  - Monitor overlay inherits DWM aspect ratio caching for consistent scaling
  - Bounds checking (0.2-5.0 ratio limits) prevents extreme distortions
  - DPI-aware thumbnail property scaling for proper rendering

#### DWM Overlay Geometry Persistence (Spec)

- Apply persisted geometry on show: On `_show_impl()`, standalone DWM overlays attempt to restore the last saved nearest-corner geometry. On success, an INFO log "Applied persisted DWM geometry" is emitted for diagnostics.
- Persist geometry on interaction and hide: Geometry is saved on hide as before, and also opportunistically after drag/resize release and wheel-apply via `WindowBehaviorManager` calling `_persist_current_geometry()` when exposed by the overlay host.
- Benign DWM errors: `DwmUpdateThumbnailProperties` E_INVALIDARG during hide or with degenerate rectangles is downgraded to DEBUG to reduce noise; ERROR severity is retained otherwise.

##### Troubleshooting: Verifying Persistence Apply

- DWM apply success emits: `[DWM_PERSIST] Applied persisted DWM geometry` in the standard app log (`logs/app_*.log` in runtime; repo dev: `app.log`).
- Docking apply success emits: `[DOCK_PERSIST] Applied persisted docking main geometry`.
- If restore does not apply:
  - Confirm keys exist: DWM uses `overlays.dwm.last_state`, Docking uses `docking.last_state`.
  - Check monitor topology changes; geometry is clamped via `ensure_within_available_desktop(...)`.
  - Check canvas insets and aspect calculations for DWM (source window rect availability).

#### DWM Capture Mode Policy (Spec)

- **Full Window Capture** (2025-10-11): DWM thumbnails use `source_client_area_only=False` to capture the entire window including title bar and borders
- **Rationale**: Modern Windows apps (Windows 11 Task Manager, Notepad, Settings, etc.) use **DWM Extended Frame** technique where the title bar is rendered inside the client area via `DwmExtendFrameIntoClientArea()`. When `source_client_area_only=True`, DWM incorrectly calculates an offset that skips valid content, causing black areas at the top of these apps
- **Behavior**: All windows (modern and traditional) are captured with their title bars visible, matching Windows taskbar preview behavior
- **Aspect Ratio Independence**: This setting does NOT affect aspect ratio calculations - AR is always calculated from `GetClientRect()` on the source window regardless of capture mode
- **Implementation**: `core/graphics/backends/dwm/integrated_dwm_backend.py` lines 531 and 569 set `source_client_area_only=False` in all `update_thumbnail()` calls
- **Impact**: Traditional apps (Chrome, Firefox, etc.) now show title bars in overlays instead of content-only. This is acceptable as it provides consistent behavior across all window types and matches user expectations from taskbar previews
- **Reference**: See `audits/dwm_modern_windows_apps_black_titlebar_research.md` for detailed technical analysis
- **Comprehensive Theming System**: 
  - Support for light/dark themes with custom styling
  - Structured QSS organization for maintainability
  - Modular theme management with centralized validation
  - Runtime theme switching with performance optimization
  - Theme event system for coordinated updates
  - Required token validation and registration
  - Consistent component styling via ThemeConsumer interface
  - Efficient resource caching for icons, colors, and stylesheets
  - Adjustable title bar icon size through QSS properties
- **Window Management**: Comprehensive window handling and manipulation
- **Event System**: Type-safe event handling with priority-based dispatching and optional UI-thread callback routing

---

## 🏗️ Architectural Principles

### Core Policies

- **Strict No-Fallback**: Either complete functionality or explicit failure with clear logging; deferred initialization with retries is acceptable
- **Window Type Handling**: Unknown window types are logged and raise `ValueError`; no fallback to `WindowType.CUSTOM`
- **Resource Management**: All managers register with ResourceManager using lambda-wrapped cleanup handlers (`cleanup_handler=lambda obj: obj._cleanup()`) for consistent parameter signatures
- **Docking Mode Integration**: Controllers must check for active docking manager via ResourceManager and delegate appropriately. Fresh lookups per operation ensure proper routing without reliance on cached state
- **Debug Logging Policy**: Debug logging disabled by default; enable via `SPQ_DEBUG` environment variable (`1`, `true`, `yes`, or `on`). All debug logs gated behind `utils.debug.debug_enabled`. Frequent logs throttled; `BorderRenderer` employs per-path rate limiting. `CursorManager` short-circuits identical cursor requests and dedupes logs to avoid ArrowCursor spam during drag/hover cycles
  - KeyPassthrough verbose decision logs: settings key `debug.keypassthrough_verbose` enables detailed non-media decision logging (target validation, early return reasons, routing results). Default: false.
  - EventSystem dispatch tracing: settings key `debug.events_trace` enables lightweight begin/end trace logs with handler identity, priority, duration, and handled flag. Default: false.
- **TODO Comment Policy** (CRITICAL - see `audits/todo_fixme_evaluation_2025_10_06.md`):
  - **Prohibition**: TODO/FIXME/XXX/HACK comments are **strongly discouraged** in production code
  - **Issue Tracking Required**: All work items must be tracked in external issue tracker, not inline comments
  - **Allowed Exceptions** (with strict requirements):
    - **Temporary TODOs**: Must include removal deadline (max 30 days): `# TODO(Remove by 2025-11-06): Workaround for Qt 6.5 bug #12345`
    - **Issue-Linked TODOs**: Must reference specific issue: `# TODO(#456): Add GPU acceleration for large overlays`
    - **Both preferred**: `# TODO(#456, Remove by 2025-11-06): Temporary fix until upstream merge`
  - **Monthly Audit**: All TODO comments reviewed monthly; expired/stale comments must be implemented or removed
  - **Rationale**: 
    - Inline TODOs become obsolete and conflict with architecture (see 2025-10-06 audit)
    - Features described in TODOs may already be implemented elsewhere
    - Commented-out "stub" code misleads developers about implementation status
    - Untracked work items create hidden technical debt
  - **Enforcement**: PR reviews must reject TODOs without issue numbers or deadlines
  - **Violation Example**: `# TODO: Implement position manager integration` ← REJECTED (no issue, no deadline, feature already exists)
  - **Valid Example**: `# TODO(#789, 2025-11-15): Remove legacy hotkey compat after migration complete`

#### KeyPassthrough Blocklist (Spec)

- **Ownership**: Settings layer owns the blocklist file; it is read-only at runtime for controllers.
- **Location**: Same directory as the active `settings.json`. In repo builds: `settings/keypassthrough_blocklist.txt`. Fallback: `~/.spqmodular/keypassthrough_blocklist.txt`.
- **Creation**: `SettingsManager` auto-creates the blocklist on first run if missing with conservative defaults and logs the outcome with the `[KEYPASS]` prefix.
- **Encoding**: UTF-8 (LF newlines tolerated); comments start with `#`.
- **Format**: One rule per line. Plain lines ending in `.exe` match process names case-insensitively; other plain lines match window title contains (ALL terms). JSON-object-per-line overrides supported: `{ "exe": "..." }`, `{ "title_exact": "..." }`, `{ "title_contains": "..." }`.
- **Defaults**: ~24+ popular anti-cheat game executables (process-name specific; conservative to avoid false positives). The file is not overwritten once created.
- **Public Settings Directory**: `SettingsManager.get_settings_dir()` exposes the resolved settings directory for helpers (used by the blocklist loader to locate the file).
- **Centralized Loader**: `core/input/keypassthrough_blocklist.py` provides a singleton loader via `get_blocklist()`. The loader parses the file, caches rules, and offers matching APIs:
  - `match(exe_name: Optional[str], title: Optional[str]) -> Optional[dict]`
  - `match_for_hwnd(hwnd: int) -> Optional[dict]` (Windows-only; fetches exe/title and delegates to `match`)
- **Caching & Reload**: Parsed rules are cached with source mtime. Reload checks are throttled (~2s) and only performed when invoked. Parse errors are logged and skipped; feature operation is never blocked by loader failures.
- **Controller Integration**: `KeyPassthroughController.passthrough_key(vk)` performs an early blocklist check when `features.keypassthrough_blocklist_enabled` is true. On match, the key is not forwarded; `_block("blocklist", extra=match_info)` is invoked.
- **Blocked Event Payload**: `_block(...)` publishes `key.passthrough.blocked` with payload `{ "reason": "blocklist", "extra": { "type", "value", "exe", "title" } }`. Additional fields may be included by the loader for diagnostics. UI feedback triggers a brief black flash on the active overlay's focus indicator.

#### KeyPassthrough Feedback Throttling (Spec)

- UI flash feedback for blocked passthrough attempts is throttled and deduplicated.
- Setting key: `ui.block_flash_min_interval_ms` (int). Default: 250.
- Behavior: `_block(reason, extra)` only triggers `OverlayHost.flash_focus_indicator(...)` when either the reason has changed since the last flash or at least `ui.block_flash_min_interval_ms` has elapsed.

### Architectural Rules (Cross-cutting)

- **Lock-free Concurrency Model**: All utility modules use UI thread confinement for state mutations (completed migration 2024-12-28)
  - `ZOrderManager`, `CursorManager`, `MouseCaptureCoordinator` - all migrated to lock-free design
  - Cross-thread calls dispatched via `ThreadManager.run_on_ui_thread()`
  - No explicit locks (RLock, Lock) in utility modules
- **No Backwards-Compatibility Layers**: Do not add legacy shims, dual paths, or compatibility wrappers. Remove redundant/legacy code during refactors
- **No Duplication**: Centralize shared logic (e.g., `ThreadManager`, `ZOrderManager`, `ThemeManager`, `OverlayManager`). Extract shared behavior once and reuse
- **No Code Bloat**: Prefer standardized interfaces, single sources of truth, and small composable helpers over ad-hoc per-backend logic
- **Restructure over Mitigate**: For persistent/systemic issues, favor architectural restructuring over debouncing or conditional guards
- **Canonical Integrations Only**: New features must integrate with centralized modules and policies (no raw `QTimer`, no direct native calls bypassing managers)
#### Windows Messaging Policy

- Prefer non-blocking `PostMessage` for standard key forwarding to avoid UI thread stalls.
- When synchronous messaging is required with time bounds, use `utils.win.winmsg.safe_send_message(hwnd, msg, wparam, lparam, timeout_ms=250)`.
  - Performs responsiveness checks and enforces a timeout.
  - Do not call raw `SendMessage` directly.
  - `safe_send_appcommand` remains the helper for APPCOMMAND-based media operations.

#### Media Routing & Volume Hold (Spec)

- **Focused-only Media Routing (Policy)**
  - Media keys (`VK_MEDIA_*`) and system volume keys (`VK_VOLUME_*`) are only routed when an overlay is focused.
  - When unfocused, these keys are blocked by `KeyPassthroughController` to avoid unintended global actions and to minimize processing.
  - Focus checks failing due to transient state are treated conservatively (fail closed) and emit `key.passthrough.blocked` with `*-focus-check-failed` reasons.

- **Ownership of Continuous Volume Adjustment**
  - `MediaController` exclusively owns the hold loop. `KeyPassthroughController` performs one immediate step and delegates hold to `MediaController`.
  - Implementation uses `ThreadManager.single_shot` with token guards (no raw `QTimer`). A hard cap (`_volume_hold_max_seconds`) prevents runaways.
  - Release semantics are robust: `KeyPassthroughController.release_passthrough_key()` calls `MediaController.handle_volume_key_release(hwnd)` when a target exists, or `MediaController.stop_all_continuous_volume_adjustments()` if the target hwnd is missing (handoff/clear) to avoid ghost loops.

- **Auto-repeat Guard (Qt)**
  - `OverlayHost` ignores auto-repeat `KeyPress` and `KeyRelease` for `VK_VOLUME_UP/DOWN` and `VK_UP/DOWN`.
  - Only the physical first press starts the hold; only the final physical key-up ends it. This prevents premature stop due to Qt auto-repeat release events.

- **Timing & Settings (Single Source of Truth)**
  - `MediaController` reads timing from settings:
    - `input.volume_hold_initial_delay_ms` (default 200)
    - `input.volume_hold_interval_ms` (default 50)
  - Progressive step sizing during continuous mode (session-volume): ~1% initially, scaling to ~2% after ~1s.

- **Active Audio Session Requirement (Behavioral Quirk)**
  - Continuous adjustment uses per-app session volume. If the target application has no active audio session (e.g., currently silent), continuous steps may no-op until a session exists.
  - The initial immediate step is still applied via the media command path; subsequent continuous steps require a session. This is acceptable by policy and may be addressed by the Media Keepalive subsystem if desired.

#### Media Keepalive Integration (Spec)

- **Subtle Activation (no focus steal)**: `core/media/keepalive.py#MediaPlayerKeepAlive.request_subtle_activation(hwnd, app_name=None)` exposes a public API to nudge background media windows without activating them. Attempts are rate-limited per HWND and respect z-order/focus policies.
- **Enhanced Browser Command Retry**: `core/media/media_controller.py#MediaController._send_browser_media_command_enhanced(hwnd, command, app_name)` wraps the child-window APPCOMMAND path with a retry after subtle activation for browsers (Chrome/Edge/Firefox/Discord). If the initial dispatch fails, it requests subtle activation, briefly yields (~30ms), and retries.
- **Activity Hinting**: On any successful media command (browser or app), `MediaController` calls `MediaPlayerKeepAlive.hint_media_activity(hwnd)` to maintain background responsiveness heuristics.
- **Import Discipline**: `get_media_keepalive()` is imported locally inside call sites to avoid import cycles between controller and keepalive modules.
### Test Logging and Runner Policy

- Pytest initializes centralized rotating file logs at `logs/tests/` via `tests/conftest.py` using `core.logging.configure_logging(...)`
- Rotation config: 2 MB/file, 3 backups. Console at INFO; file at DEBUG. `SPQ_TEST_LOG_DIR` env is exported
- For automated agents/runners: prefer reading `logs/tests/` over stdout. If a run appears hung, enforce a 20s timeout and kill the test runner, then inspect the latest test log file
- Note: Testing is currently deferred until Threading and Resource Manager are feature complete; see `audits/2025-08-19-lock-free-threading-resource-migration.md`

### Startup Hygiene & Environment Variables

- `QT_OPENGL` is deprecated in Qt6 and may force ANGLE paths or emit warnings when set.
- The application unsets `QT_OPENGL` as an early startup hygiene step in the main entry point (`main.py`) before any Qt initialization.
  - Behavior: if present, the variable is removed from `os.environ` and the action is logged via the centralized logger.
  - Rationale: overlay rendering uses DXGI/D3D11 paths, not Qt OpenGL; removing this avoids deprecated GL/ANGLE warnings and accidental GL paths.
  - Policy: the app does not use or set any OpenGL-related environment variables.

### Qt Resource System Registration

- Registration source: `main.py` and `utils/theme/theme_manager.py` both attempt to import `ui.resources_rc` at module import time to register Qt resources compiled from `resources.qrc`.
- Effect: ensures `:/themes/*.qss`, icons, and other embedded `:/` resources resolve correctly in all contexts, including tests and non-main entries.
- Failure path: if the import fails, a DEBUG log is emitted and the theming code falls back to filesystem lookup for QSS. This behavior keeps tests deterministic while avoiding hard failures when resources are not yet built.

### Icon Handling Policy

- Blank icon resource: the canonical blank icon is embedded at `:/icons/Blank.ico`.
- Enumerators must not use filesystem fallbacks for the blank icon. If a resource load fails, generate a transparent pixmap (e.g., 16×16) and log a WARN once for diagnostics.
- Current implementations:
  - `core/window/enumerator.py` and `core/application/window_enumerator.py` load the blank icon from `:/icons/Blank.ico` and generate a transparent icon if the resource is unavailable.
  - `core/window/icons.py#WindowIconManager` currently returns an empty `QIcon()` as the blank fallback; this is slated to be aligned with the resource-only policy.
- Centralization roadmap: deduplicate blank-icon initialization and expose `WindowIconManager.get_blank_icon()` (resource-backed, with transparent fallback). Enumerators should call the centralized API rather than maintain their own `_blank_icon` state.

### Overlay Minimum Size Enforcement (Spec)

- Source of truth: `ui/main_dialog.py` computes and enforces the minimum overlay window size.
- Minimum computation: `_compute_min_size_for_display(screen)` returns an aspect-ratio-aware minimum that is never smaller than 640×360 and is scaled to the active display’s aspect ratio. This ensures usable overlays on high-DPI and ultrawide displays.
- Enforcement: `_ensure_min_overlay_size(rect)` clamps the overlay rectangle to at least the computed minimum while preserving the top-left origin. Used during initial overlay creation/placement and relevant resize paths.
- Relationship to centralized constants: `utils/window/overlay_constants.py` defines `OVERLAY_MIN_WIDTH=200`, `OVERLAY_MIN_HEIGHT=180`, and `DEFAULT_ASPECT=(16, 9)`. These serve as general guardrails and canvas defaults but are not the source of truth for initial overlay window sizing.
- Roadmap: unify min-size logic behind a central helper (e.g., `utils/window/overlay_sizing.py`) consumed by UI and backends; deprecate divergent ad-hoc minima.

#### Wheel Resize Behavior (Spec)

- Dynamic fine-step near minimum bounds without modifiers to prevent step skipping at extremes.
- **Screen boundary performance** (2025-10-09 fix): Boundary smoothing logic checks for **stable pinned edges** before reducing resize steps:
  - When a window is pinned to a screen edge (e.g., left edge at x=0) and growing away from that edge, full resize deltas are applied
  - Smoothing (2px steps) only triggers when the window is actually trying to move beyond screen limits
  - Eliminates slow resize at screen corners (especially bottom-left: x=0, y=max) where boundary detection was false-positive
  - Implementation: tracks whether pinned edge positions remain constant between `cur_geo`, `new_rect`, and `clamped` - stability indicates growing away from boundary
- Clamping is performed within the current monitor's FULL geometry (not union-of-monitors). If the proposed size exceeds the monitor, size is reduced while preserving INNER aspect ratio using canvas insets, then position is clamped to the monitor bounds.
- Inner aspect-ratio lock when the canvas provides `content_aspect` and insets; AR is preserved for inner content while respecting outer min-size.
- Persistence triggers: after wheel-apply and after drag/resize release, if the overlay host exposes `_persist_current_geometry()`, it is called (standalone DWM) to keep persisted state fresh for creation-time restore.

### Portable Build & Distribution (Windows)

- Packaging:
  - Baseline: PyInstaller one-folder. See `scripts/build_pyinstaller.ps1`.
  - Optimized: Nuitka one-directory payload + native launcher. See `scripts/build_nuitka.ps1`.
- Layout (release/dist/):
  - `SPQ.exe` → native C# launcher (icon embedded). Launches `data/bin/SPQ_core.exe`.
  - `data/` → runtime assets
    - `bin/` → Nuitka one-directory payload (when using Nuitka)
    - `resources/` (copied from repo `resources/`)
    - `themes/` (copied from repo `themes/`)
  - `settings/` → user settings and auxiliary files (e.g., `keypassthrough_blocklist.txt`)
  - `logs/` → runtime logs
- Path resolution (single source of truth):
  - `utils.paths.get_runtime_root()` behavior:
    - Honors `SPQ_RUNTIME_ROOT` when set (set by the native launcher to the portable root, e.g., `release/dist`).
    - Normalizes compiled layouts: if the executable resides in `<root>/data/bin` or `<root>/data`, returns `<root>`.
    - Source/dev fallback: repo root (parent of `utils/`).
  - `utils.paths.get_data_dir()` returns `<runtime_root>/data`.
  - `main.py` and `core/ui/tray.py` resolve the app icon from `data/resources/ShittyPIP.ico` with a dev fallback to repo `resources/`.
- Environment:
  - `SPQ_PORTABLE=1` marks portable mode (set by the launcher); settings/logs are under `<runtime_root>/settings` and `<runtime_root>/logs`.
  - `SPQ_RUNTIME_ROOT=<portable_root>` explicit override for runtime root (set by the launcher).
  - `--debug` CLI flag sets `SPQ_DEBUG=1` early and the launcher keeps the child attached until exit (otherwise it exits immediately).
- Launcher (Nuitka optimized build):
  - Working directory: portable root (not `data/bin`).
  - DLL resolution: prepends `data/bin` to `PATH` and calls `AddDllDirectory(binDir)`.
  - Env: sets `SPQ_PORTABLE=1` and `SPQ_RUNTIME_ROOT=<portable_root>`.
  - Exit behavior: exits immediately after launching the core unless `--debug` is supplied (then waits and returns the child's exit code).
- Build usage:
  - PyInstaller: `pwsh -File scripts/build_pyinstaller.ps1 [-Clean] [-Debug]`
  - Nuitka: `pwsh -File scripts/build_nuitka.ps1 [-Clean]`
  - Both scripts validate tools, stage `data/` (resources/themes), create `settings/` and `logs/`, and write `PORTABLE_README.txt`.
  - Nuitka post-stage cleanup: after staging, the build script removes empty `<portable_root>/data/themes` and `<portable_root>/data/resources` directories to reflect that assets are embedded via Qt resources; non-empty directories are preserved.

#### Settings System Architecture (Spec)

**SettingsManager (`core/settings/settings_manager.py`)**

- **Singleton Pattern**: Thread-safe singleton with test isolation support via `_reset_for_testing()` class method
- **File Location Hierarchy**: Robust fallback system for settings file resolution:
  1. **Explicit Path**: When provided (for tests), uses exact path without fallback
  2. **Primary**: `<runtime_root>/settings/settings.json` (portable builds)
  3. **Fallback**: `<executable_dir>/settings.json` (legacy compatibility)
  4. **Last Resort**: `~/.spqmodular/settings.json` (user profile)
- **Comprehensive Defaults**: 25+ logically categorized settings with production-ready defaults:
  - **UI Settings**: Theme, opacity, DPI scaling, window behavior
  - **Capture Settings**: FPS (1-165), monitor selection, quality
  - **Input Settings**: Hotkeys, media control, volume behavior
  - **Debug Settings**: Logging levels, trace flags, verbose modes
  - **Feature Flags**: Component enable/disable toggles
- **Change Notifications**: Qt signal-based change notifications for live updates
- **Validation**: Strict type checking and range validation with detailed error messages
- **Test Integration**: Singleton reset capability for test isolation without breaking production usage

#### Settings File Management

- **Resolution**: `SettingsManager._resolve_settings_dir()` targets `<runtime_root>/settings` when `SPQ_PORTABLE=1` is set (by the launcher) or when a sibling `settings/` directory exists next to the executable. Otherwise, it falls back to the user-profile location (see KeyPassthrough spec for the non-portable path reference).
- **First-run file creation**: After load/migrate/validate, `SettingsManager._ensure_settings_file_exists()` persists the in-memory defaults to `<settings_dir>/settings.json` when the file is missing. Existing files are never overwritten.
- **Blocklist defaults**: `SettingsManager._ensure_keypassthrough_blocklist_defaults()` ensures `<settings_dir>/keypassthrough_blocklist.txt` exists with conservative defaults on first run. Encoding/format per the KeyPassthrough Blocklist spec. Existing files are never overwritten.
- **Directory Creation**: Settings directory is automatically created if missing during initialization
- **Logging**: Creation attempts and outcomes are logged (prefixes: `[SETTINGS]`, `[KEYPASS]`). IO errors are logged; the app continues with in-memory settings and retries on subsequent saves.
  - **Guarantee**: In portable builds, after the first successful run and clean exit, `<runtime_root>/settings/settings.json` and `<runtime_root>/settings/keypassthrough_blocklist.txt` will exist.

---

## 🧵 Threading & Resource Management

### Resource Management

**Centralized resource lifecycle via `utils/resource_manager` facade**

- **Manager**: `core/resources/manager.py#ResourceManager` — implementation tracks resources with weakrefs, reference accounting, and deterministic cleanup. Integrates with `ThreadManager` via `attach_thread_manager()` and dispatches all mutations synchronously on the UI thread (single-writer semantics). Publishes lock-free snapshots via a `TripleBuffer`. Registers an atexit cleanup hook
  - Implementation note (2025-09-18): `_enqueue_mutation_sync()` now enforces UI single-writer by dispatching via the manager's internal UI dispatcher and synchronously awaiting completion before publishing a snapshot
- **Access**: `utils/resource_manager.py` facade re-exports the public API. Always import from `utils.resource_manager` (do not import `core.resources` directly)
- **API**: `attach_thread_manager()`, `register()`, `unregister()`, `get()`, `get_typed()`, `list_resources()`, `cleanup_all()`, `shutdown()`, plus Z-order delegations and convenience helpers
- **Deprecation**: direct imports of `core.resources` in call sites are deprecated; use the `utils.resource_manager` facade exclusively
 - **Policy**: explicit failure if resource cannot be weak-referenced; all mutations execute synchronously on the UI thread via `ThreadManager.run_on_ui_thread` (single-writer semantics). A fresh `TripleBuffer` snapshot is published after each successful mutation. `attach_thread_manager()` is idempotent. Atexit cleanup remains in place. Legacy mutation-worker APIs are retained as no-ops for compatibility

### Deterministic Cleanup Ordering

- **Ordering groups**: Qt → Network/DB → Filesystem/OS → Other
- **Within-group**: stable insertion order unless `cleanup_priority` metadata is provided; lower numbers run first
- **Ties**: broken by registration timestamp
- **ResourceInfo**: carries `group` (derived from `resource_type` and metadata) and optional `cleanup_priority` metadata
- **cleanup_all()**: (on the UI thread) sorts by `(group_rank, cleanup_priority?, registered_at)` and logs a summary of ordering and per-item results

### Thread-Safe Design

**All thread, timer, and hotkey logic is routed through the canonical thread manager (`core/threading/manager.py`)**

- **No direct QTimer or thread usage permitted**; all timer-based operations use `ThreadManager.single_shot(ms, callback)` instead of `QTimer.singleShot`
- **All cross-thread UI updates** use the thread manager's `run_on_ui_thread` method
- **All imports and references are canonical**; no legacy shims or fallback logic remain
- **Lock-free policy**: primary synchronization uses lock-free Single-Producer/Single-Consumer (SPSC) ring buffers managed by `ThreadManager`
- **Frame exchange**: uses triple buffering with atomic indices; producer publishes latest complete frame, consumer reads most recent without locks
- **No shared mutable state** across threads; ownership is transferred via queue items. Bursts are coalesced at queue boundaries with explicit policies (drop/overwrite/coalesce) per channel
- **Primitives module**: `utils/lockfree/` provides `SPSCQueue` and `TripleBuffer`
- **Docking Mode Threading**: All docking system operations (overlay creation, synchronization, positioning) are dispatched to UI thread via `ThreadManager.run_on_ui_thread` for thread safety
  - Gate: first-press gate held in `HotkeyManager._kb_press_state[hotkey_id]` to prevent repeats; reset on release.
  - Watchdog: tokenized watchdog uses `ThreadManager.single_shot(interval≈85ms)` to poll `keyboard.is_pressed(key_token)`. If a release event is missed, it resets the gate and logs `[HOTKEY][WATCHDOG] gate-reset` for the affected hotkey. Scheduling is idempotent per-token to avoid races.
  - Modifiers: press handlers ignore when `ctrl` or `shift` are down to avoid unintended activation with modifiers; this policy remains unchanged.
  - Thread-safety: handlers and watchdog callbacks are lock-light, mutate only hotkey-local state, and avoid cross-thread UI calls; UI dispatch goes through `ThreadManager.run_on_ui_thread` when needed.

 - **Combination hotkeys (2+ keys)**: registered via Windows `RegisterHotKey` and processed in the message loop; suppression is not applied in this path.
 
 - **Shutdown**: watchdog tokens are invalidated on `HotkeyManager.shutdown()`; pending checks become no-ops.

  - **Manager Architecture**:
    - Message loop runs on a dedicated ThreadManager task (IO pool). A message-only window is created on that thread; `GetMessage` pumps `WM_HOTKEY` for system registrations.
    - Registration/unregistration requests are posted to the hotkey thread via a lock-free `SPSCQueue` (single writer: callers; single reader: hotkey thread). No raw `threading.Lock/RLock` in hot paths.
    - On shutdown, `WM_QUIT` is posted to cleanly break the loop; all registrations are unregistered and keyboard hooks removed.
    - Enhanced error logging for `RegisterHotKey` captures `GetLastError()` codes and logs them with context and the attempted modifier/key tuple. Retries without `MOD_NOREPEAT` are attempted when applicable.

  - **Settings (Hotkeys)**:
    - `hotkeys.prefer_keyboard_fallback` (bool; default false): When a global registration fails or for single-key-with-suppress scenarios, prefer registering the equivalent keyboard combo (e.g., `alt+<key>`) first before attempting system-wide registration. If the keyboard path succeeds, the system path is skipped.
    - `hotkeys.allow_single_digits` (bool; default false): Treat `0–9` as safe single keys for the keyboard backend suppression path (no modifiers). Affects `_is_safe_single_key()` decision logic.

#### Quickswitch (Spec)

- **Ownership & Registration**: `QuickSwitchController` owns quickswitch. Registers a global combo via the `keyboard` library from settings key `hotkeys.opacity_quickswitch` (default: `shift+x`). No fallback combos and no backtick-specific handling.
  - Centralization policy (Option A retained): QuickSwitchController keeps its own combo registration and cooldown/locking logic. `HotkeyManager` continues to own other global/system hotkeys. This avoids duplication of docking-aware selection rules and preserves the single entry point while docking mode is active (controller delegates to docking manager and returns).
  - Option B (deferred): If future work centralizes the quickswitch combo under `HotkeyManager`, extract a pure orchestrator for docking/window-mode selection (no Qt) to avoid logic drift. The manager would then execute decisions; controller becomes a thin shim. Not implemented as of 2025-09-18.
- **Settings**:
  - `hotkeys.quickswitch_enabled` (bool): toggles feature
  - `hotkeys.opacity_quickswitch` (str): stores the combo; live-updated by SubSettingsDialog
- **Threading Model**: `quickswitch()` performs an early cooldown check on any thread. If needed, it dispatches to the UI thread via `ThreadManager.run_on_ui_thread()` where `_quickswitch_impl()` executes the full flow (MRU selection, overlay swap, focus handoff). Focus handoff is deferred ~25ms using `ThreadManager.single_shot()`.
- **Lock-free Gating**: Uses an in-flight flag to prevent re-entrancy and a monotonic timestamp gate for a fixed 800ms cooldown (`time.monotonic()`). No `threading.Lock/RLock`.
- **Events**: Publishes suppression diagnostics: `switch.cooldown_suppressed`, `switch.reentry_suppressed`, `switch.lock_suppressed`.

##### Quickswitch – Implementation Flow (Authoritative)
- **Cooldown and Dispatch**
  - 800ms cooldown gate enforced both before and on the UI thread.
  - Off-UI invocations schedule `_quickswitch_impl()` on UI via `ThreadManager.run_on_ui_thread()` with a simple `_inflight` guard.
- **Docking-first Delegation**
  - If a `DockingOverlayManager` is active (found via `ResourceManager`), delegate: `handle_overlay_interaction("main", "quickswitch")` and return.
- **Active Overlay Path** (non-docking)
  - Resolve active overlay from `OverlayManager`.
  - If globally or individually locked, do not swap; instead focus the overlay's captured window (if available), emit `switch.lock_suppressed`, and return.
- **Candidate Collection**
  - Read current foreground via `win32gui.GetForegroundWindow()` and current overlay source (`overlay._current_source_hwnd` fallback `_src_hwnd`).
  - MRU list via `get_mru_manager().get_recent(limit=7)`. If MRU has <2 entries, seed from Z-order as needed.
  - Build ordered unique candidates: `[foreground, current_src, MRU...]`.
  - Optional display-locked filtering when `features.display_locked_switching` is true: restrict to windows on the same monitor as the current source.
  - Exclude overlay/host HWNDs from focus targets (forbidden set includes host, border, DWM host when present).
- **Selection and Swap**
  - Prefer swapping to the valid foreground if not forbidden and display-locked predicate passes; otherwise use `compute_next_selection(...)` over the candidate list.
  - Perform `overlay._handle_swap_window(target_hwnd)`. Update MRU for the foreground and focus target.
  - Defer focus handoff ~25ms to the pre-swap source (or chosen) and arm autoswitch suppression: `get_autoswitch_controller().suppress_for(900, last_seen_hwnd=focus_target)`.

#### Quickswitch Logging (Spec)

- **Logging**: Centralized utilities: `core.logging.throttled` and `core.logging.log_dedupe` are used inside `core/switching/quickswitch_controller.py#QuickSwitchController` to reduce debug log spam during rapid operations.
- **Categories (tags)**:
  - `"quickswitch:reentry"` — re-entrancy gate suppressions
  - `"quickswitch:lock"` — overlay lock checks and outcomes
  - `"quickswitch:foreground"` — foreground window reads
  - `"quickswitch:mru"` — MRU candidate processing and filtering
  - `"quickswitch:seed"` — seeding MRU from z-order when sparse
  - `"quickswitch:dispatch"` — UI-thread focus-change dispatch
  - `"quickswitch:fail"` — deduplicated early-abort reasons
  - `"quickswitch:cooldown"` — cooldown period and watchdog timer
- **Behavior**: high-frequency debug emits in `quickswitch()` are throttled (category-specific windows) and repeated failure messages are deduplicated to preserve clarity. Functional behavior is unchanged.
- **Fallback**: if the helpers are unavailable at runtime, initialization falls back to standard `logger.debug` so logging remains functional (no feature dependency on throttling/deduping).
- **Gating**: respects the global debug policy (`SPQ_DEBUG`, `utils.debug.debug_enabled`).

##### Docking Integration (Quickswitch)
- **CRITICAL FIX (2025-10-06)**: Fixed `_rotate_mru_forward()` to implement correct **targeted swap** behavior
- When docking is active, QuickSwitch delegates to `DockingOverlayManager.handle_overlay_interaction("main", "quickswitch")` which triggers MRU rotation with targeted swap
- **Targeted Swap Logic (Normal Mode)**: 
  - **User intent**: "I want to interact with the window shown in this overlay"
  - **Rotates MRU**: If overlay's window == MRU[0], swap MRU[0] ↔ MRU[1]
  - **Focuses outgoing window**: The window being swapped OUT goes to foreground (user can interact with it)
  - **Swaps in MRU[0]**: Overlay displays new MRU[0] after rotation
  - **Result**: User gets the window they clicked, overlay shows next most-recent window
- **Applies to all overlays**: Main (A) and secondaries (B/C/D/E) use same targeted swap pattern in Normal mode
- **Enhanced Secondary Focus**: Double-click on secondary overlays uses `_normal_mode_swap_with_foreground()` for targeted swap
- **Cycle Mode**: Double-click focuses window, all overlays update dynamically based on MRU
- Per-overlay lock (focus indicator) suppresses quickswitch automatic updates; manual double-click on overlays remains enabled with proper locked overlay handling

#### Autoswitch (Spec)

- **Ownership**: `ForegroundAutoswitchController` observes OS foreground changes and may swap overlay sources when appropriate.
- **Debounce**: Stable foreground selection window `STABLE_DEBOUNCE_MS` enforced; suppression window supported for quickswitch handoff.
- **Active Overlay Resolution**: Prefers docking main overlay when docking is active; otherwise uses `OverlayManager` active overlay.
- **Lock-aware Gating**:
  - Global lock: suppressed when `OverlayManager.is_overlay_locked()` is true.
  - Individual lock: suppressed when the active overlay is locked. For docking, this checks the DWM backend lock via the DockingOverlay wrapper (`overlay._dwm_overlay._is_window_locked`). For direct DWM overlays, the controller checks `_is_window_locked` on the overlay itself.
- **Selection Policy**:
  - Normal path (foreground == current source): cycles MRU to the next eligible window (no Z-order fallback). Display-locked filtering optional via setting `features.display_locked_switching`.
  - Emergency path (invalid/missing current source): attempts recovery by picking a valid MRU alternative (no Z-order fallback).
- **Validation**: All candidate windows must pass `utils.window_validation.is_valid_window`.

##### Autoswitch – Implementation Flow (Authoritative)
- **Polling**
  - Self-rescheduling tick via `ThreadManager.single_shot(POLL_INTERVAL_MS)`. Defaults: `POLL_INTERVAL_MS=250ms`, `STABLE_DEBOUNCE_MS=300ms`.
  - Suppression window after QuickSwitch handoff: `suppress_for(duration_ms=900, last_seen_hwnd=...)` prevents immediate re-trigger; also stabilizes `_last_seen_hwnd`.
- **Docking Path**
  - If docking is active, only trigger a cycle when the foreground equals the main overlay's current window. Otherwise, record MRU and exit.
- **Non-docking Path**
  - Resolve active overlay from `OverlayManager`. Respect global and per-overlay locks.
  - If foreground equals current source, build candidates `[foreground, current_src, MRU...]`, optionally apply display-locked filter; compute next via `compute_next_selection(...)` and swap via `overlay._handle_swap_window(chosen)`.
  - If foreground differs from current source, do nothing unless in emergency (missing/invalid current source). In emergency, compute candidates as above and swap.
  - Record MRU for chosen targets; maintain `_last_applied_src` and `_last_seen_hwnd` for stability and diagnostics.

---

### Switching – Potential Improvements (Proposal)
- **Make timings configurable**
  - Expose `quickswitch.cooldown_ms` (default 800) and `autoswitch.debounce_ms` (default 300) in settings to accommodate user preference and hardware variance.
- **Adaptive autoswitch debounce**
  - Increase debounce temporarily after a swap (e.g., +150–250ms) to reduce oscillation when users rapidly alt-tab between the same two windows.
- **Unified suppression window**
  - Align QuickSwitch cooldown (800ms) and Autoswitch suppression (900ms) under a single configurable value for more predictable interactions.
- **Display-locked switching UX**
  - Surface `features.display_locked_switching` in the UI with a brief tooltip. Optionally allow per-mode (docking vs. single overlay) overrides.
- **User feedback on lock suppression**
  - When a swap is suppressed due to an overlay lock, optionally show a brief non-intrusive toast or focus-indicator pulse explaining why nothing changed.
- **MRU quality**
  - Consider filtering known shell/system windows during MRU seeding beyond `is_valid_window` (e.g., taskbar, our own process), especially on fresh starts.

#### Event System Architecture (Spec)

**EventSystem (`core/events/event_system.py`)**

{{ ... }}
- **Rich Event Objects**: Handlers receive full `Event` objects with comprehensive metadata:
  - **Event Data**: `.data` property contains the actual event payload
  - **Metadata**: Event ID, timestamp, source, priority, and lifecycle flags
  - **Lifecycle Management**: `is_handled` flag for event consumption tracking
  - **Tracing Support**: Optional dispatch tracing via `debug.events_trace` setting
- **Thread-Safe Dispatch**: Protects subscription maps and event history using a re-entrant lock (`threading.RLock`)
- **Lock-Free Handler Execution**: Handlers execute after the lock is released to prevent contention from long-running callbacks and allow safe subscribe/unsubscribe/publish reentrancy
- **UI-Thread Routing**: Opt-in per subscription via `dispatch_on_ui=True` in `subscribe(...)` to route callbacks through `ThreadManager.run_on_ui_thread` for thread-affine UI work
- **Priority-Based Dispatch**: Handlers executed in priority order with deterministic tie-breaking
- **Resource Integration**: Registers with ResourceManager for proper cleanup during shutdown
- **Event History**: Maintains event history for debugging and replay capabilities
- **Default Behavior**: Keeps publish synchronous on the caller's thread; prefer `dispatch_on_ui=True` (or an async policy at the call site) for heavy/UI handlers

#### Event Object Structure

```python
class Event:
    def __init__(self, event_type: str, data: dict, source: str = None, priority: int = 0):
        self.id: str = uuid.uuid4().hex
        self.event_type: str = event_type
        self.data: dict = data  # Actual event payload
        self.source: str = source
        self.priority: int = priority
        self.timestamp: float = time.time()
        self.is_handled: bool = False
```

#### Handler API

- **Handler Signature**: `def handler(event: Event) -> None`
- **Event Access**: Use `event.data` to access the actual event payload
- **Metadata Access**: Use `event.timestamp`, `event.source`, etc. for debugging and tracing
- **Event Consumption**: Set `event.is_handled = True` to mark event as consumed

#### Shutdown Coordination (ThreadManager ⇄ ResourceManager)

- **Goal**: Guarantee deadlock-free application shutdown while preserving lock-free runtime behavior.
- **Sequence (non-blocking under locks)**:
  1. `ThreadManager.shutdown(wait=True|False)` acquires its internal lock only to set `_shutdown=True` and capture references (`resource_manager`, `active_task_ids`, `executors`, `resource_id`). It then releases the lock before any potentially blocking operations.
  2. Outside the lock, cancel active tasks and shut down executors. If shutdown is invoked from a worker thread of a pool, that pool is shut down with `wait=False, cancel_futures=True` to avoid self-join deadlocks; otherwise uses the requested `wait` semantics.
  3. Resource cleanup is performed via `ResourceManager.shutdown()` which synchronously routes `cleanup_all()` to the UI thread. Callers must avoid holding locks during this call.
  4. Best-effort unregister of the `ThreadManager` resource via `ResourceManager.unregister(...)` happens on the UI thread and may be skipped if already in shutdown.
- **ResourceManager behavior**:
  - `shutdown()` performs synchronous `cleanup_all()` on the UI thread in the deterministic cleanup order (Qt → Network/DB → Filesystem/OS → Other).
  - Legacy mutation-worker methods remain as no-ops for compatibility.
- **Policy**: No long waits or cross-component calls occur while holding locks. All blocking operations happen after releasing locks to preserve the lock-free design in hot paths and avoid shutdown-time deadlocks.

#### Example: Mixed-type cleanup ordering

```
Items:
- A: Qt widget (group=qt)
- B: DB session (group=network_db, cleanup_priority=0)
- C: file handle (group=filesystem)

Order → A, B, C  (Qt first; within Network/DB, lower priority first; then Filesystem)
```

##### Monitor Capture → Render Pipeline (Spec)

- Producer/Consumer: latest-frame semantics over a lock-free `TripleBuffer` via `utils.frame_exchange`.
  - Producer: `core/graphics/capture/monitor_capture_manager.py#MonitorCaptureManager` publishes `CaptureFrame` objects to `utils.frame_exchange.get_exchange(f"monitor_frames_{qt_index}")` on each successful DXGI frame (dxcam).
  - Consumer: The active renderer backend polls `acquire_latest()` on the configured exchange and triggers repaint.
  - Multi-monitor exchange naming: `monitor_frames_<qt_index>` where `<qt_index>` is derived from the selected screen's `qt_index` in `utils.monitor_utils`.
  - Compatibility signal: `MonitorCaptureManager.frame_captured` remains emitted on the UI thread for existing listeners; FrameExchange is the canonical path for rendering.
  
  ###### CaptureFrame Semantics (CAP-FRAME)
  
  - Payload: `image_data` contains raw RGB (3 bpp) or BGRA (4 bpp) with metadata (`width`, `height`, `timestamp`, `monitor_index`).
  - GPU pointer: not used in current DXGI-only pipeline; `d3d11_tex_ptr` is None. Reserved for future zero-copy integration.
  - Consumer policy: consume `image_data` bytes only at present.
  - Producer: `MonitorCaptureManager` publishes bytes-only frames via dxcam.

  ###### Capture Rate Setting (Spec)

  - Setting key: `capture.fps` (int)
  - Range: 1–165 inclusive. Default: 60.
  - Validation: strict; out-of-range or non-int values raise `ValueError` in `core/settings/settings_manager.py`.
  - Application:
    - UI: `ui/dialogs/subsettings_dialog.py` persists the value and exposes presets: 15, 30, 60, 120, 144, 165. Missing persisted custom values are inserted into the combo.
    - Runtime: `core.graphics.pipeline_manager.get_pipeline_manager().set_capture_rate(fps)` applies the effective rate to the active capture backend.
  - Policy: No legacy caps (e.g., 120) remain; UI and validator are authoritative up to 165 to cover high-refresh displays (144/165 Hz).

## Section: Project Structure

### Root Directory (`/`)
- **`/core`**: Core application modules
- **`/docs`**: Project documentation and architecture
- **`/tests`**: Test files and test utilities
- **`/utils`**: Utility modules and tools
- **`/themes`**: Theme files and assets
- **`/resources`**: Application resources (icons, images, etc.)

### Core Modules (`/core`)
- **`application/`**: Application lifecycle and core services
- **`graphics/`**: Graphics and overlay system
- **`window/`**: Window management utilities
- **`events/`**: Event system implementation
- **`settings/`**: Application settings management
- **`resources/`**: Resource management
- **`threading/`**: Thread management utilities
- **`ui/`**: User interface components
- **`theme/`**: Theming system and style management

### Graphics System (`/core/graphics`)
- **Overlay Architecture**
  - `overlay.py`: Base `Overlay` class with common interface
  - `overlay_host.py`: Host window for overlays
  - `overlay_manager.py`: Enhanced overlay tracking and lifecycle management with MRU tracking, border overlay support, and z-order enforcement
    - Canonical ID policy: all overlays are tracked by their intrinsic `overlay.id` (UUID). OverlayManager uses this canonical ID as the only key.
    - Registration: after `initialize()`, `OverlayManager` registers the main overlay's host widget with `ZOrderManager`. On removal it unregisters it and cleans up any associated border overlay.
  - `utils/z_order_manager.py`: Canonical `ZOrderManager` — centralized z-order management for all overlays; integrates context-menu priority and debounced enforcement via `ThreadManager`. The legacy `core/graphics/z_order_manager.py` is deprecated and must not be referenced.
  - `dwm_composition_manager.py`: Centralized DWM composition attribute management for consistent rendering
  - `backend_manager.py`: Backend selection and initialization
  - `types.py`: Shared type definitions
  - `ui/`: Overlay UI components
  - Context menu: centralized in `utils/overlay_context_menu.py`; all backends delegate to this handler. DWM overlays pass `border_overlay` so borders remain visible while the menu is open. No backend constructs or owns menus directly.

##### Context Menu Z-Order Enforcement (Spec)

- Purpose: ensure consistent, flicker-free visibility of the BorderOverlay while context menus are shown for DWM overlays.
- Centralized API usage only:
  - Lifecycle entrypoints: coordinated via centralized managers; use `utils.resource_manager.get_resource_manager()` (facade). 
  - Normal enforcement remains available via `ResourceManager.enforce_z_order(overlay_id)` for non-menu paths.
  - No explicit scheduling/retry calls in `OverlayContextMenu`; deferrals/debouncing are handled internally by `ZOrderManager`.
- Lifecycle hooks:
  - Before show: call `begin_context_menu(overlay_id, menu)`.
  - After hide: call `end_context_menu(overlay_id, menu)`.
- Triggers:
    - `QMenu.aboutToShow` → `_ensure_border_visible(before=True)` (delegates to `begin_context_menu`)
    - `QMenu.aboutToHide` → `_ensure_border_visible(after=True)` (delegates to `end_context_menu`)
    - Event filters on overlay host, canvas, and border frame intercept:
      - `QEvent.ContextMenu`
      - Right-click `MouseButtonPress` (and related RMB events)
      - On intercept: call `_ensure_border_visible(before=True)` (delegates to `begin_context_menu`), then show menu at the event position
    - Additionally consumes Right-button release and double-click to prevent other handlers reacting while the menu is active.
    - Context menu policy:
      - Affected widgets (host/canvas/border) are forced to `Qt.PreventContextMenu` to guarantee the filter path; defaults are restored on detach.
      - No raw QTimer usage:
        - No explicit scheduling or QTimer.singleShot in this module
        - All deferrals are implemented inside `ZOrderManager` using the centralized `ThreadManager.single_shot`
        - Enforces consistent timer management across the codebase
- Strict no-fallback logging:
  - All enforcement attempts are logged explicitly (debug for attempts and results; warnings/errors on exceptions). No silent passes.
- Cleanup:
  - `OverlayContextMenu.detach_from_overlay(overlay=None)` removes installed filters for the specified overlay (host/canvas/border) or all tracked filters when `None`.
  - Restores `Qt.DefaultContextMenu` on targets; logs any failure explicitly; returns whether any filter was removed.
  - Use during overlay teardown to prevent ghost filters and policy leakage.
- **`overlay_manager.py`**: Manages overlay lifecycle and state
- **`backend_manager.py`**: Manages rendering backends
- **`renderer.py`**: Base renderer implementation
- **`types.py`**: Graphics-related types and enums

#### DWM Overlay Initialization & Scaling (Spec)

- Flicker-free initialization: the overlay host window and DWM thumbnail remain hidden until layout is finalized; reveal occurs in a single step with a subtle fade-in.
- Destination rect source of truth: use `IntegratedBorderCanvas.content_rect()` (letter/pillarbox area) mapped through `border_frame()` into host-window coordinates; never use the full canvas rect.
- DPI/coordinates: perform physical-pixel conversion once at update time using Win32 RECT semantics `(x, y, x + width, y + height)` to avoid off-by-one bleeds. Do not pre-convert tuples passed within the code path; convert only at the final update call.
- Caching: destination rect is recomputed on each content/geometry change; rapid changes are coalesced via UI-thread scheduling. No `_last_dest_rect_phys` cache is maintained in the current implementation.
 - Content inset only: destination rect integrates an inward content inset equal to the border stroke/bleed budget; this is the sole DWM bleed mitigation. Additionally, `IntegratedBorderCanvas.content_rect()` shrinks by the inner accent inset + thickness (DPI-aware, pixel-snapped) calculated via `AccentCalculator` to prevent DWM/content overlap of the inner accent line. No QPainterPath clipping, no backdrop clipping layer, and no enhanced host masking are used.
- Minimal host round mask retained: a small QRegion mask is applied to the host window for visual consistency only; it does not clip the DWM thumbnail.
- Removed legacy methods: `_apply_content_clipping`, `_apply_enhanced_host_mask`, `_apply_minimal_host_mask`, `_ensure_backdrop_clipping_layer`, `_remove_backdrop_clipping_layer`.
- Integrated border rendering: borders are rendered directly in the canvas component:
  - `IntegratedBorderCanvas`: Unified canvas with direct border rendering
  - `BorderRenderer`: Pure rendering engine for pixel-perfect borders
  - `BorderGeometry`: DPI-aware metrics calculation with caching (main border only)
  - `BorderTheme`: Strict theme integration with fail-fast token validation
  - `AccentCalculator`: Unified inner accent calculation system providing single source of truth for thickness, inset, and radius with DPI scaling and coordinate alignment validation
  - Features theme-appropriate colors (always white in dark theme), adaptive thickness scaling, optional inner accent effect with gap-free rendering
  - Eliminates separate BorderOverlay window and all z-order coordination complexity
- Initialization race fix: removed duplicate initialization triggers that caused back-to-back initialization and incorrect scaling by using centralized `ThreadManager.single_shot` for all deferred operations.
- Opacity routing (integrated): the canvas backdrop and DWM thumbnail both use the configured opacity for unified fades, while border strokes remain fully opaque for crisp edges. Window opacity is not manipulated in this path.
- DWM source area: no source RECT is passed to DWM to avoid coordinate mismatches; `fSourceClientAreaOnly` is enabled (True) so only the client area is displayed. Destination RECT is snapped to integer physical pixels using exclusive right/bottom edges and includes the current content inset policy.

##### Canonical Border/Clipping/Masking (Spec)

- Borders are rendered directly in the canvas component (`ui/overlays/integrated_border_canvas.py#IntegratedBorderCanvas`). No separate border windows.
- Window-level masking is applied by the overlay host window (not child widgets) using a rounded-rect `QPainterPath`/`QRegion` when rounded borders are enabled. Radius/inset are slightly reduced for clean edges. This mask is purely visual; it does not clip the DWM thumbnail.
- Avoid redundant/conflicting widget-level masks. The canvas may set its own widget mask for visuals, but parent widgets in the hierarchy must be transparent to prevent QSS background bleed-through in rounded corners.
- DPI and coordinate conversions are performed once at the final update call (Win32 RECT semantics with exclusive right/bottom edges). No pre-conversion of tuples within the pipeline.
- UI-thread work (e.g., DWM thumbnail property updates) is batched via a `ThreadManager` UI coalescer. Z-order and context menu priority are centralized through `ResourceManager`/`ZOrderManager`. All timers run through `ThreadManager` (no raw `QTimer`).
- No legacy/fallback paths. A safe fallback applies/clears a parent-level mask only when no overlay is exposed (legacy hosts).

See also: `Index.md` → Graphics & Overlays → "Canonical Border/Clipping/Masking" for a quick reference link back to this section.

##### Defensive Error Handling for DWM Thumbnail Updates (Spec)

- Purpose: prevent crashes and noisy warnings caused by transiently invalid canvas content rectangles during initialization/layout or rapid geometry changes.
- Behavior:
  - `_update_thumbnail_properties(...)` is the source of truth for computing/applying the destination RECT from `IntegratedBorderCanvas.content_rect()`. When the host/canvas is missing or the content rect is empty/invalid, it logs at debug level and skips the update; subsequent events retry automatically.
  - Errors during property computation/application are trapped and logged; exceptions are not raised for transient conditions.
  - All thumbnail operations execute on the UI thread and are routed through a `UICoalescer` created via `ThreadManager.create_ui_coalescer(name, capacity=128, window_ms=7)`; rapid content-rect/geometry changes submit `_update_thumbnail_properties` to this coalescer. If coalescer init fails, falls back to `ThreadManager.run_on_ui_thread`.

  - Latest-value exchange (TripleBuffer): `IntegratedDWMOverlay` uses a `TripleBuffer[QRect]` created via `ThreadManager.create_triple_buffer()` to carry the latest canvas content rectangle across bursts safely without locks.
    - Producer: `_on_content_rect_changed(...)` publishes the new `QRect` to the triple buffer.
    - Consumer: a dedicated drain on the UI side (e.g., `_drain_and_update_thumbnail`) is invoked by the UI coalescer once per batch; it calls `consume_latest()` and applies thumbnail properties only if a fresh value is available.

  - ResourceManager registrations (deterministic cleanup):
    - `ThumbnailManager` instance registered with descriptive tags.
    - Active thumbnail binding (dest HWND ↔ source HWND) registered with a cleanup handler that unregisters the DWM thumbnail.
    - The `TripleBuffer` instance registered with a cleanup handler that resets/clears internal slots.
    - The overlay `OverlayContextMenu` registered with a cleanup handler to detach it from the overlay.
    - `_close_impl` prefers `ResourceManager`-driven teardown via unregistering all related registrations; best-effort direct cleanup remains as a fallback for transient states.
- Guarded lifecycle points (all defensively trigger updates and skip when invalid):
  - `_on_content_rect_changed(...)`
  - delayed init apply inside `_initialize_impl()`
  - `_handle_reset_position()`
  - `_ensure_stack_order()`
  - `_swap_source_hwnd(new_hwnd)`
  - `_ensure_final_state()`
- Z-order policy: centralized enforcement via `ResourceManager.enforce_z_order(overlay_id)` (strict, no fallback). Failures are logged explicitly; any deferrals are handled internally by `ZOrderManager`.
- Scheduling: no raw `QTimer` in modules. Deferrals are handled internally by `ZOrderManager` using `ThreadManager`.

###### DWM Thumbnail Retry & Shutdown Semantics

- Registration retry policy:
  - `utils/window/thumbnail_manager.py#ThumbnailManager.register_thumbnail(dest, src)` performs a one-shot synchronous retry on `E_INVALIDARG (-2147024809)` after a short (~15ms) sleep via `kernel32.Sleep`. The method records `last_hresult` for diagnostics.
  - Pre-validation is minimal: source HWND must be valid and have a non-zero client rect; destination HWND is not hard-rejected to avoid transient Qt handle races. DWM’s HRESULT is used as the authority.
- Overlay recovery flags:
  - `core/graphics/backends/dwm/integrated_dwm_backend.py#IntegratedDWMOverlay` sets `_broken=True` and `_needs_recreate=True` when registration fails (logs include `ThumbnailManager.last_hresult`). On next successful registration these flags are cleared.
  - `core/graphics/overlay_manager.py#OverlayManager.create_overlay()` detects overlays with `_needs_recreate` and tears them down instead of reusing, ensuring a clean recreate path.
- Deferred registration after show:
  - `_show_impl()` attempts immediate registration; on failure, schedules bounded deferred retries (short cadence, ~500ms total). The overlay remains usable during deferral.
- Shutdown behavior:
  - During cleanup, `ThumbnailManager.unregister_thumbnail(hwnd)` treats `E_INVALIDARG` from `DwmUnregisterThumbnail` as benign and logs at DEBUG. This avoids noisy errors when DWM has already invalidated the handle during app shutdown.

##### OverlayHost Geometry Verification (Spec)

- Purpose: reduce redundant normalization and UI work during rapid move/resize of the overlay host window.
- Wiring:
  - `core/graphics/overlay_host.py#OverlayHost` connects `geometryChanged` to `_on_geometry_changed()`.
  - `_on_geometry_changed()` submits `_verify_geometry_coalesced()` to a lazily-created UI coalescer named `overlay_geom_<id>` with `capacity=128`, `window_ms=7`.
  - Fallback: when coalescer init fails, routes via `ThreadManager.run_on_ui_thread`.
- Behavior of `_verify_geometry_coalesced()`:
  - Normalizes host geometry using existing helpers (e.g., `set_host_geometry(...)`).
  - Updates focus indicator position and lock state.
  - Triggers optional window masking on the parent overlay when supported (e.g., `_apply_window_masking()`).
  - Non-throwing UI path: exceptions are caught and logged at debug level.

##### Top-level Volume OSD Window in OverlayHost (Spec)

- Ownership: `core/graphics/overlay_host.py#OverlayHost` constructs `ui.components.volume_osd.VolumeOSDWindow(host_widget=self)` as a top-level window (`objectName="volumeOSD"`). The window inherits behavior from `VolumeOSDWidget` and self-manages show/hide timing and repaints.
- Positioning (screen coordinates): bottom-center relative to the host with a 12 px bottom margin. Uses `host.mapToGlobal(QPoint(0,0))` to compute the screen-space anchor. Maintained by wiring `geometryChanged → _safe_update_volume_osd_position()` which calls `VolumeOSDWindow.update_position(target_rect?: QRect)`.
- Z-order enforcement: `ResourceManager.place_window_above(osd_window, host_widget)` ensures the OSD window stays directly above the host without using global topmost, combined with `.raise_()` as a secondary safeguard. Window flags include `Qt.Tool | Qt.FramelessWindowHint | Qt.WindowDoesNotAcceptFocus | Qt.WindowStaysOnTopHint` with `WA_ShowWithoutActivating`.

##### Opacity Control Policy (Spec)

- **Units & Storage**:
  - UI/settings store opacity as integer percent [10–100]. Backend uses float [0.1–1.0].
- **Minimum Floor**: Absolute minimum opacity is 10% (0.1). Values below this cause flicker and are not permitted.
- **Adjustment Cadence**: Hotkey-driven adjustments use a 1-1-1-2 cadence per tick to speed up perceived changes while staying smooth.
- **Clamping Layers (defensive)**:
  - `core/opacity/manager.py` enforces [10, 100] and normalizes persisted values at startup (clamps <10 to 10 with immediate save).
  - `ui/main_dialog.py::set_opacity(...)` clamps to >=0.1 before delegating to overlay manager and logs boundary conditions.
  - `core/graphics/backends/dwm/integrated_dwm_backend.py::IntegratedDWMOverlay.set_opacity(...)` clamps to [0.1, 1.0] before applying to canvas/thumbnail and logs clamping and boundary values.
- **Logging**:
  - Debug logs with the `[OPACITY]` tag are emitted on clamp events and at boundaries (10%/100%) across the above layers to aid diagnostics.
  - Manager logs cadence steps during continuous adjustment.
- - Theming: integrated with the centralized `ThemeManager`. `_apply_theme()` applies the current theme to the host, canvas, and OSD; `theme_changed` re-applies on runtime switches. Styling is QSS-driven per project policy.
 - Styling (current defaults): volume fill bar is black (`#000000`) at ~90% opacity (alpha ≈ 230). Volume text is uppercase and white. Implemented in `ui/components/volume_osd.py#VolumeOSDWidget.paintEvent` (base class) to match the visual spec; theming overrides may adjust via QSS in the future.
 - Event-driven visibility: `VolumeOSDWindow` (via base handler) subscribes to "media.volume.changed" on the centralized EventSystem (UI-thread dispatch) and becomes visible briefly on updates. No ad-hoc timers outside the centralized threading system.
 - Threading/coalescing: UI updates and rate-limited repaints use the canonical `ThreadManager`/UI-coalescer facilities provided by the base widget; no raw `QTimer` usage.
 - Tests: `tests/ui/overlays/test_volume_osd_in_overlay_host.py` verifies instantiation (as a top-level window), positioning in screen coordinates, visibility on `media.volume.changed`, and reposition on host resize.
 - Textual fallback: when the event lacks a numeric volume level (e.g., fields `volume`/`level` are missing), the OSD displays a textual indicator — "VOLUME UP" or "VOLUME DOWN" — derived from the event's direction.

##### Windows Messaging Policy (Spec)

- Safe message sending: `safe_send_message` is the canonical API for sending messages to windows; it wraps `PostMessage`/`SendMessage` with error handling and logging.
- Volume hold timer behavior: the timer is started on press and canceled on release; a small suppression window (~35ms) applies to smooth rapid events.
- Volume hold timer settings:
  - Initial delay: 200ms
  - Repeat interval: 75ms
- OSD textual fallback: when the event lacks a numeric volume level, the widget displays a textual indicator — "VOLUME UP" or "VOLUME DOWN" — derived from the event's direction.

##### DWM Composition Attribute Handling (Spec)

- Centralized management through `DWMCompositionManager` (core/graphics/dwm_composition_manager.py)
- Defensive handling for attributes that may not be supported across all Windows versions:
{{ ... }}
  - `FREEZE_REPRESENTATION` is treated as an optional attribute
  - Failures to set unsupported attributes like `FREEZE_REPRESENTATION` are considered non-critical
  - Success count accounting adjusts for known unsupported attributes to avoid misleading log messages
  - Warnings are not raised for expected platform limitations
- Explicit logging: All attribute application attempts and results are logged at appropriate levels
- Strict no-fallback for critical attributes, graceful handling for optional attributes with clear documentation

##### Context Menu Integration and app_instance Injection

- OverlayManager injects `app_instance` via a provider set with `OverlayManager.set_app_instance_provider(...)`, wired by `ApplicationCore` during startup. This occurs BEFORE `initialize()` in `core/graphics/overlay_manager.py#create_overlay`.
- The injected `app_instance` is required by `OverlayContextMenu` to populate the "Switch To Window" submenu using `app_instance.get_windows()` (no direct `app_core` imports).
- `DWMOverlay` must expose `app_instance` and implement `_handle_swap_window(hwnd:int)`; selection triggers `_swap_source_hwnd` to switch the thumbnail source.
- Reuse path requirement: `DWMOverlay` should also implement `update_source(new_hwnd:int) -> bool` for `OverlayManager` reuse. This method validates the input and delegates to the canonical swap path (queues `_swap_source_hwnd` on the UI thread). Controllers do not call `update_source()` directly.
- Strict no-fallback: if injection fails or required members are missing, overlay creation is aborted with an explicit error.
- Centralized Settings actions: `OverlayContextMenu` owns "Main Window" and "Subsettings" actions. It locates `ui.main_dialog.MainDialog` via `QApplication.topLevelWidgets()`, calls `show()`, `raise_()`, and `activateWindow()` to foreground, and invokes `show_sub_settings()` (or `_open_subsettings_dialog()` if exposed) for subsettings. Backends MUST NOT implement `show_main_window`/`show_sub_settings`; any override goes through `config.actions`.
 
###### Window-mode Lazy Initialization (Spec)

- `ApplicationCore.ensure_window_mode_features()` lazily initializes window-only features: `QuickSwitchController`, `AutoSwitchController`, `FocusTracker`, and Media KeepAlive (guarded by `features.media_control_enabled`).
- `OverlayManager.create_overlay(...)` calls `app_instance.ensure_window_mode_features()` when creating `OverlayType.WINDOW` overlays to ensure readiness on demand without incurring startup cost in monitor-only sessions.
- Errors during lazy init are logged; overlay creation proceeds (strict logging, no silent fallback behavior).

##### Swap To Window: Flow, Threading, and Re-entrancy (Spec)

- Selection originates in `OverlayContextMenu` and invokes `DWMOverlay._handle_swap_window(hwnd)` with the chosen HWND.
- `_handle_swap_window` schedules the actual swap on the UI thread via `ThreadManager.run_on_ui_thread` to guarantee Qt/DWM API safety.
- The concrete swap `_swap_source_hwnd(new_hwnd)` performs:
  - Unregisters the current DWM thumbnail (if any) and registers a new one for `new_hwnd`.
  - Stores and uses the handle returned by `DwmRegisterThumbnail` immediately. If registration fails, logs explicitly and raises (strict no-fallback).
  - Recompute destination rect from `IntegratedBorderCanvas.content_rect()` mapped into host coordinates; convert to physical pixels at the edge call only.
  - Apply visibility and opacity (fade-in from 0.0 to configured opacity) and then enforce z-order for host and border overlays.
- Overlay reuse: when reusing an existing overlay instance, `OverlayManager` invokes `overlay.update_source(hwnd)` (if available), which delegates to `_handle_swap_window`. Controllers (QuickSwitch/AutoSwitch) continue to call `_handle_swap_window` directly on the active overlay.
- Re-entrancy guard: `_swap_in_flight` and `_pending_swap_hwnd` coalesce rapid successive requests; only the latest HWND is applied after the current swap completes. This prevents redundant unregister/register churn without violating the no-fallback policy.
- All swap steps (thumbnail ops, rect updates, visibility/opacity) run on the UI thread.

##### Context Menu Actions (Spec)

- Reset Position/Size → `DWMOverlay._handle_reset_position()`
  - Restores configured geometry, updates the destination rect, and synchronizes/raises the border overlay.
- Quit Application → `DWMOverlay._handle_quit_application()`
  - Calls `QCoreApplication.quit()` with explicit error handling and logging.
- Main Window and Subsettings actions are provided by the centralized `OverlayContextMenu` and are wired to the app core; behavior must preserve border overlay visibility while menus are open.

  ##### Switch To Monitor Submenu (Spec)

  - Availability: window and monitor overlays.
  - Population: `utils/monitor_utils.get_all_monitors()`; ignores current target screen (`overlay.capture_target_screen` when present).
  - Actions:
    - Optional "Switch to Monitor Overlay" invokes `overlay._handle_switch_to_monitor_overlay()`.
    - Per-screen entries invoke `overlay._handle_switch_monitor(screen_obj)`; strict no-fallback — missing handler is an error.
  - Naming: uses `screen_obj.name()`; entries are deduped by name.
  - Error handling: explicit logs; empty/invalid monitor lists raise a clear error.
  - Selection handling uses `QGuiApplication.screens()` in `OverlayContextMenu._handle_monitor_selection()`; population remains via `utils.monitor_utils.get_all_monitors()`.

##### Window Enumeration and Icons (Spec)

- `ApplicationCore.get_windows()` provides capturable windows for menus and features; it delegates to `WindowFilter` and `WindowEnumerator`.
- `WindowEnumerator` defers creation of a blank `QPixmap`-based icon until a `QApplication` exists. If not available, it uses an empty `QIcon` and logs explicitly (graceful and logged, not silent). Lazy initialization ensures no QPixmap is created before the application is ready.

##### Z-order / Stacking Enforcement (Spec)

- Native enforcement: both the DWM host window and the border overlay are explicitly placed in the TOPMOST band via `SetWindowPos(HWND_TOPMOST, …)`. Qt `.raise_()` and `sync_with_target()` are used as secondary safeguards.
  - DWM backend correctness: `SetWindowPos` uses `SWP_NOZORDER | SWP_NOACTIVATE` for geometry updates (not `SWP_SHOWWINDOW`). Click-through is controlled via `WS_EX_TRANSPARENT` based on `config.click_through` and is not set unconditionally via `Qt.WindowTransparentForInput`.
- Combined success logic: enforcement is considered successful only if native calls succeed for both host and border windows; failures are logged explicitly (no fallback). Qt `.raise_()` may be invoked as a secondary safeguard but does not constitute fallback semantics nor success criteria.
- Invocation: all enforcement requests are made via `ResourceManager.enforce_z_order(overlay_id)` (directly or via the `OverlayManager` wrapper).
- Canonical ID requirement: backends must pass the exact canonical `overlay.id` to the enforcement API. Constructed or legacy IDs are prohibited. `DWMOverlay._enforce_centralized_z_order()` now uses `self.id`; the legacy `f"dwm_{id(self)}"` path has been removed.
 - Cadence: immediate enforcement on show/geometry/content changes; any debouncing/deferrals are handled internally by `ZOrderManager`. Modules must not implement custom retry loops.
- Verification coalescing: `_ensure_stack_order()` uses a single deferred verification when needed to avoid piling up callbacks during rapid drag/resize operations; no raw `QTimer` usage.
   - Border refresh strategy: uses `update()` instead of `repaint()` to avoid synchronous painting during interactive moves/resizes.
   - Redundant wiring removed: the extra `geometryChanged → _verify_stack_order` connection was eliminated to prevent duplicate native calls during geometry changes.
   - BorderOverlay deferrals: post-show sync/raise behavior is coordinated by `ResourceManager` as needed; this module does not schedule timers directly (no raw `QTimer`).
   - BorderOverlay theming: programmatic border stroke color is used (not QSS) because QSS cannot style the QPainter stroke in this translucent utility window. Documented exception: on the dark theme the stroke is forced to white `#ffffff` for contrast; other themes use `ThemeManager.get_token('border')` (public API). Theme tokens remain unchanged; this exception applies to BorderOverlay only.

##### BorderOverlay Mouse Transparency (Spec)

- Policy: `BorderOverlay` must be fully mouse click-through to underlying windows.
- Qt-level: set `Qt.WA_TransparentForMouseEvents = True` during initialization and reaffirm in `showEvent()` via `_ensure_window_flags()`.
- Native-level (Windows only): apply `WS_EX_TRANSPARENT` using centralized `utils.win32constants.ExtendedWindowStyles.TRANSPARENT` with `GetWindowLongPtrW`/`SetWindowLongPtrW`.
- Lifecycle reassertion: call `_apply_mouse_transparency()` from `__init__` and `showEvent()` (after `_apply_dwm_composition_attributes()`), and ensure `WA_ShowWithoutActivating`/flags remain intact via `_ensure_window_flags()`.
- Platform guard: wrap native calls with `IS_WINDOWS`; no-ops on non-Windows platforms.
- Error policy: best-effort with explicit logs via centralized logger; failures do not crash and are reattempted on subsequent lifecycle events.
- Threading: perform native style application on the UI thread; if `winId()` is not yet available, defer via `ThreadManager.single_shot`.

### Graphics System (`/py/core/graphics`)
- **`overlay/`**: Overlay window management
  - `backends/`: Rendering backends (DWM)
  - `rendering/`: Rendering components
  - `types/`: Type definitions
  - `ui/`: Overlay UI components

### Monitor Overlay (`/py/monitor_overlay`)
- Implementation: DXGI via dxcam (CPU-copy frames).
- Status: Monitor overlay uses DXGI-only capture; no zero-copy/WGC path.

## Section: Core Functionality

### Graphics System
- DXGI-only capture via dxcam (CPU-copy frames); presentation via CPU blit
- Support for multiple monitors with independent settings
- DPI-aware rendering with proper scaling
- Backend-agnostic rendering pipeline
- Efficient texture management
- Shader-based effects and compositing

### Window Management
- Comprehensive window enumeration and filtering
- Window state tracking and event handling
- Z-order and focus management
- **WINDOW STATE PRESERVATION POLICY** (CRITICAL):
  - **NEVER USE SW_SHOWNORMAL**: Forces windows into normal mode, destroying fullscreen/maximized states
  - **PROVEN APPROACH**: Based on `quickswitch_controller._focus_window()` that has never failed
  - **FOCUS STRATEGY**:
    1. Only use `SW_RESTORE` if window is minimized (preserves fullscreen/maximized)
    2. Never use `SW_SHOWNORMAL` or other mode-forcing show commands
    3. Use `SetWindowPos` with `SWP_SHOWWINDOW` flag to ensure visibility without changing state
    4. Fallback progression: simple `SetForegroundWindow` → `SetWindowPos` TOPMOST sequence → thread input attachment → Alt keystroke trick
  - **IMPLEMENTATION**: `DockingManager._bring_hwnd_to_focus()` uses this proven approach
  - **RESULT**: Fullscreen windows stay fullscreen, maximized windows stay maximized, normal windows stay normal
- Window snapping and positioning
- Multi-monitor support
- Window thumbnail generation
 - Window type handling: creation with an unknown type fails fast (raises `ValueError`); no implicit coercion or defaulting to `CUSTOM`.

#### Centralized Window Behavior (Single Source of Truth)

- All drag/resize/snap/cursor logic is centralized in `utils/window/behavior.py`.
- Fixed constants only (non-configurable):
  - `DEFAULT_SNAP_DISTANCE = 30` pixels
  - `DEFAULT_RESIZE_MARGIN = 12` pixels
- Policy: failures are explicit with logs; partial behavior is not allowed. Non-critical issues may be handled gracefully if logged; deferred retries are acceptable.
  - Backend selection semantics: when `preferred_backend != BackendType.AUTO` and the preferred backend is unavailable, creation fails fast with an error log (no implicit substitution). When `preferred_backend == AUTO`, the best available backend is selected with explicit diagnostics.
- Integration requirements for all UI windows/dialogs (e.g., `ui/main_dialog.py`):
  - Use `WindowBehaviorManager(widget=self, min_width=..., min_height=...)`.
  - Monitor Overlay cursor policy: `ui/overlays/monitor/monitor_overlay.py` installs event filters and enables mouse tracking on the overlay and key children to maintain correct edge-resize cursors and immediately reset to Arrow on leave. Drag/resize states are respected (no interference while active). Uses centralized edge detection and cursor helpers from `utils.window.behavior`.
  - Do NOT override snap distance or resize margin anywhere.
  - Delegate mouse events (`mousePressEvent`, `mouseMoveEvent`, `mouseReleaseEvent`, `leaveEvent`) to the manager.
  - All visuals and hover states are styled via QSS; no inline Python styling.
 - Exception: Overlay border is rendered programmatically inside `IntegratedBorderCanvas` for robustness across DPI/Windows context menus; it is DPI-aware, fully opaque, and replaces previous QSS-based borders.
  - When forwarding child mouse events via an event filter, the proxy must expose a QMouseEvent-compatible API: `button()`, `buttons()`, `position()/globalPosition()` and `pos()/globalPos()` so `WindowBehaviorManager` receives complete data.

#### Hotkeys & Switching

- Ownership
  - `HotkeyManager` centralizes all global hotkeys.
  - `QuickSwitchController` owns the quickswitch hotkey registration and action; `OpacityManager` owns only opacity increase/decrease hotkeys.
- Backend policy (strict; no-fallback)
  - Single-key hotkeys (no modifiers) with `suppress=True` use the `keyboard` library backend with suppression.
  - Combinations (2+ keys) are registered via Windows `RegisterHotKey` (non-suppressed). Suppression is not attempted for combos.
  - Registration failures abort and roll back partial state with explicit error logs; no silent success.
- Key parsing
  - `VK_OEM_3` (backtick/tilde) is supported; when `win32con.VK_OEM_3` is unavailable, the canonical value `0xC0` is used explicitly.
- Logging (mandatory)
  - Registration path logs: selected backend, parsed modifiers/VK, success/failure.
  - Dispatch path logs: backend match and hotkey id; action invocation logs from controllers.
  - `core/graphics/overlay_host.py#OverlayHost.mouseDoubleClickEvent` invokes `QuickSwitchController.quickswitch("overlay_host.double_click")` with before/after debug logs; exceptions are captured and logged; event is accepted.

##### QuickSwitch: API, Gating, and Events (Spec)

- API
  - `core/switching/quickswitch_controller.py#QuickSwitchController.quickswitch(source: Optional[str] = None) -> None`
  - `source` conveys trigger origin, e.g., `"hotkey"`, `"overlay.double_click"` (from `ui/overlays/integrated_border_canvas.py`), or `"overlay_host.double_click"` (from `core/graphics/overlay_host.py`).
- Re-entrancy gating
  - Lock-free in-flight flag prevents concurrent runs; on contention the call is suppressed.
  - Publishes: `switch.reentry_suppressed` with `{ reason: "in_progress", source }` (publisher `source="QuickSwitchController"`).
- Cooldown
  - Fixed 800ms cooldown enforced via `time.monotonic()`; early gate on any thread with authoritative check on the UI thread.
  - Publishes: `switch.cooldown_suppressed` with `{ until, now, source }`.
- Overlay-lock gating
  - If the active overlay is locked, quickswitch is suppressed.
  - Publishes: `switch.lock_suppressed` with `{ overlay_id, source }` (publisher `source="QuickSwitchController"`).
- Success path
  - MRU candidates limited to 7 and optionally monitor-filtered by `features.display_locked_switching`.
  - Swap is executed on the UI thread via `ThreadManager.run_on_ui_thread` calling overlay `_handle_swap_window(hwnd)`.
  - Detailed `QUICKSWITCH` logs include selection reason and, when applicable, monitor indices.

##### Media Control Routing Policy (Browsers & Volume)

- __Firefox passthrough bypass (Media Control ON)__
  - When the overlay target resolves to Firefox, SPACE and ARROW keys are delivered via direct passthrough (nonmedia behavior) from `core/input/key_passthrough_controller.py#KeyPassthroughController.passthrough_key()`.
  - This bypasses `MediaController` for those keys to ensure reliable play/pause and seek behavior in Firefox content windows.
  - App detection uses the public helper `core/media/media_controller.py#MediaController.detect_app_for_hwnd(hwnd)`.
  - Exception (Arrow→Volume remap): when `features.media_control_enabled` is True, `VK_UP`/`VK_DOWN` are first remapped to `VK_VOLUME_UP`/`VK_VOLUME_DOWN` and handled by the volume path (press-and-hold) rather than being delivered as arrow keys.

- __Browser child-window targeting__
  - For browser key routing paths that use `MediaController` (e.g., Space with WM_CHAR on Firefox; K/Space on Chromium), `_send_browser_hotkey(...)` enumerates child HWNDs and prioritizes those whose window titles contain streaming/site keywords: `youtube, netflix, twitch, spotify, vimeo, disney, prime video, hulu, plex, jellyfin`.
  - Keys are sent to prioritized children first, then others, then the top-level as a last resort. Each attempt checks responsiveness.

- __Volume control policy (final fallback)__
  - Order of handling in `_send_media_command_safe(...)` for `APPCOMMAND_VOLUME_{UP,DOWN}`:
    1) Prefer per-app session volume via `utils.audio.session_volume.adjust_session_volume_for_hwnd(hwnd, ±0.05)`.
    2) Fallback to app-declared volume hotkeys (when present in the catalog).
    3) Final fallback to targeted global mixer via `WM_APPCOMMAND` with timeout safeguards.
  - This ensures apps like mpv.net adjust volume even when a per-app audio session is not found.

- __Public app detection helper__
  - `MediaController.detect_app_for_hwnd(hwnd)` is exposed to consumers (e.g., KeyPassthroughController) as a safe wrapper over `_detect_app_for_hwnd`.

- __Volume key press-and-hold behavior (controller)__
  - Applies to VK_VOLUME_UP (0xAF) and VK_VOLUME_DOWN (0xAE), and to VK_UP/VK_DOWN when media control is enabled (remapped early to VK_VOLUME_UP/DOWN).
  - On press: an immediate step is executed via `MediaController.volume_*_for_hwnd(target_hwnd)` and a repeating timer is started with the initial delay/interval settings described above.
  - On release: the active hold is canceled; no further repeats occur.
  - Environment gates: media control must be enabled, overlay must be focused, and a valid target HWND is required. A small suppression window (~35ms) still applies to smooth rapid events.
  - Remap implementation details: remapping occurs at the start of `KeyPassthroughController.passthrough_key(...)`, `.press_passthrough_key(...)`, and `.release_passthrough_key(...)`, ensuring consistent downstream behavior and avoiding accidental arrow-key passthrough while media control is active. `VK_LEFT`/`VK_RIGHT` are not remapped to volume keys.

- __Non-media browser-aware fallback (Media Control OFF)__
  - When `features.media_control_enabled` is OFF and the overlay target HWND belongs to a supported browser app (Chrome, Edge, Firefox, Discord), `KeyPassthroughController` attempts a browser-aware child targeting fallback for SPACE/LEFT/RIGHT.
  - Implementation: calls `MediaController._send_browser_hotkey(hwnd, vk, include_char?, char_code?)` to enumerate/prioritize child content windows and send the key there first, then falls back to the top-level HWND.
  - Firefox quirk: SPACE may include a WM_CHAR alongside WM_KEYDOWN/UP for improved reliability.
  - Success publishes `key.passthrough.forwarded` with `note="browser-child"`. If not applicable or unsuccessful, normal PostMessage passthrough is used.

##### Media Controller: App Catalog Clarifications

- Detection uses exact, case-insensitive process name matches.
- mpv.net: process corrected to `mpvnet.exe` (not `mpv.net.exe`) to ensure detection succeeds.
- **Multi-Window Targeting**: `_find_window_by_app()` prioritizes overlay target window when multiple instances exist, ensuring media commands target the exact window displayed in the overlay rather than the first found window of that application type.
- MPC variants (HC/BE 32/64): `safe_methods` include `wm_command` and `hotkeys`.
  - When `media.wm_command_ids.{app}` do not provide IDs, declared `hotkeys` are attempted for play/pause/next/previous/stop.
  - This maintains identical functionality without introducing degraded fallbacks; failures are logged explicitly.

##### Display Locked Switching (Monitor-Scoped MRU)

- Setting key: `features.display_locked_switching` (bool; default: false)
- Behavior: When enabled, both `QuickSwitchController` and `ForegroundAutoswitchController` filter MRU candidates to those on the same monitor as the overlay's current source window.
  - Monitor detection uses `utils.window.monitors.find_monitor_for_window(QPoint, QSize) -> int`.
  - Window geometry is obtained via `utils.window_validation.get_window_rect(hwnd)`; equality of monitor indices determines eligibility.
  - Applied on all selection paths: standard quickswitch candidate ordering, autoswitch (fg==current_src) cyclic selection, and autoswitch emergency recovery.
- Live updates: The SubSettingsDialog toggles this key and notifies controllers (`QuickSwitchController.update_hotkeys()`; `ForegroundAutoswitchController.apply_settings()`) so behavior updates without restart.

#### Shutdown Sequence (ApplicationCore)

- `ApplicationCore.shutdown()` performs orderly teardown and must be idempotent and non-throwing (errors are logged, not raised). Sequence includes:
  - `WindowManagerAdapter.shutdown()` — shuts down/cleans up underlying impl (prefers `impl.shutdown()` if available, else `impl.cleanup()`), clears pending operations, resets initialization flags. Never raises; logs on error.
  - `Logger.shutdown()` — restores `sys.excepthook`, flushes/closes/removes handlers, disables the logger, and clears singleton/configured state so logging can be re-initialized later. Never raises.
- Policy: strict no-fallback remains; graceful failure is allowed only with explicit logs. Shutdown paths must not attempt implicit substitutions.

##### Centralized MRU Selection (Cyclic)

- `core/switching/selection.py` provides `compute_next_selection(...)` used by both QuickSwitch and AutoSwitch for deterministic, per-overlay cycling.
  - Inputs: MRU `candidates`, `filtered` list, `cur_hwnd`, `current_src`, `cycle_last_by_overlay`, `overlay_id`, `pick_from_zorder`, `is_valid`.
  - Order of preference: after last selection → after current foreground → Z-order hint → first valid.
  - Returns: chosen hwnd, display index, reason, ordered list, start index, and reference keys for logging.
  - Strict policy: if no valid candidates remain, selection explicitly fails and logs; no implicit fallbacks.
  - Candidate construction (controllers): ordered = [foreground, current_src, MRU...], preserving order and uniqueness; focus targets are filtered by excluding only the current foreground.
  - MRU capacity and limits: MRUManager default capacity is 7; controllers request up to 7 MRU candidates to match.
  - MRU candidate limit: 7 (applied consistently in QuickSwitch and AutoSwitch).
  - Validation hardening: exclude OS/UI surfaces like Alt-Tab "Task Switching"; do not exclude typical user apps (e.g., File Explorer).
  - Centralized selection candidate ordering: MRU candidates are ordered by most recent usage, followed by the current foreground window, and then by Z-order (if specified). The first valid candidate is chosen.

##### Autoswitch Behavior (Refined)

- Debounced polling remains; however:
  - MRU is recorded only when an actual overlay source change occurs (not every poll).
  - Redundant swaps are skipped when the target equals the current overlay source (`_current_source_hwnd`) or the last applied.
  - Maintains strict window validation and UI-thread safe swap via overlay `_handle_swap_window(hwnd)`.
  - No-fallback: Autoswitch never seeds from Z-order; `pick_from_zorder` is a no-op and selection aborts explicitly when no valid target exists.
  - Cycle state reset: if filtered targets are empty or selection returns no choice, the per-overlay cycle state is cleared to avoid sticky cycling and allow deterministic recovery.
  - Monitor-awareness (backend support, logging only): logs monitor indices for foreground and current source via `utils.window.monitors.find_monitor_for_window` to aid future multi-monitor features.

##### MainDialog Badge System (UI Behavioral Spec)

- Purpose: Display a decorative badge in the main window that users can change via double-click.
- Implementation:
  - Widget: `QLabel#badgeLabel` with a drop shadow (explicitly permitted Python-side visual effect).
  - Assets: `resources/Badge*.png`.
  - Defaults: On first run, loads `resources/Badge19.png`.
  - Interaction: Double-clicking the badge selects a random `Badge*.png` (preferably different from current).
  - Persistence: Stores the selected filename (not full path) in `SettingsManager` under key `ui.badge_file`.
  - Layout: Positioned at the bottom-right of `main_frame`; re-positioned on show/resize.
  - Input: Badge must receive mouse events; it is excluded from draggable regions to avoid interfering with window dragging.

### Theme System
- Light/dark theme support
- Custom styling for all UI components
- Runtime theme switching
- Style sheet management
- High-contrast mode support
- Custom color schemes
 - Canonical setting key for theme is `theme` in `SettingsManager`.
 - Overlay border visuals are programmatic (painted in code) rather than QSS; QSS still governs backdrop bars and all other UI visuals. This is an approved, narrow exception to the no-inline-styles policy to eliminate DWM border "melt".
 - Checkbox indicator visuals in dialogs use a programmatic component `ui/components/circle_checkbox.py#CircleCheckBox`. It draws a circular indicator with an optional inner filled circle when checked, using colors fetched from `ThemeManager` tokens at paint time. All QSS styling for the `QCheckBox::indicator` was removed to avoid conflicts and ensure pixel-perfect circles across DPIs.

##### Overlay Border Shadows (Removed)

Legacy shadow-related settings and behavior have been removed for cleanliness and simplicity. The keys under `overlay.shadow.*` are no longer recognized or persisted. Border visuals remain programmatic in `ui/overlays/integrated_border_canvas.py` and are styled via theme tokens where applicable; no shadow configuration exists.

#### Theme Sync Policy (Single Source of Truth)

- `themes/dark.qss` is the canonical source. Do not diverge structure or selectors between themes.
- `themes/light.qss` is always an exact, color-inverted mirror of `dark.qss`:
  - Invert RGB components for `#RGB`, `#RRGGBB`, `rgb(r,g,b)`, and `rgba(r,g,b,a)`
  - Preserve alpha values, sizes, margins, radii, and all non-color tokens
  - Invert color tokens even when present in inline comments (e.g., color-picker annotations)
- Never edit `light.qss` manually. After any change to `dark.qss`, regenerate `light.qss` by running:

```bash
python scripts/sync_themes.py
```

- The script `scripts/sync_themes.py` performs full-file inversion and overwrites `themes/light.qss`.

##### SubSettingsDialog Styling Details

- Container: `QDialog#subsettingsDialog`
  - `border: none` (prevents double borders; visual border drawn by an inner frame)
  - `border-radius: 8px`
- Outer border frame: `QFrame#settingsDialogBorder`
  - `border: 2px solid` (white on dark, black on light)
  - `border-radius: 8px`
  - `background: transparent`
- Title bar: `#subsettingsDialog > #titleFrame`
  - `border: none`
  - `border-bottom: 2px solid` for a straight separator line (no rounding)
  - Title bar background matches theme title bar base color

#### SubSettingsDialog (UI Spec)

- Location: `ui/dialogs/subsettings_dialog.py` (exported via `ui/subsettings_dialog.py` as `SubSettingsDialog`)
- Purpose: Frameless subsettings dialog for theme selection, opacity hotkeys, and feature toggles
- Styling: QSS-first with a documented programmatic exception for checkbox indicators. Object names include:
  - Dialog: `subsettingsDialog`
  - Title bar frame: `titleFrame`; Close button: `closeButton`
  - Content frame: `settingsContentFrame`
  - Widgets: `SettingsComboBox`, `SettingsCheckBox`, `SettingsKeySequenceEdit`
    - Checkboxes use `ui/components/circle_checkbox.py#CircleCheckBox` for the indicator drawing (theme-driven via `ThemeManager`). QSS `QCheckBox::indicator` rules are intentionally absent.
- Window behavior:
  - Uses `WindowBehaviorManager(self, 380, 520)`; forwards mouse events to manager
  - Draggable region is the dynamic height of `title_bar`
  - `resizeEvent` only updates border geometry; no widget reconstruction
- Settings integration (canonical keys):
  - Theme: `theme` (string: `dark`|`light`) — applied via `ThemeManager.apply_theme()`
    - No aliases or fallback are supported; only `theme` is recognized
  - Opacity hotkeys:
    - `hotkeys.opacity_enabled` (bool)
    - `hotkeys.opacity_decrease` (string; default `-`)
    - `hotkeys.opacity_increase` (string; default `=`)
    - Changes trigger `OpacityManager.update_hotkeys()` for live updates
  - Quickswitch hotkey:
    - `hotkeys.opacity_quickswitch` (string; default `` ` ``)
    - Changes trigger `QuickSwitchController.update_hotkeys()` for live re-registration via `HotkeyManager` (system/global)
  - Feature toggles:
    - `features.autoswitch_enabled` (bool)
    - `features.keypassthrough_enabled` (bool)
    - `features.media_control_enabled` (bool; default: false) — enables media key routing and background monitoring
    - `features.display_locked_switching` (bool; default: false)
      - When enabled, restrict both QuickSwitch and AutoSwitch MRU candidates to windows on the same physical monitor as the overlay's current content window. Monitor is determined via `utils.window.monitors.find_monitor_for_window` using the window's rect from `utils.window_validation.get_window_rect`.
  - Overlay visuals:
    - `overlay.rounded_borders` (bool) — applied live to all `BorderOverlay` instances and persisted
- Behavior:
  - Close button wired to `close()`
  - No inline styles; theme application handled centrally by `ThemeManager`
  - Debug logging on setting saves and changes for auditability
  - Visual gap fix: QSS margins for `#titleFrame` and `QFrame#settingsContentFrame` adjusted to remove the transparent seam

### Resource Management
- Centralized resource tracking and cleanup via `core/resources/manager.py` (`ResourceManager`)
- No-fallback policy: invalid operations fail explicitly with clear errors; no partial behavior
- Weakref requirement: objects must be weak-referenceable; otherwise `register()` raises `ValueError` and rolls back partial state
- Deterministic cleanup: per-resource cleanup handler is executed if provided or derived from `CleanupProtocol.cleanup`
- Thread-safe access using `threading.RLock`; automatic cleanup on shutdown and interpreter exit

#### Filesystem and OS Handle Cleanup Helpers

- Helper APIs (built-in registrations in `ResourceManager`):
  - `register_temp_file(target, description="", delete=True, **metadata)`
    - Target may be a file-like object or a path-like.
    - Cleanup policy: flush if file-like, attempt `close()` if present, then unlink path when available and `delete=True`.
    - Group: `filesystem` (via `ResourceType.FILE_HANDLE`). Idempotent; failures are logged.
  - `register_temp_dir(path, description="", ignore_errors=True, **metadata)`
    - Recursively deletes directory via `shutil.rmtree(path, ignore_errors=ignore_errors)`.
    - Group: `filesystem`. Idempotent; logs failures.
  - `register_os_handle(handle, description="", **metadata)`
    - Windows: if `handle` is an `int`, attempts `CloseHandle` via Win32; otherwise tries `close()` if available.
    - Non-Windows: attempts `close()` if available; otherwise logs platform notice.
    - Group: `filesystem` (Filesystem/OS bucket). Idempotent; logs failures.

- Usage examples:

```python
from core.resources import get_resource_manager

rm = get_resource_manager()

# Temp file by path
fid = rm.register_temp_file("C:/Temp/spq_tmp_abc.txt", description="scratch file")

# Temp file via file-like
f = open("C:/Temp/spq_tmp_xyz.bin", "wb")
fid2 = rm.register_temp_file(f, description="binary tmp")

# Temp directory
did = rm.register_temp_dir("C:/Temp/spq_job_123/", description="job workspace", ignore_errors=True)

# OS handle (Windows)
hid = rm.register_os_handle(some_raw_handle_int, description="DWM dup handle")

# Later (on shutdown or explicitly)
rm.cleanup_all()
```

- Cleanup ordering context:
  - Helpers participate in deterministic ordering via `ResourceInfo.group`.
  - Filesystem/OS resources are cleaned after Qt and Network/DB groups.
  - Within-group order is by `cleanup_priority` (lower first) then registration time.

#### Network and Database Cleanup Helpers

- Helper APIs:
  - `register_network(conn, description="", **metadata)`
    - Cleanup policy: best-effort `shutdown()` (when available/signature may vary) then `close()`.
    - Group: `network_db` (via `ResourceType.NETWORK_CONNECTION`). Idempotent; failures logged.
  - `register_db(session, description="", rollback_on_cleanup: bool = False, **metadata)`
    - Cleanup policy: optional `rollback()` when `rollback_on_cleanup=True`, then `close()` and/or `dispose()` if provided.
    - Group: `network_db` (via `ResourceType.DATABASE_CONNECTION`). Idempotent; failures logged.
  - `register_network_pool(pool, description="", **metadata)`
    - Member cleanup: if iterable, each member via network helper; then pool-level `closeall()`/`close()`/`dispose()`.
    - Group: `network_db` (resource type `NETWORK_CONNECTION`, `metadata.pool=True`).
  - `register_db_pool(pool, description="", rollback_members: bool = False, **metadata)`
    - Member cleanup: if iterable, each member via DB helper with `rollback_members`; then pool-level `closeall()`/`dispose()`/`close()`.
    - Group: `network_db` (resource type `DATABASE_CONNECTION`, `metadata.pool=True`).

- Notes:
  - Even if the app has minimal/no network/DB usage today, these helpers provide a standardized, centralized path for future integrations.
  - Ordering: Network/DB resources are cleaned just after Qt and before OpenGL and Filesystem/OS.
- Public API (`core/resources/__init__.py`):
  - `get_resource_manager()` returns the singleton instance
  - `register_resource(resource, resource_type, description, **metadata)`
  - `unregister_resource(resource_id) -> bool`
  - `get_resource(resource_id)` delegates to `ResourceManager.get()`
  - `list_resources(resource_type=None, include_metadata=False)` returns IDs by default; when `include_metadata=True`, returns `ResourceInfo` objects
  - `cleanup_all()`, `shutdown()`; `cleanup` helpers include `register_pycache_cleaner()` and `register_cleanup_handler()` which now execute the stored callable during cleanup

### Media Control System

**Purpose**: Centralized media key routing and application control with crash protection and display-locked awareness.

**Architecture**:
- `core/media/media_controller.py`: `MediaController` — main controller with app catalog, window enumeration, and command routing
- `core/media/keepalive.py`: `MediaPlayerKeepAlive` — background monitoring service with responsiveness checking
- `core/input/key_passthrough_controller.py`: Enhanced with media key routing capabilities
- `ui/dialogs/subsettings_dialog.py`: "Media Control" toggle in settings UI

**Key Features**:
- **Enhanced App Catalog Management**: Comprehensive catalog of media applications with crash-prone flagging, safe methods, and extensive hotkey mappings
- **Window Enumeration**: Cached window discovery with filtering for media applications and responsiveness checking
- **Advanced Command Routing**: Multi-tier fallback system (media commands → hotkeys → spacebar) with enhanced app-specific logic
- **Enhanced Browser Support**: Advanced child window handling for Chrome/Firefox/Edge/Discord with multiple routing methods
- **Robust Crash Protection**: Pre-command responsiveness checking and automatic detection of unresponsive applications
- **Improved MPC Player Support**: Primary (arrow keys) and fallback (bracket keys) hotkey support for crash-prone MPC variants
- **Background Monitoring**: Continuous responsiveness checking with automatic catalog updates
- **Settings Integration**: Off-by-default policy with live settings updates

**Settings Keys**:
- `features.media_control_enabled` (bool; default: false) — master toggle for all media control functionality

**API Endpoints**:
- MediaController: `play_pause()`, `next()`, `previous()`, `stop()`, `get_running_media_apps()`
- MediaPlayerKeepAlive: `start()`, `stop()`, `get_monitored_apps()`, `force_check(hwnd)`

**Threading Policy**:
- All operations use `ThreadManager` for async work and UI thread dispatch
- No focus stealing; all messaging uses safe, timeout-protected Win32 calls
- Background monitoring uses self-rescheduling single-shot timers (5s intervals)

**Event Integration**:
- KeyPassthroughController publishes `key.passthrough.media_routed` events
- Media keys (VK_MEDIA_PLAY_PAUSE, VK_MEDIA_STOP, VK_MEDIA_PREV_TRACK, VK_MEDIA_NEXT_TRACK) are intercepted before standard passthrough
- Settings changes trigger automatic start/stop of monitoring service

**Logging Prefixes**:
- `MEDIA_CTRL` — MediaController operations
- `MEDIA_KEEPALIVE` — Background monitoring service
- `KEYPASS` — Enhanced with media routing logs

### Event System
- Type-safe event handling
- Priority-based event dispatching
- Thread-safe event queue
- Event filtering
- Cross-thread event delivery
- Event debugging and profiling

### Settings Management
- Hierarchical configuration
- Type-safe settings access
- Change notifications
- Settings persistence
- Validation and schema
- Import/export functionality

## Section: Architecture Overview
- **Core**: Centralized application core with explicit dependency injection for managing application state and shared functionality
  - **`core/__init__.py`**: Core package initialization and service access
    - `Core` class: Central application core with explicit dependencies
    - Provides direct access to core services via `core.resources`, `core.threads`, etc.
    - Implements the composition root pattern
    - Handles service lifecycle and cleanup
    - Type hints for all public APIs
  - **`core/application/`**: Application lifecycle and core services
    - `ApplicationCore`: Central application core implementation
      - Manages service lifecycle and dependency injection
      - Provides access to all core services through type-hinted properties
      - Handles initialization order and error handling
      - Implements proper cleanup on shutdown
    - `ApplicationInstanceManager`: Single instance enforcement using OS-level mutex
      - **Cross-process detection**: Uses Win32 named mutex for inter-process coordination (not intra-process thread synchronization)
      - **Prevents keyboard hook conflicts**: Multiple instances with conflicting system-wide keyboard hooks cause modifier key state corruption
      - **Enforced in main.py**: Checks before Qt initialization, exits cleanly if another instance detected
      - **Policy clarification**: OS mutex for cross-process detection is appropriate; ThreadManager lock-free patterns apply to intra-process threading only
    - `ApplicationLifecycle`: Application startup/shutdown logic
  - **Design Principles**:
    - **Explicit Dependency Injection**: Services are injected through constructors and properties
    - **No Global State**: No global service locator patterns
    - **Composition Root**: All services are composed at application startup
    - **Service Lifecycle**: Manages service initialization and cleanup
    - **Type Safety**: Strong typing for all service dependencies
    - **Testability**: Services can be easily mocked for testing
    - **Event System**: Thread-safe publish/subscribe with Qt signal integration
    - **Settings Management**: Hierarchical configuration with type safety and validation
    - **Resource Management**: Centralized resource tracking and cleanup
    - **Threading**: Managed thread pools and task scheduling
- **Graphics System**: Comprehensive graphics and overlay management
  - **`core/graphics/`**: Centralized graphics functionality
    - **Overlay System**: Window overlays with hardware acceleration
      - `OverlayManager`: Central overlay management with MRU tracking
      - Backend support for DWM and D3D11/WGC rendering
      - Thread-safe operations with overlay locking
      - Automatic resource cleanup
    - **Rendering**: D3D11 swapchain presenter with centralized DXGI interop (`core.graphics.dxgi_interop`)
      - Support for multiple rendering backends
      - Efficient texture management
      - Shader and rendering pipeline management
- **Screen Capture**: DXGI Desktop Duplication via dxcam (DXGI-only)
- **Threading**: Multi-threaded architecture with:
  - Main thread: UI and window management
  - Capture thread: Screen capture and frame processing
  - Render thread: D3D11/WGC presentation
  - Thread manager: core/threading/manager.py (canonical, deduplicated)
  - All timer and hotkey logic must use the canonical thread manager in core.threading.manager.
- **Window Management**: Centralized window management with:
  - `WindowManagerAdapter` in `core/window_adapter.py`: Qt-compatible adapter
  - `WindowManagerImpl` in `core/window_manager_impl.py`: Core implementation
  - `WindowEnumerator` in `core/application/window_enumerator.py`: Window enumeration
  - Thread-safe window operations and event handling
  - Support for multiple windows and overlays
- **Hotkey Management**: Centralized in `core/hotkeys/manager.py` with:
  - Thread-safe hotkey registration/unregistration
  - Support for multiple hotkey profiles
  - Conflict detection and resolution
  - Clean resource management
  - Global hotkey suppression for opacity adjustment handled by `core/opacity/manager.py`

#### QuickSwitch and AutoSwitch Controllers

- QuickSwitchController (`core/switching/quickswitch_controller.py`)
  - Ownership: Sole owner of the quickswitch feature and its hotkey.
  - Settings:
    - `hotkeys.opacity_quickswitch` (string; default "`")
  - Responsibilities:
    - Load and validate the quickswitch hotkey from settings.
    - Register/unregister the quickswitch hotkey via `HotkeyManager` as a system/global hotkey (centralized).
    - Expose `quickswitch()` to swap the active overlay’s source to the current foreground window.
    - Provide `update_hotkeys()` for live reloading when settings change.
  - Integration:
    - Uses `OverlayManager` to locate the active overlay; calls the overlay’s `_handle_swap_window(hwnd)` on the UI thread using `ThreadManager`.
    - `OverlayHost.mouseDoubleClickEvent` triggers `QuickSwitchController.quickswitch()` for mouse-driven switching.
  - Logging: All logs use the `QUICKSWITCH` prefix. No silent fallbacks.

- ForegroundAutoswitchController (`core/switching/autoswitch_controller.py`)
  - Ownership: Observes foreground window focus changes and requests overlay swaps when stable.
  - NOTE: This is NOT related to closed-window monitoring (see ClosedWindowSwitchManager).
  - Settings:
    - `features.autoswitch_enabled` (bool)
  - Responsibilities:
    - Poll foreground window on a Qt timer with debounce (`STABLE_DEBOUNCE_MS`).
    - Strictly validate target windows; skip invalid candidates.
    - Safely call the active overlay’s `_handle_swap_window(hwnd)` on the UI thread.
    - `apply_settings()` applies enable/disable immediately.
  - Logging: All logs use the `FOREGROUND_AUTOSWITCH` prefix. No silent fallbacks.

- Initialization Strategy (composition root):
  - Controllers are initialized at application startup in `core/application/core.py (ApplicationCore)` for clean, centralized lifecycle:
    - After initializing OpacityManager, construct singletons via `get_quickswitch_controller()` and `get_foreground_autoswitch_controller()`.
    - This ensures hotkeys and autoswitch observation are active whenever overlays exist and keeps ownership centralized.
  - Scope boundaries:
    - `OpacityManager` owns only opacity increase/decrease hotkeys. It does not register the quickswitch key.
    - The quickswitch key is loaded and owned by `QuickSwitchController` (move any residual quickswitch key handling out of `OpacityManager`).
  - Settings wiring:
    - `SubSettingsDialog` calls `QuickSwitchController.update_hotkeys()` when the quickswitch key changes and `ForegroundAutoswitchController.apply_settings()` when the autoswitch toggle changes.
- **Theme System**: Centralized theme management in `utils/theme/manager.py` with:
  - Support for light/dark themes
  - Resource-aware asset loading and caching
  - Automatic cleanup of theme resources
- **Resource Management**: Centralized resource handling in `core/resource/manager.py` with:
  - Thread-safe resource registration and cleanup
  - Support for common app resources; GPU resource management is handled in the D3D11 presenter path
  - Automatic cleanup of registered resources
  - Resource tracking and debugging capabilities
  - Integration with theme system and D3D11 renderer
  - **Event System**: Centralized event handling in `core/events/` with:
  - **Singleton Pattern**: Global `event_system` instance for application-wide access
  - **Modular Design**:
    - `base.py`: Base event classes and interfaces
    - `manager.py`: Core event dispatching and subscription logic
    - `window_events.py`: Window-related event types
    - `app_events.py`: Application-level event types
    - `qt_integration.py`: Qt event system integration
  - **Key Features**:
    - Type-safe event handling with dataclass-based events
    - Priority-based callback ordering (0-1000, higher executes first)
    - Thread-safe operations with `threading.RLock`
    - Support for wildcard event patterns (e.g., 'window.*')
    - Event filtering and transformation
    - Automatic cleanup of event handlers
  - **Migration**:
    - Legacy `utils/event_system` has been deprecated
    - Comprehensive migration guide available in `core/events/MIGRATION_GUIDE.md`
    - All core components and tests have been updated to use the new system
- **Settings Management**: Centralized in `core/settings.py` with:
  - Hierarchical configuration
  - Type safety and validation
  - Change notifications
  - Automatic persistence

## Section: Input and Output Contracts
### Inputs:
- System metrics (CPU, GPU, memory usage)
- Application window handles
- User configuration settings
- Keyboard/mouse input for interaction
- Theme settings (light/dark mode)
- Hotkey bindings and actions
- Window state and position data

### Outputs:
- Overlay display on target monitor(s)
- Configuration files (JSON format)
- Log files for debugging (rotating log files)
- Visual feedback for user interactions
- System notifications for important events
- Performance metrics and diagnostics

## Section: Edge Cases and Constraints
- Must handle monitor resolution changes and display reconfiguration
- Should gracefully handle loss of capture source (window/minimized/closed)
- Memory usage must be optimized for continuous operation (target <100MB idle)
- Resource cleanup must be handled through the centralized ResourceManager
- All system resources must be properly tracked and released on application exit
- Must support Windows 10/11 with various DPI scaling factors (100%-300%)
- Must maintain 60 FPS performance during normal operation
- Should provide visual feedback during all user interactions
- Must respect and adapt to system theme changes
- Hotkey conflicts must be detected and resolved gracefully
- Must handle rapid mode switching and configuration changes
- Should recover from GPU context loss
- Must maintain thread safety across all operations

## Section: Centralization Architecture

### Core Principles
1. **Single Source of Truth**: All shared state and functionality is managed by `app_core`
2. **Separation of Concerns**: Clear boundaries between UI, business logic, and utilities
3. **Dependency Injection**: Components receive their dependencies through `app_core`
4. **Testability**: Centralized state makes it easier to test components in isolation
5. **Thread Safety**: All public APIs must be thread-safe
6. **Resource Management**: All resources must be managed through centralized managers
7. **No Circular Dependencies**: Strict enforcement of acyclic dependency graph between modules

## Media Control Policy

Media control is managed by `core.media.media_controller.MediaController` with the following policies:

### Enhanced Media Control Features (2025-09-05)

- **Targeted Browser Selection**: `_send_browser_media_command_targeted()` intelligently matches DWM overlay source HWND to the correct browser child window using title/class similarity scoring and media content keyword detection (YouTube, Netflix, Spotify, etc.)
- **Enhanced Application Detection**: `_derive_app_name_from_context()` resolves UNKNOWN applications through comprehensive fallback logic using process names, window titles, class names, and media characteristics (supports 40+ media applications)
- **Continuous Volume Control**: Key hold detection enables smooth 0-100% volume ramping via `handle_volume_key_press/release()` methods:
  - Immediate response: 2% step on key press (configurable via `media.volume_step`)
  - Continuous ramping: After 0.5s hold, switches to 1% steps every 50ms
  - Auto-bounds detection: Stops at 0% or 100% volume limits
  - Key release handling: Stops continuous adjustment instantly
- **Volume Step Configuration**: Default reduced from 5% to 2% for finer control while maintaining [1%-25%] safety range

1. **Overlay-Aware Routing**: Commands prioritize the app matching the current overlay target hwnd
2. **App-Specific Routing**: Commands are routed to specific applications based on process detection
3. **Crash Protection**: Responsiveness checking before sending commands to crash-prone applications
4. **Safe Method Selection**: Each app defines safe command methods (media_command, spacebar, hotkeys)
5. **Browser Enhancement**: Special handling for browsers with child window enumeration for embedded players
6. **Fallback Chains**: Multiple delivery methods attempted in order of safety and reliability
7. **Context Priority**: `get_preferred_app()` checks overlay target before settings-based preferences

#### Volume Commands (Per-App Session Volume)
- Preferred path: adjust the application's audio session from the target window handle using `utils.audio.session_volume.adjust_session_volume_for_hwnd(hwnd, delta)`.
- Step size: ±0.05 per command for `volume_up`/`volume_down`.
- Fallback: if session control is unavailable or fails, use app-defined hotkeys when present in the catalog.
- Never perform global volume changes via `WM_APPCOMMAND` (system-wide volume is not modified by this feature).
- Platform: Windows-only; safe no-op on other platforms.
- Logging: successes and failures are explicitly logged; no silent fallbacks.

#### Capture & Presentation Policy (Spec)

- DXGI-only monitor capture via dxcam (CPU-copy frames).
- No zero-copy/WinRT path; no swapchain presentation.
- Rendering/presentation is performed via QWidget CPU blit.

## Section: Graphics Presentation

- Presentation path: QWidget blitter only.
- Setting `graphics.presentation` and swapchain host have been removed.

## Section: Technical Notes and Dependencies

## Section: Event System Architecture

### Overview
The event system provides a unified, type-safe way to handle application events with the following features:
- Centralized event dispatching through `core/events.py`
- Type-safe event definitions using Python's `typing` module
- Support for event filtering and prioritization
- Seamless Qt integration
- Thread-safe event handling
- Support for both synchronous and asynchronous event processing

### Key Components
- **EventDispatcher**: Central hub for event routing (singleton pattern)
- **Event Types**: Strongly-typed event classes in `core/events.py`
- **Event Handlers**: Type-annotated callback functions
- **Event Filters**: For intercepting and processing events before they reach their target
- **Event Filters**: For intercepting and processing events
- **Qt Integration**: Automatic conversion between Qt and internal events
- **Migration Tools**: Utilities for gradual adoption

### Event Types
- **Window Events**: Movement, resizing, focus changes
- **Input Events**: Mouse, keyboard, touch
- **Application Events**: Settings changes, theme updates
- **Custom Events**: Extensible for application-specific needs

### Best Practices
1. Use strongly-typed events when possible
2. Keep event handlers short and efficient
3. Use event filters for cross-cutting concerns
4. Always clean up event handlers when no longer needed

### Migration Status
- [x] Core event system implemented
- [x] Qt integration layer complete
- [x] Migration utilities available
- [x] BorderWidget refactored
- [x] SystemTrayManager integrated
- [x] Main application refactored to use centralized services
- [x] Circular import issues resolved
- [ ] All overlays migrated to use new event system
- [ ] Legacy event handling removed

## Section: Dependencies

### Core Dependencies
- **Python**: 3.8+
- **PySide6**: 6.9.1+ (Qt6 bindings for Python)
- **comtypes**: Optional COM interop helpers for DXGI (if needed by utilities)
- **dxcam**: DXGI Desktop Duplication capture backend (DXGI-only)
- **NumPy**: Required for array operations
- **psutil**: For system monitoring
- **keyboard**: For global hotkey support
- **pywin32**: For Windows-specific functionality

### D3D11 Implementation
- **Feature Level**: D3D11 (11_0+) with BGRA support
- **Device Management**: Centralized `ID3D11Device` via `core.graphics.d3d11_device` (if present); shared where applicable
- **Texture Handling**: Efficient texture and resource management
- **Error Handling**: Explicit logging; strict no-fallback policy

### Platform Support
- **Primary Platform**: Windows (pywin32 required)
- **Multi-Monitor**: Full support for multiple monitor configurations
- **High-DPI**: Proper handling of high-DPI displays
- **Theming**: System theme detection and support for light/dark modes

### Performance Considerations
- **Threading**: Separate threads for capture and rendering to prevent UI lag
- **Memory Management**: Efficient texture handling to minimize GPU memory usage
- **Frame Rate**: Configurable FPS for capture and rendering
- **Resource Cleanup**: Proper cleanup of OpenGL resources on window close

## Section: Resource Management

### Centralized Resource Handling
All resource management in the application must go through the centralized `ResourceManager` to ensure:
- Consistent resource lifecycle management
- Prevention of resource leaks
- Thread-safe access to shared resources
- Centralized monitoring and debugging

#### Key Components:
- `utils/resource_manager.py`: Core resource management implementation
- `utils/resource_types.py`: Resource type definitions and enums

#### Resource Types:
- `FILE_HANDLE`: File handles and I/O resources
- `MEMORY_BUFFER`: Memory allocations requiring cleanup
- `SYSTEM_HANDLE`: System resources (GDI handles, etc.)

#### Best Practices:
1. Register all resources immediately after creation
2. Use unique, descriptive resource IDs
3. Provide appropriate metadata for debugging
4. Always unregister resources when no longer needed
5. Use context managers or RAII patterns where possible

## Section: File and Module Map

### Centralized Core (`py/`)
- `app_core.py` - Central application core managing shared state and functionality
- `ai_friendly_core.py` - AI integration and automation capabilities

### Utility Modules (`py/utils/`)
- `cache_cleaner.py` - Cache management and cleanup
- `constants.py` - Application-wide constants
- `debug_utils.py` - Centralized debugging and logging utilities
- `overlay_context_menu.py` - Context menu handling for overlays
## Section: Theming System

### Overview
The theming system provides a consistent and maintainable way to style the application's user interface. It's implemented using Qt Style Sheets (QSS) with a strict organization pattern to ensure maintainability and consistency. The system enforces a QSS-first approach where all visual styling is defined in theme files, with no inline or programmatic styling in Python code.

### Key Components
- **Theme Files**:
  - `dark.qss`: Dark theme stylesheet
  - `light.qss`: Light theme stylesheet (mirrors dark theme structure with inverted colors)
  - Both themes maintain consistent structure and selector naming

- **Theme Manager** (`utils/theme/theme_manager.py`):
  - Centralized theme management
  - Loads and validates QSS files
  - Handles theme switching and signal emissions
  - No fallback mechanisms - fails explicitly if theme cannot be loaded

- **Sync Utility** (`scripts/sync_themes.py`):
  - Automatically mirrors changes from dark.qss to light.qss
  - Inverts colors appropriately for light theme
  - Preserves structure and selectors while updating colors

### Styling Guidelines
1. **QSS-First Approach**:
   - All visual styling must be defined in QSS files
   - No inline styles or style overrides in Python code
   - Clear separation of concerns between logic and presentation

2. **Naming Conventions**:
   - Use specific selectors for reliable styling (e.g., `QFrame#titleBar`)
   - Prefix custom properties with `spq-`
   - Group related styles with section comments

3. **Theme Structure**:
   - Organized into logical sections with clear headers
   - Consistent ordering of properties
   - Comments for complex or non-obvious styles

4. **Color Handling**:
   - Use rgba() for colors with opacity
   - Define color variables at the top of each theme file
   - Ensure sufficient contrast for accessibility

5. **No Fallbacks**:
   - The system will fail explicitly if styles cannot be applied
   - No silent degradation of the UI
   - Clear error messages for debugging styling issues

### Core Components
- **Theme Manager** (`utils/theme/theme_manager.py`)
  - Singleton instance for theme management
  - Handles theme loading and application
  - Manages theme assets and resources
  - Provides theme change notifications

### Theme Structure
Each theme file (`.qss`) is organized into the following sections:

1. **Base/Global Styles**
   - Root element styles
   - Global variables and colors
   - Common widget defaults

2. **Main Dialog**
   - Main window styling
   - Title bar and window controls
   - Content area layout

3. **Settings Dialog**
   - Dialog window styling
   - Form elements and controls
   - Section headers and groups

4. **Overlays & Tooltips**
   - Tooltip styling
   - Loading indicators
   - Notification popups

5. **Common Components**
   - Buttons and controls
   - Input fields and comboboxes
   - Scrollbars and sliders
   - Checkboxes and radio buttons

### Theming Guidelines
1. **Color Usage**
   - Use semantic color names (e.g., `@primary-color`, `@error-color`)
   - Define colors at the top of each theme file
   - Ensure sufficient contrast for accessibility

2. **Styling Rules**
   - Keep selectors specific but not overly complex
   - Group related styles together
   - Use comments to separate logical sections
   - Maintain consistent indentation (4 spaces)

3. **Assets**
   - Store theme-specific assets in `/themes/assets/`
   - Use vector formats (SVG) where possible
   - Follow naming convention: `component-state.ext` (e.g., `button-hover.svg`)

4. **Implementation**
   - All visual styling should be done in QSS
   - Avoid inline styles in code
   - Use object names and properties for dynamic theming
   - Test all interactive states (hover, pressed, disabled)

### Theme Switching
Theme changes are handled through the `ThemeManager` which:
1. Validates the requested theme
2. Loads the appropriate QSS file
3. Applies the stylesheet to the application
4. Emits theme change signals
5. Updates any dynamic UI elements

### Best Practices
- Always maintain both light and dark themes in sync
- Document new color variables and their usage
- Test themes on different DPIs and platforms
- Keep styles modular and reusable
- Use QSS properties for runtime theming

### Example Theme Entry
```qss
/* ===== BUTTONS ===== */
QPushButton {
    border: 2px solid @border-color;
    border-radius: 4px;
    padding: 5px 15px;
    background-color: @button-bg;
    color: @text-primary;
}

QPushButton:hover {
    background-color: @button-hover-bg;
}

QPushButton:pressed {
    background-color: @button-pressed-bg;
}
```

- `resource_manager.py` - Centralized resource management
- `snap_utils.py` - Window snapping and positioning utilities
- `style_manager.py` - Theme and style management
- `thread_manager.py` - Centralized thread management
- `universal_media_controller.py` - Media control functionality
- `validators.py` - Input validation utilities
- `window_management.py` - Window handling utilities
- `window_menu_utils.py` - Window menu utilities

### Core Modules (`py/monitor_overlay/`)

### Event System (`py/utils/event_system/`)
- `__init__.py` - Package initialization and exports
- `event.py` - Base event classes and types
- `dispatcher.py` - Event routing and handling
- `events.py` - Predefined event classes
- `qt_integration.py` - Qt event system integration
- `migration.py` - Migration utilities
- `README.md` - Documentation and usage examples

### Core Modules (`py/monitor_overlay/)
- `__init__.py` - Package initialization and exports
- `core.py` - Main MonitorOverlay class and core functionality
- `capture_worker.py` - Threaded screen capture implementation
- `opengl/` - OpenGL rendering components
  - `__init__.py` - Package initialization
  - `gl_capture.py` - Screen capture using OpenGL
  - `gl_renderer.py` - OpenGL rendering logic
  - `gl_utils.py` - OpenGL utility functions
- `opengl_overlay.py` - OpenGL overlay implementation
- `overlay_ui.py` - Themed UI components

### Support Modules
- `core/window/snapping.py` - Window positioning and snapping utilities
- `core/hotkeys/manager.py` - Centralized hotkey management
- `core/window/overlay.py` - Overlay window management
- `core/window/state.py` - Window state tracking and management
- `subsettings_dialog.py` - Configuration interface
- `monitor_utils.py` - Monitor detection and information
- `style_manager.py` - Theme and style management
- `window_management.py` - Window handling utilities
- `window_overlay.py` - Window overlay implementation
- `debug_utils.py` - Debugging and logging utilities

### Configuration
- `Settings/settings.ini` - User configuration storage
- `Resources/` - Application resources (icons, etc.)
- `themes/` - UI theme definitions

## Section: Implementation Details
### Dynamic Settings Integration:

#### 1. Settings Change Flow:
- **SubSettingsDialog** (subsettings_dialog.py, ~line 400-500)
  - Handles UI interactions for opacity sliders
  - Converts slider values (0-100) to internal format (0.0-1.0)
  - Emits signals to parent application
  - Key methods:
    - `_on_opacity_changed(value)` - Handles main overlay opacity changes
    - `_on_border_opacity_changed(value)` - Handles border opacity changes

#### 2. Application Layer (PiPApplication in main.py):
- Receives settings updates from SubSettingsDialog
- Propagates changes to all active overlays
- Key methods:
  - `update_overlay_opacity(opacity)` - Updates all overlays with new opacity
  - `update_border_opacity(border_opacity)` - Updates border opacity for all overlays

#### 3. Overlay Implementation (MonitorOverlay in monitor_overlay/core.py):
- Receives and applies settings updates
- Handles redraw when settings change
- Key method:
  ```python
  def update_settings(self, settings):
      """
      Update overlay settings dynamically.
      Args:
          settings (dict): Can include:
              - opacity (float): 0.0-1.0
              - border_opacity (float): 0.0-1.0
              - theme (str): 'light' or 'dark'
      """
  ```

#### 4. OpenGL Rendering (GLWidget in monitor_overlay/gl_renderer.py):
- Applies visual changes based on new settings
- Handles opacity and border rendering
- Key methods:
  - `setOpacity(value)` - Updates overlay opacity
  - `setBorderOpacity(value)` - Updates border opacity
  - `paintGL()` - Renders with current settings

### OverlayUI Component:
- Provides themed overlay display with configurable appearance
- Supports light/dark theme modes
- Implements 3px opaque borders (white for dark theme, black for light theme)
- Handles window management (resize, move) with snap-to-edge functionality
- Uses hardware-accelerated rendering when possible
- Provides visual feedback during interactions

### Theme Support:
- Light and dark theme support with strict token validation
- Theme components use registration system for required tokens
- Centralized theme event system for error handling and coordination
- Performance-optimized theme switching with resource caching
- Theme-appropriate border colors with validation
- Consistent with system/application theme settings
- Standardized error handling and logging for theme issues

## Section: Open Questions or TODOs
- [ ] Implement GPU acceleration for rendering (in progress)
- [ ] Add support for custom overlay themes
- [ ] Create installer/package for distribution
- [ ] Add unit tests for overlay components
- [ ] Implement performance optimizations for low-end systems

## Section: Recent Changes
- 2025-07-29: Centralized application architecture implemented
  - Moved all utility modules to `/utils`
  - Added `app_core.py` as central application hub
  - Updated all imports to use new module locations
  - Added `ai_friendly_core.py` for AI integration
  - Improved code organization and maintainability
- 2025-07-28: Added comprehensive OpenGL error handling and fallback mechanisms
- 2025-07-28: Implemented multi-threaded capture and rendering pipeline
- 2025-07-28: Added support for multiple monitor configurations
- 2025-07-28: Improved high-DPI display support
- 2025-07-28: Enhanced window management with snapping and positioning utilities
- 2025-07-28: Added theme support with light/dark mode
- 2025-07-30: Implemented centralized resource management system
- 2025-07-29: Updated OpenGL renderer to use centralized resource management
- 2025-07-28: Initial specification created
