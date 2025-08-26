### DWM Thumbnail Management (`utils/window/thumbnail_manager.py`)

- **Register**: Minimal pre-validation (valid src HWND with non-zero client rect). Destination HWND not hard-rejected to avoid transient Qt races. Records `last_hresult`.
- **Retry**: One-shot retry on `E_INVALIDARG (-2147024809)` after ~15ms sleep.
- **Update**: `update_thumbnail(...)` logs detailed flags/rects on failure and ensures composition attributes on first visible.
- **Unregister**: On shutdown, treats `E_INVALIDARG` from `DwmUnregisterThumbnail` as benign (logs DEBUG) to avoid noisy errors when DWM already invalidated the handle.
# 📚 SPQModular Codebase Index

> **AI Agents**: This is the canonical codebase index. Always regenerate wholesale based on live codebase state. Do not append or create changelogs.

## 📋 Quick Navigation

- [🏁 Entry Point](#-entry-point)
- [🧠 Core Services](#-core-services)
- [🪟 Window Management](#-window-management)
- [🎨 Graphics & Overlays](#-graphics--overlays)
- [🎯 User Interface](#-user-interface)
- [🔧 Utilities](#-utilities)
- [🧪 Testing](#-testing)
- [📁 Resources & Config](#-resources--config)

---

### KeyPassthrough Blocklist (`core/input/keypassthrough_blocklist.py`)

#### `KeyPassthroughBlocklist` (centralized loader/matcher)
- **Purpose**: Load and parse `settings/keypassthrough_blocklist.txt` and provide cached matching against the active window's process name and title.
- **Format**: One rule per line; comments with `#`. Plain lines ending in `.exe` match process name; other plain lines are title-contains (ALL terms). JSON-per-line overrides: `{ "exe": "..." }`, `{ "title_exact": "..." }`, `{ "title_contains": "..." }`.
- **Caching/Reload**: Parses once and caches with mtime tracking; throttles reload checks to ~2s.
- **API**:
  - `get_blocklist()` → singleton accessor
  - `match(exe_name: Optional[str], title: Optional[str]) -> Optional[dict]`
  - `match_for_hwnd(hwnd: int) -> Optional[dict]` (Windows only)
- **Integration**: Used by `KeyPassthroughController` for early block decisions when `features.keypassthrough_blocklist_enabled` is enabled. Emits enriched `key.passthrough.blocked` events and triggers UI flash via `_block(...)`.

## 🏁 Entry Point

### `main.py`
- **`PiPApplication`**: Main Qt application class
- **`main()`**: Application bootstrap and service injection
- **Dependencies**: `SystemTrayManager`, `ThreadManager`, `ResourceManager`
- **Pattern**: Service locator + dependency injection
- **Portable Paths**: `utils.paths.get_runtime_root()` first honors `SPQ_RUNTIME_ROOT` (set by the native launcher) and normalizes compiled layouts (`<root>/data/bin` or `<root>/data`) back to `<root>`. Logs resolve to `<root>/logs`. Application icon resolves from `data/resources/ShittyPIP.ico` with a development fallback to repository `resources/`.
- **Env Flags**: `--debug` sets `SPQ_DEBUG=1` early to enable verbose logging.

## 

### Application Bootstrap (`core/application/`)

#### `core.py` - ApplicationCore
- **Purpose**: Full application bootstrap and service wire-up
- **Initialization Order**: Resources → Threads → Events → Settings → WindowManager
- **Key Methods**:
  - `shutdown()`: Idempotent cleanup (calls `_windows.shutdown()`, `_logger.shutdown()`)
  - `ensure_window_mode_features()`: Lazy-initialize window-only systems
- **Integrations**:
  - Always: `OpacityManager`
  - Lazy (window overlays only): `QuickSwitchController`, `FocusTracker`, `AutoSwitchController`, Media KeepAlive (guarded by `features.media_control_enabled`)
  - Trigger: `OverlayManager.create_overlay(...)` calls `ensure_window_mode_features()` for `OverlayType.WINDOW`

#### `__init__.py`
- **Purpose**: Service locator exports (DI container pattern)


### Hotkey System (`core/hotkeys/`, `core/switching/`, `core/opacity/`)

#### `hotkeys/manager.py` - HotkeyManager
- **Purpose**: Centralized global hotkey registration/dispatch
- **Backend Policy** (strict, no-fallback):
  - Single keys (no modifiers): `keyboard` library (suppressed)
  - Combinations (2+ keys): Windows `RegisterHotKey` (non-suppressed)
  - Registration failures: roll back partial state with explicit logs
- **Special Keys**: Backtick/tilde via `VK_OEM_3` (fallback `0xC0`)
 - **Threading**: Message loop runs as a long-lived `ThreadManager` task on the IO pool (no raw `threading.Thread`). A message-only window is created on the hotkey thread; `GetMessage` pumps `WM_HOTKEY`. Shutdown posts `WM_QUIT` via `win32api.PostThreadMessage(thread_id, WM_QUIT, 0, 0)` to break `GetMessage` cleanly.
 - **Resource Lifecycle**: Registered with `ResourceManager` (facade `utils.resource_manager`) as a custom resource; cleanup calls `HotkeyManager.shutdown()` (clears hotkeys, uninstalls hooks, stops loop, unregisters).
 - **UI Dispatch Policy**: Any UI-affine callbacks use `ThreadManager.run_on_ui_thread`; deferrals use `ThreadManager.single_shot` (no raw `QTimer`).
 - **Single-key suppression watchdog**: For suppressed single-key hotkeys (e.g., quickswitch `` ` ``), a tokenized watchdog uses `ThreadManager.single_shot` to poll the physical key state and resets the press-gate if a release event is missed by the backend. Prevents stuck gates and key leaks without LLHOOK.
 - **Command Passing**: Registration/unregistration requests are posted to the hotkey thread via a lock-free `SPSCQueue` (single producer: callers; single consumer: hotkey thread). No raw locks on hot paths.
 - **Enhanced Error Logging**: `RegisterHotKey` failures log `GetLastError()` codes with the attempted modifiers/key; a retry without `MOD_NOREPEAT` is attempted when applicable.
 - **Settings**:
   - `hotkeys.prefer_keyboard_fallback` (bool): Prefer keyboard-combo registration first in eligible cases; skip system registration if keyboard path succeeds.
   - `hotkeys.allow_single_digits` (bool): Treat `0–9` as safe single keys for the keyboard suppression backend.

#### `switching/quickswitch_controller.py` - QuickSwitchController
- **Purpose**: Owns quickswitch hotkey lifecycle
- **Registration**: Registers the combo from `SettingsManager` key `hotkeys.opacity_quickswitch` (default: `shift+x`) via the `keyboard` library. No fallback/backtick paths. Cleaned up deterministically via `ResourceManager` as a custom resource
- **Settings**: `hotkeys.quickswitch_enabled` (bool) toggles feature; `hotkeys.opacity_quickswitch` (str) stores the combo. Live updates via `QuickSwitchController.update_hotkeys()`
- **Cooldown & Threading**: 800ms lock-free cooldown using `time.monotonic()` and an in-flight gate; authoritative checks and full quickswitch flow run on the UI thread via `ThreadManager.run_on_ui_thread()` with early cooldown check on any thread. Focus handoff deferred ~25ms via `ThreadManager.single_shot()`
- **Events**: Publishes suppression diagnostics — `"switch.cooldown_suppressed"`, `"switch.reentry_suppressed"`, `"switch.lock_suppressed"`
- **Logging**: QUICKSWITCH prefix; throttled/deduped via `core.logging.throttled` and `core.logging.log_dedupe`
  - Categories: "quickswitch:reentry", "quickswitch:lock", "quickswitch:foreground", "quickswitch:mru", "quickswitch:seed", "quickswitch:dispatch", "quickswitch:fail"
  - Behavior: high-frequency debug emits are rate-limited; repeated failure messages are deduplicated. Functional behavior unchanged
  - Fallback: if helpers unavailable, falls back to standard `logger.debug`
  - Gating: respects global debug (`SPQ_DEBUG`, `utils.debug.debug_enabled`)

#### `opacity/manager.py` - OpacityManager
- **Purpose**: Opacity increase/decrease hotkeys only
 - **Policy**:
   - Minimum opacity: 10% (storage as int percent [10–100])
   - Cadence: 1-1-1-2 step pattern per tick for smoother yet faster changes; interval ~17ms (increased ~30% from prior 24ms)
   - Startup normalization: clamps persisted opacity <10 to 10 and saves immediately
   - Emits `opacityChanged(int)`; boundary logs with `[OPACITY]` tag

 - **Defensive Clamping Layers**:
   - `core/opacity/manager.py` → clamps to [10, 100]
   - `ui/main_dialog.py::set_opacity(float)` → clamps to >=0.1 before delegating
   - `core/graphics/backends/dwm/integrated_dwm_backend.py::IntegratedDWMOverlay.set_opacity(float)` → clamps to [0.1, 1.0] for canvas + thumbnail

### Input System (`core/input/`)

#### `input/key_passthrough_controller.py` - KeyPassthroughController
- **Purpose**: Centralized PostMessage-only key passthrough with media routing
- **Settings**: `features.keypassthrough_enabled`, `features.media_control_enabled`
- **API**:
  - `set_target_hwnd(hwnd)`, `passthrough_key(vk) -> bool`
- **Signals**: `enabled_changed(bool)`, `target_changed(int)`
 - **Events**: `key.passthrough.*` (enabled, target, forwarded, media_routed, blocked)
 - **Focus Gating**: `OverlayHost` sets target HWND when overlay active
 - **Rate Limiting**: ~18ms min interval, 10ms keyup delay
 - **Threading**: `ThreadManager.run_on_ui_thread`, `ThreadManager.single_shot` (no raw QTimer)
 - **Arrow→Volume Remap (media-enabled)**: When `features.media_control_enabled` is ON, `VK_UP` and `VK_DOWN` are internally remapped to `VK_VOLUME_UP`/`VK_VOLUME_DOWN` at the start of `passthrough_key`, `press_passthrough_key`, and `release_passthrough_key`. This triggers identical hold behavior and prevents arrow key events from being forwarded while active.
 - **Logging**: KEYPASS prefix
  - Verbose diagnostics gated by `debug.keypassthrough_verbose` (includes media routing and fallbacks)
- **Overlay Suppression**: While any overlay is focused (tracked via `utils.state.focus_state.get_focus_state()`), global media keys (`VK_MEDIA_*`) and system volume keys (`VK_VOLUME_*`) are suppressed in `passthrough_key()`/`press_passthrough_key()`/`_volume_hold_tick()` to prevent duplicate adjustments. Focus check failures fail closed (block) to avoid double-handling. Events: `key.passthrough.blocked` with reasons like `media-key-overlay-focused`, `volume-overlay-focused`.
 - **Browser-aware Child Fallback (media-disabled playback controls)**: When `features.media_control_enabled` is OFF and the target HWND is a supported browser (Chrome, Edge, Firefox, Discord), playback keys SPACE/LEFT/RIGHT are sent via `MediaController._send_browser_hotkey(...)` to child content windows first; top-level is used as a fallback. Publishes `key.passthrough.forwarded` with note `browser-child` on success.
 - **Verbose Non-media Logging**: Toggle `debug.keypassthrough_verbose` to emit reasoned decision logs for non-media paths (early returns, target validation, routing results). Off by default to avoid noise.
- **Volume Keys Policy**:
  - Uses VK_VOLUME_UP (0xAF), VK_VOLUME_DOWN (0xAE), VK_VOLUME_MUTE (0xAD)
  - Routed locally via `MediaController.*_for_hwnd(hwnd)` with session-volume, app-hotkey, then mixer fallback
  - Timer-based holds implemented: on press, perform an immediate step and start a repeating timer via `ThreadManager.single_shot`; on release, cancel via token invalidation
  - Arrow-to-volume remap: When media control is enabled, `VK_UP`/`VK_DOWN` are treated equivalently to volume keys for press/hold/release semantics (hold timers apply; arrow key events are not forwarded during active media control)
  - Configurable timing via settings: `input.volume_hold_initial_delay_ms` (default 200), `input.volume_hold_interval_ms` (default 75)
  - Minimal suppression window (~35ms) is still enforced to smooth bursts and avoid flooding
  - Never intercepts globally; only routes when a valid target HWND is set
  - **Hold rescheduling**: During holds, `_volume_hold_tick` always reschedules the next tick even when gating conditions (overlay focused/invalid target hwnd) are not met; the action is suppressed but the timer persists until release. Prevents premature early stop during long holds.
  - Arrow keys LEFT/RIGHT remain previous/next track; never mapped to volume
- **UI Feedback on Block**: Centralized `_block(reason)` publishes `key.passthrough.blocked` and triggers a brief black flash (`~300ms`) on the active overlay's focus indicator via `OverlayManager.get_active_overlay()` and `OverlayHost.flash_focus_indicator(...)` on the UI thread.
  - Throttled/Deduped: controlled by `ui.block_flash_min_interval_ms` (default 250ms). Repeats within the interval or same reason are coalesced.
 - **Blocklist Pre-check**: If `features.keypassthrough_blocklist_enabled` is true, performs an early check via the centralized loader (`core/input/keypassthrough_blocklist.py#get_blocklist()`). When the active window matches a rule (exe/title), the key is not forwarded, `_block("blocklist", extra=match_info)` is invoked, and an enriched `key.passthrough.blocked` event is emitted with `{ reason: "blocklist", extra: { type, value, exe, title } }`.

#### Supporting Utilities
- **`utils/win/winmsg.py`**: Safe Windows messaging helpers
  - `is_process_responsive(hwnd, timeout_ms)`
  - `safe_send_appcommand(hwnd, command)` with hang detection
  - `safe_send_message(hwnd, msg, wparam, lparam, timeout_ms=250)` with responsiveness check and timeout (centralized; prefer over direct `SendMessage`)

### Media Control (`core/media/`)

#### `media/media_controller.py` - MediaController
- **Purpose**: Centralized media player control with app-specific routing
- **Features**:
  - Comprehensive app catalog with safe method definitions
  - Browser-specific routing with child window enumeration
  - Enhanced browser command path with subtle activation-and-retry via KeepAlive (`_send_browser_media_command_enhanced`)
  - Hints media activity to KeepAlive on successful commands to maintain background responsiveness
  - Process responsiveness checking before dispatch
  - Window enumeration caching (5s TTL)
- **Overlay Integration**: `get_preferred_app()` detects from overlay target HWND
- **Settings**: `media.app_catalog`, `media.preferred_apps`
- **API**: `play_pause()`, `next_track()`, `previous_track()`, `volume_up()`, `volume_down()`, `stop()`
- **HWND APIs**: `*_for_hwnd(hwnd)` methods for direct control
- **Volume Policy**: Session-only. Per-app audio session via `utils.audio.session_volume` (no hotkey or mixer fallback). The audio module includes a psutil-guarded child-process resolution to handle apps whose audio session runs in renderer/subprocesses (e.g., browsers).
- **Browser Support**: Chrome, Firefox, Edge, Discord with streaming site priority
- **Logging**: MediaController prefix

#### `media/keepalive.py` - MediaPlayerKeepAlive
- **Purpose**: Background monitoring and responsiveness checking
- **Features**:
  - Automatic crash-prone app detection, catalog updates
  - Media activity detection heuristics (titles, child classes, session volume)
  - Subtle activation (no focus steal) with cooldowns and z-order awareness
  - Periodic background keepalive sweep for browsers/Discord
- **Integration**: ThreadManager for all async operations
- **Settings**: `features.media_control_enabled`
- **API**: `start()`, `stop()`, `get_monitored_apps()`, `force_check(hwnd)`, `hint_media_activity(hwnd)`, `request_subtle_activation(hwnd, app_name=None)`
- **Logging**: MEDIA_KEEPALIVE prefix

### Settings Management (`core/settings/`)

#### `settings/settings_manager.py` - SettingsManager
- **Purpose**: Thread-safe configuration management
- **Storage**: JSON persistence at `<settings_dir>/settings.json`
- **Settings Directory Resolution**:
  - Portable mode when `SPQ_PORTABLE=1` is set or when `<runtime_root>/settings` exists → `settings_dir = <runtime_root>/settings`
  - Otherwise → `settings_dir = %USERPROFILE%/.spqmodular/settings`
- **Policy**: No-fallback, strict validation on load/set
- **Theme**: Canonical key `theme` (light/dark, no system)
- **Signals**: `setting_changed(key: str, value: object)`
- **API**: Thread-safe get/set, change handlers, category queries
- **Used By**: OpacityManager, ThemeManager, WindowStateManager
- **First-run File Creation (Portable Build Guarantee)**:
  - On init, `_ensure_settings_file_exists()` creates a default `settings.json` if missing (never overwrites). Logs `[SETTINGS] Created default settings.json on first run at <path>` or an error on failure.
  - `_ensure_keypassthrough_blocklist_defaults()` creates `keypassthrough_blocklist.txt` in the same directory if missing with conservative defaults. Logs with `[KEYPASS]` prefix.
- **Portable Env Vars**: Honors `SPQ_RUNTIME_ROOT` (set by native launcher) via `utils.paths.get_runtime_root()`; `SPQ_PORTABLE=1` forces portable settings under `<runtime_root>/settings`.
- **KeyPassthrough Blocklist Defaults**: On init, auto-creates `keypassthrough_blocklist.txt` in the same directory as `settings.json` (repo: `settings/`) if missing, with conservative default entries (popular anti-cheat game executables). Logs with `[KEYPASS]` prefix. File is read-only at runtime for controllers.
- **Public Directory Accessor**: `get_settings_dir()` returns the directory that contains `settings.json` and auxiliary files (like the blocklist). Used by the blocklist loader to locate `keypassthrough_blocklist.txt`.

##### Debug verbose toggles
- `debug.keypassthrough_verbose`: Verbose routing logs for key passthrough decisions (now includes media-path successes/failures), handled in `core/input/key_passthrough_controller.py`.
- `debug.events_trace`: Enable lightweight dispatch tracing in `core/events/event_system.py` (begin/end, handler identity, priority, duration, handled flag).
- `debug.window_filter_verbose`: Controls verbose exclusion logs for the window filtering module. Wired in `core/application/core.py`, which applies the initial value and registers a change handler that calls `utils/window_filter.set_verbose_exclusion_logs(...)` to toggle `VERBOSE_EXCLUSION_LOGS`. Default: false.

#### Presentation Setting
- Removed. Presentation path is QWidget blitter only.

### Threading System (`core/threading/`)

#### `threading/manager.py` - ThreadManager
- **Purpose**: Global threading + timer handling (NO raw QTimers allowed)
- **Core APIs**:
  - `single_shot(ms, callback)`: Replaces all `QTimer.singleShot` calls
  - `run_on_ui_thread(func, *args, **kwargs)`: Lock-free UI thread scheduling
- **Lock-Free Factories**:
  - `create_spsc_queue(capacity)`: Single-Producer/Single-Consumer queues
  - `create_triple_buffer()`: Atomic frame exchange buffers
  - `create_ui_coalescer(name, capacity=64, window_ms=7)`: UI task batching
- **UI Coalescers**:
  - **Purpose**: Batch UI tasks with drop-oldest overflow policy
  - **Methods**: `submit(task)`, `flush()`, `shutdown()`
  - **Registration**: Auto-registered with ResourceManager
  - **Used By**: Z-order enforcement, DWM thumbnail updates, geometry verification
- **Policy**: No shared mutable state; ownership transfer via queue items
 - **Primitives**: `utils/lockfree/` (SPSCQueue, TripleBuffer)
 - **Supporting**: `task.py` (abstractions), `priority.py` (tuning)
 - **Codebase Policy**: No raw `QTimer` or `QObject.startTimer` usage anywhere in the codebase. Always use `ThreadManager.single_shot` for timers and `ThreadManager.run_on_ui_thread` for UI dispatch. The only permitted internal `QTimer` usage is within `ThreadManager` itself.
 - **Shutdown Coordination**: After setting `_shutdown` under a short-held lock, releases the lock and shuts down executors. Resource cleanup is performed by `ResourceManager.shutdown()` on the UI thread; there is no mutation worker to stop in the new model.

### Logging System (`core/logging/`, `utils/debug/`)

#### `logging/logger_impl.py` - Logger
- **Purpose**: Thread-safe file + console logging
- **Features**: Configurable at runtime

#### `utils/debug/` - Debug Utilities
- **Purpose**: Centralized debug logging control
- **Policy**: Disabled by default; enable via `SPQ_DEBUG=1/true/yes/on`
- **Gating**: All debug logs must check `utils.debug.debug_enabled`
- **Rate Limiting**: High-frequency paths throttle logs
  - `border_renderer.py`: Per-path rate limiting
  - `overlay_context_menu.py`: Event filter logs gated

#### Test Logging
- **Location**: `logs/tests/` via `tests/conftest.py`
- **Rotation**: 2MB/file, 3 backups
- **Levels**: Console INFO, File DEBUG
- **Environment**: `SPQ_TEST_LOG_DIR` exported
- **Status**: Deferred until Threading/Resource Manager complete

### Event System (`core/events/`)

#### `events/event_system.py` - EventSystem
- **Purpose**: Centralized publish-subscribe event bus
- **Thread-Safety**: Uses `threading.RLock` for subscription maps; handlers invoked after releasing locks
- **Dispatch Policy**: Callers may request UI-thread dispatch via `dispatch_on_ui=True`; this routes callbacks through `ThreadManager.run_on_ui_thread`
- **Priority Ordering**: Single source of truth via `Subscription.__lt__` (higher numeric runs earlier; zero last)
- **UI Thread Dispatch**: `subscribe(event_type, callback, priority=0, filter_fn=None, dispatch_on_ui=False)`
  - Set `dispatch_on_ui=True` to route `callback` via `ThreadManager.run_on_ui_thread`
  - Default preserves synchronous, calling-thread behavior
- **Wildcard Support**: Patterns like `window.*` match hierarchical event types

#### Event Types
- **`window_events.py`**: Window-related events
- **`app_events.py`**: Application-level events

### Resource Management (`core/resources/`, `utils/resource_manager.py`)

#### `core/resources/manager.py` - ResourceManager (Implementation)
- **Purpose**: Centralized resource lifecycle with weakref tracking
 - **Integration**: `attach_thread_manager()`; all mutations are dispatched synchronously on the UI thread via `ThreadManager.run_on_ui_thread`
- **Snapshots**: Lock-free via TripleBuffer
- **Cleanup**: Atexit hook, deterministic ordering
- **Policy**: Explicit failure if not weakref-able (no fallback)
 - **Shutdown**: `shutdown()` performs synchronous `cleanup_all()` on the UI thread in deterministic order (Qt → Network/DB → Filesystem/OS → Other). Legacy mutation-worker APIs are retained as no-ops for compatibility.

#### `utils/resource_manager.py` - Public Facade
- **Access Policy**: Always import from facade, never `core.resources` directly
- **Core API**:
  - `register()`, `unregister()`, `get()`, `get_typed()`
  - `list_resources()`, `list_resources_snapshot()`
  - `cleanup_all()`, `shutdown()`
- **Z-Order Delegations**: Thin wrappers to `utils.z_order_manager`
  - `begin_context_menu()`, `end_context_menu()`, `enforce_z_order()`
  - `register_overlay()`, `unregister_overlay()`
- **Cleanup Ordering**: Qt → Network/DB → Filesystem/OS → Other
- **Built-in Handlers**: File, temp file/dir, OS handles, network, database


---

## 

### Core Window System (`core/window/`)

#### Key Components
- **`WindowManager`**: Main window management entry point
- **`WindowEnumerator`**: System window enumeration
- **`WindowManagerAdapter`**: Qt-friendly abstraction layer

### Window Behavior System (`utils/window/`)

#### `behavior.py` - WindowBehaviorManager
- **Purpose**: Unified window behavior (drag/resize/snap) - Single Source of Truth
- **Used By**: All dialogs and windows in application
- **Features**: Mouse events, dragging, resizing, snapping
- **Constants** (fixed, not configurable):
  - `DEFAULT_SNAP_DISTANCE = 30` pixels
  - `DEFAULT_RESIZE_MARGIN = 12` pixels
- **Policy**: No duplication; explicit failure logging; no fallbacks for unknown window types
- **Integration Requirements**:
  - Use `WindowBehaviorManager(widget, min_width, min_height)`
  - Delegate mouse events to manager
  - QSS-only styling (no programmatic styling)
  - Full QMouseEvent API for event forwarding

#### Supporting Components
- **`WindowManagementCore`**: Overlay management and state persistence
- **`WindowState` & `WindowStateManager`**: Position/size save/restore
- **Utility Functions**:
  - `apply_snap()`: Multi-monitor snapping
  - `get_resize_edge_for_pos()`: Edge detection
  - `get_cursor_for_edge()`: Cursor management

#### Window Utilities
- **`monitors.py`**: Screen enumeration + DPI
- **`types.py`**: `WindowType`, `WindowInfo` enums
- **`icons.py`**: Icon extraction and scaling
  - Blank icon policy: resource-only `:/icons/Blank.ico`; if load fails, generate a transparent 16×16 pixmap; no filesystem fallback. Warnings are logged on failure. Implemented in `core/application/window_enumerator.py::_init_blank_icon` and `core/window/enumerator.py::_init_blank_icon`.

---

## 

### Core Graphics System (`core/graphics/`)

#### Overlay Architecture
- **`overlay.py`**: Base `Overlay` class with common interface
- **`overlay_host.py`**: Host window for overlays
- **`overlay_manager.py`**: Centralized lifecycle management with MRU tracking
  - **API**: `get_active_overlay() -> Optional[Overlay]` thread-safe accessor for the currently active overlay
- **`backend_manager.py`**: Backend selection and initialization
- **`types.py`**: Shared type definitions

#### Supporting Systems
- **`utils/z_order_manager.py`**: Unified Z-Order Management (ThreadManager-based, no raw QTimer)
- **`dwm_composition_manager.py`**: DWM composition attribute management
  - Platform-specific attribute application
  - Graceful handling for unsupported attributes (e.g., FREEZE_REPRESENTATION)
  - Adjusted success accounting for known limitations

#### Backend Selection Policy
- **Strict**: If `preferred_backend != AUTO` and unavailable, fail fast with error
- **Auto**: Best available backend selected and logged

#### `d3d11_device.py` - D3D11 Device Manager (Centralized)
- **Purpose**: Single source of truth for native `ID3D11Device` + immediate context with BGRA support.
- **API**: `ensure_initialized()`, `get_device()`, `get_immediate_context()`, `get_feature_level()`
- **Lifecycle**: Registered with `ResourceManager` as a custom resource; cleanup releases COM pointers best-effort. Idempotent init; recreate on device-lost.
- **Spec**: See `Spec.md` → "D3D11 Device/Context Management (Spec)"

#### `dxgi_interop.py` – DXGI Interop (Centralized)
- **Purpose**: Centralized helpers for DXGI interop
  - Safe release of COM pointers
- **Dependencies**:
  - `comtypes` (if used by future DXGI helpers)
- **API**:
  - `release_com_ptr(ptr: ctypes.c_void_p) -> None`
- **Usage**:
- **Rationale**: Single source of truth for DXGI creation helpers; consumers stay minimal and focus on lifecycle

### Rendering Backends (`core/graphics/backends/`)

#### `software/backend.py` - SoftwareOverlay
- **Purpose**: QWidget host for `ui/overlays/integrated_border_canvas.py`

#### `dwm/integrated_dwm_backend.py` - IntegratedDWMOverlay
- **Purpose**: DWM thumbnail-backed window overlays with integrated borders
- **Integration**: Centralized OverlayManager lifecycle
- **Key Features**:
  - `update_source(new_hwnd) -> bool`: OverlayManager reuse path
  - **Aspect Ratio Caching**: `_source_aspect` for consistent scaling
  - **Multi-display Fix**: Uses client area (`GetClientRect`) for aspect calculation
  - **Consistent Scaling**: Cached ratio in `_update_thumbnail_properties()`
  - **Threading (Latest-Value Pipeline)**: Uses `ThreadManager.create_triple_buffer()` to publish latest canvas content rects from `_on_content_rect_changed(...)`; a UI `UICoalescer` drains once per batch and applies updates via `_drain_and_update_thumbnail()` → `_update_thumbnail_properties()`
  - **Resource Management**: Registers `ThumbnailManager`, the active thumbnail binding (cleanup unregisters the thumbnail), the `TripleBuffer` (cleanup resets slots), and the `OverlayContextMenu` (cleanup detaches) with `ResourceManager`; `_close_impl` first unregisters resources deterministically, then best-effort fallbacks
  - **Z-Order Enforcement**: Reaffirms focus indicator/overlay z-order after each thumbnail property update via centralized `ZOrderManager` delegation
  - **Recovery Flags**: On DWM registration failure, sets `_broken=True`, `_needs_recreate=True`; cleared on next successful registration. `OverlayManager.create_overlay()` detects `_needs_recreate` and tears down for a clean recreate instead of reusing.

#### `monitor/monitor_backend.py` - MonitorBackend
- **Purpose**: Monitor capture overlays
- **Host Widget**: `ui/overlays/monitor/monitor_overlay.py` (exposed via `_host`)
- **Lifecycle**: Implements all `Overlay` hooks (`_initialize_impl`, `_show_impl`, etc.)
- **Requirements**: `OverlayConfig.properties["monitor_target"]` (strict, no-fallback)
- **Integration**: OverlayManager registration, ZOrderManager z-order
- **Context menu**: Uses unified `OverlayContextMenu` (`utils/overlay_context_menu.py`) via `attach_to_overlay(self)` for consistent theming and z-order.
  - Actions implemented on `MonitorOverlay`: `_handle_reset_position()` (resets to centered default size), `_handle_quit_application()` (clean quit)
- **Aspect ratio**: `set_target_monitor()` forwards the monitor `rect.width()/rect.height()` to `IntegratedBorderCanvas.set_content_aspect(w, h)` for correct letterbox/pillarbox.
- **Scope**: Focus indicator and MRU switching/collection are not part of the monitor overlay by design.
- **Frame Pipeline**: 
  - **Producer**: `core.graphics.pipeline_manager.PipelineManager` (single source of truth)
    - Publishes to `utils.frame_exchange.get_exchange(f"monitor_frames_{qt_index}")`
  - **Consumer**: Active renderer backend polls `acquire_latest()` and repaints; backend is selected by policy
    - Accepts `CaptureFrame` (RGB/BGRA) via an `update_frame(...)`-style API
    - **Multi-monitor**: Exchange naming `monitor_frames_<qt_index>`
    - **Timer Policy**: `ThreadManager.single_shot(~16ms)` polling (no raw QTimer)
  - **Setting**: `graphics.pipeline` supported values: `'dxgi'` only.

### Capture System (`core/graphics/capture/`)

#### `pipeline_manager.py` - PipelineManager
- **Purpose**: Single source of truth for capture pipeline selection and lifecycle
- **Accessor**: `core.graphics.pipeline_manager.get_pipeline_manager()`
- **Backends**:
  - `'dxgi'` → `MonitorCaptureManager` (DXGI via dxcam; CPU-copy frames)
- **Signals**: Forwards `frame_captured`, `capture_started`, `capture_stopped`, `capture_error(str)` from the active backend
- **API**: `set_capture_rate(fps: float)` forwards to active backend and stores a pending rate for application on next bind
- **Shutdown Synchronization**: Centralized in `PipelineManager.stop_capture()`
  - Calls backend `stop_capture()` then, if `_capture_task_id` exists, joins via `ThreadManager.get_task_result(task_id, timeout=2.5s)`
  - Bounded poll of `backend.is_capturing()` for up to 2.5s with 10ms intervals; logs WARN if still capturing at deadline, DEBUG when stopped

#### `monitor_capture_manager.py` - MonitorCaptureManager
- **Purpose**: Screen capture via DXGI Desktop Duplication (dxcam)
- **FrameExchange**: Publishes to `monitor_frames_{qt_index}` exchange
- **Compatibility**: Maintains Qt signal emission for existing listeners
{{ ... }}
<!-- Swapchain host removed in DXGI-only pipeline. -->

#### `wgc_capture.py` - WgcCaptureManager (Removed)
- This capture backend has been removed in favor of a DXGI-only pipeline. All references are deprecated; monitor capture is provided exclusively by `MonitorCaptureManager` via dxcam.

#### `dwm_capture_manager.py` - DwmCaptureManager
- **Purpose**: DWM-based window capture manager (no pixel copies). Polls source HWND rect and manages DWM thumbnail lifecycle.
- **FrameExchange**: Publishes `DwmContentRect` metadata to `dwm_content_rects_{hwnd_src}` exchange.
- **Signals**: `content_rect_updated(DwmContentRect)`, `capture_started`, `capture_stopped`, `capture_error(str)` (UI thread).
- **Threading**: Runs capture loop on `ThreadManager` CAPTURE pool using `capture_context().submit_capture(...)`; loop lightly paced (~200Hz).
- **UI Coalescer**: Batches UI work via `ThreadManager.create_ui_coalescer('dwm_capture_updates', 128, 7ms)` for signal emission and `ThumbnailManager.update_thumbnail(...)`.
- **Resource Management**: Registers instance, UI coalescer, and active thumbnail binding with `ResourceManager` for deterministic cleanup.

---

## 🎯 User Interface

### Main Application UI (`ui/`)

#### `main_dialog.py` - MainDialog
- **Purpose**: Primary application window
- **Overlay Integration**: Wires to `OverlayManager`
- **Creation Policy**: Explicit via launch button ("»"), not auto-create
- **Backend Selection**:
  - Window overlays: `BackendType.DWM` (strict HWND validation)
  - Monitor overlays: `BackendType.MONITOR`
- **Size Constraints**: Main window minimum `MIN_WINDOW_WIDTH=700`, `MIN_WINDOW_HEIGHT=540` (defined in `ui/main_dialog.py`); enforced via `WindowBehaviorManager` in `MainDialog._setup_window_behavior`.
- **Opacity**: Routes to `OverlayManager.set_opacity()`, hotkey integration
- **Mouse Events**: Forwards to `WindowBehaviorManager`
- **Wheel Resize**: Tuned for speed/smoothness (28px base, 2.0/2.1 scale)
- **ESC Behavior**: Calls `QCoreApplication.quit()`
#### `dialogs/subsettings_dialog.py` - SubSettingsDialog
- **Purpose**: Application settings interface
- **Controls**:
  - Capture FPS: persists `capture.fps` (int 1–165) via `SettingsManager`; live-applies via `core.graphics.pipeline_manager.get_pipeline_manager().set_capture_rate(fps)`
    - Presets: 15, 30, 60, 120, 144, 165 (missing persisted custom value is inserted)
  - Quickswitch Enable: `CircleCheckBox` bound to `hotkeys.quickswitch_enabled` (default: false). Toggling invokes `QuickSwitchController.update_hotkeys()` to (un)register the current combo via the keyboard backend.
  - Quickswitch Hotkey: Editable. Persists to `hotkeys.opacity_quickswitch` (default `shift+x`) and live-applies via `QuickSwitchController.update_hotkeys()`. No fallback or backtick-specific handling.
  - **ESC Behavior**: Closes dialog

#### `core/graphics/overlay_host.py` - OverlayHost
 - **Purpose**: Host window for overlays
 - **Features**:
  - Double-click quickswitch integration
  - Cursor reset hardening
  - Focus indicator management
  - Creates top-level `VolumeOSDWindow` for the volume OSD (owns HWND; no activation). Themed via `ThemeManager`.
  - Positions the OSD bottom-center relative to the host in screen coordinates and updates on `geometryChanged`.
  - Enforces z-order above the host via `ResourceManager.place_window_above(...)` while avoiding global topmost.
  - Event-driven visibility from `media.volume.changed`. When numeric volume is missing, shows a textual indicator ("VOLUME UP"/"VOLUME DOWN").
  - **Automatic Flash Trigger**: `KeyPassthroughController` invokes the flash on blocked passthrough decisions via `OverlayHost.flash_focus_indicator(...)` for UI feedback.
  - **Active Overlay Accessor**: `OverlayManager.get_active_overlay() -> Optional[Overlay]` thread-safe accessor for the currently active overlay.
  - **Minimum Size Enforcement**: Initializes `WindowBehaviorManager` with `min_w=IntegratedBorderCanvas.MIN_WIDTH` and `min_h=IntegratedBorderCanvas.MIN_HEIGHT` to gate drag/resize to at least 200×180.
  - Safety: releases any active volume holds on deactivate, focus-out, and hide to prevent stuck repeat loops

### Overlay UI Components (`ui/overlays/`)

#### `integrated_border_canvas.py` - IntegratedBorderCanvas
- **Purpose**: Primary canvas for all overlays with integrated border rendering
- **Features**:
  - Direct border rendering in `paintEvent` (no z-order issues)
  - Centralized `ThemeManager` integration via `BorderTheme` tokens
  - DPI-aware geometry
  - Mouse event forwarding to `WindowBehaviorManager`
  - Cursor hardening: overlays ensure Arrow cursor on init/show and clear any managed resize/drag cursors via `utils.cursor_manager`. `MonitorOverlay` additionally installs event filters and mouse tracking to clear edge cursors immediately on leave and maintain correct cursor state even when child widgets consume events.
  - Scroll-wheel resize with batching
  - Aspect ratio and minimum size maintenance
  - Window-level masking defers to the owning overlay when available; the canvas applies only its own widget mask. A safe fallback applies the parent mask when no overlay is exposed (legacy hosts).
  - Aspect Ratio Fit Policy: floor-based fit (width/height rounded down) to avoid overshoot or clamping in letterbox/pillarbox scenarios; stable across DPIs. Validated by offscreen Qt tests in `tests/ui/overlays/test_integrated_border_canvas_ar.py`.
  - Quick link: See `Spec.md` → "Canonical Border/Clipping/Masking (Spec)"
  - Minimum size: `setMinimumSize(MIN_WIDTH=200, MIN_HEIGHT=180)`; constants sourced from `utils/window/overlay_constants.py` via `IntegratedBorderCanvas.MIN_WIDTH/MIN_HEIGHT`.
  - Aspect ratio enforcement and signaling: `_calc_content_rect()` + `_update_content_rect()` compute and cache the aspect-fit content rect; `set_content_aspect(w, h)` sets target AR; emits `contentRectChanged(QRect)` on changes.

#### `geometry/border_geometry.py` - BorderGeometry, BorderMetrics
- **Purpose**: Complete border metrics for rendering
- **Features**: Size-scaled inner accent (up to 2x for large overlays)
- **Scaling**: `size_scale = min(min_dim / 200.0, 2.0)`

#### `geometry/focus_indicator.py` - Focus Indicators
- **`FocusIndicatorWidget`**: Bottom-right indicator hosted by OverlayHost
- **`FocusIndicatorWindow`**: Top-level window variant with separate HWND
- **States**: 
  - `locked` (padlock glyph)
  - `passthrough_enabled` (red circle)
  - Normal focus hint (subtle circle)
- **Integration**:
  - Signals: `lock_toggled()` wired to overlay `toggle_window_lock()`
  - Listens to `KeyPassthroughController.enabled_changed(bool)`
  - Passthrough state sync on initialization
  - Geometry verification via UI coalescer
- **Z-Order**: Positioned via `ZOrderManager.place_window_above()`
- **Transparency**: `WA_TranslucentBackground` for per-pixel alpha
 - **Flash API**: `flash_block(duration_ms=300)` temporarily renders the indicator black for the specified duration. Thread-safe; scheduled via `ThreadManager.run_on_ui_thread` and `single_shot`.
 - **Automatic Trigger**: `KeyPassthroughController` invokes the flash on blocked passthrough decisions via `OverlayHost.flash_focus_indicator(...)`.

#### `components/volume_osd.py` - VolumeOSDWidget / VolumeOSDWindow (Top-level OSD)
 - **Purpose**: `VolumeOSDWidget` is the base component; `VolumeOSDWindow` is the top-level window variant used by `OverlayHost` by default. Both use `objectName="volumeOSD"` and share visuals/behavior.
 - **Styling Defaults**: Fill bar is black (`#000000`) at ~90% opacity (alpha ≈ 230); volume text is uppercase and white. Implemented in `VolumeOSDWidget.paintEvent` and can be themed via QSS later.
 - **Behavior**: Subscribes to `media.volume.changed`; shows briefly on updates with internal hide scheduling via `ThreadManager` (no raw `QTimer`). Displays textual fallback ("VOLUME UP"/"VOLUME DOWN") when numeric level is missing.
 - **Positioning**: For the window variant, positioned bottom-center relative to the host in screen coordinates via `host.mapToGlobal(...)` and raised/z-placed above the host via `ResourceManager.place_window_above(...)`. The base widget supports embedded hosting when needed.

#### `core/graphics/overlay_host.py` additions
 - **`OverlayHost.flash_focus_indicator(duration_ms=300)`**: UI-thread-safe entrypoint that triggers `FocusIndicatorWindow.flash_block(...)` on the host's indicator. Used by `KeyPassthroughController._block(...)` for UI feedback.

---

### Utilities

### DWM Thumbnail Retry Policy
- **Purpose**: Robust DWM thumbnail registration handling
- **Policy**: Retry on transient failures (e.g., E_INVALIDARG), fail-fast on persistent errors
- **OverlayManager Integration**: `OverlayManager.create_overlay()` checks recreate flags and tears down for a clean recreate instead of reusing

### Overlay Recreate Flags
- **Purpose**: Overlay lifecycle management
- **Flags**: `_needs_recreate` (cleared on successful registration), `_broken` (set on registration failure)
- **OverlayManager Usage**: `OverlayManager.create_overlay()` detects `_needs_recreate` and tears down for a clean recreate

### Benign Shutdown Handling
- **Purpose**: Graceful shutdown handling for E_INVALIDARG from DwmUnregisterThumbnail
- **Behavior**: Ignore E_INVALIDARG errors during shutdown, log and continue

### Context Menu System (`utils/overlay_context_menu.py`)

#### `OverlayContextMenu`
- **Purpose**: Centralized right-click menu handler for all overlay types
{{ ... }}
- **Z-Order Management**: Via `ResourceManager.begin_context_menu()` / `end_context_menu()`
- **Lifecycle Triggers**:
  - Menu signals: `QMenu.aboutToShow` / `aboutToHide`
  - Event filters: `QEvent.ContextMenu`, right-click `MouseButtonPress`
- **Error Handling**: Enhanced with specific exception handling
- **Logging**: Strict no-fallback, detailed debug logs
- **Actions**:
  - Lock Overlay (window-only): Checkable, bound to `toggle_window_lock`
  - Switch To Monitor: Available for window/monitor overlays
- **Cleanup**: `detach_from_overlay()` removes filters, restores `Qt.DefaultContextMenu`

### Frame Exchange System (`utils/frame_exchange.py`)

#### `FrameExchange`
- **Purpose**: Lock-free triple-buffer frame handoff with latest-frame semantics
- **Factory**: `get_exchange(name)` returns shared instances
- **API**: `publish(obj)`, `acquire_latest()`, `set_on_drop(callback)`, `set_min_publish_interval(seconds)`, `clear()`, `stats()`, `reset_stats()`
- **Used By**:
  - Monitor capture/render pipeline: `"monitor_frames_{qt_index}"`

### Z-Order Management (`utils/z_order_manager.py`)

#### `ZOrderManager`
- **Purpose**: Unified Z-Order Management System
- **Integration**: ThreadManager-based debounce (no raw QTimer)
- **Context Menu Priority**: Built-in support
- **Policy**: Strict no-fallback, explicit failure logging
- **Used By**: All overlay backends via ResourceManager delegation

### Lock-Free Primitives (`utils/lockfree/`)

#### `spsc_queue.py` - SPSCQueue
- **Purpose**: Single-Producer/Single-Consumer ring buffer
- **Features**: Atomic operations, fixed capacity

#### `TripleBuffer` 
- **Purpose**: Atomic frame exchange with rotating buffers
- **Features**: Latest-frame semantics, no locks

#### `focus_state.py` - FocusState (Lock-free)
- **Purpose**: Centralized overlay focus flag used by controllers (e.g., `KeyPassthroughController`) to gate behavior while an overlay is active.
- **Threading Policy**: Lock-free, single-writer on the UI thread. Callers must route writes via `ThreadManager.run_on_ui_thread`. Reads are safe from any thread under CPython due to GIL-protected bool load/store.
- **API**: `get_focus_state()`, `set_overlay_focused(bool)`, `is_overlay_focused()`
- **Singleton**: Eagerly-initialized singleton (no locks). No raw mutexes anywhere in the module.

### Audio System (`utils/audio/`)

#### `session_volume.py`
- **Purpose**: Per-application audio session volume control
- **API**: `adjust_session_volume_for_hwnd(hwnd, delta)`, `get_session_volume_for_hwnd(hwnd)`
- **Integration**: Used by `MediaController` volume policy
- **Child-process Fallback**: If the top-level window PID has no direct audio session, attempts to resolve sessions from any child processes (e.g., browser renderers) using an optional `psutil`-guarded search. Falls back silently when unavailable. Detailed DEBUG logs remain under the `AUDIO_SESSION` logger.

### Windows Integration (`utils/win/`)

#### `winmsg.py`
- **Purpose**: Safe Windows messaging helpers
- **Features**: PostMessage, SendMessageTimeout, lParam packing
- **Media Support**: `is_process_responsive()`, `safe_send_appcommand()`

### Monitor Utilities (`utils/monitor_utils.py`)
- **Purpose**: Screen enumeration and management
- **API**: `get_all_monitors()`, screen information
- **Used By**: Context menus, overlay targeting

---

## 🧪 Testing

### Test Configuration (`tests/`)

#### `conftest.py`
- **Purpose**: Pytest configuration and logging setup
- **Logging**: Centralized rotating logs at `logs/tests/`
- **Environment**: `SPQ_TEST_LOG_DIR` exported
- **Status**: Deferred until Threading/Resource Manager complete

#### Test Structure
- **`core/`**: Core module tests
- **`integration/`**: Integration tests
- **`fixtures/`**: Test fixtures and utilities

---

## 📁 Resources & Config

### Application Resources
- **`resources/`**: Icons, images, assets
- **`themes/`**: QSS theme files (dark.qss, light.qss)
- **`settings/`**: User configuration (settings.json)
 - **Portable Runtime (release/dist/SPQ/)**:
   - `SPQ.exe` (embedded icon)
   - `data/` → staged runtime assets
     - `resources/` (copied from repo `resources/`)
     - `themes/` (copied from repo `themes/`)
   - `settings/` → portable settings directory (contains `settings.json` and auxiliary files like `keypassthrough_blocklist.txt`)
   - `logs/` → runtime logs
   - Note: Legacy code paths referencing `resources/...` continue to work via PyInstaller `--add-data`.

### Qt Resource Registration & Build Cleanup
 - **Qt resources registration**: Qt resources are compiled into `ui.resources_rc` and registered by both `main.py` and `utils/theme/theme_manager.py` at import time. This ensures `:/themes/*.qss` and other `:/` resource paths are available in tests and non-main contexts. On failure, a DEBUG log is emitted and the ThemeManager falls back to filesystem paths.
 - **Nuitka post-stage cleanup**: `scripts/build_nuitka.ps1` removes empty `<portable_root>/data/themes` and `<portable_root>/data/resources` directories after staging, reflecting that assets are embedded via Qt resources. Non-empty directories are preserved.

### Configuration Files
- **`pyproject.toml`**: Python project configuration
- **`Spec.md`**: Project specification
- **`Index.md`**: This codebase index
- **`ThemeIndex.md`**: Theme system documentation
 - **Build Script**: `scripts/build_pyinstaller.ps1` creates a PyInstaller one-folder portable build, stages `data/`, and creates `settings/` and `logs/`.

### Audit Trail
- **`audits/`**: Architecture migration documentation
- **`docs/`**: Additional project documentation

---

## 🔗 Key Integration Points

### Service Dependencies
1. **ThreadManager** ← All modules (no raw QTimer allowed)
2. **ResourceManager** ← All modules (via `utils.resource_manager` facade)
3. **SettingsManager** ← Theme, Opacity, Window managers
4. **OverlayManager** ← UI, backends, context menus
5. **ZOrderManager** ← All overlays (via ResourceManager delegation)

### Frame Pipeline Flow
1. **Capture** (GL/Monitor) → **FrameExchange** → **Renderer** → **Display**
2. **Threading**: All timers via `ThreadManager.single_shot`
3. **Cleanup**: All resources via `utils.resource_manager` helpers

### Event Flow
1. **Input** → **KeyPassthroughController** → **MediaController** (if enabled)
2. **Hotkeys** → **HotkeyManager** → **Controllers** (QuickSwitch, Opacity)
3. **UI Events** → **WindowBehaviorManager** → **Overlay Actions**

---

*This index reflects the current state of the SPQModular codebase. Always regenerate based on live code when making significant changes.*
