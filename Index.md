# 📚 SPQModular Codebase Index

> **AI Agents**: This is the canonical codebase index. Always regenerate based on live codebase state. Do not append or create changelogs.

## 📋 Quick Navigation

- [🏁 Entry Point](#-entry-point)
- [🧠 Core Services](#-core-services)
- [🪟 Window Management](#-window-management)
- [🎨 Graphics & Overlays](#-graphics--overlays)
- [🎛️ Input Handling](#-input-handling)
- [🔊 Media Control](#-media-control)
- [🎯 User Interface](#-user-interface)
- [🔧 Utilities](#-utilities)
- [🧪 Testing](#-testing)
- [📁 Resources & Config](#-resources--config)
- [🧵 Concurrency](#-concurrency)
- [📊 Events](#-events)

## 📎 Related Docs

- [Understanding Guide](./Understanding.md) — end-to-end operational overview of how the system works (docking lifecycle, switching, validation, threading, debugging recipes).
- [Spec](./Spec.md) — architecture policies and cross-cutting standards.

## 🧠 Core Services

### Application Core (`core/application/`)
- **`ApplicationCore`**: Central application controller
  - **Lifecycle Management**:
    - `initialize()`: Sets up all core services
    - `shutdown()`: Cleanup in reverse initialization order
    - `restart()`: Graceful application restart
  - **Service Access**:
    - `get_service(service_type)`: Service locator pattern
    - `has_service(service_type)`: Check service availability
  - **State Management**:
    - `state_changed` signal
    - `get_state()`: Current application state
    - `set_state(state)`: Update application state

### Settings System (`core/settings/`)
- **`SettingsManager`**: Centralized configuration with comprehensive defaults
  - **Features**:
    - JSON-based persistence with fallback hierarchy
    - Type-safe get/set operations with validation
    - Change notifications via signals
    - Comprehensive settings structure (25+ settings)
    - Singleton pattern with test isolation support
  - **File Location Hierarchy**:
    1. Explicit path (for tests): `settings_file` parameter
    2. Primary: `<runtime_root>/settings/settings.json`
    3. Fallback: `<executable_dir>/settings.json`
    4. Last resort: `~/.spqmodular/settings.json`
  - **Key Methods**:
    - `get(key, default=None)`: Get setting value with type conversion
    - `set(key, value)`: Update setting with validation
    - `save()`: Persist changes to correct file location
    - `reset_to_defaults()`: Restore comprehensive defaults
    - `_reset_for_testing()`: Reset singleton for test isolation
  - **Settings Categories**: Appearance, Behavior, Hotkeys, Graphics, General, Experimental
  - **Thread Safety**: All operations are thread-safe with UI-thread enforcement

### Resource Management (`core/resources/`)
- **`ResourceManager`**: Centralized resource lifecycle
  - **Features**:
    - Automatic cleanup on shutdown with proper handler signatures
    - Memory leak prevention
    - Resource tracking and monitoring
    - Lambda-wrapped cleanup handlers for consistent parameter passing
  - **Key Methods**:
    - `register(resource, cleanup_fn)`: Track resource with lambda wrapper
    - `unregister(resource_id)`: Remove tracking
    - `cleanup_all()`: Force cleanup of all resources
  - **Thread Safety**: All operations are thread-safe
  - **Cleanup Handler Pattern**: `cleanup_handler=lambda obj: obj._cleanup()` for consistent signatures

### Event System (`core/events/`)
- **`EventSystem`**: Publish-subscribe bus with rich Event objects
  - **Features**:
    - Rich Event objects with metadata (id, timestamp, source, is_handled)
    - Priority-based dispatch with proper ordering
    - Thread-safe operations with UI dispatch support
    - Event lifecycle management and history tracking
    - Debugging capabilities with event tracing
  - **Event Object Structure**:
    - `event.type`: Event type string
    - `event.data`: Payload data dictionary
    - `event.id`: Unique event identifier
    - `event.timestamp`: Event creation time
    - `event.source`: Event source object
    - `event.is_handled`: Consumption flag for stopping propagation
  - **Key Methods**:
    - `subscribe(event_type, callback, dispatch_on_ui=False)`: Add subscriber
    - `unsubscribe(subscription_id)`: Remove subscriber by ID
    - `publish(event_type, data, source)`: Dispatch event with Event object
    - `wait_for(event_type, timeout, condition)`: Wait for specific events
  - **Handler API**: Handlers receive Event objects, access data via `event.data`

## 🎨 Graphics & Overlays

### Backend Management (`core/graphics/`)
- **`BackendManager`**: Overlay backend selection
  - **Available Backends**: DWM, Software, Monitor
  - **Selection Logic**: Priority-based with fallback
  - **Docking Mode Support**: Automatically selects DWM backend for `OverlayType.DOCKING`
  - **Key Methods**:
    - `get_available_backends()`: List available backends
    - `select_backend(preference, overlay_type)`: Choose optimal backend with type-specific preferences
    - `create_overlay(config)`: Factory method for overlays
  - **ResourceManager Integration**: Proper cleanup handlers for deterministic resource management

### Docking System (`core/graphics/docking/`)
- **`DockingOverlayManager`**: Multi-overlay docking mode controller
  - **Features**:
    - Three-overlay system (main 100%, secondary strictly decreasing: B≈70% of A, C/D/E decay by ~15% each; floors via min sizes)
    - **SINGLE SOURCE OF TRUTH MRU**: Complete architectural refactor - eliminated all local MRU caches. DockingManager reads directly from `MRUManager` via `_get_current_mru()` helper. No listeners, no synchronization, no stale data. See `audits/mru_single_source_of_truth_COMPLETE.md`
    - **Quickswitch Flow**: Hotkey entry remains centralized in `core/switching/quickswitch_controller.py`; when docking is active, the controller calls the docking manager (no callback expected). The docking manager handles quickswitch locally (no recursion to controller).
    - **App Instance Injection**: Proper app instance injection for context menus using centralized provider pattern
    - **Debug Log Suppression**: Uses `utils.debug.log_suppressor` to reduce excessive debug spam while preserving errors/warnings
    - **Context Menu SST Violation Prevention**: Fixed timing issue where DWM overlays initialized before docking manager became active, causing context menu creation errors
    - **Lock-Aware Autoswitch**: Enhanced autoswitch to properly respect individual overlay lock states through `_is_window_locked` property
    - **Initialization Sequencing**: Proper `_is_initializing` flag prevents context menu creation during overlay initialization phase
    - **Reduced Debug Spam**: Eliminated repeated debug messages in opacity animation loops and thumbnail property updates
    - Automatic positioning with screen-aware bounds checking to prevent overlay C overlap/shifting during resize
    - Tight content framing on creation/recreation/Correct AR: secondary widths include canvas content insets and cached AR so OUTER geometry hugs INNER content (no side gaps)
    - Docking overlays skip extra DWM-level thumbnail inset (canvas `content_rect()` already accounts for borders/accents) to avoid double-inset blank space
    - Inward/outward positioning based on screen edge proximity with vertical clamp to the screen work-area bottom (`availableGeometry`) and bounds validation
    - Min-size-aware fit scaling of secondary overlays when space is constrained; floors enforced via `utils/window/overlay_constants.py` (`OVERLAY_MIN_WIDTH`, `OVERLAY_MIN_HEIGHT`)
    - Overlay interaction handling (mode-aware):
      - **Normal Mode**: Sticky window assignments - overlays maintain content unless explicitly changed
        - Double-click main (A): Quickswitch to next MRU item (only A updates)
        - Double-click secondary (B/C/D/E): Swap overlay content with foreground
        - Quickswitch hotkey: Focuses overlay A's window, moves foreground to A, replaces source overlay if needed
      - **Cycle Mode**: Dynamic MRU-based assignment - all overlays update on MRU changes
        - Double-click any overlay: Focus that window, all overlays update to show next N MRU items
        - Quickswitch hotkey: Rotates MRU, focuses next window, all overlays update
      - Quickswitch (hotkey): Centralized controller triggers DockingOverlayManager, which performs mode-aware swap locally (no recursion)
    - Lock-aware automatic switching via centralized switching controllers
    - Resource lifecycle management with proper cleanup and ResourceManager integration
    - Secondary overlays apply direct geometry with screen bounds validation to prevent overlap
    - Movement binding uses event filter with proper host availability checking
    - Batch geometry guard: `_batch_applying` suppresses host Move/Resize reactions during group apply to eliminate feedback loops/teleporting; cleared after scheduled UI tasks complete
    - Secondary drag delegation: secondary hosts do not start local drags; `DockingOverlayManager.eventFilter()` translates the main overlay by global mouse deltas during a secondary-held drag and coalesces a sync
    - Unified wheel-resize: secondary wheel is intercepted to scale the main overlay; prevents one-frame host distortion on secondaries
    - Wheel-resize clamping: main overlay wheel-resize is clamped within the current monitor’s FULL bounds; size is reduced if necessary while preserving INNER AR using canvas insets
  - **Geometry Persistence Architecture** (CRITICAL - 2025-10-09):
    - Physical pixel-based persistence to `docking.last_state` using nearest-corner state
    - Proper DPR scaling: converts logical ↔ physical pixels for multi-DPI correctness
    - Anti-cascade protection: eliminated bidirectional sizing (secondaries NEVER resize main)
    - Batch mode protection: `_batch_applying` flag blocks resize events during sync operations
    - Initialization gating: `_is_initializing` flag blocks all syncs until system stabilizes
    - Delayed initial sync: 100ms delay after overlays shown prevents restoration cascade
    - 2-second suppression window after restoration prevents spurious saves
    - Single sync source architecture: only delayed initial sync runs on startup
    - See `audits/docking_persistence_complete_call_chain_analysis.md` for full details
  - **Overlay E Bottom Alignment Fix** (2025-10-09):
    - Fixed asymmetric canvas inset handling in secondary overlay sizing
    - Canvas insets now properly added to both width AND height (was only width)
    - Eliminates 12px downward drift at minimum sizes for rightmost overlays
    - Top and bottom alignment now work identically at all sizes
    - See `audits/overlay_e_bottom_alignment_investigation.md` for technical details
  - **Key Methods**:
    - `create_docking_system(config)`: Initialize 3-overlay system
    - `destroy_docking_system()`: Clean shutdown with resource cleanup
    - `sync_overlay_properties()`: Synchronize size, opacity, position
      - Single source of truth for positioning (all legacy positioners removed). Implements inward/outward placement with min-size-aware scaling and sequence-wide clamping. Emits `FIT` and `POSITION` diagnostics.
    - `handle_overlay_interaction(overlay_id, interaction_type)`: Process user interactions
    - `update_mru_list(hwnd_list)`: Update MRU and sync with main system; dedupe and clamp capacity
    - `eventFilter(obj, event)`: Handle drag hide/show, secondary-drag group translation, batch-guard suppression, and coalesced sync
    - `_populate_mru_with_visible_windows()`: Fallback enumeration for insufficient MRU

- **`DockingOverlay`**: Individual overlay wrapper for docking mode
  - **Features**:
    - Size ratio-based scaling relative to main overlay
    - Target window management with HWND tracking
    - Host wiring: injects `_parent_overlay` into `OverlayHost` so focus indicator lock toggle and key passthrough routing operate in docking mode
    - Mouse event handling (double-click, context menu)
    - ResourceManager integration for proper cleanup
    - Batch-aware event filtering: during `_batch_applying`, overlay-level sync/wheel handling is suppressed to avoid recursive updates
    - Secondary overlays do not initiate local drag/resize; wheel on a secondary scales the main overlay and triggers a coalesced group sync (smooth resize without host jitter)
  - **Key Methods**:
    - `initialize()`: Setup overlay with backend integration
    - `set_target_window(hwnd)`: Assign window to display
    - `sync_with_main(geometry, opacity)`: Synchronize with main overlay
    - `cleanup()`: Resource cleanup including ResourceManager unregistration

#### Focus Indicator & Lock (Docking)
- `ui/overlays/geometry/focus_indicator.py` emits `lock_toggled` on click. `core/graphics/overlay_host.py` handles it and forwards to the underlying DWM overlay `toggle_window_lock()`, then reflects the state via `FocusIndicator.set_locked(...)`. Added `FOCUS_AUDIT` logs for before/after state.
- Docking explicitly sets `OverlayHost._parent_overlay = IntegratedDWMOverlay` during `_setup_interaction_handling()` to make this path functional in docking mode.

#### Media Key Passthrough Targeting (Docking)
- `OverlayHost` resolves the routing target HWND in this order for docking: `_parent_overlay._source_hwnd`, `_parent_overlay._captured_hwnd`, `overlay._config.properties['hwnd']`. This is applied on key events, activation changes, and showEvent to avoid "no target hwnd" gaps after swaps.

#### Docking Context Menu (Styling Parity)
- `core/graphics/docking/context_menu.py` uses `QMenu` with `objectName='overlayContextMenu'` and applies theme. Menu and its submenus now use overlay-style QSS, matching DWM/Monitor menus in border, text weight, and sizing.
 

### DWM Integration (`core/graphics/backends/dwm/`)
- **`IntegratedDWMOverlay`**: Hardware-accelerated window thumbnails
  - **Features**:
    - Native DWM thumbnail rendering
    - Integrated border canvas with aspect ratio preservation
    - Hardware acceleration with robust aspect ratio caching
    - Minimal CPU usage
    - **Full window capture mode**: Captures entire window including title bar to fix modern Windows apps (Task Manager, Notepad, Settings) that use DWM Extended Frame technique
  - **Aspect Ratio Management**:
    - `_cache_source_aspect()`: Robust client area and window rect fallback
    - Bounds checking (0.2-5.0 ratio limits)
    - DPI-aware scaling for thumbnail properties
    - **AR calculation independent of capture mode**: Always uses `GetClientRect()` from source window
  - **Capture Mode** (2025-10-11):
    - Uses `source_client_area_only=False` in all thumbnail property updates
    - Fixes black title bar issue in modern Windows apps with extended frames
    - Shows title bars for all windows (matches Windows taskbar preview behavior)
  - **Key Methods**:
    - `set_source_window(hwnd)`: Set target window
    - `update_thumbnail()`: Refresh thumbnail with aspect preservation
    - `set_opacity(value)`: Adjust transparency

### Monitor Capture (`ui/overlays/monitor/`)
- **`MonitorOverlay`**: Screen capture overlay with DWM-quality rendering
  - **Features**:
    - Real-time monitor capture via DXGI
    - Integrated border canvas matching DWM overlay styling
    - Aspect ratio preservation using DWM overlay logic
    - Hall-of-mirrors detection
  - **Aspect Ratio Management**:
    - `_cache_monitor_aspect()`: Monitor dimension caching with bounds checking
    - Consistent scaling during manual resize operations
    - Proper letterbox/pillarbox handling
  - **Key Methods**:
    - `set_target_monitor(monitor_info)`: Configure capture target
    - `start_capture()`: Begin screen capture
    - `stop_capture()`: End capture with cleanup

## 🔊 Media Control

### Audio System (`utils/audio/`)
- **`AudioManager`**: Central audio control
  - **Features**:
    - Volume control
    - Mute/unmute
    - Session management
  - **Key Methods**:
    - `set_volume(level)`: 0.0-1.0
    - `get_volume()`: Current volume
    - `set_mute(muted)`: Toggle mute
    - `get_sessions()`: List audio sessions

### Media Session (`core/media/session.py`)
- **`MediaSession`**: Platform media integration
  - **Platforms**:
    - Windows: SMTC (System Media Transport Controls)
    - macOS: Now Playing Info
    - Linux: MPRIS (Media Player Remote Interfacing Specification)
  - **Events**:
    - `playback_state_changed`
    - `metadata_updated`
    - `volume_changed`

### Media Keys (`core/input/media_keys.py`)
- **`MediaKeyHandler`**: System media key interception
  - **Supported Keys**:
    - Play/Pause
    - Next Track
    - Previous Track
    - Volume Up/Down
  - **Integration**:
    - Global hotkey registration
    - Event emission

## 🎯 User Interface

### UI Framework (`ui/`)
- **`ThemeManager`**: Centralized theming
  - **Features**:
    - Light/dark theme support
    - Custom color schemes
    - Runtime theme switching
  - **Usage**:
    ```python
    theme = ThemeManager.instance()
    theme.set_theme('dark')
    ```

### Components (`ui/components/`)
- **`Toast`**: Notification system
  - **Features**:
    - Stacking notifications
    - Custom durations
    - Rich content support
  - **Example**:
    ```python
    Toast.show("Operation completed", "The task was successful", icon=Toast.ICON_SUCCESS)
    ```

- **`CircleCheckBox`**: Custom checkbox
  - **Features**:
    - Smooth animations
    - Custom colors
    - State management

### Windows (`ui/windows/`)
- **`MainWindow`**: Primary application window
  - **Features**:
    - Menu bar
    - Status bar
    - Central widget area
    - Dock widgets
  - **Key Methods**:
    - `add_dock_widget()`
    - `set_central_widget()`
    - `show_message()`

### Dialogs (`ui/dialogs/`)
- **`SettingsDialog`**: Application settings
  - **Sections**:
    - General
    - Appearance
    - Hotkeys
    - Advanced
  - **Features**:
    - Category navigation
    - Search functionality
    - Reset to defaults

- **`AboutDialog`**: Application information
  - **Sections**:
    - Version info
    - Credits
    - License
    - System information

## 🧵 Concurrency

### ThreadManager (`core/threading/manager.py`)
- **Purpose**: Centralized thread and timer management with lock-free design
- **Architecture**: **LOCK-FREE** - No raw threading locks or mutexes
- **Recent Fix**: ✅ UICoalescer lock reference bug resolved (2024-12-28)
- **Features**:
  - UI thread confinement enforcement
  - Worker thread pools (IO, COMPUTE, CAPTURE, RENDER, UI)
  - Single-shot and recurring timers
  - Task submission and lifecycle management
  - Lock-free singleton creation and state management
  - UICoalescer for batching UI operations with atomic scheduling
- **Key Methods**:
  - `run_on_ui_thread(func, *args, **kwargs)`: Execute on UI thread
  - `single_shot(ms, func, *args, **kwargs)`: One-time timer
  - `submit_task(func, pool_type, *args, **kwargs)`: Background task
  - `shutdown()`: Clean shutdown of all pools
  - `create_ui_coalescer()`: Factory for UI operation batching
- **Concurrency Model**: UI thread confinement + atomic operations + message passing
- **Migration Status**: ✅ Complete - All raw mutexes eliminated, UICoalescer fully lock-free

### Lock-Free Data Structures (`utils/lockfree/`)
- **`SPSCQueue`**: Single Producer Single Consumer Queue
  - **Features**:
    - Fixed-size ring buffer
    - Wait-free operations
    - Thread-safe for single producer/consumer
  - **Key Methods**:
    - `push(item)`: Add item (non-blocking)
    - `pop()`: Remove and return item (non-blocking)
    - `is_empty()`: Check if queue is empty

- **`TripleBuffer`**: Triple buffering for graphics
  - **Features**:
    - Lock-free reads/writes
    - Always-consistent reads
    - No allocation during operation
  - **Key Methods**:
    - `write(new_data)`: Update buffer
    - `read()`: Get latest consistent data
    - `try_read()`: Non-blocking read

## 📊 Events

### Event Types (`core/events/types/`)
- **`WindowEvents`**: Window-related events
  - `WindowCreated(hwnd, rect)`
  - `WindowDestroyed(hwnd)`
  - `WindowMoved(hwnd, new_rect)`
  - `WindowFocused(hwnd)`
  - `WindowMinimized(hwnd)`
  - `WindowRestored(hwnd)`

- **`InputEvents`**: User input events
  - `KeyPressed(key, modifiers)`
  - `KeyReleased(key, modifiers)`
  - `MouseMoved(x, y, buttons)`
  - `MousePressed(button, x, y)`
  - `MouseReleased(button, x, y)`
  - `WheelScrolled(delta, orientation)`

- **`SystemEvents`**: System-level events
  - `DisplayChanged()`
  - `SystemThemeChanged(theme_name)`
  - `SystemShutdown()`
  - `SystemWake()`

### Event Handling Patterns
- **Event Handlers**:
  - Should be fast and non-blocking
  - Use `ThreadManager` for heavy operations
  - Handle exceptions gracefully
  - Unsubscribe when no longer needed

- **Common Pitfalls**:
  - Blocking the event loop
  - Memory leaks from unremoved handlers
  - Race conditions in event handlers
  - Deadlocks from nested event processing

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

### `version.py`
- **Single Source of Truth** for application version
  - **Version Components**:
    - `VERSION_MAJOR`, `VERSION_MINOR`, `VERSION_PATCH`: Semantic version numbers
    - `VERSION_SUFFIX`: Alpha/beta/RC suffix (e.g., "a", "b", "rc")
    - `__version__`: Full version string (e.g., "2.1.0a")
    - `VERSION_WIN32`: Windows PE metadata format (4-part, e.g., "2.1.0.0")
  - **Application Metadata**:
    - `APP_NAME`, `APP_DISPLAY_NAME`, `APP_COMPANY`, `APP_DESCRIPTION`
  - **Build Metadata** (set by build scripts):
    - `BUILD_DATE`, `BUILD_TIME`, `BUILD_COMMIT`
  - **Functions**:
    - `get_version_string(include_build=False)`: Formatted version with optional build info
    - `get_full_version_info()`: Complete version dictionary
  - **Usage**:
    - Python runtime: `from version import __version__`
    - Build scripts: PowerShell `Get-AppVersion` function parses version.py
  - **Versioning Strategy**: Semantic versioning (Major.Minor.Patch)

### `main.py`
- **`PiPApplication`**: Main Qt application class
  - **Key Methods**:
    - `__init__`: Initializes Qt application and core services
    - `_setup_connections`: Sets up signal/slot connections
    - `_on_about_to_quit`: Cleanup handler
  - **Signals**: `shutdown_initiated`, `all_windows_closed`
  - **Dependencies**: `SystemTrayManager`, `ThreadManager`, `ResourceManager`
  - **Pattern**: Service locator + dependency injection

- **`main()`**: Application bootstrap and service injection
  - **Path Resolution**: Uses `utils.paths.get_runtime_root()`
    - Honors `SPQ_RUNTIME_ROOT` (set by native launcher)
    - Normalizes compiled layouts (`<root>/data/bin` or `<root>/data`) to `<root>`
    - Logs directory: `<root>/logs`
    - Icons: `data/resources/ShittyPIP.ico` (fallback to `resources/` in dev)
  - **Environment Flags**:
    - `--debug`: Sets `SPQ_DEBUG=1` for verbose logging
    - `--portable`: Forces portable mode for settings
  - **Error Handling**: Global exception hook for unhandled exceptions

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
  - Lazy (window overlays only): `QuickSwitchController`, `FocusTracker`, `ForegroundAutoswitchController`, Media KeepAlive (guarded by `features.media_control_enabled`)
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

#### `hide_show_controller.py` - HideShowController
- **Purpose**: Global hotkey (Ctrl+Shift+H) to toggle overlay visibility
- **Implementation**: Uses `OverlayStateManager` for robust state capture/restore
- **Integration**: All hide/show mechanisms (context menu, tray, hotkey) now use `OverlayStateManager` as single implementation
- **Behavior**: 
  - First press: Hides all overlays, captures full state (geometry, opacity, locks, assignments)
  - Second press: Restores overlays from saved state
  - Works with both single overlay and docking modes
- **No suppression**: Like other multipress hotkeys, doesn't block key from reaching applications

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
  - **Docking integration**: when docking is active, QuickSwitch delegates to `DockingOverlayManager.handle_overlay_interaction("main", "quickswitch")` which triggers `_rotate_mru_forward()` for proper Alt+Tab behavior. Per-overlay locks prevent automatic switching while still allowing manual double-click actions via the docking manager. Fresh ResourceManager lookup per call ensures proper delegation even without cached references
  - **CRITICAL FIX (2025-09-18)**: Fixed broken `_rotate_mru_forward()` method in docking manager that only worked once - now properly rotates MRU list like classic Alt+Tab
  - **MAJOR MODE-AWARE IMPLEMENTATION (2025-10-05)**: Refactored docking mode to properly distinguish Normal vs Cycle mode behaviors:
    - **Normal Mode**: Sticky window assignments - overlays maintain content unless explicitly changed
      - `_update_normal_mode_displays()`: Preserves assignments, only updates invalid/closed windows
      - Quickswitch/double-click: Targeted updates (only affected overlay changes)
      - Assignment tracking via `_normal_mode_assignments` dict
    - **Cycle Mode**: Dynamic MRU-based assignment (existing behavior preserved)
      - `_update_cycle_mode_displays()`: Updates all overlays based on MRU order (excluding foreground)
      - Quickswitch/double-click: Full update of all overlays
    - Helper methods: `_update_mru_order_only()`, `_assign_overlay()`, `_normal_mode_swap_with_foreground()`
    - Mode transition: Captures current state as sticky baseline when switching Cycle→Normal
    - Result: Normal mode no longer cascades changes to all overlays; B/C/D/E maintain content correctly
- **Polling**: 250ms interval using `ThreadManager.single_shot` self-rescheduling (no raw QTimer)
- **Debouncing**: 300ms stability requirement before triggering swaps
- **Lock Awareness**: Respects both global (`OverlayManager.is_overlay_locked()`) and individual overlay locks (`_is_window_locked`)
- **Docking Integration**: 
  - **MRU-Aware Logic**: Only triggers autoswitch when focusing the SAME window as main overlay (follows Spec.md policy)
  - **Cycle Mode Gating**: Automatically disabled when `docking.mode="cycle"` is active
  - **Lock Checking**: Properly checks docking overlay lock states before triggering cycles
- **Emergency Recovery**: Handles invalid/missing overlay sources by selecting valid MRU alternatives
- **Display Lock Support**: Optional monitor-based switching via `features.display_locked_switching`
- **Suppression API**: `suppress_for(duration_ms, last_seen_hwnd)` for coordination with QuickSwitch
- **Events**: Publishes `switch.lock_suppressed` when blocked by overlay locks
- **Threading**: All operations run on UI thread via `ThreadManager`
- **Settings**: `features.autoswitch_enabled` (bool), `docking.mode` (gates cycle mode)
- **Logging**: AUTOSWITCH prefix with detailed cycle/emergency diagnostics

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
 - **Focus Gating & Routing**: `OverlayHost` sets target HWND when the overlay is active; `KeyPassthroughController` now routes media/volume keys ONLY when the overlay is focused. When unfocused, media/volume key requests are blocked (telemetry includes explicit reasons like `media-key-overlay-not-focused`).
 - **Rate Limiting**: ~18ms min interval, 10ms keyup delay
 - **Threading**: `ThreadManager.run_on_ui_thread`, `ThreadManager.single_shot` (no raw QTimer)
 - **Arrow→Volume Remap (media-enabled)**: When `features.media_control_enabled` is ON, `VK_UP` and `VK_DOWN` are internally remapped to `VK_VOLUME_UP`/`VK_VOLUME_DOWN` at the start of `passthrough_key`, `press_passthrough_key`, and `release_passthrough_key`. This triggers identical hold behavior and prevents arrow key events from being forwarded while active.
 - **Logging**: KEYPASS prefix
  - Verbose diagnostics gated by `debug.keypassthrough_verbose` (includes media routing and fallbacks)
 - **Focused-only Media Routing**: Media and system volume keys are processed only when an overlay is focused; otherwise they are blocked to avoid unintended global handling. Focus check failures fail closed (block) and emit `key.passthrough.blocked` with reasons like `media-key-focus-check-failed`.
 - **Browser-aware Child Fallback (media-disabled playback controls)**: When `features.media_control_enabled` is OFF and the target HWND is a supported browser (Chrome, Edge, Firefox, Discord), playback keys SPACE/LEFT/RIGHT are sent via `MediaController._send_browser_hotkey(...)` to child content windows first; top-level is used as a fallback. Publishes `key.passthrough.forwarded` with note `browser-child` on success.
 - **Verbose Non-media Logging**: Toggle `debug.keypassthrough_verbose` to emit reasoned decision logs for non-media paths (early returns, target validation, routing results). Off by default to avoid noise.
 - **Volume Keys Policy**:
  - Uses VK_VOLUME_UP (0xAF), VK_VOLUME_DOWN (0xAE), VK_VOLUME_MUTE (0xAD)
  - Routed locally via `MediaController.*_for_hwnd(hwnd)` with session-volume, app-hotkey, then mixer fallback
  - Continuous hold is owned by `MediaController`: on press, `KeyPassthroughController` performs an immediate step then delegates the hold loop to `MediaController` (token-guarded, `ThreadManager.single_shot`). On release, KPC calls `MediaController.handle_volume_key_release(hwnd)`; if the target hwnd is missing (handoff), it calls `MediaController.stop_all_continuous_volume_adjustments()` to prevent runaways.
  - Arrow-to-volume remap: When media control is enabled, `VK_UP`/`VK_DOWN` are treated equivalently to volume keys for press/hold/release semantics (hold timers apply; arrow key events are not forwarded during active media control)
  - Configurable timing via settings (applied by MediaController): `input.volume_hold_initial_delay_ms` (default 200), `input.volume_hold_interval_ms` (default 50)
  - Minimal suppression window (~35ms) is still enforced to smooth bursts and avoid flooding
  - Never intercepts globally; only routes when a valid target HWND is set and the overlay is focused
  - Arrow keys LEFT/RIGHT remain previous/next track; never mapped to volume
  - Auto-repeat guard: `OverlayHost` ignores auto-repeat `KeyPress` and `KeyRelease` for volume and arrow keys so only physical press and the final physical release are honored.
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
  - **Targeted Browser Selection**: `_send_browser_media_command_targeted()` matches DWM overlay source HWND to correct browser child window using title/class similarity scoring and media content detection
  - **Enhanced App Detection**: `_derive_app_name_from_context()` resolves UNKNOWN applications using process names, window titles, class names, and media characteristics (40+ applications supported)
  - **Continuous Volume Control**: Key hold detection with smooth 0-100% ramping via `handle_volume_key_press/release()` - immediate 2% step + continuous 1% steps after 0.5s hold
  - **OSD Feedback Policy**: `media.volume.changed` events are ALWAYS published immediately on volume key press/hold for consistent user feedback, even when no audio session exists yet (numeric level shows when session becomes active)
  - **Volume Key Gating**: Aligned with spacebar policy - requires app/overlay to be handling keys and valid target hwnd (no strict overlay-focus requirement)
  - Hints media activity to KeepAlive on successful commands to maintain background responsiveness
  - Process responsiveness checking before dispatch
  - Window enumeration caching (5s TTL)
- **Overlay Integration**: `get_preferred_app()` detects from overlay target HWND
- **Multi-Window Targeting**: `_find_window_by_app()` prioritizes overlay target window over generic window selection for precise control with multiple browser/app instances
- **Settings**: `media.app_catalog`, `media.preferred_apps`, `media.volume_step` (default 2%, range 1%-25%)
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

- **Wheel Resize Behavior (tuned)**:
  - Dynamic fine-step near minimum bounds (no modifiers required) to prevent step skipping at extremes
  - **Screen boundary performance** (2025-10-09 fix): Boundary smoothing logic now checks for **stable pinned edges** before reducing resize steps. When a window is pinned to a screen edge (e.g., left edge at x=0) and growing away from that edge, full resize deltas are applied without smoothing. Only triggers 2px smoothing when the window is actually trying to move beyond screen limits. This eliminates the slow resize issue at screen corners (especially bottom-left) while maintaining proper boundary handling.
  - Min-size inner-AR lock when `content_aspect` is provided (maintains aspect ratio at minimum sizes using canvas insets)
  - Opportunistic persistence: after wheel-apply and drag/resize release, if the host exposes `_persist_current_geometry()`, it is called to persist geometry (standalone DWM)

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

- **`overlay.py`**: Base `Overlay` class with common interface
- **`overlay_host.py`**: Host window for overlays
- **`OverlayManager`**: Centralized overlay lifecycle management with MRU tracking and locking support
- **`BackendManager`**: Graphics backend selection and management (DXGI, fallbacks)
- **`ZOrderManager`**: Window z-order enforcement and management
- **Backends**: Hardware-accelerated capture and rendering backends
  - **DXGI Backend**: Primary Windows capture backend using DXGI desktop duplication
  - **DWM Backend**: Integrated DWM overlay system with hardware acceleration
- **Docking System (`core/graphics/docking/`)**: Three-overlay docking mode implementation
  - **`DockingOverlayManager`**: Orchestrates 3-overlay docking system with expanded MRU
  - **`DockingOverlay`**: Individual overlay wrapper with synchronization and interaction handling
  - **`DockingPositioner`**: Spatial positioning logic with directional priority (right > left > bottom > top)
  - **`DockingContextMenu`**: Docking-specific context menu with limited options
- **Types**: Core graphics type definitions and configurations
    - `set_overlay_lock(locked)`: Enable/disable overlay modifications
    - `get_mru_overlays(limit)`: Get overlays sorted by recent use
    - `set_auto_switch_enabled(enabled)`: Enable/disable auto-switching
- **`backend_manager.py`**: Backend selection and initialization
- **`types.py`**: Shared type definitions
- **`window_monitor.py`**: Closed-window switching functionality for overlays
  - **`WindowMonitor`**: Monitors window validity and detects closures (2s polling)
  - **`ClosedWindowSwitchManager`**: Manages auto-switching when windows close
  - **Features**: Automatic overlay source switching to next valid window from MRU list when a window closes
  - **NOTE**: This is separate from foreground-based autoswitch (see ForegroundAutoswitchController)

#### Border Rendering System (`ui/overlays/`)
- **`integrated_border_canvas.py`**: Unified canvas with integrated border rendering
  - **Features**: Direct border rendering in paintEvent, no separate border windows
  - **Masking**: Window-level masking via QPainterPath/QRegion for rounded borders
  - **Theme Integration**: Automatic theme updates and DPI scaling
- **`accent_calculator.py`**: Unified accent calculation system
  - **Purpose**: Single source of truth for inner accent thickness, inset, and radius
  - **Features**: Bounded size scaling, DPI-aware calculations, coordinate alignment validation
  - **API**: `calculate_accent_properties()` with guaranteed DWM/border alignment
- **`rendering/border_renderer.py`**: Core border drawing primitives
- **`geometry/border_geometry.py`**: Border metrics calculation (main border only)
- **`theming/border_theme.py`**: Theme token integration

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

### Cursor Management (`utils/cursor_manager.py`)
- **Purpose**: Centralized cursor state management
- **Concurrency**: Lock-free design with UI thread confinement (migrated 2024-12-28)
- **Priority System**: Request-based cursor control
- **Integration**: Works with overlay focus states
- **Used By**: Overlay interactions, resize operations

### Z-Order Management (`utils/z_order_manager.py`)
- **Purpose**: Unified Z-Order Management System
- **Concurrency**: Lock-free design with UI thread confinement (migrated 2024-12-28)
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

### Mouse Capture Coordination (`utils/mouse_capture_coordinator.py`)

#### `MouseCaptureCoordinator`
- **Purpose**: Centralized mouse capture management
- **Concurrency**: Lock-free design with UI thread confinement (migrated 2024-12-28)
- **Priority System**: Context menu > Window behavior > Default
- **Stack Management**: Automatic capture restoration
- **Resource Registration**: ✅ Registered with ResourceManager
- **Used By**: Context menus, window drag/resize operations

---

## 🧪 Testing

### Test Configuration (`tests/`)

#### `conftest.py`
- **Purpose**: Pytest configuration and logging setup
- **Logging**: Centralized rotating logs at `logs/tests/`
- **Environment**: `SPQ_TEST_LOG_DIR` exported
- **Timeout**: Extended to 60 seconds for real integration tests
- **Status**: Active with comprehensive real functionality testing

#### Test Structure
- **`test_real_functionality.py`**: Comprehensive real integration tests (replaces mocked unit tests)
  - **TestRealResourceManagement**: Manager registration and cleanup validation
  - **TestRealSettingsManager**: Settings file I/O, fallback hierarchy, comprehensive defaults
  - **TestRealEventSystem**: Event subscription, dispatch with Event objects, resource cleanup
  - **TestRealBackendManagement**: Backend discovery and resource integration
  - **TestRealWindowValidation**: Window API validation functions
  - **TestRealCrossManagerInteractions**: Full system integration testing
  - **TestRealImportIntegrity**: Module import validation across codebase
- **`integration/`**: Legacy integration tests (deprecated in favor of real functionality tests)
- **`fixtures/`**: Test fixtures and utilities

#### Testing Philosophy
- **Real Integration Tests**: Uses actual components without mocks to expose genuine system issues
- **Two+ Way Testing**: Tests validate multiple components and interactions simultaneously
- **Singleton Reset**: SettingsManager includes `_reset_for_testing()` for test isolation
- **Resource Cleanup**: All tests validate proper ResourceManager integration

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

### Qt Resource Registration
 - **Qt resources registration**: Qt resources are compiled into `ui.resources_rc` and registered by both `main.py` and `utils/theme/theme_manager.py` at import time. This ensures `:/themes/*.qss` and other `:/` resource paths are available in tests and non-main contexts. On failure, a DEBUG log is emitted and the ThemeManager falls back to filesystem paths.

### Build Scripts
- **`scripts/build_onefile_local.ps1`** ⭐ RECOMMENDED: Nuitka onefile with persistent cache
  - Creates single EXE with persistent cache extraction
  - Output: `release/onefile_local/SPQ.exe`
  - Extracts to `%LOCALAPPDATA%\SPQ\runtime` (reused between launches)
  - AV-safe (proven - onefile builds pass Windows Defender)
  - First launch ~2-3s, subsequent launches instant
- **`scripts/build_onefile.ps1`**: Nuitka single-file build (temp extraction)
  - Creates single portable EXE
  - Output: `release/onefile/SPQ.exe`
  - Extracts to temp on every launch
  - AV-safe baseline (proven in testing)
- **`scripts/build_standalone.ps1`**: Nuitka standalone build (clean layout)
  - Creates directory: `SPQ.exe` in root, all DLLs in `data/` subdirectory
  - Output: `release/standalone/SPQ.exe`
  - ⚠️ KNOWN ISSUE: Triggers Windows Defender on static file analysis
  - Fast startup, no unpacking delay
  - Minimal flags + module exclusions
- **`scripts/build_nuitka_single_av_safe_minimal.ps1`**: Original baseline (kept for reference)
- **See:** `audits/build_script_comparison.md` for detailed comparison and recommendation

### Configuration Files
- **`pyproject.toml`**: Python project configuration
- **`Spec.md`**: Project specification
- **`Index.md`**: This codebase index
- **`ThemeIndex.md`**: Theme system documentation

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
