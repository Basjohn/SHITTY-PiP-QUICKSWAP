# Debug Package

This package provides centralized debugging and performance measurement utilities for the application.

## Features

### Logging Module
- **Thread-safe logging** with rotation and file handling
- **Environment variable configuration** via `SPQ_DEBUG` and `SPQ_PERF`
- **Runtime control** of debug and performance logging levels
- **Robust file handling** with automatic recovery from permission issues

### Performance Module
- **Function timing** with `@log_perf` decorator
- **Block timing** with `DebugTimer` context manager
- **Configurable thresholds** for performance logging

## Usage

### Basic Logging

```python
from utils.debug import get_logger, debug_enabled, debug_print, log_exception

# Get a logger
logger = get_logger('my_module')

# Log messages
logger.info("This is an info message")
logger.debug("Debug message")

# Conditional debug printing
if debug_enabled():
    print("Debug information")

# Or use the convenience function
debug_print("This will only print in debug mode")

# Log exceptions safely
try:
    # Code that might fail
    result = 1 / 0
except Exception as e:
    log_exception("Division failed", exc=e)
```

### Performance Measurement

```python
from utils.debug import log_perf, DebugTimer
import time

# Time a function
@log_perf(level=logging.INFO, threshold_ms=10.0)
def slow_operation():
    time.sleep(0.1)
    return "Done"

# Time a code block
def process_data(data):
    with DebugTimer("Data processing"):
        # Time-consuming operations
        time.sleep(0.2)
        return [x * 2 for x in data]
```

## Configuration

### Environment Variables
- `SPQ_DEBUG=1`: Enable debug mode (logs DEBUG level messages)
- `SPQ_PERF=1`: Enable performance logging
- `LOG_LEVEL`: Set the minimum log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `LOG_FILE`: Path to the log file (default: `app.log` in the current directory)

### Runtime Configuration

```python
from utils.debug import set_debug_mode, set_perf_logging

# Enable/disable debug mode at runtime
set_debug_mode(True)

# Enable/disable performance logging at runtime
set_perf_logging(True)
```

## Best Practices

1. **Use named loggers** for better log filtering:
   ```python
   logger = get_logger(__name__)
   ```

2. **Use debug_print** for temporary debugging that should be removed later.

3. **Wrap external calls** with DebugTimer to identify performance bottlenecks.

4. **Use log_exception** instead of logger.exception() for better error context.

5. **Set appropriate log levels**:
   - DEBUG: Detailed information for debugging
   - INFO: General operational information
   - WARNING: Indicates potential issues
   - ERROR: Serious problems that prevent normal execution
   - CRITICAL: Severe errors causing application failure

## Dependencies

- Python Standard Library
- No external dependencies
