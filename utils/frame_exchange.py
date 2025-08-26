"""
TripleBuffer-based frame exchange utilities.

Provides a centralized FrameExchange abstraction for producer/consumer handoff
with latest-frame semantics and optional back-pressure hints.

Usage:
    fx = FrameExchange()
    # Producer thread
    fx.publish(frame_data)
    # Consumer thread
    frame = fx.acquire_latest()

Notes:
- Uses utils.lockfree.TripleBuffer internally
- Single producer / single consumer recommended
- Payload is opaque (any Python object). Prefer immutable or copy-on-write data.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Generic, Optional, TypeVar
import time

from utils.lockfree import TripleBuffer

T = TypeVar("T")


@dataclass
class ExchangeStats:
    produced: int = 0
    consumed: int = 0
    drops: int = 0
    last_publish_ts: float = 0.0
    last_consume_ts: float = 0.0


class FrameExchange(Generic[T]):
    """Latest-frame TripleBuffer exchange with simple policies.

    - publish(obj): non-blocking; overwrites back buffer via TripleBuffer.publish()
    - acquire_latest(): returns most recent committed snapshot (may be None initially)
    - Optional coalescing via min_publish_interval_sec
    """

    def __init__(
        self,
        *,
        min_publish_interval_sec: float = 0.0,
        on_drop: Optional[Callable[[T], None]] = None,
    ) -> None:
        self._tb: TripleBuffer = TripleBuffer()
        self._min_pub_interval = max(0.0, float(min_publish_interval_sec))
        self._on_drop = on_drop
        self._stats = ExchangeStats()
        self._last_obj: Optional[T] = None

    def publish(self, obj: T) -> None:
        """Publish a new frame/object into the exchange.

        Applies optional coalescing based on min_publish_interval_sec. If a publish
        happens sooner than the interval, the previous unpublished object is
        considered dropped and on_drop is invoked.
        """
        now = time.perf_counter()
        if self._min_pub_interval > 0 and (now - self._stats.last_publish_ts) < self._min_pub_interval:
            # Coalesce/detect drop of the last unpublished object
            if self._on_drop and self._last_obj is not None:
                try:
                    self._on_drop(self._last_obj)
                except Exception:
                    pass
            self._stats.drops += 1
        # Publish the value to the TripleBuffer
        self._tb.publish(obj)
        self._last_obj = obj
        self._stats.produced += 1
        self._stats.last_publish_ts = now

    def acquire_latest(self) -> Optional[T]:
        """Acquire the latest committed object (may be None if none published)."""
        obj = self._tb.consume_latest()
        self._stats.consumed += 1
        self._stats.last_consume_ts = time.perf_counter()
        return obj  # type: ignore[return-value]

    def stats(self) -> ExchangeStats:
        return self._stats

    def reset_stats(self) -> None:
        self._stats = ExchangeStats()

    def set_on_drop(self, cb: Optional[Callable[[T], None]]) -> None:
        """Configure the drop callback used during coalesced publishes.

        Args:
            cb: Callback invoked with the previously unpublished object when a
                publish occurs sooner than min_publish_interval_sec.
        """
        self._on_drop = cb

    def set_min_publish_interval(self, seconds: float) -> None:
        """Configure the coalescing window for publishes.

        When greater than zero, a publish occurring sooner than this interval
        since the last publish will be considered coalesced; the previously
        unpublished object may be dropped and the configured on_drop callback
        will be invoked with it.

        Args:
            seconds: Minimum interval in seconds between publishes before
                     considering the previous value dropped/coalesced.
        """
        try:
            self._min_pub_interval = max(0.0, float(seconds))
        except Exception:
            # Keep previous value on invalid input
            pass

    def clear(self) -> None:
        """Explicitly clear retained references to help GC when stopping.

        This drops the last published object reference and resets counters.
        """
        try:
            self._last_obj = None
            # Best-effort: if TripleBuffer has a clear/reset API, call it.
            for meth in ("clear", "reset"):
                fn = getattr(self._tb, meth, None)
                if callable(fn):
                    try:
                        fn()
                    except Exception:
                        pass
        finally:
            self.reset_stats()


# Convenience factory
_def_exchanges: dict[str, FrameExchange] = {}


def get_exchange(name: str) -> FrameExchange:
    fx = _def_exchanges.get(name)
    if fx is None:
        fx = FrameExchange()
        _def_exchanges[name] = fx
    return fx
