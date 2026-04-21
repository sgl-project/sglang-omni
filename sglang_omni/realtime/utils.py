# SPDX-License-Identifier: Apache-2.0
"""Small realtime utilities."""

from __future__ import annotations

import functools
import inspect
import time
from typing import Any, Callable, TypeVar

F = TypeVar("F", bound=Callable[..., Any])


def throttle(
    interval_s: float,
    *,
    timestamp_kw: str | None = None,
    state_attr: str = "_throttle_state",
) -> Callable[[F], F]:
    """Throttle instance method calls to at most once per interval.

    The decorated method must be called on an object instance. Per-instance
    throttle state is stored on ``state_attr`` as a dictionary keyed by the
    wrapped method name.
    """

    def decorator(func: F) -> F:
        key = func.__qualname__
        is_async = inspect.iscoroutinefunction(func)

        def resolve_timestamp(args: tuple[Any, ...], kwargs: dict[str, Any]) -> float:
            if timestamp_kw is not None:
                value = kwargs.get(timestamp_kw)
                if value is not None:
                    return float(value)
            return time.monotonic()

        def should_run(instance: Any, ts: float) -> bool:
            state = getattr(instance, state_attr, None)
            if state is None:
                state = {}
                setattr(instance, state_attr, state)
            last_ts = state.get(key)
            if last_ts is not None and (ts - float(last_ts)) < interval_s:
                return False
            state[key] = ts
            return True

        if is_async:

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if not args:
                    raise TypeError(
                        "throttle-decorated methods must be bound to an instance"
                    )
                ts = resolve_timestamp(args, kwargs)
                if not should_run(args[0], ts):
                    return None
                return await func(*args, **kwargs)

            return async_wrapper  # type: ignore[return-value]

        @functools.wraps(func)
        def sync_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not args:
                raise TypeError(
                    "throttle-decorated methods must be bound to an instance"
                )
            ts = resolve_timestamp(args, kwargs)
            if not should_run(args[0], ts):
                return None
            return func(*args, **kwargs)

        return sync_wrapper  # type: ignore[return-value]

    return decorator
