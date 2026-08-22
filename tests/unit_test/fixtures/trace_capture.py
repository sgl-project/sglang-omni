# SPDX-License-Identifier: Apache-2.0
"""Capture helper for SGLANG_OMNI_COMM_TRACE events in tests.

Consumers of ``sglang_omni.profiler.comm_trace`` bind ``emit`` via
from-imports, so monkeypatching ``comm_trace.emit`` does not intercept their
calls. Instead this enables the env gate and attaches a logging handler to the
``sglang_omni.comm_trace`` logger, decoding each ``COMM_TRACE {json}`` line
back into a dict.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from contextlib import contextmanager


class _CommTraceHandler(logging.Handler):
    def __init__(self, events: list[dict]) -> None:
        super().__init__(level=logging.INFO)
        self.events = events

    def emit(self, record: logging.LogRecord) -> None:
        message = record.getMessage()
        prefix = "COMM_TRACE "
        if message.startswith(prefix):
            self.events.append(json.loads(message[len(prefix) :]))


@contextmanager
def capture_comm_trace(monkeypatch, *, enable: bool = True) -> Iterator[list[dict]]:
    """Yield the list of decoded trace events.

    ``enable=False`` leaves the env gate unset and still attaches the handler,
    so a test can assert that the layer emits nothing when tracing is off.
    """
    if enable:
        monkeypatch.setenv("SGLANG_OMNI_COMM_TRACE", "1")
    else:
        monkeypatch.delenv("SGLANG_OMNI_COMM_TRACE", raising=False)
    events: list[dict] = []
    handler = _CommTraceHandler(events)
    trace_logger = logging.getLogger("sglang_omni.comm_trace")
    previous_level = trace_logger.level
    trace_logger.addHandler(handler)
    trace_logger.setLevel(logging.INFO)
    try:
        yield events
    finally:
        trace_logger.removeHandler(handler)
        trace_logger.setLevel(previous_level)


def events_named(events: list[dict], name: str) -> list[dict]:
    return [event for event in events if event["event"] == name]
