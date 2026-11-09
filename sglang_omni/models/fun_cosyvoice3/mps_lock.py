# SPDX-License-Identifier: Apache-2.0
"""Serialize selected Fun-CosyVoice3 AR and vocoder calls on MPS."""

from __future__ import annotations

import threading
from types import TracebackType

MPS_LOCK = threading.RLock()


class MpsSerializedStep:
    """Share a reentrant host-side lock across AR and vocoder steps."""

    def __enter__(self) -> None:
        MPS_LOCK.acquire()
        return None

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> bool:
        MPS_LOCK.release()
        return False
