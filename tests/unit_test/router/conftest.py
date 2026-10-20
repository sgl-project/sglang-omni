from __future__ import annotations

import logging
import sys
from collections.abc import Iterator
from pathlib import Path

import pytest

from sglang_omni_router.python.update_journal import STATE_DIR_ENV


@pytest.fixture(autouse=True)
def isolate_router_state_dir(
    tmp_path_factory: pytest.TempPathFactory, monkeypatch: pytest.MonkeyPatch
) -> Path:
    """Keep the default journal path out of the developer's real home.

    The router resolves it under $XDG_STATE_HOME or ~/.local/state, so a test
    that triggers a weight update would otherwise write there.
    """
    state_dir = tmp_path_factory.mktemp("router-state")
    monkeypatch.setenv(STATE_DIR_ENV, str(state_dir))
    return state_dir


@pytest.fixture(autouse=True)
def restore_router_logging(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Restore router logging and keep Abseil off pytest-owned capture streams."""
    absl_logging = sys.modules.get("absl.logging")
    if absl_logging is not None:
        monkeypatch.setattr(
            absl_logging.get_absl_handler().python_handler,
            "stream",
            sys.__stderr__,
        )
    else:
        pass
    logger = logging.getLogger("sglang_omni_router.python")
    saved = (list(logger.handlers), logger.propagate, logger.level)
    yield
    logger.handlers, logger.propagate, logger.level = (
        saved[0],
        saved[1],
        saved[2],
    )
