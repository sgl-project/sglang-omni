# SPDX-License-Identifier: Apache-2.0
"""Failure reporting for explicitly optional Relay dependencies."""

import logging
from typing import NoReturn


class UnavailableDependency:
    """Retain a caught ImportError until an unavailable backend is selected.

    Call only from the backend's ImportError handler, passing its top-level
    import name. Only exact top-level absence is routine availability
    information; broken APIs and missing transitive imports remain errors.
    Backends must explicitly guard construction with raise_unavailable().
    """

    def __init__(
        self,
        *,
        package: str,
        backend: str,
        error: ImportError,
        logger: logging.Logger,
        install_hint: str,
    ):
        self._cause = error
        self._message = (
            f"{backend} is unavailable because importing {package!r} failed. "
            f"{install_hint}"
        )
        if isinstance(error, ModuleNotFoundError) and error.name == package:
            logger.debug("Optional dependency %s is unavailable: %s", package, error)
        else:
            logger.error(
                "Failed to import %s: %s. %s is unavailable.", package, error, backend
            )

    def raise_unavailable(self) -> NoReturn:
        raise RuntimeError(self._message) from self._cause
