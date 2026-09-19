# SPDX-License-Identifier: Apache-2.0
"""Errors that require replacing the encoder process, not retrying a request."""


class EncoderGraphUnrecoverableError(RuntimeError):
    """An NPU graph may contain an incomplete capture or replay/update handshake."""
