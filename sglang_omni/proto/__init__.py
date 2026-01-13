# SPDX-License-Identifier: Apache-2.0
# Import SHMMetadata from relay.nixl where it has full support for new format
from sglang_omni.relay.nixl import SHMMetadata

from .messages import (
    AbortMessage,
    CompleteMessage,
    DataReadyMessage,
    ShutdownMessage,
    SubmitMessage,
    parse_message,
)
from .request import RequestInfo, RequestState
from .stage import StageInfo

__all__ = [
    "SHMMetadata",
    "DataReadyMessage",
    "AbortMessage",
    "CompleteMessage",
    "SubmitMessage",
    "ShutdownMessage",
    "parse_message",
    "RequestState",
    "RequestInfo",
    "StageInfo",
]
