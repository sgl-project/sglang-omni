# SPDX-License-Identifier: Apache-2.0
"""Relay module for inter-stage data transfer.

This module provides NIXL-based relay implementation for transferring data
between pipeline stages using RDMA.
"""

from sglang_omni_v1.relay.base import Relay
from sglang_omni_v1.relay.mooncake import MOONCAKE_AVAILABLE, MooncakeRelay
from sglang_omni_v1.relay.nixl import (
    NIXL_AVAILABLE,
    Connection,
    NixlOperation,
    NixlRelay,
)

__all__ = [
    "Relay",
    "NixlRelay",
    "NixlOperation",
    "Connection",
    "NIXL_AVAILABLE",
    "MooncakeRelay",
    "MOONCAKE_AVAILABLE",
]
