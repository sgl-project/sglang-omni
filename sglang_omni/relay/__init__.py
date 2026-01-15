# SPDX-License-Identifier: Apache-2.0
"""Relay module for inter-stage data transfer.

This module provides NIXL-based relay implementation for transferring data
between pipeline stages using RDMA.
"""

from sglang_omni.relay.base import Relay
from sglang_omni.relay.nixl import NixlRelay, NixlOperation, Connection, NIXL_AVAILABLE

__all__ = [
    "Relay",
    "NixlRelay",
    "NixlOperation",
    "Connection",
    "NIXL_AVAILABLE",
]
