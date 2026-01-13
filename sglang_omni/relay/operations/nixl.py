# SPDX-License-Identifier: Apache-2.0
"""NIXL operation classes."""


from sglang_omni.relay.nixl import (
    RdmaMetadata,
    ReadableOperation,
    ReadOperation,
    WritableOperation,
    WriteOperation,
)

# Re-export NIXL operation classes
# These classes are defined in sglang_omni.relay.nixl.connector
# and are re-exported here for API consistency

__all__ = [
    "ReadOperation",
    "ReadableOperation",
    "WriteOperation",
    "WritableOperation",
    "RdmaMetadata",
]
