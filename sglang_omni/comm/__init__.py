# SPDX-License-Identifier: Apache-2.0
"""Unified communication primitives for stage, rank, and node data movement."""

from sglang_omni.comm.data_ref import (
    BackendRef,
    DataKind,
    DataLayout,
    DataRef,
    MetadataTensorRef,
    TensorMeta,
    TransportKind,
)
from sglang_omni.comm.engine import CommEngine
from sglang_omni.comm.router import CommRouter

__all__ = [
    "BackendRef",
    "CommEngine",
    "CommRouter",
    "MetadataTensorRef",
    "TensorMeta",
    "DataRef",
    "DataLayout",
    "DataKind",
    "TransportKind",
]
