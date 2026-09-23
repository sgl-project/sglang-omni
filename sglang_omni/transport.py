# SPDX-License-Identifier: Apache-2.0
"""Transport identifiers shared by platform policy and communication."""

from enum import Enum


class TransportKind(str, Enum):
    LOCAL_OBJECT = "local_object"
    CUDA_IPC = "cuda_ipc"
    SHM = "shm"
    MOONCAKE = "mooncake"
