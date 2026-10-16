# SPDX-License-Identifier: Apache-2.0
"""Locality classification and relay ownership for Omni communication."""
from __future__ import annotations

import logging
from collections.abc import Mapping
from contextlib import suppress
from typing import Literal

import torch

from sglang_omni.comm.data_ref import TransportKind
from sglang_omni.comm.stage_io import extract_tensors, packed_size_bytes
from sglang_omni.config.schema import (
    LEVEL_ZERO_IPC_MIN_TRANSFER_KB,
    LEVEL_ZERO_IPC_POOL_SIZE_MB,
    LEVEL_ZERO_IPC_SLOT_SIZE_KB,
    check_level_zero_ipc_sizes,
)
from sglang_omni.platforms import current_platform
from sglang_omni.profiler.comm_trace import emit as _comm_trace
from sglang_omni.profiler.comm_trace import enabled as _comm_trace_enabled
from sglang_omni.proto.request import StagePayload
from sglang_omni.relay.base import Relay, create_relay
from sglang_omni.relay.slot_pool import slots_for_size

logger = logging.getLogger(__name__)

_CUDA_IPC_FALLBACK_WARNED: set[tuple[str, str, int | None, int | None, str]] = set()


class CommRouter:
    """Maps an edge to the physical mover and owns relay instances.

    The router classifies locality. It does not define the stage protocol and it
    does not expose Mooncake-specific handles to stage code.
    """

    def __init__(
        self,
        *,
        stage_name: str,
        gpu_id: int | None,
        placement_gpu_id: int | None = None,
        same_process_targets: set[str] | None,
        gpu_stage_names: set[str] | None,
        stage_gpu_ids: dict[str, tuple[int, ...]] | None = None,
        remote_stage_names: set[str] | None = None,
        comm_config: Mapping[str, int | float | str | None] | None = None,
        injected_relay: Relay | None = None,
    ) -> None:
        self.stage_name = stage_name
        self.gpu_id = gpu_id
        self.placement_gpu_id = gpu_id if placement_gpu_id is None else placement_gpu_id
        self.same_process_targets = set(same_process_targets or ())
        self.gpu_stage_names = set(gpu_stage_names or ())
        self.stage_gpu_ids = {
            name: tuple(int(gpu_id) for gpu_id in gpu_ids)
            for name, gpu_ids in (stage_gpu_ids or {}).items()
        }
        self.remote_stage_names = set(remote_stage_names or ())
        self.direct_cuda_ipc_targets = frozenset(
            name
            for name, gpu_ids in self.stage_gpu_ids.items()
            if self.gpu_id == self.placement_gpu_id
            and gpu_ids == (self.placement_gpu_id,)
            and name not in self.same_process_targets
            and name not in self.remote_stage_names
        )
        self.comm_config = dict(comm_config or {})
        communication_config = self.comm_config
        self.level_zero_ipc_slot_size_kb = int(
            communication_config["level_zero_ipc_slot_size_kb"]
            if "level_zero_ipc_slot_size_kb" in communication_config
            else LEVEL_ZERO_IPC_SLOT_SIZE_KB
        )
        self.level_zero_ipc_pool_size_mb = int(
            communication_config["level_zero_ipc_pool_size_mb"]
            if "level_zero_ipc_pool_size_mb" in communication_config
            else LEVEL_ZERO_IPC_POOL_SIZE_MB
        )
        self.level_zero_ipc_min_transfer_kb = int(
            communication_config["level_zero_ipc_min_transfer_kb"]
            if "level_zero_ipc_min_transfer_kb" in communication_config
            else LEVEL_ZERO_IPC_MIN_TRANSFER_KB
        )
        # A direct caller skips CommConfig, and a zero slot would divide by zero.
        check_level_zero_ipc_sizes(
            slot_size_kb=self.level_zero_ipc_slot_size_kb,
            pool_size_mb=self.level_zero_ipc_pool_size_mb,
            min_transfer_kb=self.level_zero_ipc_min_transfer_kb,
        )
        self.injected_relay = injected_relay
        self.relays: dict[TransportKind, Relay] = {}
        self.traced_transports: dict[tuple[str, str], str] = {}
        self.device_ipc_peer_cache: dict[str, bool] = {}

    def device_ipc_peer_available(
        self, target: str, device_kind: Literal["cuda", "xpu"]
    ) -> bool:
        """Return whether a GPU edge can use device-IPC peer copies."""
        cached = self.device_ipc_peer_cache.get(target)
        if cached is not None:
            return cached
        else:
            pass
        if target not in self.stage_gpu_ids:
            return True
        else:
            pass
        target_gpu_ids = self.stage_gpu_ids[target]
        if not target_gpu_ids or self.placement_gpu_id is None:
            return True
        else:
            pass

        source_gpu = int(self.placement_gpu_id)
        target_gpu = int(target_gpu_ids[0])
        if source_gpu == target_gpu:
            self.device_ipc_peer_cache[target] = True
            return True
        else:
            pass

        if self.gpu_id is None or int(self.gpu_id) != source_gpu:
            self.warn_device_ipc_fallback(
                target,
                source_gpu,
                target_gpu,
                device_kind,
                "source process uses a remapped GPU",
            )
            self.device_ipc_peer_cache[target] = False
            return False
        else:
            pass
        try:
            source_local = int(self.gpu_id)
            target_local = target_gpu
            if device_kind == "cuda":
                device_count = torch.cuda.device_count()
            else:
                device_count = torch.xpu.device_count()
            if source_local >= device_count or target_local >= device_count:
                raise RuntimeError(
                    f"GPU is outside this process's visible {device_kind} range"
                )
            else:
                pass
            if device_kind == "cuda":
                available = bool(
                    torch.cuda.can_device_access_peer(target_local, source_local)
                )
            else:
                available = bool(
                    torch.xpu.can_device_access_peer(target_local, source_local)
                )
        except Exception as exc:
            self.warn_device_ipc_fallback(
                target, source_gpu, target_gpu, device_kind, f"peer query failed: {exc}"
            )
            self.device_ipc_peer_cache[target] = False
            return False
        if not available:
            self.warn_device_ipc_fallback(
                target,
                source_gpu,
                target_gpu,
                device_kind,
                "peer access is unsupported",
            )
        else:
            pass
        self.device_ipc_peer_cache[target] = available
        return available

    def warn_device_ipc_fallback(
        self,
        target: str,
        source_gpu: int,
        target_gpu: int,
        device_kind: Literal["cuda", "xpu"],
        reason: str,
    ) -> None:
        key = (self.stage_name, target, source_gpu, target_gpu, reason)
        if key in _CUDA_IPC_FALLBACK_WARNED:
            return
        else:
            pass
        _CUDA_IPC_FALLBACK_WARNED.add(key)
        logger.warning(
            f"CommRouter: using SHM for {device_kind} edge "
            f"{self.stage_name}(gpu={source_gpu}) -> {target}(gpu={target_gpu}): "
            f"{reason}"
        )

    @property
    def self_is_gpu(self) -> bool:
        return self.gpu_id is not None

    def is_local_object(self, target: str) -> bool:
        return target in self.same_process_targets

    def can_use_direct_cuda_ipc(self, target: str) -> bool:
        return (
            target in self.direct_cuda_ipc_targets
            and current_platform.get_intra_node_transport() == TransportKind.CUDA_IPC
        )

    def intra_node_transport(self, target: str) -> TransportKind:
        # The peer probe is vendor-specific, so it waits for the platform's choice.
        transport = current_platform.get_intra_node_transport()
        if transport is TransportKind.CUDA_IPC and not self.device_ipc_peer_available(
            target, "cuda"
        ):
            return TransportKind.SHM
        elif (
            transport is TransportKind.LEVEL_ZERO_IPC
            and not self.device_ipc_peer_available(target, "xpu")
        ):
            return TransportKind.SHM
        else:
            pass
        return transport

    def level_zero_ipc_carries(self, num_bytes: int) -> bool:
        """Whether one buffer this size goes over Level-Zero IPC rather than shm."""
        slot_bytes = self.level_zero_ipc_slot_size_kb * 1024
        slot_count = self.level_zero_ipc_pool_size_mb * 1024 * 1024 // slot_bytes
        return (
            num_bytes >= self.level_zero_ipc_min_transfer_kb * 1024
            and slots_for_size(num_bytes, slot_bytes) <= slot_count
        )

    @staticmethod
    def edge_transport(kind: TransportKind) -> TransportKind:
        """Route edge-only callers such as paged KV to shm instead of Level-Zero."""
        if kind is TransportKind.LEVEL_ZERO_IPC:
            return TransportKind.SHM
        else:
            return kind

    def note_transport_choice(
        self,
        direction: str,
        target: str,
        transport: str,
    ) -> None:
        """Record the transport an edge uses, the first time and on every change.

        The routing methods run once per transfer, so emitting on every call
        would bury the log. An edge that changes transport mid-run still reads as
        healthy in every other counter, so record the change itself.

        Takes a plain string because the same-GPU direct path names a transport
        (``torch_cuda_ipc``) that TransportKind does not carry.
        """
        if not _comm_trace_enabled():
            return
        else:
            pass
        key = (direction, target)
        previous = self.traced_transports.get(key)
        if previous == transport:
            return
        else:
            pass
        self.traced_transports[key] = transport
        _comm_trace(
            "comm_transport_selected",
            stage=self.stage_name,
            direction=direction,
            peer_stage=target,
            transport=transport,
            previous=previous,
        )

    def note_transport(
        self,
        direction: str,
        target: str,
        kind: TransportKind,
    ) -> TransportKind:
        self.note_transport_choice(direction, target, kind.value)
        return kind

    def outbound(self, target: str) -> TransportKind:
        if target in self.same_process_targets:
            kind = TransportKind.LOCAL_OBJECT
        else:
            kind = self.physical_outbound(target)
        return self.note_transport("outbound", target, kind)

    def physical_outbound(self, target: str) -> TransportKind:
        if target in self.remote_stage_names:
            return TransportKind.MOONCAKE
        else:
            pass
        if self.self_is_gpu and target in self.gpu_stage_names:
            return self.edge_transport(self.intra_node_transport(target))
        else:
            pass
        return TransportKind.SHM

    def outbound_stream(
        self,
        target: str,
        data: torch.Tensor,
        metadata: dict[str, object] | None = None,
    ) -> TransportKind:
        if not isinstance(data, torch.Tensor):
            raise TypeError(
                "relay-backed stream chunks must be torch.Tensor, got "
                f"{type(data).__name__}"
            )
        else:
            pass
        if target in self.remote_stage_names:
            kind = TransportKind.MOONCAKE
        elif data.device.type != current_platform.device_type:
            kind = TransportKind.SHM
        elif self.self_is_gpu and target in self.gpu_stage_names:
            kind = self.intra_node_transport(target)
            # Metadata tensors take slots of their own; two such streams can deadlock.
            if kind is TransportKind.LEVEL_ZERO_IPC and (
                (metadata is not None and extract_tensors(metadata)[1])
                or not self.level_zero_ipc_carries(data.numel() * data.element_size())
            ):
                kind = TransportKind.SHM
            else:
                pass
        else:
            raise ValueError(
                f"{current_platform.device_type} stream chunk cannot be sent from "
                f"{self.stage_name!r} to non-GPU target {target!r}"
            )
        return self.note_transport("stream", target, kind)

    def inbound(self, from_stage: str) -> TransportKind:
        if from_stage in self.remote_stage_names:
            kind = TransportKind.MOONCAKE
        elif self.self_is_gpu and from_stage in self.gpu_stage_names:
            kind = self.edge_transport(current_platform.get_intra_node_transport())
        else:
            kind = TransportKind.SHM
        return self.note_transport("inbound", from_stage, kind)

    def relay(self, kind: TransportKind) -> Relay:
        if kind is TransportKind.LOCAL_OBJECT:
            raise ValueError("local_object has no relay")
        else:
            pass
        if self.injected_relay is not None:
            return self.injected_relay
        else:
            pass
        relay = self.relays.get(kind)
        if relay is None:
            relay = self.build_relay(kind)
            self.relays[kind] = relay
        else:
            pass
        return relay

    def relay_for(self, target: str) -> tuple[TransportKind, Relay]:
        kind = self.outbound(target)
        if kind is TransportKind.LOCAL_OBJECT:
            raise ValueError(
                f"same-process target {target!r} has no relay transport; "
                "use local-object dispatch"
            )
        else:
            pass
        return kind, self.relay(kind)

    def relay_for_payload(
        self, target: str, payload: StagePayload
    ) -> tuple[TransportKind, Relay]:
        kind = self.outbound_payload(target, payload)
        if kind is TransportKind.LOCAL_OBJECT:
            raise ValueError("local_object has no relay")
        else:
            pass
        return kind, self.relay(kind)

    def outbound_payload(self, target: str, payload: object) -> TransportKind:
        if target in self.remote_stage_names:
            kind = TransportKind.MOONCAKE
        else:
            devices = tensor_devices(getattr(payload, "data", payload))
            if not devices or devices == {"cpu"}:
                kind = TransportKind.SHM
            elif current_platform.device_type in devices and devices <= {
                "cpu",
                current_platform.device_type,
            }:
                if self.self_is_gpu and target in self.gpu_stage_names:
                    kind = self.intra_node_transport(target)
                    if (
                        kind is TransportKind.LEVEL_ZERO_IPC
                        and not self.level_zero_ipc_carries(
                            packed_size_bytes(
                                extract_tensors(getattr(payload, "data", payload))[1]
                            )
                        )
                    ):
                        kind = TransportKind.SHM
                    else:
                        pass
                else:
                    kind = TransportKind.SHM
            else:
                raise ValueError(
                    f"mixed or unsupported tensor devices in payload: {devices}"
                )
        return self.note_transport("payload", target, kind)

    def relay_for_stream(
        self,
        target: str,
        data: torch.Tensor,
        metadata: dict[str, object] | None = None,
    ) -> tuple[TransportKind, Relay]:
        kind = self.outbound_stream(target, data, metadata)
        if kind is TransportKind.LOCAL_OBJECT:
            raise ValueError(
                f"same-process stream target {target!r} has no relay transport; "
                "use local-object dispatch"
            )
        else:
            pass
        return kind, self.relay(kind)

    def inbound_relay(self, from_stage: str) -> Relay:
        return self.relay(self.inbound(from_stage))

    def build_relay(self, kind: TransportKind) -> Relay:
        cfg = self.comm_config
        engine_id = (
            cfg["worker_id"] if "worker_id" in cfg else f"{self.stage_name}_relay"
        )
        slot_size_mb = cfg["slot_size_mb"] if "slot_size_mb" in cfg else 512
        credits = cfg["credits"] if "credits" in cfg else 2
        cuda_ipc_slot_size_kb = (
            cfg["cuda_ipc_slot_size_kb"] if "cuda_ipc_slot_size_kb" in cfg else 64
        )
        cuda_ipc_pool_size_mb = (
            cfg["cuda_ipc_pool_size_mb"] if "cuda_ipc_pool_size_mb" in cfg else None
        )
        if kind is TransportKind.LEVEL_ZERO_IPC:
            if self.gpu_id is None:
                raise ValueError(
                    f"level_zero_ipc relay requested for non-GPU stage "
                    f"{self.stage_name!r}"
                )
            else:
                pass
            return create_relay(
                "level_zero_ipc",
                engine_id=engine_id,
                device=f"xpu:{self.gpu_id}",
                slot_size_kb=self.level_zero_ipc_slot_size_kb,
                pool_size_mb=self.level_zero_ipc_pool_size_mb,
            )
        else:
            pass
        if kind is TransportKind.CUDA_IPC:
            if self.gpu_id is None:
                raise ValueError(
                    f"cuda_ipc relay requested for non-GPU stage {self.stage_name!r}"
                )
            else:
                pass
            return create_relay(
                "cuda_ipc",
                engine_id=engine_id,
                device=f"cuda:{self.gpu_id}",
                slot_size_mb=slot_size_mb,
                credits=credits,
                slot_size_kb=cuda_ipc_slot_size_kb,
                pool_size_mb=cuda_ipc_pool_size_mb,
            )
        else:
            pass
        if kind is TransportKind.MOONCAKE:
            if self.gpu_id is not None and not current_platform.is_cuda_alike():
                raise NotImplementedError(
                    f"cross-node transfer from {self.stage_name!r} needs a mooncake "
                    f"relay, which is built on a literal cuda device; "
                    f"{current_platform.device_type} is not supported yet"
                )
            else:
                pass
            device = f"cuda:{self.gpu_id}" if self.gpu_id is not None else "cpu"
            return create_relay(
                "mooncake",
                engine_id=engine_id,
                device=device,
                slot_size_mb=slot_size_mb,
                credits=credits,
                protocol=(
                    cfg["mooncake_protocol"] if "mooncake_protocol" in cfg else "rdma"
                ),
                hostname=(
                    cfg["mooncake_hostname"] if "mooncake_hostname" in cfg else None
                ),
                device_name=(
                    cfg["mooncake_device_name"] if "mooncake_device_name" in cfg else ""
                ),
            )
        else:
            pass
        if kind is TransportKind.SHM:
            return create_relay(
                "shm",
                engine_id=engine_id,
                device="cpu",
                slot_size_mb=slot_size_mb,
                credits=credits,
            )
        else:
            pass
        raise ValueError(f"CommRouter cannot build a relay for {kind}")

    def cleanup(self, request_id: str) -> None:
        for relay in self.active_relays():
            with suppress(Exception):
                relay.cleanup(request_id)

    def close(self) -> None:
        for relay in self.active_relays():
            with suppress(Exception):
                relay.close()
        self.relays.clear()

    def active_relays(self) -> list[Relay]:
        if self.injected_relay is not None:
            return [self.injected_relay]
        else:
            pass
        return list(self.relays.values())


def tensor_devices(obj: object, seen: set[int] | None = None) -> set[str]:
    if obj is None:
        return set()
    else:
        pass
    seen = set() if seen is None else seen
    obj_id = id(obj)
    if obj_id in seen:
        return set()
    else:
        pass
    seen.add(obj_id)
    if isinstance(obj, torch.Tensor):
        if obj.device.type in ("cpu", "musa"):
            return {obj.device.type}
        else:
            pass
        if obj.is_cuda:
            return {"cuda"}
        else:
            pass
        return {obj.device.type}
    else:
        pass
    if isinstance(obj, dict):
        devices: set[str] = set()
        for value in obj.values():
            devices.update(tensor_devices(value, seen))
        return devices
    else:
        pass
    if isinstance(obj, (list, tuple, set, frozenset)):
        devices = set()
        for value in obj:
            devices.update(tensor_devices(value, seen))
        return devices
    else:
        pass
    return set()
