# SPDX-License-Identifier: Apache-2.0
"""Stream native SenseNova layer weights from CPU during inference."""

from __future__ import annotations

import logging
import threading
from _thread import LockType
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Literal, Protocol

import torch
from torch import nn
from torch.utils.hooks import RemovableHandle

logger = logging.getLogger(__name__)


class LayerPreHook(Protocol):
    def __call__(self, module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        """Prepare layer weights before the module executes."""


class LayerPostHook(Protocol):
    def __call__(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | None,
    ) -> None:
        """Release streamed layer weights after the module executes."""


@dataclass(frozen=True)
class LayerwiseOffloadConfig:
    prefetch_size: int
    resident_layers: int
    residency_policy: Literal["leading", "strided"]
    pin_cpu_memory: bool

    def __post_init__(self) -> None:
        for name in ("prefetch_size", "resident_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
            else:
                pass
        if self.residency_policy not in ("leading", "strided"):
            raise ValueError("residency_policy must be 'leading' or 'strided'")
        else:
            pass

    def resident_indices(self, num_layers: int) -> frozenset[int]:
        if self.resident_layers > num_layers:
            raise ValueError("resident_layers cannot exceed the model's layer count")
        else:
            pass
        if self.residency_policy == "leading":
            return frozenset(range(self.resident_layers))
        else:
            pass
        # note (Wes): Residency is the complement of evenly spaced streamed layers.
        streamed_count = num_layers - self.resident_layers
        streamed = {
            round(index * num_layers / streamed_count)
            for index in range(streamed_count)
        }
        return frozenset(range(num_layers)) - streamed


@dataclass
class LayerWeight:
    module: nn.Module
    name: str
    is_parameter: bool
    host_tensor: torch.Tensor

    def get(self) -> torch.Tensor:
        if self.is_parameter:
            return self.module._parameters[  # noqa: leading-underscore -- PyTorch registered tensor storage
                self.name
            ].data
        else:
            pass
        return self.module._buffers[  # noqa: leading-underscore -- PyTorch registered tensor storage
            self.name
        ]

    def set(self, tensor: torch.Tensor) -> None:
        if self.is_parameter:
            self.module._parameters[
                self.name
            ].data = (  # noqa: leading-underscore -- PyTorch registered tensor storage
                tensor
            )
        else:
            self.module._buffers[self.name] = (
                tensor  # noqa: leading-underscore -- PyTorch registered tensor storage
            )


class SenseNovaLayerwiseOffload:
    """Serialize generation within request() while streaming layer weights."""

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: LayerwiseOffloadConfig,
        *,
        layers_path: str = "language_model.model.layers",
    ) -> None:
        self.device: torch.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("SenseNova layerwise offload supports CUDA only")
        else:
            pass
        if model.training:
            raise ValueError("SenseNova layerwise offload requires an eval model")
        else:
            pass
        self.config: LayerwiseOffloadConfig = config
        layers = model.get_submodule(layers_path)
        if not isinstance(layers, nn.ModuleList) or not len(layers):
            raise ValueError(f"{layers_path} must be a non-empty ModuleList")
        else:
            pass
        self.layers: nn.ModuleList = layers
        self.residents: frozenset[int] = config.resident_indices(len(self.layers))
        self.lock: LockType = threading.Lock()
        self.is_active: bool = False
        self.is_closed: bool = False
        self.loaded: set[int] = set()
        self.ready: dict[int, torch.cuda.Event] = {}
        self.hooks: list[RemovableHandle] = []
        self.weights: list[list[LayerWeight]] = []
        block_modules = {
            id(module) for layer in self.layers for module in layer.modules()
        }
        owners: dict[int, int | None] = {}
        # note (Wes): Shared weights cannot move independently between devices.
        for index, layer in enumerate(self.layers):
            weights = []
            for module in layer.modules():
                for is_parameter, values in (
                    (
                        True,
                        module._parameters,  # noqa: leading-underscore -- PyTorch registered tensor storage
                    ),
                    (
                        False,
                        module._buffers,  # noqa: leading-underscore -- PyTorch registered tensor storage
                    ),
                ):
                    for name, tensor in values.items():
                        if tensor is None:
                            continue
                        else:
                            pass
                        if tensor.device.type != "cpu":
                            raise ValueError(
                                "Load the model on CPU before enabling offload"
                            )
                        else:
                            pass
                        if id(tensor) in owners and owners[id(tensor)] != index:
                            raise ValueError("Offloaded blocks cannot share weights")
                        else:
                            pass
                        owners[id(tensor)] = index
                        weights.append(
                            LayerWeight(module, name, is_parameter, tensor.detach())
                        )
            self.weights.append(weights)
        for module in model.modules():
            if id(module) in block_modules:
                continue
            else:
                pass
            if any(
                tensor is not None and id(tensor) in owners
                for tensor in (
                    *module._parameters.values(),  # noqa: leading-underscore -- PyTorch registered tensor storage
                    *module._buffers.values(),  # noqa: leading-underscore -- PyTorch registered tensor storage
                )
            ):
                raise ValueError("Offloaded blocks cannot share non-block weights")
            else:
                pass

        self.copy_stream: torch.cuda.Stream | None = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        if config.pin_cpu_memory and self.device.type == "cuda":
            for weights in self.weights:
                # note (Wes): Pinning needs an additional host copy during initialization.
                pinned: dict[
                    tuple[int, torch.Size, tuple[int, ...], torch.dtype], torch.Tensor
                ] = {}
                for weight in weights:
                    key = (
                        weight.host_tensor.data_ptr(),
                        weight.host_tensor.shape,
                        weight.host_tensor.stride(),
                        weight.host_tensor.dtype,
                    )
                    if key not in pinned:
                        pinned[key] = weight.host_tensor.pin_memory()
                    else:
                        pass
                    weight.host_tensor = pinned[key]
                    weight.set(weight.host_tensor)
        else:
            pass

        # note (Wes): Detaching layers prevents moving all weights to the GPU at once.
        parent_path, name = layers_path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        setattr(parent, name, nn.ModuleList())
        try:
            model.to(self.device)
        finally:
            setattr(parent, name, self.layers)
        for index in sorted(self.residents):
            self.load_weights(index)
        if self.copy_stream is not None:
            self.copy_stream.synchronize()
        else:
            pass
        try:
            for index, layer in enumerate(self.layers):
                self.hooks.append(
                    layer.register_forward_pre_hook(self.before_forward(index))
                )
                self.hooks.append(
                    layer.register_forward_hook(
                        self.after_forward(index), always_call=True
                    )
                )
        except BaseException:
            self.close()
            raise
        logger.info(
            f"SenseNova layerwise offload ready: layers={len(self.layers)}, "
            f"resident={len(self.residents)}, prefetch={config.prefetch_size}, "
            f"policy={config.residency_policy}, pinned={config.pin_cpu_memory}"
        )

    def load_weights(self, index: int) -> None:
        if index in self.loaded:
            return
        else:
            pass
        # note (Wes): Track partial copies so failures release all allocated weights.
        self.loaded.add(index)
        try:
            if self.copy_stream is None:
                for weight in self.weights[index]:
                    weight.set(weight.host_tensor.to(self.device))
            else:
                with torch.cuda.stream(self.copy_stream):
                    for weight in self.weights[index]:
                        weight.set(
                            weight.host_tensor.to(self.device, non_blocking=True)
                        )
                    event = torch.cuda.Event()
                    event.record(self.copy_stream)
                    self.ready[index] = event
        except BaseException:
            self.release_weights(index)
            raise

    def release_weights(self, index: int) -> None:
        if index not in self.loaded:
            return
        else:
            pass
        stream = (
            torch.cuda.current_stream(self.device)
            if self.copy_stream is not None
            else None
        )
        for weight in self.weights[index]:
            if stream is not None and weight.get().device.type == "cuda":
                # note (Wes): Pending kernels must finish before storage is recycled.
                weight.get().record_stream(stream)
            else:
                pass
            weight.set(weight.host_tensor)
        self.loaded.remove(index)
        self.ready.pop(index, None)

    def before_forward(self, index: int) -> LayerPreHook:
        # note (Wes): PyTorch requires module and input arguments even when unused.
        def hook(module: nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
            if not self.is_active:
                raise RuntimeError("Run offloaded generation inside manager.request()")
            else:
                pass
            self.load_weights(index)
            if self.copy_stream is not None:
                torch.cuda.current_stream(self.device).wait_event(self.ready[index])
            else:
                pass
            upcoming = (
                i for i in range(index + 1, len(self.layers)) if i not in self.residents
            )
            for _, next_index in zip(range(self.config.prefetch_size), upcoming):
                self.load_weights(next_index)

        return hook

    def after_forward(self, index: int) -> LayerPostHook:
        # note (Wes): PyTorch supplies None when an always-called hook follows failure.
        def hook(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor | None,
        ) -> None:
            if index not in self.residents:
                self.release_weights(index)
            else:
                pass

        return hook

    @contextmanager
    def request(self) -> Iterator[None]:
        if not self.lock.acquire(blocking=False):
            raise RuntimeError(
                "Concurrent use of an offloaded SenseNova model is unsupported"
            )
        else:
            pass
        if self.is_closed:
            self.lock.release()
            raise RuntimeError("The SenseNova offload manager is closed")
        else:
            pass
        self.is_active = True
        try:
            with torch.inference_mode():
                yield
        finally:
            try:
                # note (Wes): Failed requests may leave unused prefetched copies.
                if self.copy_stream is not None:
                    self.copy_stream.synchronize()
                else:
                    pass
                for index in sorted(self.loaded - self.residents):
                    self.release_weights(index)
            finally:
                self.is_active = False
                self.lock.release()

    def close(self) -> None:
        """Remove hooks and return all block weights to CPU after use ends."""
        if not self.lock.acquire(blocking=False):
            raise RuntimeError("Cannot close an active offload manager")
        else:
            pass
        try:
            if self.copy_stream is not None:
                self.copy_stream.synchronize()
            else:
                pass
            for handle in self.hooks:
                handle.remove()
            self.hooks.clear()
            for index in sorted(self.loaded):
                self.release_weights(index)
            self.is_closed = True
        finally:
            self.lock.release()
