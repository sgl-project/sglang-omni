# SPDX-License-Identifier: Apache-2.0
"""Inference-only layerwise weight streaming for the native SenseNova stage.

CPU weights remain authoritative. CUDA copies use a separate stream, while
forward hooks load each block and release its device weights after execution.
Non-block weights and the selected resident blocks stay on the compute device.
"""

from __future__ import annotations

import logging
import threading
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LayerwiseOffloadConfig:
    prefetch_size: int = 1
    resident_layers: int = 0
    residency_policy: str = "leading"
    pin_cpu_memory: bool = False

    def __post_init__(self):
        for name in ("prefetch_size", "resident_layers"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.residency_policy not in ("leading", "strided"):
            raise ValueError("residency_policy must be 'leading' or 'strided'")

    def resident_indices(self, num_layers: int) -> frozenset[int]:
        if self.resident_layers > num_layers:
            raise ValueError("resident_layers cannot exceed the model's layer count")
        if self.residency_policy == "leading":
            return frozenset(range(self.resident_layers))
        # Match SGLang's policy: distribute streamed blocks evenly, then keep
        # their complement resident (not an independent ramp of residents).
        streamed_count = num_layers - self.resident_layers
        streamed = {
            round(index * num_layers / streamed_count)
            for index in range(streamed_count)
        }
        return frozenset(range(num_layers)) - streamed


@dataclass
class _Weight:
    module: nn.Module
    name: str
    parameter: bool
    host: torch.Tensor

    def get(self) -> torch.Tensor:
        if self.parameter:
            return self.module._parameters[self.name].data
        return self.module._buffers[self.name]

    def set(self, tensor: torch.Tensor) -> None:
        if self.parameter:
            self.module._parameters[self.name].data = tensor
        else:
            self.module._buffers[self.name] = tensor


class SenseNovaLayerwiseOffload:
    """Stream a CPU-loaded block list; serialize use of the mutable weights.

    CUDA is the serving backend. A synchronous CPU path also permits small
    numerical/lifecycle tests without requiring the model checkpoint or a GPU.
    Call ``request()`` around the entire generation, not around each image or
    denoising step. Repeated CFG/prefix/denoise forwards use the same hooks.
    """

    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        config: LayerwiseOffloadConfig,
        *,
        layers_path: str = "language_model.model.layers",
    ):
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("SenseNova layerwise offload supports CUDA only")
        if model.training:
            raise ValueError("SenseNova layerwise offload requires an eval model")
        self.config = config
        self.layers = model.get_submodule(layers_path)
        if not isinstance(self.layers, nn.ModuleList) or not len(self.layers):
            raise ValueError(f"{layers_path} must be a non-empty ModuleList")
        self.residents = config.resident_indices(len(self.layers))
        self._lock = threading.Lock()
        self._active = False
        self._closed = False
        self._loaded: set[int] = set()
        self._ready: dict[int, torch.cuda.Event] = {}
        self._hooks = []
        self._weights: list[list[_Weight]] = []
        block_modules = {
            id(module) for layer in self.layers for module in layer.modules()
        }
        owners: dict[int, int | None] = {}
        # Cross-block or block/non-block ties would make independent placement
        # ambiguous. Reject them before changing any device placement.
        for index, layer in enumerate(self.layers):
            weights = []
            for module in layer.modules():
                for parameter, values in (
                    (True, module._parameters),
                    (False, module._buffers),
                ):
                    for name, tensor in values.items():
                        if tensor is None:
                            continue
                        if tensor.device.type != "cpu":
                            raise ValueError(
                                "Load the model on CPU before enabling offload"
                            )
                        if id(tensor) in owners and owners[id(tensor)] != index:
                            raise ValueError("Offloaded blocks cannot share weights")
                        owners[id(tensor)] = index
                        weights.append(
                            _Weight(module, name, parameter, tensor.detach())
                        )
            self._weights.append(weights)
        for module in model.modules():
            if id(module) in block_modules:
                continue
            if any(
                tensor is not None and id(tensor) in owners
                for tensor in (*module._parameters.values(), *module._buffers.values())
            ):
                raise ValueError("Offloaded blocks cannot share non-block weights")

        self._copy_stream = (
            torch.cuda.Stream(device=self.device)
            if self.device.type == "cuda"
            else None
        )
        if config.pin_cpu_memory and self.device.type == "cuda":
            for weights in self._weights:
                # Pinned memory is opt-in: it creates an additional host copy
                # during initialization and must fit the deployment's RAM budget.
                pinned: dict[tuple, torch.Tensor] = {}
                for weight in weights:
                    key = (
                        weight.host.data_ptr(),
                        weight.host.shape,
                        weight.host.stride(),
                        weight.host.dtype,
                    )
                    if key not in pinned:
                        pinned[key] = weight.host.pin_memory()
                    weight.host = pinned[key]
                    weight.set(weight.host)

        # Move only non-block modules. Moving the root with its block list
        # attached would transiently materialize the full model on the GPU.
        parent_path, name = layers_path.rsplit(".", 1)
        parent = model.get_submodule(parent_path)
        setattr(parent, name, nn.ModuleList())
        try:
            model.to(self.device)
        finally:
            setattr(parent, name, self.layers)
        for index in sorted(self.residents):
            self._load(index)
        if self._copy_stream is not None:
            self._copy_stream.synchronize()
        try:
            for index, layer in enumerate(self.layers):
                self._hooks.append(layer.register_forward_pre_hook(self._before(index)))
                self._hooks.append(
                    layer.register_forward_hook(self._after(index), always_call=True)
                )
        except BaseException:
            self.close()
            raise
        logger.info(
            "SenseNova layerwise offload ready: layers=%d, resident=%d, "
            "prefetch=%d, policy=%s, pinned=%s",
            len(self.layers),
            len(self.residents),
            config.prefetch_size,
            config.residency_policy,
            config.pin_cpu_memory,
        )

    def _load(self, index: int) -> None:
        if index in self._loaded:
            return
        # Track partial materialization too: an allocation/copy failure midway
        # through a block must not leave its earlier weights on the GPU.
        self._loaded.add(index)
        try:
            if self._copy_stream is None:
                for weight in self._weights[index]:
                    weight.set(weight.host.to(self.device))
            else:
                with torch.cuda.stream(self._copy_stream):
                    for weight in self._weights[index]:
                        weight.set(weight.host.to(self.device, non_blocking=True))
                    event = torch.cuda.Event()
                    event.record(self._copy_stream)
                    self._ready[index] = event
        except BaseException:
            self._release(index)
            raise

    def _release(self, index: int) -> None:
        if index not in self._loaded:
            return
        stream = (
            torch.cuda.current_stream(self.device)
            if self._copy_stream is not None
            else None
        )
        for weight in self._weights[index]:
            if stream is not None and weight.get().device.type == "cuda":
                # GPU kernels may still hold the old weights after the CPU hook
                # returns. Do not recycle their storage until compute completes.
                weight.get().record_stream(stream)
            weight.set(weight.host)
        self._loaded.remove(index)
        self._ready.pop(index, None)

    def _before(self, index: int):
        def hook(_module, _inputs):
            if not self._active:
                raise RuntimeError("Run offloaded generation inside manager.request()")
            self._load(index)
            if self._copy_stream is not None:
                torch.cuda.current_stream(self.device).wait_event(self._ready[index])
            upcoming = (
                i for i in range(index + 1, len(self.layers)) if i not in self.residents
            )
            for _, next_index in zip(range(self.config.prefetch_size), upcoming):
                self._load(next_index)

        return hook

    def _after(self, index: int):
        def hook(_module, _inputs, _output):
            if index not in self.residents:
                self._release(index)

        return hook

    @contextmanager
    def request(self):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError(
                "Concurrent use of an offloaded SenseNova model is unsupported"
            )
        if self._closed:
            self._lock.release()
            raise RuntimeError("The SenseNova offload manager is closed")
        self._active = True
        try:
            with torch.inference_mode():
                yield
        finally:
            try:
                # Also drains copies prefetched for blocks skipped after a failure.
                if self._copy_stream is not None:
                    self._copy_stream.synchronize()
                for index in sorted(self._loaded - self.residents):
                    self._release(index)
            finally:
                self._active = False
                self._lock.release()

    def close(self):
        """Remove hooks and return all block weights to CPU after use ends."""
        if not self._lock.acquire(blocking=False):
            raise RuntimeError("Cannot close an active offload manager")
        try:
            if self._copy_stream is not None:
                self._copy_stream.synchronize()
            for handle in self._hooks:
                handle.remove()
            self._hooks.clear()
            for index in sorted(self._loaded):
                self._release(index)
            self._closed = True
        finally:
            self._lock.release()
