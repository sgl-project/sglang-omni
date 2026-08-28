# SPDX-License-Identifier: Apache-2.0
"""Ascend NPUGraph replay for the buffered CosyVoice3 Flow estimator."""

from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Callable

import torch

_INPUT_NAMES = ("x", "mask", "mu", "t", "spks", "cond")
logger = logging.getLogger(__name__)


def _graph_safe_non_streaming_mask(
    xs: torch.Tensor,
    masks: torch.Tensor,
    use_dynamic_chunk: bool,
    use_dynamic_left_chunk: bool,
    decoding_chunk_size: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
    enable_full_context: bool = True,
) -> torch.Tensor:
    del xs, use_dynamic_left_chunk, decoding_chunk_size
    del num_decoding_left_chunks, enable_full_context
    if use_dynamic_chunk or static_chunk_size != 0:
        raise RuntimeError("Flow NPUGraph supports only buffered non-streaming masks")
    masks = masks.bool()
    return torch.where(~masks.any(dim=-1, keepdim=True), True, masks)


@contextmanager
def _graph_safe_mask_scope():
    from cosyvoice.flow.DiT import dit as dit_module

    original = dit_module.add_optional_chunk_mask
    dit_module.add_optional_chunk_mask = _graph_safe_non_streaming_mask
    try:
        yield
    finally:
        dit_module.add_optional_chunk_mask = original


@dataclass
class _GraphEntry:
    inputs: dict[str, torch.Tensor]
    graph: Any
    output: torch.Tensor
    stream: Any


def _replay_after_capture(graph: Any, capture_stream: Any) -> None:
    current_stream = torch.npu.current_stream()
    current_stream.wait_stream(capture_stream)
    graph.replay()
    current_stream.wait_stream(capture_stream)
    torch.npu.synchronize()


class FlowNPUGraphRunner:
    """Capture one estimator graph per static input signature."""

    def __init__(self, estimator: torch.nn.Module, *, max_graphs: int = 8) -> None:
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        self._estimator = estimator
        self._eager_forward: Callable[..., torch.Tensor] = estimator.forward
        self._max_graphs = max_graphs
        self._graphs: dict[tuple[Any, ...], _GraphEntry] = {}
        self._capacity_warning_logged = False

    @staticmethod
    def _signature(inputs: dict[str, torch.Tensor]) -> tuple[Any, ...]:
        return tuple(
            (
                name,
                tuple(value.shape),
                value.dtype,
                value.device.type,
                value.device.index,
            )
            for name, value in inputs.items()
        )

    def __call__(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        mu: torch.Tensor,
        t: torch.Tensor,
        spks: torch.Tensor,
        cond: torch.Tensor,
        streaming: bool = False,
    ) -> torch.Tensor:
        if streaming:
            return self._eager_forward(x, mask, mu, t, spks, cond, streaming=True)

        inputs = dict(zip(_INPUT_NAMES, (x, mask, mu, t, spks, cond), strict=True))
        signature = self._signature(inputs)
        entry = self._graphs.get(signature)
        if entry is None:
            if len(self._graphs) >= self._max_graphs:
                if not self._capacity_warning_logged:
                    logger.warning(
                        "CosyVoice3 Flow NPUGraph cache reached max_graphs=%s; "
                        "new signatures will use NPU eager execution",
                        self._max_graphs,
                    )
                    self._capacity_warning_logged = True
                return self._eager_forward(x, mask, mu, t, spks, cond, streaming=False)
            entry = self._capture(inputs)
            self._graphs[signature] = entry
            return entry.output

        for name in _INPUT_NAMES:
            entry.inputs[name].copy_(inputs[name])
        current_stream = torch.npu.current_stream()
        entry.stream.wait_stream(current_stream)
        entry.graph.replay()
        current_stream.wait_stream(entry.stream)
        return entry.output

    def _capture(self, inputs: dict[str, torch.Tensor]) -> _GraphEntry:
        started = time.perf_counter()
        for _ in range(2):
            self._eager_forward(
                *(inputs[name] for name in _INPUT_NAMES), streaming=False
            )
        torch.npu.synchronize()

        static_inputs = {name: value.clone() for name, value in inputs.items()}
        graph = torch.npu.NPUGraph()
        capture_stream = torch.npu.Stream()
        capture_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(capture_stream), _graph_safe_mask_scope():
            with torch.npu.graph(
                graph,
                pool=torch.npu.graph_pool_handle(),
                stream=capture_stream,
                auto_dispatch_capture=True,
            ):
                output = self._eager_forward(
                    *(static_inputs[name] for name in _INPUT_NAMES), streaming=False
                )
        _replay_after_capture(graph, capture_stream)
        logger.info(
            "Captured CosyVoice3 Flow NPUGraph signature=%s elapsed=%.3fs",
            self._signature(inputs),
            time.perf_counter() - started,
        )
        return _GraphEntry(
            inputs=static_inputs,
            graph=graph,
            output=output,
            stream=capture_stream,
        )


def enable_flow_npugraph(flow: Any, *, max_graphs: int = 8) -> bool:
    from sglang_omni.platforms import current_platform

    if current_platform.device_type != "npu":
        return False
    estimator = getattr(getattr(flow, "decoder", None), "estimator", None)
    if not isinstance(estimator, torch.nn.Module):
        return False
    estimator.forward = FlowNPUGraphRunner(estimator, max_graphs=max_graphs)
    logger.info("Enabled CosyVoice3 Flow NPUGraph with max_graphs=%s", max_graphs)
    return True


def prepare_flow_npugraph_environment() -> bool:
    from sglang_omni.platforms import current_platform

    if current_platform.device_type != "npu":
        return False
    import torch_npu

    torch_npu.npu.config.allow_internal_format = False
    return True


__all__ = [
    "FlowNPUGraphRunner",
    "enable_flow_npugraph",
    "prepare_flow_npugraph_environment",
]
