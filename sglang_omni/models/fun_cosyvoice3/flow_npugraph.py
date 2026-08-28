# SPDX-License-Identifier: Apache-2.0
"""Ascend NPUGraph replay for the buffered CosyVoice3 Flow estimator."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
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


class FlowDiTNPUGraphRunner:
    """Reusable NPU Graph runner for a fixed-input Flow DiT estimator.

    ``prepare_inputs`` may pad inputs to a bucket and ``restore_output`` may
    crop the estimator result back to the request shape. The runner owns the
    bounded LRU graph cache; model-specific tensor conventions stay in the
    adapter callbacks.
    """

    def __init__(
        self,
        estimator: torch.nn.Module,
        *,
        input_names: tuple[str, ...],
        max_graphs: int = 8,
        bucket_sizes: tuple[int, ...] = (),
        warmup_buckets: tuple[int, ...] = (),
        prepare_inputs: Callable[[dict[str, torch.Tensor], int], dict[str, torch.Tensor]] | None = None,
        restore_output: Callable[[torch.Tensor, dict[str, torch.Tensor]], torch.Tensor] | None = None,
        streaming_kwarg: str | None = None,
    ) -> None:
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        if any(size < 1 for size in bucket_sizes + warmup_buckets):
            raise ValueError("Graph bucket sizes must be positive")
        if tuple(sorted(set(bucket_sizes))) != bucket_sizes:
            raise ValueError("bucket_sizes must be sorted and unique")
        self._estimator = estimator
        self._eager_forward: Callable[..., torch.Tensor] = estimator.forward
        self._input_names = input_names
        self._max_graphs = max_graphs
        self._bucket_sizes = bucket_sizes
        self._warmup_buckets = tuple(sorted(set(warmup_buckets)))
        self._prepare_inputs = prepare_inputs or (lambda inputs, _target: inputs)
        self._restore_output = restore_output or (lambda output, _inputs: output)
        self._streaming_kwarg = streaming_kwarg
        self._graphs: OrderedDict[tuple[Any, ...], _GraphEntry] = OrderedDict()
        self._warmed = False

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

    def _bucket_for(self, inputs: dict[str, torch.Tensor]) -> int | None:
        if not self._bucket_sizes:
            return None
        length = inputs[self._input_names[0]].shape[-1]
        return next((size for size in self._bucket_sizes if size >= length), None)

    def _prepare(self, inputs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        bucket = self._bucket_for(inputs)
        if self._bucket_sizes and bucket is None:
            return None
        return self._prepare_inputs(inputs, bucket) if bucket is not None else inputs

    def __call__(self, *args: torch.Tensor, streaming: bool = False, **kwargs: torch.Tensor) -> torch.Tensor:
        if args and kwargs:
            raise TypeError("Flow DiT runner accepts positional or keyword inputs, not both")
        if kwargs:
            try:
                values = tuple(kwargs[name] for name in self._input_names)
            except KeyError as exc:
                raise TypeError(f"missing Flow DiT input {exc.args[0]!r}") from exc
            unexpected = set(kwargs) - set(self._input_names)
            if unexpected:
                raise TypeError(f"unexpected Flow DiT inputs: {sorted(unexpected)}")
        else:
            values = args
        if len(values) != len(self._input_names):
            raise TypeError(
                f"expected {len(self._input_names)} Flow DiT inputs, got {len(values)}"
            )
        if streaming:
            return self._call_eager(values, streaming=True)

        inputs = dict(zip(self._input_names, values))
        original_inputs = inputs
        inputs = self._prepare(inputs)
        if inputs is None:
            return self._call_eager(tuple(original_inputs.values()), streaming=False)
        if not self._warmed and self._warmup_buckets:
            self._prewarm(inputs)
            self._warmed = True
        signature = self._signature(inputs)
        entry = self._graphs.get(signature)
        if entry is None:
            if len(self._graphs) >= self._max_graphs:
                evicted_signature, _ = self._graphs.popitem(last=False)
                logger.info(
                    "Flow DiT NPUGraph LRU evicted signature=%s for new signature=%s",
                    evicted_signature,
                    signature,
                )
            entry = self._capture(inputs)
            self._graphs[signature] = entry
            return self._restore_output(entry.output, original_inputs)

        self._graphs.move_to_end(signature)
        for name in self._input_names:
            entry.inputs[name].copy_(inputs[name])
        current_stream = torch.npu.current_stream()
        entry.stream.wait_stream(current_stream)
        entry.graph.replay()
        current_stream.wait_stream(entry.stream)
        return self._restore_output(entry.output, original_inputs)

    def _call_eager(self, values: tuple[torch.Tensor, ...], *, streaming: bool) -> torch.Tensor:
        if self._streaming_kwarg is None:
            return self._eager_forward(*values)
        return self._eager_forward(*values, **{self._streaming_kwarg: streaming})

    def _prewarm(self, inputs: dict[str, torch.Tensor]) -> None:
        current_length = inputs[self._input_names[0]].shape[-1]
        for bucket in self._warmup_buckets:
            if bucket < current_length or len(self._graphs) >= self._max_graphs:
                continue
            warm_inputs = self._prepare_inputs(inputs, bucket)
            signature = self._signature(warm_inputs)
            if signature not in self._graphs:
                self._graphs[signature] = self._capture(warm_inputs)

    def _capture(self, inputs: dict[str, torch.Tensor]) -> _GraphEntry:
        started = time.perf_counter()
        for _ in range(2):
            self._call_eager(
                tuple(inputs[name] for name in self._input_names), streaming=False
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
                output = self._call_eager(
                    tuple(static_inputs[name] for name in self._input_names), streaming=False
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


class FlowNPUGraphRunner(FlowDiTNPUGraphRunner):
    """CosyVoice3 adapter for the reusable Flow DiT Graph runner."""

    def __init__(
        self,
        estimator: torch.nn.Module,
        *,
        max_graphs: int = 8,
        bucket_sizes: tuple[int, ...] = (),
        warmup_buckets: tuple[int, ...] = (),
    ) -> None:
        super().__init__(
            estimator,
            input_names=_INPUT_NAMES,
            max_graphs=max_graphs,
            bucket_sizes=bucket_sizes,
            warmup_buckets=warmup_buckets,
            prepare_inputs=_prepare_cosyvoice_inputs,
            restore_output=_restore_cosyvoice_output,
            streaming_kwarg="streaming",
        )


def _prepare_cosyvoice_inputs(
    inputs: dict[str, torch.Tensor], target_length: int
) -> dict[str, torch.Tensor]:
    current_length = inputs["x"].shape[-1]
    if target_length == current_length:
        return inputs
    padding = target_length - current_length
    padded = dict(inputs)
    for name in ("x", "mu", "cond"):
        padded[name] = torch.nn.functional.pad(inputs[name], (0, padding))
    padded["mask"] = torch.nn.functional.pad(inputs["mask"], (0, padding), value=False)
    return padded


def _restore_cosyvoice_output(
    output: torch.Tensor, original_inputs: dict[str, torch.Tensor]
) -> torch.Tensor:
    return output[..., : original_inputs["x"].shape[-1]]


def enable_flow_npugraph(
    flow: Any,
    *,
    max_graphs: int = 8,
    bucket_sizes: tuple[int, ...] = (),
    warmup_buckets: tuple[int, ...] = (),
) -> bool:
    from sglang_omni.platforms import current_platform

    if current_platform.device_type != "npu":
        return False
    estimator = getattr(getattr(flow, "decoder", None), "estimator", None)
    if not isinstance(estimator, torch.nn.Module):
        return False
    estimator.forward = FlowNPUGraphRunner(
        estimator,
        max_graphs=max_graphs,
        bucket_sizes=bucket_sizes,
        warmup_buckets=warmup_buckets,
    )
    logger.info(
        "Enabled CosyVoice3 Flow NPUGraph with max_graphs=%s buckets=%s warmup=%s",
        max_graphs,
        bucket_sizes,
        warmup_buckets,
    )
    return True


def prepare_flow_npugraph_environment() -> bool:
    from sglang_omni.platforms import current_platform

    if current_platform.device_type != "npu":
        return False
    import torch_npu

    torch_npu.npu.config.allow_internal_format = False
    return True


__all__ = [
    "FlowDiTNPUGraphRunner",
    "FlowNPUGraphRunner",
    "enable_flow_npugraph",
    "prepare_flow_npugraph_environment",
]
