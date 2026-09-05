# SPDX-License-Identifier: Apache-2.0
"""Reusable Ascend NPU Graph execution for fixed-shape Flow DiT modules."""

from __future__ import annotations

import logging
import time
from collections import OrderedDict
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Callable, ContextManager

import torch

logger = logging.getLogger(__name__)

TensorInputs = dict[str, torch.Tensor]


@dataclass
class GraphEntry:
    inputs: TensorInputs
    graph: Any
    output: torch.Tensor
    stream: Any


def replay_after_capture(graph: Any, capture_stream: Any) -> None:
    current_stream = torch.npu.current_stream()
    current_stream.wait_stream(capture_stream)
    graph.replay()
    current_stream.wait_stream(capture_stream)
    torch.npu.synchronize()


class FlowDiTNPUGraphRunner:
    """Capture, bucket, prewarm, and LRU-cache Flow DiT estimator graphs.

    Model adapters own input semantics through callbacks. The generic runner
    only requires tensor inputs and a stable estimator output tensor.
    """

    def __init__(
        self,
        estimator: torch.nn.Module,
        *,
        input_names: tuple[str, ...],
        max_graphs: int = 8,
        bucket_sizes: tuple[int, ...] = (),
        warmup_buckets: tuple[int, ...] = (),
        get_bucket_length: Callable[[TensorInputs], int] | None = None,
        prepare_inputs: Callable[[TensorInputs, int], TensorInputs] | None = None,
        restore_output: Callable[[torch.Tensor, TensorInputs], torch.Tensor]
        | None = None,
        capture_context: Callable[[], ContextManager[Any]] | None = None,
    ) -> None:
        if max_graphs < 1:
            raise ValueError("max_graphs must be positive")
        if not input_names:
            raise ValueError("input_names must not be empty")
        bucket_sizes = tuple(int(size) for size in bucket_sizes)
        warmup_buckets = tuple(int(size) for size in warmup_buckets)
        if any(size < 1 for size in bucket_sizes + warmup_buckets):
            raise ValueError("Graph bucket sizes must be positive")
        if tuple(sorted(set(bucket_sizes))) != bucket_sizes:
            raise ValueError("bucket_sizes must be sorted and unique")
        unknown_warmups = set(warmup_buckets) - set(bucket_sizes)
        if unknown_warmups:
            raise ValueError("warmup_buckets must be present in bucket_sizes")

        self._eager_forward: Callable[..., torch.Tensor] = estimator.forward
        self._input_names = input_names
        self._max_graphs = max_graphs
        self._bucket_sizes = bucket_sizes
        self._warmup_buckets = tuple(sorted(set(warmup_buckets)))
        self._get_bucket_length = get_bucket_length or (
            lambda inputs: inputs[self._input_names[0]].shape[-1]
        )
        self._prepare_inputs = prepare_inputs or (lambda inputs, _target: inputs)
        self._restore_output = restore_output or (lambda output, _inputs: output)
        self._capture_context = capture_context or nullcontext
        self._graphs: OrderedDict[tuple[Any, ...], GraphEntry] = OrderedDict()
        self._warmed = False

    @staticmethod
    def _signature(inputs: TensorInputs) -> tuple[Any, ...]:
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

    def _bucket_for(self, inputs: TensorInputs) -> int | None:
        if not self._bucket_sizes:
            return None
        length = self._get_bucket_length(inputs)
        return next((size for size in self._bucket_sizes if size >= length), None)

    def _prepare(self, inputs: TensorInputs) -> TensorInputs | None:
        bucket = self._bucket_for(inputs)
        if self._bucket_sizes and bucket is None:
            return None
        return self._prepare_inputs(inputs, bucket) if bucket is not None else inputs

    def __call__(self, *args: torch.Tensor, **kwargs: torch.Tensor) -> torch.Tensor:
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

        original_inputs = dict(zip(self._input_names, values))
        inputs = self._prepare(original_inputs)
        if inputs is None:
            return self._eager_forward(*original_inputs.values())
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

    def _prewarm(self, inputs: TensorInputs) -> None:
        current_length = self._get_bucket_length(inputs)
        for bucket in self._warmup_buckets:
            if bucket < current_length or len(self._graphs) >= self._max_graphs:
                continue
            warm_inputs = self._prepare_inputs(inputs, bucket)
            signature = self._signature(warm_inputs)
            if signature not in self._graphs:
                self._graphs[signature] = self._capture(warm_inputs)

    def _capture(self, inputs: TensorInputs) -> GraphEntry:
        started = time.perf_counter()
        for _ in range(2):
            self._eager_forward(*(inputs[name] for name in self._input_names))
        torch.npu.synchronize()

        static_inputs = {name: value.clone() for name, value in inputs.items()}
        graph = torch.npu.NPUGraph()
        capture_stream = torch.npu.Stream()
        capture_stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(capture_stream), self._capture_context():
            with torch.npu.graph(
                graph,
                pool=torch.npu.graph_pool_handle(),
                stream=capture_stream,
                auto_dispatch_capture=True,
            ):
                output = self._eager_forward(
                    *(static_inputs[name] for name in self._input_names)
                )
        replay_after_capture(graph, capture_stream)
        logger.info(
            "Captured Flow DiT NPUGraph signature=%s elapsed=%.3fs",
            self._signature(inputs),
            time.perf_counter() - started,
        )
        return GraphEntry(static_inputs, graph, output, capture_stream)


__all__ = ["FlowDiTNPUGraphRunner", "GraphEntry", "replay_after_capture"]
