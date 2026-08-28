# SPDX-License-Identifier: Apache-2.0
"""Ascend NPUGraph replay for the buffered CosyVoice3 Flow estimator."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from typing import Any

import torch

from sglang_omni.utils.flow_dit_npugraph import (
    FlowDiTNPUGraphRunner,
    GraphEntry as _GraphEntry,
    replay_after_capture as _replay_after_capture,
)

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
            capture_context=_graph_safe_mask_scope,
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
            return self._eager_forward(
                x, mask, mu, t, spks, cond, streaming=True
            )
        return super().__call__(x, mask, mu, t, spks, cond)


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
