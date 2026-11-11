# SPDX-License-Identifier: Apache-2.0
"""Capture decode metadata without synchronizing the live encoder stream."""

from __future__ import annotations

import logging

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph

logger = logging.getLogger(__name__)

GlueGraphKey = tuple[int, str, str]


def run_with_thread_local_capture(
    self: MetadataGlueGraph,
    attention_backend: AttentionBackend,
    forward_batch: ForwardBatch,
    replay_key: GlueGraphKey,
) -> None:
    """Prepare decode metadata, capturing only operations owned by this thread."""
    state = self._states.get(
        replay_key
    )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    if state is None:
        state = {"warmups": 0, "graph": None, "meta": None}
        self._states[replay_key] = (
            state  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        )
    else:
        pass

    if state["graph"] is not None:
        for backend, metadata in state["meta"]:
            backend.forward_metadata = metadata
        state["graph"].replay()
        return
    else:
        pass

    if state["warmups"] < self.NUM_WARMUP:
        state["warmups"] += 1
        attention_backend.init_forward_metadata_out_graph(forward_batch)
        return
    else:
        pass

    if (
        self._capture_stream is None
    ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self._capture_stream = (
            torch.cuda.Stream()
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    else:
        pass
    graph = torch.cuda.CUDAGraph()
    try:
        # note (Zhuofeng Li): torch.cuda.graph synchronizes the live encoder on entry.
        with torch.cuda.stream(
            self._capture_stream
        ):  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
            graph.capture_begin(capture_error_mode="thread_local")
            try:
                attention_backend.init_forward_metadata_out_graph(forward_batch)
            finally:
                graph.capture_end()
    except Exception:
        logger.warning(
            f"Metadata glue-graph capture failed for key {replay_key}; falling back "
            "to eager metadata prep permanently.",
            exc_info=True,
        )
        self.disabled = True
        # Ops under a failed capture were recorded, not executed.
        attention_backend.init_forward_metadata_out_graph(forward_batch)
        return

    state["meta"] = [
        (backend, backend.forward_metadata)
        for backend in self._leaves(
            attention_backend
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    ]
    state["graph"] = graph
    # Capture records without executing; replay once for this step's prep.
    graph.replay()


def apply_thread_local_glue_capture() -> None:
    """Route MetadataGlueGraph captures through the thread-local variant."""
    if MetadataGlueGraph.run is run_with_thread_local_capture:
        return
    else:
        pass
    MetadataGlueGraph.run = run_with_thread_local_capture
    logger.info("[qwen3-asr] metadata glue graph captures in thread-local mode")
