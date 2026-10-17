# SPDX-License-Identifier: Apache-2.0
"""Decode metadata graph capture lifecycle and eager recovery."""

from contextlib import nullcontext
from unittest.mock import MagicMock

import pytest
import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.runner.metadata_glue_graph import MetadataGlueGraph

from sglang_omni.models.qwen3_asr.metadata_glue import run_with_thread_local_capture


@pytest.mark.parametrize("capture_fails", [False, True])
def test_metadata_capture_replays_first_step_or_recovers_eagerly(
    monkeypatch: pytest.MonkeyPatch, capture_fails: bool
) -> None:
    """A recorded first step must run, and failed captures must restore eager prep."""
    graph = MagicMock(spec=torch.cuda.CUDAGraph)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: graph)
    monkeypatch.setattr(torch.cuda, "Stream", MagicMock())
    monkeypatch.setattr(torch.cuda, "stream", lambda stream: nullcontext())
    backend = MagicMock(spec=AttentionBackend)
    backend.attn_backend_list = None
    metadata = {"positions": torch.arange(4)}
    backend.forward_metadata = metadata
    forward_batch = MagicMock(spec=ForwardBatch)
    glue = MetadataGlueGraph(torch.device("cuda"))
    for _ in range(glue.NUM_WARMUP):
        run_with_thread_local_capture(
            glue, backend, forward_batch, (4, "decode", "default")
        )
    backend.init_forward_metadata_out_graph.reset_mock()
    if capture_fails:
        backend.init_forward_metadata_out_graph.side_effect = [
            RuntimeError("capture invalidated"),
            None,
        ]
    run_with_thread_local_capture(
        glue, backend, forward_batch, (4, "decode", "default")
    )
    graph.capture_begin.assert_called_once_with(capture_error_mode="thread_local")
    graph.capture_end.assert_called_once_with()
    if capture_fails:
        assert glue.disabled
        assert backend.init_forward_metadata_out_graph.call_count == 2
        graph.replay.assert_not_called()
    else:
        graph.replay.assert_called_once_with()
        backend.forward_metadata = None
        run_with_thread_local_capture(
            glue, backend, forward_batch, (4, "decode", "default")
        )
        assert backend.forward_metadata is metadata
        assert graph.replay.call_count == 2
        assert backend.init_forward_metadata_out_graph.call_count == 1


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph capture requires CUDA"
)
def test_metadata_graph_refreshes_live_inputs_after_capture() -> None:
    """Replay must refresh tensors from current inputs rather than captured values."""
    positions = torch.arange(4, device="cuda")
    output = torch.empty_like(positions)
    backend = MagicMock(spec=AttentionBackend)
    backend.attn_backend_list = None

    def prepare_metadata(forward_batch: ForwardBatch) -> None:
        torch.add(positions, 1, out=output)
        backend.forward_metadata = output

    backend.init_forward_metadata_out_graph.side_effect = prepare_metadata
    forward_batch = MagicMock(spec=ForwardBatch)
    glue = MetadataGlueGraph(torch.device("cuda"))
    for value in range(glue.NUM_WARMUP + 3):
        positions.fill_(value)
        run_with_thread_local_capture(
            glue, backend, forward_batch, (4, "decode", "default")
        )
        assert not glue.disabled
        torch.testing.assert_close(output, positions + 1, rtol=0, atol=0)
        assert backend.forward_metadata is output
        backend.forward_metadata = None
