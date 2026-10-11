# SPDX-License-Identifier: Apache-2.0
"""CPU-only graph startup and routing checks; numerical replay tests require CUDA."""

from contextlib import nullcontext
from typing import Literal
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.nemotron3_5_asr import encoder
from sglang_omni.models.nemotron3_5_asr.encoder import (
    CapturedEncoderBatch,
    NemotronStreamingEncoderGraphRunner,
    encode_pooled_streaming_batch,
)
from sglang_omni.models.nemotron3_5_asr.encoder_state_pool import (
    EncoderPoolLayout,
    NemotronEncoderStatePool,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)


@pytest.mark.parametrize("capture_fails", [False, True])
def test_startup_capture_and_slot_cleanup(
    model: Nemotron3_5AsrForRNNT,
    monkeypatch: pytest.MonkeyPatch,
    capture_fails: bool,
) -> None:
    pool = NemotronEncoderStatePool(EncoderPoolLayout.from_model(model), 8)
    shared_pool = (1, 2)
    capture_context = (
        Mock(side_effect=RuntimeError("capture failed"))
        if capture_fails
        else Mock(return_value=nullcontext())
    )
    monkeypatch.setattr(torch.cuda, "graph_pool_handle", Mock(return_value=shared_pool))
    monkeypatch.setattr(torch.cuda, "Stream", Mock())
    monkeypatch.setattr(torch.cuda, "current_stream", Mock())
    monkeypatch.setattr(torch.cuda, "stream", Mock(return_value=nullcontext()))
    monkeypatch.setattr(torch.cuda, "synchronize", Mock())
    monkeypatch.setattr(torch.cuda, "CUDAGraph", Mock())
    monkeypatch.setattr(torch.cuda, "graph", capture_context)
    monkeypatch.setattr(
        encoder,
        "encode_pooled_windows",
        Mock(return_value=torch.empty(0)),
    )
    with (
        pytest.raises(RuntimeError, match="capture failed")
        if capture_fails
        else nullcontext()
    ):
        runner = NemotronStreamingEncoderGraphRunner(
            model,
            pool,
            subsequent_mel_frames=32,
            num_lookahead_tokens=3,
            max_batch_size=8,
        )
    assert not pool.active_slots
    if not capture_fails:
        assert list(runner.captured_batches) == list(range(8, 0, -1))
        assert all(
            call.kwargs["pool"] is shared_pool
            for call in capture_context.call_args_list
        )
        runner.close()
    else:
        pass
    pool.close()


@pytest.mark.parametrize("execution", ["replay", "first", "uncaptured_batch"])
def test_batch_routes_first_chunks_and_uncaptured_sizes_to_eager(
    model: Nemotron3_5AsrForRNNT,
    monkeypatch: pytest.MonkeyPatch,
    execution: Literal["replay", "first", "uncaptured_batch"],
) -> None:
    pool = NemotronEncoderStatePool(EncoderPoolLayout.from_model(model), 2)
    runner = object.__new__(NemotronStreamingEncoderGraphRunner)
    runner.subsequent_mel_frames = 32
    graph = Mock()
    runner.captured_batches = {
        1: CapturedEncoderBatch(
            input_features=torch.empty(1, 32, 4),
            prompt_ids=torch.empty(1, dtype=torch.long),
            slot_ids=torch.empty(1, dtype=torch.long),
            encoded_frames=torch.empty(1, 4, 8),
            graph=graph,
        )
    }
    batch_size = 2 if execution == "uncaptured_batch" else 1
    slots = [pool.acquire() for _ in range(batch_size)]
    for slot in slots:
        slot.seen_frames = 0 if execution == "first" else 4
    eager = Mock(return_value=torch.empty(batch_size, 4, 8))
    capture = Mock()
    monkeypatch.setattr(encoder, "encode_pooled_windows", eager)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", capture)
    encode_pooled_streaming_batch(
        model,
        [torch.zeros(1, 32, 4) for _ in slots],
        torch.zeros(batch_size, dtype=torch.long),
        encoder_slots=slots,
        num_lookahead_tokens=3,
        graph_runner=runner,
    )
    capture.assert_not_called()
    if execution == "replay":
        graph.replay.assert_called_once_with()
        eager.assert_not_called()
    else:
        graph.replay.assert_not_called()
        eager.assert_called_once()
    pool.close()
