# SPDX-License-Identifier: Apache-2.0
"""CPU-only graph routing checks."""

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

POOL_LAYOUT = EncoderPoolLayout(
    attention_shape=(1, 1, 4, 2),
    convolution_shapes={"conv.0": (2, 2)},
    dtype=torch.float32,
    device=torch.device("cpu"),
)


@pytest.mark.parametrize("execution", ["replay", "first", "uncaptured_batch"])
def test_batch_routes_first_chunks_and_uncaptured_sizes_to_eager(
    monkeypatch: pytest.MonkeyPatch,
    execution: Literal["replay", "first", "uncaptured_batch"],
) -> None:
    pool = NemotronEncoderStatePool(POOL_LAYOUT, 2)
    model = Mock(training=False)
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
    monkeypatch.setattr(encoder, "encode_pooled_windows", eager)
    encode_pooled_streaming_batch(
        model,
        [torch.zeros(1, 32, 4) for _ in slots],
        torch.zeros(batch_size, dtype=torch.long),
        encoder_slots=slots,
        num_lookahead_tokens=3,
        graph_runner=runner,
    )
    if execution == "replay":
        graph.replay.assert_called_once_with()
        eager.assert_not_called()
    else:
        graph.replay.assert_not_called()
        eager.assert_called_once()
    pool.close()
