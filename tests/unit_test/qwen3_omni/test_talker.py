# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import torch

from sglang_omni_v1.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni_v1.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni_v1.models.qwen3_omni.components.talker_input import (
    build_assistant_part,
)
from sglang_omni_v1.models.qwen3_omni.components.talker_prefill import (
    TalkerPrefillBuilder,
)
from sglang_omni_v1.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni_v1.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler


def _sched_req(**data_kwargs: object) -> SimpleNamespace:
    return SimpleNamespace(data=SimpleNamespace(**data_kwargs))


def _take_decode_input(sched_req: SimpleNamespace) -> torch.Tensor | None:
    return QwenTalkerModelRunner._take_next_decode_input_embed(
        sched_req=sched_req,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_qwen_talker_decode_input_consumes_feedback_and_text_or_pad() -> None:
    """Preserves FIFO consumption for ordinary text and final pad fallback."""
    text_req = _sched_req(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque([torch.tensor([20.0, 20.0])]),
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=False,
    )

    assert torch.equal(
        _take_decode_input(text_req),
        torch.tensor([21.0, 22.0]),
    )
    assert len(text_req.data.pending_feedback_queue) == 0
    assert len(text_req.data.pending_text_queue) == 0

    pad_req = _sched_req(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque(),
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=True,
    )
    assert torch.equal(_take_decode_input(pad_req), torch.tensor([8.0, 10.0]))
    assert len(pad_req.data.pending_feedback_queue) == 0
    assert len(pad_req.data.pending_text_queue) == 0


def test_qwen_talker_decode_input_preserves_feedback_until_text_arrives() -> None:
    """Preserves queued feedback when neither text nor final pad is ready."""
    sched_req = _sched_req(
        pending_feedback_queue=deque(
            [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        ),
        pending_text_queue=deque(),
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=False,
    )

    assert _take_decode_input(sched_req) is None
    assert len(sched_req.data.pending_feedback_queue) == 2

    sched_req.data.pending_text_queue.append(torch.tensor([20.0, 20.0]))
    assert torch.equal(_take_decode_input(sched_req), torch.tensor([21.0, 22.0]))
    assert len(sched_req.data.pending_feedback_queue) == 1
    assert torch.equal(
        sched_req.data.pending_feedback_queue[0],
        torch.tensor([3.0, 4.0]),
    )


def test_qwen_talker_decode_readiness_requires_feedback_and_text_or_pad() -> None:
    """Preserves decode gating across no-text, text-ready, and pad-ready states."""
    no_text = SimpleNamespace(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque(),
        thinker_chunks_done=False,
        tts_pad_embed=torch.tensor([7.0, 8.0]),
    )
    with_text = SimpleNamespace(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque([torch.tensor([20.0, 20.0])]),
        thinker_chunks_done=False,
        tts_pad_embed=torch.tensor([7.0, 8.0]),
    )
    with_pad = SimpleNamespace(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque(),
        thinker_chunks_done=True,
        tts_pad_embed=torch.tensor([7.0, 8.0]),
    )

    assert not QwenTalkerModelRunner._data_has_next_decode_input(no_text)
    assert QwenTalkerModelRunner._data_has_next_decode_input(with_text)
    assert QwenTalkerModelRunner._data_has_next_decode_input(with_pad)


def test_qwen_talker_scheduler_waits_for_stream_done_without_replay() -> None:
    """Preserves build gating and avoids replaying prefetched text chunks."""
    scheduler = object.__new__(QwenTalkerScheduler)
    payload = SimpleNamespace(prefetched_chunks=[], prefetched_stream_done=False)

    assert not scheduler._is_request_build_ready(
        payload,
        pending_stream_done=False,
    )
    assert scheduler._is_request_build_ready(
        payload,
        pending_stream_done=True,
    )

    req_data = SimpleNamespace(
        pending_text_queue=deque([torch.tensor([11.0, 12.0])]),
        thinker_chunks_done=True,
    )
    payload = SimpleNamespace(
        prefetched_chunks=[SimpleNamespace(data=torch.tensor([20.0, 20.0]))],
        prefetched_stream_done=True,
    )
    assert scheduler._is_request_build_ready(payload, pending_stream_done=True)
    scheduler._initialize_request_stream_state(req_data, payload)
    assert len(req_data.pending_text_queue) == 1
    assert torch.equal(req_data.pending_text_queue[0], torch.tensor([11.0, 12.0]))


def test_qwen_talker_assistant_part_handles_short_prefix() -> None:
    """Preserves the 9-row assistant layout before a fourth text token exists."""
    assistant_embed = torch.tensor(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ],
        dtype=torch.float32,
    )

    def zero_codec_embed(token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((token_ids.shape[0], 2), dtype=torch.float32)

    result = build_assistant_part(
        assistant_embed=assistant_embed,
        text_projection=lambda tensor: tensor,
        codec_embed_fn=zero_codec_embed,
        tts_bos_embed=torch.tensor([[10.0, 11.0]], dtype=torch.float32),
        tts_eos_embed=torch.tensor([[12.0, 13.0]], dtype=torch.float32),
        tts_pad_embed=torch.tensor([[7.0, 8.0]], dtype=torch.float32),
        speaker_id=1,
        codec_nothink_id=2,
        codec_think_bos_id=3,
        codec_think_eos_id=4,
        codec_pad_id=5,
        codec_bos_id=6,
        tts_pad_token_id=99,
    )

    assert result["input_embeds"].shape == (9, 2)
    assert result["input_ids"].tolist() == [99] * 9
    assert torch.equal(result["input_embeds"][:3], assistant_embed)
    assert torch.equal(
        result["input_embeds"][3:7],
        torch.tensor(
            [[7.0, 8.0], [7.0, 8.0], [7.0, 8.0], [7.0, 8.0]],
            dtype=torch.float32,
        ),
    )
    assert torch.equal(result["input_embeds"][7], torch.tensor([10.0, 11.0]))
    assert torch.equal(result["input_embeds"][8], torch.zeros(2, dtype=torch.float32))
    assert torch.equal(
        result["future_text_rows"],
        torch.tensor([[12.0, 13.0]], dtype=torch.float32),
    )


def test_qwen_talker_prefill_ignores_late_text_after_thinker_done() -> None:
    """Preserves completed thinker streams against late text chunk appends."""
    builder = object.__new__(TalkerPrefillBuilder)
    req_data = SimpleNamespace(
        thinker_chunks_done=True,
        pending_text_queue=deque(),
    )
    chunk = SimpleNamespace(
        data=torch.tensor([1.0], dtype=torch.float32),
        metadata={},
    )

    builder.append_text_chunk(req_data, chunk)

    assert list(req_data.pending_text_queue) == []


def test_qwen_code_predictor_keeps_4d_logits_token_shape() -> None:
    """Preserves 4D code-predictor logits as a two-dimensional token tensor."""
    logits = torch.tensor(
        [
            [[[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]]],
        ],
        dtype=torch.float32,
    )

    sampled = Qwen3OmniTalker._sample_code_predictor_token(logits)

    assert sampled.shape == (1, 2)
    assert sampled.tolist() == [[2, 0]]


def test_qwen_model_runner_and_code_predictor_tensor_contracts() -> None:
    """Preserves multimodal embed injection and code-predictor token shape."""

    class RecordingEmbed:
        num_embeddings = 10

        def __init__(self) -> None:
            self.seen: torch.Tensor | None = None

        def __call__(self, input_ids: torch.Tensor) -> torch.Tensor:
            self.seen = input_ids.clone()
            return torch.zeros((input_ids.shape[0], 4), dtype=torch.float32)

    runner = ThinkerModelRunner.__new__(ThinkerModelRunner)
    runner._embed_tokens = RecordingEmbed()
    runner._image_token_id = 5
    runner._video_token_id = 6
    runner._audio_token_id = 7
    req = SimpleNamespace(
        omni_model_inputs={
            "audio_embeds": torch.tensor([[1.0, 2.0, 3.0, 4.0]]),
            "pad_values": {"audio": 999},
        },
        _omni_consumed=None,
        is_chunked=0,
    )
    input_embeds, _, _ = runner._inject_multimodal_embeds(
        SimpleNamespace(input_ids=torch.tensor([1, 999, 2]), extend_seq_lens_cpu=[3]),
        SimpleNamespace(reqs=[req]),
    )

    assert (
        int(runner._embed_tokens.seen.max().item())
        < runner._embed_tokens.num_embeddings
    )
    assert torch.equal(input_embeds[1], torch.tensor([1.0, 2.0, 3.0, 4.0]))

    logits = torch.tensor([[[0.0, 1.0, 2.0]], [[2.0, 1.0, 0.0]]])
    sampled = Qwen3OmniTalker._sample_code_predictor_token(logits)
    assert sampled.shape == (2, 1)
    assert sampled[:, 0].tolist() == [2, 0]
