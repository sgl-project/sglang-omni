# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen3_omni.components.talker import Qwen3OmniTalker
from sglang_omni.models.qwen3_omni.components.talker_input import build_assistant_part
from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.pending_text_queue import (
    PendingTextTensorQueue,
    coerce_pending_text_queue,
)
from sglang_omni.models.qwen3_omni.request_builders import build_sglang_talker_request
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_omni.talker_scheduler import QwenTalkerScheduler
from tests.unit_test.fixtures.qwen_fakes import FakeQwenTokenizer


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


def test_qwen_talker_decode_input_consumes_device_text_queue() -> None:
    """Preserves FIFO decode semantics for tensor-backed future text rows."""
    text_req = _sched_req(
        pending_feedback_queue=deque(
            [torch.tensor([1.0, 2.0]), torch.tensor([3.0, 4.0])]
        ),
        pending_text_queue=PendingTextTensorQueue.from_tensor(
            torch.tensor([[20.0, 20.0], [30.0, 30.0]])
        ),
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=False,
    )

    assert torch.equal(_take_decode_input(text_req), torch.tensor([21.0, 22.0]))
    assert len(text_req.data.pending_text_queue) == 1
    assert torch.equal(_take_decode_input(text_req), torch.tensor([33.0, 34.0]))
    assert len(text_req.data.pending_text_queue) == 0


def test_qwen_talker_decode_input_rejects_implicit_row_transfer() -> None:
    """Keeps decode hot path free of implicit dtype/device conversions."""
    sched_req = _sched_req(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque([torch.tensor([20.0, 20.0], dtype=torch.float64)]),
        tts_pad_embed=torch.tensor([7.0, 8.0]),
        thinker_chunks_done=False,
    )

    with pytest.raises(RuntimeError, match="must already match"):
        _take_decode_input(sched_req)


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


def test_qwen_talker_prefill_keeps_future_rows_device_backed() -> None:
    """Avoids splitting future text rows into per-row CPU tensors."""
    builder = object.__new__(TalkerPrefillBuilder)
    rows = torch.empty((2, 3), device="meta")

    queue = builder.tensor_rows_to_queue(rows)

    assert isinstance(queue, PendingTextTensorQueue)
    assert len(queue) == 2
    assert queue[0].device.type == "meta"


def test_pending_text_queue_rejects_unexpected_rank() -> None:
    """Keeps queue shape handling explicit instead of flattening unknown ranks."""
    queue = PendingTextTensorQueue()

    with pytest.raises(ValueError, match="1D row tensor or a 2D row batch"):
        queue.append_rows(torch.zeros((1, 2, 3)))
    with pytest.raises(ValueError, match="non-empty hidden dimension"):
        queue.append_rows(torch.zeros((1, 0)))


def test_pending_text_queue_rejects_non_tensor_input() -> None:
    """Keeps conversion failures explicit instead of skipping invalid rows."""
    with pytest.raises(TypeError, match="pending text rows must be tensors"):
        PendingTextTensorQueue.from_tensor(None)

    with pytest.raises(TypeError, match="pending text rows must be tensors"):
        coerce_pending_text_queue([torch.tensor([1.0]), object()])
    with pytest.raises(TypeError, match="pending text queue must be None"):
        coerce_pending_text_queue(object())


def test_coerce_pending_text_queue_copies_cursor_state() -> None:
    """Avoids sharing mutable FIFO cursor state across request data objects."""
    queue = PendingTextTensorQueue.from_tensor(torch.tensor([[1.0], [2.0]]))

    copied = coerce_pending_text_queue(queue)
    copied.popleft()

    assert copied is not queue
    assert len(copied) == 1
    assert len(queue) == 2


def test_qwen_talker_prefill_appends_text_chunks_to_tensor_queue() -> None:
    """Preserves incremental text appends without switching back to deque."""
    builder = object.__new__(TalkerPrefillBuilder)
    builder._im_end_token_id = 99

    def project_assistant_chunk(chunk: SimpleNamespace) -> torch.Tensor:
        del chunk
        return torch.tensor([11.0, 12.0])

    builder.project_assistant_chunk = project_assistant_chunk
    req_data = SimpleNamespace(
        thinker_chunks_done=False,
        pending_text_queue=None,
    )
    chunk = SimpleNamespace(data=None, metadata={})

    builder.append_text_chunk(req_data, chunk)

    assert isinstance(req_data.pending_text_queue, PendingTextTensorQueue)
    assert torch.equal(req_data.pending_text_queue[0], torch.tensor([11.0, 12.0]))


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


@pytest.fixture()
def _patch_sampling(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.normalize",
        lambda _self, _tok: None,
    )
    monkeypatch.setattr(
        "sglang.srt.sampling.sampling_params.SamplingParams.verify",
        lambda _self, _vs: None,
    )


@pytest.mark.usefixtures("_patch_sampling")
class TestBuildTalkerRequestTensorStorage:
    """build_sglang_talker_request stores the tensor and honours the Req list contract."""

    def test_projected_embeds_path(self) -> None:
        seq_len, hidden = 64, 128
        embeds = torch.randn(seq_len, hidden)
        ids = torch.arange(seq_len, dtype=torch.long)

        data = build_sglang_talker_request(
            thinker_hidden_states=torch.empty(0),
            tokenizer=FakeQwenTokenizer(),
            codec_vocab_size=4096,
            talker_input_embeds=embeds,
            talker_input_ids=ids,
            input_embeds_are_projected=True,
        )

        assert data.prefill_input_embeds is embeds
        assert data.req.input_embeds is None
        assert data.req._input_embeds_are_projected is True
        assert data.input_embeds_are_projected is True

    def test_hidden_states_path(self) -> None:
        seq_len, hidden = 32, 256
        hidden_states = torch.randn(seq_len, hidden)

        data = build_sglang_talker_request(
            thinker_hidden_states=hidden_states,
            tokenizer=FakeQwenTokenizer(),
            codec_vocab_size=4096,
        )

        assert data.prefill_input_embeds is None
        assert isinstance(data.req.input_embeds, list)
        assert len(data.req.input_embeds) == seq_len
        assert data.req._input_embeds_are_projected is False


def test_projected_prefill_reads_tensor_from_data() -> None:
    """Model runner reads prefill_input_embeds, not Req.input_embeds."""
    embeds = torch.randn(10, 64)
    sched_req = _sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=embeds,
        req=SimpleNamespace(input_embeds=None, prefix_indices=[], extend_input_len=10),
    )
    forward_batch = SimpleNamespace(
        input_embeds=None,
        input_ids=torch.zeros(10, dtype=torch.long),
    )

    runner = object.__new__(QwenTalkerModelRunner)
    runner._forward_with_input_embeds = (
        lambda self, fb, *, input_embeds, **kw: SimpleNamespace(
            next_token_ids=None, logits_output=None, _embeds=input_embeds
        )
    ).__get__(runner)

    result = runner._run_projected_prefill_forward(
        forward_batch, schedule_batch=None, requests=[sched_req]
    )

    assert torch.equal(result._embeds, embeds)


def test_projected_prefill_slices_tensor_by_prefix_indices() -> None:
    """Tensor path slices by prefix_indices, matching the list fallback."""
    full_embeds = torch.randn(10, 64)
    prefix_len = 3
    sched_req = _sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=full_embeds,
        req=SimpleNamespace(
            input_embeds=None,
            prefix_indices=list(range(prefix_len)),
            extend_input_len=7,
        ),
    )
    forward_batch = SimpleNamespace(
        input_embeds=None,
        input_ids=torch.zeros(7, dtype=torch.long),
    )

    runner = object.__new__(QwenTalkerModelRunner)
    runner._forward_with_input_embeds = (
        lambda self, fb, *, input_embeds, **kw: SimpleNamespace(
            next_token_ids=None, logits_output=None, _embeds=input_embeds
        )
    ).__get__(runner)

    result = runner._run_projected_prefill_forward(
        forward_batch, schedule_batch=None, requests=[sched_req]
    )

    expected = full_embeds[prefix_len:]
    assert result._embeds.shape == expected.shape
    assert torch.equal(result._embeds, expected)


def test_projected_prefill_slices_tensor_by_extend_input_len() -> None:
    """Tensor path slices by prefix and extend length, matching SGLang prefill."""
    full_embeds = torch.randn(10, 64)
    prefix_len = 3
    extend_len = 4
    sched_req = _sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=full_embeds,
        req=SimpleNamespace(
            input_embeds=None,
            prefix_indices=list(range(prefix_len)),
            extend_input_len=extend_len,
        ),
    )
    forward_batch = SimpleNamespace(
        input_embeds=None,
        input_ids=torch.zeros(extend_len, dtype=torch.long),
    )

    runner = object.__new__(QwenTalkerModelRunner)
    runner._forward_with_input_embeds = (
        lambda self, fb, *, input_embeds, **kw: SimpleNamespace(
            next_token_ids=None, logits_output=None, _embeds=input_embeds
        )
    ).__get__(runner)

    result = runner._run_projected_prefill_forward(
        forward_batch, schedule_batch=None, requests=[sched_req]
    )

    expected = full_embeds[prefix_len : prefix_len + extend_len]
    assert result._embeds.shape == expected.shape
    assert torch.equal(result._embeds, expected)


def test_projected_prefill_list_fallback_slices_by_extend_input_len() -> None:
    """List fallback keeps the same prefill slice contract as the tensor path."""
    full_embeds = torch.randn(10, 64)
    prefix_len = 2
    extend_len = 5
    sched_req = _sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=None,
        req=SimpleNamespace(
            input_embeds=full_embeds.tolist(),
            prefix_indices=list(range(prefix_len)),
            extend_input_len=extend_len,
        ),
    )
    forward_batch = SimpleNamespace(
        input_embeds=None,
        input_ids=torch.zeros(extend_len, dtype=torch.long),
    )

    runner = object.__new__(QwenTalkerModelRunner)
    runner._forward_with_input_embeds = (
        lambda self, fb, *, input_embeds, **kw: SimpleNamespace(
            next_token_ids=None, logits_output=None, _embeds=input_embeds
        )
    ).__get__(runner)

    result = runner._run_projected_prefill_forward(
        forward_batch, schedule_batch=None, requests=[sched_req]
    )

    expected = full_embeds[prefix_len : prefix_len + extend_len]
    assert result._embeds.shape == expected.shape
    assert torch.allclose(result._embeds, expected)


def test_projected_prefill_prefers_request_data_over_forward_embeds() -> None:
    """Projected rows live on request data, not ForwardBatch.input_embeds."""
    embeds = torch.randn(4, 8)
    stale_forward_embeds = torch.full((2, 8), -1.0)
    sched_req = _sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=embeds,
        req=SimpleNamespace(input_embeds=None, prefix_indices=[], extend_input_len=4),
    )
    forward_batch = SimpleNamespace(
        input_embeds=stale_forward_embeds,
        input_ids=torch.zeros(4, dtype=torch.long),
    )

    runner = object.__new__(QwenTalkerModelRunner)
    runner._forward_with_input_embeds = (
        lambda self, fb, *, input_embeds, **kw: SimpleNamespace(
            next_token_ids=None, logits_output=None, _embeds=input_embeds
        )
    ).__get__(runner)

    result = runner._run_projected_prefill_forward(
        forward_batch, schedule_batch=None, requests=[sched_req]
    )

    assert torch.equal(result._embeds, embeds)


def test_projected_prefill_rejects_mixed_projected_and_list_batch() -> None:
    """The model forward has one projection mode, so mixed batches are invalid."""
    projected_req = _sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=torch.randn(2, 8),
        req=SimpleNamespace(input_embeds=None, prefix_indices=[], extend_input_len=2),
    )
    list_req = _sched_req(
        input_embeds_are_projected=False,
        prefill_input_embeds=None,
        req=SimpleNamespace(
            input_embeds=torch.randn(2, 8).tolist(),
            prefix_indices=[],
            extend_input_len=2,
        ),
    )
    forward_batch = SimpleNamespace(
        input_embeds=torch.randn(2, 8),
        input_ids=torch.zeros(4, dtype=torch.long),
    )

    runner = object.__new__(QwenTalkerModelRunner)

    with pytest.raises(RuntimeError, match="cannot be batched together"):
        runner._run_projected_prefill_forward(
            forward_batch, schedule_batch=None, requests=[projected_req, list_req]
        )


def test_projected_prefill_full_prefix_hit_returns_none() -> None:
    """Full prefix hit produces no embeds, method returns None."""
    embeds = torch.randn(5, 64)
    sched_req = _sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=embeds,
        req=SimpleNamespace(
            input_embeds=None, prefix_indices=list(range(5)), extend_input_len=0
        ),
    )
    forward_batch = SimpleNamespace(
        input_embeds=None,
        input_ids=torch.zeros(0, dtype=torch.long),
    )

    runner = object.__new__(QwenTalkerModelRunner)

    result = runner._run_projected_prefill_forward(
        forward_batch, schedule_batch=None, requests=[sched_req]
    )

    assert result is None


def test_post_prefill_preserves_prefill_embeds_for_retract() -> None:
    """post_prefill keeps prefill_input_embeds so retract can re-prefill."""
    embeds = torch.randn(4, 8)
    sched_req = _sched_req(
        prefill_input_embeds=embeds,
        pending_feedback_queue=deque(),
        pending_text_queue=deque(),
        tts_pad_embed=None,
        thinker_chunks_done=True,
    )

    runner = object.__new__(QwenTalkerModelRunner)
    runner._feedback_enabled = False

    runner.post_prefill(
        SimpleNamespace(next_token_ids=None),
        forward_batch=None,
        schedule_batch=None,
        requests=[sched_req],
    )
    assert sched_req.data.prefill_input_embeds is embeds


def test_projected_prefill_survives_decode_retract() -> None:
    """Re-prefill after a simulated decode retract still feeds projected embeds."""
    full_embeds = torch.randn(10, 64)
    sched_req = _sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=full_embeds,
        req=SimpleNamespace(
            input_embeds=None,
            prefix_indices=[],
            extend_input_len=10,
        ),
        pending_feedback_queue=deque(),
        pending_text_queue=deque(),
        tts_pad_embed=None,
        thinker_chunks_done=True,
    )
    forward_batch = SimpleNamespace(
        input_embeds=None,
        input_ids=torch.zeros(10, dtype=torch.long),
    )

    runner = object.__new__(QwenTalkerModelRunner)
    runner._feedback_enabled = False
    runner._forward_with_input_embeds = (
        lambda self, fb, *, input_embeds, **kw: SimpleNamespace(
            next_token_ids=None, logits_output=None, _embeds=input_embeds
        )
    ).__get__(runner)

    first = runner._run_projected_prefill_forward(
        forward_batch, schedule_batch=None, requests=[sched_req]
    )
    assert torch.equal(first._embeds, full_embeds)

    runner.post_prefill(
        first,
        forward_batch=None,
        schedule_batch=None,
        requests=[sched_req],
    )

    sched_req.data.req.prefix_indices = []
    sched_req.data.req.extend_input_len = 10

    second = runner._run_projected_prefill_forward(
        forward_batch, schedule_batch=None, requests=[sched_req]
    )
    assert second is not None, "retract+re-prefill must not silently lose embeds"
    assert torch.equal(second._embeds, full_embeds)


def test_write_feedback_buffers_records_decode_input_history() -> None:
    """Decode inputs consumed by the feedback buffer are replayable after retract."""
    feedback_buffer = torch.zeros(1, 2)
    feedback_mask = torch.zeros(1, dtype=torch.bool)
    sched_req = _sched_req(
        pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
        pending_text_queue=deque([torch.tensor([20.0, 30.0])]),
        decode_input_embeds=[],
    )

    runner = object.__new__(QwenTalkerModelRunner)
    runner.model = SimpleNamespace(
        _feedback_buffer=feedback_buffer,
        _feedback_mask=feedback_mask,
    )

    runner._write_feedback_buffers([sched_req])

    assert feedback_mask.tolist() == [True]
    assert torch.equal(feedback_buffer[0], torch.tensor([21.0, 32.0]))
    assert len(sched_req.data.decode_input_embeds) == 1
    assert torch.equal(
        sched_req.data.decode_input_embeds[0],
        torch.tensor([21.0, 32.0]),
    )


def test_projected_prefill_retract_replays_generated_decode_inputs() -> None:
    """Retracted prefill can span prompt suffix and generated codec tokens."""
    full_embeds = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    decode_history = [
        torch.tensor([100.0, 101.0]),
        torch.tensor([200.0, 201.0]),
    ]
    sched_req = _sched_req(
        input_embeds_are_projected=True,
        prefill_input_embeds=full_embeds,
        decode_input_embeds=decode_history,
        pending_feedback_queue=deque([torch.tensor([3.0, 4.0])]),
        pending_text_queue=deque([torch.tensor([30.0, 40.0])]),
        req=SimpleNamespace(
            input_embeds=None,
            prefix_indices=list(range(8)),
            extend_input_len=5,
            output_ids=[11, 12, 13],
        ),
    )
    forward_batch = SimpleNamespace(
        input_embeds=None,
        input_ids=torch.zeros(5, dtype=torch.long),
    )

    runner = object.__new__(QwenTalkerModelRunner)
    runner._forward_with_input_embeds = (
        lambda self, fb, *, input_embeds, **kw: SimpleNamespace(
            next_token_ids=None, logits_output=None, _embeds=input_embeds
        )
    ).__get__(runner)

    result = runner._run_projected_prefill_forward(
        forward_batch, schedule_batch=None, requests=[sched_req]
    )

    expected = torch.cat(
        [
            full_embeds[8:10],
            torch.stack(
                [
                    torch.tensor([100.0, 101.0]),
                    torch.tensor([200.0, 201.0]),
                    torch.tensor([33.0, 44.0]),
                ]
            ),
        ],
        dim=0,
    )
    assert torch.equal(result._embeds, expected)
    assert len(sched_req.data.decode_input_embeds) == 3
    assert len(sched_req.data.pending_feedback_queue) == 0
    assert len(sched_req.data.pending_text_queue) == 0


@pytest.mark.benchmark
@pytest.mark.usefixtures("_patch_sampling")
@pytest.mark.parametrize("seq_len", [256, 2048, 4096])
def test_build_talker_request_wall_clock(seq_len: int) -> None:
    """Wall-clock for request build at representative seq_lens."""
    embeds = torch.randn(seq_len, 2048)
    ids = torch.arange(seq_len, dtype=torch.long)
    tokenizer = FakeQwenTokenizer()

    def _build():
        return build_sglang_talker_request(
            thinker_hidden_states=torch.empty(0),
            tokenizer=tokenizer,
            codec_vocab_size=4096,
            talker_input_embeds=embeds,
            talker_input_ids=ids,
            input_embeds_are_projected=True,
        )

    for _ in range(3):
        _build()

    t0 = time.perf_counter()
    for _ in range(20):
        _build()
    mean_ms = (time.perf_counter() - t0) / 20 * 1000

    print(f"\n[seq_len={seq_len}] mean={mean_ms:.2f}ms  floats={seq_len * 2048:,}")
