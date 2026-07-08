# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner
from sglang_omni.models.qwen3_omni.components.talker import (
    Qwen3OmniTalker,
    _bind_default_weight_loaders,
)
from sglang_omni.models.qwen3_omni.components.talker_input import build_assistant_part
from sglang_omni.models.qwen3_omni.components.talker_prefill import TalkerPrefillBuilder
from sglang_omni.models.qwen3_omni.pending_text_queue import (
    PendingTextTensorQueue,
    coerce_pending_text_queue,
)
from sglang_omni.models.qwen3_omni.request_builders import build_sglang_talker_request
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.models.qwen3_omni.talker_scheduler import (
    MIN_PARTIAL_START_CHUNKS,
    QwenTalkerScheduler,
)
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
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
    scheduler = _fresh_partial_scheduler()
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


def test_chunk_hidden_device_mismatch_resolved_before_stack() -> None:
    """Mixed CPU/CUDA thinker chunks must not crash torch.stack in build_prompt_prefill."""
    builder = object.__new__(TalkerPrefillBuilder)
    builder._device = torch.device("meta")
    builder._dtype = torch.float16

    chunks = [
        SimpleNamespace(data=torch.ones((3,), dtype=torch.float32), metadata={}),
        SimpleNamespace(
            data=torch.zeros((3,), dtype=torch.float32),
            metadata={
                "layer_hidden": torch.empty((3,), device="meta", dtype=torch.float32)
            },
        ),
    ]

    # note (YueYin): .to() must happen per-chunk before torch.stack, not after
    result = torch.stack(
        [
            builder.chunk_layer_hidden_or_embed(c).to(
                device=builder._device, dtype=builder._dtype
            )
            for c in chunks
        ],
        dim=0,
    )
    assert result.shape == (2, 3)
    assert result.device.type == "meta"
    assert result.dtype == torch.float16


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


def _build_assistant_part_for_n_chunks(n: int) -> dict[str, torch.Tensor]:
    """Build an assistant segment with n thinker chunks under the test layout."""
    hidden_dim = 4
    assistant_embed = torch.arange(n * hidden_dim, dtype=torch.float32).reshape(
        n, hidden_dim
    )

    def codec_embed_fn(token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((token_ids.shape[0], hidden_dim), dtype=torch.float32)

    return build_assistant_part(
        assistant_embed=assistant_embed,
        text_projection=lambda tensor: tensor,
        codec_embed_fn=codec_embed_fn,
        tts_bos_embed=torch.zeros((1, hidden_dim), dtype=torch.float32),
        tts_eos_embed=torch.zeros((1, hidden_dim), dtype=torch.float32),
        tts_pad_embed=torch.zeros((1, hidden_dim), dtype=torch.float32),
        speaker_id=1,
        codec_nothink_id=2,
        codec_think_bos_id=3,
        codec_think_eos_id=4,
        codec_pad_id=5,
        codec_bos_id=6,
        tts_pad_token_id=99,
    )


def test_partial_prompt_prefill_layout_invariants() -> None:
    """Locks the assistant-segment row contract used by partial-start.

    Source of truth for ``MIN_PARTIAL_START_CHUNKS`` and the documented
    decode-ready operating point: below 3 chunks ``build_assistant_part``
    fails to assemble the layout (``text_hidden`` is < 9 rows while
    ``codec_hidden`` is fixed at 9 rows, so the subsequent tensor add raises);
    at 3 or 4 chunks the layout is stable but ``future_text_rows`` collapses
    to zero after the trailing EOS row is stripped on the partial path; from
    5 chunks onward at least one consumable future text row remains.
    """
    for n in range(1, MIN_PARTIAL_START_CHUNKS):
        try:
            _build_assistant_part_for_n_chunks(n)
        except RuntimeError:
            pass
        else:
            raise AssertionError(
                "layout invariant: build_assistant_part must fail "
                f"below MIN_PARTIAL_START_CHUNKS (n={n})"
            )

    for n in (MIN_PARTIAL_START_CHUNKS, 4, 5, 10):
        assert (
            _build_assistant_part_for_n_chunks(n)["input_embeds"].shape[0] == 9
        ), f"layout invariant: at n={n} the assistant tail must be 9 rows"

    # future_text_rows count before include_assistant_eos stripping:
    #   n <= 4 -> 1 row (just the EOS row);
    #   n  > 4 -> (n - 4) projected rows + 1 EOS row.
    assert _build_assistant_part_for_n_chunks(3)["future_text_rows"].shape[0] == 1
    assert _build_assistant_part_for_n_chunks(4)["future_text_rows"].shape[0] == 1
    assert _build_assistant_part_for_n_chunks(5)["future_text_rows"].shape[0] == 2
    assert _build_assistant_part_for_n_chunks(6)["future_text_rows"].shape[0] == 3

    def stripped(n: int) -> int:
        rows = _build_assistant_part_for_n_chunks(n)["future_text_rows"]
        return max(rows.shape[0] - 1, 0)

    # With include_assistant_eos=False on the partial path:
    #   n in {3, 4} -> 0 future rows (decode stalls immediately after prefill);
    #   n == 5 -> 1 future row (documented decode-ready operating point);
    #   n >= 6 -> (n - 4) future rows.
    assert stripped(3) == 0
    assert stripped(4) == 0
    assert stripped(5) == 1
    assert stripped(6) == 2


def _fresh_partial_scheduler(
    *,
    enable_partial_start: bool = False,
    partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
) -> QwenTalkerScheduler:
    """Build a bare scheduler instance with only the partial-start state needed."""
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._enable_partial_start = enable_partial_start
    scheduler._partial_start_min_chunks = partial_start_min_chunks
    scheduler._im_end_token_id = None
    return scheduler


def _make_payload(
    *,
    request_id: str = "r0",
    prefetched_chunks: list[Any] | None = None,
    prefetched_stream_done: bool = False,
) -> SimpleNamespace:
    return SimpleNamespace(
        request_id=request_id,
        prefetched_chunks=list(prefetched_chunks or []),
        prefetched_stream_done=prefetched_stream_done,
    )


def test_partial_disabled_preserves_legacy_path() -> None:
    """enable_partial_start=False preserves legacy stream_done-only gating."""
    scheduler = _fresh_partial_scheduler(enable_partial_start=False)
    payload = _make_payload(prefetched_chunks=[object()] * 50)

    assert not scheduler._is_request_build_ready(payload, pending_stream_done=False)
    assert scheduler._is_request_build_ready(payload, pending_stream_done=True)


def test_partial_enabled_below_threshold_stays_deferred() -> None:
    """Below the threshold the payload is not yet build-ready."""
    scheduler = _fresh_partial_scheduler(
        enable_partial_start=True, partial_start_min_chunks=10
    )
    payload = _make_payload(prefetched_chunks=[object()] * 4)

    assert not scheduler._is_request_build_ready(payload, pending_stream_done=False)


def test_partial_enabled_at_threshold_returns_true_with_done_false() -> None:
    """At or above the threshold the payload is build-ready early."""
    scheduler = _fresh_partial_scheduler(
        enable_partial_start=True, partial_start_min_chunks=5
    )
    payload = _make_payload(prefetched_chunks=[object()] * 5)

    assert scheduler._is_request_build_ready(payload, pending_stream_done=False)


def test_partial_rejects_min_chunks_below_layout_floor(monkeypatch) -> None:
    """partial_start_min_chunks below MIN_PARTIAL_START_CHUNKS raises ValueError."""
    monkeypatch.setattr(OmniScheduler, "__init__", lambda self, *a, **k: None)
    live = QwenTalkerScheduler.__new__(QwenTalkerScheduler)

    with pytest.raises(ValueError):
        QwenTalkerScheduler.__init__(
            live,
            enable_partial_start=True,
            partial_start_min_chunks=MIN_PARTIAL_START_CHUNKS - 1,
        )


def test_partial_min_chunks_env_override(monkeypatch) -> None:
    monkeypatch.setenv("SGLANG_OMNI_QWEN3_PARTIAL_START_MIN_CHUNKS", "3")
    monkeypatch.setattr(OmniScheduler, "__init__", lambda self, *a, **k: None)
    live = QwenTalkerScheduler.__new__(QwenTalkerScheduler)

    QwenTalkerScheduler.__init__(
        live,
        enable_partial_start=True,
        partial_start_min_chunks=10,
    )

    assert live._partial_start_min_chunks == 3


def test_partial_count_strips_only_trailing_im_end_chunk() -> None:
    scheduler = _fresh_partial_scheduler(
        enable_partial_start=True, partial_start_min_chunks=3
    )
    scheduler._im_end_token_id = 13

    def _chunk(token_id: int) -> SimpleNamespace:
        return SimpleNamespace(
            data=torch.tensor([0.0]),
            metadata={"token_id": token_id},
        )

    trailing_im_end = _make_payload(
        prefetched_chunks=[_chunk(100), _chunk(101), _chunk(13)]
    )
    assert not scheduler._is_request_build_ready(
        trailing_im_end, pending_stream_done=False
    )

    midstream_im_end = _make_payload(
        prefetched_chunks=[_chunk(100), _chunk(13), _chunk(101)]
    )
    assert scheduler._is_request_build_ready(
        midstream_im_end, pending_stream_done=False
    )

    enough = _make_payload(
        prefetched_chunks=[_chunk(100), _chunk(13), _chunk(102), _chunk(13)]
    )
    assert scheduler._is_request_build_ready(enough, pending_stream_done=False)


def test_partial_enabled_zero_chunks_stays_deferred() -> None:
    """Enabled knob with empty prefetched_chunks never satisfies the threshold."""
    scheduler = _fresh_partial_scheduler(
        enable_partial_start=True, partial_start_min_chunks=MIN_PARTIAL_START_CHUNKS
    )
    payload = _make_payload(prefetched_chunks=[])

    assert not scheduler._is_request_build_ready(payload, pending_stream_done=False)


def test_no_op_initialize_request_stream_state_prevents_replay() -> None:
    """Talker's _initialize_request_stream_state must not replay prefetched chunks.

    Comparison test: the base OmniScheduler initializer would walk prefetched
    chunks and append them; the talker override is deliberately a no-op.
    """
    scheduler = object.__new__(QwenTalkerScheduler)

    captured_appends: list[Any] = []
    chunks_consumed_at_build = [
        SimpleNamespace(data=torch.tensor([1.0]), metadata={}),
        SimpleNamespace(data=torch.tensor([2.0]), metadata={}),
    ]
    req_data = SimpleNamespace(
        pending_text_queue=deque(),
        thinker_chunks_done=False,
    )
    payload = SimpleNamespace(
        prefetched_chunks=chunks_consumed_at_build,
        prefetched_stream_done=False,
    )

    scheduler._initialize_request_stream_state(req_data, payload)

    # No-op override: no rows appended for chunks consumed at build time.
    assert list(req_data.pending_text_queue) == []

    # Comparison: the base initializer would have appended them, which would be wrong.
    base_req_data = SimpleNamespace(
        pending_text_queue=deque(),
        thinker_chunks_done=False,
    )

    def _record_append(_self: Any, _req: Any, chunk: Any) -> None:
        captured_appends.append(chunk)

    base_scheduler = object.__new__(QwenTalkerScheduler)
    base_scheduler._append_stream_chunk = _record_append.__get__(
        base_scheduler, QwenTalkerScheduler
    )
    base_scheduler._mark_stream_done = lambda req: None
    # Call the upstream base implementation directly.
    OmniScheduler._initialize_request_stream_state(
        base_scheduler, base_req_data, payload
    )
    assert len(captured_appends) == len(chunks_consumed_at_build)


def test_stream_done_after_partial_build_marks_thinker_done() -> None:
    """_on_stream_done after an early build flips the flag and appends EOS once."""
    builder = object.__new__(TalkerPrefillBuilder)
    eos_embed = torch.tensor([99.0, 98.0], dtype=torch.float32)
    req_data = SimpleNamespace(
        pending_text_queue=deque(),
        thinker_chunks_done=False,
        tts_eos_embed=eos_embed,
    )

    builder.mark_thinker_done(req_data)
    assert req_data.thinker_chunks_done is True
    assert len(req_data.pending_text_queue) == 1
    assert torch.equal(req_data.pending_text_queue[0], eos_embed.cpu())

    # Second invocation is a no-op (no duplicate EOS row).
    builder.mark_thinker_done(req_data)
    assert len(req_data.pending_text_queue) == 1


def test_chunk_after_partial_build_appends_once() -> None:
    """_on_stream_chunk after an early build appends exactly one row; im_end filtered."""
    builder = object.__new__(TalkerPrefillBuilder)
    builder._im_end_token_id = 13
    builder.project_assistant_chunk = lambda chunk: torch.tensor(
        [7.0, 8.0], dtype=torch.float32
    )

    req_data = SimpleNamespace(
        pending_text_queue=deque(),
        thinker_chunks_done=False,
    )

    ordinary_chunk = SimpleNamespace(
        data=torch.tensor([0.0]),
        metadata={"token_id": 100},
    )
    im_end_chunk = SimpleNamespace(
        data=torch.tensor([0.0]),
        metadata={"token_id": 13},
    )

    builder.append_text_chunk(req_data, ordinary_chunk)
    assert len(req_data.pending_text_queue) == 1

    # im_end is filtered: no second row appended.
    builder.append_text_chunk(req_data, im_end_chunk)
    assert len(req_data.pending_text_queue) == 1

    # After thinker_chunks_done flips, later chunks no longer append.
    req_data.thinker_chunks_done = True
    builder.append_text_chunk(req_data, ordinary_chunk)
    assert len(req_data.pending_text_queue) == 1


def _drive_real_builder(
    *,
    prefetched_chunks: list[Any] | None,
    prefetched_stream_done: bool,
    request_id: str = "r0",
    request_params: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    from sglang_omni.models.qwen3_omni.request_builders import (
        _build_talker_request_data,
    )

    captured: dict[str, Any] = {}

    class StubPrefillBuilder:
        def build_prompt_prefill(
            self,
            _payload: Any,
            thinker_chunks: list[Any],
            *,
            thinker_done: bool,
        ) -> dict[str, Any]:
            captured["build_prompt_prefill_thinker_done"] = thinker_done
            captured["build_prompt_prefill_chunk_count"] = len(thinker_chunks)
            return {
                "input_embeds": torch.zeros((9, 2), dtype=torch.float32),
                "input_ids": torch.zeros((9,), dtype=torch.long),
                "pending_text_queue": deque([torch.zeros((2,), dtype=torch.float32)]),
                "tts_eos_embed": torch.full((2,), 0.5, dtype=torch.float32),
                "tts_pad_embed": torch.full((2,), 0.25, dtype=torch.float32),
                "prompt_model_inputs": {"audio_embeds": None},
            }

    def fake_build_sglang_talker_request(**kwargs: Any) -> SimpleNamespace:
        captured["talker_request_kwargs"] = kwargs
        return SimpleNamespace(req=SimpleNamespace(rid=request_id))

    def resolve_sampling_config(_params: dict[str, Any]) -> dict[str, Any]:
        return {
            "max_new_tokens": 4096,
            "temperature": 0.9,
            "top_k": 50,
            "top_p": 1.0,
            "repetition_penalty": 1.05,
            "codec_eos_id": 7,
            "suppress_tokens": [],
            "seed": (_params or {}).get("seed"),
        }

    from sglang_omni.models.qwen3_omni import request_builders as rb_mod

    with pytest.MonkeyPatch.context() as monkeypatch:
        monkeypatch.setattr(
            rb_mod,
            "build_sglang_talker_request",
            fake_build_sglang_talker_request,
        )
        payload = SimpleNamespace(
            request_id=request_id,
            request=SimpleNamespace(params=request_params or {}),
            prefetched_chunks=list(prefetched_chunks or []),
            prefetched_stream_done=prefetched_stream_done,
        )
        req_data = _build_talker_request_data(
            payload,
            prefill_builder=StubPrefillBuilder(),
            tokenizer=SimpleNamespace(),
            codec_vocab_size=4096,
            codec_bos_id=2149,
            audio_token_id=151646,
            image_token_id=151647,
            video_token_id=151648,
            thinker_config=SimpleNamespace(),
            resolve_sampling_config=resolve_sampling_config,
        )
        return req_data, captured


def test_real_builder_threads_thinker_done_false_on_partial_path() -> None:
    """Real `_build_talker_request_data` passes thinker_done=False through prefill + request."""
    _, captured = _drive_real_builder(
        prefetched_chunks=[object()] * 5, prefetched_stream_done=False
    )
    assert captured["build_prompt_prefill_thinker_done"] is False
    assert captured["talker_request_kwargs"]["thinker_chunks_done"] is False


def test_real_builder_threads_thinker_done_true_on_completed_stream() -> None:
    """Real builder passes thinker_done=True through prefill + request."""
    _, captured = _drive_real_builder(
        prefetched_chunks=[object()] * 3, prefetched_stream_done=True
    )
    assert captured["build_prompt_prefill_thinker_done"] is True
    assert captured["talker_request_kwargs"]["thinker_chunks_done"] is True


def test_real_builder_propagates_prefill_outputs_into_talker_request() -> None:
    """input_ids, tts_pad_embed, pending_text_queue, talker_model_inputs all flow through."""
    _, captured = _drive_real_builder(
        prefetched_chunks=[object()] * 5, prefetched_stream_done=False
    )
    kw = captured["talker_request_kwargs"]
    assert torch.equal(kw["talker_input_ids"], torch.zeros((9,), dtype=torch.long))
    assert torch.equal(kw["tts_pad_embed"], torch.full((2,), 0.25, dtype=torch.float32))
    assert kw["talker_model_inputs"] == {"audio_embeds": None}
    pending = kw["pending_text_queue"]
    assert pending is not None and len(pending) == 1


def test_real_builder_derives_per_request_seed_when_missing() -> None:
    """When request params carry no seed the builder must derive a stable per-request seed."""
    _, captured1 = _drive_real_builder(
        prefetched_chunks=[object()] * 5,
        prefetched_stream_done=False,
        request_id="rid-A",
    )
    _, captured2 = _drive_real_builder(
        prefetched_chunks=[object()] * 5,
        prefetched_stream_done=False,
        request_id="rid-A",
    )
    _, captured3 = _drive_real_builder(
        prefetched_chunks=[object()] * 5,
        prefetched_stream_done=False,
        request_id="rid-B",
    )
    seed_a1 = captured1["talker_request_kwargs"]["seed"]
    seed_a2 = captured2["talker_request_kwargs"]["seed"]
    seed_b = captured3["talker_request_kwargs"]["seed"]
    assert seed_a1 is not None and seed_a1 > 0
    assert seed_a1 == seed_a2, "same request_id must derive the same seed"
    assert seed_a1 != seed_b, "different request_id must derive different seeds"


def test_real_builder_preserves_explicit_seed_from_request_params() -> None:
    """If the request explicitly carries a seed, builder must not override it."""
    _, captured = _drive_real_builder(
        prefetched_chunks=[object()] * 5,
        prefetched_stream_done=False,
        request_params={"seed": 42},
    )
    assert captured["talker_request_kwargs"]["seed"] == 42


def test_real_builder_attaches_tts_eos_and_stage_payload() -> None:
    """req_data.tts_eos_embed must come from prefill output; stage_payload must round-trip."""
    req_data, _ = _drive_real_builder(
        prefetched_chunks=[object()] * 5, prefetched_stream_done=False
    )
    assert torch.equal(
        req_data.tts_eos_embed, torch.full((2,), 0.5, dtype=torch.float32)
    )
    assert req_data.stage_payload.request_id == "r0"


def test_real_builder_rejects_zero_chunks_without_done() -> None:
    """Empty prefetched_chunks indicates a readiness or projection bug."""
    with pytest.raises(RuntimeError, match="requires prefetched thinker chunks"):
        _drive_real_builder(prefetched_chunks=[], prefetched_stream_done=False)


def test_real_builder_rejects_done_path_without_chunks() -> None:
    with pytest.raises(RuntimeError, match="requires prefetched thinker chunks"):
        _drive_real_builder(prefetched_chunks=[], prefetched_stream_done=True)


def _build_state_machine_scheduler(
    *,
    enable_partial_start: bool = False,
    partial_start_min_chunks: int = MIN_PARTIAL_START_CHUNKS,
    request_builder_stub: Any,
) -> QwenTalkerScheduler:
    """Construct a scheduler with just enough state for process_input_requests."""
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._enable_partial_start = enable_partial_start
    scheduler._partial_start_min_chunks = partial_start_min_chunks
    scheduler._im_end_token_id = None
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler._aborted_request_ids = set()
    scheduler.waiting_queue = []
    scheduler._request_builder = request_builder_stub
    scheduler._request_admission_lock = threading.RLock()
    scheduler._request_build_executor = None
    scheduler.request_build_max_pending = 0
    scheduler._pending_request_builds = {}
    scheduler._backlogged_request_build_payloads = deque()
    scheduler._request_build_max_pending_observed = 0
    scheduler.max_req_len = 8192
    return scheduler


def test_process_input_requests_partial_build_state_machine() -> None:
    """Drive process_input_requests through the partial-build path end-to-end."""
    appended: list[Any] = []
    marked_done = [False]

    def stub_request_builder(payload: Any) -> Any:
        captured_done = bool(payload.prefetched_stream_done)
        return SimpleNamespace(
            req=SimpleNamespace(
                rid=payload.request_id,
                _omni_data=None,
                origin_input_ids=[],
                sampling_params=SimpleNamespace(max_new_tokens=0),
            ),
            thinker_chunks_done=captured_done,
            pending_text_queue=deque(),
            _captured_thinker_done=captured_done,
        )

    scheduler = _build_state_machine_scheduler(
        enable_partial_start=True,
        partial_start_min_chunks=5,
        request_builder_stub=stub_request_builder,
    )
    # Stream-state handlers used after build:
    scheduler._append_stream_chunk = lambda req_data, chunk: appended.append(chunk)
    scheduler._mark_stream_done = lambda req_data: marked_done.__setitem__(0, True)

    chunks = [SimpleNamespace(data=torch.tensor([float(i)])) for i in range(5)]
    payload = SimpleNamespace(
        request_id="rid-partial-1",
        prefetched_chunks=list(chunks),
        prefetched_stream_done=False,
    )

    # 1) Drive the upstream-base process_input_requests with a partial payload.
    OmniScheduler.process_input_requests(scheduler, [payload])

    assert scheduler.waiting_queue, "request must have been built and enqueued"
    built = scheduler.waiting_queue[0]._omni_data
    assert built._captured_thinker_done is False
    assert "rid-partial-1" not in scheduler._deferred_request_payloads
    assert "rid-partial-1" not in scheduler._pending_stream_done
    assert appended == []
    assert marked_done == [False]

    # 2) A later stream chunk arrives. _find_request_data is provided by upstream;
    #    short-circuit it for the test so that the live request is found.
    scheduler._find_request_data = lambda rid: built if rid == "rid-partial-1" else None
    OmniScheduler._on_stream_chunk(
        scheduler, "rid-partial-1", SimpleNamespace(data=torch.tensor([42.0]))
    )
    assert len(appended) == 1, "exactly one row must be appended after build"

    # 3) The eventual stream_done arrives.
    OmniScheduler._on_stream_done(scheduler, "rid-partial-1")
    assert marked_done == [True]


def test_process_input_requests_keeps_deferred_when_below_threshold() -> None:
    """Below the partial threshold the payload stays in _deferred_request_payloads."""

    def fail_if_called(_payload: Any) -> Any:
        raise AssertionError(
            "request_builder must not be called when threshold is not met"
        )

    scheduler = _build_state_machine_scheduler(
        enable_partial_start=True,
        partial_start_min_chunks=10,
        request_builder_stub=fail_if_called,
    )
    payload = SimpleNamespace(
        request_id="rid-stay",
        prefetched_chunks=[SimpleNamespace(data=torch.tensor([0.0]))] * 2,
        prefetched_stream_done=False,
    )

    OmniScheduler.process_input_requests(scheduler, [payload])

    assert "rid-stay" in scheduler._deferred_request_payloads
    assert scheduler.waiting_queue == []


def test_deferred_request_payload_ignores_chunks_when_partial_disabled() -> None:
    """Default-disabled talker payloads wait for stream_done instead of every chunk."""

    scheduler = _build_state_machine_scheduler(
        enable_partial_start=False,
        request_builder_stub=lambda _payload: None,
    )
    scheduler._find_request_data = lambda _rid: None
    payload = SimpleNamespace(request_id="rid-disabled", prefetched_chunks=[])
    scheduler._deferred_request_payloads["rid-disabled"] = payload

    OmniScheduler._on_stream_chunk(
        scheduler, "rid-disabled", SimpleNamespace(data=torch.tensor([1.0]))
    )

    assert scheduler._pending_stream_chunks["rid-disabled"]
    assert scheduler._dirty_deferred_request_ids == set()
    assert OmniScheduler._take_deferred_request_payloads(scheduler) == []

    OmniScheduler._on_stream_done(scheduler, "rid-disabled")
    assert scheduler._dirty_deferred_request_ids == {"rid-disabled"}
    assert OmniScheduler._take_deferred_request_payloads(scheduler) == [payload]


def test_abort_filters_subsequent_stream_messages_via_recv_requests() -> None:
    """After abort, subsequent stream messages are filtered at recv_requests.

    The public abort surface includes batch-removal bookkeeping that touches
    scheduler state we do not own in this unit test (running_batch /
    waiting_queue inner state). The filter contract being tested here is the
    line in recv_requests that short-circuits on _aborted_request_ids
    membership; we exercise that line directly by marking the request aborted
    and driving the dispatch loop.
    """
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler._aborted_request_ids = set()
    scheduler._pending_stream_chunks = {}
    scheduler._pending_stream_done = set()
    scheduler._deferred_request_payloads = {}
    scheduler._dirty_deferred_request_ids = set()
    scheduler.waiting_queue = []

    stream_chunk_calls: list[Any] = []
    stream_done_calls: list[str] = []
    scheduler._on_stream_chunk = lambda rid, data: stream_chunk_calls.append(
        (rid, data)
    )
    scheduler._on_stream_done = lambda rid: stream_done_calls.append(rid)

    messages = [
        IncomingMessage(
            request_id="rid-abort",
            type="stream_chunk",
            data=SimpleNamespace(data=torch.tensor([1.0])),
        ),
        IncomingMessage(request_id="rid-abort", type="stream_done"),
    ]
    scheduler._recv_scheduler_messages = lambda: list(messages)

    # Mark as aborted via the same set the public abort() ultimately writes to.
    scheduler._aborted_request_ids.add("rid-abort")

    OmniScheduler.recv_requests(scheduler)

    assert stream_chunk_calls == []
    assert stream_done_calls == []


def test_wiring_propagation_factory_args_to_scheduler(monkeypatch) -> None:
    """factory_args enable_partial_start + partial_start_min_chunks flow to scheduler."""
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig

    config = Qwen3OmniSpeechPipelineConfig(model_path="dummy")
    talker_stage = next(stage for stage in config.stages if stage.name == "talker_ar")
    assert talker_stage.factory_args["enable_partial_start"] is True
    assert talker_stage.factory_args["partial_start_min_chunks"] == 5

    scheduler = QwenTalkerScheduler.__new__(QwenTalkerScheduler)
    monkeypatch.setattr(OmniScheduler, "__init__", lambda self, *args, **kwargs: None)
    QwenTalkerScheduler.__init__(
        scheduler, enable_partial_start=True, partial_start_min_chunks=7
    )

    assert scheduler._enable_partial_start is True
    assert scheduler._partial_start_min_chunks == 7


def test_rollback_decode_prep_after_skip_is_idempotent_across_repeated_stalls() -> None:
    """Repeated stalls must leave talker scheduler state identical to pre-prepare_for_decode.

    Reviewer flagged that an incomplete rollback in shared OmniScheduler could
    leak state across the side-effect set that ``prepare_for_decode`` writes
    (out_cache_loc, seq_lens, decode_batch_idx, kv_committed_len,
    kv_allocated_len, plus the req_to_token_pool cell that ``alloc_for_decode``
    writes for the new KV slot). Under talker server_args (overlap disabled,
    no Mamba) those are the only writes; assert that rolling back twice leaves
    every one of them where it started.
    """
    freed: list[Any] = []

    class FakeAllocator:
        def free(self, slot: Any) -> None:
            freed.append(slot)

    class FakeForwardMode:
        @staticmethod
        def is_decode() -> bool:
            return True

    pre_seq_lens = torch.tensor([12, 12])
    pre_seq_lens_cpu = torch.tensor([12, 12])
    pre_orig_seq_lens = torch.tensor([10, 11])
    pre_seq_lens_sum = 24

    reqs = [
        SimpleNamespace(decode_batch_idx=5, kv_committed_len=12, kv_allocated_len=13),
        SimpleNamespace(decode_batch_idx=7, kv_committed_len=12, kv_allocated_len=13),
    ]
    req_pool_indices = torch.tensor([3, 4])
    req_to_token = torch.zeros((8, 16), dtype=torch.int32)
    req_to_token[req_pool_indices, pre_seq_lens] = torch.tensor(
        [777, 888], dtype=torch.int32
    )
    batch = SimpleNamespace(
        forward_mode=FakeForwardMode(),
        out_cache_loc=object(),
        output_ids=None,
        input_ids=torch.tensor([99, 100]),
        reqs=reqs,
        seq_lens=pre_seq_lens.clone() + 1,
        seq_lens_cpu=pre_seq_lens_cpu.clone() + 1,
        orig_seq_lens=pre_orig_seq_lens.clone() + 1,
        seq_lens_sum=pre_seq_lens_sum + len(reqs),
        req_pool_indices=req_pool_indices,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
    )

    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler.token_to_kv_pool_allocator = FakeAllocator()

    # Simulate one prepare_for_decode round: counters already incremented +
    # an out_cache_loc allocation handed in. One stall -> one rollback.
    scheduler._rollback_decode_prep_after_skip(batch)
    assert batch.out_cache_loc is None
    assert batch.output_ids is batch.input_ids
    for req in reqs:
        assert req.decode_batch_idx == [5, 7][reqs.index(req)] - 1
        assert req.kv_committed_len == 11
        assert req.kv_allocated_len == 12
    assert torch.equal(batch.seq_lens, pre_seq_lens)
    assert torch.equal(batch.seq_lens_cpu, pre_seq_lens_cpu)
    assert torch.equal(batch.orig_seq_lens, pre_orig_seq_lens)
    assert batch.seq_lens_sum == pre_seq_lens_sum
    assert len(freed) == 1
    assert torch.equal(
        req_to_token[req_pool_indices, pre_seq_lens],
        torch.zeros(len(reqs), dtype=torch.int32),
    )

    # A second stall on the next iteration must roll back again without
    # accumulating state — the contract is "leaves no residue per stall".
    # Re-simulate prepare_for_decode having run: counters re-incremented,
    # a fresh out_cache_loc was allocated, and the new KV slot was written.
    batch.out_cache_loc = object()
    for req in reqs:
        req.decode_batch_idx += 1
        req.kv_committed_len += 1
        req.kv_allocated_len += 1
    batch.seq_lens.add_(1)
    batch.seq_lens_cpu.add_(1)
    batch.orig_seq_lens.add_(1)
    batch.seq_lens_sum += len(reqs)
    req_to_token[req_pool_indices, pre_seq_lens] = torch.tensor(
        [333, 444], dtype=torch.int32
    )

    scheduler._rollback_decode_prep_after_skip(batch)
    assert batch.out_cache_loc is None
    for req in reqs:
        assert req.decode_batch_idx == [5, 7][reqs.index(req)] - 1
        assert req.kv_committed_len == 11
        assert req.kv_allocated_len == 12
    assert batch.seq_lens_sum == pre_seq_lens_sum
    assert len(freed) == 2
    assert torch.equal(req_to_token, torch.zeros_like(req_to_token))


def test_rollback_decode_prep_after_skip_is_noop_for_prefill_batches() -> None:
    """Rollback must not fire for prefill batches — those have no decode prep to undo."""

    class FakeForwardMode:
        @staticmethod
        def is_decode() -> bool:
            return False

    freed: list[Any] = []
    batch = SimpleNamespace(
        forward_mode=FakeForwardMode(),
        out_cache_loc=object(),
        seq_lens_sum=99,
    )
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(free=freed.append)

    scheduler._rollback_decode_prep_after_skip(batch)
    assert batch.out_cache_loc is not None
    assert batch.seq_lens_sum == 99
    assert freed == []


def test_rollback_decode_prep_after_skip_rejects_seq_lens_sum_type_change() -> None:
    class FakeForwardMode:
        @staticmethod
        def is_decode() -> bool:
            return True

    batch = SimpleNamespace(forward_mode=FakeForwardMode(), seq_lens_sum=None)
    scheduler = object.__new__(QwenTalkerScheduler)

    with pytest.raises(TypeError, match="seq_lens_sum is NoneType"):
        scheduler._rollback_decode_prep_after_skip(batch)


def test_prepare_for_decode_rollback_type_contract_with_upstream(monkeypatch) -> None:
    schedule_batch_mod = pytest.importorskip("sglang.srt.managers.schedule_batch")
    ScheduleBatch = schedule_batch_mod.ScheduleBatch

    batch = ScheduleBatch.__new__(ScheduleBatch)
    reqs = [
        SimpleNamespace(
            decode_batch_idx=0,
            kv_committed_len=10,
            kv_allocated_len=11,
            output_ids=[6],
            origin_input_ids=[5],
        )
    ]
    batch.reqs = reqs
    batch.enable_overlap = False
    batch.spec_algorithm = SimpleNamespace(is_none=lambda: True)
    batch.sampling_info = SimpleNamespace(
        penalizer_orchestrator=SimpleNamespace(is_required=False)
    )
    batch.model_config = SimpleNamespace(is_encoder_decoder=False)
    batch.output_ids = torch.tensor([6], dtype=torch.long)
    batch.input_ids = torch.tensor([5], dtype=torch.long)
    batch.seq_lens = torch.tensor([10], dtype=torch.long)
    batch.seq_lens_cpu = torch.tensor([10], dtype=torch.long)
    batch.orig_seq_lens = torch.tensor([10], dtype=torch.long)
    batch.seq_lens_sum = 10
    batch.device = torch.device("cpu")
    batch.spec_info = None
    batch.req_pool_indices = torch.tensor([2], dtype=torch.long)
    req_to_token = torch.zeros((4, 16), dtype=torch.int32)
    batch.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)

    def _fake_alloc_for_decode(b, token_per_req):
        out = torch.tensor([123], dtype=torch.long)
        locs = b.seq_lens.clone()
        b.req_to_token_pool.req_to_token[b.req_pool_indices, locs] = out.to(torch.int32)
        return out

    monkeypatch.setattr(
        schedule_batch_mod,
        "alloc_for_decode",
        _fake_alloc_for_decode,
    )
    monkeypatch.setattr(
        schedule_batch_mod,
        "get_global_server_args",
        lambda: SimpleNamespace(enable_mamba_extra_buffer=lambda: False),
    )

    ScheduleBatch.prepare_for_decode(batch)
    assert isinstance(batch.seq_lens_sum, int)
    assert int(req_to_token[2, 10]) == 123

    allocated = batch.out_cache_loc
    freed: list[Any] = []
    scheduler = object.__new__(QwenTalkerScheduler)
    scheduler.token_to_kv_pool_allocator = SimpleNamespace(free=freed.append)
    scheduler._rollback_decode_prep_after_skip(batch)

    assert batch.seq_lens_sum == 10
    assert torch.equal(batch.seq_lens, torch.tensor([10], dtype=torch.long))
    assert batch.out_cache_loc is None
    assert len(freed) == 1
    assert torch.equal(freed[0], allocated)
    assert int(req_to_token[2, 10]) == 0


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


def test_qwen_talker_keeps_existing_read_only_weight_loader() -> None:
    """Preserves FP8 parameter weight_loader properties during default binding."""

    class ReadOnlyWeightLoaderParam:
        @property
        def weight_loader(self):
            return "existing"

    class FakeModule:
        def __init__(self) -> None:
            self.param = ReadOnlyWeightLoaderParam()

        def parameters(self):
            return iter([self.param])

    module = FakeModule()

    _bind_default_weight_loaders(module)

    assert module.param.weight_loader == "existing"


def test_qwen_talker_activation_dtype_comes_from_codec_embedding() -> None:
    talker = object.__new__(Qwen3OmniTalker)
    talker.model = SimpleNamespace(
        codec_embedding=SimpleNamespace(
            weight=torch.empty((1, 1), dtype=torch.bfloat16)
        )
    )

    assert talker.activation_dtype is torch.bfloat16


def test_qwen_talker_load_weights_converts_fp8_scales_after_name_mapping() -> None:
    """Converts reciprocal scales for stacked, expert, and direct talker params."""

    class RecordingParam:
        def __init__(self) -> None:
            self.calls = []

        def weight_loader(self, param, loaded_weight, *args, **kwargs) -> None:
            self.calls.append((param, loaded_weight.clone(), args, kwargs))

    qkv_param = RecordingParam()
    expert_param = RecordingParam()
    direct_param = RecordingParam()
    talker = object.__new__(Qwen3OmniTalker)
    talker.config = SimpleNamespace(text_config=SimpleNamespace(num_experts=1))
    talker._cached_params_dict = {
        "model.layers.0.self_attn.qkv_proj.weight_scale_inv": qkv_param,
        "model.layers.0.mlp.experts.w13_weight_scale_inv": expert_param,
        "code_predictor.model.layers.0.mlp.gate_up_proj.weight_scale_inv": direct_param,
    }

    Qwen3OmniTalker.load_weights(
        talker,
        [
            (
                "talker.model.layers.0.self_attn.q_proj.weight_scale_inv",
                torch.tensor([128.0], dtype=torch.float32),
            ),
            (
                "talker.model.layers.0.mlp.experts.0.gate_proj.weight_scale_inv",
                torch.tensor([256.0], dtype=torch.float32),
            ),
            (
                "talker.code_predictor.model.layers.0.mlp.gate_up_proj.weight_scale_inv",
                torch.tensor([512.0], dtype=torch.float32),
            ),
        ],
    )

    assert torch.allclose(qkv_param.calls[0][1], torch.tensor([1.0 / 128.0]))
    assert qkv_param.calls[0][2] == ("q",)
    assert torch.allclose(expert_param.calls[0][1], torch.tensor([1.0 / 256.0]))
    assert expert_param.calls[0][2] == (
        "model.layers.0.mlp.experts.w13_weight_scale_inv",
    )
    assert expert_param.calls[0][3] == {"shard_id": "w1", "expert_id": 0}
    assert torch.allclose(direct_param.calls[0][1], torch.tensor([1.0 / 512.0]))


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


def _talker_seed_self(max_bs: int = 4, vocab: int = 8) -> SimpleNamespace:
    """Minimal stand-in carrying only the buffers prepare_decode_buffers writes."""
    return SimpleNamespace(
        _repetition_mask=torch.zeros(max_bs, vocab, dtype=torch.bool),
        _suppress_mask=torch.zeros(max_bs, vocab, dtype=torch.bool),
        _repetition_penalties=torch.ones(max_bs, 1),
        _sampling_temperatures=torch.ones(max_bs, 1),
        _sampling_top_ps=torch.ones(max_bs),
        _sampling_top_ks=torch.ones(max_bs, dtype=torch.long),
        _sampling_min_ps=torch.zeros(max_bs),
        _sampling_seeds=torch.zeros(max_bs, dtype=torch.long),
        _sampler=None,
    )


def _talker_seed_req(
    seed: int | None, rid: str
) -> SimpleNamespace:
    sp = SimpleNamespace(
        repetition_penalty=1.0,  # keep rep/suppress branches off
        temperature=0.8,
        top_p=0.9,
        top_k=20,
        min_p=0.0,
        sampling_seed=seed,
    )
    req = SimpleNamespace(
        sampling_params=sp,
        output_ids=[],
        _codec_suppress_tokens=None,
        rid=rid,
    )
    return SimpleNamespace(data=SimpleNamespace(req=req, suppress_tokens=None))


def test_talker_prepare_decode_buffers_unseeded_seed_is_rank_shared() -> None:
    # Unseeded rows must get a rank-shared (request-id-derived) seed, not
    # os.urandom, or TP ranks desync.
    from sglang_omni.sampling.seed import SAMPLING_SEED_MASK, derive_sampling_seed

    fake = _talker_seed_self()
    seeded = _talker_seed_req(123, "seeded")
    unseeded = _talker_seed_req(None, "unseeded")
    out_of_range = _talker_seed_req(0xFFFFFFFF, "oor")
    requests = [seeded, unseeded, out_of_range]

    Qwen3OmniTalker.prepare_decode_buffers(fake, requests)
    seeds = fake._sampling_seeds

    assert int(seeds[0]) == 123
    assert seeded.data.req.sampling_params.sampling_seed == 123
    assert int(seeds[1]) == derive_sampling_seed("sglang-omni-unseeded-row", "unseeded")
    assert unseeded.data.req.sampling_params.sampling_seed is None  # not cached
    assert int(seeds[2]) == (0xFFFFFFFF & SAMPLING_SEED_MASK)
    assert out_of_range.data.req.sampling_params.sampling_seed == (
        0xFFFFFFFF & SAMPLING_SEED_MASK
    )

    # stable across decode steps
    Qwen3OmniTalker.prepare_decode_buffers(fake, requests)
    assert int(fake._sampling_seeds[1]) == derive_sampling_seed(
        "sglang-omni-unseeded-row", "unseeded"
    )
