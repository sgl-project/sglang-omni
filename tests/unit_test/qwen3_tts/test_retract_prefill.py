# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS re-prefill after KV-pressure retraction (#1555)."""

from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.qwen3_tts.model_runner import Qwen3TTSModelRunner


class _TinyModel(torch.nn.Module):
    def __init__(self, hidden: int = 4) -> None:
        super().__init__()
        self.p = torch.nn.Parameter(torch.zeros(hidden))


def _runner() -> Qwen3TTSModelRunner:
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.model = _TinyModel()
    return runner


def _sched_req(
    *,
    prompt: torch.Tensor,
    extend_len: int,
    prefix_len: int = 0,
    history: list[torch.Tensor] | None = None,
    pending_feedback: list[torch.Tensor] | None = None,
    pending_text: list[torch.Tensor] | None = None,
    rid: str = "req",
) -> SimpleNamespace:
    return SimpleNamespace(
        data=SimpleNamespace(
            req=SimpleNamespace(
                rid=rid,
                prefix_indices=list(range(prefix_len)),
                extend_range=SimpleNamespace(length=extend_len),
            ),
            prompt_input_embeds=prompt,
            prefill_input_embeds=prompt,
            decode_input_embeds=list(history or []),
            pending_feedback_queue=deque(pending_feedback or []),
            pending_text_queue=deque(pending_text or []),
            thinker_chunks_done=True,
            tts_pad_embed=torch.zeros(prompt.shape[-1], dtype=prompt.dtype),
        )
    )


def test_write_feedback_buffers_records_decode_input_history() -> None:
    embedding = torch.nn.Embedding(4, 2)
    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.model = SimpleNamespace(
        _decode_feedback_embedding=embedding,
        get_input_embeddings=lambda: embedding,
    )
    sched_req = SimpleNamespace(
        data=SimpleNamespace(
            pending_feedback_queue=deque([torch.tensor([1.0, 2.0])]),
            pending_text_queue=deque([torch.tensor([20.0, 30.0])]),
            decode_input_embeds=[],
            thinker_chunks_done=True,
            tts_pad_embed=torch.zeros(2),
        )
    )
    forward_batch = SimpleNamespace(input_ids=torch.tensor([99], dtype=torch.long))

    runner._write_feedback_buffers(forward_batch, [sched_req])

    assert len(sched_req.data.decode_input_embeds) == 1
    assert torch.equal(
        sched_req.data.decode_input_embeds[0],
        torch.tensor([21.0, 32.0]),
    )
    assert torch.equal(embedding.weight[0].detach(), torch.tensor([21.0, 32.0]))
    assert forward_batch.input_ids.tolist() == [0]
    assert list(sched_req.data.pending_feedback_queue) == []
    assert list(sched_req.data.pending_text_queue) == []


def test_reprefill_after_retract_replays_prompt_plus_generated() -> None:
    # note (Richard Wang): 460 plus 134 is the #1555 594 vs 460 mismatch
    prompt_len, generated_len, hidden = 460, 134, 4
    prompt = torch.arange(prompt_len * hidden, dtype=torch.float32).reshape(
        prompt_len, hidden
    )
    history = [
        torch.full((hidden,), float(1000 + i), dtype=torch.float32)
        for i in range(generated_len - 1)
    ]
    leftover = torch.full((hidden,), 2000.0, dtype=torch.float32)
    extend_len = prompt_len + generated_len
    sched_req = _sched_req(
        prompt=prompt,
        extend_len=extend_len,
        history=history,
        pending_feedback=[leftover],
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(extend_len, dtype=torch.long))

    out = _runner()._build_prefill_input_embeds(forward_batch, [sched_req])

    assert out.shape[0] == extend_len
    assert torch.equal(out[:prompt_len], prompt)
    assert torch.equal(out[prompt_len:-1], torch.stack(history))
    assert torch.equal(out[-1], leftover)
    assert list(sched_req.data.pending_feedback_queue) == []
    assert len(sched_req.data.decode_input_embeds) == generated_len


def test_reprefill_replays_prompt_tail_and_generated_tail() -> None:
    prompt = torch.arange(20, dtype=torch.float32).reshape(10, 2)
    history = [
        torch.tensor([100.0, 101.0]),
        torch.tensor([200.0, 201.0]),
        torch.tensor([300.0, 301.0]),
    ]
    sched_req = _sched_req(prompt=prompt, prefix_len=8, extend_len=5, history=history)
    forward_batch = SimpleNamespace(input_ids=torch.zeros(5, dtype=torch.long))

    out = _runner()._build_prefill_input_embeds(forward_batch, [sched_req])

    expected = torch.cat([prompt[8:10], torch.stack(history)], dim=0)
    assert torch.equal(out, expected)


def test_reprefill_drains_leftover_feedback_when_history_is_short() -> None:
    prompt = torch.arange(8, dtype=torch.float32).reshape(4, 2)
    history = [torch.tensor([10.0, 11.0])]
    sched_req = _sched_req(
        prompt=prompt,
        extend_len=6,
        history=history,
        pending_feedback=[torch.tensor([3.0, 4.0])],
        pending_text=[torch.tensor([30.0, 40.0])],
    )
    forward_batch = SimpleNamespace(input_ids=torch.zeros(6, dtype=torch.long))

    out = _runner()._build_prefill_input_embeds(forward_batch, [sched_req])

    expected = torch.cat(
        [
            prompt,
            torch.stack([torch.tensor([10.0, 11.0]), torch.tensor([33.0, 44.0])]),
        ],
        dim=0,
    )
    assert torch.equal(out, expected)
    assert len(sched_req.data.decode_input_embeds) == 2
    assert list(sched_req.data.pending_feedback_queue) == []
    assert list(sched_req.data.pending_text_queue) == []


def test_reprefill_without_generated_history_fails_loudly() -> None:
    prompt = torch.randn(460, 4)
    sched_req = _sched_req(prompt=prompt, extend_len=594)
    forward_batch = SimpleNamespace(input_ids=torch.zeros(594, dtype=torch.long))

    with pytest.raises(RuntimeError, match="missing feedback/text input embeds"):
        _runner()._build_prefill_input_embeds(forward_batch, [sched_req])


def test_decode_then_retract_reprefill_roundtrip() -> None:
    """N completed decodes leave G=N+1 tokens: history N plus one queued row."""
    hidden, prompt_len, n_decode = 2, 8, 3
    prompt = torch.arange(prompt_len * hidden, dtype=torch.float32).reshape(
        prompt_len, hidden
    )

    class _Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self._decode_feedback_embedding = torch.nn.Embedding(8, hidden)
            self.embed = torch.nn.Embedding(8, hidden)

        def get_input_embeddings(self):
            return self.embed

    runner = Qwen3TTSModelRunner.__new__(Qwen3TTSModelRunner)
    runner.model = _Model()
    generated = n_decode + 1
    sched_req = _sched_req(prompt=prompt, extend_len=prompt_len)
    decode_batch = SimpleNamespace(input_ids=torch.tensor([99], dtype=torch.long))
    for step in range(n_decode):
        sched_req.data.pending_feedback_queue.append(
            torch.full((hidden,), float(step + 1), dtype=torch.float32)
        )
        sched_req.data.pending_text_queue.append(torch.zeros(hidden))
        runner._write_feedback_buffers(decode_batch, [sched_req])

    leftover = torch.full((hidden,), float(generated), dtype=torch.float32)
    sched_req.data.pending_feedback_queue.append(leftover)
    sched_req.data.pending_text_queue.append(torch.zeros(hidden))
    assert len(sched_req.data.decode_input_embeds) == n_decode
    assert len(sched_req.data.pending_feedback_queue) == 1

    sched_req.data.req.prefix_indices = []
    sched_req.data.req.extend_range = SimpleNamespace(length=prompt_len + generated)
    prefill_batch = SimpleNamespace(
        input_ids=torch.zeros(prompt_len + generated, dtype=torch.long)
    )
    out = runner._build_prefill_input_embeds(prefill_batch, [sched_req])

    assert out.shape[0] == prompt_len + generated
    assert torch.equal(out[:prompt_len], prompt)
    assert torch.equal(
        out[prompt_len:-1],
        torch.stack([torch.full((hidden,), float(i + 1)) for i in range(n_decode)]),
    )
    assert torch.equal(out[-1], leftover)
    assert list(sched_req.data.pending_feedback_queue) == []
    assert len(sched_req.data.decode_input_embeds) == generated


def test_fresh_prefill_still_uses_prompt_only_buffer() -> None:
    prompt = torch.arange(12, dtype=torch.float32).reshape(6, 2)
    sched_req = _sched_req(prompt=prompt, prefix_len=1, extend_len=4)
    forward_batch = SimpleNamespace(input_ids=torch.zeros(4, dtype=torch.long))

    out = _runner()._build_prefill_input_embeds(forward_batch, [sched_req])

    assert torch.equal(out, prompt[1:5])
