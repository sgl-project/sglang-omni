# SPDX-License-Identifier: Apache-2.0
"""Pipeline-level unit tests for csm_tts (CPU, no GPU, no sglang engine —
CPU-only sglang stub; the fixtures double as upstream API-drift
tripwires perR5).


"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.csm_tts.audio_codec import (
    CsmMimiCodec,
    sanitize_for_mimi,
    trim_at_first_all_zero_frame,
)
from sglang_omni.models.csm_tts.config import CsmTtsPipelineConfig
from sglang_omni.models.csm_tts.text_tokenizer import (
    AUDIO_EOS_TOKEN_ID,
    AUDIO_TOKEN_ID,
    BOS,
    EOT,
    CsmPromptBuilder,
)
from sglang_omni.models.csm_tts.vocoder_scheduler import CsmStreamingVocoderScheduler
from sglang_omni.pipeline.stage.stream_queue import StreamItem

SPF = CsmMimiCodec.SAMPLES_PER_FRAME  # 1920
_STREAM_META = {
    "modality": "audio_codes",
    "stream": True,
    "num_codebooks": 32,
    "codebook_size": 2051,
}


def test_registry_smoke() -> None:
    """``PIPELINE_CONFIG_REGISTRY.get_config("CsmForConditionalGeneration")``
    resolves to ``CsmTtsPipelineConfig`` via the pkgutil scan — i.e. the
    csm_tts ``__init__``/``config`` import chain stays sglang/CUDA-free
    (contract; a silent registry skip would fail this)."""
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    config_cls = PIPELINE_CONFIG_REGISTRY.get_config("CsmForConditionalGeneration")
    assert config_cls is CsmTtsPipelineConfig


def test_config_topology() -> None:
    """Stage topology: next= chain preprocessing→audio_encoder→tts_engine→
    vocoder, ``tts_engine.stream_to=["vocoder"]``, vocoder ``terminal=True``
    + ``can_accept_stream_before_payload=True``, all ``process="pipeline"``
    (schema demands exactly one of next/terminal per stage)."""
    config = CsmTtsPipelineConfig(model_path="fake-model")
    stages = {stage.name: stage for stage in config.stages}

    assert list(stages) == ["preprocessing", "audio_encoder", "tts_engine", "vocoder"]
    assert stages["preprocessing"].next == "audio_encoder"
    assert stages["audio_encoder"].next == "tts_engine"
    assert stages["tts_engine"].next == "vocoder"
    assert stages["tts_engine"].stream_to == ["vocoder"]
    assert stages["vocoder"].terminal is True
    assert stages["vocoder"].can_accept_stream_before_payload is True
    assert all(stage.process == "pipeline" for stage in stages.values())
    assert CsmTtsPipelineConfig.architecture == "CsmForConditionalGeneration"


class _CharTokenizer:
    """1 token per char, ids far outside the special range — structural
    prompt-layout checks without the checkpoint tokenizer."""

    def encode(self, text: str, add_special_tokens: bool = True) -> list[int]:
        assert add_special_tokens is False  # hard requirement
        return [200_000 + ord(c) for c in text]


def test_prompt_builder_layout_matches_hub_chat_template() -> None:
    """Structural layout per context message ``128000 ⊕ tok("[spk]text") ⊕
    128001 ⊕ 128002×F ⊕ 128003`` and final message without audio framing;
    spans cover exactly the 128002 runs. (Token-for-token golden fixtures
    against the real checkpoint tokenizer are part of the container gate —
    parity_csm_hf.py asserts prompt-token equality vs the HF processor.)"""
    builder = CsmPromptBuilder.__new__(CsmPromptBuilder)
    builder._tok = _CharTokenizer()

    ids, spans = builder.build_prompt(
        text="ok",
        speaker_id=1,
        context=[
            {"speaker_id": 0, "text": "ab"},
            {"speaker_id": 1, "text": "c"},
        ],
        context_frame_counts=[3, 2],
    )
    seg0 = builder.tokenize_segment_text(0, "ab")  # literal "[0]ab"
    seg1 = builder.tokenize_segment_text(1, "c")
    final = builder.tokenize_segment_text(1, "ok")
    expected = (
        [BOS]
        + seg0
        + [EOT]
        + [AUDIO_TOKEN_ID] * 3
        + [AUDIO_EOS_TOKEN_ID]
        + [BOS]
        + seg1
        + [EOT]
        + [AUDIO_TOKEN_ID] * 2
        + [AUDIO_EOS_TOKEN_ID]
        + [BOS]
        + final
        + [EOT]
    )
    assert ids == expected
    assert spans == [
        (1 + len(seg0) + 1, 3),
        (len(seg0) + len(seg1) + 8, 2),
    ]
    for start, length in spans:
        assert ids[start : start + length] == [AUDIO_TOKEN_ID] * length
        assert ids[start + length] == AUDIO_EOS_TOKEN_ID

    ids2, spans2 = builder.build_prompt(text="ok", speaker_id=0)
    assert ids2 == [BOS] + builder.tokenize_segment_text(0, "ok") + [EOT]
    assert spans2 == []

    with pytest.raises(ValueError):
        builder.build_prompt(
            text="x",
            speaker_id=0,
            context=[{"speaker_id": 0, "text": "a"}],
            context_frame_counts=[0],  # every non-final message carries audio
        )
    with pytest.raises(ValueError):
        builder.build_prompt(
            text="x",
            speaker_id=0,
            context=[{"speaker_id": 0, "text": "a"}],  # missing frame counts
        )


class _MockMimi:
    """Identity-traceable decoder: a frame with codes[:, 0] == v decodes to
    1920 samples all equal to float(v) — every emitted sample is traceable to
    its source frame."""

    SAMPLE_RATE = CsmMimiCodec.SAMPLE_RATE
    SAMPLES_PER_FRAME = SPF

    def __init__(self) -> None:
        self.calls: list[torch.Tensor] = []

    def decode(self, codes_F32: torch.Tensor) -> torch.Tensor:
        assert codes_F32.ndim == 2 and codes_F32.shape[1] == 32
        assert (
            int(codes_F32.min()) >= 0 and int(codes_F32.max()) <= 2047
        ), "OOB code reached Mimi — fence #2 failed"
        self.calls.append(codes_F32.clone())
        ids = codes_F32[:, 0].to(torch.float32)
        return ids.repeat_interleave(SPF).reshape(1, -1)

    def decode_batch(self, items: list[torch.Tensor]) -> list[torch.Tensor]:
        return [self.decode(item) for item in items]


def _row(value: int) -> torch.Tensor:
    return torch.full((32,), value, dtype=torch.long)


def _chunk_samples(message) -> np.ndarray:
    data = message.data
    assert data["modality"] == "audio"
    assert data["sample_rate"] == 24000
    samples = np.frombuffer(data["audio_waveform"], dtype=np.float32)
    assert samples.size % SPF == 0, f"chunk not 1920-aligned: {samples.size}"
    return samples


def _run_stream(sched, rid, rows, extra_meta=None) -> list[np.ndarray]:
    meta = dict(_STREAM_META)
    if extra_meta:
        meta.update(extra_meta)
    chunks: list[np.ndarray] = []
    for i, row in enumerate(rows):
        msgs = sched.on_stream_chunk(rid, StreamItem(i, row, "tts_engine", meta))
        chunks.extend(_chunk_samples(m) for m in msgs)
    return chunks


def _assert_gapless(chunks, ids) -> None:
    got = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    want = np.repeat(np.asarray(list(ids), dtype=np.float32), SPF)
    assert got.size == want.size, f"sample count {got.size} != {want.size}"
    assert np.array_equal(got, want), "emitted samples not gapless/in-order"


def test_vocoder_framing_math_default_strides() -> None:
    """Delay-free framing at 12.5 Hz: first chunk at stride=13
    rows emits 13-2(holdback)=11 frames, steady chunks of followup=6, the
    overlap window is re-decoded then trimmed, final flush is gapless."""
    mock = _MockMimi()
    sched = CsmStreamingVocoderScheduler(mock)
    rid = "req-default"
    n_rows = 26
    chunks = _run_stream(sched, rid, [_row(i + 1) for i in range(n_rows)])
    state = sched._stream_states[rid]

    assert chunks and chunks[0].size == 11 * SPF
    assert chunks[1].size == 6 * SPF and chunks[2].size == 6 * SPF
    # 2nd decode window starts at emitted - overlap = 11 - 4 → frame id 8...
    assert int(mock.calls[1][0, 0]) == 8
    # ...but the re-decoded overlap is trimmed: chunk 2 starts at id 12.
    assert chunks[1][0] == 12.0

    final = sched._decode_delta(rid, state, final=True)
    chunks.extend(_chunk_samples(m) for m in final)
    assert state.frames_emitted == n_rows
    _assert_gapless(chunks, range(1, n_rows + 1))

    sched.clear_stream_state(rid)
    assert rid not in sched._stream_states


def test_vocoder_initial_chunk_knob_ttfa_path() -> None:
    """``initial_codec_chunk_frames=1`` → exactly 1 frame after row 1 (the
    TTFA knob), then gapless through the final flush."""
    mock = _MockMimi()
    sched = CsmStreamingVocoderScheduler(mock)
    rid = "req-initial"
    chunks = _run_stream(
        sched,
        rid,
        [_row(i + 1) for i in range(20)],
        extra_meta={"initial_codec_chunk_frames": 1},
    )
    state = sched._stream_states[rid]
    assert state.initial_codec_chunk_frames == 1
    assert chunks[0].size == 1 * SPF and chunks[0][0] == 1.0
    assert int(mock.calls[0].shape[0]) == 1

    final = sched._decode_delta(rid, state, final=True)
    chunks.extend(_chunk_samples(m) for m in final)
    _assert_gapless(chunks, range(1, 21))


def test_vocoder_trim_and_clamp_fences() -> None:
    """Fences #2/#3: the final flush trims at the first
    all-32-zero frame (tail never reaches Mimi); a cb31!=0 EOS frame is KEPT
    by the trim (HF stop(31)/trim(32) split); OOB codes 2048-2050 are counted
    then clamped."""
    # trim helper semantics
    frames = torch.stack([_row(3), _row(7), _row(0), _row(9)])
    assert trim_at_first_all_zero_frame(frames).shape[0] == 2
    eos_cb31 = torch.tensor([[0] * 31 + [5]], dtype=torch.long)
    assert trim_at_first_all_zero_frame(eos_cb31).shape[0] == 1
    assert CsmStreamingVocoderScheduler._frames_to_codes([[0] * 32]) is None
    kept = CsmStreamingVocoderScheduler._frames_to_codes([[0] * 31 + [5]])
    assert kept is not None and kept.shape[0] == 1

    # sanitize counts then clamps; valid codes untouched
    bad = torch.tensor([[2048, 2049, 2050, 5, 0] + [1] * 27], dtype=torch.long)
    clamped, count = sanitize_for_mimi(bad.clone())
    assert count == 3
    assert int(clamped.max()) <= 2047 and int(clamped.min()) >= 0
    assert clamped[0, 3] == 5 and clamped[0, 4] == 0

    # streamed: all-zero frame cuts the final flush; bad row clamp counted
    mock = _MockMimi()
    sched = CsmStreamingVocoderScheduler(mock)
    rid = "req-trim"
    rows = [_row(i + 1) for i in range(24)] + [_row(0), _row(99)]
    chunks = _run_stream(sched, rid, rows)
    state = sched._stream_states[rid]
    final = sched._decode_delta(rid, state, final=True)
    chunks.extend(_chunk_samples(m) for m in final)
    assert state.frames_emitted == 24
    _assert_gapless(chunks, range(1, 25))

    mock = _MockMimi()
    sched = CsmStreamingVocoderScheduler(mock)
    rid = "req-clamp"
    rows = [_row(i + 1) for i in range(12)]
    bad_row = _row(7)
    bad_row[0:3] = torch.tensor([2048, 2049, 2050])
    rows.append(bad_row)  # 13th row triggers the first decode (bad row held back)
    _run_stream(sched, rid, rows)
    state = sched._stream_states[rid]
    assert state.clamp_count == 0  # held-back row not decoded mid-stream
    sched._decode_delta(rid, state, final=True)
    assert state.clamp_count == 3
    assert sched._clamp_count_total == 3


def _overlay_fixture():
    """Runner + fake model with traceable embeddings for the prefill overlay."""
    from sglang_omni.models.csm_tts.model_runner import CsmTTSModelRunner

    dim = 4

    def embed_tokens(ids: torch.Tensor) -> torch.Tensor:
        return ids.to(torch.float32).unsqueeze(-1).repeat(1, dim)

    def frame_embedding(codes: torch.Tensor) -> torch.Tensor:
        # traceable: negative sum of the frame's codes
        return -codes.to(torch.float32).sum(dim=-1, keepdim=True).repeat(1, dim)

    fake_model = SimpleNamespace(
        backbone=SimpleNamespace(model=SimpleNamespace(embed_tokens=embed_tokens)),
        frame_embedding=frame_embedding,
    )
    tp_worker = SimpleNamespace(
        gpu_id=0, model_runner=SimpleNamespace(model=fake_model)
    )
    runner = CsmTTSModelRunner(tp_worker, output_processor=None)
    return runner, frame_embedding, embed_tokens


def test_prefill_overlay_paste_positions() -> None:
    """Overlay pastes context-frame embeds at real-id 128002 positions, the
    all-zeros-frame embed at 128003 (HF ``_merge_input_ids_with_input_values``
    parity), leaves text rows on the backbone embedding, and returns None for
    batches with no context audio."""
    runner, frame_embedding, embed_tokens = _overlay_fixture()

    codes = torch.randint(1, 2048, (3, 32), dtype=torch.long)
    prompt = (
        [BOS, 200_065, EOT]
        + [AUDIO_TOKEN_ID] * 3
        + [AUDIO_EOS_TOKEN_ID]
        + [BOS, 200_066, EOT]
    )
    input_ids = torch.tensor(prompt, dtype=torch.long)
    data = SimpleNamespace(
        req=SimpleNamespace(extend_input_len=len(prompt)),
        context_codes=[codes],
        num_ctx_codes_consumed=0,
    )
    forward_batch = SimpleNamespace(input_ids=input_ids)
    requests = [SimpleNamespace(request_id="r0", data=data)]

    embeds = runner._build_prefill_input_embeds(forward_batch, requests)
    assert embeds is not None and embeds.shape == (len(prompt), 4)

    expected_paste = frame_embedding(codes)
    assert torch.equal(embeds[3:6], expected_paste)  # 128002 run
    eos_embed = frame_embedding(torch.zeros(1, 32, dtype=torch.long))
    assert torch.equal(embeds[6], eos_embed[0])  # 128003 → all-zeros frame
    text_positions = [0, 1, 2, 7, 8, 9]
    assert torch.equal(embeds[text_positions], embed_tokens(input_ids[text_positions]))
    assert data.num_ctx_codes_consumed == 3

    # no context audio in the batch → None (backbone embedding is exact)
    data_no_ctx = SimpleNamespace(
        req=SimpleNamespace(extend_input_len=3),
        context_codes=None,
        num_ctx_codes_consumed=0,
    )
    none_embeds = runner._build_prefill_input_embeds(
        SimpleNamespace(input_ids=torch.tensor([BOS, 200_065, EOT])),
        [SimpleNamespace(request_id="r1", data=data_no_ctx)],
    )
    assert none_embeds is None


def test_prefill_overlay_chunked_cursor() -> None:
    """``num_ctx_codes_consumed`` keeps the paste chunked-prefill-correct:
    a placeholder run split across two extend chunks consumes the code matrix
    sequentially; overrunning the codes raises (off-by-one
    class)."""
    runner, frame_embedding, _ = _overlay_fixture()

    codes = torch.randint(1, 2048, (5, 32), dtype=torch.long)
    data = SimpleNamespace(
        req=SimpleNamespace(extend_input_len=3),
        context_codes=[codes],
        num_ctx_codes_consumed=0,
    )

    # chunk 1: first 3 placeholders
    chunk1 = torch.tensor([AUDIO_TOKEN_ID] * 3, dtype=torch.long)
    embeds1 = runner._build_prefill_input_embeds(
        SimpleNamespace(input_ids=chunk1),
        [SimpleNamespace(request_id="r0", data=data)],
    )
    assert torch.equal(embeds1, frame_embedding(codes[:3]))
    assert data.num_ctx_codes_consumed == 3

    # chunk 2: remaining 2 placeholders + audio_eos
    chunk2 = torch.tensor([AUDIO_TOKEN_ID] * 2 + [AUDIO_EOS_TOKEN_ID], dtype=torch.long)
    embeds2 = runner._build_prefill_input_embeds(
        SimpleNamespace(input_ids=chunk2),
        [SimpleNamespace(request_id="r0", data=data)],
    )
    assert torch.equal(embeds2[:2], frame_embedding(codes[3:5]))
    assert data.num_ctx_codes_consumed == 5

    # a further placeholder would overrun the code matrix → loud error
    data.req.extend_input_len = 1
    with pytest.raises(ValueError, match="overruns context codes"):
        runner._build_prefill_input_embeds(
            SimpleNamespace(input_ids=torch.tensor([AUDIO_TOKEN_ID])),
            [SimpleNamespace(request_id="r0", data=data)],
        )

    # placeholders without any context codes → loud error (prompt mismatch)
    data_missing = SimpleNamespace(
        req=SimpleNamespace(extend_input_len=1),
        context_codes=[],
        num_ctx_codes_consumed=0,
    )
    # NB: the batch must contain at least one request WITH context codes for
    # the overlay to run at all; pair the broken request with a healthy one.
    healthy = SimpleNamespace(
        req=SimpleNamespace(extend_input_len=1),
        context_codes=[codes],
        num_ctx_codes_consumed=0,
    )
    ids = torch.tensor([AUDIO_TOKEN_ID, AUDIO_TOKEN_ID], dtype=torch.long)
    with pytest.raises(ValueError, match="no context_codes"):
        runner._build_prefill_input_embeds(
            SimpleNamespace(input_ids=ids),
            [
                SimpleNamespace(request_id="r-h", data=healthy),
                SimpleNamespace(request_id="r-m", data=data_missing),
            ],
        )
