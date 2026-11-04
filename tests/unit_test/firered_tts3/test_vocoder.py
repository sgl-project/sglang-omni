# SPDX-License-Identifier: Apache-2.0
"""Audio output, streaming boundaries, and decoder state through the vocoder lifecycle."""

from __future__ import annotations

import asyncio
from typing import Literal
from unittest.mock import patch

import pytest
import torch

from sglang_omni.models.firered_tts3.redae_decoder import (
    ISTFT,
    RedAEAudioDecoder,
    rematerialize_istft_window,
)
from sglang_omni.models.firered_tts3.vocoder import FireRedTTS3BatchVocoder
from sglang_omni.models.firered_tts3.vocoder_graph import RedAEVocoderGraphRunner
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from tests.unit_test.fixtures.firered_tts3 import (
    audio_tensor,
    make_payload,
    make_streaming_vocoder,
    run_streams,
)


@pytest.mark.parametrize("finish_reason", ["length", None])
def test_batch_decode_trims_prompt_and_preserves_result_metadata(
    decoder: RedAEAudioDecoder, finish_reason: str | None
) -> None:
    generator = torch.Generator().manual_seed(7)
    prompt = torch.randn(1, 8, 64, generator=generator)
    generated = torch.randn(1, 12, 64, generator=generator)
    vocoder = FireRedTTS3BatchVocoder(decoder, redae_scale=1.0, sample_rate=24000)
    payload = make_payload(prompt, generated, finish_reason=finish_reason)
    [result] = asyncio.run(vocoder.decode_payloads([payload]))
    with torch.inference_mode():
        full_audio, _ = decoder.forward_stream(
            torch.cat([prompt, generated], dim=1), None, last_chunk=True
        )
    assert torch.equal(audio_tensor(result), full_audio.flatten()[8 * 960 :])
    assert audio_tensor(result).shape == (12 * 960,)
    assert result.data["sample_rate"] == 24000
    assert result.data["finish_reason"] == finish_reason
    assert result.data["text"] == "voice plan"
    assert result.data["usage"]["completion_tokens"] == 3
    with pytest.raises(RuntimeError, match="no generated latents"):
        vocoder.prepare_item(
            StagePayload(request_id="empty", request=payload.request, data={})
        )


@pytest.mark.parametrize("prompt_predecode", [False, True])
def test_streamed_audio_and_empty_tail_flush_match_full_decode(
    decoder: RedAEAudioDecoder, prompt_predecode: bool
) -> None:
    generator = torch.Generator().manual_seed(11)
    prompt, generated = torch.randn(1, 8, 64, generator=generator), torch.randn(
        1, 28, 64, generator=generator
    )
    batch = FireRedTTS3BatchVocoder(decoder, redae_scale=1.0, sample_rate=24000)
    [expected] = asyncio.run(batch.decode_payloads([make_payload(prompt, generated)]))
    vocoder = make_streaming_vocoder(
        decoder,
        redae_scale=1.0,
        sample_rate=24000,
        prompt_predecode=prompt_predecode,
        stream_batching=False,
    )
    actual = run_streams(vocoder, {"request": prompt}, {"request": generated})[
        "request"
    ]
    torch.testing.assert_close(actual, audio_tensor(expected), rtol=1e-3, atol=1e-3)
    fallback = make_streaming_vocoder(
        decoder, redae_scale=1.0, sample_rate=24000, stream_batching=False
    )
    fallback.handle_streaming_new_request("request", make_payload(prompt, generated))
    fallback.handle_stream_done("request")
    waveform = fallback.outbox.get_nowait()
    assert waveform.type == "stream" and isinstance(waveform.data, dict)
    assert torch.equal(
        torch.frombuffer(
            bytearray(waveform.data["audio_waveform"]), dtype=torch.float32
        ),
        audio_tensor(expected),
    )
    assert fallback.outbox.get_nowait().type == "result"
    assert fallback.stream_states == {}


@pytest.mark.parametrize("request_count", [1, 3])
def test_coalesced_streams_keep_request_audio_and_respect_slot_limit(
    decoder: RedAEAudioDecoder, request_count: int
) -> None:
    prompts, generated = {}, {}
    for index in range(request_count):
        generator = torch.Generator().manual_seed(100 + index)
        prompts[str(index)] = torch.randn(
            1, 12 if index == 2 else 8, 64, generator=generator
        )
        generated[str(index)] = torch.randn(1, 28, 64, generator=generator)
    vocoder = make_streaming_vocoder(
        decoder,
        redae_scale=1.0,
        sample_rate=24000,
        prompt_predecode=False,
        stream_batching=True,
        stream_slots=2,
    )
    with patch.object(
        decoder, "forward_stream", wraps=decoder.forward_stream
    ) as forward:
        actual = run_streams(vocoder, prompts, generated)
        widths = [call.args[0].shape[0] for call in forward.call_args_list]
    assert max(widths) == min(request_count, 2)
    for request_id in prompts:
        serial = make_streaming_vocoder(
            decoder,
            redae_scale=1.0,
            sample_rate=24000,
            prompt_predecode=False,
            stream_batching=False,
        )
        expected = run_streams(
            serial,
            {request_id: prompts[request_id]},
            {request_id: generated[request_id]},
        )[request_id]
        if request_count == 1:
            assert torch.equal(actual[request_id], expected)
        else:
            torch.testing.assert_close(
                actual[request_id], expected, rtol=1e-3, atol=5e-3
            )


def test_stream_rejects_duplicate_prompt_and_nonpositive_context(
    decoder: RedAEAudioDecoder,
) -> None:
    vocoder = make_streaming_vocoder(
        decoder,
        redae_scale=1.0,
        sample_rate=24000,
        prompt_predecode=False,
        stream_batching=False,
    )
    prompt = torch.ones(1, 8, 64)
    metadata = {"stream": True, "prompt_chunk": True, "left_context_size": 8}
    vocoder.handle_stream_chunk("request", StreamItem(0, prompt, "talker", metadata))
    with pytest.raises(RuntimeError, match="second"):
        vocoder.handle_stream_chunk(
            "request", StreamItem(1, prompt, "talker", metadata)
        )
    with pytest.raises(ValueError, match="left_context_size"):
        vocoder.handle_stream_chunk(
            "other",
            StreamItem(0, prompt, "talker", {**metadata, "left_context_size": 0}),
        )


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "mode,prompt_frames,generated_frames,seed",
    [
        ("warm", 8, 64, 29),
        ("restricted", 40, 64, 31),
        ("cold_exact", 28, 64, 90),
        ("cold_padded", 49, 64, 41),
        ("cold_short", 8, 32, 43),
    ],
)
def test_graph_stream_replay_and_fallback_match_eager(
    decoder: RedAEAudioDecoder,
    mode: Literal["warm", "restricted", "cold_exact", "cold_padded", "cold_short"],
    prompt_frames: int,
    generated_frames: int,
    seed: int,
) -> None:
    generator = torch.Generator().manual_seed(seed)
    prompt = torch.randn(1, prompt_frames, 64, generator=generator)
    generated = torch.randn(1, generated_frames, 64, generator=generator)
    eager = make_streaming_vocoder(
        decoder,
        redae_scale=1.0,
        sample_rate=24000,
        prompt_predecode=False,
        stream_batching=False,
    )
    expected = run_streams(eager, {"request": prompt}, {"request": generated})[
        "request"
    ]
    uses_cold_graph = mode.startswith("cold_")
    vocoder = make_streaming_vocoder(
        decoder,
        redae_scale=1.0,
        sample_rate=24000,
        prompt_predecode=False,
        incremental_cuda_graph=mode == "warm",
        cold_cuda_graph=uses_cold_graph,
        cold_cuda_graph_latent_sizes=[32, 64],
        cold_cuda_graph_batch_sizes=[1],
        stream_batching=False,
    )
    if mode == "restricted":
        vocoder.graph_runner = RedAEVocoderGraphRunner(decoder, latent_sizes=(8,))
        vocoder.graph_runner.capture_all()
    else:
        vocoder.warmup_now()
    actual = run_streams(vocoder, {"request": prompt}, {"request": generated})[
        "request"
    ]
    if mode == "cold_padded":
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-3)
    else:
        assert torch.equal(actual, expected)
    graphs = vocoder.cold_graph_runner if uses_cold_graph else vocoder.graph_runner
    stats = graphs.stats()
    if mode == "cold_short":
        assert stats["replays"] == 0 and stats["eager"] > 0
    elif mode == "restricted":
        assert stats["replays"] > 0 and stats["eager"] > 0
    elif uses_cold_graph:
        assert stats["replays"] == 1
    else:
        assert stats["replays"] > 0


@pytest.mark.accelerator
@pytest.mark.parametrize(
    "cold_graph,prompt_frames,seed", [(False, 8, 200), (True, 49, 300)]
)
def test_coalesced_graph_stream_matches_eager(
    decoder: RedAEAudioDecoder, cold_graph: bool, prompt_frames: int, seed: int
) -> None:
    prompts, generated = {}, {}
    for index in range(2):
        generator = torch.Generator().manual_seed(seed + index)
        prompts[str(index)] = torch.randn(1, prompt_frames, 64, generator=generator)
        generated[str(index)] = torch.randn(
            1, (8 if cold_graph else 12) * 4, 64, generator=generator
        )
    eager = make_streaming_vocoder(
        decoder,
        redae_scale=1.0,
        sample_rate=24000,
        prompt_predecode=False,
        stream_batching=True,
    )
    expected = run_streams(eager, prompts, generated)
    vocoder = make_streaming_vocoder(
        decoder,
        redae_scale=1.0,
        sample_rate=24000,
        prompt_predecode=False,
        stream_batching=True,
        incremental_cuda_graph=not cold_graph,
        cold_cuda_graph=cold_graph,
        cold_cuda_graph_latent_sizes=[64],
        cold_cuda_graph_batch_sizes=[2],
    )
    vocoder.warmup_now()
    actual = run_streams(vocoder, prompts, generated)
    for request_id in prompts:
        if cold_graph:
            torch.testing.assert_close(
                actual[request_id], expected[request_id], atol=1e-4, rtol=1e-3
            )
        else:
            assert torch.equal(actual[request_id], expected[request_id])
    graphs = vocoder.cold_graph_runner if cold_graph else vocoder.graph_runner
    assert graphs.stats()["replays"] > 0


@pytest.mark.accelerator
def test_decode_stream_waits_for_chunk_event(
    decoder: RedAEAudioDecoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    chunk = torch.randn(1, 16, 64, generator=torch.Generator().manual_seed(23)).cuda()
    event = torch.cuda.Event()
    event.record()
    vocoder = make_streaming_vocoder(
        decoder,
        redae_scale=1.0,
        sample_rate=24000,
        separate_decode_stream=True,
        stream_batching=False,
    )
    waits: list[tuple[torch.cuda.Stream, torch.cuda.Event]] = []
    original_wait = torch.cuda.Stream.wait_event

    def record_wait(stream: torch.cuda.Stream, ready_event: torch.cuda.Event) -> None:
        waits.append((stream, ready_event))
        original_wait(stream, ready_event)

    monkeypatch.setattr(torch.cuda.Stream, "wait_event", record_wait)
    state = vocoder.create_stream_state("request")
    vocoder.latch_stream_contract(
        "request",
        state,
        {"stream": True, "codes_ready_event": event},
        origin="stream metadata",
    )
    vocoder.ingest("request", state, vocoder.validate_chunk("request", state, chunk))
    actual = vocoder.decode_delta("request", state, is_final=True)
    assert (vocoder.batch.decode_stream, event) in waits
    with torch.inference_mode():
        expected, _ = decoder.forward_stream(chunk, None, last_chunk=True)
    assert torch.equal(actual.flatten(), expected.cpu().flatten())


def test_rematerialize_istft_window_repairs_meta_materialization(
    decoder: RedAEAudioDecoder,
) -> None:
    decoder.to(device="meta").to_empty(device="cpu")
    rematerialize_istft_window(decoder, "cpu")
    assert torch.equal(decoder.istft_head.istft.window, torch.hann_window(1920))


def test_istft_stream_envelope_clamp_keeps_output_finite() -> None:
    """A degenerate (zeroed) window collapses the fold envelope; the clamp
    keeps the normalized output finite instead of emitting NaN/inf."""
    n_fft, hop = 1920, 480
    istft = ISTFT(n_fft=n_fft, hop_length=hop, win_length=n_fft)
    istft.window = torch.zeros_like(istft.window)
    spec = torch.randn(2, n_fft // 2 + 1, 8, dtype=torch.float32)
    spec = torch.view_as_complex(torch.stack([spec, -spec], dim=-1))
    audio, cache = istft.forward_stream(spec, None, last_chunk=False)
    assert torch.isfinite(audio).all()
    assert torch.isfinite(cache).all()
    audio, _ = istft.forward_stream(spec, cache, last_chunk=True)
    assert torch.isfinite(audio).all()
    assert torch.isfinite(istft.flush_stream(cache)).all()


@pytest.mark.parametrize("flush_tail", [False, True])
def test_istft_streaming_preserves_samples_across_chunk_boundaries(
    flush_tail: bool,
) -> None:
    istft = ISTFT(n_fft=1920, hop_length=480, win_length=1920)
    generator = torch.Generator().manual_seed(84)
    spectrum = torch.randn(2, 961, 12, dtype=torch.complex64, generator=generator)
    expected = istft(spectrum)
    cache = None
    chunks = []
    for index, chunk in enumerate(spectrum.split(4, dim=-1)):
        audio, cache = istft.forward_stream(
            chunk, cache, last_chunk=index == 2 and not flush_tail
        )
        chunks.append(audio)
    if flush_tail:
        chunks.append(istft.flush_stream(cache))
    else:
        pass
    assert torch.equal(torch.cat(chunks, dim=-1), expected)
