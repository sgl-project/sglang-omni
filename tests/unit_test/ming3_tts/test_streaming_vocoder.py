# SPDX-License-Identifier: Apache-2.0
"""Incremental AudioVAE output and request lifecycle contracts."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest
import torch

from sglang_omni.models.ming3_tts import vocoder as factory_module
from sglang_omni.models.ming3_tts.streaming_vocoder import Ming3StreamingVocoder
from sglang_omni.modules.ming_audio.vae.configuration_audio_vae import AudioVAEconfig
from sglang_omni.modules.ming_audio.vae.modeling_audio_vae import AudioVAE
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from sglang_omni.utils.audio_payload import audio_waveform_payload


@pytest.fixture
def vocoder() -> Ming3StreamingVocoder:
    backbone = {
        "_attn_implementation": "sdpa",
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "num_hidden_layers": 1,
        "max_position_embeddings": 512,
        "sliding_window": 64,
        "use_sliding_window": True,
        "max_window_layers": 0,
        "vocab_size": 1,
    }
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        vae = AudioVAE(
            AudioVAEconfig(
                sample_rate=44100,
                patch_size=4,
                enc_kwargs={
                    "backbone": {**backbone, "num_hidden_layers": 4},
                    "input_dim": 4,
                    "latent_dim": 4,
                },
                dec_kwargs={
                    "backbone": backbone,
                    "output_dim": 4,
                    "latent_dim": 4,
                },
            )
        ).eval()

    @torch.inference_mode()
    def decode(payload: StagePayload) -> StagePayload:
        waveform, _, _ = vae.decode(payload.data["latents"].reshape(1, -1, 4))
        payload.data = audio_waveform_payload(waveform, sample_rate=44100)
        return payload

    return Ming3StreamingVocoder(
        decode, vae=vae, sample_rate=44100, stream_chunk_steps=8, stream_capacity=2
    )


@pytest.mark.parametrize("latent_steps", [1, 5, 8, 9, 16, 17, 39, 66])
@torch.inference_mode()
def test_incremental_audio_matches_full_decode_and_flushes_tail(
    vocoder: Ming3StreamingVocoder, latent_steps: int
) -> None:
    """Chunk boundaries must neither drop interpolation/ISTFT tails nor repeat audio."""
    latents = torch.randn(
        latent_steps, 1, 4, generator=torch.Generator().manual_seed(8)
    )
    expected, _, _ = vocoder.vae.decode(latents.reshape(1, -1, 4))
    payload = StagePayload(
        "speech",
        OmniRequest("text", params={"stream": True}),
        {"usage": {"completion_tokens": latent_steps}, "finish_reason": "stop"},
    )
    messages = []
    for chunk_id, latent in enumerate(latents):
        vocoder.on_stream_chunk_batch(
            [
                (
                    "speech",
                    StreamItem(
                        chunk_id,
                        latent,
                        "tts_engine",
                        {
                            "modality": "audio_latents",
                            "stream": True,
                            "is_last": chunk_id == latent_steps - 1,
                        },
                    ),
                )
            ]
        )
    if latent_steps > 16:
        assert list(vocoder.outbox.queue)
    else:
        pass
    vocoder.stream_payloads["speech"] = payload
    terminal_messages = vocoder.on_stream_done("speech")
    messages.extend(list(vocoder.outbox.queue))
    messages.extend(terminal_messages)
    actual = torch.from_numpy(
        np.concatenate(
            [
                np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
                for message in messages
                if message.type == "stream"
            ]
        )
    )
    torch.testing.assert_close(actual, expected.reshape(-1), atol=1e-5, rtol=1e-4)
    assert messages[-1].type == "result"
    assert messages[-1].data.data["usage"] == payload.data["usage"]
    assert messages[-1].data.data["finish_reason"] == "stop"
    assert "audio_waveform" not in messages[-1].data.data


@torch.inference_mode()
def test_abort_discards_cache_and_rejects_late_latents(
    vocoder: Ming3StreamingVocoder,
) -> None:
    """An aborted request cannot retain its decoder cache or create it from late chunks."""
    for chunk_id in range(17):
        vocoder.on_stream_chunk_batch(
            [
                (
                    "aborted",
                    StreamItem(
                        chunk_id,
                        torch.ones(1, 4),
                        "tts_engine",
                        {"stream": True, "is_last": False},
                    ),
                )
            ]
        )
    assert vocoder.slot_bindings.slot_for("aborted") is not None
    vocoder.abort("aborted")
    assert "aborted" not in vocoder.stream_states
    vocoder.on_stream_chunk_batch(
        [
            (
                "aborted",
                StreamItem(
                    17,
                    torch.ones(1, 4),
                    "tts_engine",
                    {"stream": True, "is_last": False},
                ),
            )
        ]
    )
    assert "aborted" not in vocoder.stream_states
    vocoder.drain_pending_releases()
    assert vocoder.slot_bindings.slot_for("aborted") is None


def test_reordered_chunks_fail_before_decoding(vocoder: Ming3StreamingVocoder) -> None:
    vocoder.on_stream_chunk_batch(
        [
            (
                "speech",
                StreamItem(
                    1,
                    torch.zeros(1, 4),
                    "tts_engine",
                    {"stream": True, "is_last": False},
                ),
            )
        ]
    )
    messages = list(vocoder.outbox.queue)
    assert any(message.type == "error" for message in messages)
    assert "speech" not in vocoder.stream_states


@torch.inference_mode()
def test_interleaved_streams_finish_after_late_payload_and_release_state(
    vocoder: Ming3StreamingVocoder,
) -> None:
    """Independent decoder caches must survive stream-done arriving before each payload."""
    sequences = {
        "first": torch.randn(33, 1, 4, generator=torch.Generator().manual_seed(1)),
        "second": torch.randn(33, 1, 4, generator=torch.Generator().manual_seed(2)),
    }
    for chunk_id in range(33):
        for request_id, latents in sequences.items():
            vocoder.handle_stream_chunk(
                request_id,
                StreamItem(
                    chunk_id,
                    latents[chunk_id],
                    "tts_engine",
                    {"stream": True, "is_last": chunk_id == 32},
                ),
            )
    for request_id in sequences:
        vocoder.handle_stream_done(request_id)
        assert request_id in vocoder.stream_states
        payload = StagePayload(
            request_id,
            OmniRequest("text", params={"stream": True}),
            {"usage": {}, "finish_reason": "length"},
        )
        vocoder.handle_new_request_batch(
            [IncomingMessage(request_id=request_id, type="new_request", data=payload)]
        )
        assert request_id not in vocoder.stream_states
        assert request_id not in vocoder.stream_payloads
    messages = list(vocoder.outbox.queue)
    for request_id, latents in sequences.items():
        expected, _, _ = vocoder.vae.decode(latents.reshape(1, -1, 4))
        waveform = np.concatenate(
            [
                np.frombuffer(message.data["audio_waveform"], dtype=np.float32)
                for message in messages
                if message.request_id == request_id and message.type == "stream"
            ]
        )
        torch.testing.assert_close(
            torch.from_numpy(waveform), expected.reshape(-1), atol=1e-5, rtol=1e-4
        )
        assert (
            len(
                [
                    message
                    for message in messages
                    if message.request_id == request_id and message.type == "result"
                ]
            )
            == 1
        )


@torch.inference_mode()
def test_ready_streams_share_one_decoder_step(vocoder: Ming3StreamingVocoder) -> None:
    """Ready requests must use one decoder call with isolated cache rows."""
    for chunk_id in range(8):
        vocoder.on_stream_chunk_batch(
            [
                (
                    request_id,
                    StreamItem(
                        chunk_id,
                        torch.ones(1, 4) * value,
                        "tts_engine",
                        {"stream": True},
                    ),
                )
                for request_id, value in [("one", 1), ("two", 2)]
            ]
        )
    with patch.object(
        vocoder.decoder, "run_streaming", wraps=vocoder.decoder.run_streaming
    ) as decode:
        vocoder.on_stream_chunk_batch(
            [
                (
                    request_id,
                    StreamItem(
                        8, torch.ones(1, 4) * value, "tts_engine", {"stream": True}
                    ),
                )
                for request_id, value in [("one", 1), ("two", 2)]
            ]
        )
    assert decode.call_count == 1
    assert len(decode.call_args.kwargs["slot_ids"]) == 2
    assert len(set(decode.call_args.kwargs["slot_ids"])) == 2


@torch.inference_mode()
def test_first_packet_waits_for_lookahead(vocoder: Ming3StreamingVocoder) -> None:
    """The eight-latent default emits its first non-empty packet at latent seventeen."""
    for chunk_id in range(16):
        vocoder.handle_stream_chunk(
            "speech",
            StreamItem(chunk_id, torch.ones(1, 4), "tts_engine", {"stream": True}),
        )
    assert vocoder.outbox.empty()
    vocoder.handle_stream_chunk(
        "speech", StreamItem(16, torch.ones(1, 4), "tts_engine", {"stream": True})
    )
    assert any(message.type == "stream" for message in vocoder.outbox.queue)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA graph routing requires CUDA"
)
@torch.inference_mode()
def test_nonstream_keeps_graph_dispatch_and_stream_uses_cache(
    vocoder: Ming3StreamingVocoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Streaming integration must not replace the non-streaming graph dispatcher."""

    monkeypatch.setattr(factory_module, "resolve_checkpoint", lambda model: ".")
    monkeypatch.setattr(
        factory_module,
        "load_checkpoint_config",
        lambda checkpoint: {"model_type": "dense"},
    )
    monkeypatch.setattr(
        factory_module, "load_audio_vae", lambda *args, **kwargs: (vocoder.vae, 44100)
    )
    vocoder.vae.to(device="cuda", dtype=torch.bfloat16)
    original = factory_module.DecoderGraphs.run
    with patch.object(
        factory_module.DecoderGraphs, "run", autospec=True, side_effect=original
    ) as full_decode:
        scheduler = factory_module.create_vocoder_executor(
            "model", graph_latent_steps=8, device="cuda"
        )
        latents = torch.randn(17, 1, 4, generator=torch.Generator().manual_seed(3))
        payload = StagePayload(
            "full",
            OmniRequest("text", params={"stream": False}),
            {"latents": latents, "usage": {}, "finish_reason": "length"},
        )
        scheduler.handle_new_request_batch(
            [IncomingMessage(request_id="full", type="new_request", data=payload)]
        )
        result = next(
            message.data
            for message in scheduler.outbox.queue
            if message.type == "result"
        )
        assert result.data["finish_reason"] == "length"
        assert full_decode.call_count == 1
        for index, latent in enumerate(latents):
            scheduler.handle_stream_chunk(
                "stream", StreamItem(index, latent, "tts_engine", {"stream": True})
            )
        scheduler.stream_payloads["stream"] = StagePayload(
            "stream",
            OmniRequest("text", params={"stream": True}),
            {"usage": {}, "finish_reason": "stop"},
        )
        scheduler.handle_stream_done("stream")
        assert full_decode.call_count == 1
        assert "stream" not in scheduler.stream_states


@torch.inference_mode()
def test_cpu_factory_preserves_full_decode(
    vocoder: Ming3StreamingVocoder, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The fixed-slot streaming backend must not remove CPU full-decoder fallback."""
    monkeypatch.setattr(factory_module, "resolve_checkpoint", lambda model: ".")
    monkeypatch.setattr(
        factory_module,
        "load_checkpoint_config",
        lambda checkpoint: {"model_type": "dense"},
    )
    vocoder.vae.bfloat16()
    monkeypatch.setattr(
        factory_module, "load_audio_vae", lambda *args, **kwargs: (vocoder.vae, 44100)
    )
    scheduler = factory_module.create_vocoder_executor(
        "model", graph_latent_steps=8, device="cpu"
    )
    payload = StagePayload(
        "cpu",
        OmniRequest("text", params={"stream": False}),
        {"latents": torch.ones(5, 1, 4), "usage": {}, "finish_reason": "stop"},
    )
    result = scheduler.fn(payload)
    assert result.data["finish_reason"] == "stop"
    assert result.data["audio_waveform"]
