# SPDX-License-Identifier: Apache-2.0
"""Incremental AudioVAE output and request lifecycle contracts."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Literal
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch

import sglang_omni.platforms as platforms
from sglang_omni.config.runtime import resolve_stage_factory_kwargs
from sglang_omni.models.ming3_tts import stages as engine_module
from sglang_omni.models.ming3_tts import vocoder as factory_module
from sglang_omni.models.ming3_tts.config import Ming3TTSPipelineConfig, default_stages
from sglang_omni.models.ming3_tts.model_runner import Ming3TTSRequestData
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


@pytest.mark.parametrize(
    "dense,vocoder_device,stream,vocoder_factory,platform_device_type",
    [
        (False, "cpu", True, "current", "cuda"),
        (False, "cpu", True, "legacy", "cuda"),
        (True, "cpu", True, "current", "cuda"),
        (True, "cpu", True, "legacy", "cuda"),
        (True, "cpu", False, "current", "cuda"),
        (True, "cpu", False, "legacy", "cuda"),
        (False, None, True, "current", "cuda"),
        (False, None, True, "legacy", "cuda"),
        (True, None, True, "current", "cuda"),
        (True, None, True, "legacy", "cuda"),
        (True, None, True, "current", "xpu"),
        (True, None, True, "legacy", "xpu"),
        (True, None, True, "custom", "cuda"),
    ],
)
@torch.inference_mode()
def test_gpu_generation_retains_full_latents_for_cpu_vocoder(
    vocoder: Ming3StreamingVocoder,
    monkeypatch: pytest.MonkeyPatch,
    dense: bool,
    vocoder_device: str | None,
    stream: bool,
    vocoder_factory: Literal["current", "legacy", "custom"],
    platform_device_type: str,
) -> None:
    """GPU generation dropped full latents for stream=true even with a CPU vocoder."""
    monkeypatch.setattr(
        platforms, "current_platform", SimpleNamespace(device_type=platform_device_type)
    )
    stages = default_stages(dense)
    stages[-1].factory.device = vocoder_device
    if vocoder_factory == "legacy":
        stages[-1].factory_path = (
            "sglang_omni.models.ming3_tts.stages.create_vocoder_executor"
        )
    elif vocoder_factory == "custom":
        stages[-1].factory_path = "test.custom_vocoder"
    else:
        pass
    config = Ming3TTSPipelineConfig(model_path="checkpoint", stages=stages, mps="off")
    engine_kwargs = resolve_stage_factory_kwargs(
        config.stage_named("tts_engine"), config
    )
    checkpoint_config = {
        "model_type": "dense" if dense else "tiny",
        "audio_tokenizer_config": {"enc_kwargs": {"latent_dim": 4}},
        "tts_config": {
            "prompt_format": "legacy_antmmf",
            "with_rml": True,
            "conditioning_mode": "speaker_only",
            "decode_alignment": "old_shift",
            "patch_size": 1,
            "history_size": 1,
            "latent_dim": 4,
            "speaker_embedding_dim": 4,
            "sampling_temperature": 1.5,
            "stop_threshold": 0.5,
            "minimum_decode_steps": 1,
            "maximum_decode_steps": 32,
            "audio_patch_token_id": 1,
            "audio_start_token_id": 2,
            "audio_end_token_id": 3,
            "speaker_start_token_id": 4,
            "speaker_end_token_id": 5,
        },
    }
    scheduler_factory = Mock()
    with monkeypatch.context() as engine_patches:
        engine_patches.setattr(
            engine_module,
            "resolve_concrete_device",
            lambda *args: torch.device("cuda", 0),
        )
        engine_patches.setattr(engine_module, "resolve_checkpoint", lambda path: path)
        engine_patches.setattr(
            engine_module, "load_checkpoint_config", lambda path: checkpoint_config
        )
        engine_patches.setattr(
            engine_module,
            "build_sglang_server_args",
            lambda *args, **kwargs: SimpleNamespace(max_running_requests=32),
        )
        engine_patches.setattr(
            engine_module,
            "create_sglang_infrastructure",
            lambda *args, **kwargs: (None, None, None, None, None),
        )
        engine_patches.setattr(
            engine_module, "get_tokenizer", lambda *args, **kwargs: None
        )
        engine_patches.setattr(engine_module, "Ming3TTSModelRunner", Mock())
        engine_patches.setattr(engine_module, "OmniScheduler", scheduler_factory)
        engine_patches.setattr(
            torch,
            "get_device_module",
            lambda device: SimpleNamespace(
                Stream=lambda **kwargs: None, stream=lambda stream: nullcontext()
            ),
        )
        engine_module.create_tts_engine_executor(
            "checkpoint",
            max_seq_len=4096,
            enable_acoustic_graph=False,
            **engine_kwargs,
        )
        generated_latents = torch.randn(
            5, 1, 4, generator=torch.Generator().manual_seed(19)
        )
        request = Ming3TTSRequestData(
            stage_payload=StagePayload(
                "cpu-fallback", OmniRequest("text", params={"stream": stream}), {}
            ),
            input_ids=torch.tensor([1, 2]),
            output_ids=[3] * 5,
            speaker_embedding=torch.zeros(4),
            speaker_position=0,
            sampling_temperature=0.0,
            noise_generator=torch.Generator(),
            latent_history=torch.zeros(1, 4),
            latents=list(generated_latents),
            finish_reason="length",
        )
        result = scheduler_factory.call_args.kwargs["result_adapter"](request)
    streaming_builder = scheduler_factory.call_args.kwargs["stream_output_builder"]
    if vocoder_device is None:
        assert result.data["latents"] is None
        assert streaming_builder is engine_module.build_stream_output
    else:
        assert streaming_builder is None
        torch.testing.assert_close(result.data["latents"], generated_latents)
        monkeypatch.setattr(factory_module, "resolve_checkpoint", lambda path: ".")
        monkeypatch.setattr(
            factory_module, "load_checkpoint_config", lambda path: checkpoint_config
        )
        cpu_vae = vocoder.vae.to(dtype=torch.bfloat16)
        monkeypatch.setattr(
            factory_module, "load_audio_vae", lambda *args, **kwargs: (cpu_vae, 44100)
        )
        cpu_scheduler = factory_module.create_vocoder_executor(
            "checkpoint", device="cpu"
        )
        loop = asyncio.new_event_loop()
        try:
            cpu_scheduler.run_single(
                IncomingMessage(
                    request_id="cpu-fallback", type="new_request", data=result
                ),
                loop,
            )
        finally:
            loop.close()
        response = next(
            message.data
            for message in cpu_scheduler.outbox.queue
            if message.type == "result"
        )
        waveform, _, _ = cpu_vae.decode(
            generated_latents.reshape(1, -1, 4).to(torch.bfloat16)
        )
        actual = torch.from_numpy(
            np.frombuffer(response.data["audio_waveform"], dtype=np.float32).copy()
        )
        torch.testing.assert_close(
            actual, waveform.float().reshape(-1), atol=1e-5, rtol=1e-4
        )
        assert response.data["finish_reason"] == "length"
        assert response.data["usage"] == {"prompt_tokens": 2, "completion_tokens": 5}
