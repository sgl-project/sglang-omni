# SPDX-License-Identifier: Apache-2.0
"""Shared decoder and streaming vocoder test fixtures."""

from __future__ import annotations

from typing import TypedDict

import pytest
import torch
from typing_extensions import Unpack

from sglang_omni.models.firered_tts3.config import FireRedTTS3PipelineConfig
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.redae_decoder import RedAEAudioDecoder
from sglang_omni.models.firered_tts3.vocoder import FireRedTTS3StreamingVocoder
from sglang_omni.models.firered_tts3.weights import drop_embedding_tables
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage


@pytest.fixture
def decoder(request: pytest.FixtureRequest) -> RedAEAudioDecoder:
    uses_cuda = request.node.get_closest_marker("accelerator") is not None
    if uses_cuda and not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    else:
        pass
    torch.manual_seed(1234)
    module = RedAEAudioDecoder(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
    )
    drop_embedding_tables(module)
    return module.to(device="cuda" if uses_cuda else "cpu").eval()


def make_payload(
    prompt: torch.Tensor,
    generated: torch.Tensor,
    request_id: str = "request",
    finish_reason: str | None = "length",
) -> StagePayload:
    state = FireRedTTS3State(
        prompt_latents=prompt,
        generated_latents=generated,
        left_context_size=prompt.shape[1],
        finish_reason=finish_reason,
        gen_text="voice plan",
        completion_tokens=generated.shape[1] // 4,
    )
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={"text": "hello"}, params={"stream": True}),
        data=state.to_dict(),
    )


def audio_tensor(payload: StagePayload) -> torch.Tensor:
    assert payload.data["audio_waveform_dtype"] == "float32"
    return torch.frombuffer(
        bytearray(payload.data["audio_waveform"]), dtype=torch.float32
    )


def run_streams(
    vocoder: FireRedTTS3StreamingVocoder,
    prompts: dict[str, torch.Tensor],
    generated: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    for request_id, prompt in prompts.items():
        vocoder.handle_streaming_new_request(
            request_id, make_payload(prompt, generated[request_id], request_id)
        )
    chunk_id = 0
    cursor = 0
    while cursor < next(iter(generated.values())).shape[1]:
        messages = []
        for request_id, prompt in prompts.items():
            is_prompt = chunk_id == 0
            frames = (4, 8, 16)[min(max(chunk_id - 1, 0), 2)]
            chunk = (
                prompt
                if is_prompt
                else generated[request_id][:, cursor : cursor + frames]
            )
            metadata = {"modality": "audio_latents", "stream": True}
            if is_prompt:
                metadata.update(prompt_chunk=True, left_context_size=prompt.shape[1])
            else:
                pass
            messages.append(
                IncomingMessage(
                    request_id,
                    "stream_chunk",
                    StreamItem(chunk_id, chunk, "talker", metadata),
                )
            )
        if vocoder.can_batch_stream_chunks:
            vocoder.handle_stream_chunk_batch(messages)
        else:
            for message in messages:
                vocoder.handle_stream_chunk(message.request_id, message.data)
        if chunk_id > 0:
            cursor += frames
        else:
            assert vocoder.outbox.empty()
        chunk_id += 1
    states = list(vocoder.stream_states.values())
    for request_id in prompts:
        vocoder.handle_stream_done(request_id)
    waveforms: dict[str, list[torch.Tensor]] = {
        request_id: [] for request_id in prompts
    }
    completed: set[str] = set()
    while not vocoder.outbox.empty():
        message = vocoder.outbox.get_nowait()
        if message.type == "stream":
            assert message.request_id not in completed
            assert isinstance(message.data, dict)
            assert message.metadata == {"modality": "audio"}
            waveforms[message.request_id].append(
                torch.frombuffer(
                    bytearray(message.data["audio_waveform"]), dtype=torch.float32
                )
            )
        else:
            assert message.type == "result", message.data
            assert isinstance(message.data, StagePayload)
            assert message.data.data["finish_reason"] == "length"
            assert message.data.data["text"] == "voice plan"
            assert (
                message.data.data["usage"]["completion_tokens"]
                == generated[message.request_id].shape[1] // 4
            )
            completed.add(message.request_id)
    assert completed == set(prompts)
    assert vocoder.stream_states == {} and vocoder.stream_payloads == {}
    assert all(
        state.decoder_state is None and state.pending_latents == [] for state in states
    )
    return {request_id: torch.cat(chunks) for request_id, chunks in waveforms.items()}


class StreamingVocoderOverrides(TypedDict, total=False):
    max_batch_size: int
    max_batch_wait_ms: float
    stream_slots: int
    prompt_predecode: bool
    stream_batching: bool
    separate_decode_stream: bool
    incremental_cuda_graph: bool
    cuda_graph_latent_sizes: list[int] | None
    cuda_graph_batch_sizes: list[int] | None
    cold_cuda_graph: bool
    cold_cuda_graph_latent_sizes: list[int] | None
    cold_cuda_graph_batch_sizes: list[int] | None


def make_streaming_vocoder(
    decoder: RedAEAudioDecoder,
    *,
    redae_scale: float,
    sample_rate: int,
    **overrides: Unpack[StreamingVocoderOverrides],
) -> FireRedTTS3StreamingVocoder:
    configuration = FireRedTTS3PipelineConfig(model_path="model")
    factory_parameters = configuration.stage_factory_kwargs("vocoder")
    factory_parameters.pop("warmup_on_init")
    return FireRedTTS3StreamingVocoder(
        decoder,
        redae_scale=redae_scale,
        sample_rate=sample_rate,
        **{**factory_parameters, **overrides},
    )
