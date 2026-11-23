# SPDX-License-Identifier: Apache-2.0
"""Stable Audio 3: request contract, official layout values, batch invariance, fp16 audio transport and checkpoint detection."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn
from transformers import BatchEncoding

from sglang_omni.client.audio import encode_audio, to_numpy
from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateChunk
from sglang_omni.models.stable_audio3.autoencoder import AudioDecoder
from sglang_omni.models.stable_audio3.conditioning import Conditioner
from sglang_omni.models.stable_audio3.cuda_graphs import (
    MAX_GRAPH_TOKENS,
    bucket_batch,
    bucket_latents,
)
from sglang_omni.models.stable_audio3.dit import DiffusionTransformer
from sglang_omni.models.stable_audio3.model_config import StableAudio3Config
from sglang_omni.models.stable_audio3.request_builders import (
    DEFAULT_DURATION_SECONDS,
    build_state,
)
from sglang_omni.models.stable_audio3.runtime import (
    GenerationRequest,
    StableAudio3Runtime,
    sampling_schedule,
)
from sglang_omni.models.stable_audio3.speech_validation import (
    StableAudio3SpeechRequestValidator,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.protocol import CreateSpeechRequest
from sglang_omni.serve.speech_errors import SpeechAPIError
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.hf import try_resolve_arch_from_stable_audio3_layout

PROMPT_TOKENS = 6
TEXT_DIM = 12


def payload_with(params: dict[str, object]) -> StagePayload:
    return StagePayload(
        request_id="request",
        request=OmniRequest(inputs="ambient pads", params=params, metadata={}),
        data={},
    )


def test_request_defaults_to_the_official_duration_and_a_random_seed() -> None:
    first, second = (build_state(payload_with({})) for _ in range(2))
    assert first.duration_seconds == DEFAULT_DURATION_SECONDS
    assert first.seed != second.seed
    state = build_state(
        payload_with(
            {"seed": 7, "stage_params": {"generation": {"duration_seconds": 30}}}
        )
    )
    assert (state.prompt, state.duration_seconds, state.seed) == ("ambient pads", 30, 7)


@pytest.mark.parametrize(
    ("fields", "param"),
    [
        ({"voice": "alloy"}, "voice"),
        ({"stream": True, "response_format": "wav"}, "stream"),
        ({"seed": 2**63}, "seed"),
        ({"stage_params": {"vocoder": {}}}, "stage_params"),
        (
            {"stage_params": {"generation": {"duration_seconds": 381}}},
            "stage_params.generation.duration_seconds",
        ),
        (
            {"stage_params": {"generation": {"duration_seconds": 0}}},
            "stage_params.generation.duration_seconds",
        ),
        (
            {"stage_params": {"generation": {"steps": 4}}},
            "stage_params.generation.steps",
        ),
    ],
)
def test_unsupported_requests_are_client_errors(
    fields: dict[str, object], param: str
) -> None:
    validator = StableAudio3SpeechRequestValidator(default_model="stable-audio-3")
    with pytest.raises(SpeechAPIError) as error:
        validator.prepare_generation_updates(
            CreateSpeechRequest(input="ambient pads", **fields)
        )
    assert (error.value.status_code, error.value.param) == (400, param)


def tiny_config() -> StableAudio3Config:
    """The Medium layout with tiny widths; the sample geometry matches the release."""
    return StableAudio3Config.model_validate(
        {
            "model_type": "diffusion_cond_inpaint",
            "sample_size": 16777216,
            "sample_rate": 44100,
            "audio_channels": 2,
            "model": {
                "pretransform": {
                    "type": "autoencoder",
                    "chunked": True,
                    "config": {
                        "pretransform": {
                            "type": "patched",
                            "config": {"patch_size": 256, "channels": 2},
                        },
                        "encoder": {"config": {"strides": [16]}},
                        "decoder": {
                            "type": "taae_v2",
                            "config": {
                                "out_channels": 512,
                                "channels": 8,
                                "c_mults": [4],
                                "strides": [16],
                                "latent_dim": 8,
                                "transformer_depths": [2],
                                "sinusoidal_blocks": [2],
                                "sliding_window": [1, 1],
                                "dim_heads": 16,
                                "differential": True,
                                "variable_stride": True,
                                "mask_noise": 0.1,
                            },
                        },
                        "bottleneck": {
                            "type": "softnorm",
                            "config": {
                                "noise_augment_dim": 0,
                                "noise_regularize": True,
                                "auto_scale": True,
                            },
                        },
                        "latent_dim": 8,
                        "downsampling_ratio": 4096,
                    },
                },
                "conditioning": {
                    "configs": [
                        {
                            "id": "prompt",
                            "type": "t5gemma",
                            "config": {
                                "max_length": PROMPT_TOKENS,
                                "padding_mode": "learned",
                                "subfolder": "t5gemma-b-b-ul2",
                            },
                        },
                        {
                            "id": "seconds_total",
                            "type": "number",
                            "config": {
                                "min_val": 0,
                                "max_val": 384,
                                "fourier_features_type": "expo",
                            },
                        },
                    ],
                    "cond_dim": TEXT_DIM,
                },
                "diffusion": {
                    "type": "dit",
                    "diffusion_objective": "rf_denoiser",
                    "config": {
                        "io_channels": 8,
                        "embed_dim": 64,
                        "depth": 2,
                        "num_heads": 1,
                        "cond_token_dim": TEXT_DIM,
                        "global_cond_dim": TEXT_DIM,
                        "local_add_cond_dim": 9,
                        "num_memory_tokens": 4,
                        "global_cond_type": "adaLN",
                        "timestep_features_type": "expo",
                        "norm_type": "rms_norm",
                        "attn_kwargs": {"qk_norm": "rms", "differential": True},
                        "norm_kwargs": {"force_fp32": True},
                        "ff_kwargs": {"mult": 2.0},
                    },
                },
            },
        }
    )


class CharacterTokenizer:
    def __call__(
        self,
        prompts: list[str],
        *,
        truncation: bool,
        max_length: int,
        padding: str,
        return_tensors: str,
    ) -> BatchEncoding:
        input_ids = torch.zeros(len(prompts), max_length, dtype=torch.long)
        attention_mask = torch.zeros_like(input_ids)
        for row, prompt in enumerate(prompts):
            codes = [ord(character) % 64 for character in prompt[:max_length]]
            input_ids[row, : len(codes)] = torch.tensor(codes)
            attention_mask[row, : len(codes)] = 1
        return BatchEncoding({"input_ids": input_ids, "attention_mask": attention_mask})


class EmbeddingTextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(64, TEXT_DIM)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> BatchEncoding:
        return BatchEncoding({"last_hidden_state": self.embedding(input_ids)})


@pytest.fixture(scope="module")
def runtime() -> StableAudio3Runtime:
    torch.manual_seed(0)
    config = tiny_config()
    modules = [
        DiffusionTransformer(config.model.diffusion.config, backend="sdpa"),
        Conditioner(config.model.conditioning),
        AudioDecoder(config.model.pretransform.config, backend="sdpa"),
    ]
    for module in modules:
        module.eval()
    dit, conditioner, decoder = modules
    return StableAudio3Runtime(
        config=config,
        dit=dit,
        conditioner=conditioner,
        decoder=decoder,
        tokenizer=CharacterTokenizer(),
        text_encoder=EmbeddingTextEncoder().eval(),
        device=torch.device("cpu"),
    )


def test_layout_values_match_the_official_implementation(
    runtime: StableAudio3Runtime,
) -> None:
    """Latent lengths, valid lengths and noise levels read from the official Medium inference."""
    assert [
        runtime.latent_length(seconds) for seconds in (4.0, 5.0, 20.0, 30.0, 380.0)
    ] == [
        108,
        120,
        280,
        388,
        4096,
    ]
    assert [runtime.valid_length(5.0, 120), runtime.valid_length(30.0, 388)] == [
        118,
        387,
    ]
    assert torch.allclose(
        sampling_schedule(torch.device("cpu")),
        torch.tensor(
            [1.0, 0.9944, 0.9845, 0.9579, 0.8909, 0.7455, 0.5125, 0.2739, 0.0]
        ),
        atol=1e-4,
    )


def test_batching_does_not_change_any_request(runtime: StableAudio3Runtime) -> None:
    """A short request padded next to a long one, decoded in chunks, matches its own single run."""
    short = GenerationRequest(prompt="lofi", seconds=4.0, seed=3)
    long = GenerationRequest(prompt="orchestral swell", seconds=20.0, seed=4)
    batched = runtime.generate([short, long])
    singles = [runtime.generate([request])[0] for request in (short, long)]
    for batched_audio, single_audio, request in zip(
        batched, singles, (short, long), strict=True
    ):
        assert batched_audio.shape == (2, int(request.seconds * 44100))
        assert batched_audio.abs().max() > 0
        assert torch.allclose(batched_audio, single_audio, atol=1e-6)


@pytest.mark.parametrize(
    ("batch_size", "max_batch_size", "bucket"),
    [(1, 8, 1), (3, 8, 4), (8, 8, 8), (3, 3, 3), (12, 16, 16), (5, 4, None)],
)
def test_batch_buckets_stay_within_the_stage_batch(
    batch_size: int, max_batch_size: int, bucket: int | None
) -> None:
    assert bucket_batch(batch_size, max_batch_size) == bucket


def test_latent_buckets_cover_every_length_within_the_graph_budget() -> None:
    for latent_length in range(1, MAX_GRAPH_TOKENS):
        bucket = bucket_latents(latent_length)
        assert latent_length <= bucket < latent_length + 32
        assert bucket % 32 == 0


def test_float16_waveform_reaches_the_client_as_the_same_audio() -> None:
    """The fp16 decoder output travels at half the bytes; the client widens it to the same float32 values and WAV."""
    waveform = (torch.rand(2, 4410) * 2.4 - 1.2).half().clamp(-1, 1)
    payload = audio_waveform_payload(
        waveform, sample_rate=44100, keep_channels=True, keep_float16=True
    )
    chunk = GenerateChunk(request_id="stable-audio-3")
    Client.set_audio_data(chunk, payload)
    widened = waveform.float().numpy()
    assert payload["audio_waveform_dtype"] == "float16"
    assert len(payload["audio_waveform"]) == waveform.numel() * 2
    assert np.array_equal(to_numpy(chunk.audio_data), widened)
    assert (
        encode_audio(chunk.audio_data, sample_rate=44100).audio_bytes
        == encode_audio(widened, sample_rate=44100).audio_bytes
    )


@pytest.mark.parametrize(
    ("model_type", "prompt_encoder", "architecture"),
    [
        ("diffusion_cond_inpaint", "t5gemma", "StableAudio3ForConditionalGeneration"),
        ("diffusion_cond", "t5", None),
    ],
)
def test_checkpoint_layout_detection(
    tmp_path: Path, model_type: str, prompt_encoder: str, architecture: str | None
) -> None:
    (tmp_path / "model_config.json").write_text(
        json.dumps(
            {
                "model_type": model_type,
                "model": {"conditioning": {"configs": [{"type": prompt_encoder}]}},
            }
        )
    )
    assert try_resolve_arch_from_stable_audio3_layout(str(tmp_path)) == architecture
