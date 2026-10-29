# SPDX-License-Identifier: Apache-2.0
"""Registration, request validation and batched decoding for Parakeet TDT."""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import (
    ParakeetEncoderConfig,
    ParakeetFeatureExtractor,
    ParakeetForTDT,
    ParakeetProcessor,
    ParakeetTDTConfig,
    ParakeetTokenizer,
)

from sglang_omni.models.parakeet_tdt.config import (
    MAX_NATIVE_CLIP_SECONDS,
    ParakeetTDTPipelineConfig,
)
from sglang_omni.models.parakeet_tdt.stages import SAMPLE_RATE, ParakeetTDTTranscriber
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.serve.openai_errors import is_bad_request_error

WORDS = ["<unk>", "<pad>", "alpha", "beta", "gamma", "delta"]
BLANK_TOKEN_ID = len(WORDS)
PAD_TOKEN_ID = 1
MEL_BINS = 16
MAX_AUDIO_SECONDS = 6.0


@pytest.fixture(scope="module")
def tiny_checkpoint(tmp_path_factory: pytest.TempPathFactory) -> Path:
    checkpoint = tmp_path_factory.mktemp("parakeet_tdt")
    tokenizer = ParakeetTokenizer(
        tokenizer_object=Tokenizer(
            WordLevel({word: i for i, word in enumerate(WORDS)}, unk_token="<unk>")
        ),
        unk_token="<unk>",
        pad_token="<pad>",
    )
    ParakeetProcessor(
        feature_extractor=ParakeetFeatureExtractor(feature_size=MEL_BINS),
        tokenizer=tokenizer,
    ).save_pretrained(checkpoint)
    encoder_config = ParakeetEncoderConfig(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_mel_bins=MEL_BINS,
        subsampling_conv_channels=8,
    )
    torch.manual_seed(0)
    model = ParakeetForTDT(
        ParakeetTDTConfig(
            encoder_config=encoder_config.to_dict(),
            vocab_size=len(WORDS) + 1,
            blank_token_id=BLANK_TOKEN_ID,
            pad_token_id=PAD_TOKEN_ID,
            decoder_hidden_size=16,
            num_decoder_layers=1,
            durations=[0, 1, 2, 3, 4],
        )
    )
    model.generation_config.decoder_start_token_id = BLANK_TOKEN_ID
    model.generation_config.pad_token_id = PAD_TOKEN_ID
    model.save_pretrained(checkpoint)
    return checkpoint


@pytest.fixture(scope="module")
def transcriber(tiny_checkpoint: Path) -> ParakeetTDTTranscriber:
    return ParakeetTDTTranscriber(
        str(tiny_checkpoint),
        device=torch.device("cpu"),
        dtype=torch.float32,
        max_audio_seconds=MAX_AUDIO_SECONDS,
    )


def noise_wav(duration_s: float, seed: int) -> bytes:
    samples = 0.1 * np.random.default_rng(seed).standard_normal(
        int(duration_s * SAMPLE_RATE)
    )
    stream = io.BytesIO()
    sf.write(
        stream, samples.astype(np.float32), SAMPLE_RATE, format="WAV", subtype="FLOAT"
    )
    return stream.getvalue()


def make_payload(
    request_id: str, audio_bytes: bytes, params: dict[str, object] | None = None
) -> StagePayload:
    request = OmniRequest(
        inputs={"audio_bytes": audio_bytes},
        params={"temperature": 0.0, **(params or {})},
    )
    return StagePayload(request_id=request_id, request=request, data={})


def test_checkpoint_architecture_resolves_to_parakeet_tdt_pipeline() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("ParakeetForTDT")
        is ParakeetTDTPipelineConfig
    )


def test_server_chunks_fit_the_stage_audio_limit() -> None:
    config = ParakeetTDTPipelineConfig(
        model_path="ConwayResearch/Underdog-Bark-3.0-Pro"
    )
    chunking = config.resolved_audio_chunking
    assert chunking.allow_audio_chunking is True
    assert chunking.max_native_clip_s == MAX_NATIVE_CLIP_SECONDS
    assert config.stages[0].factory.max_audio_seconds == MAX_NATIVE_CLIP_SECONDS


@pytest.mark.parametrize(
    "params",
    [
        {"prompt": "cars"},
        {"temperature": 0.7},
        {"max_new_tokens": 5},
        {"repetition_penalty": 1.1},
        {"task": "translate"},
    ],
)
def test_unsupported_parameters_fail_as_bad_requests(
    transcriber: ParakeetTDTTranscriber, params: dict[str, object]
) -> None:
    [result] = transcriber.transcribe_batch(
        [make_payload("r", noise_wav(1.0, 0), params)]
    )
    assert isinstance(result, ValueError)
    assert is_bad_request_error(result)


def test_audio_over_the_limit_fails_as_bad_request(
    transcriber: ParakeetTDTTranscriber,
) -> None:
    [result] = transcriber.transcribe_batch(
        [make_payload("r", noise_wav(MAX_AUDIO_SECONDS + 1.0, 0))]
    )
    assert isinstance(result, ValueError)
    assert is_bad_request_error(result)


def test_batched_requests_match_solo_and_fail_only_invalid_ones(
    transcriber: ParakeetTDTTranscriber,
) -> None:
    """A short clip batched with a long one must not decode into its padding."""
    short_audio = noise_wav(0.5, 1)
    long_audio = noise_wav(4.0, 2)
    solo_short = transcriber.transcribe(make_payload("solo-short", short_audio)).data
    solo_long = transcriber.transcribe(make_payload("solo-long", long_audio)).data
    results = transcriber.transcribe_batch(
        [
            make_payload("short", short_audio),
            make_payload("corrupt", b"not audio"),
            make_payload("long", long_audio, {"language": "en"}),
        ]
    )
    assert solo_short["text"]
    assert [result.data for result in results if isinstance(result, StagePayload)] == [
        solo_short,
        solo_long,
    ]
    assert isinstance(results[1], ValueError)
    assert is_bad_request_error(results[1])
