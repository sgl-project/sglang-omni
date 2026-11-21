# SPDX-License-Identifier: Apache-2.0
"""Shared FireRedTTS3 preprocessing request builders for unit tests."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from unittest.mock import patch

import torch
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import FireRedTTS3State
from sglang_omni.models.firered_tts3.preprocessing import (
    preprocess_firered_tts3_payload,
)
from sglang_omni.models.firered_tts3.ref_encode import (
    FireRedReferenceBundle,
    FireRedReferenceEncoder,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.utils.json import JsonValue
from tests.unit_test.fixtures.firered_tts3 import make_speaker_tokenizer


class ReferenceLatentEncoder(torch.nn.Module):
    """Keep RedAE's sample-to-frame boundary while decoding real test audio."""

    def __init__(self, config: FireRedTTS3Config) -> None:
        super().__init__()
        self.samples_per_latent_frame: int = (
            config.audio_patch_size * config.enc_extra_downsample_rate
        )
        self.latent_channels: int = config.redae_dim

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        return (
            waveform.reshape(1, -1, self.samples_per_latent_frame)
            .mean(dim=-1)
            .unsqueeze(-1)
            .expand(-1, -1, self.latent_channels)
        )


@contextmanager
def make_reference_encoder(
    config: FireRedTTS3Config,
) -> Iterator[FireRedReferenceEncoder]:
    with patch(
        "sglang_omni.models.firered_tts3.ref_encode.load_redae_encoder",
        return_value=ReferenceLatentEncoder(config),
    ):
        bundle = FireRedReferenceBundle("unused", config, "cpu")
    encoder = FireRedReferenceEncoder(
        bundle,
        model_id="checkpoint",
        max_batch_size=1,
        max_batch_wait_ms=0,
        cache_enabled=True,
    )
    try:
        yield encoder
    finally:
        encoder.close()


def make_preprocessing_payload(
    *,
    inputs: dict[str, JsonValue] | None = None,
    params: dict[str, JsonValue] | None = None,
    tts_params: dict[str, JsonValue] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id="rid",
        request=OmniRequest(
            inputs=inputs
            or {
                "text": "欢迎使用点点语音合成系统。",
                "references": [{"audio_path": "ref.wav", "text": "今天天气很好。"}],
            },
            params=params or {},
            metadata={"tts_params": tts_params or {}},
        ),
        data={},
    )


def preprocess_payload(
    payload: StagePayload,
    *,
    config: FireRedTTS3Config | None = None,
    tokenizer: PreTrainedTokenizerFast | None = None,
    n_timesteps: int = 10,
    max_patches: int = 400,
) -> FireRedTTS3State:
    preprocess_firered_tts3_payload(
        payload,
        tokenizer=tokenizer or make_speaker_tokenizer(),
        config=config or FireRedTTS3Config(spk_in_dim=512, dit_in_channels=1600),
        n_timesteps=n_timesteps,
        max_patches=max_patches,
    )
    return FireRedTTS3State.from_dict(payload.data)
