# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from contextlib import contextmanager

import pytest
import torch
from torch import nn

from sglang_omni.models.moss_tts_realtime.vocoder_decoder import (
    configure_moss_tts_realtime_vocoder_decoder,
    moss_tts_realtime_vocoder_decoder_dtype,
)


class MossAudioTokenizerMultiheadAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self._streaming_state = None


class _DecoderBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.self_attn = MossAudioTokenizerMultiheadAttention()


class _Quantizer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(()))

    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        return codes.to(torch.float32) * self.scale


class _Codec(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.decoder = nn.ModuleList([_DecoderBlock()])
        self.quantizer = _Quantizer()

    @contextmanager
    def streaming(self):
        self.decoder[0].self_attn._streaming_state = object()
        try:
            yield
        finally:
            self.decoder[0].self_attn._streaming_state = None


def test_vocoder_decoder_bfloat16_keeps_native_attention_and_fp32_quantizer() -> None:
    codec = _Codec()
    attention = codec.decoder[0].self_attn

    count = configure_moss_tts_realtime_vocoder_decoder(
        codec,
        dtype=torch.bfloat16,
    )

    assert count == 1
    assert codec.decoder[0].self_attn is attention
    assert next(codec.decoder.parameters()).dtype is torch.bfloat16
    assert next(codec.quantizer.parameters()).dtype is torch.float32
    assert codec.quantizer.decode_codes(torch.ones(1)).dtype is torch.bfloat16
    assert moss_tts_realtime_vocoder_decoder_dtype(codec) is torch.bfloat16


def test_vocoder_decoder_configuration_is_idempotent() -> None:
    codec = _Codec()

    assert (
        configure_moss_tts_realtime_vocoder_decoder(
            codec,
            dtype=torch.bfloat16,
        )
        == 1
    )
    wrapped_decode = codec.quantizer.decode_codes
    assert (
        configure_moss_tts_realtime_vocoder_decoder(
            codec,
            dtype=torch.bfloat16,
        )
        == 1
    )

    assert codec.quantizer.decode_codes == wrapped_decode


def test_vocoder_decoder_rejects_dtype_change() -> None:
    codec = _Codec()
    configure_moss_tts_realtime_vocoder_decoder(codec, dtype=torch.bfloat16)

    with pytest.raises(RuntimeError, match="already configured"):
        configure_moss_tts_realtime_vocoder_decoder(codec, dtype=torch.float32)


def test_vocoder_decoder_rejects_active_streaming_state() -> None:
    codec = _Codec()

    with (
        codec.streaming(),
        pytest.raises(
            RuntimeError,
            match="before opening codec.streaming",
        ),
    ):
        configure_moss_tts_realtime_vocoder_decoder(
            codec,
            dtype=torch.bfloat16,
        )


def test_vocoder_decoder_rejects_unvalidated_dtype() -> None:
    codec = _Codec()

    with pytest.raises(ValueError, match="unsupported.*vocoder dtype"):
        configure_moss_tts_realtime_vocoder_decoder(codec, dtype=torch.float16)
