# SPDX-License-Identifier: Apache-2.0
"""MOSS-TTS-Realtime streaming vocoder decoder configuration."""

from __future__ import annotations

import logging
from types import MethodType
from typing import Any

import torch
from torch import nn

logger = logging.getLogger(__name__)

_DECODER_DTYPE_ATTR = "_sglang_omni_vocoder_decoder_dtype"
_QUANTIZER_ORIGINAL_DECODE_ATTR = "_sglang_omni_original_decode_codes"
_QUANTIZER_OUTPUT_DTYPE_ATTR = "_sglang_omni_decode_codes_output_dtype"
_SUPPORTED_DECODER_DTYPES = frozenset((torch.float32, torch.bfloat16))


def _native_sdpa_attention_count(codec: Any) -> int:
    return sum(
        module.__class__.__name__ == "MossAudioTokenizerMultiheadAttention"
        for decoder_module in codec.decoder
        for module in decoder_module.modules()
    )


def _decode_codes_in_decoder_dtype(
    self: nn.Module,
    codes: torch.Tensor,
) -> torch.Tensor:
    original = getattr(self, _QUANTIZER_ORIGINAL_DECODE_ATTR)
    output_dtype = getattr(self, _QUANTIZER_OUTPUT_DTYPE_ATTR)
    decoded = original(codes)
    if not isinstance(decoded, torch.Tensor):
        raise RuntimeError("MOSS codec quantizer decode_codes() must return a tensor")
    return decoded.to(dtype=output_dtype)


def configure_moss_tts_realtime_vocoder_decoder(
    codec: Any,
    *,
    dtype: torch.dtype,
) -> int:
    """Configure decoder precision while retaining the native streaming SDPA."""
    if dtype not in _SUPPORTED_DECODER_DTYPES:
        supported = ", ".join(sorted(str(item) for item in _SUPPORTED_DECODER_DTYPES))
        raise ValueError(
            f"unsupported MOSS-TTS-Realtime vocoder dtype {dtype}; "
            f"expected one of {supported}"
        )

    attention_count = _native_sdpa_attention_count(codec)
    configured_dtype = getattr(codec, _DECODER_DTYPE_ATTR, None)
    if configured_dtype is not None:
        if configured_dtype != dtype:
            raise RuntimeError(
                "MOSS-TTS-Realtime vocoder decoder is already configured with "
                f"dtype {configured_dtype}; cannot change it to {dtype}"
            )
        return attention_count

    active_streaming_modules = [
        module.__class__.__name__
        for module in codec.decoder.modules()
        if getattr(module, "_streaming_state", None) is not None
    ]
    if active_streaming_modules:
        raise RuntimeError(
            "vocoder decoder dtype must be configured before opening "
            f"codec.streaming(); found {len(active_streaming_modules)} active "
            "decoder states"
        )

    codec.decoder.to(dtype=dtype)
    if dtype != torch.float32:
        quantizer = codec.quantizer
        original_decode = getattr(quantizer, "decode_codes", None)
        if not callable(original_decode):
            raise RuntimeError("MOSS codec quantizer does not expose decode_codes()")
        setattr(quantizer, _QUANTIZER_ORIGINAL_DECODE_ATTR, original_decode)
        setattr(quantizer, _QUANTIZER_OUTPUT_DTYPE_ATTR, dtype)
        quantizer.decode_codes = MethodType(
            _decode_codes_in_decoder_dtype,
            quantizer,
        )

    setattr(codec, _DECODER_DTYPE_ATTR, dtype)
    logger.info(
        "Configured MOSS-TTS-Realtime vocoder decoder dtype=%s with %d native "
        "streaming SDPA attention layers",
        dtype,
        attention_count,
    )
    return attention_count


def moss_tts_realtime_vocoder_decoder_dtype(codec: Any) -> torch.dtype:
    configured_dtype = getattr(codec, _DECODER_DTYPE_ATTR, None)
    if isinstance(configured_dtype, torch.dtype):
        return configured_dtype

    decoder = getattr(codec, "decoder", None)
    if isinstance(decoder, nn.Module):
        try:
            return next(decoder.parameters()).dtype
        except StopIteration:
            pass
    return torch.float32
