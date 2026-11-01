# SPDX-License-Identifier: Apache-2.0
"""Adapt NeMo SpeechLM configuration to the hybrid NemotronH decoder."""

from __future__ import annotations

import math

from sglang.srt.configs.nemotron_h import NemotronHConfig
from transformers import AutoConfig

MODEL_TYPE = "nemotron_transcribe"
MODEL_ARCH_OVERRIDE = "NemotronTranscribeForConditionalGeneration"
BACKBONE_ARCH = "NemotronHForCausalLM"
AUDIO_TOKEN_ID = 131072
EOS_TOKEN_ID = 11


def decode_nemo_floats(value: object) -> object:
    """Decode JSON float wrappers used for non-finite checkpoint values."""
    if isinstance(value, dict):
        if set(value) == {"__float__"}:
            return float(value["__float__"])
        else:
            return {name: decode_nemo_floats(nested) for name, nested in value.items()}
    elif isinstance(value, list):
        return [decode_nemo_floats(nested) for nested in value]
    else:
        return value


class NemotronTranscribeConfig(NemotronHConfig):
    model_type = MODEL_TYPE

    def __init__(
        self,
        perception: dict[str, object] | None = None,
        pe_encoder_config: dict[str, object] | None = None,
        audio_token_estimator: dict[str, object] | None = None,
        audio_token_id: int = AUDIO_TOKEN_ID,
        **kwargs: object,
    ) -> None:
        super().__init__(**kwargs)
        self.perception = dict(perception or {})
        self.pe_encoder_config = dict(pe_encoder_config or {})
        self.audio_token_estimator = dict(audio_token_estimator or {})
        self.audio_token_id = audio_token_id

    @classmethod
    def from_nemo_dict(cls, nemo_config: dict[str, object]) -> NemotronTranscribeConfig:
        decoder_config = decode_nemo_floats(nemo_config["llm_config"])
        if not isinstance(decoder_config, dict):
            raise TypeError("llm_config must be a dictionary")
        else:
            pass
        for name in ("architectures", "model_type", "torch_dtype", "dtype"):
            decoder_config.pop(name, None)
        timestep_limits = decoder_config.pop("time_step_limit", None)
        if timestep_limits is not None:
            if not isinstance(timestep_limits, list):
                raise TypeError("time_step_limit must be a list")
            else:
                decoder_config["mamba_dt_limit"] = tuple(
                    float(limit) for limit in timestep_limits
                )
        else:
            pass
        decoder_config["num_nextn_predict_layers"] = 0
        eos_token_ids = nemo_config.get("eos_token_id", EOS_TOKEN_ID)
        if isinstance(eos_token_ids, list):
            if not eos_token_ids:
                raise ValueError("eos_token_id must not be empty")
            else:
                eos_token_id = eos_token_ids[0]
        else:
            eos_token_id = eos_token_ids
        if not isinstance(eos_token_id, int):
            raise TypeError("eos_token_id must be an integer")
        else:
            decoder_config["eos_token_id"] = eos_token_id
        perception_config = nemo_config["perception"]
        encoder_config = nemo_config["pe_encoder_config"]
        token_estimator = nemo_config.get("audio_token_estimator", {})
        if (
            not isinstance(perception_config, dict)
            or not isinstance(encoder_config, dict)
            or not isinstance(token_estimator, dict)
        ):
            raise TypeError(
                "perception, pe_encoder_config and audio_token_estimator must be dictionaries"
            )
        else:
            pass
        # note (yhayaran): NemotronH architecture keeps hybrid cache selection intact.
        return cls(
            **decoder_config,
            architectures=[BACKBONE_ARCH],
            torch_dtype="bfloat16",
            perception=perception_config,
            pe_encoder_config=encoder_config,
            audio_token_estimator=token_estimator,
        )


def num_audio_frames(
    num_samples: int, estimator: dict[str, object] | None = None
) -> int:
    estimator = estimator or {}
    preprocessor = estimator.get("preprocessor", {})
    if not isinstance(preprocessor, dict):
        raise TypeError("audio_token_estimator.preprocessor must be a dictionary")
    else:
        pass
    fft_samples = preprocessor.get("n_fft", 512)
    hop_samples = preprocessor.get("hop_length", 160)
    padding_samples = preprocessor.get("stft_pad_amount", 256)
    if (
        not isinstance(fft_samples, int)
        or not isinstance(hop_samples, int)
        or not isinstance(padding_samples, int)
    ):
        raise TypeError("audio frame geometry must contain integer sample counts")
    elif fft_samples <= 0 or hop_samples <= 0 or padding_samples < 0:
        raise ValueError(
            "audio frame geometry must use positive FFT/hop and nonnegative padding"
        )
    else:
        return max(0, (num_samples + 2 * padding_samples - fft_samples) // hop_samples)


def num_audio_tokens(
    num_samples: int, estimator: dict[str, object] | None = None
) -> int:
    """Count audio placeholders after centered STFT and feature stacking."""
    estimator = estimator or {}
    subsampling = estimator.get("subsampling", {})
    if not isinstance(subsampling, dict):
        raise TypeError("audio_token_estimator.subsampling must be a dictionary")
    else:
        pass
    subsampling_factor = subsampling.get("factor", 8)
    if not isinstance(subsampling_factor, int) or subsampling_factor <= 0:
        raise ValueError("audio subsampling factor must be a positive integer")
    else:
        return max(
            1, math.ceil(num_audio_frames(num_samples, estimator) / subsampling_factor)
        )


def register_nemotron_transcribe_hf_config() -> None:
    AutoConfig.register(MODEL_TYPE, NemotronTranscribeConfig, exist_ok=True)


__all__ = [
    "AUDIO_TOKEN_ID",
    "EOS_TOKEN_ID",
    "MODEL_ARCH_OVERRIDE",
    "NemotronTranscribeConfig",
    "num_audio_frames",
    "num_audio_tokens",
    "register_nemotron_transcribe_hf_config",
]
