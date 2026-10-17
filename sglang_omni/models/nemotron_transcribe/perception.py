# SPDX-License-Identifier: Apache-2.0
"""Build native log-mel, speech encoders, speaker fusion and projection."""

from __future__ import annotations

import math

import torch
from torch import nn

from sglang_omni.models.nemotron_transcribe.hf_config import decode_nemo_floats
from sglang_omni.models.nemotron_transcribe.mel_features import LogMelFrontend
from sglang_omni.models.nemotron_transcribe.parallel_expert import ParallelExpertEncoder
from sglang_omni.models.nemotron_transcribe.sortformer import SortformerModules
from sglang_omni.models.nemotron_transcribe.sortformer_diarizer import (
    SortformerDiarizer,
)
from sglang_omni.models.nemotron_transcribe.speech_transformer import (
    SpeechTransformerEncoder,
)


class MelPreprocessor(nn.Module):
    """Holds the featurizer at preprocessor.featurizer, matching the checkpoint."""

    def __init__(self, featurizer: LogMelFrontend) -> None:
        super().__init__()
        self.featurizer = featurizer

    def forward(
        self, waveforms: torch.Tensor, sample_counts: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.featurizer(waveforms, sample_counts)


class AudioPerception(nn.Module):
    """16 kHz audio to language-model embeddings shaped (batch, time, width)."""

    def __init__(
        self,
        preprocessor: MelPreprocessor,
        encoder: ParallelExpertEncoder,
        output_width: int,
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.encoder = encoder
        self.proj = nn.Linear(encoder.asr_encoder.d_model, output_width)

    def forward(
        self, input_signal: torch.Tensor, input_signal_length: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        mel, frame_counts = self.preprocessor(input_signal, input_signal_length)
        encoded, encoded_counts = self.encoder(mel, frame_counts)
        return self.proj(encoded.transpose(1, 2)), encoded_counts


def build_audio_perception(
    perception_config: object, pe_encoder_config: object
) -> AudioPerception:
    """Build the perception module from the checkpoint's two config sections."""
    perception = expect_dict(decode_nemo_floats(perception_config), "perception")
    encoder_config = expect_dict(
        decode_nemo_floats(pe_encoder_config), "pe_encoder_config"
    )
    if (
        str(encoder_config.get("asr_encoder_type", "transformer")).lower()
        != "transformer"
    ):
        raise ValueError("only the transformer ASR encoder is supported")
    else:
        pass
    asr_cfg = expect_dict(encoder_config.get("asr_encoder_cfg"), "asr_encoder_cfg")
    diar_cfg = expect_dict(
        encoder_config.get("diarization_model_cfg"), "diarization_model_cfg"
    )
    asr_encoder = build_transformer(asr_cfg, rotary_fraction=0.5)
    diar_encoder_cfg = expect_dict(diar_cfg.get("encoder"), "diarization encoder")
    diar_encoder = build_transformer(diar_encoder_cfg, rotary_fraction=1.0)
    modules_cfg = expect_dict(diar_cfg.get("sortformer_modules"), "sortformer_modules")
    high_resolution = read_bool(diar_cfg, "high_resolution", False)
    upsample = diar_encoder.subsampling_factor if high_resolution else 1
    modules = build_sortformer(modules_cfg, upsample)
    diarizer = SortformerDiarizer(
        diar_encoder,
        modules,
        high_resolution=high_resolution,
        output_subsampling_factor=asr_encoder.subsampling_factor,
    )
    diar_frontend_cfg = expect_dict(
        diar_cfg.get("preprocessor"), "diarization preprocessor"
    )
    diarizer.preprocessor = MelPreprocessor(
        build_frontend(diar_frontend_cfg, pad_to=16)
    )
    mode = str(encoder_config.get("speaker_feature_mode", "thresholded"))
    threshold = encoder_config.get("speaker_activity_threshold", 0.5)
    if threshold is None:
        threshold_value = None
    elif isinstance(threshold, (int, float)) and not isinstance(threshold, bool):
        threshold_value = float(threshold)
    else:
        raise TypeError("speaker_activity_threshold must be a number or null")
    encoder = ParallelExpertEncoder(
        asr_encoder,
        diarizer,
        asr_normalize=optional_text(encoder_config, "asr_normalize_type")
        or "per_feature",
        diar_normalize=optional_text(encoder_config, "diar_normalize_type"),
        speaker_feature_mode=mode,
        speaker_activity_threshold=threshold_value,
        speaker_kernel_scale=read_float(encoder_config, "spk_kernel_scale", 1.0),
        online_frames=read_int(encoder_config, "online_inference_length", 500),
        left_context_frames=read_int(encoder_config, "chunk_left_context", 50),
        right_context_frames=read_int(encoder_config, "chunk_right_context", 50),
        cache_limit=read_int(encoder_config, "diar_spkcache_len", 188),
        recent_limit=read_int(encoder_config, "diar_fifo_len", 40),
        cache_update_frames=read_int(
            encoder_config, "diar_spkcache_update_period", 300
        ),
    )
    if read_bool(encoder_config, "freeze_diar", True):
        diarizer.requires_grad_(False)
        diarizer.eval()
    else:
        pass
    output_width = read_int(perception, "output_dim", asr_encoder.d_model)
    frontend_cfg = expect_dict(
        perception.get("preprocessor"), "perception preprocessor"
    )
    perception_module = AudioPerception(
        MelPreprocessor(build_frontend(frontend_cfg, pad_to=0)),
        encoder,
        output_width,
    )
    # The fused encoder normalizes each branch itself.
    perception_module.preprocessor.featurizer.normalize = None
    perception_module.eval()
    return perception_module


def build_frontend(config: dict[str, object], pad_to: int) -> LogMelFrontend:
    return LogMelFrontend(
        sample_rate=read_int(config, "sample_rate", 16000),
        window_size=read_float(config, "window_size", 0.025),
        window_stride=read_float(config, "window_stride", 0.01),
        n_fft=read_int(config, "n_fft", 512),
        features=read_int(config, "features", 128),
        preemph=read_float(config, "preemph", 0.97),
        pad_to=read_int(config, "pad_to", pad_to),
        mag_power=read_float(config, "mag_power", 2.0),
        normalize=optional_text(config, "normalize"),
    )


def build_transformer(
    config: dict[str, object], rotary_fraction: float
) -> SpeechTransformerEncoder:
    subsampling = str(config.get("subsampling", "feature_stacking"))
    if subsampling != "feature_stacking":
        raise ValueError(f"unsupported subsampling {subsampling!r}")
    else:
        pass
    attention = str(config.get("self_attention_model", "rope"))
    if attention != "rope":
        raise ValueError(f"unsupported self attention {attention!r}")
    else:
        pass
    width = read_int(config, "d_model", 512)
    if "feat_out" in config and config["feat_out"] is not None:
        feat_out = read_int(config, "feat_out", -1)
        if feat_out not in (-1, width):
            raise ValueError(f"feat_out {feat_out} is not implemented")
        else:
            pass
    else:
        pass
    if read_bool(config, "xscaling", False):
        scale = math.sqrt(read_float(config, "d_model", 512.0))
    else:
        scale = None
    return SpeechTransformerEncoder(
        feat_in=read_int(config, "feat_in", 128),
        d_model=width,
        n_heads=read_int(config, "n_heads", 8),
        n_layers=read_int(config, "n_layers", 1),
        subsampling_factor=read_int(config, "subsampling_factor", 8),
        drop_rate=read_float(config, "drop_rate", 0.1),
        qkv_bias=read_bool(config, "qkv_bias", False),
        qk_norm=read_bool(config, "qk_norm", False),
        ff_expansion=read_float(config, "ff_expansion", 4.0),
        pre_block_norm=read_bool(config, "pre_block_norm", True),
        rope_base=read_float(config, "rope_base", 10000.0),
        rotary_fraction=read_float(config, "rotary_fraction", rotary_fraction),
        pos_emb_max_len=read_int(config, "pos_emb_max_len", 5000),
        xscale=scale,
    )


def build_sortformer(
    config: dict[str, object], upsample_factor: int
) -> SortformerModules:
    return SortformerModules(
        speaker_count=read_int(config, "num_spks", 8),
        fc_d_model=read_int(config, "fc_d_model", 512),
        tf_d_model=read_int(config, "tf_d_model", 192),
        dropout_rate=read_float(config, "dropout_rate", 0.5),
        upsample_factor=upsample_factor,
        spkcache_len=read_int(config, "spkcache_len", 188),
        fifo_len=read_int(config, "fifo_len", 0),
        chunk_len=read_int(config, "chunk_len", 188),
        spkcache_update_period=read_int(config, "spkcache_update_period", 188),
        spkcache_sil_frames_per_spk=read_int(config, "spkcache_sil_frames_per_spk", 1),
        pred_score_threshold=read_float(config, "pred_score_threshold", 0.25),
        max_index=read_int(config, "max_index", 99999),
        scores_boost_latest=read_float(config, "scores_boost_latest", 0.05),
        sil_threshold=read_float(config, "sil_threshold", 0.2),
        strong_boost_rate=read_float(config, "strong_boost_rate", 0.75),
        weak_boost_rate=read_float(config, "weak_boost_rate", 1.5),
        min_pos_scores_rate=read_float(config, "min_pos_scores_rate", 0.5),
        use_learnable_sil_emb=read_bool(config, "use_learnable_sil_emb", False),
    )


def expect_dict(value: object, label: str) -> dict[str, object]:
    if isinstance(value, dict):
        return value
    else:
        raise TypeError(f"{label} must be a dict, got {type(value).__name__}")


def optional_text(config: dict[str, object], key: str) -> str | None:
    if key not in config or config[key] is None:
        return None
    else:
        return str(config[key])


def read_int(config: dict[str, object], key: str, default: int) -> int:
    value = config.get(key, default)
    if value is None:
        return default
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be an integer")
    else:
        return int(value)


def read_float(config: dict[str, object], key: str, default: float) -> float:
    value = config.get(key, default)
    if value is None:
        return default
    elif isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{key} must be a number")
    else:
        return float(value)


def read_bool(config: dict[str, object], key: str, default: bool) -> bool:
    value = config.get(key, default)
    if isinstance(value, bool):
        return value
    else:
        raise TypeError(f"{key} must be a boolean")
