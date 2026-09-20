# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import torch
from sglang.srt.multimodal.customized_mm_processor_utils import (
    register_customized_processor,
)
from sglang.utils import logger
from transformers import (
    AutoConfig,
    AutoFeatureExtractor,
    AutoTokenizer,
    PretrainedConfig,
)
from transformers.feature_extraction_sequence_utils import SequenceFeatureExtractor

from sglang_omni.utils.audio_features import cached_fbank

from .tool_funcs.audio_lengths import fun_asr_low_frame_rate_length

AUDIO_PLACEHOLDER_TOKEN = "<|object_ref_start|>"


class FunAsrNanoFeatureExtractor(SequenceFeatureExtractor):
    """80-mel log-mel fbank + LFR stacking, matching Fun-ASR's WavFrontend.

    Output ``input_features`` shape is ``[batch, num_frames_lfr * n_mels, T_lfr]`` =
    ``[batch, 560, T_lfr]`` where ``T_lfr = ceil(T_mel / stride_lfr)``. The encoder's
    ``input_size`` is 560 (= 7 * 80). ``attention_mask`` tracks valid LFR
    frames; its per-row sum is the post-LFR frame count fed to
    :func:`fun_asr_low_frame_rate_length`.
    """

    model_input_names = ["input_features"]

    def __init__(
        self,
        feature_size: int = 80,
        sampling_rate: int = 16000,
        frame_length: int = 25,
        frame_shift: int = 10,
        num_frames_lfr: int = 7,
        stride_lfr: int = 6,
        window: str = "hamming",
        padding_value: float = 0.0,
        return_attention_mask: bool = True,
        **kwargs,
    ):
        super().__init__(
            feature_size=feature_size,
            sampling_rate=sampling_rate,
            padding_value=padding_value,
            return_attention_mask=return_attention_mask,
            **kwargs,
        )
        self.feature_size = feature_size
        self.n_mels = feature_size
        self.sampling_rate = sampling_rate
        # frame_length/shift in ms (for torchaudio.compliance.kaldi.fbank)
        self.frame_length = frame_length
        self.frame_shift = frame_shift
        # transformers expresses window length/shift in samples at sampling_rate
        self.n_fft = int(round(frame_length * sampling_rate / 1000))  # 400 @ 16k
        self.hop_length = int(round(frame_shift * sampling_rate / 1000))  # 160 @ 16k
        self.win_length = self.n_fft
        self.num_frames_lfr = num_frames_lfr
        self.stride_lfr = stride_lfr
        self.window = window
        self.padding_value = padding_value
        self.return_attention_mask = return_attention_mask

        # _extract_fbank builds its own mel filterbank/window via
        # torchaudio.compliance.kaldi.fbank, so we don't precompute them here.
        if window != "hamming":
            raise ValueError(f"Unsupported window: {window!r} (Fun-ASR uses hamming)")

    @property
    def nb_max_frames(self) -> int:
        """Max LFR frames for a 30s clip — used for context_length sizing."""
        max_mel = int(round(30.0 * self.sampling_rate / self.hop_length))
        return (max_mel + self.stride_lfr - 1) // self.stride_lfr

    def extract_fbank(self, waveform: np.ndarray) -> tuple[torch.Tensor, int]:
        """Compute 80-mel log-mel fbank via Kaldi compliance (matches funasr WavFrontend).

        Mirrors ``funasr.frontends.wav_frontend.WavFrontend.forward``:
        ``waveform * (1 << 15)`` (int16 scale) → ``torchaudio.compliance.kaldi.fbank``
        (hamming window, energy_floor=0, dither=0, snip_edges=True).
        Returns ``(fbank, T_mel)`` where ``fbank`` is ``[T_mel, n_mels]``.
        """
        wav = np.asarray(waveform, dtype=np.float32)
        if wav.ndim != 1:
            wav = wav.reshape(-1)
        # WavFrontend.forward: waveform * (1 << 15) — scale to int16 range.
        # Without this, log-mel values are ~21 lower (2*log(32768)) and the
        # encoder/adaptor produce embeddings the LLM cannot decode (→ /sil).
        wav_t = torch.from_numpy(wav).unsqueeze(0) * (1 << 15)
        # Cap frame_length for very short audio (funasr WavFrontend does this).
        frame_length = min(self.frame_length, wav.shape[0] / self.sampling_rate * 1000)
        mat = cached_fbank(
            wav_t,
            num_mel_bins=self.n_mels,
            frame_length=frame_length,
            frame_shift=self.frame_shift,
            window_type=self.window,
            sample_frequency=self.sampling_rate,
        )  # [T_mel, n_mels]
        return mat, mat.shape[0]

    def lfr(self, fbank: torch.Tensor) -> tuple[torch.Tensor, int]:
        """Low frame rate stacking (matches funasr ``apply_lfr``).

        Stacks ``num_frames_lfr`` frames every ``stride_lfr`` stride: left-pad by repeating
        the first frame ``(num_frames_lfr-1)//2`` times, right-pad the last frame to
        fill the final window, then gather via ``as_strided``.
        Returns ``(lfr_out, T_lfr)`` where ``lfr_out`` is ``[T_lfr, num_frames_lfr*n_mels]``.
        """
        t_mel = fbank.shape[0]
        t_lfr = int(np.ceil(t_mel / self.stride_lfr))
        pad_left = (self.num_frames_lfr - 1) // 2
        left_padding = fbank[0:1].repeat(pad_left, 1)
        inputs = torch.vstack([left_padding, fbank])
        t_padded = inputs.shape[0]
        feat_dim = inputs.shape[-1]
        strides = (self.stride_lfr * feat_dim, 1)
        sizes = (t_lfr, self.num_frames_lfr * feat_dim)
        last_idx = (t_padded - self.num_frames_lfr) // self.stride_lfr + 1
        num_padding = self.num_frames_lfr - (t_padded - last_idx * self.stride_lfr)
        if num_padding > 0:
            num_padding = (
                (
                    2 * self.num_frames_lfr
                    - 2 * t_padded
                    + (t_lfr - 1 + last_idx) * self.stride_lfr
                )
                / 2
                * (t_lfr - last_idx)
            )
            inputs = torch.vstack([inputs] + [inputs[-1:]] * int(num_padding))
        out = inputs.as_strided(sizes, strides)
        return out.clone().to(torch.float32), t_lfr

    def __call__(
        self,
        audio,
        sampling_rate: int | None = None,
        return_tensors=None,
        return_attention_mask: bool | None = None,
        padding: str | bool = False,
        truncation: bool = False,
        max_length: int | None = None,
        **kwargs,
    ):
        if isinstance(audio, np.ndarray) and audio.ndim == 1:
            waveforms = [audio]
        elif isinstance(audio, (list, tuple)) and (
            not audio or isinstance(audio[0], (int, float, np.floating))
        ):
            waveforms = [np.asarray(audio, dtype=np.float32)]
        elif isinstance(audio, np.ndarray) and audio.ndim == 2:
            waveforms = [audio[i] for i in range(audio.shape[0])]
        else:
            waveforms = list(audio)

        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            logger.warning(
                f"FunAsrNanoFeatureExtractor: sampling_rate {sampling_rate} != "
                f"{self.sampling_rate}; resampling is the caller's responsibility."
            )

        feats, masks = [], []
        for wav in waveforms:
            fbank, t_mel = self.extract_fbank(wav)
            lfr_feat, t_lfr = self.lfr(fbank)  # [t_lfr, num_frames_lfr*n_mels=560]
            # Transpose to [num_frames_lfr * n_mels, t_lfr] = [560, t_lfr] (encoder expects [B, T, 560])
            lfr_feat = lfr_feat.t().contiguous()
            feats.append(lfr_feat)
            masks.append([1] * t_lfr)

        if padding == "max_length":
            max_t = self.nb_max_frames
        elif padding == "longest" or padding is True:
            max_t = max(f.shape[1] for f in feats)
        else:
            max_t = max(f.shape[1] for f in feats)

        n_feat = self.num_frames_lfr * self.n_mels
        batched = np.full(
            (len(feats), n_feat, max_t), self.padding_value, dtype=np.float32
        )
        attention = np.zeros((len(feats), max_t), dtype=np.int64)
        for i, f in enumerate(feats):
            t = f.shape[1]
            batched[i, :, :t] = f
            attention[i, :t] = masks[i]

        return_attention_mask = (
            self.return_attention_mask
            if return_attention_mask is None
            else return_attention_mask
        )
        out = {"input_features": batched}
        if return_attention_mask:
            out["attention_mask"] = attention
        if return_tensors == "pt":
            out["input_features"] = torch.from_numpy(out["input_features"])
            if "attention_mask" in out:
                out["attention_mask"] = torch.from_numpy(out["attention_mask"])
        return out


# ---------------------------------------------------------------------------
# Processor — feature extractor + tokenizer + placeholder expansion.
# ---------------------------------------------------------------------------


class FunAsrNanoProcessor:
    """Composite processor: FunAsrNanoFeatureExtractor + Qwen2Tokenizer.

    Mirrors ``Qwen3ASRProcessor``. AutoProcessor.from_pretrained for the HF
    Fun-ASR checkpoint expects remote code (``processing_fun_asr_nano.py``)
    that is not bundled, so sglang-omni provides this processor directly and
    registers it via ``register_customized_processor``.
    """

    attributes = ["feature_extractor", "tokenizer"]
    feature_extractor_class = "FunAsrNanoFeatureExtractor"
    tokenizer_class = "AutoTokenizer"

    def __init__(self, feature_extractor=None, tokenizer=None, **kwargs):
        self.feature_extractor = feature_extractor
        self.tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, **kwargs):
        trust_remote_code = kwargs.pop("trust_remote_code", True)
        feature_extractor = FunAsrNanoFeatureExtractor.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            pretrained_model_name_or_path,
            trust_remote_code=trust_remote_code,
        )
        return cls(feature_extractor=feature_extractor, tokenizer=tokenizer)

    def get_feat_extract_output_lengths(self, input_lengths):
        """LFR frames -> adaptor audio-token count (3x stride-2)."""
        return fun_asr_low_frame_rate_length(input_lengths)

    def __call__(self, text=None, audio=None, audio_kwargs=None, **kwargs):
        inputs: dict[str, Any] = {}
        if audio is not None:
            audio_kwargs = audio_kwargs or {}
            audio_inputs = self.feature_extractor(
                audio,
                sampling_rate=self.feature_extractor.sampling_rate,
                return_tensors=kwargs.get("return_tensors"),
                return_attention_mask=True,
                **audio_kwargs,
            )
            inputs["input_features"] = audio_inputs["input_features"]
            if "attention_mask" in audio_inputs:
                inputs["feature_attention_mask"] = audio_inputs["attention_mask"]

        if text is not None:
            text_inputs = self.tokenizer(
                text,
                return_tensors=kwargs.get("return_tensors"),
                padding=kwargs.get("padding", False),
            )
            input_ids = text_inputs["input_ids"]

            # Expand the single <|object_ref_start|> placeholder in the prompt
            # to N copies, where N is the adaptor's audio-token count for this
            # clip. Without this, the model sees only 1 audio token for hundreds
            # of LFR frames and cannot align audio embeddings with positions.
            if audio is not None and "feature_attention_mask" in inputs:
                audio_pad_id = self.tokenizer.convert_tokens_to_ids(
                    AUDIO_PLACEHOLDER_TOKEN
                )
                feat_lengths = inputs["feature_attention_mask"].sum(dim=-1)
                audio_token_counts = self.get_feat_extract_output_lengths(feat_lengths)
                expanded = []
                for seq_idx in range(input_ids.shape[0]):
                    ids = (
                        input_ids[seq_idx].tolist()
                        if hasattr(input_ids[seq_idx], "tolist")
                        else list(input_ids[seq_idx])
                    )
                    audio_idx = 0
                    new_ids = []
                    for tid in ids:
                        if tid == audio_pad_id and audio_idx < len(audio_token_counts):
                            n = int(audio_token_counts[audio_idx].item())
                            new_ids.extend([audio_pad_id] * n)
                            audio_idx += 1
                        else:
                            new_ids.append(tid)
                    expanded.append(new_ids)
                max_len = max(len(s) for s in expanded)
                pad_id = self.tokenizer.pad_token_id or 0
                padded = [s + [pad_id] * (max_len - len(s)) for s in expanded]
                input_ids = torch.tensor(padded, dtype=torch.long)

            inputs["input_ids"] = input_ids
        return inputs


class FunAsrNanoEncoderConfig(PretrainedConfig):
    """SenseVoice SANM encoder configuration."""

    model_type = "fun_asr_nano_encoder"

    def __init__(
        self,
        num_mel_bins: int = 80,
        num_stacked_frames: int = 7,
        hidden_size: int = 512,
        num_attention_heads: int = 4,
        intermediate_size: int = 2048,
        num_hidden_layers: int = 70,
        num_timestamp_prediction_layers: int = 20,
        fsmn_kernel_size: int = 11,
        hidden_dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        hidden_act: str = "relu",
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.num_stacked_frames = num_stacked_frames
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_timestamp_prediction_layers = num_timestamp_prediction_layers
        self.fsmn_kernel_size = fsmn_kernel_size
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.activation_dropout = activation_dropout
        self.hidden_act = hidden_act
        self.layer_norm_eps = layer_norm_eps

    @property
    def input_size(self) -> int:
        return self.num_mel_bins * self.num_stacked_frames


class FunAsrNanoAdaptorConfig(PretrainedConfig):
    """Bidirectional adaptor and projection configuration in HF field names."""

    model_type = "fun_asr_nano_adaptor"

    def __init__(
        self,
        hidden_size: int = 1024,
        intermediate_size: int = 256,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 8,
        projector_hidden_size: int = 2048,
        projector_hidden_act: str = "relu",
        hidden_act: str = "relu",
        hidden_dropout: float = 0.0,
        attention_dropout: float = 0.0,
        layer_norm_eps: float = 1e-5,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.projector_hidden_size = projector_hidden_size
        self.projector_hidden_act = projector_hidden_act
        self.hidden_act = hidden_act
        self.hidden_dropout = hidden_dropout
        self.attention_dropout = attention_dropout
        self.layer_norm_eps = layer_norm_eps


@register_customized_processor(FunAsrNanoProcessor)
class FunAsrNanoConfig(PretrainedConfig):
    """Configuration for the Fun-ASR-Nano checkpoint."""

    model_type = "fun_asr_nano"
    sub_configs: ClassVar[dict[str, Any]] = {
        "audio_config": FunAsrNanoEncoderConfig,
        "adaptor_config": FunAsrNanoAdaptorConfig,
    }

    def __init__(
        self,
        audio_config=None,
        adaptor_config=None,
        text_config=None,
        audio_token_id: int = 151646,
        initializer_range: float = 0.02,
        tie_word_embeddings: bool = True,
        **kwargs,
    ):
        if "encoder_config" in kwargs:
            raise ValueError(
                "Fun-ASR supports only the current flat HF checkpoint; "
                "download the latest FunAudioLLM/Fun-ASR-Nano-2512-hf revision."
            )
        if isinstance(audio_config, dict):
            audio_config = FunAsrNanoEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = FunAsrNanoEncoderConfig()
        self.audio_config = audio_config
        if isinstance(adaptor_config, dict):
            adaptor_config = FunAsrNanoAdaptorConfig(**adaptor_config)
        elif adaptor_config is None:
            adaptor_config = FunAsrNanoAdaptorConfig()
        self.adaptor_config = adaptor_config

        from transformers.models.qwen3.configuration_qwen3 import (
            Qwen3Config as HFQwen3Config,
        )

        if isinstance(text_config, dict):
            text_config = HFQwen3Config(**text_config)
        elif text_config is None:
            text_config = HFQwen3Config()
        self.text_config = text_config
        self.audio_token_id = audio_token_id
        self.initializer_range = initializer_range

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )

    def get_text_config(self, decoder: bool = False) -> PretrainedConfig:
        text_config = getattr(self, "text_config", None)
        if text_config is None:
            return self
        return text_config


AutoConfig.register("fun_asr_nano", FunAsrNanoConfig)
AutoConfig.register("fun_asr_nano_encoder", FunAsrNanoEncoderConfig)

# note(LauraGPT): The checkpoint metadata names this local feature extractor.
AutoFeatureExtractor.register("FunAsrNanoFeatureExtractor", FunAsrNanoFeatureExtractor)
