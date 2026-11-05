# SPDX-License-Identifier: Apache-2.0
"""Load codec weights once and condition decoding on each request's reference audio."""

from __future__ import annotations

import random
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Protocol, TypedDict

import numpy as np
import onnxruntime
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from safetensors.torch import load_file
from transformers import DynamicCache, EncoderDecoderCache, WhisperFeatureExtractor
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang_omni.models.moss_speech.request_builders import (
    MossSpeechState,
    RequestValidationError,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.utils.audio import load_audio

SAMPLE_RATE_HZ = 24000
MAX_REFERENCE_SECONDS = 30


class VoiceConditioning(TypedDict):
    prompt_token: torch.Tensor
    prompt_feat: torch.Tensor
    embedding: torch.Tensor


class AudioEncoderOutput(Protocol):
    quantized_token_ids: torch.Tensor


class AudioEncoder(Protocol):
    def __call__(
        self,
        *,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor,
        return_dict: bool,
        past_key_values: EncoderDecoderCache,
    ) -> AudioEncoderOutput: ...


class MelExtractor(Protocol):
    def __call__(self, waveform: torch.Tensor) -> torch.Tensor: ...


class AudioDecoder(Protocol):
    feat_extractor: MelExtractor
    campplus_session: onnxruntime.InferenceSession

    def token2wav(
        self,
        token: torch.Tensor,
        uuid: str,
        *,
        prompt_token: torch.Tensor,
        prompt_feat: torch.Tensor,
        embedding: torch.Tensor,
        finalize: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


@contextmanager
def codec_rng() -> Iterator[None]:
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    try:
        with torch.random.fork_rng():
            yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)


class MossSpeechCodec:
    def __init__(self, codec_path: str, device: torch.device) -> None:
        self.device: torch.device = device
        codec_directory = Path(codec_path)
        decoder_class = get_class_from_dynamic_module(
            "modeling_moss_speech_codec.AudioDecoder", codec_path, local_files_only=True
        )
        encoder_class = get_class_from_dynamic_module(
            "modeling_whisper.WhisperVQEncoder", codec_path, local_files_only=True
        )
        with codec_rng():
            self.decoder: AudioDecoder | None = decoder_class(
                config_path=codec_directory / "flow/config.yaml",
                flow_ckpt_path=codec_directory / "flow/flow.pt",
                hift_ckpt_path=codec_directory / "flow/hift.pt",
                campplus_model=codec_directory / "flow/campplus.onnx",
                device=device,
            ).eval()
            configuration = encoder_class.config_class.from_pretrained(codec_path)
            encoder = encoder_class(configuration).to(device, torch.float32).eval()
            checkpoint = load_file(str(codec_directory / "model.safetensors"))
            missing, unexpected = encoder.load_state_dict(
                {
                    name.removeprefix("encoder."): tensor
                    for name, tensor in checkpoint.items()
                    if name.startswith("encoder.")
                    and name not in {"encoder.ema_count", "encoder.ema_weight"}
                },
                strict=False,
            )
            if [
                name for name in missing if "masked_spec_embed" not in name
            ] or unexpected:
                raise RuntimeError(
                    f"Codec encoder weights do not match: {missing}, {unexpected}"
                )
            else:
                pass
            self.encoder: AudioEncoder | None = encoder
            self.feature_frames_per_token: int = (
                encoder.conv1.stride[0]
                * encoder.conv2.stride[0]
                * configuration.pooling_kernel_size
            )
            self.feature_extractor: WhisperFeatureExtractor = (
                WhisperFeatureExtractor.from_pretrained(codec_path)
            )

    @torch.inference_mode()
    def encode_reference(self, reference_audio: str) -> VoiceConditioning:
        assert self.encoder is not None and self.decoder is not None
        samples = load_audio(
            reference_audio,
            source_name="reference audio",
            target_sample_rate=SAMPLE_RATE_HZ,
        )
        minimum_reference_samples = (
            self.feature_frames_per_token
            * self.feature_extractor.hop_length
            * SAMPLE_RATE_HZ
            // self.feature_extractor.sampling_rate
        )
        if len(samples) < minimum_reference_samples:
            raise RequestValidationError(
                "Reference audio is too short for codec conditioning"
            )
        elif (
            len(samples) > MAX_REFERENCE_SECONDS * SAMPLE_RATE_HZ
            or not np.isfinite(samples).all()
        ):
            raise RequestValidationError(
                f"Reference audio must contain finite audio of at most {MAX_REFERENCE_SECONDS} seconds"
            )
        else:
            pass
        waveform = torch.from_numpy(samples).unsqueeze(0)
        waveform_16khz = torchaudio.functional.resample(waveform, SAMPLE_RATE_HZ, 16000)
        audio_features = self.feature_extractor(
            [waveform_16khz[0].numpy()],
            sampling_rate=16000,
            return_attention_mask=True,
            return_tensors="pt",
            device=self.device,
            padding="longest",
            pad_to_multiple_of=self.feature_frames_per_token
            * self.feature_extractor.hop_length,
        ).to(self.device)
        # Explicit cache and return type avoid removed Transformers legacy defaults.
        encoded = self.encoder(
            input_features=audio_features.input_features,
            attention_mask=audio_features.attention_mask,
            return_dict=True,
            past_key_values=EncoderDecoderCache(DynamicCache(), DynamicCache()),
        )
        prompt_tokens = encoded.quantized_token_ids[0][
            audio_features.attention_mask[0, :: self.feature_frames_per_token].bool()
        ]
        prompt_mel = (
            self.decoder.feat_extractor(waveform)
            .squeeze(0)
            .transpose(0, 1)
            .unsqueeze(0)
        )
        prompt_code_count = min(prompt_mel.shape[1] // 4, prompt_tokens.numel())
        if prompt_code_count == 0:
            raise RequestValidationError(
                "Reference audio is too short for codec conditioning"
            )
        else:
            pass
        speaker_mel = kaldi.fbank(
            waveform_16khz, num_mel_bins=80, dither=0, sample_frequency=16000
        )
        speaker_mel -= speaker_mel.mean(dim=0, keepdim=True)
        speaker_encoder = self.decoder.campplus_session
        embedding = speaker_encoder.run(
            None,
            {speaker_encoder.get_inputs()[0].name: speaker_mel.unsqueeze(0).numpy()},
        )[0]
        return VoiceConditioning(
            prompt_token=prompt_tokens[:prompt_code_count]
            .unsqueeze(0)
            .to(self.device, torch.int32),
            prompt_feat=prompt_mel[:, : 4 * prompt_code_count, :].to(self.device),
            embedding=torch.from_numpy(np.asarray(embedding).flatten())
            .float()
            .unsqueeze(0)
            .to(self.device),
        )

    @torch.inference_mode()
    def decode(self, payload: StagePayload) -> StagePayload:
        state = MossSpeechState.from_dict(payload.data)
        if not state.audio_codes:
            raise RequestValidationError(
                "Generation ended before producing audio; increase max_new_tokens or retry"
            )
        else:
            pass
        assert self.decoder is not None
        with (
            codec_rng(),
            torch.backends.cudnn.flags(
                enabled=torch.backends.cudnn.enabled,
                benchmark=False,
                deterministic=True,
                allow_tf32=torch.backends.cudnn.allow_tf32,
            ),
        ):
            voice = self.encode_reference(state.reference_audio)
            random.seed(state.effective_seed)
            np.random.seed(state.effective_seed % (1 << 32))
            torch.manual_seed(state.effective_seed)
            waveform, _ = self.decoder.token2wav(
                torch.tensor([state.audio_codes], dtype=torch.long, device=self.device),
                uuid=payload.request_id,
                **voice,
                finalize=True,
            )
            waveform = waveform.detach().cpu().reshape(-1)
        assert waveform.numel() and torch.isfinite(waveform).all()
        return StagePayload(
            payload.request_id,
            payload.request,
            {
                "audio_data": waveform.tolist(),
                "sample_rate": SAMPLE_RATE_HZ,
                "finish_reason": state.finish_reason,
                "usage": {
                    "prompt_tokens": state.prompt_token_count,
                    "completion_tokens": state.completion_token_count,
                    "total_tokens": state.prompt_token_count
                    + state.completion_token_count,
                },
            },
        )

    def close(self) -> None:
        self.decoder = None
        self.encoder = None
