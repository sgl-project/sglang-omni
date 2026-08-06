# SPDX-License-Identifier: Apache-2.0
"""Reference audio conditioning and schedule assembly for dots.tts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from sglang_omni.models.dots_tts.components.utils.audio import high_quality_resample
from sglang_omni.models.dots_tts.components.vocoder.vocoder_inference import (
    VocoderInference,
)
from sglang_omni.models.dots_tts.payload_types import (
    DotsTtsState,
    load_dots_tts_state,
    store_dots_tts_state,
)
from sglang_omni.models.dots_tts.prompt_builder import (
    DOTS_TTS_MAX_SEQUENCE_LENGTH,
    build_dots_tts_schedule_ids,
)
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.reference_encoder import (
    KeyedReferenceEncodeHook,
    ReferenceEncodeService,
)

REFERENCE_TRIM_TOP_DB = 30


def load_reference_waveform(
    reference_path: str, *, target_sample_rate: int
) -> torch.Tensor:
    """Load one reference file as a mono ``[1, samples]`` float32 waveform."""
    import librosa

    path = str(Path(reference_path).expanduser())
    audio, source_sample_rate = librosa.load(path, sr=None, mono=True)
    audio, _trim_interval = librosa.effects.trim(audio, top_db=REFERENCE_TRIM_TOP_DB)
    waveform = torch.from_numpy(audio).float().reshape(1, -1)
    if int(source_sample_rate) != int(target_sample_rate):
        waveform = high_quality_resample(
            waveform,
            orig_sr=int(source_sample_rate),
            target_sr=int(target_sample_rate),
        )
    if waveform.ndim == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.numel() == 0:
        raise ValueError(
            f"dots.tts reference audio is empty after silence trimming: {path}"
        )
    return waveform.contiguous()


@dataclass(frozen=True)
class DotsTtsPromptFeatures:
    """Text-independent prompt conditioning cached per reference file.

    ``latent_distribution`` is the AudioVAE posterior ``[1, 2 * latent_dim,
    frames]``; it is cached instead of a sample so a seeded request stays
    reproducible on a cache hit.
    """

    latent_distribution: torch.Tensor
    speaker_embedding: torch.Tensor
    patch_count: int


class DotsTtsPromptFeatureHook(
    KeyedReferenceEncodeHook[str, DotsTtsPromptFeatures, DotsTtsPromptFeatures]
):
    """Run the AudioVAE encoder and the CAM++ speaker encoder on the GPU."""

    model_revision = ""
    encoder_id = "dots_tts_prompt_features"
    artifact_kind = "prompt_latent_distribution"

    def __init__(
        self,
        *,
        model_identity: str,
        vocoder_inference: VocoderInference,
        speaker_encoder: torch.nn.Module,
        sample_rate: int,
        samples_per_patch: int,
        device: torch.device,
    ) -> None:
        self.model_id = str(model_identity)
        self._vocoder_inference = vocoder_inference
        self._speaker_encoder = speaker_encoder
        self._sample_rate = int(sample_rate)
        self._samples_per_patch = int(samples_per_patch)
        self._device = device
        self.encoder_config_hash = hash_bytes(
            f"sample_rate:{self._sample_rate}"
            f":samples_per_patch:{self._samples_per_patch}"
            f":trim_top_db:{REFERENCE_TRIM_TOP_DB}".encode("utf-8")
        )

    def normalize_input(self, raw_input: object) -> str:
        return str(raw_input)

    def input_key(self, item: str) -> str | None:
        # Full-content memoized hash; the sampled variant collides for
        # same-size files that differ only in the middle. None means the file
        # is unreadable: bypass the cache and let encode_one raise the real
        # error to the caller.
        return reference_path_cache_key(item, trust_stat=False)

    @torch.no_grad()
    def encode_one(self, item: str) -> DotsTtsPromptFeatures:
        waveform = load_reference_waveform(item, target_sample_rate=self._sample_rate)
        patch_count = math.ceil(waveform.shape[-1] / self._samples_per_patch)
        padded = F.pad(
            waveform, (0, patch_count * self._samples_per_patch - waveform.shape[-1])
        ).to(device=self._device)
        return DotsTtsPromptFeatures(
            latent_distribution=self._vocoder_inference.extract_latents(padded[None]),
            speaker_embedding=self._speaker_encoder(padded[None]),
            patch_count=patch_count,
        )

    def store_artifact(self, artifact: DotsTtsPromptFeatures) -> DotsTtsPromptFeatures:
        return DotsTtsPromptFeatures(
            latent_distribution=artifact.latent_distribution.detach().to(
                device="cpu", dtype=torch.float32, copy=True
            ),
            speaker_embedding=artifact.speaker_embedding.detach().to(
                device="cpu", dtype=torch.float32, copy=True
            ),
            patch_count=int(artifact.patch_count),
        )

    def load_artifact(self, stored: DotsTtsPromptFeatures) -> DotsTtsPromptFeatures:
        return stored


class DotsTtsReferenceEncoder:
    """Attach prompt latents, speaker embedding, and schedule to a request."""

    def __init__(
        self,
        *,
        tokenizer: object,
        vocoder_inference: VocoderInference,
        speaker_encoder: torch.nn.Module,
        sample_rate: int,
        patch_size: int,
        samples_per_patch: int,
        model_identity: str,
        device: torch.device,
        max_sequence_length: int = DOTS_TTS_MAX_SEQUENCE_LENGTH,
        cache_max_items: int | None = 64,
        cache_max_bytes: int | None = 512 * 1024 * 1024,
    ) -> None:
        if samples_per_patch <= 0:
            raise ValueError(
                f"dots.tts samples_per_patch must be positive, got {samples_per_patch}"
            )
        if max_sequence_length <= 0:
            raise ValueError(
                "dots.tts max_sequence_length must be positive, got "
                f"{max_sequence_length}"
            )
        self._tokenizer = tokenizer
        self._patch_size = int(patch_size)
        self._max_sequence_length = int(max_sequence_length)
        self._service: ReferenceEncodeService[
            str, DotsTtsPromptFeatures, DotsTtsPromptFeatures
        ] = ReferenceEncodeService(
            DotsTtsPromptFeatureHook(
                model_identity=model_identity,
                vocoder_inference=vocoder_inference,
                speaker_encoder=speaker_encoder,
                sample_rate=int(sample_rate),
                samples_per_patch=int(samples_per_patch),
                device=device,
            ),
            max_items=cache_max_items,
            max_bytes=cache_max_bytes,
            log_prefix="dots.tts reference cache",
        )

    def encode_payload(self, payload: StagePayload) -> StagePayload:
        state = load_dots_tts_state(payload)
        if not state.ref_audio_path:
            raise ValueError("dots.tts requires reference audio for every request")
        if not state.prompt_text:
            raise ValueError(
                "dots.tts continuation cloning requires the reference transcript"
            )

        features = self._service.get_or_encode(
            state.ref_audio_path, desc=repr(state.ref_audio_path)
        )
        self._validate_prompt_capacity(state, features)

        schedule_ids = build_dots_tts_schedule_ids(
            tokenizer=self._tokenizer,
            text=state.text,
            prompt_text=state.prompt_text,
            max_audio_patches=state.max_audio_patches,
        )
        if len(schedule_ids) > self._max_sequence_length:
            raise ValueError(
                "dots.tts generation schedule exceeds max_sequence_length: "
                f"schedule_length={len(schedule_ids)} "
                f"max_sequence_length={self._max_sequence_length}"
            )

        prompt_latent = self._sample_prompt_latent(features, seed=state.seed)
        state.prompt_latent = prompt_latent
        state.spk_emb = features.speaker_embedding
        state.prompt_patch_count = int(prompt_latent.shape[1]) // self._patch_size
        state.generation_schedule_ids = schedule_ids
        state.prompt_tokens = len(schedule_ids)
        return store_dots_tts_state(payload, state)

    def _sample_prompt_latent(
        self, features: DotsTtsPromptFeatures, *, seed: int | None
    ) -> torch.Tensor:
        # note (chenyang): the AR loop regenerates the final prompt patch as its
        # first decode step, so the conditioning latents stop one patch short.
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        mean, log_std = features.latent_distribution.chunk(2, dim=1)
        noise = torch.randn(mean.shape, generator=generator, dtype=mean.dtype)
        sampled = (mean + noise * log_std.exp()).transpose(1, 2)
        return sampled[:, : -self._patch_size].contiguous()

    @staticmethod
    def _validate_prompt_capacity(
        state: DotsTtsState, features: DotsTtsPromptFeatures
    ) -> None:
        if features.patch_count < 2:
            raise ValueError(
                "dots.tts reference audio is too short for continuation cloning: "
                f"prompt_audio_patch_count={features.patch_count}"
            )
        if state.max_audio_patches <= features.patch_count:
            raise ValueError(
                "dots.tts max_audio_patches must exceed the prompt audio patch "
                f"count: max_audio_patches={state.max_audio_patches} "
                f"prompt_audio_patch_count={features.patch_count}"
            )


__all__ = [
    "REFERENCE_TRIM_TOP_DB",
    "DotsTtsPromptFeatureHook",
    "DotsTtsPromptFeatures",
    "DotsTtsReferenceEncoder",
    "load_reference_waveform",
]
