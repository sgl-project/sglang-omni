# SPDX-License-Identifier: Apache-2.0
"""Reference-audio conditioning through the shared content-addressed encoder cache."""

from __future__ import annotations

import logging
import threading
from _thread import LockType

import torch

from sglang_omni.models.firered_tts3.campp import CamppEmbedding
from sglang_omni.models.firered_tts3.hf_config import FireRedTTS3Config
from sglang_omni.models.firered_tts3.payload_types import (
    LEFT_CONTEXT_METADATA_KEY,
    MAX_EDIT_SECONDS,
    PROMPT_CHUNK_METADATA_KEY,
    load_firered_tts3_state,
    store_firered_tts3_state,
)
from sglang_omni.models.firered_tts3.profiling import ProfilingProbe
from sglang_omni.models.firered_tts3.redae_encoder import RedAEAudioEncoder
from sglang_omni.models.firered_tts3.weights import load_campp, load_redae_encoder
from sglang_omni.preprocessing.cache_key import (
    hash_media_item,
    reference_path_cache_key,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.reference_encoder import (
    KeyedReferenceEncodeHook,
    ReferenceEncodeService,
)
from sglang_omni.utils.audio import load_audio

ReferenceAudioInput = tuple[str, float | None]
ReferenceConditioning = dict[str, torch.Tensor]

logger = logging.getLogger(__name__)
CACHE_MAX_ITEMS = 256
CACHE_MAX_BYTES = 64 * 1024 * 1024
WARMUP_AUDIO_SECONDS = 8.0


class FireRedReferenceBundle:
    """CAM++ + RedAE-encoder pair for one checkpoint (fp32, autocast at call)."""

    def __init__(self, model_path: str, config: FireRedTTS3Config, device: str) -> None:
        self.config: FireRedTTS3Config = config
        self.device: torch.device = torch.device(device)
        self.encoder: RedAEAudioEncoder = load_redae_encoder(
            model_path, config, device=self.device
        )
        self.spk_extractor: CamppEmbedding | None = (
            load_campp(model_path, device=self.device)
            if config.spk_in_dim is not None
            else None
        )
        self.sample_rate: int = int(config.audio_sample_rate)
        self.pad_multiple: int = (
            int(config.audio_patch_size)
            * int(config.enc_extra_downsample_rate)
            * int(config.patch_size)
        )
        self.redae_scale: float = float(config.redae_scale or 1.0)
        self.lock: LockType = threading.Lock()

    def normalize(
        self, path_or_uri: str, max_seconds: float | None = None
    ) -> torch.Tensor:
        waveform = load_audio(
            path_or_uri,
            source_name="FireRedTTS3 reference",
            target_sample_rate=self.sample_rate,
            mono=True,
        )
        audio = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1)
        if max_seconds is not None:
            audio = audio[:, : int(self.sample_rate * max_seconds)]
        else:
            pass
        padded_sample_count = (
            -(-int(audio.shape[-1]) // self.pad_multiple) * self.pad_multiple
        )
        padding_samples = padded_sample_count - int(audio.shape[-1])
        if padding_samples > 0:
            audio = torch.nn.functional.pad(audio, (padding_samples, 0))
        else:
            pass
        return audio

    @torch.no_grad()
    def encode(
        self, path_or_uri: str, max_seconds: float | None = None
    ) -> ReferenceConditioning:
        audio = self.normalize(path_or_uri, max_seconds).to(self.device)
        probe = ProfilingProbe.create()
        with self.lock:
            with probe.segment("campp"):
                speaker_embedding = (
                    self.spk_extractor(audio, self.sample_rate).float()
                    if self.spk_extractor is not None
                    else None
                )
            with probe.segment("redae_encode"):
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                    latents = self.encoder(audio)
        probe.log(path_or_uri)
        artifact = {"prompt_latents": (latents.float() * self.redae_scale).cpu()}
        if speaker_embedding is not None:
            artifact["speaker_embedding"] = speaker_embedding.cpu()
        else:
            pass
        return artifact

    def warmup(self) -> None:
        """Run one real-shaped dummy clip through CAM++ and the RedAE encoder."""
        samples = int(self.sample_rate * WARMUP_AUDIO_SECONDS)
        samples = -(-samples // self.pad_multiple) * self.pad_multiple
        audio = torch.zeros(1, samples, device=self.device, dtype=torch.float32)
        with self.lock, torch.no_grad():
            if self.spk_extractor is not None:
                self.spk_extractor(audio, self.sample_rate)
            else:
                pass
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
                self.encoder(audio)
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        else:
            pass
        logger.info("FireRedTTS3 reference encoders warmup complete")


class FireRedReferenceHook(
    KeyedReferenceEncodeHook[
        ReferenceAudioInput,
        ReferenceConditioning,
        ReferenceConditioning,
        ReferenceAudioInput,
    ]
):
    model_revision: str = ""
    encoder_id: str = "fireredtts3_redae_campp"
    artifact_kind: str = "reference_conditioning"

    def __init__(
        self, bundle: FireRedReferenceBundle, *, model_id: str, cache_enabled: bool
    ) -> None:
        self.bundle: FireRedReferenceBundle = bundle
        self.model_id: str = model_id
        self.cache_enabled: bool = bool(cache_enabled)
        self.encoder_config_hash: str = (
            f"sr{bundle.sample_rate}:pad{bundle.pad_multiple}:scale{bundle.redae_scale}:spk{bundle.spk_extractor is not None}"
        )

    def input_key(self, item: ReferenceAudioInput) -> str | None:
        if not self.cache_enabled:
            return None
        else:
            pass
        path = item[0]
        return reference_path_cache_key(path, trust_stat=False) or hash_media_item(path)

    def options_key(self, item: ReferenceAudioInput) -> str:
        cap_seconds = item[1]
        return "" if cap_seconds is None else f"cap{cap_seconds}"

    def encode_one(self, item: ReferenceAudioInput) -> ReferenceConditioning:
        path, cap_seconds = item
        return self.bundle.encode(path, max_seconds=cap_seconds)

    @staticmethod
    def store_artifact(artifact: ReferenceConditioning) -> ReferenceConditioning:
        return {
            name: tensor.detach().cpu().float().clone()
            for name, tensor in artifact.items()
        }

    @staticmethod
    def load_artifact(stored: ReferenceConditioning) -> ReferenceConditioning:
        return {name: tensor.clone() for name, tensor in stored.items()}


class FireRedReferenceEncoder:
    """Apply cached reference conditioning to pipeline payloads."""

    def __init__(
        self,
        bundle: FireRedReferenceBundle,
        *,
        model_id: str,
        max_batch_size: int,
        max_batch_wait_ms: float,
        cache_enabled: bool,
    ) -> None:
        self.bundle: FireRedReferenceBundle = bundle
        self.service: ReferenceEncodeService[
            ReferenceAudioInput,
            ReferenceConditioning,
            ReferenceConditioning,
            ReferenceAudioInput,
        ] = ReferenceEncodeService(
            FireRedReferenceHook(
                bundle, model_id=model_id, cache_enabled=cache_enabled
            ),
            max_items=CACHE_MAX_ITEMS,
            max_bytes=CACHE_MAX_BYTES,
            log_prefix="fireredtts3",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_worker_name="fireredtts3-ref-encode-batch",
        )

    def close(self) -> None:
        self.service.close()

    def encode_payload(self, payload: StagePayload) -> StagePayload:
        state = load_firered_tts3_state(payload)
        if state.prompt_audio_path is not None:
            artifact = self.service.get_or_encode(
                (state.prompt_audio_path, None), desc=repr(state.prompt_audio_path)
            )
            state.prompt_latents = artifact["prompt_latents"]
            state.speaker_embedding = artifact.get("speaker_embedding")
            state.left_context_size = int(state.prompt_latents.shape[1])
        else:
            pass
        if state.audio_in_path is not None:
            artifact = self.service.get_or_encode(
                (state.audio_in_path, MAX_EDIT_SECONDS), desc=repr(state.audio_in_path)
            )
            state.audio_in_latents = artifact["prompt_latents"]
        else:
            pass
        return store_firered_tts3_state(payload, state)


def prompt_stream_message(payload: StagePayload) -> OutgoingMessage | None:
    """The prompt-latent chunk handed to the vocoder ahead of the talker."""
    params = payload.request.params
    if not isinstance(params, dict) or not params.get("stream"):
        return None
    else:
        pass
    state = load_firered_tts3_state(payload)
    if state.prompt_latents is None or state.prompt_latents.numel() == 0:
        return None
    else:
        pass
    return OutgoingMessage(
        request_id=payload.request_id,
        type="stream",
        target="vocoder",
        data=state.prompt_latents,
        metadata={
            "modality": "audio_latents",
            "stream": True,
            PROMPT_CHUNK_METADATA_KEY: True,
            LEFT_CONTEXT_METADATA_KEY: int(state.left_context_size),
        },
    )


__all__ = [
    "FireRedReferenceBundle",
    "FireRedReferenceEncoder",
    "FireRedReferenceHook",
    "prompt_stream_message",
]
