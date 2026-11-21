# SPDX-License-Identifier: Apache-2.0
"""Reference-audio conditioning through the shared content-addressed encoder cache."""

from __future__ import annotations

import logging
import threading
from _thread import LockType
from typing import NamedTuple

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


class ReferenceAudioInput(NamedTuple):
    audio_path: str
    max_audio_seconds: float | None
    # Filled by prepare_batch_input in the calling thread, only for requests that reach the batch worker.
    normalized_audio: torch.Tensor | None = None


ReferenceConditioning = dict[str, torch.Tensor]

logger = logging.getLogger(__name__)
CACHE_MAX_ITEMS = 256
CACHE_MAX_BYTES = 64 * 1024 * 1024
WARMUP_AUDIO_SECONDS = 8.0
# note (yucheng): Rows x longest clip per batched RedAE forward; 8 clips of 12 s fill it, longer references split into smaller forwards or run alone.
MAX_BATCH_AUDIO_SECONDS = 96.0


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
        self.batch_sample_budget: int = int(MAX_BATCH_AUDIO_SECONDS * self.sample_rate)
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

    @torch.no_grad()
    def encode_batch(
        self, clips: list[torch.Tensor]
    ) -> list[ReferenceConditioning | Exception]:
        """Encode normalized clips, one right-padded RedAE forward per length-sorted group."""
        # note (yucheng): Sorting keeps padding small, and the budget bounds rows x longest clip so one long reference cannot multiply memory by the batch size.
        groups: list[list[int]] = []
        for index in sorted(range(len(clips)), key=lambda i: clips[i].shape[-1]):
            if (
                groups
                and (len(groups[-1]) + 1) * clips[index].shape[-1]
                <= self.batch_sample_budget
            ):
                groups[-1].append(index)
            else:
                groups.append([index])
        outcomes: dict[int, ReferenceConditioning | Exception] = {}
        probe = ProfilingProbe.create()
        with self.lock:
            for group in groups:
                # note (yucheng): A failed group fails only its own rows, so one oversize clip does not discard its batch-mates. The stored exception drops its traceback because the traceback frames pin the group's activations while the larger groups run.
                try:
                    padded_audio = torch.nn.utils.rnn.pad_sequence(
                        [clips[index][0] for index in group], batch_first=True
                    ).to(self.device)
                    with probe.segment("campp"):
                        # CAM++ pools over time, so each clip goes through it unpadded.
                        speaker_embeddings = (
                            torch.cat(
                                [
                                    self.spk_extractor(
                                        padded_audio[
                                            row : row + 1, : clips[index].shape[-1]
                                        ],
                                        self.sample_rate,
                                    ).float()
                                    for row, index in enumerate(group)
                                ]
                            )
                            if self.spk_extractor is not None
                            else None
                        )
                    with probe.segment("redae_encode"):
                        with torch.autocast(
                            device_type=self.device.type, dtype=torch.bfloat16
                        ):
                            latents = self.encoder(padded_audio)
                    latents = (latents.float() * self.redae_scale).cpu()
                    # note (yucheng): Copying the embeddings to the host after the RedAE launch lets CAM++ and RedAE overlap on the GPU, as in the per-clip encode.
                    if speaker_embeddings is not None:
                        speaker_embeddings = speaker_embeddings.cpu()
                    else:
                        pass
                except Exception as exc:
                    outcomes.update(dict.fromkeys(group, exc.with_traceback(None)))
                else:
                    for row, index in enumerate(group):
                        # note (yucheng): The encoder is causal, so right padding leaves every valid latent frame unchanged.
                        frame_count = (
                            clips[index].shape[-1] // self.encoder.downsample_rate
                        )
                        artifact = {
                            "prompt_latents": latents[
                                row : row + 1, :frame_count
                            ].clone()
                        }
                        if speaker_embeddings is not None:
                            artifact["speaker_embedding"] = speaker_embeddings[
                                row : row + 1
                            ].clone()
                        else:
                            pass
                        outcomes[index] = artifact
        probe.log(f"batch of {len(clips)}")
        return [outcomes[index] for index in range(len(clips))]

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
        return reference_path_cache_key(
            item.audio_path, trust_stat=False
        ) or hash_media_item(item.audio_path)

    def options_key(self, item: ReferenceAudioInput) -> str:
        return "" if item.max_audio_seconds is None else f"cap{item.max_audio_seconds}"

    def encode_one(self, item: ReferenceAudioInput) -> ReferenceConditioning:
        return self.bundle.encode(item.audio_path, max_seconds=item.max_audio_seconds)

    def can_encode_batch(self) -> bool:
        return True

    def can_encode_alone(self) -> bool:
        return True

    def prepare_batch_input(self, item: ReferenceAudioInput) -> ReferenceAudioInput:
        normalized_audio = self.bundle.normalize(
            item.audio_path, item.max_audio_seconds
        )
        # CAM++ cannot embed an empty clip, and rejecting it here keeps its batch-mates unaffected.
        if normalized_audio.shape[-1] == 0:
            raise ValueError("FireRedTTS3 reference audio has no samples")
        else:
            pass
        return ReferenceAudioInput(
            item.audio_path, item.max_audio_seconds, normalized_audio
        )

    def encode_batch(
        self, items: list[ReferenceAudioInput]
    ) -> list[ReferenceConditioning | Exception]:
        assert all(item.normalized_audio is not None for item in items)
        return self.bundle.encode_batch([item.normalized_audio for item in items])

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
                ReferenceAudioInput(
                    audio_path=state.prompt_audio_path, max_audio_seconds=None
                ),
                desc=repr(state.prompt_audio_path),
            )
            state.prompt_latents = artifact["prompt_latents"]
            state.speaker_embedding = artifact.get("speaker_embedding")
            state.left_context_size = int(state.prompt_latents.shape[1])
        else:
            pass
        if state.audio_in_path is not None:
            artifact = self.service.get_or_encode(
                ReferenceAudioInput(
                    audio_path=state.audio_in_path, max_audio_seconds=MAX_EDIT_SECONDS
                ),
                desc=repr(state.audio_in_path),
            )
            state.audio_in_latents = artifact["prompt_latents"]
        else:
            pass
        if state.reference_pad_token_id is not None:
            reference_latents = (
                state.prompt_latents
                if state.task_type == "tts"
                else state.audio_in_latents
            )
            assert reference_latents is not None
            assert reference_latents.shape[1] % self.bundle.config.patch_size == 0
            reference_patch_count = (
                reference_latents.shape[1] // self.bundle.config.patch_size
            )
            reference_pad_position = state.input_ids.index(state.reference_pad_token_id)
            state.input_ids[reference_pad_position : reference_pad_position + 1] = [
                state.reference_pad_token_id
            ] * reference_patch_count
            state.reference_pad_token_id = None
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
