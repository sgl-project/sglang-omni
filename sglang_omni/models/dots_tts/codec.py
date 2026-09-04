# SPDX-License-Identifier: Apache-2.0
"""dots.tts reference encoder and AudioVAE model operators."""

from __future__ import annotations

import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from sglang_omni.models.dots_tts.compat import import_dots_tts
from sglang_omni.models.dots_tts.payload_types import (
    load_dots_tts_state,
    store_dots_tts_state,
)
from sglang_omni.preprocessing.cache_key import (
    hash_media_item,
    reference_path_cache_key,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.reference_encoder import (
    KeyedReferenceEncodeHook,
    ReferenceEncodeService,
)
from sglang_omni.utils.audio import load_audio
from sglang_omni.utils.checkpoint import resolve_checkpoint

logger = logging.getLogger(__name__)


def _load_module(module: torch.nn.Module, path: Path) -> None:
    mismatch = module.load_state_dict(load_file(path, device="cpu"), strict=False)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        raise RuntimeError(f"Failed to load {path}: {mismatch}")


class DotsAudioCodec:
    """Model-only AudioVAE/speaker bundle shared by reference and vocoder stages."""

    def __init__(self, checkpoint: str, *, device: str) -> None:
        import_dots_tts()
        from dots_tts.models.dots_tts.config import ModelConfig
        from dots_tts.modules.speaker.encoder import SpeakerXVectorFeatures
        from dots_tts.modules.vocoder.bigvgan import AudioVAE
        from dots_tts.modules.vocoder.vocoder_inference import VocoderInference

        root = Path(checkpoint)
        config = ModelConfig.model_validate(
            json.loads((root / "config.json").read_text(encoding="utf-8"))
        )
        vocoder = AudioVAE(config.vocoder).eval()
        vocoder.remove_weight_norm()
        speaker = SpeakerXVectorFeatures(
            sample_rate=vocoder.sample_rate,
            campplus_embedding_size=config.campplus_embedding_size,
            max_audio_seconds=config.xvec_max_audio_seconds,
        ).eval()
        _load_module(vocoder, root / "vocoder.safetensors")
        _load_module(speaker, root / "speaker_encoder.safetensors")
        self.vocoder = vocoder.to(device=torch.device(device)).eval()
        self.speaker = speaker.to(device=torch.device(device)).eval()
        self.inference = VocoderInference(self.vocoder)
        self.patch_size = int(config.patch_size)
        self.latent_dim = int(config.latent_dim)
        self.sample_rate = int(vocoder.sample_rate)
        self.hop_size = int(vocoder.hop_size)
        self.device = torch.device(device)
        self.lock = threading.RLock()

    @staticmethod
    def _reference_load_workers(count: int) -> int:
        return max(1, min(int(count), 8))

    def _load_reference_waveform(self, path: str) -> torch.Tensor:
        waveform = load_audio(
            path,
            source_name="dots.tts reference",
            target_sample_rate=self.sample_rate,
            mono=True,
            trim_top_db=30,
            resample_kwargs={
                "lowpass_filter_width": 64,
                "rolloff": 0.95,
                "resampling_method": "sinc_interp_kaiser",
            },
        )
        audio = torch.as_tensor(waveform, dtype=torch.float32).reshape(1, -1)
        samples_per_patch = self.patch_size * self.hop_size
        target = math.ceil(audio.shape[-1] / samples_per_patch) * samples_per_patch
        return F.pad(audio, (0, target - audio.shape[-1]))

    @torch.inference_mode()
    def _encode_waveforms(
        self, waveforms: list[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        if not waveforms:
            return []
        lengths = {int(w.shape[-1]) for w in waveforms}
        if len(lengths) != 1:
            raise ValueError(
                "dots.tts batched reference encode requires equal-length "
                f"waveforms, got {sorted(lengths)}; padding is not parity-safe"
            )
        length = lengths.pop()
        batch = torch.stack([w.reshape(-1) for w in waveforms]).unsqueeze(1)
        batch = batch.to(self.device)
        audio_lengths = torch.full(
            (len(waveforms),), length, dtype=torch.long, device=self.device
        )
        speaker_batch, speaker_lengths = self._speaker_input(batch, audio_lengths)

        with self.lock:
            speaker = self.speaker(speaker_batch, audio_lengths=speaker_lengths)
            latent_distribution = self.inference.extract_latents(batch)

        frames = int(latent_distribution.shape[-1])
        expected_frames = length // self.hop_size
        if frames != expected_frames:
            raise RuntimeError(
                "dots.tts reference encode produced "
                f"{frames} latent frames for {length} samples, expected "
                f"{expected_frames}; latent frame rate is not hop-aligned"
            )
        return [
            {
                "speaker_embedding": speaker[index : index + 1].detach().cpu().float(),
                "latent_distribution": latent_distribution[index : index + 1]
                .detach()
                .cpu()
                .float(),
            }
            for index in range(len(waveforms))
        ]

    def _speaker_input(
        self, batch: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use deterministic speaker cropping for long references.

        Pre-truncate to the leading window to avoid upstream global-RNG crop;
        AudioVAE latents still use full-length audio.
        """
        limit = self._speaker_sample_limit()
        if limit is None or batch.shape[-1] <= limit:
            return batch, audio_lengths
        cropped = batch[..., :limit].contiguous()
        return cropped, audio_lengths.clamp(max=limit)

    def _speaker_sample_limit(self) -> int | None:
        """Samples the speaker encoder keeps, or ``None`` if it never crops."""
        max_seconds = float(getattr(self.speaker, "max_audio_seconds", 0.0) or 0.0)
        if max_seconds <= 0:
            return None
        rate = int(getattr(self.speaker, "sample_rate", self.sample_rate))
        return round(rate * max_seconds)

    def encode_reference(self, path: str) -> dict[str, torch.Tensor]:
        return self._encode_waveforms([self._load_reference_waveform(path)])[0]

    def encode_reference_batch(self, paths: list[str]) -> list[dict[str, torch.Tensor]]:
        if not paths:
            return []
        if len(paths) == 1:
            return [self.encode_reference(paths[0])]

        with ThreadPoolExecutor(
            max_workers=self._reference_load_workers(len(paths)),
            thread_name_prefix="dots-ref-load",
        ) as pool:
            waveforms = list(pool.map(self._load_reference_waveform, paths))

        results: list[dict[str, torch.Tensor] | None] = [None] * len(paths)
        for group in self._length_groups(waveforms).values():
            encoded = self._encode_waveforms([waveforms[i] for i in group])
            for index, artifact in zip(group, encoded):
                results[index] = artifact
        if any(item is None for item in results):
            raise RuntimeError("dots.tts batched reference encode dropped an item")
        return [item for item in results if item is not None]

    @staticmethod
    def _length_groups(waveforms: list[torch.Tensor]) -> dict[int, list[int]]:
        groups: dict[int, list[int]] = {}
        for index, waveform in enumerate(waveforms):
            groups.setdefault(int(waveform.shape[-1]), []).append(index)
        return groups

    def sample_prompt_latents(
        self, latent_distribution: torch.Tensor, *, seed: int | None
    ) -> torch.Tensor:
        mean, log_std = latent_distribution.chunk(2, dim=1)
        generator = None
        if seed is not None:
            generator = torch.Generator(device="cpu").manual_seed(int(seed))
        noise = torch.randn(mean.shape, dtype=mean.dtype, generator=generator)
        sampled = (mean + noise * torch.exp(log_std)).transpose(1, 2)
        return sampled[:, : -self.patch_size].contiguous()


_CODEC_CACHE: dict[tuple[str, str], DotsAudioCodec] = {}
_CODEC_CACHE_LOCK = threading.Lock()


def load_dots_audio_codec(model_path: str, *, device: str) -> DotsAudioCodec:
    checkpoint = str(Path(resolve_checkpoint(model_path)).resolve())
    key = (checkpoint, str(device))
    with _CODEC_CACHE_LOCK:
        codec = _CODEC_CACHE.get(key)
        if codec is None:
            codec = DotsAudioCodec(checkpoint, device=device)
            _CODEC_CACHE[key] = codec
        return codec


class _DotsReferenceHook(KeyedReferenceEncodeHook[str, dict, dict]):
    model_revision = ""
    encoder_id = "dots_audio_vae_campplus"
    artifact_kind = "reference_conditioning"

    def __init__(self, codec: DotsAudioCodec, *, model_id: str) -> None:
        self.codec = codec
        self.model_id = model_id
        self.encoder_config_hash = (
            f"sr{codec.sample_rate}:patch{codec.patch_size}:latent{codec.latent_dim}"
        )

    def input_key(self, item: str) -> str | None:
        return reference_path_cache_key(item, trust_stat=False) or hash_media_item(item)

    def encode_one(self, item: str) -> dict:
        return self.codec.encode_reference(item)

    def can_encode_batch(self) -> bool:
        if self.codec.device.type != "cuda":
            return True
        return not (
            torch.backends.cuda.matmul.allow_tf32 or torch.backends.cudnn.allow_tf32
        )

    def encode_batch(self, items: list[str]) -> list[dict]:
        return self.codec.encode_reference_batch(list(items))

    @staticmethod
    def store_artifact(artifact: dict) -> dict:
        return {
            name: tensor.detach().cpu().float().clone()
            for name, tensor in artifact.items()
        }

    @staticmethod
    def load_artifact(stored: dict) -> dict:
        return {name: tensor.clone() for name, tensor in stored.items()}


class DotsReferenceEncoder:
    def __init__(
        self,
        codec: DotsAudioCodec,
        *,
        model_id: str,
        max_batch_size: int = 1,
        max_batch_wait_ms: float = 0.0,
    ) -> None:
        self.codec = codec
        self.service = ReferenceEncodeService(
            _DotsReferenceHook(codec, model_id=model_id),
            max_items=256,
            max_bytes=64 * 1024 * 1024,
            log_prefix="dots.tts",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            batch_worker_name="dots-ref-encode-batch",
        )
        logger.info(
            "dots.tts reference encode backend: %s (max_batch_size=%d, "
            "max_batch_wait_ms=%g)",
            "coalesced batch" if self.service.batching_enabled else "per-request",
            max_batch_size,
            max_batch_wait_ms,
        )

    def close(self) -> None:
        self.service.close()

    def encode_payload(self, payload: StagePayload) -> StagePayload:
        state = load_dots_tts_state(payload)
        if state.prompt_audio_path is None:
            return payload
        artifact = self.service.get_or_encode(
            state.prompt_audio_path,
            desc=repr(state.prompt_audio_path),
        )
        state.speaker_embedding = artifact["speaker_embedding"]
        if state.use_prompt_prefill:
            state.prompt_latents = self.codec.sample_prompt_latents(
                artifact["latent_distribution"],
                seed=state.seed,
            )
        return store_dots_tts_state(payload, state)


__all__ = [
    "DotsAudioCodec",
    "DotsReferenceEncoder",
    "load_dots_audio_codec",
]
