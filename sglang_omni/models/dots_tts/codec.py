# SPDX-License-Identifier: Apache-2.0
"""dots.tts reference encoder and AudioVAE model operators."""

from __future__ import annotations

import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from sglang_omni.models.dots_tts.alias_free import install_alias_free_fusion
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

# note (0xtoward): dots.tts is optional on CPU hosts that only import the codec.
if TYPE_CHECKING:
    from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
else:
    pass

logger = logging.getLogger(__name__)

# note (0xtoward): Longer references use eager encoding.
REFERENCE_PATCH_BUCKETS = (8, 12, 16, 20, 24, 32, 40, 48, 64, 80)


def load_module(module: torch.nn.Module, path: Path) -> None:
    mismatch = module.load_state_dict(load_file(path, device="cpu"), strict=False)
    if mismatch.missing_keys or mismatch.unexpected_keys:
        raise RuntimeError(f"Failed to load {path}: {mismatch}")
    else:
        pass


def fold_weight_norm(module: torch.nn.Module) -> None:
    """Fold weight-norm reparametrizations so forward passes stop rewriting weights."""
    for submodule in module.modules():
        if hasattr(submodule, "weight_g"):
            torch.nn.utils.remove_weight_norm(submodule)
        else:
            pass


class ReferenceEncoderGraphs:
    """Replay causal reference encodes whose lookahead fits in the dropped patch."""

    def __init__(
        self,
        inference: VocoderInference,
        *,
        samples_per_patch: int,
        hop_size: int,
        device: torch.device,
    ) -> None:
        config = inference.vocoder.h
        if not config.causal_encoder:
            raise ValueError("Encoder graphs require a causal reference encoder")
        else:
            pass
        lookahead_frames = int(config.get("num_encoder_lookahead", 2))
        if samples_per_patch < lookahead_frames * hop_size:
            raise ValueError(
                "Encoder graphs need the dropped last patch to cover "
                f"the {lookahead_frames}-frame lookahead"
            )
        else:
            pass
        self.inference = inference
        self.hop_size = hop_size
        self.lock = threading.Lock()
        self.graphs: dict[
            int, tuple[torch.cuda.CUDAGraph, torch.Tensor, torch.Tensor]
        ] = {}
        started_seconds = time.perf_counter()
        capture_stream = torch.cuda.Stream(device=device)
        with torch.no_grad():
            for patches in REFERENCE_PATCH_BUCKETS:
                static_input = torch.zeros(
                    1, 1, patches * samples_per_patch, device=device
                )
                capture_stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(capture_stream):
                    for _ in range(2):
                        inference.extract_latents(static_input)
                torch.cuda.current_stream(device).wait_stream(capture_stream)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=capture_stream):
                    static_output = inference.extract_latents(static_input)
                self.graphs[int(static_input.shape[-1])] = (
                    graph,
                    static_input,
                    static_output,
                )
        logger.info(
            f"Reference encoder graphs: buckets={sorted(self.graphs)} "
            f"capture_seconds={time.perf_counter() - started_seconds:.1f}"
        )

    def extract_latents(self, batch: torch.Tensor) -> torch.Tensor:
        length = int(batch.shape[-1])
        bucket = min(
            (samples for samples in self.graphs if samples >= length), default=None
        )
        if batch.shape[0] != 1 or bucket is None:
            return self.inference.extract_latents(batch)
        else:
            graph, static_input, static_output = self.graphs[bucket]
            with self.lock:
                static_input[..., :length].copy_(batch)
                static_input[..., length:].zero_()
                graph.replay()
                return static_output[..., : length // self.hop_size].clone()


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
        load_module(vocoder, root / "vocoder.safetensors")
        load_module(speaker, root / "speaker_encoder.safetensors")
        # note (0xtoward): remove_weight_norm above only folds the decoder.
        fold_weight_norm(vocoder.audio_encoder)
        fold_weight_norm(speaker)
        self.vocoder = vocoder.to(device=torch.device(device)).eval()
        self.speaker = speaker.to(device=torch.device(device)).eval()
        self.inference = VocoderInference(self.vocoder)
        self.patch_size = int(config.patch_size)
        self.latent_dim = int(config.latent_dim)
        self.sample_rate = int(vocoder.sample_rate)
        self.hop_size = int(vocoder.hop_size)
        self.device = torch.device(device)
        self.lock = threading.RLock()
        self.alias_free_fusion_enabled: bool | None = None
        self.encoder_graphs: ReferenceEncoderGraphs | None = None
        self.speaker_streams = threading.local()

    def configure_alias_free_fusion(self, enabled: bool) -> None:
        """Fix the shared codec's decoder mode before creating a vocoder."""
        with self.lock:
            if self.alias_free_fusion_enabled is not None:
                if self.alias_free_fusion_enabled != enabled:
                    raise RuntimeError(
                        "The shared dots.tts codec already has a different "
                        "enable_alias_free_fusion setting"
                    )
                else:
                    pass
            else:
                if enabled:
                    install_alias_free_fusion(self.vocoder.decoder)
                else:
                    pass
                self.alias_free_fusion_enabled = enabled

    @staticmethod
    def reference_load_workers(count: int) -> int:
        return max(1, min(int(count), 8))

    def load_reference_waveform(self, path: str) -> torch.Tensor:
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
    def encode_waveforms(
        self, waveforms: list[torch.Tensor]
    ) -> list[dict[str, torch.Tensor]]:
        if not waveforms:
            return []
        else:
            pass
        lengths = {int(w.shape[-1]) for w in waveforms}
        if len(lengths) != 1:
            raise ValueError(
                "dots.tts batched reference encode requires equal-length "
                f"waveforms, got {sorted(lengths)}; padding is not parity-safe"
            )
        else:
            pass
        length = lengths.pop()
        batch = torch.stack([w.reshape(-1) for w in waveforms]).unsqueeze(1)
        batch = batch.to(self.device)
        audio_lengths = torch.full(
            (len(waveforms),), length, dtype=torch.long, device=self.device
        )
        speaker_batch, speaker_lengths = self.speaker_input(batch, audio_lengths)

        # note (0xtoward): Folded weights let encodes bypass the vocoder state lock.
        if self.device.type == "cuda":
            # note (0xtoward): the speaker model and the AudioVAE encoder are
            # independent. Each encoding thread runs the speaker on its own stream,
            # so it overlaps the encoder and other references' speaker work instead
            # of queueing behind them on the shared stream.
            main_stream = torch.cuda.current_stream(self.device)
            speaker_stream = self.speaker_stream()
            speaker_stream.wait_stream(main_stream)
            with torch.cuda.stream(speaker_stream):
                speaker = self.speaker(speaker_batch, audio_lengths=speaker_lengths)
        else:
            speaker = self.speaker(speaker_batch, audio_lengths=speaker_lengths)
        if self.encoder_graphs is None:
            latent_distribution = self.inference.extract_latents(batch)
        else:
            latent_distribution = self.encoder_graphs.extract_latents(batch)
        if self.device.type == "cuda":
            main_stream.wait_stream(speaker_stream)
        else:
            pass

        frames = int(latent_distribution.shape[-1])
        expected_frames = length // self.hop_size
        if frames != expected_frames:
            raise RuntimeError(
                "dots.tts reference encode produced "
                f"{frames} latent frames for {length} samples, expected "
                f"{expected_frames}; latent frame rate is not hop-aligned"
            )
        else:
            pass
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

    def speaker_stream(self) -> torch.cuda.Stream:
        """The calling thread's stream for the speaker model."""
        stream = getattr(self.speaker_streams, "stream", None)
        if stream is None:
            stream = torch.cuda.Stream(device=self.device)
            self.speaker_streams.stream = stream
        else:
            pass
        return stream

    def speaker_input(
        self, batch: torch.Tensor, audio_lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Use deterministic speaker cropping for long references.

        Pre-truncate to the leading window to avoid upstream global-RNG crop;
        AudioVAE latents still use full-length audio.
        """
        limit = self.speaker_sample_limit()
        if limit is None or batch.shape[-1] <= limit:
            return batch, audio_lengths
        else:
            pass
        cropped = batch[..., :limit].contiguous()
        return cropped, audio_lengths.clamp(max=limit)

    def speaker_sample_limit(self) -> int | None:
        """Samples the speaker encoder keeps, or ``None`` if it never crops."""
        max_seconds = float(getattr(self.speaker, "max_audio_seconds", 0.0) or 0.0)
        if max_seconds <= 0:
            return None
        else:
            pass
        rate = int(getattr(self.speaker, "sample_rate", self.sample_rate))
        return round(rate * max_seconds)

    def compile_speaker_model(self, *, warmup_seconds: float = 5.0) -> None:
        """Compile the CAM++ speaker model with dynamic lengths and warm it before serving."""
        started_seconds = time.perf_counter()
        self.speaker.model = torch.compile(self.speaker.model, dynamic=True)
        samples = int(warmup_seconds * self.sample_rate)
        audio = torch.zeros(1, 1, samples, device=self.device)
        with torch.inference_mode():
            for count in (samples, int(samples * 0.8)):
                lengths = torch.full((1,), count, dtype=torch.long, device=self.device)
                batch, batch_lengths = self.speaker_input(audio[..., :count], lengths)
                self.speaker(batch, audio_lengths=batch_lengths)
        torch.cuda.synchronize(self.device)
        logger.info(
            f"Reference speaker model compiled in {time.perf_counter() - started_seconds:.1f}s"
        )

    def encode_reference(self, path: str) -> dict[str, torch.Tensor]:
        return self.encode_waveforms([self.load_reference_waveform(path)])[0]

    def encode_reference_batch(self, paths: list[str]) -> list[dict[str, torch.Tensor]]:
        if not paths:
            return []
        else:
            pass
        if len(paths) == 1:
            return [self.encode_reference(paths[0])]
        else:
            pass

        with ThreadPoolExecutor(
            max_workers=self.reference_load_workers(len(paths)),
            thread_name_prefix="dots-ref-load",
        ) as pool:
            waveforms = list(pool.map(self.load_reference_waveform, paths))

        results: list[dict[str, torch.Tensor] | None] = [None] * len(paths)
        for group in self.length_groups(waveforms).values():
            encoded = self.encode_waveforms([waveforms[i] for i in group])
            for index, artifact in zip(group, encoded):
                results[index] = artifact
        if any(item is None for item in results):
            raise RuntimeError("dots.tts batched reference encode dropped an item")
        else:
            pass
        return [item for item in results if item is not None]

    @staticmethod
    def length_groups(waveforms: list[torch.Tensor]) -> dict[int, list[int]]:
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
        else:
            pass
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
        else:
            pass
        return codec


class DotsReferenceHook(KeyedReferenceEncodeHook[str, dict, dict, str]):
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
        else:
            pass
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
            DotsReferenceHook(codec, model_id=model_id),
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
        else:
            pass
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
        else:
            pass
        return store_dots_tts_state(payload, state)


__all__ = [
    "DotsAudioCodec",
    "DotsReferenceEncoder",
    "load_dots_audio_codec",
]
