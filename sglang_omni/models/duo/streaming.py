"""Session hooks for continuous causal Realization blocks and H50 conditions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import torch
from sglang.multimodal_gen.runtime.distributed.parallel_state import get_ulysses_ctx
from sglang.multimodal_gen.runtime.models.dits.duo.cache import RealizationKVCache
from sglang.multimodal_gen.runtime.models.dits.duo.model import (
    CausalRealizationModel,
    RealizationSegment,
    RealizationSequence,
)
from sglang.multimodal_gen.runtime.models.dits.duo.sequence import AttentionSpan

from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext, SessionHooks


class RealizationLatentDecoder(Protocol):
    def __call__(
        self, video_latent: torch.Tensor, audio_latent: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor | None]: ...


@dataclass(kw_only=True)
class RealizationSessionState:
    cache: RealizationKVCache
    reference: RealizationSegment
    shared: RealizationSegment
    next_block_index: int = 0
    is_ended: bool = False


def segment_from_h50(
    tokens: torch.Tensor,
    *,
    kind: str,
    block_index: int = -1,
    positions: torch.Tensor | None = None,
    roles: torch.Tensor | None = None,
) -> RealizationSegment:
    if tokens.ndim != 2:
        raise ValueError("H50 tokens must have shape [tokens, hidden]")
    else:
        pass
    token_count = tokens.shape[0]
    if positions is None:
        positions = torch.zeros(token_count, 3, device=tokens.device)
        positions[:, 1] = torch.arange(token_count, device=tokens.device)
    else:
        pass
    if roles is None:
        roles = torch.full((token_count,), -1, dtype=torch.long, device=tokens.device)
    else:
        pass
    return RealizationSegment(
        span=AttentionSpan(kind=kind, token_count=token_count, block_index=block_index),
        modality=1,
        tokens=tokens,
        positions=positions,
        roles=roles,
    )


def cache_for_model(
    model: CausalRealizationModel, *, capacity_tokens: int, plan_capacity_tokens: int
) -> RealizationKVCache:
    ulysses_size, _ = get_ulysses_ctx()
    architecture = model.backbone.arch
    return RealizationKVCache(
        layer_count=architecture.num_layers,
        capacity_tokens=capacity_tokens,
        plan_capacity_tokens=plan_capacity_tokens,
        local_head_count=architecture.num_attention_heads // ulysses_size,
        head_dimension=architecture.attention_head_dim,
        dtype=torch.bfloat16,
        device=model.backbone.video_patch_proj.weight.device,
        use_flash_attention=model.use_flash_attention,
        flash_num_splits=model.flash_num_splits,
    )


class RealizationSessionHooks(SessionHooks):
    """Run one causal Realization block per append and retain clean media K/V."""

    @property
    def device(self) -> torch.device:
        return self.model.backbone.video_patch_proj.weight.device

    @property
    def graph_counts(self) -> tuple[int, int]:
        return self.model.graph_capture_count, self.model.graph_replay_count

    def __init__(
        self,
        model: CausalRealizationModel,
        *,
        capacity_tokens: int,
        plan_capacity_tokens: int,
        denoise_timesteps: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75),
        decoder: RealizationLatentDecoder | None = None,
    ) -> None:
        if not denoise_timesteps or any(
            timestep < 0 or timestep >= 1 for timestep in denoise_timesteps
        ):
            raise ValueError("Denoise timesteps must be in [0, 1)")
        else:
            pass
        if denoise_timesteps[0] != 0 or any(
            right <= left
            for left, right in zip(denoise_timesteps, denoise_timesteps[1:])
        ):
            raise ValueError(
                "Realization timesteps must increase from noise=0 toward clean=1"
            )
        else:
            pass
        self.model = model
        self.capacity_tokens = capacity_tokens
        self.plan_capacity_tokens = plan_capacity_tokens
        self.denoise_timesteps = denoise_timesteps
        self.decoder = decoder
        self.states: dict[SessionIdentity, RealizationSessionState] = {}
        self.idle_cache: RealizationKVCache | None = None

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.close(session_identity)

    @torch.inference_mode()
    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if chunk.modality != "realization_block" or not isinstance(payload.data, dict):
            raise ValueError("Realization session requires realization_block payloads")
        else:
            pass
        state = self.states.get(context.session_identity)
        data = payload.data
        if state is None:
            reference = data.get("reference")
            shared = data.get("shared")
            if not isinstance(reference, RealizationSegment) or not isinstance(
                shared, RealizationSegment
            ):
                raise ValueError(
                    "The first Realization block must provide reference and shared"
                )
            else:
                pass
            state = RealizationSessionState(
                cache=(
                    self.idle_cache
                    if self.idle_cache is not None
                    else cache_for_model(
                        self.model,
                        capacity_tokens=self.capacity_tokens,
                        plan_capacity_tokens=self.plan_capacity_tokens,
                    )
                ),
                reference=reference,
                shared=shared,
            )
            self.idle_cache = None
            state.cache.begin_prefix(
                reference_token_count=reference.span.token_count,
                shared_token_count=shared.span.token_count,
            )
            self.model(
                RealizationSequence(
                    segments=(reference, shared), refiner_segments=(shared,)
                ),
                noisy_timestep=self.model.clean_timestep,
                cache=state.cache,
            )
            self.states[context.session_identity] = state
        else:
            pass
        if state.is_ended:
            raise ValueError("Realization session already ended")
        else:
            pass
        block_index = data.get("block_index")
        if not isinstance(block_index, int) or block_index != state.next_block_index:
            raise ValueError(f"Expected Realization block {state.next_block_index}")
        else:
            pass
        plan = data.get("plan")
        video = data.get("video")
        audio = data.get("audio")
        if not isinstance(plan, RealizationSegment) or not isinstance(
            video, RealizationSegment
        ):
            raise ValueError("Realization block requires plan and video segments")
        else:
            pass
        if audio is not None and not isinstance(audio, RealizationSegment):
            raise ValueError("Realization audio segment must be an RealizationSegment")
        else:
            pass
        media_segments = (video,) if audio is None else (video, audio)
        state.cache.begin_block(
            block_index=block_index,
            plan_token_count=plan.span.token_count,
            media_token_count=sum(
                segment.span.token_count for segment in media_segments
            ),
        )
        state.cache.begin_forward("plan")
        self.model(
            RealizationSequence(
                segments=(plan,), refiner_segments=(state.shared, plan)
            ),
            noisy_timestep=self.model.clean_timestep,
            cache=state.cache,
        )
        video_tokens = video.tokens.float().clone()
        audio_tokens = None if audio is None else audio.tokens.float().clone()
        for timestep, next_timestep in zip(
            self.denoise_timesteps,
            (*self.denoise_timesteps[1:], 1.0),
            strict=True,
        ):
            if context.cancelled.is_set():
                raise RuntimeError("Realization block was cancelled")
            else:
                pass
            current_video = RealizationSegment(
                span=video.span,
                modality=video.modality,
                tokens=video_tokens,
                positions=video.positions,
                roles=video.roles,
            )
            current_audio = (
                None
                if audio is None
                else RealizationSegment(
                    span=audio.span,
                    modality=audio.modality,
                    tokens=audio_tokens,
                    positions=audio.positions,
                    roles=audio.roles,
                )
            )
            state.cache.begin_forward("denoise")
            output = self.model(
                RealizationSequence(
                    segments=(
                        (current_video,)
                        if current_audio is None
                        else (current_video, current_audio)
                    ),
                    refiner_segments=(),
                ),
                noisy_timestep=timestep,
                cache=state.cache,
            )
            delta = next_timestep - timestep
            video_tokens = video_tokens + output.video.float() * delta
            if audio_tokens is not None:
                audio_tokens = audio_tokens + output.audio.float() * delta
            else:
                pass
        clean_video = RealizationSegment(
            span=AttentionSpan(
                kind="clean",
                token_count=video.span.token_count,
                block_index=block_index,
            ),
            modality=video.modality,
            tokens=video_tokens,
            positions=video.positions,
            roles=video.roles,
        )
        clean_audio = (
            None
            if audio is None
            else RealizationSegment(
                span=AttentionSpan(
                    kind="clean",
                    token_count=audio.span.token_count,
                    block_index=block_index,
                ),
                modality=audio.modality,
                tokens=audio_tokens,
                positions=audio.positions,
                roles=audio.roles,
            )
        )
        if context.cancelled.is_set():
            raise RuntimeError("Realization block was cancelled")
        else:
            pass
        state.cache.begin_forward("clean")
        self.model(
            RealizationSequence(
                segments=(
                    (clean_video,)
                    if clean_audio is None
                    else (clean_video, clean_audio)
                ),
                refiner_segments=(),
            ),
            noisy_timestep=self.model.clean_timestep,
            cache=state.cache,
        )
        state.next_block_index += 1
        state.is_ended = chunk.eos
        output_data: dict[str, object] = {
            "video_latent": video_tokens,
            "audio_latent": audio_tokens,
            "block_index": block_index,
            "eos": chunk.eos,
        }
        payload.data = output_data
        if self.decoder is not None:
            video_media, audio_media = self.decoder(video_tokens, audio_tokens)
            context.emit(
                TimedChunk(
                    modality="video",
                    t_start_ms=chunk.t_start_ms,
                    duration_ms=chunk.duration_ms,
                    seq=chunk.seq,
                    payload=video_media,
                    format="realization_video",
                    eos=chunk.eos,
                )
            )
            context.emit(
                TimedChunk(
                    modality="audio",
                    t_start_ms=chunk.t_start_ms,
                    duration_ms=chunk.duration_ms,
                    seq=chunk.seq,
                    payload=audio_media,
                    format="realization_audio",
                    eos=chunk.eos,
                )
            )
        else:
            context.emit(
                TimedChunk(
                    modality="realization_latent",
                    t_start_ms=chunk.t_start_ms,
                    duration_ms=chunk.duration_ms,
                    seq=chunk.seq,
                    payload=output_data,
                    format="realization_latent",
                    eos=chunk.eos,
                )
            )
        return payload

    @torch.inference_mode()
    def close(self, session_identity: SessionIdentity) -> None:
        state = self.states.pop(session_identity, None)
        if (
            state is not None
            and self.model.enable_cuda_graph
            and self.idle_cache is None
        ):
            state.cache.reset()
            self.idle_cache = state.cache
        else:
            pass

    def usage(self, session_identity: SessionIdentity) -> ResourceUsage:
        state = self.states.get(session_identity)
        if state is None:
            return ResourceUsage()
        else:
            pass
        cache_bytes = sum(
            layer.k.numel() * layer.k.element_size()
            + layer.v.numel() * layer.v.element_size()
            for layer in state.cache.layers
        )
        return ResourceUsage(
            kv_tokens=state.cache.committed_token_count,
            bytes=cache_bytes,
        )


@dataclass(kw_only=True)
class LiveSessionInput:
    reference_rgb: bytes
    height: int
    width: int
    description: dict[str, str]
    seed: int = 42
    sample_id: str = "realtime"


def discard_latents(chunk: TimedChunk) -> None:
    """The live owner emits decoded media instead of the internal latent event."""
