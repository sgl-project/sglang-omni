# SPDX-License-Identifier: Apache-2.0
"""dots.tts-specific continuous-latent head for an SGLang Qwen2 backbone."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from einops import rearrange
from torch import nn


@dataclass
class DotsFlowState:
    fm_sequence: torch.Tensor
    fm_cfg_sequence: torch.Tensor
    fm_null_g_cond: torch.Tensor
    fm_capacity: int
    g_cond: torch.Tensor | None
    rng_state: torch.Tensor | None = None
    fm_seq_len: int = 0
    patch_encoder_state: Any = None
    dit_state: Any = None
    prompt_patches: torch.Tensor | None = None
    drop_regenerated_prompt_patch: bool = False
    suppress_first_eos_check: bool = False
    decoded_patches: int = 0
    slot: int | None = None
    all_mods: torch.Tensor | None = None


@dataclass(frozen=True)
class DotsFlowStep:
    latent_patch: torch.Tensor
    feedback_embedding: torch.Tensor
    finished: bool
    emit: bool


class DotsTTSFlowHead(nn.Module):
    """Patch encoder, Flow/MeanFlow DiT and EOS head from the dots model."""

    _LENGTH_BUCKETS = (64, 128, 256, 512)

    def __init__(
        self,
        config_dict: dict[str, Any],
        *,
        llm_hidden_size: int,
        latent_stats_path: str,
        optimize: bool,
    ) -> None:
        super().__init__()
        from dots_tts.models.dots_tts.config import ModelConfig
        from dots_tts.models.dots_tts.core import IOHelper
        from dots_tts.modules.backbone.dit import DiT
        from dots_tts.modules.backbone.encoder import VAESemanticEncoder

        config = ModelConfig.model_validate(config_dict)
        self.config = config
        self.fm_hidden_size = int(config.DiT.hidden_size)
        self.hidden_patch_size = 1
        self.latent_patch_size = int(config.patch_size)
        self.latent_dim = int(config.latent_dim)
        self.optimize = bool(optimize)
        self.mode = (
            "meanflow"
            if config.meanflow is not None and config.meanflow.enabled
            else "flow_matching"
        )
        dit_mode = (
            "meanflow"
            if self.mode == "meanflow" and config.meanflow.use_duration_embedding
            else "flow_matching"
        )

        self.patch_encoder = VAESemanticEncoder(
            in_dim=self.latent_dim,
            out_dim=int(llm_hidden_size),
            config=config,
        )
        self.hidden_proj = nn.Linear(int(llm_hidden_size), self.fm_hidden_size)
        self.latent_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.coordinate_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.xvec_proj = nn.Sequential(
            nn.Linear(int(config.campplus_embedding_size), self.fm_hidden_size),
            nn.LayerNorm(self.fm_hidden_size),
        )
        self.velocity_field_predictor = DiT(
            in_dim=self.fm_hidden_size,
            out_dim=self.latent_dim,
            transformer_config=config.DiT,
            mode=dit_mode,
        )
        self.eos_proj = nn.Sequential(
            nn.Linear(int(llm_hidden_size), int(llm_hidden_size)),
            nn.SiLU(),
            nn.Linear(int(llm_hidden_size), 2),
        )
        self.io = IOHelper(Path(latent_stats_path))
        self._patch_inference: Any = None
        self._dit_solver: Any = None
        self._tail: Any = None
        self._batched_nfe: int | None = None
        self._eos_pinned: torch.Tensor | None = None
        self._eos_event: torch.cuda.Event | None = None

    def _bucket(self, requested: int) -> int:
        if requested <= 0:
            raise ValueError("dots.tts max audio patch count must be positive")
        for bucket in self._LENGTH_BUCKETS:
            if requested <= bucket:
                return bucket
        raise ValueError(
            f"dots.tts supports at most {self._LENGTH_BUCKETS[-1]} audio patches"
        )

    def _patch_encoder_inference(self):
        if self._patch_inference is None:
            from dots_tts.modules.backbone.encoder_inference import (
                SemanticEncoderInference,
            )

            self._patch_inference = SemanticEncoderInference(self.patch_encoder)
        return self._patch_inference

    def _solver(self):
        if self._dit_solver is None:
            from dots_tts.modules.backbone.dit_inference import (
                DiTInferenceContext,
                DiTSolver,
            )

            self._dit_solver = DiTSolver(
                DiTInferenceContext.from_core(self),
                optimize=self.optimize,
                bucket_resolver=self._bucket,
                meanflow=self.mode == "meanflow",
            )
        return self._dit_solver

    @property
    def is_batched(self) -> bool:
        return self._tail is not None

    def init_batched_tail(
        self,
        *,
        num_slots: int,
        nfe: int,
        max_audio_patches: int,
        optimize: bool = False,
    ) -> None:
        if self.mode != "meanflow":
            # note (luojiaxuan): flow-matching checkpoints (SOAR, base) need the
            # CFG double branch, which the batched tail does not implement. They
            # serve on the single-request solver instead.
            raise ValueError(
                "dots.tts continuous batching requires a MeanFlow checkpoint; "
                f"this checkpoint is {self.mode}. Serve it with "
                "max_running_requests=1 (see examples/configs/dots_tts_soar.yaml)"
            )
        from sglang_omni.models.dots_tts.tail import (
            DotsTtsAcousticTail,
            DotsTtsTailSpec,
            fuse_dit_for_inference,
        )

        parameter = next(self.parameters())
        self._tail = DotsTtsAcousticTail(
            dit=fuse_dit_for_inference(self),
            coordinate_proj=self.coordinate_proj,
            latent_proj=self.latent_proj,
            patch_encoder=self.patch_encoder,
            spec=DotsTtsTailSpec(
                nfe=int(nfe),
                patch_capacity=int(max_audio_patches) + 1,
                num_slots=int(num_slots),
                hidden_patch_size=self.hidden_patch_size,
                latent_patch_size=self.latent_patch_size,
                latent_dim=self.latent_dim,
                fm_hidden_size=self.fm_hidden_size,
            ),
            device=parameter.device,
            dtype=parameter.dtype,
            optimize=optimize,
        )
        self._batched_nfe = int(nfe)

    def validate_request(
        self,
        *,
        num_steps: int,
        ode_method: str,
        prompt_patch_count: int | None = None,
        total_span_count: int | None = None,
    ) -> None:
        if not self.is_batched:
            return
        if int(num_steps) != self._batched_nfe:
            raise ValueError(
                "dots.tts num_steps is fixed for continuous batching: "
                f"requested={num_steps} engine={self._batched_nfe}"
            )
        if str(ode_method) != "euler":
            raise ValueError("dots.tts continuous batching currently requires euler")
        if prompt_patch_count is not None and int(prompt_patch_count) <= 0:
            raise ValueError(
                "dots.tts continuous batching requires reference audio with its transcript"
            )
        # The patch-encoder KV pool is the binding pool: it holds one row per
        # audio span, so prompt + generated spans must fit patch_capacity.
        max_spans = int(self._tail.spec.patch_capacity)
        if total_span_count is not None and int(total_span_count) > max_spans:
            raise ValueError(
                f"dots.tts request needs {total_span_count} audio spans but the "
                f"engine tail fits {max_spans}; raise the latent_engine "
                "max_generate_length or lower the request limit"
            )

    @torch.inference_mode()
    def new_request(
        self,
        *,
        max_audio_patch_count: int,
        prompt_latents: torch.Tensor | None,
        speaker_embedding: torch.Tensor | None,
        speaker_scale: float,
        rng: int | torch.Tensor | None,
    ) -> tuple[DotsFlowState, torch.Tensor | None]:
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype
        if self.is_batched:
            if prompt_latents is None or prompt_latents.numel() == 0:
                raise ValueError(
                    "dots.tts continuous batching requires reference audio"
                )
            slot = self._tail.acquire_slot()
            try:
                self._tail.initialize_slot_rng(slot, rng)
                g_cond = None
                if speaker_embedding is not None:
                    speaker_embedding = speaker_embedding.to(device=device, dtype=dtype)
                    g_cond = self.xvec_proj(speaker_embedding * float(speaker_scale))
                grid = torch.linspace(
                    0.0,
                    1.0,
                    int(self._batched_nfe) + 1,
                    device=device,
                    dtype=dtype,
                )
                all_mods = self._tail.dit.build_mods(
                    grid[:-1],
                    duration=grid[1:] - grid[:-1],
                    g_cond=g_cond,
                )
                prompt_latents = prompt_latents.to(device=device, dtype=dtype)
                prompt_embeddings = self._tail.encode_prompt_patches(
                    slot, self._patch_encoder_input(prompt_latents)
                )
                prompt_patches = rearrange(
                    self.io.normalize(prompt_latents).to(dtype=dtype),
                    "b (s p) d -> b s p d",
                    p=self.latent_patch_size,
                )
                empty = torch.empty(0, device=device, dtype=dtype)
                return (
                    DotsFlowState(
                        fm_sequence=empty,
                        fm_cfg_sequence=empty,
                        fm_null_g_cond=empty,
                        fm_capacity=0,
                        g_cond=g_cond,
                        prompt_patches=prompt_patches,
                        drop_regenerated_prompt_patch=True,
                        suppress_first_eos_check=True,
                        slot=slot,
                        all_mods=all_mods,
                    ),
                    prompt_embeddings.unsqueeze(0),
                )
            except BaseException:
                self._tail.release_slot(slot)
                raise
        capacity_patches = (
            self._bucket(max_audio_patch_count)
            if self.optimize
            else int(max_audio_patch_count)
        )
        fm_capacity = capacity_patches * (
            self.hidden_patch_size + self.latent_patch_size
        )
        g_cond = None
        if speaker_embedding is not None:
            speaker_embedding = speaker_embedding.to(device=device, dtype=dtype)
            g_cond = self.xvec_proj(speaker_embedding * float(speaker_scale))
        rng_state = None
        if isinstance(rng, torch.Tensor):
            rng_state = rng.detach().cpu().clone()
        elif rng is not None:
            rng_state = torch.Generator(device=device).manual_seed(int(rng)).get_state()
        state = DotsFlowState(
            fm_sequence=torch.zeros(
                (1, fm_capacity, self.fm_hidden_size), device=device, dtype=dtype
            ),
            fm_cfg_sequence=torch.zeros(
                (1, fm_capacity, self.fm_hidden_size), device=device, dtype=dtype
            ),
            fm_null_g_cond=torch.zeros(
                (1, self.fm_hidden_size), device=device, dtype=dtype
            ),
            fm_capacity=fm_capacity,
            g_cond=g_cond,
            rng_state=rng_state,
        )
        if prompt_latents is None or prompt_latents.numel() == 0:
            return state, None

        prompt_latents = prompt_latents.to(device=device, dtype=dtype)
        patch_input = self._patch_encoder_input(prompt_latents)
        (
            prompt_embeddings,
            state.patch_encoder_state,
        ) = self._patch_encoder_inference().prefill_with_state(
            patch_input,
            None,
            optimize=self.optimize,
            bucket_resolver=self._bucket,
            dtype=dtype,
        )
        state.prompt_patches = rearrange(
            self.io.normalize(prompt_latents).to(dtype=dtype),
            "b (s p) d -> b s p d",
            p=self.latent_patch_size,
        )
        state.drop_regenerated_prompt_patch = True
        state.suppress_first_eos_check = True
        return state, prompt_embeddings

    @torch.inference_mode()
    def replay_feedback(
        self,
        state: DotsFlowState,
        decoded_latent_patches: list[torch.Tensor],
    ) -> torch.Tensor:
        assert decoded_latent_patches
        rows = []
        for patch in decoded_latent_patches:
            normalized = self.io.normalize(patch.to(device=state.fm_sequence.device))
            patch_input = self._patch_encoder_input(
                normalized,
                already_normalized=True,
            )
            if state.slot is None:
                (
                    feedback,
                    state.patch_encoder_state,
                ) = self._patch_encoder_inference().decode_patch_with_state(
                    patch_input,
                    state.patch_encoder_state,
                    optimize=self.optimize,
                    bucket_resolver=self._bucket,
                    dtype=state.fm_sequence.dtype,
                )
            else:
                feedback = self._tail.encode_feedback([state.slot], patch_input)
            rows.append(feedback.reshape(-1, feedback.shape[-1])[-1])
        return torch.stack(rows)

    @torch.inference_mode()
    def initialize_history(
        self,
        state: DotsFlowState,
        *,
        hidden_states: torch.Tensor,
        prompt_span_positions: torch.Tensor,
        audio_span_token_ids: set[int],
        generation_schedule: torch.Tensor,
        prefill_end: int,
        decoded_latent_patches: list[torch.Tensor],
    ) -> None:
        if hidden_states.ndim == 2:
            hidden_states = hidden_states.unsqueeze(0)
        decoded_count = len(decoded_latent_patches)
        assert hidden_states.shape[:2] == (1, int(prefill_end) + decoded_count)
        if state.slot is not None:
            prompt_patches = state.prompt_patches
            if prompt_patches is None or state.all_mods is None:
                raise RuntimeError("dots.tts batched prompt state is incomplete")
            positions = prompt_span_positions.to(
                device=hidden_states.device, dtype=torch.long
            )
            if positions.numel() != prompt_patches.size(1) or bool(
                torch.any(positions <= 0)
            ):
                raise RuntimeError("dots.tts prompt spans do not match prompt latents")
            hidden_rows = self.hidden_proj(hidden_states[0, positions - 1])
            latent_rows = self.latent_proj(prompt_patches[0])
            prompt_fm_rows = torch.cat(
                [hidden_rows[:, None], latent_rows], dim=1
            ).reshape(-1, self.fm_hidden_size)
            if not decoded_count:
                self._tail.seed_fm_history(
                    state.slot,
                    fm_rows=prompt_fm_rows,
                    all_mods=state.all_mods,
                )
                return
            fm_parts = [prompt_fm_rows]
            conditioning_hidden = hidden_states[
                0,
                prefill_end - 1 : prefill_end - 1 + decoded_count,
            ]
            normalized = self.io.normalize(
                torch.cat(
                    [
                        patch.to(device=hidden_states.device)
                        for patch in decoded_latent_patches
                    ],
                    dim=1,
                )
            ).to(dtype=state.fm_sequence.dtype)
            decoded_patches = normalized.reshape(
                decoded_count,
                self.latent_patch_size,
                self.latent_dim,
            )
            decoded_hidden_rows = self.hidden_proj(conditioning_hidden)
            decoded_latent_rows = self.latent_proj(decoded_patches)
            fm_parts.append(
                torch.cat(
                    [decoded_hidden_rows[:, None], decoded_latent_rows],
                    dim=1,
                ).reshape(-1, self.fm_hidden_size)
            )
            self._tail.seed_fm_history(
                state.slot,
                fm_rows=torch.cat(fm_parts, dim=0),
                all_mods=state.all_mods,
            )
            state.drop_regenerated_prompt_patch = False
            state.suppress_first_eos_check = False
            state.decoded_patches = decoded_count
            return
        prompt_patches = state.prompt_patches
        cursor = 0
        for prompt_index, span_position in enumerate(
            prompt_span_positions.detach().cpu().tolist()
        ):
            if span_position > cursor:
                self.append_hidden(
                    state, hidden_states[:, span_position - 1 : span_position]
                )
            if prompt_patches is None:
                raise RuntimeError("dots.tts prompt spans require prompt latents")
            self._append_history(state, prompt_patches[:, prompt_index])
            next_position = span_position + 1
            if (
                next_position < generation_schedule.shape[1]
                and int(generation_schedule[0, next_position]) in audio_span_token_ids
            ):
                self.append_hidden(
                    state, hidden_states[:, span_position : span_position + 1]
                )
            cursor = next_position
        if prefill_end > cursor:
            self.append_hidden(state, hidden_states[:, prefill_end - 1 : prefill_end])
        for patch_index, patch in enumerate(decoded_latent_patches):
            self._append_history(
                state,
                self.io.normalize(patch.to(device=hidden_states.device)).to(
                    dtype=state.fm_sequence.dtype
                ),
            )
            next_hidden_position = prefill_end + patch_index
            self.append_hidden(
                state,
                hidden_states[
                    :,
                    next_hidden_position : next_hidden_position + 1,
                ],
            )
        if decoded_count:
            state.drop_regenerated_prompt_patch = False
            state.suppress_first_eos_check = False
            state.decoded_patches = decoded_count

    @torch.inference_mode()
    def append_hidden(self, state: DotsFlowState, hidden_states: torch.Tensor) -> None:
        hidden = hidden_states[:, -self.hidden_patch_size :]
        projected = self.hidden_proj(hidden)
        null_projected = self.hidden_proj(torch.zeros_like(hidden))
        start, end = self._reserve(state, projected.shape[1])
        state.fm_sequence[:, start:end].copy_(projected)
        state.fm_cfg_sequence[:, start:end].copy_(null_projected)
        state.fm_seq_len = end

    @torch.inference_mode()
    def decode_next(
        self,
        state: DotsFlowState,
        *,
        hidden_states: torch.Tensor,
        num_steps: int,
        ode_method: str,
        guidance_scale: float,
        eos_threshold: float,
    ) -> DotsFlowStep:
        should_check_eos = not (
            state.suppress_first_eos_check and state.decoded_patches == 0
        )
        # note (luojiaxuan): compute the EOS flag now but keep it on the GPU;
        # reading it here would stall the host before any tail work is queued.
        # The readback happens after the DiT/patch-encoder launches below, via
        # a pinned staging buffer + event so the copy overlaps the tail kernels
        # (same idea as the batched path's single deferred readback).
        eos_hit: torch.Tensor | None = None
        if should_check_eos:
            eos_hit = (
                self.eos_proj(hidden_states)
                .softmax(dim=-1)[:, -1, 1]
                .gt(float(eos_threshold))
                .reshape(())
            )
            if eos_hit.is_cuda:
                if self._eos_pinned is None:
                    self._eos_pinned = torch.zeros(
                        (), dtype=torch.bool, pin_memory=True
                    )
                    self._eos_event = torch.cuda.Event()
                self._eos_pinned.copy_(eos_hit, non_blocking=True)
                self._eos_event.record()
        from dots_tts.modules.backbone.dit_inference import DiTSolverState

        if state.dit_state is None:
            state.dit_state = DiTSolverState()
        solver = self._solver()
        dtype = state.fm_sequence.dtype
        device_type = state.fm_sequence.device.type
        with (
            self._request_rng(state),
            torch.autocast(
                device_type=device_type,
                dtype=dtype,
                enabled=device_type == "cuda"
                and dtype in {torch.float16, torch.bfloat16},
            ),
        ):
            normalized_patch = solver.decode_next(
                state.dit_state,
                sequence=state.fm_sequence,
                cfg_sequence=state.fm_cfg_sequence,
                fm_seq_len=state.fm_seq_len,
                null_g_cond=state.fm_null_g_cond,
                g_cond=state.g_cond,
                nfe=int(num_steps),
                ode_method=str(ode_method),
                guidance_scale=float(guidance_scale),
            )
        self._append_history(state, normalized_patch)
        (
            feedback,
            state.patch_encoder_state,
        ) = self._patch_encoder_inference().decode_patch_with_state(
            self._patch_encoder_input(normalized_patch, already_normalized=True),
            state.patch_encoder_state,
            optimize=self.optimize,
            bucket_resolver=self._bucket,
            dtype=state.fm_sequence.dtype,
        )
        if eos_hit is None:
            finished = False
        elif eos_hit.is_cuda:
            self._eos_event.synchronize()
            finished = bool(self._eos_pinned.item())
        else:
            finished = bool(eos_hit.item())
        emit = not state.drop_regenerated_prompt_patch
        state.drop_regenerated_prompt_patch = False
        state.decoded_patches += 1
        return DotsFlowStep(
            latent_patch=self.io.denormalize(normalized_patch),
            feedback_embedding=feedback,
            finished=finished,
            emit=emit,
        )

    @torch.inference_mode()
    def decode_batch(
        self,
        states: list[DotsFlowState],
        *,
        hidden_states: torch.Tensor,
        num_steps: list[int],
        ode_methods: list[str],
        guidance_scales: list[float],
        eos_thresholds: list[float],
        append_hidden: bool,
    ) -> list[DotsFlowStep]:
        if not self.is_batched:
            if len(states) != 1:
                raise RuntimeError("dots.tts single-stream tail received a batch")
            hidden = hidden_states[:1]
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(1)
            if append_hidden:
                self.append_hidden(states[0], hidden)
            return [
                self.decode_next(
                    states[0],
                    hidden_states=hidden,
                    num_steps=num_steps[0],
                    ode_method=ode_methods[0],
                    guidance_scale=guidance_scales[0],
                    eos_threshold=eos_thresholds[0],
                )
            ]

        for steps, method in zip(num_steps, ode_methods, strict=True):
            self.validate_request(num_steps=steps, ode_method=method)
        hidden = hidden_states[:, -1] if hidden_states.ndim == 3 else hidden_states
        probabilities = self.eos_proj(hidden).softmax(dim=-1)[:, 1]
        eos_hits = probabilities.gt(probabilities.new_tensor(eos_thresholds))
        slots = [state.slot for state in states]
        if any(slot is None for slot in slots):
            raise RuntimeError("dots.tts batched request is missing its tail slot")
        normalized = self._tail.sample_patches(
            slots,
            fm_hidden_rows=self.hidden_proj(hidden),
        )
        feedback = self._tail.encode_feedback(
            slots,
            self._patch_encoder_input(normalized, already_normalized=True),
        )
        # Single readback after the tail work is queued so the sync overlaps it.
        eos_flags = eos_hits.tolist()
        finished = [
            (
                False
                if state.suppress_first_eos_check and state.decoded_patches == 0
                else bool(eos_flags[row])
            )
            for row, state in enumerate(states)
        ]
        results = []
        for row, state in enumerate(states):
            emit = not state.drop_regenerated_prompt_patch
            state.drop_regenerated_prompt_patch = False
            state.decoded_patches += 1
            results.append(
                DotsFlowStep(
                    latent_patch=self.io.denormalize(normalized[row : row + 1]),
                    feedback_embedding=feedback[row],
                    finished=finished[row],
                    emit=emit,
                )
            )
        return results

    def release_request(self, state: DotsFlowState | None) -> None:
        if state is not None and state.slot is not None and self._tail is not None:
            slot, state.slot = state.slot, None
            self._tail.release_slot(slot)

    def suspend_request(self, state: DotsFlowState) -> torch.Tensor | None:
        if state.slot is not None:
            rng_state = self._tail.slot_rng_state(state.slot)
            self.release_request(state)
            return rng_state
        return None if state.rng_state is None else state.rng_state.clone()

    @contextmanager
    def _request_rng(self, state: DotsFlowState) -> Iterator[None]:
        if state.rng_state is None:
            yield
            return
        device = state.fm_sequence.device
        cuda_device = None
        if device.type == "cuda":
            cuda_device = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
        with torch.random.fork_rng(
            devices=[] if cuda_device is None else [cuda_device]
        ):
            if cuda_device is None:
                torch.set_rng_state(state.rng_state)
            else:
                torch.cuda.set_rng_state(state.rng_state, cuda_device)
            try:
                yield
            finally:
                state.rng_state = (
                    torch.get_rng_state()
                    if cuda_device is None
                    else torch.cuda.get_rng_state(cuda_device)
                )

    def _patch_encoder_input(
        self, latents: torch.Tensor, *, already_normalized: bool = False
    ) -> torch.Tensor:
        if self.patch_encoder.expects_normalized_input:
            value = latents if already_normalized else self.io.normalize(latents)
        else:
            value = self.io.denormalize(latents) if already_normalized else latents
        return value.to(dtype=next(self.patch_encoder.parameters()).dtype)

    def _append_history(self, state: DotsFlowState, latent_patch: torch.Tensor) -> None:
        projected = self.latent_proj(latent_patch)
        start, end = self._reserve(state, projected.shape[1])
        state.fm_sequence[:, start:end].copy_(projected)
        state.fm_cfg_sequence[:, start:end].copy_(projected)
        state.fm_seq_len = end

    @staticmethod
    def _reserve(state: DotsFlowState, length: int) -> tuple[int, int]:
        start = int(state.fm_seq_len)
        end = start + int(length)
        if end > state.fm_capacity:
            raise RuntimeError(
                f"dots.tts flow history exceeded capacity ({end}>{state.fm_capacity})"
            )
        return start, end


__all__ = ["DotsFlowState", "DotsFlowStep", "DotsTTSFlowHead"]
