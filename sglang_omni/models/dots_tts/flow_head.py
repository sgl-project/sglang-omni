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

from sglang_omni.models.dots_tts.compat import import_dots_tts


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

    LENGTH_BUCKETS = (64, 128, 256, 512)

    def __init__(
        self,
        config_dict: dict[str, Any],
        *,
        llm_hidden_size: int,
        latent_stats_path: str,
        optimize: bool,
    ) -> None:
        super().__init__()
        import_dots_tts()
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
            in_dim=self.latent_dim, out_dim=int(llm_hidden_size), config=config
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
        self.patch_inference: Any = None
        self.dit_solver: Any = None
        self.tail: Any = None
        self.batched_nfe: int | None = None
        self.eos_pinned: torch.Tensor | None = None
        self.eos_event: torch.cuda.Event | None = None
        self.batched_eos_pinned: torch.Tensor | None = None
        self.batched_eos_event: torch.cuda.Event | None = None
        self.batched_eos_host: list[bool] | None = None
        self.batched_eos_pending: int = 0

    def bucket(self, requested: int) -> int:
        if requested <= 0:
            raise ValueError("dots.tts max audio patch count must be positive")
        else:
            pass
        for bucket in self.LENGTH_BUCKETS:
            if requested <= bucket:
                return bucket
            else:
                pass
        raise ValueError(
            f"dots.tts supports at most {self.LENGTH_BUCKETS[-1]} audio patches"
        )

    def patch_encoder_inference(self):
        if self.patch_inference is None:
            import_dots_tts()
            from dots_tts.modules.backbone.encoder_inference import (
                SemanticEncoderInference,
            )

            self.patch_inference = SemanticEncoderInference(self.patch_encoder)
        else:
            pass
        return self.patch_inference

    def solver(self):
        if self.dit_solver is None:
            import_dots_tts()
            from dots_tts.modules.backbone.dit_inference import (
                DiTInferenceContext,
                DiTSolver,
            )

            self.dit_solver = DiTSolver(
                DiTInferenceContext.from_core(self),
                optimize=self.optimize,
                bucket_resolver=self.bucket,
                meanflow=self.mode == "meanflow",
            )
        else:
            pass
        return self.dit_solver

    @property
    def is_batched(self) -> bool:
        return self.tail is not None

    @property
    def batched_tail(self) -> Any | None:
        return self.tail

    def init_batched_tail(
        self,
        *,
        num_slots: int,
        nfe: int,
        max_audio_patches: int,
        optimize: bool = False,
        pad_to_bucket: bool = True,
    ) -> None:
        if self.mode != "meanflow":
            raise ValueError(
                f"dots.tts continuous batching requires a MeanFlow checkpoint; this checkpoint is {self.mode}. Serve it with max_running_requests=1 (see examples/configs/dots_tts_soar.yaml)"
            )
        else:
            pass
        from sglang_omni.models.dots_tts.tail import (
            DotsTtsAcousticTail,
            DotsTtsTailSpec,
            fuse_dit_for_inference,
        )

        parameter = next(self.parameters())
        self.tail = DotsTtsAcousticTail(
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
            pad_to_bucket=pad_to_bucket,
        )
        self.batched_nfe = int(nfe)
        self.prepare_batched_eos_staging(int(num_slots), parameter.device)

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
        else:
            pass
        if int(num_steps) != self.batched_nfe:
            raise ValueError(
                f"dots.tts num_steps is fixed for continuous batching: requested={num_steps} engine={self.batched_nfe}"
            )
        else:
            pass
        if str(ode_method) != "euler":
            raise ValueError("dots.tts continuous batching currently requires euler")
        else:
            pass
        if prompt_patch_count is not None and int(prompt_patch_count) <= 0:
            raise ValueError(
                "dots.tts continuous batching requires reference audio with its transcript"
            )
        else:
            pass
        max_spans = int(self.tail.spec.patch_capacity)
        if total_span_count is not None and int(total_span_count) > max_spans:
            raise ValueError(
                f"dots.tts request needs {total_span_count} audio spans but the engine tail fits {max_spans}; raise the latent_engine max_generate_length or lower the request limit"
            )
        else:
            pass

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
        device, dtype = (parameter.device, parameter.dtype)
        if self.is_batched:
            if prompt_latents is None or prompt_latents.numel() == 0:
                raise ValueError(
                    "dots.tts continuous batching requires reference audio"
                )
            else:
                pass
            slot = self.tail.acquire_slot()
            try:
                self.tail.initialize_slot_rng(slot, rng)
                g_cond = None
                if speaker_embedding is not None:
                    speaker_embedding = speaker_embedding.to(device=device, dtype=dtype)
                    g_cond = self.xvec_proj(speaker_embedding * float(speaker_scale))
                else:
                    pass
                grid = torch.linspace(
                    0.0, 1.0, int(self.batched_nfe) + 1, device=device, dtype=dtype
                )
                all_mods = self.tail.dit.build_mods(
                    grid[:-1], duration=grid[1:] - grid[:-1], g_cond=g_cond
                )
                prompt_latents = prompt_latents.to(device=device, dtype=dtype)
                prompt_embeddings = self.tail.encode_prompt_patches(
                    slot, self.patch_encoder_input(prompt_latents)
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
                self.tail.release_slot(slot)
                raise
        else:
            pass
        capacity_patches = (
            self.bucket(max_audio_patch_count)
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
        else:
            pass
        rng_state = None
        if isinstance(rng, torch.Tensor):
            rng_state = rng.detach().cpu().clone()
        elif rng is not None:
            rng_state = torch.Generator(device=device).manual_seed(int(rng)).get_state()
        else:
            pass
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
            return (state, None)
        else:
            pass
        prompt_latents = prompt_latents.to(device=device, dtype=dtype)
        patch_input = self.patch_encoder_input(prompt_latents)
        (
            prompt_embeddings,
            state.patch_encoder_state,
        ) = self.patch_encoder_inference().prefill_with_state(
            patch_input,
            None,
            optimize=self.optimize,
            bucket_resolver=self.bucket,
            dtype=dtype,
        )
        state.prompt_patches = rearrange(
            self.io.normalize(prompt_latents).to(dtype=dtype),
            "b (s p) d -> b s p d",
            p=self.latent_patch_size,
        )
        state.drop_regenerated_prompt_patch = True
        state.suppress_first_eos_check = True
        return (state, prompt_embeddings)

    @torch.inference_mode()
    def replay_feedback(
        self, state: DotsFlowState, decoded_latent_patches: list[torch.Tensor]
    ) -> torch.Tensor:
        assert decoded_latent_patches
        rows = []
        for patch in decoded_latent_patches:
            normalized = self.io.normalize(patch.to(device=state.fm_sequence.device))
            patch_input = self.patch_encoder_input(normalized, already_normalized=True)
            if state.slot is None:
                (
                    feedback,
                    state.patch_encoder_state,
                ) = self.patch_encoder_inference().decode_patch_with_state(
                    patch_input,
                    state.patch_encoder_state,
                    optimize=self.optimize,
                    bucket_resolver=self.bucket,
                    dtype=state.fm_sequence.dtype,
                )
            else:
                feedback = self.tail.encode_feedback([state.slot], patch_input)
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
        else:
            pass
        decoded_count = len(decoded_latent_patches)
        assert hidden_states.shape[:2] == (1, int(prefill_end) + decoded_count)
        if state.slot is not None:
            prompt_patches = state.prompt_patches
            if prompt_patches is None or state.all_mods is None:
                raise RuntimeError("dots.tts batched prompt state is incomplete")
            else:
                pass
            positions = prompt_span_positions.to(
                device=hidden_states.device, dtype=torch.long
            )
            if positions.numel() != prompt_patches.size(1) or bool(
                torch.any(positions <= 0)
            ):
                raise RuntimeError("dots.tts prompt spans do not match prompt latents")
            else:
                pass
            hidden_rows = self.hidden_proj(hidden_states[0, positions - 1])
            latent_rows = self.latent_proj(prompt_patches[0])
            prompt_fm_rows = torch.cat(
                [hidden_rows[:, None], latent_rows], dim=1
            ).reshape(-1, self.fm_hidden_size)
            if not decoded_count:
                self.tail.seed_fm_history(
                    state.slot, fm_rows=prompt_fm_rows, all_mods=state.all_mods
                )
                return
            else:
                pass
            fm_parts = [prompt_fm_rows]
            conditioning_hidden = hidden_states[
                0, prefill_end - 1 : prefill_end - 1 + decoded_count
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
                decoded_count, self.latent_patch_size, self.latent_dim
            )
            decoded_hidden_rows = self.hidden_proj(conditioning_hidden)
            decoded_latent_rows = self.latent_proj(decoded_patches)
            fm_parts.append(
                torch.cat(
                    [decoded_hidden_rows[:, None], decoded_latent_rows], dim=1
                ).reshape(-1, self.fm_hidden_size)
            )
            self.tail.seed_fm_history(
                state.slot, fm_rows=torch.cat(fm_parts, dim=0), all_mods=state.all_mods
            )
            state.drop_regenerated_prompt_patch = False
            state.suppress_first_eos_check = False
            state.decoded_patches = decoded_count
            return
        else:
            pass
        prompt_patches = state.prompt_patches
        cursor = 0
        for prompt_index, span_position in enumerate(
            prompt_span_positions.detach().cpu().tolist()
        ):
            if span_position > cursor:
                self.append_hidden(
                    state, hidden_states[:, span_position - 1 : span_position]
                )
            else:
                pass
            if prompt_patches is None:
                raise RuntimeError("dots.tts prompt spans require prompt latents")
            else:
                pass
            self.append_history(state, prompt_patches[:, prompt_index])
            next_position = span_position + 1
            if (
                next_position < generation_schedule.shape[1]
                and int(generation_schedule[0, next_position]) in audio_span_token_ids
            ):
                self.append_hidden(
                    state, hidden_states[:, span_position : span_position + 1]
                )
            else:
                pass
            cursor = next_position
        if prefill_end > cursor:
            self.append_hidden(state, hidden_states[:, prefill_end - 1 : prefill_end])
        else:
            pass
        for patch_index, patch in enumerate(decoded_latent_patches):
            self.append_history(
                state,
                self.io.normalize(patch.to(device=hidden_states.device)).to(
                    dtype=state.fm_sequence.dtype
                ),
            )
            next_hidden_position = prefill_end + patch_index
            self.append_hidden(
                state, hidden_states[:, next_hidden_position : next_hidden_position + 1]
            )
        if decoded_count:
            state.drop_regenerated_prompt_patch = False
            state.suppress_first_eos_check = False
            state.decoded_patches = decoded_count
        else:
            pass

    @torch.inference_mode()
    def append_hidden(self, state: DotsFlowState, hidden_states: torch.Tensor) -> None:
        hidden = hidden_states[:, -self.hidden_patch_size :]
        projected = self.hidden_proj(hidden)
        null_projected = self.hidden_proj.bias.view(1, 1, -1).expand_as(projected)
        start, end = self.reserve(state, projected.shape[1])
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
        eos_hit: torch.Tensor | None = None
        if should_check_eos:
            eos_hit = (
                self.eos_proj(hidden_states)
                .softmax(dim=-1)[:, -1, 1]
                .gt(float(eos_threshold))
                .reshape(())
            )
            if eos_hit.is_cuda:
                if self.eos_pinned is None:
                    self.eos_pinned = torch.zeros((), dtype=torch.bool, pin_memory=True)
                    self.eos_event = torch.cuda.Event()
                else:
                    pass
                self.eos_pinned.copy_(eos_hit, non_blocking=True)
                self.eos_event.record()
            else:
                pass
        else:
            pass
        import_dots_tts()
        from dots_tts.modules.backbone.dit_inference import DiTSolverState

        if state.dit_state is None:
            state.dit_state = DiTSolverState()
        else:
            pass
        solver = self.solver()
        dtype = state.fm_sequence.dtype
        device_type = state.fm_sequence.device.type
        with (
            self.request_rng(state),
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
        self.append_history(state, normalized_patch)
        (
            feedback,
            state.patch_encoder_state,
        ) = self.patch_encoder_inference().decode_patch_with_state(
            self.patch_encoder_input(normalized_patch, already_normalized=True),
            state.patch_encoder_state,
            optimize=self.optimize,
            bucket_resolver=self.bucket,
            dtype=state.fm_sequence.dtype,
        )
        if eos_hit is None:
            finished = False
        elif eos_hit.is_cuda:
            self.eos_event.synchronize()
            finished = bool(self.eos_pinned.item())
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
            else:
                pass
            hidden = hidden_states[:1]
            if hidden.ndim == 2:
                hidden = hidden.unsqueeze(1)
            else:
                pass
            if append_hidden:
                self.append_hidden(states[0], hidden)
            else:
                pass
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
        else:
            pass
        if self.has_pending_batched_eos:
            raise RuntimeError(
                "dots.tts batched EOS staging overwritten before resolve_batched_eos"
            )
        else:
            pass
        for steps, method in zip(num_steps, ode_methods, strict=True):
            self.validate_request(num_steps=steps, ode_method=method)
        hidden = hidden_states[:, -1] if hidden_states.ndim == 3 else hidden_states
        probabilities = self.eos_proj(hidden).softmax(dim=-1)[:, 1]
        eos_hits = probabilities.gt(probabilities.new_tensor(eos_thresholds))
        for row, state in enumerate(states):
            if state.suppress_first_eos_check and state.decoded_patches == 0:
                eos_hits[row] = False
            else:
                pass
        slots = [state.slot for state in states]
        if any((slot is None for slot in slots)):
            raise RuntimeError("dots.tts batched request is missing its tail slot")
        else:
            pass
        normalized = self.tail.sample_patches(
            slots, fm_hidden_rows=self.hidden_proj(hidden)
        )
        latent_patches = self.io.denormalize(normalized)
        patch_encoder_input = (
            normalized
            if self.patch_encoder.expects_normalized_input
            else latent_patches
        ).to(dtype=next(self.patch_encoder.parameters()).dtype)
        feedback = self.tail.encode_feedback(slots, patch_encoder_input)
        self.tail.note_decode_cycle()
        self.stage_batched_eos(eos_hits)
        results = []
        for row, state in enumerate(states):
            emit = not state.drop_regenerated_prompt_patch
            state.drop_regenerated_prompt_patch = False
            state.decoded_patches += 1
            results.append(
                DotsFlowStep(
                    latent_patch=latent_patches[row : row + 1],
                    feedback_embedding=feedback[row],
                    finished=False,
                    emit=emit,
                )
            )
        return results

    @property
    def has_pending_batched_eos(self) -> bool:
        return self.batched_eos_pending > 0

    def resolve_batched_eos(self) -> list[bool]:
        """Return staged batched EOS flags. Call before the next decode_batch."""
        n = int(self.batched_eos_pending)
        if n <= 0:
            return []
        else:
            pass
        if self.batched_eos_host is not None:
            flags = self.batched_eos_host
            self.batched_eos_host = None
        else:
            assert self.batched_eos_event is not None
            assert self.batched_eos_pinned is not None
            self.batched_eos_event.synchronize()
            flags = [bool(value) for value in self.batched_eos_pinned[:n].tolist()]
        self.batched_eos_pending = 0
        return flags

    def prepare_batched_eos_staging(self, capacity: int, device: torch.device) -> None:
        """Reset pending EOS staging and size the pinned buffer at init."""
        self.batched_eos_pending = 0
        self.batched_eos_host = None
        if device.type != "cuda" or capacity <= 0:
            self.batched_eos_pinned = None
            self.batched_eos_event = None
            return
        else:
            pass
        self.ensure_batched_eos_capacity(capacity, device)

    def ensure_batched_eos_capacity(self, capacity: int, device: torch.device) -> None:
        if device.type != "cuda" or capacity <= 0:
            return
        else:
            pass
        if (
            self.batched_eos_pinned is None
            or int(self.batched_eos_pinned.numel()) < capacity
        ):
            self.batched_eos_pinned = torch.zeros(
                capacity, dtype=torch.bool, pin_memory=True
            )
            self.batched_eos_event = torch.cuda.Event()
        else:
            pass

    def stage_batched_eos(self, eos_hits: torch.Tensor) -> None:
        if eos_hits.ndim != 1:
            raise RuntimeError(
                f"dots.tts batched EOS flags must be rank-1, got {tuple(eos_hits.shape)}"
            )
        else:
            pass
        n = int(eos_hits.shape[0])
        if n <= 0:
            self.batched_eos_pending = 0
            self.batched_eos_host = None
            return
        else:
            pass
        if eos_hits.is_cuda:
            self.ensure_batched_eos_capacity(n, eos_hits.device)
            assert self.batched_eos_pinned is not None
            assert self.batched_eos_event is not None
            self.batched_eos_pinned[:n].copy_(
                eos_hits.to(dtype=torch.bool), non_blocking=True
            )
            self.batched_eos_event.record()
            self.batched_eos_host = None
        else:
            self.batched_eos_host = [bool(value) for value in eos_hits.tolist()]
        self.batched_eos_pending = n

    def release_request(self, state: DotsFlowState | None) -> None:
        if state is not None and state.slot is not None and (self.tail is not None):
            slot, state.slot = (state.slot, None)
            self.tail.release_slot(slot)
        else:
            pass

    def suspend_request(self, state: DotsFlowState) -> torch.Tensor | None:
        if state.slot is not None:
            rng_state = self.tail.slot_rng_state(state.slot)
            self.release_request(state)
            return rng_state
        else:
            pass
        return None if state.rng_state is None else state.rng_state.clone()

    @contextmanager
    def request_rng(self, state: DotsFlowState) -> Iterator[None]:
        if state.rng_state is None:
            yield
            return
        else:
            pass
        device = state.fm_sequence.device
        cuda_device = None
        if device.type == "cuda":
            cuda_device = (
                device.index
                if device.index is not None
                else torch.cuda.current_device()
            )
        else:
            pass
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

    def patch_encoder_input(
        self, latents: torch.Tensor, *, already_normalized: bool = False
    ) -> torch.Tensor:
        if self.patch_encoder.expects_normalized_input:
            value = latents if already_normalized else self.io.normalize(latents)
        else:
            value = self.io.denormalize(latents) if already_normalized else latents
        return value.to(dtype=next(self.patch_encoder.parameters()).dtype)

    def append_history(self, state: DotsFlowState, latent_patch: torch.Tensor) -> None:
        projected = self.latent_proj(latent_patch)
        start, end = self.reserve(state, projected.shape[1])
        state.fm_sequence[:, start:end].copy_(projected)
        state.fm_cfg_sequence[:, start:end].copy_(projected)
        state.fm_seq_len = end

    @staticmethod
    def reserve(state: DotsFlowState, length: int) -> tuple[int, int]:
        start = int(state.fm_seq_len)
        end = start + int(length)
        if end > state.fm_capacity:
            raise RuntimeError(
                f"dots.tts flow history exceeded capacity ({end}>{state.fm_capacity})"
            )
        else:
            pass
        return (start, end)


__all__ = ["DotsFlowState", "DotsFlowStep", "DotsTTSFlowHead"]
