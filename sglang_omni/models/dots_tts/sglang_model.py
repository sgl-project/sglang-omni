# SPDX-License-Identifier: Apache-2.0
"""SGLang wrapper for the dots.tts Qwen2 AR backbone and its acoustic tail."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Optional

import torch
import torch.nn as nn
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2Model
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import add_prefix

from sglang_omni.models.dots_tts.components.backbone.dit import (
    DiT,
    fuse_dit_for_inference,
)
from sglang_omni.models.dots_tts.components.backbone.encoder import VAESemanticEncoder
from sglang_omni.models.dots_tts.hf_config import DOTS_TTS_LATENT_STATS_FILENAME
from sglang_omni.models.dots_tts.tail import DotsTtsAcousticTail, DotsTtsTailSpec
from sglang_omni.models.dots_tts.weight_loading import (
    AR_BACKBONE_PREFIX,
    OWNER_AR_BACKBONE,
    DotsTtsWeightReport,
    assert_dots_tts_weight_coverage,
    classify_dots_tts_weight,
    load_dots_tts_latent_stats,
)

logger = logging.getLogger(__name__)

_RUNTIME_PARAM_NAMES = frozenset({"_decode_input_embedding.weight"})
_QWEN2_STACKED_PARAMS = (
    ("qkv_proj", "q_proj", "q"),
    ("qkv_proj", "k_proj", "k"),
    ("qkv_proj", "v_proj", "v"),
    ("gate_up_proj", "gate_proj", 0),
    ("gate_up_proj", "up_proj", 1),
)


class DotsTTSSGLangModel(nn.Module):
    """dots.tts AR backbone on SGLang with the MeanFlow acoustic tail attached.

    ``forward`` owns the Qwen2 hidden-state path that SGLang schedules and
    caches. Latent sampling, the patch-encoder feedback, and the EOS head run in
    :class:`DotsTtsAcousticTail`, batched across the whole running batch once
    that forward returns.
    """

    def __init__(
        self,
        config: Any,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.llm_config = config.llm_config
        self.quant_config = quant_config
        self.model = Qwen2Model(
            self.llm_config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        self.hidden_size = int(self.llm_config.hidden_size)

        model_config = config.dots_model_config()
        self.latent_dim = int(model_config.latent_dim)
        self.latent_patch_size = int(model_config.patch_size)
        self.hidden_patch_size = 1
        self.fm_hidden_size = int(model_config.DiT.hidden_size)
        meanflow = model_config.meanflow
        if meanflow is None or not meanflow.enabled:
            raise ValueError(
                "dots.tts serving currently supports MeanFlow checkpoints only; "
                "this checkpoint has meanflow disabled"
            )

        self.patch_encoder = VAESemanticEncoder(
            in_dim=self.latent_dim,
            out_dim=self.hidden_size,
            config=model_config,
        )
        self.velocity_field_predictor = DiT(
            in_dim=self.fm_hidden_size,
            out_dim=self.latent_dim,
            transformer_config=model_config.DiT,
            mode="meanflow" if meanflow.use_duration_embedding else "flow_matching",
        )
        self.hidden_proj = nn.Linear(self.hidden_size, self.fm_hidden_size)
        self.latent_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.coordinate_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.xvec_proj = nn.Sequential(
            nn.Linear(int(model_config.campplus_embedding_size), self.fm_hidden_size),
            nn.LayerNorm(self.fm_hidden_size),
        )
        self.eos_proj = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.SiLU(),
            nn.Linear(self.hidden_size, 2),
        )

        weight = self.model.embed_tokens.weight
        max_batch_size = self._resolve_max_batch_size()
        self._decode_input_embedding = nn.Embedding(
            max_batch_size,
            self.hidden_size,
            device=weight.device,
            dtype=weight.dtype,
        )
        self._decode_input_embedding.weight.requires_grad_(False)
        self.register_buffer(
            "_decode_input_row_ids",
            torch.arange(max_batch_size, dtype=torch.long, device=weight.device),
            persistent=False,
        )
        self.register_buffer(
            "latent_mean",
            torch.zeros(self.latent_dim, dtype=torch.float32, device=weight.device),
            persistent=False,
        )
        self.register_buffer(
            "latent_std",
            torch.ones(self.latent_dim, dtype=torch.float32, device=weight.device),
            persistent=False,
        )
        self._tail: DotsTtsAcousticTail | None = None

    @staticmethod
    def _resolve_max_batch_size() -> int:
        try:
            server_args = get_global_server_args()
        except ValueError:
            return 1
        return max(1, int(server_args.max_running_requests))

    # ------------------------------------------------------------------
    # Post-load wiring
    # ------------------------------------------------------------------

    @torch.no_grad()
    def init_tail(
        self,
        checkpoint_dir: str,
        *,
        nfe: int,
        max_audio_patches: int,
    ) -> None:
        """Fuse the tail modules and allocate their pools after weight load.

        Both the fused adaLN DiT and the fused QKV patch encoder copy trained
        weights into new parameters, so this must run once weights are in.
        """
        mean, std = load_dots_tts_latent_stats(
            Path(checkpoint_dir) / DOTS_TTS_LATENT_STATS_FILENAME
        )
        self.latent_mean.copy_(mean.to(self.latent_mean.device))
        self.latent_std.copy_(std.to(self.latent_std.device))

        weight = self._decode_input_embedding.weight
        self._tail = DotsTtsAcousticTail(
            dit=fuse_dit_for_inference(self),
            coordinate_proj=self.coordinate_proj,
            patch_encoder=self.patch_encoder,
            spec=DotsTtsTailSpec(
                nfe=int(nfe),
                # note (chenyang): one spare patch so a request that runs its
                # whole decode budget cannot land exactly on the cache edge.
                patch_capacity=int(max_audio_patches) + 1,
                num_slots=int(self._decode_input_embedding.num_embeddings),
                hidden_patch_size=self.hidden_patch_size,
                latent_patch_size=self.latent_patch_size,
                latent_dim=self.latent_dim,
                fm_hidden_size=self.fm_hidden_size,
            ),
            device=weight.device,
            dtype=weight.dtype,
        )

    @property
    def tail(self) -> DotsTtsAcousticTail:
        if self._tail is None:
            raise RuntimeError(
                "dots.tts acoustic tail is not initialized; call init_tail() "
                "after the checkpoint is loaded"
            )
        return self._tail

    # ------------------------------------------------------------------
    # Latent statistics and conditioning
    # ------------------------------------------------------------------

    def normalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return (latents - self.latent_mean.to(latents.dtype)) / self.latent_std.to(
            latents.dtype
        )

    def denormalize_latents(self, latents: torch.Tensor) -> torch.Tensor:
        return latents * self.latent_std.to(latents.dtype) + self.latent_mean.to(
            latents.dtype
        )

    def patch_encoder_input(
        self, latents: torch.Tensor, *, normalized: bool
    ) -> torch.Tensor:
        if self.patch_encoder.expects_normalized_input:
            return latents if normalized else self.normalize_latents(latents)
        return self.denormalize_latents(latents) if normalized else latents

    @torch.no_grad()
    def speaker_condition(
        self, speaker_embedding: torch.Tensor, speaker_scale: float
    ) -> torch.Tensor:
        """Flow-space speaker conditioning ``g_cond`` for one request."""
        dtype = self._decode_input_embedding.weight.dtype
        scaled = speaker_embedding.to(dtype=dtype) * float(speaker_scale)
        return self.xvec_proj(scaled)

    @torch.no_grad()
    def meanflow_modulations(self, g_cond: torch.Tensor, *, nfe: int) -> torch.Tensor:
        """Per-ODE-step fused AdaLN modulations for one request's ``g_cond``.

        The MeanFlow time grid is fixed for a given ``nfe`` and ``g_cond`` is
        constant across a generation, so the whole schedule is precomputed once
        per request instead of per decode step.
        """
        grid = torch.linspace(
            0.0, 1.0, int(nfe) + 1, device=g_cond.device, dtype=g_cond.dtype
        )
        return self.tail.dit.build_mods(
            grid[:-1], duration=grid[1:] - grid[:-1], g_cond=g_cond.reshape(1, -1)
        )

    @torch.no_grad()
    def prompt_flow_rows(
        self,
        *,
        hidden_states: torch.Tensor,
        prompt_latents: torch.Tensor,
    ) -> torch.Tensor:
        """Projected flow history for one request's prompt audio spans.

        ``hidden_states`` is ``[patches, hidden_size]``: the backbone hidden at
        the token before the first audio span followed by the hidden at every
        prompt span but the last. ``prompt_latents`` is the raw (unnormalized)
        prompt latent sequence ``[patches * latent_patch_size, latent_dim]``.
        Rows interleave one projected hidden with one projected latent patch,
        matching the DiT's ``unit_len`` layout.
        """
        patches = int(hidden_states.size(0))
        hidden_rows = self.hidden_proj(hidden_states).unsqueeze(1)
        latent_rows = self.latent_proj(
            self.normalize_latents(
                prompt_latents.reshape(patches, self.latent_patch_size, self.latent_dim)
            )
        )
        return torch.cat([hidden_rows, latent_rows], dim=1).reshape(
            -1, self.fm_hidden_size
        )

    @torch.no_grad()
    def project_hidden(self, hidden: torch.Tensor) -> torch.Tensor:
        """Flow-space rows for backbone hidden states ``[rows, hidden_size]``."""
        return self.hidden_proj(hidden)

    @torch.no_grad()
    def stop_probability(self, hidden: torch.Tensor) -> torch.Tensor:
        """EOS probability per row for ``hidden`` shaped ``[rows, hidden_size]``."""
        return self.eos_proj(hidden).softmax(dim=-1)[:, 1]

    # ------------------------------------------------------------------
    # Backbone
    # ------------------------------------------------------------------

    def get_input_embeddings(self) -> nn.Module:
        return self.model.get_input_embeddings()

    @torch.no_grad()
    def stage_decode_feedback(self, feedback_embeddings: torch.Tensor) -> torch.Tensor:
        batch_size = int(feedback_embeddings.shape[0])
        weight = self._decode_input_embedding.weight
        weight[:batch_size].copy_(
            feedback_embeddings.to(device=weight.device, dtype=weight.dtype)
        )
        return self._decode_input_row_ids[:batch_size]

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Any = None,
    ) -> LogitsProcessorOutput:
        del pp_proxy_tensors

        if input_embeds is None:
            input_embeds = forward_batch.input_embeds
        is_extend = bool(forward_batch.forward_mode.is_extend())
        if input_embeds is None and not is_extend:
            input_embeds = self._decode_input_embedding(input_ids)
            input_ids = None

        hidden_states = self.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )
        # note (chenyang): prefill returns every position because the tail
        # rebuilds its flow sequence from the prompt audio spans, not just the
        # last token. Sizing the dummy logits by request keeps the scheduler's
        # per-row bookkeeping intact.
        row_count = (
            int(forward_batch.batch_size) if is_extend else int(hidden_states.shape[0])
        )
        return LogitsProcessorOutput(
            next_token_logits=hidden_states.new_empty((row_count, 1)),
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        params_dict = dict(self.named_parameters())
        report = DotsTtsWeightReport()
        loaded_names: set[str] = set()

        for original_name, loaded_weight in weights:
            owner = classify_dots_tts_weight(original_name)
            if owner is None:
                report.skipped.append(original_name)
                continue
            name = (
                original_name[len(AR_BACKBONE_PREFIX) :]
                if owner == OWNER_AR_BACKBONE
                else original_name
            )
            target = self._load_one_weight(params_dict, name, loaded_weight)
            if target is None:
                report.leftovers.append(original_name)
                continue
            loaded_names.add(target)
            report.add_loaded(owner)

        report.missing = sorted(set(params_dict) - loaded_names - _RUNTIME_PARAM_NAMES)
        assert_dots_tts_weight_coverage(report)
        logger.info("%s", report.summary())

    @staticmethod
    def _load_one_weight(
        params_dict: dict[str, nn.Parameter],
        name: str,
        loaded_weight: torch.Tensor,
    ) -> str | None:
        for param_name, weight_name, shard_id in _QWEN2_STACKED_PARAMS:
            if weight_name not in name:
                continue
            stacked_name = name.replace(weight_name, param_name)
            param = params_dict.get(stacked_name)
            if param is None:
                continue
            param.weight_loader(param, loaded_weight, shard_id)
            return stacked_name
        param = params_dict.get(name)
        if param is None:
            return None
        weight_loader = getattr(param, "weight_loader", None)
        if weight_loader is None:
            param.data.copy_(loaded_weight.to(dtype=param.dtype))
        else:
            weight_loader(param, loaded_weight)
        return name


EntryClass = DotsTTSSGLangModel

__all__ = [
    "DotsTTSSGLangModel",
    "EntryClass",
]
