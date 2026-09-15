# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 AR backbone on SGLang: the base and residual stacks in one model."""

from __future__ import annotations

import math
from collections.abc import Iterable
from typing import Any

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.minicpm import MiniCPMDecoderLayer
from torch import nn

from sglang_omni.models.voxcpm2.components.cfm import CfmConfig, UnifiedCFM
from sglang_omni.models.voxcpm2.components.local_dit import VoxCPMLocDiT
from sglang_omni.models.voxcpm2.components.local_encoder import VoxCPMLocEnc
from sglang_omni.models.voxcpm2.components.minicpm import (
    MiniCPM4Config,
    MiniCPMRMSNorm,
    MiniCPMSiluAndMul,
    RopeScalingConfig,
    align_rope_buffers,
)
from sglang_omni.models.voxcpm2.components.projections import VoxCPM2Projections
from sglang_omni.models.weight_loader import default_weight_loader


def rope_settings(lm_config: Any) -> tuple[float, RopeScalingConfig]:
    """Read the rope settings from either shape this config arrives in.

    note (Xinhao Tan): transformers 5 folds rope_theta and rope_scaling into a
    single rope_parameters dict, so a config that came through AutoConfig has
    neither of the two original fields. The checkpoint's own JSON still has
    them, and the parity test builds its config straight from that JSON, so
    both shapes reach this function and dropping either one breaks a path that
    the other path's tests would not catch.
    """
    parameters = getattr(lm_config, "rope_parameters", None)
    if parameters:
        parameters = dict(parameters)
        theta = float(parameters["rope_theta"])
        scaling = parameters
    else:
        theta = float(lm_config.rope_theta)
        scaling = lm_config.rope_scaling
        scaling = scaling if isinstance(scaling, dict) else scaling.to_dict()
    return theta, RopeScalingConfig(
        type=str(scaling.get("type") or scaling["rope_type"]),
        long_factor=list(scaling["long_factor"]),
        short_factor=list(scaling["short_factor"]),
        original_max_position_embeddings=int(
            scaling["original_max_position_embeddings"]
        ),
    )


def local_config(lm_config: Any, overrides: dict[str, Any]) -> MiniCPM4Config:
    """Build a local encoder / DiT config the way upstream derives it from the LM."""
    rope_theta, rope_scaling = rope_settings(lm_config)
    return MiniCPM4Config(
        hidden_size=int(overrides["hidden_dim"]),
        intermediate_size=int(overrides["ffn_dim"]),
        max_position_embeddings=int(lm_config.max_position_embeddings),
        num_attention_heads=int(overrides["num_heads"]),
        num_hidden_layers=int(overrides["num_layers"]),
        num_key_value_heads=int(lm_config.num_key_value_heads),
        rms_norm_eps=float(lm_config.rms_norm_eps),
        rope_theta=rope_theta,
        rope_scaling=rope_scaling,
        scale_depth=float(lm_config.scale_depth),
        use_mup=bool(getattr(lm_config, "use_mup", False)),
        kv_channels=overrides.get("kv_channels"),
    )


def stack_config(base: Any, *, num_layers: int) -> Any:
    """Copy an HF config for one stack, neutralizing SGLang's muP depth scaling."""
    config = base.__class__(**base.to_dict()) if hasattr(base, "to_dict") else base
    config.num_hidden_layers = num_layers
    # note (Xinhao Tan): do not restore the checkpoint's scale_depth here.
    # SGLang's MiniCPMDecoderLayer always multiplies each residual branch by
    # scale_depth / sqrt(num_hidden_layers), with no use_mup check, while
    # VoxCPM2 ships use_mup=False and adds the branch unscaled. Setting
    # scale_depth to sqrt(num_hidden_layers) makes that factor exactly 1.0.
    # Passing the real value silently scales every layer by ~0.265 instead.
    config.scale_depth = math.sqrt(num_layers)
    return config


class VoxCPM2SGLangModel(nn.Module):
    """The base and residual MiniCPM stacks sharing one paged KV pool.

    The two stacks advance in lockstep over identical positions and have
    identical KV geometry, so they are laid out as one flat list of layers with
    unique layer ids rather than two models with two caches.
    """

    graph_feedback_buffer: torch.Tensor | None = None
    last_lm_hidden: torch.Tensor | None = None
    last_residual_hidden: torch.Tensor | None = None

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        self.config = config
        lm_config = getattr(config, "lm_config", None)
        voxcpm_config = getattr(config, "voxcpm2_config", None)
        if lm_config is None or not isinstance(voxcpm_config, dict):
            raise ValueError("VoxCPM2 requires its lm_config and top-level config")

        self.num_base_layers = int(lm_config.num_hidden_layers)
        self.num_residual_layers = int(voxcpm_config.get("residual_lm_num_layers", 0))
        if self.num_residual_layers <= 0:
            raise ValueError("VoxCPM2 requires a positive residual_lm_num_layers")

        base_config = stack_config(lm_config, num_layers=self.num_base_layers)
        residual_config = stack_config(lm_config, num_layers=self.num_residual_layers)

        layers: list[nn.Module] = [
            MiniCPMDecoderLayer(
                base_config,
                layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.base_lm.layers.{layer_id}",
            )
            for layer_id in range(self.num_base_layers)
        ]
        for index in range(self.num_residual_layers):
            layer_id = self.num_base_layers + index
            layer = MiniCPMDecoderLayer(
                residual_config,
                layer_id,
                quant_config=quant_config,
                prefix=f"{prefix}.residual_lm.layers.{index}",
            )
            if voxcpm_config.get("residual_lm_no_rope", False):
                # note (Xinhao Tan): SGLang's MiniCPM attention carries an
                # attn_use_rope switch, but MiniCPMDecoderLayer only reads it
                # off a MiniCPMHybridConfig and hardcodes True otherwise, so
                # the residual stack has to set it after construction.
                layer.self_attn.attn_use_rope = False
            layers.append(layer)
        self.layers = nn.ModuleList(layers)

        # note (Xinhao Tan): upstream rounds RMSNorm and SiLU intermediates to
        # model dtype before multiplication. Fusing across these rounding steps
        # changes the values reaching FSQ, so these operations must preserve
        # upstream's rounding order.
        for layer in self.layers:
            layer.input_layernorm = MiniCPMRMSNorm(
                int(lm_config.hidden_size), eps=float(lm_config.rms_norm_eps)
            )
            layer.post_attention_layernorm = MiniCPMRMSNorm(
                int(lm_config.hidden_size), eps=float(lm_config.rms_norm_eps)
            )
            layer.mlp.act_fn = MiniCPMSiluAndMul()

        hidden_size = int(lm_config.hidden_size)
        eps = float(lm_config.rms_norm_eps)
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        self.embed_tokens = VocabParallelEmbedding(
            int(lm_config.vocab_size),
            hidden_size,
            prefix=(
                f"{prefix}.base_lm.embed_tokens" if prefix else "base_lm.embed_tokens"
            ),
        )
        # note (Xinhao Tan): upstream multiplies the token embedding by
        # scale_emb only when use_mup is set, and VoxCPM2 ships it false, so
        # this is 1.0 for the released checkpoint rather than the config's 12.
        self.scale_emb = (
            float(getattr(lm_config, "scale_emb", 1.0))
            if bool(getattr(lm_config, "use_mup", False))
            else 1.0
        )

        self.base_norm = MiniCPMRMSNorm(hidden_size, eps=eps)
        self.residual_norm = MiniCPMRMSNorm(hidden_size, eps=eps)

        encoder_config = local_config(lm_config, voxcpm_config["encoder_config"])
        dit_config = local_config(lm_config, voxcpm_config["dit_config"])
        self.patch_size = int(voxcpm_config["patch_size"])
        self.feat_dim = int(voxcpm_config["feat_dim"])

        self.projections = VoxCPM2Projections(
            lm_hidden_size=hidden_size,
            encoder_hidden_size=encoder_config.hidden_size,
            dit_hidden_size=dit_config.hidden_size,
            quantization_latent_dim=int(
                voxcpm_config["scalar_quantization_latent_dim"]
            ),
            quantization_scale=int(voxcpm_config["scalar_quantization_scale"]),
        )
        self.feat_encoder = VoxCPMLocEnc(encoder_config, input_dim=self.feat_dim)
        self.feat_decoder = UnifiedCFM(
            in_channels=self.feat_dim,
            cfm_params=CfmConfig(
                **(voxcpm_config["dit_config"].get("cfm_config") or {})
            ),
            estimator=VoxCPMLocDiT(dit_config, in_channels=self.feat_dim),
            mean_mode=bool(voxcpm_config["dit_config"].get("mean_mode", False)),
        )
        self.graph_feedback_buffer = None

    def enable_graph_feedback(self, max_batch_size: int) -> None:
        """Own the static buffer the decode CUDA graph reads its input from.

        Every AR step's input embedding is produced by the local encoder from
        the previous step's sampled latent, so a captured graph must read from
        an address this model controls rather than from forward_batch.
        """
        if max_batch_size <= 0:
            raise ValueError("VoxCPM2 graph feedback buffer needs a positive size")
        parameter = next(self.parameters())
        self.graph_feedback_buffer = torch.zeros(
            (int(max_batch_size), int(self.config.lm_config.hidden_size)),
            device=parameter.device,
            dtype=parameter.dtype,
        )

    def run_stack(
        self,
        layers: Iterable[nn.Module],
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
    ) -> torch.Tensor:
        for layer in layers:
            hidden_states, _ = layer(positions, hidden_states, forward_batch, None)
        return hidden_states

    def forward_base(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, forward_batch: Any
    ) -> torch.Tensor:
        hidden_states = self.run_stack(
            self.layers[: self.num_base_layers], hidden_states, positions, forward_batch
        )
        return self.base_norm(hidden_states)

    def forward_residual(
        self, hidden_states: torch.Tensor, positions: torch.Tensor, forward_batch: Any
    ) -> torch.Tensor:
        hidden_states = self.run_stack(
            self.layers[self.num_base_layers :], hidden_states, positions, forward_batch
        )
        return self.residual_norm(hidden_states)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        audio_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        """Run both stacks for one step and stash what the decode head needs."""
        if (
            self.graph_feedback_buffer is not None
            and forward_batch.forward_mode.is_decode()
        ):
            input_embeds = self.graph_feedback_buffer[: input_ids.shape[0]]
        elif input_embeds is None:
            input_embeds = forward_batch.input_embeds

        lm_hidden = self.forward_base(input_embeds, positions, forward_batch)
        if forward_batch.forward_mode.is_extend() and audio_mask is None:
            raise RuntimeError("VoxCPM2 prefill requires an audio_mask")
        quantized = self.projections.quantize(lm_hidden)
        if audio_mask is None:
            lm_hidden = quantized
            audio_embed = input_embeds
        else:
            mask = audio_mask.to(device=lm_hidden.device, dtype=lm_hidden.dtype)
            mask = mask.unsqueeze(-1)
            lm_hidden = quantized * mask + lm_hidden * (1 - mask)
            audio_embed = input_embeds * mask
        residual_inputs = self.projections.fuse(lm_hidden, audio_embed)
        residual_hidden = self.forward_residual(
            residual_inputs, positions, forward_batch
        )
        self.last_lm_hidden = lm_hidden
        self.last_residual_hidden = residual_hidden

        # note (Xinhao Tan): VoxCPM2 never samples a token - the runner reads
        # the stashed hidden states and overwrites next_token_ids - so these
        # logits exist only to satisfy the return contract. Prefill stashes one
        # row per position, so the row count is the request count only on
        # decode; using it on prefill hands the sampler one row per token.
        if forward_batch.forward_mode.is_extend():
            extend_seq_lens = getattr(forward_batch, "extend_seq_lens", None)
            request_count = (
                int(extend_seq_lens.numel()) if extend_seq_lens is not None else 1
            )
        else:
            request_count = int(lm_hidden.shape[0])
        return LogitsProcessorOutput(
            next_token_logits=lm_hidden.new_empty((request_count, 1)),
            # CUDA graph replay updates tensors, not Python attribute assignments.
            # Return both stacks so the runner reads this replay's output instead
            # of the tensors last assigned during capture or a previous prefill.
            hidden_states=torch.cat((lm_hidden, residual_hidden), dim=-1),
        )

    def set_hidden_states(self, hidden: torch.Tensor) -> None:
        width = int(self.config.lm_config.hidden_size)
        if (
            not isinstance(hidden, torch.Tensor)
            or hidden.ndim != 2
            or hidden.shape[1] != 2 * width
        ):
            raise RuntimeError("VoxCPM2 forward must return both AR hidden states")
        self.last_lm_hidden, self.last_residual_hidden = hidden.split(width, dim=-1)

    @torch.inference_mode()
    def build_input_embeds(
        self,
        text_token: torch.Tensor,
        audio_feat: torch.Tensor,
        text_mask: torch.Tensor,
        audio_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Combine token and latent positions into one embedding sequence.

        Every position carries exactly one of the two, chosen by the masks, so
        the sum is a selection rather than a blend.
        """
        parameter = next(self.parameters())
        device, dtype = parameter.device, parameter.dtype

        text_embed = self.embed_tokens(text_token.to(device)) * self.scale_emb
        patch_embed = self.projections.enc_to_lm_proj(
            self.feat_encoder(audio_feat.to(device=device, dtype=dtype).unsqueeze(0))
        )[0]

        text_mask = text_mask.to(device=device, dtype=dtype).unsqueeze(-1)
        audio_mask = audio_mask.to(device=device, dtype=dtype).unsqueeze(-1)
        return text_mask * text_embed.to(dtype) + audio_mask * patch_embed

    def rows(self, rows: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        """Select one hidden row per request from the last forward.

        Decode stashes one row per request, but prefill stashes every position
        in the batch, so the caller passes the index of each request's last
        position rather than assuming row order matches request order.
        """
        if rows is None:
            return self.last_lm_hidden, self.last_residual_hidden
        index = rows.to(self.last_lm_hidden.device)
        return self.last_lm_hidden[index], self.last_residual_hidden[index]

    @torch.inference_mode()
    def decode_patch(
        self,
        cond: torch.Tensor,
        *,
        inference_timesteps: int,
        cfg_value: float,
        rows: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample one latent patch and fold it back into the next step embedding.

        Returns the patch as ``[B, patch_size, feat_dim]`` and the embedding the
        next step feeds to the base stack.
        """
        lm_hidden, residual_hidden = self.rows(rows)
        mu = self.projections.to_dit(lm_hidden, residual_hidden)
        patch = self.feat_decoder(
            mu=mu,
            n_timesteps=int(inference_timesteps),
            patch_size=self.patch_size,
            cond=cond.transpose(1, 2).contiguous(),
            cfg_value=float(cfg_value),
            noise=noise,
        ).transpose(1, 2)
        embedding = self.projections.enc_to_lm_proj(
            self.feat_encoder(patch.unsqueeze(1))
        )[:, 0, :]
        return patch, embedding

    def stop_flags(self, rows: torch.Tensor | None = None) -> torch.Tensor:
        """Per-request stop decision from the base stack's last hidden state."""
        lm_hidden, _ = self.rows(rows)
        return self.projections.stop_logits(lm_hidden).argmax(dim=-1).bool()

    def write_feedback(self, embedding: torch.Tensor) -> None:
        """Stage the next step's input where a captured decode graph reads it."""
        if self.graph_feedback_buffer is None:
            raise RuntimeError("VoxCPM2 graph feedback buffer is not enabled")
        self.graph_feedback_buffer[: embedding.shape[0]].copy_(embedding)

    # note (Xinhao Tan): SGLang fuses the attention and MLP projections, so the
    # checkpoint's separate q/k/v and gate/up tensors each load into one shard
    # of a merged parameter. Without this the names resolve to parameters the
    # model genuinely does not have.
    STACKED_PARAMS = (
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
    )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        params = dict(self.named_parameters())
        loaded: set[str] = set()
        loaded_shards: dict[str, set[str | int | None]] = {}
        for name, tensor in weights:
            target = map_checkpoint_name(name, self.num_base_layers)
            if target is None:
                continue

            shard_id = None
            if target.startswith("layers."):
                for fused, separate, shard in self.STACKED_PARAMS:
                    if separate in target:
                        target = target.replace(separate, fused)
                        shard_id = shard
                        break

            parameter = params.get(target)
            if parameter is None:
                raise ValueError(
                    f"VoxCPM2 checkpoint weight {name!r} mapped to {target!r}, "
                    "which the AR model does not define"
                )
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            if shard_id is None:
                loader(parameter, tensor)
            else:
                loader(parameter, tensor, shard_id)
            loaded.add(target)
            loaded_shards.setdefault(target, set()).add(shard_id)
        # note (Xinhao Tan): a parameter the checkpoint never reaches keeps its
        # random init and still produces audio, just the wrong audio. The name
        # mapping is hand-written and the fused projections make it easy to
        # miss one, so the load says so rather than leaving it to listening.
        missing = sorted(set(params) - loaded)
        if missing:
            raise ValueError(
                f"VoxCPM2 checkpoint left {len(missing)} parameters at their "
                f"initial values, starting with {missing[:5]}"
            )
        for target, shards in loaded_shards.items():
            if None in shards:
                continue
            expected = {"q", "k", "v"} if ".qkv_proj." in target else {0, 1}
            missing_shards = expected - shards
            if missing_shards:
                raise ValueError(
                    f"VoxCPM2 checkpoint left {target!r} without shards "
                    f"{sorted(missing_shards)}"
                )
        align_rope_buffers(self.feat_encoder)
        align_rope_buffers(self.feat_decoder)
        return loaded


PROJECTION_PREFIXES = (
    "fsq_layer.",
    "enc_to_lm_proj.",
    "lm_to_dit_proj.",
    "res_to_dit_proj.",
    "fusion_concat_proj.",
    "stop_proj.",
    "stop_head.",
)


def map_checkpoint_name(name: str, num_base_layers: int) -> str | None:
    """Map a checkpoint parameter onto this model's module tree."""
    if name.startswith("base_lm.layers."):
        return f"layers.{name.removeprefix('base_lm.layers.')}"
    if name.startswith("residual_lm.layers."):
        rest = name.removeprefix("residual_lm.layers.")
        index, _, tail = rest.partition(".")
        return f"layers.{num_base_layers + int(index)}.{tail}"
    if name == "base_lm.embed_tokens.weight":
        return "embed_tokens.weight"
    if name == "base_lm.norm.weight":
        return "base_norm.weight"
    if name == "residual_lm.norm.weight":
        return "residual_norm.weight"
    if name.startswith(PROJECTION_PREFIXES):
        return f"projections.{name}"
    if name.startswith(("feat_encoder.", "feat_decoder.")):
        return name
    # note (Xinhao Tan): audio_vae weights ride in the same checkpoint but the
    # reference-encode and vocoder stages load them from audiovae.pth into
    # their own float32 copies, so the AR model drops them here.
    return None


EntryClass = VoxCPM2SGLangModel

__all__ = ["EntryClass", "VoxCPM2SGLangModel"]
