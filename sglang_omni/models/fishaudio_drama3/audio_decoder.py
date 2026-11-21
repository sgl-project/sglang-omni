# SPDX-License-Identifier: Apache-2.0
"""Drama 3 frame embeddings and the codebook tower that predicts codes
1..num_codebooks - 1."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from sglang.srt.mem_cache.memory_pool import HybridLinearKVPool
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from torch import nn

from sglang_omni.models.fishaudio_drama3.config import ReferenceSamplingFallback
from sglang_omni.models.fishaudio_drama3.decoder_layers import (
    DecoderAttention,
    DecoderLayer,
    DecoderMlp,
    DecoderSelfAttention,
    InterleavedRotaryEmbedding,
    PagedDecoderAttention,
    norm_weights,
)
from sglang_omni.models.fishaudio_drama3.frame_kernels import (
    MINIMUM_EMBEDDING_BLOCK,
    embedding_block,
    frame_backbone_input,
    frame_shared_input,
)
from sglang_omni.models.fishaudio_drama3.head_sampling import (
    HeadSampling,
    head_has_forward_hooks,
    head_sampling_unsupported_reason,
)
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.reference_sampling import (
    sample_main_tokens,
    sample_residual_codes,
)
from sglang_omni.models.fishaudio_drama3.sampling import (
    MINIMUM_TEMPERATURE,
    RAS_TEMPERATURE,
    RAS_TOP_K,
    RAS_TOP_P,
    RAS_WINDOW_FRAMES,
    sample_semantic_with_ras,
    sample_top_k_top_p,
)
from sglang_omni.models.fishaudio_drama3.sampling_kernels import (
    FIRST_CODEBOOK_HEAD_STREAM,
    CodebookSampler,
    sample_codebook_head,
    sample_semantic_candidates,
)
from sglang_omni.models.fishaudio_drama3.tower_kernels import (
    FusedTowerLayers,
    fused_tower_layers_unsupported_reason,
)
from sglang_omni.models.fishaudio_drama3.tower_tiles import tower_layer_shape

__all__ = [
    "AudioDecoder",
    "DecoderAttention",
    "DecoderLayer",
    "DecoderMlp",
    "DecoderSelfAttention",
    "InterleavedRotaryEmbedding",
    "PagedDecoderAttention",
    "norm_weights",
]

# note (SunskyXH): the codebook heads draw with top-k at most 30 whatever the
# request asks; wider caps there cost speaker similarity.
CODEBOOK_TOP_K = 30


class AudioDecoder(nn.Module):
    """Frame embeddings and the codebook tower of Drama 3.

    A frame row is [main token id, code 0, ..., code num_codebooks - 1]; code 0
    is the main token's offset in the semantic band.
    """

    def __init__(
        self, config: FishOmniConfig, *, tp_rank: int = 0, tp_size: int = 1
    ) -> None:
        super().__init__()
        decoder_config = config.audio_decoder_config
        self.tp_rank: int = tp_rank
        self.tp_size: int = tp_size
        self.semantic_begin_id = config.semantic_start_token_id
        self.semantic_end_id = config.semantic_end_token_id
        self.codebook_size = decoder_config.vocab_size
        self.num_codebooks = decoder_config.num_codebooks
        self.backbone_codebook_count = decoder_config.vq_backbone_codebooks
        rotary_embedding = InterleavedRotaryEmbedding(
            decoder_config, config.text_config.max_position_embeddings
        )
        self.codebook_embeddings = nn.Embedding(
            self.codebook_size * self.num_codebooks, decoder_config.dim
        )
        self.register_buffer(
            "codebook_row_offsets",
            torch.arange(self.num_codebooks) * self.codebook_size,
            persistent=False,
        )
        self.shared_layers = nn.ModuleList(
            DecoderLayer(decoder_config, rotary_embedding, tp_size)
            for _ in range(decoder_config.shared_enrichment_layers)
        )
        self.residual_layers = nn.ModuleList(
            DecoderLayer(decoder_config, rotary_embedding, tp_size)
            for _ in range(self.num_codebooks - 1)
        )
        self.codebook_heads = nn.ModuleList(
            nn.Linear(decoder_config.dim, self.codebook_size, bias=False)
            for _ in range(self.num_codebooks - 1)
        )
        self.register_buffer(
            "main_token_candidate_ids",
            torch.tensor(
                sorted(
                    {
                        config.eos_token_id,
                        *range(self.semantic_begin_id, self.semantic_end_id + 1),
                    }
                )
            ),
            persistent=False,
        )
        self.register_buffer(
            "sampling_step", torch.zeros((), dtype=torch.int64), persistent=False
        )
        self.sampler: CodebookSampler | None = None
        # Set by the engine for reference sampling; None keeps the samplers above.
        self.reference_sampling_fallback: ReferenceSamplingFallback | None = None
        self.fused_layers: FusedTowerLayers | None = None
        # Set with the fused layers: frame embeddings run as single Triton kernels.
        self.fused_frame_embeddings: bool = False
        # Set with the Triton samplers: codebook heads rank candidates in their
        # projection epilogue for passes up to MAX_FUSED_ROWS rows.
        self.head_sampling: HeadSampling | None = None

    def fused_layers_unsupported_reason(
        self,
        *,
        kv_pool: HybridLinearKVPool,
        tower_layer_ids: list[int],
        context_length: int,
        device: torch.device,
    ) -> str | None:
        if self.tp_size > 1:
            # The fused kernels add each rank's partial sums into the residual unreduced.
            return (
                f"the tower layers are split across {self.tp_size} tensor-parallel "
                f"ranks"
            )
        else:
            attention = self.shared_layers[0].self_attn
            return fused_tower_layers_unsupported_reason(
                layers=[
                    layer.tower_layer_weights()
                    for layer in (*self.shared_layers, *self.residual_layers)
                ],
                head_count=attention.head_count,
                kv_head_count=attention.kv_head_count,
                rotary_dim=attention.rotary_embedding.rotary_dim,
                rotary_cos_sin=attention.rotary_embedding.cos_sin_cache,
                context_length=context_length,
                kv_pool=kv_pool,
                tower_layer_ids=tower_layer_ids,
                device=device,
            )

    def enable_fused_layers(self, device: torch.device) -> None:
        layer = self.shared_layers[0]
        attention = layer.self_attn
        self.fused_layers = FusedTowerLayers(
            shape=tower_layer_shape(
                weights=layer.tower_layer_weights(),
                head_count=attention.head_count,
                kv_head_count=attention.kv_head_count,
                rotary_dim=attention.rotary_embedding.rotary_dim,
                multiprocessor_count=torch.cuda.get_device_properties(
                    device
                ).multi_processor_count,
            ),
            rotary_cos_sin=attention.rotary_embedding.cos_sin_cache,
            device=device,
        )
        self.fused_frame_embeddings = (
            embedding_block(self.codebook_embeddings.embedding_dim)
            >= MINIMUM_EMBEDDING_BLOCK
            and self.codebook_embeddings.weight.dtype == torch.bfloat16
            and self.codebook_embeddings.weight.is_contiguous()
        )

    def enable_head_sampling(self, device: torch.device) -> str | None:
        """Fuse the codebook heads with candidate ranking; returns why not, if not."""
        if self.sampler is None or self.reference_sampling_fallback is not None:
            return "the codebook heads do not sample with the Triton kernels"
        else:
            pass
        reason = head_sampling_unsupported_reason(
            heads=list(self.codebook_heads),
            top_k_limit=CODEBOOK_TOP_K,
            device=device,
        )
        if reason is None:
            self.head_sampling = HeadSampling.allocate(
                codebook_size=self.codebook_size,
                dim=self.codebook_heads[0].weight.shape[1],
                top_k_limit=CODEBOOK_TOP_K,
                minimum_temperature=MINIMUM_TEMPERATURE,
                device=device,
            )
        else:
            pass
        return reason

    def is_semantic(self, token_ids: torch.Tensor) -> torch.Tensor:
        return (token_ids >= self.semantic_begin_id) & (
            token_ids <= self.semantic_end_id
        )

    def codebook_embedding_sum(
        self, frames: torch.Tensor, first_codebook: int, end_codebook: int
    ) -> torch.Tensor:
        """Summed embeddings of codes first_codebook..end_codebook - 1 per frame."""
        if first_codebook == end_codebook:
            return self.codebook_embeddings.weight.new_zeros(
                frames.shape[0], self.codebook_embeddings.embedding_dim
            )
        else:
            table_rows = (
                frames[:, 1 + first_codebook : 1 + end_codebook].clamp(
                    0, self.codebook_size - 1
                )
                + self.codebook_row_offsets[first_codebook:end_codebook]
            )
            return F.embedding_bag(
                table_rows, self.codebook_embeddings.weight, mode="sum"
            )

    def embed_backbone(
        self, frames: torch.Tensor, text_embeddings: torch.Tensor
    ) -> torch.Tensor:
        """Backbone input: semantic frames add codes 1..backbone_codebook_count to the
        token embedding."""
        mixed = (
            text_embeddings
            + self.codebook_embedding_sum(
                frames, 1, 1 + self.backbone_codebook_count
            ).to(text_embeddings.dtype)
        ) * (1.0 / math.sqrt(1.0 + self.backbone_codebook_count))
        return torch.where(
            self.is_semantic(frames[:, 0]).unsqueeze(-1), mixed, text_embeddings
        )

    def backbone_input(
        self, frames: torch.Tensor, text_embedding: nn.Embedding | nn.Module
    ) -> torch.Tensor:
        """embed_backbone of frames and the text embedding of their main tokens, in
        one kernel when the frame embeddings are fused."""
        text_weight = getattr(text_embedding, "weight", None)
        if (
            self.fused_frame_embeddings
            and frames.is_cuda
            and frames.stride(1) == 1
            and isinstance(text_weight, torch.Tensor)
            and text_weight.dtype == self.codebook_embeddings.weight.dtype
            and text_weight.is_contiguous()
            and text_weight.shape[1] == self.codebook_embeddings.embedding_dim
        ):
            return frame_backbone_input(
                frames=frames,
                text_weight=text_weight,
                codebook_weight=self.codebook_embeddings.weight,
                semantic_begin_id=self.semantic_begin_id,
                semantic_end_id=self.semantic_end_id,
                codebook_size=self.codebook_size,
                backbone_codebook_count=self.backbone_codebook_count,
                scale=1.0 / math.sqrt(1.0 + self.backbone_codebook_count),
            )
        else:
            return self.embed_backbone(frames, text_embedding(frames[:, 0]))

    def shared_input(self, frames: torch.Tensor) -> torch.Tensor:
        """Extra input of the first shared layer: codes backbone_codebook_count + 1.. of
        semantic frames."""
        if self.fused_frame_embeddings and frames.is_cuda and frames.stride(1) == 1:
            return frame_shared_input(
                frames=frames,
                codebook_weight=self.codebook_embeddings.weight,
                semantic_begin_id=self.semantic_begin_id,
                semantic_end_id=self.semantic_end_id,
                codebook_size=self.codebook_size,
                first_codebook=1 + self.backbone_codebook_count,
                end_codebook=self.num_codebooks,
            )
        else:
            pass
        return torch.where(
            self.is_semantic(frames[:, 0]).unsqueeze(-1),
            self.codebook_embedding_sum(
                frames, 1 + self.backbone_codebook_count, self.num_codebooks
            ),
            0.0,
        )

    def run_layer(
        self,
        layer: DecoderLayer,
        layer_index: int,
        hidden_states: torch.Tensor,
        code_ids: torch.Tensor,
        label_is_semantic: torch.Tensor,
        shared_input: torch.Tensor | None,
        positions: torch.Tensor,
        attention: DecoderAttention,
        decode_pass: bool,
    ) -> torch.Tensor:
        fused_layers = self.fused_layers
        if (
            fused_layers is not None
            and decode_pass
            and isinstance(attention, PagedDecoderAttention)
            and attention.forward_mode() == ForwardMode.DECODE
            and fused_layers.covers(hidden_states.shape[0])
        ):
            activated, residual = fused_layers.forward_layer(
                weights=layer.tower_layer_weights(),
                layer_index=layer_index,
                hidden_states=hidden_states,
                code_ids=code_ids,
                label_is_semantic=label_is_semantic,
                shared_input=shared_input,
                positions=positions,
                cache_slots=attention.cache_slots(),
                key_cache=attention.key_buffer(layer_index),
                value_cache=attention.value_buffer(layer_index),
                attention=attention.attend,
            )
            normed, _ = layer.final_layernorm(layer.mlp.down_proj(activated), residual)
            return normed
        else:
            return layer(
                hidden_states,
                code_ids,
                label_is_semantic,
                shared_input,
                positions,
                attention,
                layer_index,
            )

    def run_shared_layers(
        self,
        hidden_states: torch.Tensor,
        codebook_zero: torch.Tensor,
        label_is_semantic: torch.Tensor,
        shared_input: torch.Tensor,
        positions: torch.Tensor,
        attention: DecoderAttention,
        decode_pass: bool = False,
    ) -> torch.Tensor:
        for layer_index, layer in enumerate(self.shared_layers):
            hidden_states = self.run_layer(
                layer,
                layer_index,
                hidden_states,
                codebook_zero,
                label_is_semantic,
                shared_input if layer_index == 0 else None,
                positions,
                attention,
                decode_pass,
            )
        return hidden_states

    def prefill(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_frames: torch.Tensor,
        next_frames: torch.Tensor,
        attention: DecoderAttention,
    ) -> None:
        """Fill the tower cache of prompt rows; row p gets frame p + 1 as next frame.

        hidden_states is the backbone's final-norm output (the lm_head input).
        """
        label_is_semantic = self.is_semantic(next_frames[:, 0])
        hidden_states = self.run_shared_layers(
            hidden_states,
            next_frames[:, 1],
            label_is_semantic,
            self.shared_input(input_frames),
            positions,
            attention,
        )
        for residual_index, layer in enumerate(self.residual_layers):
            hidden_states = self.run_layer(
                layer,
                len(self.shared_layers) + residual_index,
                hidden_states,
                next_frames[:, 1 + residual_index],
                label_is_semantic,
                None,
                positions,
                attention,
                False,
            )

    def decode_step(
        self,
        semantic_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_frames: torch.Tensor,
        previous_main_tokens: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        attention: DecoderAttention,
        epsilon_cutoff: torch.Tensor | None = None,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Sample the next int64 frame [rows, 1 + num_codebooks] per backbone row,
        filling the tower cache at its position.

        semantic_logits are the raw lm_head logits (no bias or temperature) of
        hidden_states, the backbone's final-norm output; an end-of-speech frame still
        runs every layer. Only reference sampling reads epsilon_cutoff. The frame is
        written to out when given, and out is returned.
        """
        sampler = self.sampler
        reference_sampling_fallback = self.reference_sampling_fallback
        if sampler is not None and reference_sampling_fallback is None:
            return self.decode_step_in_frame(
                semantic_logits,
                hidden_states,
                positions,
                input_frames,
                previous_main_tokens,
                temperature,
                top_k,
                top_p,
                attention,
                sampler,
                out,
            )
        elif sampler is not None:
            sampler.step.add_(1)
        else:
            pass
        if reference_sampling_fallback is not None:
            assert epsilon_cutoff is not None
            main_tokens, codebook_zero = sample_main_tokens(
                semantic_logits,
                self.main_token_candidate_ids,
                previous_main_tokens,
                temperature,
                top_k,
                top_p,
                epsilon_cutoff,
                self.semantic_begin_id,
                self.codebook_size,
                reference_sampling_fallback,
            )
        else:
            main_tokens, codebook_zero = sample_semantic_with_ras(
                semantic_logits,
                self.main_token_candidate_ids,
                previous_main_tokens,
                temperature,
                top_k,
                top_p,
                self.semantic_begin_id,
                self.codebook_size,
                sampler,
            )
        label_is_semantic = self.is_semantic(main_tokens)
        hidden_states = self.run_shared_layers(
            hidden_states,
            codebook_zero,
            label_is_semantic,
            self.shared_input(input_frames),
            positions,
            attention,
            True,
        )
        if reference_sampling_fallback is not None:
            codebook_top_k = top_k
        else:
            codebook_top_k = top_k.clamp(max=CODEBOOK_TOP_K)
        codes = [codebook_zero]
        for residual_index, (layer, head) in enumerate(
            zip(self.residual_layers, self.codebook_heads)
        ):
            hidden_states = self.run_layer(
                layer,
                len(self.shared_layers) + residual_index,
                hidden_states,
                codes[-1],
                label_is_semantic,
                None,
                positions,
                attention,
                True,
            )
            if reference_sampling_fallback is not None:
                codes.append(
                    sample_residual_codes(
                        head(hidden_states), temperature, codebook_top_k, top_p
                    )
                )
            else:
                codes.append(
                    sample_top_k_top_p(
                        head(hidden_states),
                        temperature,
                        codebook_top_k,
                        top_p,
                        sampler,
                        FIRST_CODEBOOK_HEAD_STREAM + residual_index,
                    )
                )
        frames = torch.stack([main_tokens, *codes], dim=-1)
        if out is None:
            return frames
        else:
            out.copy_(frames)
            return out

    def decode_step_in_frame(
        self,
        semantic_logits: torch.Tensor,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        input_frames: torch.Tensor,
        previous_main_tokens: torch.Tensor,
        temperature: torch.Tensor,
        top_k: torch.Tensor,
        top_p: torch.Tensor,
        attention: DecoderAttention,
        sampler: CodebookSampler,
        out: torch.Tensor | None,
    ) -> torch.Tensor:
        """decode_step with the Triton samplers writing each code straight into its
        frame column: the semantic kernel also yields the semantic labels, and the
        codebook kernels apply the codebook top-k cap themselves."""
        row_count = hidden_states.shape[0]
        if previous_main_tokens.shape[-1] != RAS_WINDOW_FRAMES:
            raise ValueError(
                f"previous_main_tokens must hold the last {RAS_WINDOW_FRAMES} frames; "
                f"got shape {tuple(previous_main_tokens.shape)}"
            )
        else:
            pass
        candidate_token_ids = self.main_token_candidate_ids
        if semantic_logits.shape[-1] != candidate_token_ids.shape[0]:
            semantic_logits = semantic_logits.index_select(-1, candidate_token_ids)
        else:
            pass
        if out is None:
            frames = torch.empty(
                (row_count, 1 + self.num_codebooks),
                dtype=torch.int64,
                device=hidden_states.device,
            )
        else:
            frames = out
        label_is_semantic = torch.empty(
            row_count, dtype=torch.bool, device=hidden_states.device
        )
        sampler.step.add_(1)
        sample_semantic_candidates(
            semantic_logits,
            candidate_token_ids,
            previous_main_tokens,
            temperature,
            top_k,
            top_p,
            self.semantic_begin_id,
            self.codebook_size,
            sampler,
            minimum_temperature=MINIMUM_TEMPERATURE,
            ras_temperature=RAS_TEMPERATURE,
            ras_top_k=RAS_TOP_K,
            ras_top_p=RAS_TOP_P,
            main_tokens_out=frames[:, 0],
            codebook_zero_out=frames[:, 1],
            label_is_semantic_out=label_is_semantic,
            semantic_end_id=self.semantic_end_id,
        )
        hidden_states = self.run_shared_layers(
            hidden_states,
            frames[:, 1],
            label_is_semantic,
            self.shared_input(input_frames),
            positions,
            attention,
            True,
        )
        head_sampling = self.head_sampling
        for residual_index, (layer, head) in enumerate(
            zip(self.residual_layers, self.codebook_heads)
        ):
            hidden_states = self.run_layer(
                layer,
                len(self.shared_layers) + residual_index,
                hidden_states,
                frames[:, 1 + residual_index],
                label_is_semantic,
                None,
                positions,
                attention,
                True,
            )
            stream_id = FIRST_CODEBOOK_HEAD_STREAM + residual_index
            # Forward hooks observe a head's logits through its module call.
            if (
                head_sampling is not None
                and head_sampling.covers(row_count)
                and not head_has_forward_hooks(head)
                and hidden_states.stride(1) == 1
            ):
                head_sampling.sample(
                    head=head,
                    hidden_states=hidden_states,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    sampler=sampler,
                    stream_id=stream_id,
                    out=frames[:, 2 + residual_index],
                )
            else:
                sample_codebook_head(
                    head(hidden_states),
                    temperature,
                    top_k,
                    top_p,
                    sampler,
                    stream_id,
                    MINIMUM_TEMPERATURE,
                    top_k_limit=CODEBOOK_TOP_K,
                    out=frames[:, 2 + residual_index],
                )
        return frames
