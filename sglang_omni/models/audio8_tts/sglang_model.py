# SPDX-License-Identifier: Apache-2.0
"""Serve Audio8 Slow AR with SGLang and generate its coupled codec frames."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.rotary_embedding import RotaryEmbedding
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from torch import nn

from sglang_omni.models.audio8_tts.fast_ar import (
    Audio8FastAR,
    CodebookSampler,
    precompute_rotary_values,
)
from sglang_omni.models.audio8_tts.hf_config import ArkttsConfig

BACKBONE_ROOT_WEIGHT_NAMES = {
    "embeddings.weight": "model.embed_tokens.weight",
    "norm.weight": "model.norm.weight",
}
BACKBONE_LAYER_WEIGHT_NAMES = {
    "attention.wo.weight": "self_attn.o_proj.weight",
    "feed_forward.w1.weight": "mlp.gate_proj.weight",
    "feed_forward.w2.weight": "mlp.down_proj.weight",
    "feed_forward.w3.weight": "mlp.up_proj.weight",
    "attention_norm.weight": "input_layernorm.weight",
    "ffn_norm.weight": "post_attention_layernorm.weight",
}
BACKBONE_QKV_SUFFIXES = ("attention.wqkv.weight", "attention.wqkv.bias")


class Audio8SGLangModel(Qwen2ForCausalLM):
    config: ArkttsConfig
    codebook_offsets: torch.Tensor

    def __init__(
        self,
        config: ArkttsConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        if (
            not config.norm_fastlayer_input
            or not config.tie_word_embeddings
            or not config.attention_qkv_bias
            or config.attention_qk_norm
            or config.attention_o_bias
        ):
            raise ValueError("Unsupported Audio8 Slow AR checkpoint layout")
        else:
            pass
        super().__init__(config, quant_config=quant_config, prefix=prefix)
        self.codebook_embeddings: nn.Embedding = nn.Embedding(
            config.num_codebooks * config.codebook_size, config.hidden_size
        )
        self.fast_ar: Audio8FastAR = Audio8FastAR(config)
        self.decode_input_embeds: torch.Tensor | None = None
        self.register_buffer(
            "codebook_offsets",
            torch.arange(
                config.num_codebooks, device=self.codebook_embeddings.weight.device
            )
            * config.codebook_size,
            persistent=False,
        )
        # note (BeiningWu): Audio8 rotates adjacent coordinates, while the stock backbone rotates split halves.
        interleaved_rope = RotaryEmbedding(
            config.head_dim,
            rotary_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            base=int(config.rope_base),
            is_neox_style=False,
            dtype=self.codebook_embeddings.weight.dtype,
        )
        rotary_values = precompute_rotary_values(
            config.max_position_embeddings, config.head_dim, config.rope_base
        )
        cosine, sine = rotary_values.unbind(dim=-1)
        interleaved_rope.cos_sin_cache.copy_(torch.cat((cosine, sine), dim=-1))
        for layer in self.model.layers:
            layer.self_attn.rotary_emb = interleaved_rope

    def embed_tokens(
        self, input_ids: torch.Tensor, codec_codes: torch.Tensor
    ) -> torch.Tensor:
        assert input_ids.ndim == 1
        assert codec_codes.shape == (input_ids.shape[0], self.config.num_codebooks)
        semantic_mask = (input_ids >= self.config.semantic_begin_id) & (
            input_ids <= self.config.semantic_end_id
        )
        semantic_codes = codec_codes.masked_fill(~semantic_mask.unsqueeze(1), 0)
        codec_embeddings = self.codebook_embeddings(
            semantic_codes + self.codebook_offsets
        ).sum(dim=1)
        text_embeddings = self.get_input_embeddings()(input_ids)
        return text_embeddings + codec_embeddings.masked_fill(
            ~semantic_mask.unsqueeze(1), 0.0
        )

    def generate_codebooks(
        self,
        hidden_states: torch.Tensor,
        semantic_token_ids: torch.Tensor,
        sample_codebook: CodebookSampler,
    ) -> torch.Tensor:
        return self.fast_ar(hidden_states, semantic_token_ids, sample_codebook)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        omni_prefill_rids: list[str] | tuple[str, ...] | None = None,
        input_embeds_are_projected: bool | None = None,
    ) -> LogitsProcessorOutput:
        # note (BeiningWu): Omni passes prefill metadata. Audio8's fused embeddings already have the backbone's hidden size.
        if (
            forward_batch.forward_mode.is_decode()
            and self.decode_input_embeds is not None
        ):
            assert input_embeds is None
            input_embeds = self.decode_input_embeds
            self.decode_input_embeds = None
        else:
            pass
        output = super().forward(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        assert isinstance(output, LogitsProcessorOutput)
        return output

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        fast_weights: dict[str, torch.Tensor] = {}
        codebook_embedding_weights: dict[str, torch.Tensor] = {}
        checkpoint_backbone_weight_names: set[str] = set()
        query_width = self.config.num_attention_heads * self.config.head_dim
        key_value_width = self.config.num_key_value_heads * self.config.head_dim

        def backbone_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for weight_name, weight in weights:
                if weight_name in ("freqs_cis", "fast_freqs_cis"):
                    continue
                elif weight_name.startswith("fast_"):
                    fast_weights[weight_name] = weight
                elif weight_name == "codebook_embeddings.weight":
                    codebook_embedding_weights["weight"] = weight
                elif weight_name in BACKBONE_ROOT_WEIGHT_NAMES:
                    checkpoint_backbone_weight_names.add(weight_name)
                    yield BACKBONE_ROOT_WEIGHT_NAMES[weight_name], weight
                elif weight_name.startswith("layers."):
                    checkpoint_backbone_weight_names.add(weight_name)
                    layer_index, _, weight_suffix = weight_name.removeprefix(
                        "layers."
                    ).partition(".")
                    layer_prefix = f"model.layers.{layer_index}."
                    if weight_suffix in BACKBONE_QKV_SUFFIXES:
                        projection_weights = weight.split(
                            (query_width, key_value_width, key_value_width), dim=0
                        )
                        parameter_name = weight_suffix.rsplit(".", 1)[1]
                        for projection, projection_weight in zip(
                            "qkv", projection_weights, strict=True
                        ):
                            yield (
                                f"{layer_prefix}self_attn.{projection}_proj.{parameter_name}",
                                projection_weight,
                            )
                    else:
                        yield (
                            layer_prefix + BACKBONE_LAYER_WEIGHT_NAMES[weight_suffix],
                            weight,
                        )
                else:
                    raise ValueError(
                        f"Unexpected Audio8 checkpoint weight {weight_name!r}"
                    )

        super().load_weights(backbone_weights())
        expected_backbone_weight_names = {
            *BACKBONE_ROOT_WEIGHT_NAMES,
            *(
                f"layers.{layer_index}.{weight_suffix}"
                for layer_index in range(self.config.num_hidden_layers)
                for weight_suffix in (
                    *BACKBONE_LAYER_WEIGHT_NAMES,
                    *BACKBONE_QKV_SUFFIXES,
                )
            ),
        }
        if checkpoint_backbone_weight_names != expected_backbone_weight_names:
            missing_weight_names = (
                expected_backbone_weight_names - checkpoint_backbone_weight_names
            )
            unexpected_weight_names = (
                checkpoint_backbone_weight_names - expected_backbone_weight_names
            )
            raise ValueError(
                f"Missing Audio8 backbone weights {sorted(missing_weight_names)}. "
                f"Unexpected weights {sorted(unexpected_weight_names)}"
            )
        else:
            pass
        self.codebook_embeddings.load_state_dict(codebook_embedding_weights)
        self.fast_ar.load_state_dict(fast_weights)
