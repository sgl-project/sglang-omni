"""Qwen decoder."""

import torch
import torch._dynamo
from transformers import Qwen3Config
from transformers.cache_utils import Cache
from transformers.modeling_layers import GradientCheckpointingLayer
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.deprecation import deprecate_kwarg

from sglang_omni.models.sensenova_u1.neo_unify.qwen_attention import Qwen3Attention
from sglang_omni.models.sensenova_u1.neo_unify.qwen_feed_forward import (
    Qwen3MLP,
    SenseNovaRMSNorm,
    make_qwen3_rms_norm,
)
from sglang_omni.models.sensenova_u1.neo_unify.rotary_embeddings import (
    PositionEmbeddings,
)

# Modified for SGLang; see this directory's README.md for upstream source.


class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int) -> None:
        super().__init__()
        self.hidden_size: int = config.hidden_size

        self.self_attn: Qwen3Attention = Qwen3Attention(
            config=config, layer_idx=layer_idx
        )

        self.mlp: Qwen3MLP = Qwen3MLP(config)
        self.mlp_mot_gen: Qwen3MLP = Qwen3MLP(config)
        self.input_layernorm: SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.input_layernorm_mot_gen: SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm: SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm_mot_gen: SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.attention_type: str = config.layer_types[layer_idx]

    def forward_und(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: PositionEmbeddings | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward_gen(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: PositionEmbeddings | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm_mot_gen(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            image_gen_indicators=image_gen_indicators,
            exist_non_image_gen_tokens=exist_non_image_gen_tokens,
            exist_image_gen_tokens=exist_image_gen_tokens,
            indexes=indexes,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm_mot_gen(hidden_states)
        hidden_states = self.mlp_mot_gen(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    @deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
    def forward(
        self,
        hidden_states: torch.Tensor,
        image_gen_indicators: torch.Tensor,
        exist_non_image_gen_tokens: bool,
        exist_image_gen_tokens: bool,
        indexes: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        use_cache: bool | None = False,
        cache_position: torch.LongTensor | None = None,
        position_embeddings: PositionEmbeddings | None = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        if exist_non_image_gen_tokens and not exist_image_gen_tokens:
            return self.forward_und(
                hidden_states,
                image_gen_indicators,
                exist_non_image_gen_tokens,
                exist_image_gen_tokens,
                indexes,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            pass
        if not exist_non_image_gen_tokens and exist_image_gen_tokens:
            return self.forward_gen(
                hidden_states,
                image_gen_indicators,
                exist_non_image_gen_tokens,
                exist_image_gen_tokens,
                indexes,
                attention_mask,
                position_ids,
                past_key_values,
                use_cache,
                cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        else:
            pass

        raise NotImplementedError(
            "Mixed und/gen decoder-layer forward is not yet validated (issue #207). "
            "Split the sequence at token-type boundaries and use forward_und / forward_gen."
        )
