"""Modeling qwen3."""

import torch
import torch._dynamo
from torch import nn
from transformers import Qwen3Config
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.processing_utils import Unpack
from transformers.utils import TransformersKwargs
from transformers.utils.generic import merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from sglang_omni.models.sensenova_u1.cache_dit import (
    decoder_attention_type,
    decoder_layers,
)
from sglang_omni.models.sensenova_u1.neo_unify.qwen_attention import Qwen3Attention
from sglang_omni.models.sensenova_u1.neo_unify.qwen_decoder import Qwen3DecoderLayer
from sglang_omni.models.sensenova_u1.neo_unify.qwen_feed_forward import (
    SenseNovaRMSNorm,
    make_qwen3_rms_norm,
)

# Modified for SGLang; see this directory's README.md for upstream source.


class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True
    _no_split_modules: list[str] = ["Qwen3DecoderLayer"]
    _skip_keys_device_placement: list[str] = ["past_key_values"]
    _supports_flash_attn: bool = True
    _supports_sdpa: bool = True
    _supports_flex_attn: bool = True

    _can_compile_fullgraph: bool = True
    _supports_attention_backend: bool = True
    _can_record_outputs: dict[str, type[nn.Module]] = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }


class Qwen3Model(Qwen3PreTrainedModel):
    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.padding_idx: int | None = config.pad_token_id
        self.vocab_size: int = config.vocab_size

        self.embed_tokens: nn.Embedding = nn.Embedding(
            config.vocab_size, config.hidden_size, self.padding_idx
        )
        self.layers: nn.ModuleList = nn.ModuleList(
            [
                Qwen3DecoderLayer(config, layer_idx)
                for layer_idx in range(config.num_hidden_layers)
            ]
        )
        self.norm: SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.norm_mot_gen: SenseNovaRMSNorm = make_qwen3_rms_norm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.gradient_checkpointing: bool = False
        self.has_sliding_layers: bool = "sliding_attention" in self.config.layer_types

        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        image_gen_indicators: torch.Tensor | None = None,
        indexes: torch.LongTensor | None = None,
        attention_mask: dict[str, torch.Tensor | None] | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        use_cache: bool | None = None,
        cache_position: torch.LongTensor | None = None,
        image_only: bool = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:

        if image_only:
            exist_non_image_gen_tokens = False
            exist_image_gen_tokens = True
        elif image_gen_indicators is None:
            exist_non_image_gen_tokens = True
            exist_image_gen_tokens = False
        else:
            exist_non_image_gen_tokens = bool((~image_gen_indicators).any().item())
            exist_image_gen_tokens = bool(image_gen_indicators.any().item())

        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError(
                "You must specify exactly one of input_ids or inputs_embeds"
            )
        else:
            pass

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        else:
            pass

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)
        else:
            pass

        if cache_position is None:
            past_seen_tokens = (
                past_key_values.get_seq_length() if past_key_values is not None else 0
            )
            cache_position = torch.arange(
                past_seen_tokens,
                past_seen_tokens + inputs_embeds.shape[1],
                device=inputs_embeds.device,
            )
        else:
            pass

        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)
        else:
            pass

        hidden_states = inputs_embeds

        layers = decoder_layers(
            self,
            update_cache=kwargs.get("update_cache", True),
            has_non_image_tokens=exist_non_image_gen_tokens,
            has_image_tokens=exist_image_gen_tokens,
        )
        position_embeddings = None
        if layers:
            native_layers = self.__dict__.get("sensenova_cache_dit_native_layers")
            rope_layer = native_layers[0] if native_layers is not None else layers[0]
            position_embeddings = rope_layer.self_attn.resolve_rope_tables(
                indexes, hidden_states
            )
        else:
            pass

        for decoder_layer in layers[: self.config.num_hidden_layers]:
            attention_type = decoder_attention_type(self, decoder_layer)
            hidden_states = decoder_layer(
                hidden_states,
                image_gen_indicators=image_gen_indicators,
                exist_non_image_gen_tokens=exist_non_image_gen_tokens,
                exist_image_gen_tokens=exist_image_gen_tokens,
                indexes=indexes,
                attention_mask=attention_mask[attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        if not exist_image_gen_tokens:
            hidden_states = self.norm(hidden_states)
        elif not exist_non_image_gen_tokens:
            hidden_states = self.norm_mot_gen(hidden_states)
        else:
            raise NotImplementedError(
                "Mixed und/gen backbone forward is not supported. "
                "Split the sequence at token-type boundaries."
            )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class Qwen3ForCausalLM(Qwen3PreTrainedModel):
    _tied_weights_keys: dict[str, str] = {"lm_head.weight": "model.embed_tokens.weight"}
    _tp_plan: dict[str, str] = {"lm_head": "colwise_rep"}
    _pp_plan: dict[str, tuple[list[str], list[str]]] = {
        "lm_head": (["hidden_states"], ["logits"])
    }

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__(config)
        self.model: Qwen3Model = Qwen3Model(config)
        self.vocab_size: int = config.vocab_size
        self.lm_head: nn.Linear = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

        self.post_init()


__all__ = ["Qwen3ForCausalLM", "Qwen3Model", "Qwen3PreTrainedModel"]
