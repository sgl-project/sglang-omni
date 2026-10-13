"""Native speech projections and residual prediction using MLX-LM decoders."""

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3 import ModelArgs as DenseArgs
from mlx_lm.models.qwen3 import TransformerBlock
from mlx_lm.models.qwen3_moe import MLP, ModelArgs, Qwen3MoeSparseMoeBlock
from mlx_lm.sample_utils import make_sampler
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
    Qwen3OmniMoeTalkerConfig,
)

from sglang_omni.models.qwen3_omni.mlx.language import MultimodalDecoderLayer, text_args


class SharedExpertMoe(Qwen3MoeSparseMoeBlock):
    def __init__(self, args: ModelArgs, intermediate_size: int) -> None:
        super().__init__(args)
        self.shared_expert = MLP(args.hidden_size, intermediate_size)
        self.shared_expert_gate = nn.Linear(args.hidden_size, 1, bias=False)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        # note (Copilot): Talker routing accumulates in float32, unlike the thinker.
        probabilities = mx.softmax(
            self.gate(hidden_states).astype(mx.float32), axis=-1, precise=True
        )
        indices = mx.argpartition(probabilities, kth=-self.top_k, axis=-1)[
            ..., -self.top_k :
        ]
        scores = mx.take_along_axis(probabilities, indices, axis=-1)
        if self.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        else:
            pass
        routed = (self.switch_mlp(hidden_states, indices) * scores[..., None]).sum(
            axis=-2
        )
        return routed + mx.sigmoid(self.shared_expert_gate(hidden_states)) * (
            self.shared_expert(hidden_states)
        )


class ResizeMLP(nn.Module):
    def __init__(
        self, input_size: int, intermediate_size: int, hidden_size: int
    ) -> None:
        super().__init__()
        self.linear_fc1 = nn.Linear(input_size, intermediate_size)
        self.linear_fc2 = nn.Linear(intermediate_size, hidden_size)

    def __call__(self, hidden_states: mx.array) -> mx.array:
        return self.linear_fc2(nn.silu(self.linear_fc1(hidden_states)))


class CodePredictor(nn.Module):
    def __init__(self, config: Qwen3OmniMoeTalkerCodePredictorConfig) -> None:
        super().__init__()
        self.config = config
        raw = config.to_dict()
        raw.update(
            rope_theta=config.rope_parameters["rope_theta"],
            rope_scaling=None,
            head_dim=config.head_dim,
            tie_word_embeddings=raw.get("tie_word_embeddings", False),
        )
        args = DenseArgs.from_dict(raw)
        self.model = nn.Module()
        self.model.layers = [
            TransformerBlock(args) for _ in range(config.num_hidden_layers)
        ]
        if config.attention_bias:
            for layer in self.model.layers:
                attention = layer.self_attn
                for projection in (
                    attention.q_proj,
                    attention.k_proj,
                    attention.v_proj,
                    attention.o_proj,
                ):
                    projection.bias = mx.zeros((projection.weight.shape[0],))
        else:
            pass
        self.model.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.model.codec_embedding = [
            nn.Embedding(config.vocab_size, config.hidden_size)
            for _ in range(config.num_code_groups - 1)
        ]
        self.lm_head = [
            nn.Linear(config.hidden_size, config.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        ]

    def __call__(self, hidden: mx.array, cache: list[KVCache]) -> mx.array:
        mask = create_attention_mask(hidden, cache[0])
        for layer, layer_cache in zip(self.model.layers, cache, strict=True):
            hidden = layer(hidden, mask, layer_cache)
        return self.model.norm(hidden)

    def generate(
        self,
        token: mx.array,
        embedding: mx.array,
        hidden: mx.array,
        temperature: float,
        top_p: float,
    ) -> tuple[mx.array, mx.array]:
        cache = [KVCache() for _ in self.model.layers]
        hidden = self(mx.concatenate([hidden, embedding], axis=1), cache)
        feedback = embedding
        codes = [token]
        sampler = make_sampler(temp=temperature, top_p=top_p)
        for index, head in enumerate(self.lm_head):
            logits = head(hidden[:, -1, :]).astype(mx.float32)
            logits = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            token = sampler(logits)[:, None]
            codes.append(token)
            embedding = self.model.codec_embedding[index](token)
            feedback = feedback + embedding
            if index + 1 < len(self.lm_head):
                hidden = self(embedding, cache)
            else:
                pass
        return feedback, mx.concatenate(codes, axis=-1)


class Talker(nn.Module):
    def __init__(self, config: Qwen3OmniMoeTalkerConfig) -> None:
        super().__init__()
        self.config = config
        text = config.text_config
        self.args = text_args(text)
        if config.num_code_groups != config.code_predictor_config.num_code_groups:
            raise ValueError("Talker and predictor code-group counts disagree")
        else:
            pass
        self.model = nn.Module()
        self.model.codec_embedding = nn.Embedding(text.vocab_size, text.hidden_size)
        self.model.layers = [
            MultimodalDecoderLayer(
                self.args,
                text.rope_parameters["mrope_section"],
                index,
                text.attention_bias,
            )
            for index in range(text.num_hidden_layers)
        ]
        for layer in self.model.layers:
            shared_size = getattr(text, "shared_expert_intermediate_size", 0)
            if shared_size:
                layer.mlp = SharedExpertMoe(self.args, shared_size)
            else:
                pass
        self.model.norm = nn.RMSNorm(text.hidden_size, eps=text.rms_norm_eps)
        self.text_projection = ResizeMLP(
            config.thinker_hidden_size, text.intermediate_size, text.hidden_size
        )
        self.hidden_projection = ResizeMLP(
            config.thinker_hidden_size, text.intermediate_size, text.hidden_size
        )
        self.codec_head = nn.Linear(text.hidden_size, text.vocab_size, bias=False)
        self.code_predictor = CodePredictor(config.code_predictor_config)

    @property
    def layers(self) -> list[MultimodalDecoderLayer]:
        return self.model.layers

    def make_cache(self) -> list[KVCache]:
        return [KVCache() for _ in self.layers]

    def __call__(
        self,
        *,
        inputs_embeds: mx.array,
        position_ids: mx.array,
        past_key_values: list[KVCache],
        use_cache: bool = True,
    ) -> tuple[mx.array, mx.array]:
        if not use_cache:
            raise ValueError("Native talker requires its request cache")
        else:
            pass
        hidden = inputs_embeds
        mask = create_attention_mask(hidden, past_key_values[0])
        for layer, cache in zip(self.layers, past_key_values, strict=True):
            hidden = layer(hidden, mask, cache, position_ids)
        hidden = self.model.norm(hidden)
        return self.codec_head(hidden), hidden

    def predict_codes(
        self,
        token: mx.array,
        hidden: mx.array,
        temperature: float,
        top_p: float,
    ) -> tuple[mx.array, mx.array]:
        embedding = self.model.codec_embedding(token)
        return self.code_predictor.generate(
            token, embedding, hidden, temperature, top_p
        )
