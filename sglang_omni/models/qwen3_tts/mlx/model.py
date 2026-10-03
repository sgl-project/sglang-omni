# SPDX-License-Identifier: Apache-2.0
"""Native MLX talker and code predictor for Qwen3-TTS CustomVoice."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.qwen3 import ModelArgs, Qwen3Model
from pydantic import BaseModel, ConfigDict
from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousAttentionKVCache
from transformers import PreTrainedTokenizerBase


class Qwen3TTSMlxCodePredictorConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    vocab_size: int
    num_code_groups: int


class Qwen3TTSMlxTalkerConfig(Qwen3TTSMlxCodePredictorConfig):
    code_predictor_config: Qwen3TTSMlxCodePredictorConfig
    text_hidden_size: int
    text_vocab_size: int
    codec_eos_token_id: int
    codec_think_id: int
    codec_nothink_id: int
    codec_think_bos_id: int
    codec_think_eos_id: int
    codec_pad_id: int
    codec_bos_id: int
    codec_language_id: dict[str, int]
    spk_id: dict[str, int]
    spk_is_dialect: dict[str, bool | str]


class Qwen3TTSMlxArtifactConfig(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tts_model_type: Literal["custom_voice"]
    tts_model_size: Literal["0b6"]
    talker_config: Qwen3TTSMlxTalkerConfig
    tts_bos_token_id: int
    tts_eos_token_id: int
    tts_pad_token_id: int


def qwen3_model_args(
    config: Qwen3TTSMlxCodePredictorConfig,
    *,
    vocab_size: int,
) -> ModelArgs:
    return ModelArgs(
        model_type="qwen3",
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        num_hidden_layers=config.num_hidden_layers,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_key_value_heads,
        head_dim=config.head_dim,
        rms_norm_eps=config.rms_norm_eps,
        rope_theta=config.rope_theta,
        max_position_embeddings=config.max_position_embeddings,
        vocab_size=vocab_size,
        tie_word_embeddings=False,
    )


class TextProjection(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.linear_fc1: nn.Linear = nn.Linear(input_size, hidden_size, bias=True)
        self.linear_fc2: nn.Linear = nn.Linear(hidden_size, output_size, bias=True)

    def __call__(self, values: mx.array) -> mx.array:
        return self.linear_fc2(nn.silu(self.linear_fc1(values)))


class Qwen3TTSMlxTalker(nn.Module):
    """Qwen3 trunk with the speech and text heads of the converted artifact."""

    def __init__(self, artifact: Qwen3TTSMlxArtifactConfig) -> None:
        super().__init__()
        config = artifact.talker_config
        self.artifact: Qwen3TTSMlxArtifactConfig = artifact
        self.model: Qwen3Model = Qwen3Model(
            qwen3_model_args(config, vocab_size=config.vocab_size)
        )
        self.text_embedding: nn.Embedding = nn.Embedding(
            config.text_vocab_size, config.text_hidden_size
        )
        self.text_projection: TextProjection = TextProjection(
            config.text_hidden_size, config.text_hidden_size, config.hidden_size
        )
        self.codec_head: nn.Linear = nn.Linear(
            config.hidden_size, config.vocab_size, bias=False
        )

    @property
    def lm_head(self) -> nn.Linear:
        return self.codec_head

    def forward_embeddings(
        self,
        embeddings: mx.array,
        cache: list[ContiguousAttentionKVCache] | None = None,
    ) -> tuple[mx.array, mx.array]:
        hidden = self.model(inputs=None, cache=cache, input_embeddings=embeddings)
        last_hidden = hidden[:, -1:, :]
        return self.codec_head(last_hidden), last_hidden

    def __call__(
        self,
        input_ids: mx.array,
        cache: list[ContiguousAttentionKVCache] | None = None,
    ) -> mx.array:
        embeddings = self.model.embed_tokens(input_ids)
        logits, _ = self.forward_embeddings(embeddings, cache)
        return logits

    def build_prompt_embeddings(
        self,
        tokenizer: PreTrainedTokenizerBase,
        *,
        text: str,
        voice: str,
        language: str,
    ) -> tuple[mx.array, mx.array, mx.array]:
        """Build the first-frame prompt and text queue for CustomVoice."""
        config = self.artifact.talker_config
        voice_key = voice.casefold()
        if voice_key not in config.spk_id:
            raise ValueError(f"Qwen3-TTS MLX does not support voice {voice!r}")
        else:
            pass
        chat_text = f"<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"
        text_ids = mx.array(tokenizer.encode(chat_text), dtype=mx.int32)[None, :]
        text_embeddings = self.text_projection(self.text_embedding(text_ids))
        tts_ids = mx.array(
            [
                [
                    self.artifact.tts_bos_token_id,
                    self.artifact.tts_eos_token_id,
                    self.artifact.tts_pad_token_id,
                ]
            ],
            dtype=mx.int32,
        )
        tts_embeddings = self.text_projection(self.text_embedding(tts_ids))
        bos_embedding = tts_embeddings[:, 0:1, :]
        eos_embedding = tts_embeddings[:, 1:2, :]
        pad_embedding = tts_embeddings[:, 2:3, :]

        language_key = language.casefold()
        dialect = config.spk_is_dialect.get(voice_key)
        if language_key in {"auto", "chinese"} and isinstance(dialect, str):
            language_key = dialect
        else:
            pass
        language_id = config.codec_language_id.get(language_key)
        if language_id is None:
            codec_ids = [
                config.codec_nothink_id,
                config.codec_think_bos_id,
                config.codec_think_eos_id,
            ]
        else:
            codec_ids = [
                config.codec_think_id,
                config.codec_think_bos_id,
                language_id,
                config.codec_think_eos_id,
            ]
        codec_ids.extend(
            [config.spk_id[voice_key], config.codec_pad_id, config.codec_bos_id]
        )
        codec_embeddings = self.model.embed_tokens(
            mx.array([codec_ids], dtype=mx.int32)
        )
        padding = mx.broadcast_to(
            pad_embedding,
            (1, len(codec_ids) - 2, pad_embedding.shape[-1]),
        )
        prefix = mx.concatenate([padding, bos_embedding], axis=1)
        prefix = prefix + codec_embeddings[:, :-1, :]
        first_text = text_embeddings[:, 3:4, :] + codec_embeddings[:, -1:, :]
        prompt = mx.concatenate([text_embeddings[:, :3, :], prefix, first_text], axis=1)
        trailing_text = mx.concatenate(
            [text_embeddings[:, 4:-5, :], eos_embedding], axis=1
        )
        return prompt, trailing_text, pad_embedding


class Qwen3TTSMlxCodePredictor(nn.Module):
    """Short Qwen3 decoder for the remaining fifteen codec groups."""

    def __init__(self, config: Qwen3TTSMlxTalkerConfig) -> None:
        super().__init__()
        predictor = config.code_predictor_config
        self.model: Qwen3Model = Qwen3Model(
            qwen3_model_args(predictor, vocab_size=predictor.vocab_size)
        )
        self.codec_embedding: list[nn.Embedding] = [
            nn.Embedding(predictor.vocab_size, config.hidden_size)
            for _ in range(config.num_code_groups - 1)
        ]
        self.lm_head: list[nn.Linear] = [
            nn.Linear(predictor.hidden_size, predictor.vocab_size, bias=False)
            for _ in range(config.num_code_groups - 1)
        ]
        self.small_to_mtp_projection: nn.Linear | None = (
            nn.Linear(config.hidden_size, predictor.hidden_size, bias=True)
            if config.hidden_size != predictor.hidden_size
            else None
        )

    def forward_embeddings(
        self,
        embeddings: mx.array,
        *,
        cache: list[ContiguousAttentionKVCache],
        code_group: int,
    ) -> mx.array:
        if self.small_to_mtp_projection is not None:
            embeddings = self.small_to_mtp_projection(embeddings)
        else:
            pass
        hidden = self.model(inputs=None, cache=cache, input_embeddings=embeddings)
        return self.lm_head[code_group](hidden[:, -1:, :])


def load_qwen3_tts_mlx_talker(
    model_dir: Path,
) -> tuple[Qwen3TTSMlxTalker, Qwen3TTSMlxCodePredictor]:
    """Load CustomVoice talker weights without materializing the vocoder."""
    artifact = Qwen3TTSMlxArtifactConfig.model_validate_json(
        (model_dir / "config.json").read_text(encoding="utf-8")
    )
    weights = mx.load(str(model_dir / "model.safetensors"))
    talker = Qwen3TTSMlxTalker(artifact)
    predictor = Qwen3TTSMlxCodePredictor(artifact.talker_config)

    talker_weights: dict[str, mx.array] = {}
    predictor_weights: dict[str, mx.array] = {}
    for name, weight in weights.items():
        if name.startswith("talker.model.layers.") or name.startswith(
            "talker.model.norm."
        ):
            talker_weights[name.removeprefix("talker.")] = weight
        elif name == "talker.model.codec_embedding.weight":
            talker_weights["model.embed_tokens.weight"] = weight
        elif name.startswith("talker.model.text_embedding."):
            talker_weights[
                name.replace("talker.model.text_embedding.", "text_embedding.")
            ] = weight
        elif name.startswith("talker.text_projection.") or name.startswith(
            "talker.codec_head."
        ):
            talker_weights[name.removeprefix("talker.")] = weight
        elif name.startswith("talker.code_predictor.model.layers.") or name.startswith(
            "talker.code_predictor.model.norm."
        ):
            predictor_weights[name.removeprefix("talker.code_predictor.")] = weight
        elif name.startswith("talker.code_predictor.model.codec_embedding."):
            predictor_weights[
                name.replace(
                    "talker.code_predictor.model.codec_embedding.", "codec_embedding."
                )
            ] = weight
        elif name.startswith("talker.code_predictor.lm_head.") or name.startswith(
            "talker.code_predictor.small_to_mtp_projection."
        ):
            predictor_weights[name.removeprefix("talker.code_predictor.")] = weight
        else:
            pass

    talker_expected = {name for name, _ in tree_flatten(talker.parameters())}
    predictor_expected = {name for name, _ in tree_flatten(predictor.parameters())} - {
        "model.embed_tokens.weight"
    }
    if talker_weights.keys() != talker_expected:
        raise ValueError(
            "Qwen3-TTS MLX talker weight mismatch: "
            f"missing={sorted(talker_expected - talker_weights.keys())}, "
            f"extra={sorted(talker_weights.keys() - talker_expected)}"
        )
    else:
        pass
    if predictor_weights.keys() != predictor_expected:
        raise ValueError(
            "Qwen3-TTS MLX predictor weight mismatch: "
            f"missing={sorted(predictor_expected - predictor_weights.keys())}, "
            f"extra={sorted(predictor_weights.keys() - predictor_expected)}"
        )
    else:
        pass
    talker.load_weights(list(talker_weights.items()), strict=True)
    predictor.load_weights(list(predictor_weights.items()), strict=False)
    mx.eval(talker.parameters(), predictor.parameters())
    return talker, predictor
