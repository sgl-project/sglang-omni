# SPDX-License-Identifier: Apache-2.0
"""HuggingFace config adapters for Ming-Omni-TTS 16B."""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig

from sglang_omni.models.ming_tts.audio_config import AudioVAEconfig
from sglang_omni.models.ming_tts.payload_types import MING_TTS_SAMPLE_RATE

MING_TTS_MODEL_ARCH_OVERRIDE = "MingTTSSGLangModel"
MING_TTS_MROPE_SECTION = [16, 24, 24]
MING_TTS_AUDIO_VAE_ATTN_IMPLEMENTATION = "sdpa"
MING_TTS_TAIL_ATTN_BACKEND = "torch"

_ming_tts_hf_config_registered = False


class BailingMoeTTSConfig(PretrainedConfig):
    """Nested BailingMoe config used by the Ming-Omni-TTS AR engine."""

    model_type = "bailing_moe"

    def __init__(
        self,
        vocab_size: int | None = None,
        hidden_size: int | None = None,
        intermediate_size: int | None = None,
        moe_intermediate_size: int | None = None,
        num_hidden_layers: int | None = None,
        num_attention_heads: int | None = None,
        num_key_value_heads: int | None = None,
        head_dim: int | None = None,
        hidden_act: str = "silu",
        rms_norm_eps: float = 1e-5,
        initializer_range: float = 0.006,
        max_position_embeddings: int = 32768,
        rope_theta: float = 600000.0,
        rope_scaling: dict[str, Any] | None = None,
        use_cache: bool = True,
        use_sliding_window: bool = False,
        sliding_window: int = 4096,
        max_window_layers: int = 28,
        attention_dropout: float = 0.0,
        output_dropout: float = 0.0,
        use_qkv_bias: bool = False,
        use_bias: bool = False,
        norm_head: bool = False,
        tie_word_embeddings: bool = False,
        pad_token_id: int | None = 126081,
        eos_token_id: int | None = 126081,
        num_experts: int | None = None,
        num_shared_experts: int | None = None,
        num_experts_per_tok: int | None = None,
        norm_topk_prob: bool = True,
        first_k_dense_replace: int = 0,
        output_router_logits: bool = False,
        multi_gate: bool | None = None,
        image_patch_token: int | None = None,
        image_start_token: int | None = None,
        video_start_token: int | None = None,
        use_grouped_gemm: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            pad_token_id=pad_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.moe_intermediate_size = moe_intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.rms_norm_eps = rms_norm_eps
        self.initializer_range = initializer_range
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.raw_rope_scaling = dict(rope_scaling) if rope_scaling is not None else None
        if rope_scaling is None:
            self.rope_scaling = None
        else:
            raw_rope_scaling = dict(rope_scaling)
            if (
                raw_rope_scaling.get("type") == "3D"
                and raw_rope_scaling.get("factor") is None
            ):
                self.rope_scaling = {
                    "type": "default",
                    "rope_type": "default",
                    "mrope_section": list(MING_TTS_MROPE_SECTION),
                    "mrope_interleaved": False,
                }
            else:
                raise ValueError(
                    "Ming-Omni-TTS currently supports only rope_scaling "
                    "{'type': '3D', 'factor': None}; got "
                    f"{raw_rope_scaling!r}"
                )
        self.runtime_rope_scaling = self.rope_scaling
        self.use_cache = use_cache
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window
        self.max_window_layers = max_window_layers
        self.attention_dropout = attention_dropout
        self.output_dropout = output_dropout
        self.use_qkv_bias = use_qkv_bias
        self.use_bias = use_bias
        self.norm_head = norm_head
        self.num_experts = num_experts
        self.num_shared_experts = num_shared_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.first_k_dense_replace = first_k_dense_replace
        self.output_router_logits = output_router_logits
        self.multi_gate = multi_gate
        self.image_patch_token = image_patch_token
        self.image_start_token = image_start_token
        self.video_start_token = video_start_token
        self.use_grouped_gemm = use_grouped_gemm


class BailingMMTTSConfig(PretrainedConfig):
    """Top-level Ming-Omni-TTS composite config."""

    model_type = "bailingmm"
    sub_configs = {
        "llm_config": BailingMoeTTSConfig,
    }

    def __init__(
        self,
        llm_config: BailingMoeTTSConfig | dict[str, Any] | None = None,
        audio_tokenizer_config: PretrainedConfig | dict[str, Any] | None = None,
        ditar_config: dict[str, Any] | None = None,
        aggregator_config: dict[str, Any] | None = None,
        model_type: str | None = None,
        **kwargs: Any,
    ) -> None:
        if isinstance(llm_config, dict):
            llm_config = BailingMoeTTSConfig(**llm_config)
        if isinstance(audio_tokenizer_config, dict):
            audio_tokenizer_config = AudioVAEconfig(**audio_tokenizer_config)

        self.raw_model_type = model_type or self.model_type
        self.llm_config = llm_config
        self.audio_tokenizer_config = audio_tokenizer_config
        self.ditar_config = dict(ditar_config) if ditar_config is not None else None
        self.aggregator_config = (
            dict(aggregator_config) if aggregator_config is not None else None
        )
        is_default_init = (
            llm_config is None
            and audio_tokenizer_config is None
            and ditar_config is None
            and aggregator_config is None
            and model_type is None
            and not kwargs
        )
        super().__init__(**kwargs)
        if is_default_init:
            return

        if self.raw_model_type != self.model_type:
            raise ValueError(
                "Ming-Omni-TTS config must use model_type "
                f"{self.model_type!r}; got {self.raw_model_type!r}"
            )

        if self.llm_config is None:
            raise ValueError("Ming-Omni-TTS config is missing llm_config")
        if getattr(self.llm_config, "model_type", None) != "bailing_moe":
            raise ValueError(
                "Ming-Omni-TTS currently supports only the 16.8B MoE TTS "
                "checkpoint (llm_config.model_type='bailing_moe'); dense "
                "branch is currently unsupported."
            )

        for field in (
            "hidden_size",
            "num_hidden_layers",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "num_experts",
            "num_experts_per_tok",
            "num_shared_experts",
            "moe_intermediate_size",
            "multi_gate",
        ):
            if getattr(self.llm_config, field, None) is None:
                raise ValueError(f"Ming-Omni-TTS llm_config is missing {field}")

        if self.audio_tokenizer_config is None:
            raise ValueError("Ming-Omni-TTS config is missing audio_tokenizer_config")
        if (
            int(getattr(self.audio_tokenizer_config, "sample_rate", 0))
            != MING_TTS_SAMPLE_RATE
        ):
            raise ValueError(
                "Ming-Omni-TTS currently supports only "
                f"audio_tokenizer_config.sample_rate {MING_TTS_SAMPLE_RATE}"
            )
        if not isinstance(
            getattr(self.audio_tokenizer_config, "enc_kwargs", None),
            dict,
        ):
            raise ValueError(
                "Ming-Omni-TTS audio_tokenizer_config.enc_kwargs is missing"
            )
        if not isinstance(
            getattr(self.audio_tokenizer_config, "dec_kwargs", None),
            dict,
        ):
            raise ValueError(
                "Ming-Omni-TTS audio_tokenizer_config.dec_kwargs is missing"
            )
        if self.audio_tokenizer_config.enc_kwargs.get("latent_dim") is None:
            raise ValueError(
                "Ming-Omni-TTS audio_tokenizer_config.enc_kwargs.latent_dim "
                "is missing"
            )

        if not isinstance(self.ditar_config, dict):
            raise ValueError("Ming-Omni-TTS config is missing ditar_config")
        for field in ("patch_size", "history_patch_size"):
            if self.ditar_config.get(field) is None:
                raise ValueError(f"Ming-Omni-TTS ditar_config is missing {field}")

        if not isinstance(self.aggregator_config, dict):
            raise ValueError("Ming-Omni-TTS config is missing aggregator_config")
        for field in ("hidden_size", "depth", "num_heads"):
            if self.aggregator_config.get(field) is None:
                raise ValueError(f"Ming-Omni-TTS aggregator_config is missing {field}")

    @property
    def sample_rate(self) -> int:
        return int(self.audio_tokenizer_config.sample_rate)

    @property
    def latent_dim(self) -> int:
        return int(self.audio_tokenizer_config.enc_kwargs["latent_dim"])

    @property
    def audio_patch_size(self) -> int:
        return int(self.ditar_config["patch_size"])

    @property
    def history_patch_size(self) -> int:
        return int(self.ditar_config.get("history_patch_size", self.audio_patch_size))


def register_ming_tts_hf_config() -> None:
    """Register Ming-Omni-TTS local HF configs before SGLang loads ModelConfig."""

    global _ming_tts_hf_config_registered
    if _ming_tts_hf_config_registered:
        return

    from transformers import AutoConfig

    AutoConfig.register("bailing_moe", BailingMoeTTSConfig, exist_ok=True)
    AutoConfig.register("bailingmm", BailingMMTTSConfig, exist_ok=True)
    _ming_tts_hf_config_registered = True


__all__ = [
    "BailingMMTTSConfig",
    "BailingMoeTTSConfig",
    "MING_TTS_AUDIO_VAE_ATTN_IMPLEMENTATION",
    "MING_TTS_MODEL_ARCH_OVERRIDE",
    "MING_TTS_MROPE_SECTION",
    "MING_TTS_SAMPLE_RATE",
    "MING_TTS_TAIL_ATTN_BACKEND",
    "register_ming_tts_hf_config",
]
