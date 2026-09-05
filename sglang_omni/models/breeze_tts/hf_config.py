# SPDX-License-Identifier: Apache-2.0
"""Expose Breeze's bundled Qwen3 backbone config to SGLang."""

from transformers import AutoConfig, Qwen3Config


class BreezeConfig(Qwen3Config):
    model_type = "breeze"

    def __init__(
        self,
        backbone_config=None,
        depth_decoder_config=None,
        text_encoder_config=None,
        audio_vocab_size=None,
        vocab_size=2051,
        **kwargs,
    ):
        # The outer config contains legacy Llama RoPE/norm settings. The
        # official adapter constructs Qwen3 layers from backbone_config instead.
        backbone = dict(backbone_config or {})
        audio_vocab_size = vocab_size if audio_vocab_size is None else audio_vocab_size
        config = {**kwargs, **backbone}
        config.pop("audio_token_id", None)
        config.pop("audio_eos_token_id", None)
        config.update(
            model_type="breeze",
            bos_token_id=0,
            vocab_size=audio_vocab_size + 1,
            tie_word_embeddings=False,
            architectures=["BreezeForConditionalGeneration"],
            eos_token_id=audio_vocab_size,
        )
        super().__init__(**config)
        self.backbone_config = backbone
        self.depth_decoder_config = depth_decoder_config
        self.text_encoder_config = text_encoder_config
        self.audio_vocab_size = audio_vocab_size
        self.num_codebooks = kwargs.get("num_codebooks", 16)
        self.audio_embed_size = kwargs.get("audio_embed_size", self.hidden_size)
        self.codec_codebook_size = kwargs.get("codec_config", {}).get(
            "codebook_size", 2048
        )


def register_breeze_config() -> None:
    AutoConfig.register("breeze", BreezeConfig, exist_ok=True)
