"""
FishQwen3 model configuration.

This follows HuggingFace's PretrainedConfig pattern while maintaining compatibility
with the original BaseModelArgs interface.
"""

import logging
from typing import Optional

from transformers import PretrainedConfig

from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.text2semantic.utils import (
    find_multiple,
)

log = logging.getLogger(__name__)


JUNK_KEYS = {
    # Classification stuff
    "id2label",
    "label2id",
    "problem_type",
    "finetuning_task",
    # Generation defaults
    "bad_words_ids",
    "begin_suppress_tokens",
    "suppress_tokens",
    "diversity_penalty",
    "do_sample",
    "early_stopping",
    "encoder_no_repeat_ngram_size",
    "exponential_decay_length_penalty",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "length_penalty",
    "max_length",
    "min_length",
    "no_repeat_ngram_size",
    "num_beam_groups",
    "num_beams",
    "num_return_sequences",
    "output_scores",
    "repetition_penalty",
    "return_dict_in_generate",
    "temperature",
    "top_k",
    "top_p",
    "typical_p",
    # Other noise
    "add_cross_attention",
    "chunk_size_feed_forward",
    "cross_attention_hidden_size",
    "decoder_start_token_id",
    "is_decoder",
    "is_encoder_decoder",
    "prefix",
    "pruned_heads",
    "remove_invalid_values",
    "sep_token_id",
    "task_specific_params",
    "tf_legacy_loss",
    "tie_encoder_decoder",
    "tokenizer_class",
    "torchscript",
    "output_attentions",
    "output_hidden_states",
    "return_dict",
    "_name_or_path",
    "architectures",
}


def clean_config_dict(d):
    if not isinstance(d, dict):
        return d

    return {
        k: clean_config_dict(v)
        for k, v in d.items()
        if k not in JUNK_KEYS and v is not None
    }


class FishQwen3Config(PretrainedConfig):
    """
    Configuration class for FishAudioQwen3 models.

    This is the base configuration for all FishAudioQwen3-based models. It stores the configuration
    of a FishAudioQwen3 model and is used to instantiate the model according to the specified arguments.

    Args:
        vocab_size (int, optional): Vocabulary size of the model. Defaults to 32000.
        n_layer (int, optional): Number of transformer layers. Defaults to 32.
        n_head (int, optional): Number of attention heads. Defaults to 32.
        dim (int, optional): Dimensionality of the embeddings and hidden states. Defaults to 4096.
        intermediate_size (int, optional): Dimensionality of the MLP representations. If None, computed automatically. Defaults to None.
        n_local_heads (int, optional): Number of local attention heads for GQA. If -1, uses n_head. Defaults to -1.
        head_dim (int, optional): Dimensionality of attention heads. If None, computed as dim // n_head. Defaults to None.
        rope_base (float, optional): Base for RoPE (Rotary Position Embedding). Defaults to 10000.
        norm_eps (float, optional): Epsilon for RMSNorm layers. Defaults to 1e-5.
        max_seq_len (int, optional): Maximum sequence length. Defaults to 2048.
        dropout (float, optional): Dropout probability (not supported with flash-attn-3). Defaults to 0.0.
        tie_word_embeddings (bool, optional): Whether to tie input and output embeddings. Defaults to True.
        attention_qkv_bias (bool, optional): Whether to use bias in QKV projection. Defaults to False.
        attention_o_bias (bool, optional): Whether to use bias in output projection. Defaults to False.
        attention_qk_norm (bool, optional): Whether to apply RMSNorm to Q and K. Defaults to False.
        use_moe (bool, optional): Whether to use Mixture of Experts. Defaults to False.
        num_experts (int, optional): Number of experts in MoE. Defaults to 1.
        num_experts_per_tok (int, optional): Number of experts to route to per token. Defaults to 1.
        norm_topk_prob (bool, optional): Whether to normalize top-k probabilities in MoE. Defaults to True.
        moe_intermediate_size (int, optional): Intermediate size for MoE experts. Defaults to 768.
        audio_embed_dim (int, optional): Dimensionality of audio embeddings. Defaults to None.
        audio_hidden_dim (int, optional): Hidden dimensionality for audio projector. Defaults to None.
        use_gradient_checkpointing (bool, optional): Whether to use gradient checkpointing. Defaults to True.
        initializer_range (float, optional): Standard deviation for weight initialization. If None, computed as dim^-0.5. Defaults to None.
    """

    model_type = "fish_qwen3"
    has_no_defaults_at_init = True

    def __init__(
        self,
        vocab_size: int = 32000,
        n_layer: int = 32,
        n_head: int = 32,
        dim: int = 4096,
        intermediate_size: Optional[int] = None,
        n_local_heads: int = -1,
        head_dim: Optional[int] = None,
        rope_base: float = 10000,
        norm_eps: float = 1e-5,
        max_seq_len: int = 2048,
        dropout: float = 0.0,
        tie_word_embeddings: bool = True,
        attention_qkv_bias: bool = False,
        attention_o_bias: bool = False,
        attention_qk_norm: bool = False,
        # MoE configs
        use_moe: bool = False,
        num_experts: int = 1,
        num_experts_per_tok: int = 1,
        norm_topk_prob: bool = True,
        moe_intermediate_size: int = 768,
        use_aux_loss_free: bool = False,
        router_gamma: float = 1e-3,
        # Audio configs
        audio_embed_dim: Optional[int] = None,
        audio_hidden_dim: Optional[int] = None,
        # Gradient checkpointing
        use_gradient_checkpointing: bool = True,
        # Initialize the model
        initializer_range: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)

        self.vocab_size = vocab_size
        self.n_layer = n_layer
        self.n_head = n_head
        self.dim = dim
        self.intermediate_size = intermediate_size
        self.n_local_heads = n_local_heads
        self.head_dim = head_dim
        self.rope_base = rope_base
        self.norm_eps = norm_eps
        self.max_seq_len = max_seq_len
        self.dropout = dropout
        self.attention_qkv_bias = attention_qkv_bias
        self.attention_o_bias = attention_o_bias
        self.attention_qk_norm = attention_qk_norm

        # MoE configs
        self.use_moe = use_moe
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok
        self.norm_topk_prob = norm_topk_prob
        self.moe_intermediate_size = moe_intermediate_size
        self.use_aux_loss_free = use_aux_loss_free
        self.router_gamma = router_gamma

        # Audio configs
        self.audio_embed_dim = audio_embed_dim
        self.audio_hidden_dim = audio_hidden_dim

        # Gradient checkpointing
        self.use_gradient_checkpointing = use_gradient_checkpointing

        # Initialize the model
        self.initializer_range = initializer_range

        # Post-initialization
        self._post_init_config()

    def _post_init_config(self):
        """Post-initialization to compute derived values."""
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head

        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)

        if self.head_dim is None:
            self.head_dim = self.dim // self.n_head

        assert self.dropout == 0.0, "Dropout is not supported in flash-attn-3"

        if self.audio_hidden_dim is None:
            self.audio_hidden_dim = self.dim * 2

        if self.initializer_range is None:
            self.initializer_range = self.dim**-0.5
            log.info(f"Using default initializer range: {self.initializer_range}")

    @staticmethod
    def attention_flops_per_token(n_layers, seq_len, dim, causal):
        """Calculate attention FLOPs per token."""
        # Formula from https://github.com/Dao-AILab/flash-attention/blob/main/benchmarks/benchmark_flash_attention.py#L27-L30
        return 3.5 * (4 * n_layers * seq_len * dim // (2 if causal else 1))

    def get_non_embed_params(self, include_output: bool = False):
        """Calculate number of non-embedding parameters."""
        base = self.n_layer * (
            # Attention, wqkv, wo
            (self.n_head + 2 * self.n_local_heads) * self.head_dim * self.dim
            + (self.dim * self.dim)
            +
            # Feed forward, w1, w3, w2
            self.dim * self.intermediate_size * 3
        )

        if include_output:
            base += self.dim * self.vocab_size

        return base

    def get_num_flop_per_token(self):
        """Calculate total FLOPs per token."""
        return 6 * self.get_non_embed_params(True) + self.attention_flops_per_token(
            self.n_layer, self.max_seq_len, self.dim, True
        )


class FishQwen3AudioDecoderConfig(FishQwen3Config):
    """
    Configuration class for FishAudioQwen3 Dual-AR models.

    This extends FishQwen3Config with additional parameters for the fast (codebook) decoder.

    Args:
        text_dim (int, optional): Input dimensionality for the fast decoder. Defaults to None.
        **kwargs: Additional arguments passed to FishQwen3Config.
        codebook_size (int, optional): Size of the codebook for audio tokens. Defaults to 160.
        num_codebooks (int, optional): Number of codebooks. Defaults to 4.
    """

    model_type = "fish_qwen3_audio_decoder"

    def __init__(
        self,
        text_dim: int = 1024,
        num_codebooks: int = 4,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.text_dim = text_dim
        self.num_codebooks = num_codebooks
        self.max_seq_len = num_codebooks + 1

    def get_num_flop_per_token(self):
        return super().get_num_flop_per_token() * self.num_codebooks


class FishQwen3AudioEncoderConfig(PretrainedConfig):
    """
    Configuration class for FishQwen3 Audio Encoder.

    This configuration is based on Qwen3OmniMoeAudioEncoder and adapted for packed (causal) version.

    Args:
        num_mel_bins (int, optional): Number of mel filterbank bins. Defaults to 128.
        d_model (int, optional): Hidden dimension. Defaults to 512.
        encoder_layers (int, optional): Number of encoder layers. Defaults to 6.
        encoder_attention_heads (int, optional): Number of attention heads. Defaults to 8.
        encoder_ffn_dim (int, optional): FFN intermediate dimension. Defaults to 2048.
        output_dim (int, optional): Output dimension after projection. Defaults to 512.
        downsample_hidden_size (int, optional): Hidden size for convolutional downsampling. Defaults to 512.
        max_source_positions (int, optional): Maximum source positions. Defaults to 1500.
        n_window (int, optional): Window size for chunking. Defaults to 50.
        n_window_infer (int, optional): Window size for inference. Defaults to 1500.
        conv_chunksize (int, optional): Chunk size for convolution to avoid OOM. Defaults to 100.
        scale_embedding (bool, optional): Whether to scale embeddings. Defaults to False.
        dropout (float, optional): Dropout rate. Defaults to 0.0.
        attention_dropout (float, optional): Attention dropout rate (deprecated). Defaults to 0.0.
        activation_function (str, optional): Activation function. Defaults to "gelu".
        activation_dropout (float, optional): Activation dropout rate. Defaults to 0.0.
        **kwargs: Additional arguments passed to PretrainedConfig.
    """

    model_type = "fish_qwen3_audio_encoder"

    def __init__(
        self,
        num_mel_bins: int = 128,
        d_model: int = 512,
        encoder_layers: int = 6,
        encoder_attention_heads: int = 8,
        encoder_ffn_dim: int = 2048,
        output_dim: int = 512,
        downsample_hidden_size: int = 512,
        max_source_positions: int = 1500,
        n_window: int = 50,
        n_window_infer: int = 1500,
        conv_chunksize: int = 100,
        scale_embedding: bool = False,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        activation_function: str = "gelu",
        activation_dropout: float = 0.0,
        use_gradient_checkpointing: bool = False,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.output_dim = output_dim
        self.downsample_hidden_size = downsample_hidden_size
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.scale_embedding = scale_embedding
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.use_gradient_checkpointing = use_gradient_checkpointing

        assert (
            self.attention_dropout == 0.0
        ), "attention_dropout is deprecated and should be 0.0"


class FishQwen3OmniConfig(PretrainedConfig):
    """
    Configuration class for FishQwen3Omni models.

    This configuration follows HuggingFace's nested config pattern, similar to Qwen3OmniMoe.
    It contains separate configs for text model, audio encoder, and audio decoder.

    Args:
        text_config (FishQwen3Config or dict): Configuration for the text model. Required.
        audio_encoder_config (FishQwen3AudioEncoderConfig or dict, optional): Configuration for audio encoder. Can be None.
        audio_decoder_config (FishQwen3AudioDecoderConfig or dict, optional): Configuration for audio decoder (if using dual-AR). Can be None.
        eos_token_id (int, optional): Token ID for end of sequence (<|im_end|>). Defaults to None.
        pad_token_id (int, optional): Token ID for padding (<|pad|>). Defaults to None.
        audio_pad_token_id (int, optional): Token ID for audio padding (<|audio_pad|>). Defaults to None.
        semantic_start_token_id (int, optional): Token ID for first semantic token (<|semantic:0|>). Defaults to None.
        semantic_end_token_id (int, optional): Token ID for last semantic token (<|semantic:{codebook_size-1}|>). Defaults to None.
        **kwargs: Additional arguments passed to PretrainedConfig.
    """

    model_type = "fish_qwen3_omni"
    is_composition = True
    has_no_defaults_at_init = True

    def __init__(
        self,
        text_config: Optional[FishQwen3Config | dict] = None,
        audio_encoder_config: Optional[FishQwen3AudioEncoderConfig | dict] = None,
        audio_decoder_config: Optional[FishQwen3AudioDecoderConfig | dict] = None,
        eos_token_id: Optional[int] = None,
        pad_token_id: Optional[int] = None,
        audio_pad_token_id: Optional[int] = None,
        semantic_start_token_id: Optional[int] = None,
        semantic_end_token_id: Optional[int] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)

        # Special token IDs
        self.eos_token_id = eos_token_id
        self.pad_token_id = pad_token_id
        self.audio_pad_token_id = audio_pad_token_id
        self.semantic_start_token_id = semantic_start_token_id
        self.semantic_end_token_id = semantic_end_token_id

        # Initialize text config (mandatory, but allow None for default instance creation by HF)
        if isinstance(text_config, dict):
            self.text_config = FishQwen3Config(**text_config)
        else:
            self.text_config = text_config

        # Initialize audio encoder config (optional)
        if isinstance(audio_encoder_config, dict):
            self.audio_encoder_config = FishQwen3AudioEncoderConfig(
                **audio_encoder_config
            )
        else:
            self.audio_encoder_config = audio_encoder_config

        # Initialize audio decoder config (optional)
        if isinstance(audio_decoder_config, dict):
            self.audio_decoder_config = FishQwen3AudioDecoderConfig(
                **audio_decoder_config
            )
        else:
            self.audio_decoder_config = audio_decoder_config

        # Validate dimension consistency
        if self.text_config is not None:
            if self.audio_encoder_config is not None:
                assert self.audio_encoder_config.output_dim == self.text_config.dim, (
                    f"AudioEncoder output_dim ({self.audio_encoder_config.output_dim}) "
                    f"must equal text_config dim ({self.text_config.dim})"
                )

            if self.audio_decoder_config is not None:
                assert self.audio_decoder_config.text_dim == self.text_config.dim, (
                    f"AudioDecoder text_dim ({self.audio_decoder_config.text_dim}) "
                    f"must equal text_config dim ({self.text_config.dim})"
                )

    def get_num_flop_per_token(self):
        """Calculate total FLOPs per token for the omni model."""
        total_flops = 0
        if self.text_config is not None:
            total_flops += self.text_config.get_num_flop_per_token()
        if self.audio_decoder_config is not None:
            total_flops += self.audio_decoder_config.get_num_flop_per_token()
        return total_flops

    def to_dict(self):
        output = super().to_dict()
        return clean_config_dict(output)
