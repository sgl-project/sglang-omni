from __future__ import annotations

from transformers import PretrainedConfig


class Qwen3OmniMoeAudioEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        num_mel_bins=128,
        encoder_layers=32,
        encoder_attention_heads=20,
        encoder_ffn_dim=5120,
        d_model=1280,
        dropout=0,
        attention_dropout=0,
        activation_function="gelu",
        activation_dropout=0,
        scale_embedding=False,
        initializer_range=0.02,
        max_source_positions=1500,
        n_window=100,
        output_dim=3584,
        n_window_infer=400,
        conv_chunksize=500,
        downsample_hidden_size=480,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.num_mel_bins = num_mel_bins
        self.d_model = d_model
        self.encoder_layers = encoder_layers
        self.encoder_attention_heads = encoder_attention_heads
        self.encoder_ffn_dim = encoder_ffn_dim
        self.dropout = dropout
        self.attention_dropout = attention_dropout
        self.activation_function = activation_function
        self.activation_dropout = activation_dropout
        self.num_hidden_layers = encoder_layers
        self.initializer_range = initializer_range
        self.scale_embedding = (
            scale_embedding  # scale factor will be sqrt(d_model) if True
        )
        self.max_source_positions = max_source_positions
        self.n_window = n_window
        self.output_dim = output_dim
        self.n_window_infer = n_window_infer
        self.conv_chunksize = conv_chunksize
        self.downsample_hidden_size = downsample_hidden_size


class Qwen3OmniMoeVisionEncoderConfig(PretrainedConfig):
    def __init__(
        self,
        depth=27,
        hidden_size=1152,
        hidden_act="gelu_pytorch_tanh",
        intermediate_size=4304,
        num_heads=16,
        in_channels=3,
        patch_size=16,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=3584,
        num_position_embeddings=2304,
        deepstack_visual_indexes=[8, 16, 24],
        tokens_per_second=None,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.depth = depth
        self.hidden_size = hidden_size
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.num_heads = num_heads
        self.in_channels = in_channels
        self.patch_size = patch_size
        self.spatial_merge_size = spatial_merge_size
        self.temporal_patch_size = temporal_patch_size
        self.out_hidden_size = out_hidden_size
        self.num_position_embeddings = num_position_embeddings
        self.initializer_range = initializer_range
        self.deepstack_visual_indexes = deepstack_visual_indexes
        self.tokens_per_second = tokens_per_second


class Qwen3OmniMoeTextConfig(PretrainedConfig):
    def __init__(
        self,
        vocab_size=3584,
        hidden_size=2048,
        intermediate_size=18944,
        num_hidden_layers=28,
        num_attention_heads=28,
        num_key_value_heads=4,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        sliding_window=None,
        attention_dropout=0,
        decoder_sparse_step=1,
        moe_intermediate_size=768,
        num_experts_per_tok=8,
        num_experts=128,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        **kwargs,
    ):
        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.sliding_window = sliding_window

        self.num_key_value_heads = num_key_value_heads
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]

        # MoE arguments
        self.decoder_sparse_step = decoder_sparse_step
        self.moe_intermediate_size = moe_intermediate_size
        self.num_experts_per_tok = num_experts_per_tok
        self.num_experts = num_experts
        self.norm_topk_prob = norm_topk_prob
        self.output_router_logits = output_router_logits
        self.router_aux_loss_coef = router_aux_loss_coef
        self.mlp_only_layers = [] if mlp_only_layers is None else mlp_only_layers


class Qwen3OmniMoeThinkerConfig(PretrainedConfig):
    def __init__(
        self,
        audio_config=None,
        vision_config=None,
        text_config=None,
        audio_token_id=151646,
        image_token_id=151655,
        video_token_id=151656,
        vision_start_token_id=151652,
        position_id_per_seconds=25,
        audio_start_token_id=151647,
        user_token_id=872,
        initializer_range=0.02,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.user_token_id = user_token_id
        self.vision_start_token_id = vision_start_token_id
        self.position_id_per_seconds = position_id_per_seconds
        self.audio_start_token_id = audio_start_token_id
        self.initializer_range = initializer_range

        if isinstance(vision_config, dict):
            vision_config = Qwen3OmniMoeVisionEncoderConfig(**vision_config)
        elif vision_config is None:
            vision_config = Qwen3OmniMoeVisionEncoderConfig()
        self.vision_config = vision_config

        if isinstance(audio_config, dict):
            audio_config = Qwen3OmniMoeAudioEncoderConfig(**audio_config)
        elif audio_config is None:
            audio_config = Qwen3OmniMoeAudioEncoderConfig()
        self.audio_config = audio_config

        if isinstance(text_config, dict):
            text_config = Qwen3OmniMoeTextConfig(**text_config)
        elif text_config is None:
            text_config = Qwen3OmniMoeTextConfig()
        self.text_config = text_config
        self.audio_token_id = audio_token_id
        self.image_token_id = image_token_id
        self.video_token_id = video_token_id


# ---------------------------------------------------------------------------
# Talker configs
# ---------------------------------------------------------------------------


class Qwen3OmniMoeTalkerTextConfig(Qwen3OmniMoeTextConfig):
    """Text config for the talker MoE backbone (20-layer, shared expert).

    Inherits from Qwen3OmniMoeTextConfig, adds shared_expert_intermediate_size.
    """

    model_type = "qwen3_omni_moe_talker_text"

    def __init__(
        self,
        vocab_size=3072,
        hidden_size=1024,
        intermediate_size=2048,
        num_hidden_layers=20,
        num_attention_heads=16,
        num_key_value_heads=2,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=65536,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        sliding_window=None,
        attention_dropout=0,
        decoder_sparse_step=1,
        moe_intermediate_size=384,
        shared_expert_intermediate_size=768,  # Talker-specific
        num_experts_per_tok=6,
        num_experts=128,
        norm_topk_prob=True,
        output_router_logits=False,
        router_aux_loss_coef=0.001,
        mlp_only_layers=None,
        **kwargs,
    ):
        # Call parent (Qwen3OmniMoeTextConfig) with all standard MoE params
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            num_key_value_heads=num_key_value_heads,
            hidden_act=hidden_act,
            max_position_embeddings=max_position_embeddings,
            initializer_range=initializer_range,
            rms_norm_eps=rms_norm_eps,
            use_cache=use_cache,
            tie_word_embeddings=tie_word_embeddings,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            attention_bias=attention_bias,
            sliding_window=sliding_window,
            attention_dropout=attention_dropout,
            decoder_sparse_step=decoder_sparse_step,
            moe_intermediate_size=moe_intermediate_size,
            num_experts_per_tok=num_experts_per_tok,
            num_experts=num_experts,
            norm_topk_prob=norm_topk_prob,
            output_router_logits=output_router_logits,
            router_aux_loss_coef=router_aux_loss_coef,
            mlp_only_layers=mlp_only_layers,
            **kwargs,
        )

        # Add Talker-specific field
        self.head_dim = head_dim
        self.shared_expert_intermediate_size = shared_expert_intermediate_size


class Qwen3OmniMoeTalkerCodePredictorConfig(PretrainedConfig):
    """Config for the CodePredictor (5-layer dense transformer)."""

    model_type = "qwen3_omni_moe_talker_code_predictor"

    def __init__(
        self,
        vocab_size=2048,
        hidden_size=1024,
        intermediate_size=3072,
        num_hidden_layers=5,
        num_attention_heads=16,
        num_key_value_heads=8,
        head_dim=128,
        hidden_act="silu",
        max_position_embeddings=32768,
        initializer_range=0.02,
        rms_norm_eps=1e-6,
        use_cache=True,
        tie_word_embeddings=False,
        rope_theta=1000000.0,
        rope_scaling=None,
        attention_bias=False,
        sliding_window=None,
        attention_dropout=0,
        num_code_groups=16,
        **kwargs,
    ):
        super().__init__(tie_word_embeddings=tie_word_embeddings, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.sliding_window = sliding_window
        self.attention_dropout = attention_dropout
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        self.num_code_groups = num_code_groups


class Qwen3OmniMoeTalkerConfig(PretrainedConfig):
    """Top-level talker config combining text model + code predictor."""

    model_type = "qwen3_omni_moe_talker"

    def __init__(
        self,
        text_config: dict | Qwen3OmniMoeTalkerTextConfig | None = None,
        code_predictor_config: (
            dict | Qwen3OmniMoeTalkerCodePredictorConfig | None
        ) = None,
        num_code_groups=16,
        thinker_hidden_size=2048,
        accept_hidden_layer=24,
        codec_eos_token_id=2150,
        codec_nothink_id=2155,
        codec_think_bos_id=2156,
        codec_think_eos_id=2157,
        codec_pad_id=2148,
        codec_bos_id=2149,
        **kwargs,
    ):
        super().__init__(**kwargs)
        if text_config is None:
            self.text_config = Qwen3OmniMoeTalkerTextConfig()
        elif isinstance(text_config, dict):
            self.text_config = Qwen3OmniMoeTalkerTextConfig(**text_config)
        else:
            self.text_config = text_config

        if code_predictor_config is None:
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig()
        elif isinstance(code_predictor_config, dict):
            self.code_predictor_config = Qwen3OmniMoeTalkerCodePredictorConfig(
                **code_predictor_config
            )
        else:
            self.code_predictor_config = code_predictor_config

        self.num_code_groups = num_code_groups
        self.thinker_hidden_size = thinker_hidden_size
        self.accept_hidden_layer = accept_hidden_layer
        self.codec_eos_token_id = codec_eos_token_id
        self.codec_nothink_id = codec_nothink_id
        self.codec_think_bos_id = codec_think_bos_id
        self.codec_think_eos_id = codec_think_eos_id
        self.codec_pad_id = codec_pad_id
        self.codec_bos_id = codec_bos_id
