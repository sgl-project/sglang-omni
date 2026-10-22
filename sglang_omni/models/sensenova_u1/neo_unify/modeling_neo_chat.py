"""Modeling neo chat."""

from typing import Literal, Never

import torch.utils.checkpoint
import transformers
from torch import nn
from transformers import PreTrainedTokenizerBase
from transformers.modeling_utils import PreTrainedModel

from sglang_omni.models.sensenova_u1.neo_unify.chat_generation import chat, generate
from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOChatConfig,
    NEOMoELLMConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.conversation import (
    Conversation,
    get_conv_template,
)
from sglang_omni.models.sensenova_u1.neo_unify.flash_cache import (
    clear_flash_kv_cache as clear_flash_kv_cache,
)
from sglang_omni.models.sensenova_u1.neo_unify.flash_cache import (
    prepare_flash_kv_cache as prepare_flash_kv_cache,
)
from sglang_omni.models.sensenova_u1.neo_unify.flow_matching import (
    FlowMatchingHead,
    TimestepEmbedder,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_inputs import (
    apply_time_schedule,
    build_cfg_schedule,
    build_i2i_cfg_schedule,
    build_it2i_inputs,
    build_t2i_image_indexes,
    build_t2i_query,
    build_t2i_text_inputs,
    calculate_dynamic_mu,
    euler_step,
    get_thw_indexes,
    patchify,
    unpatchify,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_math import (
    SenseNovaStageTimer as _SenseNovaStageTimer,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_math import (
    build_abs_positions_from_grid_hw as build_abs_positions_from_grid_hw,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_math import (
    optimized_scale as optimized_scale,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_math import (
    randn_with_seed as randn_with_seed,
)
from sglang_omni.models.sensenova_u1.neo_unify.generation_math import (
    version_cmp as version_cmp,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_conditioned_generation import (
    it2i_generate,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_decoders import ConvDecoder
from sglang_omni.models.sensenova_u1.neo_unify.interleaved_generation import (
    interleave_gen,
)
from sglang_omni.models.sensenova_u1.neo_unify.interleaved_images import (
    interleave_gen_image_only,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_vit import NEOVisionModel
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import Qwen3ForCausalLM
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
)
from sglang_omni.models.sensenova_u1.neo_unify.native_thinking import (
    append_text_tokens_to_cache,
    generate_think,
    it2i_prefix_forward,
    t2i_predict_v,
    t2i_prefix_forward,
    think_prefix_forward,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    KV_DIAGNOSTIC_DIR as _KV_DIAGNOSTIC_DIR,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    KV_IMPORT_PINNED_CHUNK_BYTES as _KV_IMPORT_PINNED_CHUNK_BYTES,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    KV_TRANSFER_DIR as _KV_TRANSFER_DIR,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    USE_SRT_KV_TRANSFER as _USE_SRT_KV_TRANSFER,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    BatchedSRTThinkingFallbackRequired as BatchedSRTThinkingFallbackRequired,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    SenseNovaPreallocatedLayer as _SenseNovaPreallocatedLayer,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    dump_native_prefix_kv as _dump_native_prefix_kv,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    load_srt_prefix_kv as load_srt_prefix_kv,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    merge_srt_prefix_caches as _merge_srt_prefix_caches,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    preallocate_think_cache as preallocate_think_cache,
)
from sglang_omni.models.sensenova_u1.neo_unify.prefix_cache import (
    token_ids_sha256 as _token_ids_sha256,
)
from sglang_omni.models.sensenova_u1.neo_unify.srt_thinking import (
    import_srt_prefix_tokens,
    replay_srt_think_prefix,
    request_srt_think_tokens,
    try_import_srt_text_prefix,
    try_import_srt_text_prefix_batch,
)
from sglang_omni.models.sensenova_u1.neo_unify.text_to_image_generation import (
    t2i_generate,
)

# Modified for SGLang; see this directory's README.md for upstream source.


GenerationValue = bool | int | float | str | torch.Tensor | list[int] | None


class NEOChatModel(PreTrainedModel):
    config_class: type[NEOChatConfig] = NEOChatConfig
    main_input_name: str = "pixel_values"
    base_model_prefix: str = "language_model"
    _supports_flash_attn_2: bool = True
    supports_gradient_checkpointing: bool = True
    _no_split_modules: list[str] = [
        "NEOVisionModel",
        "Qwen3DecoderLayer",
        "Qwen3MoeDecoderLayer",
    ]
    _denoise_offload_module_paths: tuple[str, ...] = (
        "language_model.model.embed_tokens",
        "language_model.lm_head",
    )

    _tp_plan: str = ""

    def __init__(
        self,
        config: NEOChatConfig,
        vision_model: NEOVisionModel | None = None,
        language_model: Qwen3ForCausalLM | Qwen3MoeForCausalLM | None = None,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__(config)

        assert version_cmp(transformers.__version__, "4.37.0", "ge")
        patch_size = config.vision_config.patch_size
        self.patch_size: int = patch_size
        self.template: str = config.template
        self.downsample_ratio: float = config.downsample_ratio
        config.llm_config._attn_implementation = (  # noqa: leading-underscore -- Transformers attention backend field
            "eager"
        )

        if vision_model is not None:
            self.vision_model: NEOVisionModel = vision_model
        else:
            self.vision_model = NEOVisionModel(config.vision_config)
        vision_model_mot_gen = NEOVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model: Qwen3ForCausalLM | Qwen3MoeForCausalLM = language_model
        else:
            if isinstance(config.llm_config, NEOMoELLMConfig):
                self.language_model = Qwen3MoeForCausalLM(config.llm_config)
            else:
                self.language_model = Qwen3ForCausalLM(config.llm_config)

        merge_size = int(1 / self.downsample_ratio)
        output_dim = 3 * (patch_size * merge_size) ** 2
        llm_hidden_size = self.config.llm_config.hidden_size
        self.use_deep_fm_head: bool = self.config.fm_head_layers > 2
        self.use_pixel_head: bool = self.config.use_pixel_head
        if self.use_pixel_head:
            fm_head = ConvDecoder(llm_hidden_size)
        elif self.use_deep_fm_head:
            fm_head = FlowMatchingHead(
                llm_hidden_size,
                output_dim,
                dim=self.config.fm_head_dim,
                layers=self.config.fm_head_layers,
                mlp_ratio=self.config.fm_head_mlp_ratio,
            )
        else:
            fm_head = nn.Sequential(
                nn.Linear(llm_hidden_size, 4096, bias=True),
                nn.GELU(),
                nn.Linear(4096, output_dim, bias=True),
            )

        timestep_embedder = TimestepEmbedder(llm_hidden_size)
        self.fm_modules: nn.ModuleDict = nn.ModuleDict(
            {
                "vision_model_mot_gen": vision_model_mot_gen,
                "timestep_embedder": timestep_embedder,
                "fm_head": fm_head,
            }
        )

        self.concat_time_token_num: int = config.concat_time_token_num
        self.noise_scale: float = config.noise_scale
        self.noise_scale_mode: str = config.noise_scale_mode
        self.noise_scale_base_image_seq_len: int = config.noise_scale_base_image_seq_len
        self.add_noise_scale_embedding: bool = config.add_noise_scale_embedding
        self.noise_scale_max_value: float = config.noise_scale_max_value
        self.time_schedule: Literal["standard", "dynamic"] = config.time_schedule
        self.time_shift_type: Literal["exponential", "linear"] = config.time_shift_type
        self.base_shift: float = config.base_shift
        self.max_shift: float = config.max_shift
        self.base_image_seq_len: int = config.base_image_seq_len
        self.max_image_seq_len: int = config.max_image_seq_len

        if self.add_noise_scale_embedding:
            noise_scale_embedder = TimestepEmbedder(llm_hidden_size)
            self.fm_modules["noise_scale_embedder"] = noise_scale_embedder
        else:
            pass

        self.img_context_token_id: int | None = None
        self.img_start_token_id: int = 151670
        self.last_think_content: str | list[str] = ""
        self.last_think_token_count: int = 0
        self.last_profile_timings_ms: dict[str, float] = {}
        self.conv_template: Conversation = get_conv_template(self.template)
        self.system_message: str = self.conv_template.system_message

        # note (Wes): Transformers 5 requires composite tied-weight metadata before loading.
        self.post_init()

    def notify_layer_offload_phase(self, phase: str) -> None:
        callback = getattr(self, "layer_offload_phase_callback", None)
        if callback is not None:
            callback(phase)
        else:
            pass

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        image_flags: torch.LongTensor | None = None,
        past_key_values: list[torch.FloatTensor] | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        output_attentions: bool | None = None,
        output_hidden_states: bool | None = None,
        return_dict: bool | None = None,
    ) -> Never:
        raise NotImplementedError("forward")

    def extract_feature(
        self,
        pixel_values: torch.Tensor | None,
        gen_model: bool = False,
        grid_hw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if gen_model:
            return self.fm_modules["vision_model_mot_gen"](
                pixel_values=pixel_values,
                output_hidden_states=False,
                grid_hw=grid_hw,
            ).last_hidden_state
        else:
            return self.vision_model(
                pixel_values=pixel_values,
                output_hidden_states=False,
                grid_hw=grid_hw,
            ).last_hidden_state

    def batch_chat(
        self,
        tokenizer: PreTrainedTokenizerBase,
        pixel_values: torch.Tensor | None,
        questions: list[str],
        generation_config: dict[str, GenerationValue],
        num_patches_list: list[int] | None = None,
        history: list[tuple[str, str]] | None = None,
        return_history: bool = False,
        IMG_START_TOKEN: str = "<img>",
        IMG_END_TOKEN: str = "</img>",
        IMG_CONTEXT_TOKEN: str = "<IMG_CONTEXT>",
        verbose: bool = False,
        image_counts: list[int] | None = None,
    ) -> Never:
        raise NotImplementedError("batch_chat")

    patchify = patchify

    unpatchify = unpatchify

    euler_step = euler_step

    build_cfg_schedule = staticmethod(build_cfg_schedule)

    build_i2i_cfg_schedule = staticmethod(build_i2i_cfg_schedule)

    calculate_dynamic_mu = calculate_dynamic_mu

    apply_time_schedule = apply_time_schedule

    build_t2i_query = build_t2i_query

    build_t2i_text_inputs = build_t2i_text_inputs

    build_t2i_image_indexes = build_t2i_image_indexes

    t2i_prefix_forward = t2i_prefix_forward

    it2i_prefix_forward = it2i_prefix_forward

    think_prefix_forward = think_prefix_forward

    append_text_tokens_to_cache = append_text_tokens_to_cache

    request_srt_think_tokens = request_srt_think_tokens

    import_srt_prefix_tokens = import_srt_prefix_tokens

    try_import_srt_text_prefix = try_import_srt_text_prefix

    try_import_srt_text_prefix_batch = try_import_srt_text_prefix_batch

    replay_srt_think_prefix = replay_srt_think_prefix

    generate_think = generate_think

    t2i_predict_v = t2i_predict_v

    build_it2i_inputs = build_it2i_inputs

    interleave_gen_image_only = interleave_gen_image_only

    interleave_gen = interleave_gen

    it2i_generate = it2i_generate

    t2i_generate = t2i_generate

    chat = chat

    generate = generate

    @property
    def lm_head(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        return self.language_model.set_input_embeddings(value)

    def set_output_embeddings(self, value: nn.Module) -> None:
        return self.language_model.set_output_embeddings(value)

    get_thw_indexes = get_thw_indexes


__all__ = [
    "BatchedSRTThinkingFallbackRequired",
    "NEOChatModel",
    "_KV_DIAGNOSTIC_DIR",
    "_KV_IMPORT_PINNED_CHUNK_BYTES",
    "_KV_TRANSFER_DIR",
    "_SenseNovaPreallocatedLayer",
    "_SenseNovaStageTimer",
    "_USE_SRT_KV_TRANSFER",
    "_dump_native_prefix_kv",
    "load_srt_prefix_kv",
    "_merge_srt_prefix_caches",
    "preallocate_think_cache",
    "randn_with_seed",
    "_token_ids_sha256",
    "build_abs_positions_from_grid_hw",
    "clear_flash_kv_cache",
    "optimized_scale",
    "prepare_flash_kv_cache",
    "version_cmp",
]
