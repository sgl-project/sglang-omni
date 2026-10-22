"""Typed state shared by the generation routines."""

from typing import Literal, Protocol, TypedDict, Unpack

import torch.utils.checkpoint
from torch import nn
from transformers import GenerationConfig, PreTrainedTokenizerBase
from transformers.cache_utils import DynamicCache
from transformers.modeling_outputs import CausalLMOutputWithPast

from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOChatConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_vit import NEOVisionModel
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import Qwen3ForCausalLM
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
)
from sglang_omni.models.sensenova_u1.neo_unify.thinking_types import (
    PrefixTimings,
    ThinkingBackend,
    ThinkingSession,
)

# Modified for SGLang; see this directory's README.md for upstream source.


GenerationValue = bool | int | float | str | torch.Tensor | list[int] | None


class ThinkPrefixInputs(TypedDict, total=False):
    input_ids: torch.Tensor
    inputs_embeds: torch.Tensor
    indexes: torch.Tensor
    attention_mask: dict[str, torch.Tensor]
    image_gen_indicators: torch.Tensor


class GenerationState(Protocol):
    config: NEOChatConfig
    language_model: Qwen3ForCausalLM | Qwen3MoeForCausalLM
    vision_model: NEOVisionModel
    device: torch.device
    fm_modules: nn.ModuleDict
    patch_size: int
    template: str
    downsample_ratio: float
    system_message: str
    concat_time_token_num: int
    noise_scale: float
    # note (Wes): Unrecognized noise modes retain the configured constant scale.
    noise_scale_mode: str
    noise_scale_base_image_seq_len: int
    noise_scale_max_value: float
    add_noise_scale_embedding: bool
    time_schedule: Literal["standard", "dynamic"]
    time_shift_type: Literal["exponential", "linear"]
    base_shift: float
    max_shift: float
    base_image_seq_len: int
    max_image_seq_len: int
    use_pixel_head: bool
    use_deep_fm_head: bool
    img_context_token_id: int | None
    img_start_token_id: int
    last_think_content: str | list[str]
    last_think_token_count: int
    last_think_token_counts: list[int]
    last_profile_timings_ms: dict[str, float]
    last_thinking_backend: Literal["native", "srt"] | None
    last_thinking_backends: list[Literal["native", "srt"] | None]
    last_srt_kv_transfer_used: bool
    last_srt_kv_transfer_timings: PrefixTimings
    last_srt_kv_transfer_timings_list: list[PrefixTimings]
    last_srt_kv_transferred_prefixes: list[str]

    def notify_layer_offload_phase(self, phase: str) -> None: ...

    def extract_feature(
        self,
        pixel_values: torch.Tensor | None,
        gen_model: bool = False,
        grid_hw: torch.Tensor | None = None,
    ) -> torch.Tensor: ...

    def patchify(
        self, images: torch.Tensor, patch_size: int, channel_first: bool = False
    ) -> torch.Tensor: ...

    def unpatchify(
        self,
        x: torch.Tensor,
        patch_size: int,
        h: int | None = None,
        w: int | None = None,
    ) -> torch.Tensor: ...

    def euler_step(
        self,
        v_pred: torch.Tensor,
        z: torch.Tensor,
        t: torch.Tensor,
        t_next: torch.Tensor,
    ) -> torch.Tensor: ...

    @staticmethod
    def build_cfg_schedule(
        timesteps: torch.Tensor, cfg_interval: tuple[float, float], needs_cfg: bool
    ) -> list[bool]: ...

    @staticmethod
    def build_i2i_cfg_schedule(
        timesteps: torch.Tensor, cfg_interval: tuple[float, float]
    ) -> list[bool]: ...

    def calculate_dynamic_mu(self, image_seq_len: int) -> float: ...

    def apply_time_schedule(
        self, t: torch.Tensor, image_seq_len: int, timestep_shift: float
    ) -> torch.Tensor: ...

    def build_t2i_query(
        self,
        prompt_text: str,
        system_message: str | None = None,
        append_text: str | None = None,
    ) -> str: ...

    def build_t2i_text_inputs(
        self, tokenizer: PreTrainedTokenizerBase, query: str | list[str]
    ) -> tuple[
        torch.Tensor, torch.Tensor, dict[str, torch.Tensor], torch.Tensor, torch.Tensor
    ]: ...

    def build_t2i_image_indexes(
        self,
        token_h: int,
        token_w: int,
        text_len: int | torch.Tensor,
        device: torch.device | None,
    ) -> torch.Tensor: ...

    def t2i_prefix_forward(
        self,
        input_ids: torch.Tensor,
        indexes: torch.Tensor,
        attention_mask: dict[str, torch.Tensor],
    ) -> tuple[DynamicCache, torch.Tensor]: ...

    def it2i_prefix_forward(
        self,
        input_imbeds: torch.Tensor,
        indexes: torch.Tensor,
        attention_mask: dict[str, torch.Tensor],
        gen_indicators: torch.Tensor | None = None,
    ) -> tuple[DynamicCache, torch.Tensor]: ...

    def think_prefix_forward(
        self, **kwargs: Unpack[ThinkPrefixInputs]
    ) -> CausalLMOutputWithPast: ...

    def append_text_tokens_to_cache(
        self, cache: DynamicCache, t_idx: int, input_ids: torch.Tensor
    ) -> int: ...

    def request_srt_think_tokens(
        self,
        tokenizer: PreTrainedTokenizerBase,
        thinking_backend: ThinkingBackend,
        input_ids: torch.Tensor,
        prefix_lengths: torch.Tensor,
        max_think_tokens: int,
    ) -> tuple[
        list[list[int]], list[str], ThinkingSession | list[ThinkingSession] | None
    ]: ...

    def import_srt_prefix_tokens(
        self,
        thinking_backend: ThinkingBackend,
        token_ids: list[int],
        session_context: ThinkingSession | None = None,
    ) -> tuple[DynamicCache, PrefixTimings]: ...

    def try_import_srt_text_prefix(
        self,
        thinking_backend: ThinkingBackend,
        input_ids: torch.Tensor,
        prefix_length: int | torch.Tensor,
        name: str,
    ) -> tuple[DynamicCache | None, PrefixTimings]: ...

    def try_import_srt_text_prefix_batch(
        self,
        thinking_backend: ThinkingBackend,
        input_ids: torch.Tensor,
        prefix_lengths: torch.Tensor,
        name: str,
    ) -> tuple[DynamicCache | None, list[PrefixTimings]]: ...

    def replay_srt_think_prefix(
        self,
        tokenizer: PreTrainedTokenizerBase,
        input_ids: torch.Tensor,
        prefix_lengths: torch.Tensor,
        think_token_ids: list[list[int]],
        IMG_START_TOKEN: str,
        thinking_backend: ThinkingBackend | None = None,
        transfer_context: ThinkingSession | list[ThinkingSession] | None = None,
        require_transfer: bool = False,
    ) -> tuple[
        DynamicCache, torch.Tensor | None, torch.Tensor, torch.Tensor, torch.Tensor
    ]: ...

    def generate_think(
        self,
        tokenizer: PreTrainedTokenizerBase,
        prefix_outputs: CausalLMOutputWithPast,
        past_key_values: DynamicCache,
        t_idx: int,
        IMG_START_TOKEN: str,
        max_think_tokens: int = 1024,
    ) -> tuple[DynamicCache, int, str]: ...

    def t2i_predict_v(
        self,
        input_embeds: torch.Tensor,
        indexes_image: torch.Tensor,
        attn_mask: dict[str, torch.Tensor | None] | None,
        past_key_values: DynamicCache,
        t: torch.Tensor,
        z: torch.Tensor,
        image_token_num: int,
        timestep_embeddings: torch.Tensor | None = None,
        image_size: tuple[int, int] | None = None,
    ) -> torch.Tensor: ...

    def build_it2i_inputs(
        self,
        tokenizer: PreTrainedTokenizerBase,
        query: str,
        pixel_values: torch.Tensor | None = None,
        grid_hw: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]: ...

    def generate(
        self,
        pixel_values: torch.FloatTensor | None = None,
        input_ids: torch.FloatTensor | None = None,
        grid_hw: torch.LongTensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        visual_features: torch.FloatTensor | None = None,
        generation_config: GenerationConfig | None = None,
        output_hidden_states: bool | None = None,
        **generate_kwargs: GenerationValue,
    ) -> torch.LongTensor: ...

    def get_output_embeddings(self) -> nn.Module: ...

    def get_input_embeddings(self) -> nn.Module: ...

    def get_thw_indexes(
        self, input_ids: torch.Tensor, grid_hw: torch.Tensor | None = None
    ) -> torch.Tensor: ...
