"""Modeling neo chat."""

from typing import Never

import torch.utils.checkpoint
from torch import nn
from transformers.modeling_utils import PreTrainedModel

from sglang_omni.models.sensenova_u1.neo_unify.configuration_neo_chat import (
    NEOChatConfig,
    NEOMoELLMConfig,
)
from sglang_omni.models.sensenova_u1.neo_unify.conversation import (
    Conversation,
    get_conv_template,
)
from sglang_omni.models.sensenova_u1.neo_unify.flow_matching import (
    FlowMatchingHead,
    TimestepEmbedder,
)
from sglang_omni.models.sensenova_u1.neo_unify.image_decoders import ConvDecoder
from sglang_omni.models.sensenova_u1.neo_unify.modeling_neo_vit import NEOVisionModel
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3 import Qwen3ForCausalLM
from sglang_omni.models.sensenova_u1.neo_unify.modeling_qwen3_moe import (
    Qwen3MoeForCausalLM,
)

# Modified for SGLang; see this directory's README.md for upstream source.


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
    _tp_plan: str = ""

    def __init__(
        self,
        config: NEOChatConfig,
        vision_model: NEOVisionModel | None = None,
        language_model: Qwen3ForCausalLM | Qwen3MoeForCausalLM | None = None,
    ) -> None:
        super().__init__(config)

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

        if self.add_noise_scale_embedding:
            noise_scale_embedder = TimestepEmbedder(llm_hidden_size)
            self.fm_modules["noise_scale_embedder"] = noise_scale_embedder
        else:
            pass

        self.img_context_token_id: int | None = None
        self.img_start_token_id: int = 151670
        self.conv_template: Conversation = get_conv_template(self.template)
        self.system_message: str = self.conv_template.system_message

        # note (Wes): Transformers 5 requires composite tied-weight metadata before loading.
        self.post_init()

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


__all__ = ["NEOChatModel"]
