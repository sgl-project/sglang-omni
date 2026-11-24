# SPDX-License-Identifier: Apache-2.0
"""LFM3 AR stage model: LFM3 backbone, speech-frame embeddings, depth decoder and media encoders.

Serves LFM3-Audio (audio encoder) and LFM3-Omni (audio encoder plus vision tower). The encoders are the checkpoint's own modules (custom code).

One step produces the backbone's last hidden state and text logits. The model runner samples a speech frame from the hidden state when the request is speaking and feeds the frame back as the next step's input embedding; every decode row reads its input from decode_input_embed_buffer.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_schedule
from torch import nn
from transformers import AutoConfig
from transformers.dynamic_module_utils import get_class_from_dynamic_module

from sglang_omni.models.lfm3_audio.backbone import CHECKPOINT_PREFIX, Lfm3TextModel
from sglang_omni.models.lfm3_audio.depth import DEPTH_PREFIXES, Lfm3DepthDecoder
from sglang_omni.models.lfm3_audio.depth_cuda_graph import Lfm3DepthCudaGraphs
from sglang_omni.models.lfm3_audio.hf_config import (
    Lfm3SGLangConfig,
    is_export_format,
    lfm3_tokens_from_checkpoint,
    load_checkpoint_config,
)

PROJECTOR_PREFIX = "mlp.model."
# LFM3-Audio's projector is an nn.Sequential(LayerNorm, Linear, GELU, Linear).
PROJECTOR_RENAMES = {
    "mlp.model.0.": "layer_norm.",
    "mlp.model.1.": "linear_1.",
    "mlp.model.3.": "linear_2.",
}
# LFM3-Omni's Transformers names, mapped onto the export names of the shared parts.
OMNI_RENAMES = (
    ("model.language_model.", "lfm.model."),
    ("model.codebook_embedding.", "codebook_embedding."),
    ("model.depth_decoder.model.", "depthformer.model."),
    ("model.depth_decoder.", ""),
    ("model.audio_encoder.downsample.", "encoder.downsample."),
    ("model.audio_encoder.lfm.", "encoder.lfm.lfm."),
)
# LFM3-Omni's checkpoint-code modules whose weights load under their own names.
OMNI_MODULE_PREFIXES = (
    "model.audio_projector.",
    "model.vision_tower.",
    "model.multi_modal_projector.",
)


def canonical_weight_name(name: str) -> str:
    """LFM3-Omni's names of the shared parts, as LFM3-Audio's export names them."""
    for old, new in OMNI_RENAMES:
        if name.startswith(old):
            return new + name[len(old) :]
        else:
            pass
    return name


class Lfm3AudioSGLangModel(nn.Module):
    def __init__(
        self,
        config: Lfm3SGLangConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        del quant_config, prefix
        super().__init__()
        self.config = config
        checkpoint_dir = config.checkpoint_dir
        self.tokens = lfm3_tokens_from_checkpoint(checkpoint_dir)
        self.hidden_size = config.hidden_size
        self.language_model = Lfm3TextModel(config, prefix="language_model")
        self.codebook_embedding = nn.Embedding(
            config.num_codebooks * config.codebook_size, config.hidden_size
        )
        transformers_config = AutoConfig.from_pretrained(
            checkpoint_dir, trust_remote_code=True
        )

        def checkpoint_class(class_path: str) -> type[nn.Module]:
            return get_class_from_dynamic_module(class_path, checkpoint_dir)

        # note (ruilin): built on the CPU as Transformers builds them, so the rotary frequency buffers match the checkpoint's code; load_weights moves them to the device.
        with torch.device("cpu"):
            self.depth_decoder = Lfm3DepthDecoder(
                config.depth_config,
                config.hidden_size,
                config.num_codebooks,
                config.codebook_size,
            )
            if is_export_format(load_checkpoint_config(checkpoint_dir)):
                self.audio_encoder = checkpoint_class(
                    "modeling_lfm3_audio.Lfm3AudioEncoder"
                )(transformers_config.audio_encoder_config)
                self.audio_projector = checkpoint_class(
                    "modeling_lfm3_audio.Lfm3AudioProjector"
                )(transformers_config)
                self.vision_tower = None
                self.multi_modal_projector = None
            else:
                self.audio_encoder = checkpoint_class(
                    "modeling_lfm3_omni.Lfm3OmniEncoder"
                )(transformers_config.audio_encoder_config)
                self.audio_projector = checkpoint_class(
                    "modeling_lfm3_omni.Lfm3OmniProjector"
                )(transformers_config)
                self.vision_tower = checkpoint_class(
                    "modeling_lfm3_vl.Lfm3VlVisionModel"
                )(transformers_config.vision_config)
                self.multi_modal_projector = checkpoint_class(
                    "modeling_lfm3_omni.Lfm3OmniMultiModalProjector"
                )(transformers_config)
        # Set by the engine builder after the decode graphs are captured.
        self.depth_graphs: Lfm3DepthCudaGraphs | None = None
        max_batch_size = get_schedule().max_running_requests
        self.register_buffer(
            "decode_input_embed_buffer",
            torch.zeros(max_batch_size, config.hidden_size),
            persistent=False,
        )
        # A row whose request asked for text only may not open speech.
        self.register_buffer(
            "text_only_buffer",
            torch.zeros(max_batch_size, dtype=torch.bool),
            persistent=False,
        )
        self.register_buffer(
            "codebook_offsets",
            torch.arange(config.num_codebooks) * config.codebook_size,
            persistent=False,
        )
        self.register_buffer(
            "placeholder_token_ids",
            torch.tensor(self.tokens.placeholder_token_ids),
            persistent=False,
        )
        self.register_buffer(
            "speech_marker_ids",
            torch.tensor(
                [self.tokens.mixed_start_token_id, self.tokens.audio_start_token_id]
            ),
            persistent=False,
        )

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.embed_tokens

    def get_attention_sliding_window_size(self) -> int:
        # note (ruilin): the runtime's window excludes the current token; without this the attention backend builds no window KV indices for the sliding layers.
        return self.config.sliding_window - 1

    def frame_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum of the codebook embeddings of each frame: (N, 12) -> (N, hidden)."""
        return self.codebook_embedding(codes + self.codebook_offsets).sum(-2)

    @torch.no_grad()
    def encode_audio(
        self, features: torch.Tensor, feature_mask: torch.Tensor
    ) -> torch.Tensor:
        """Projected speech features of the clips in order, one row per audio token."""
        dtype = self.audio_encoder.downsample.conv_in.weight.dtype
        lengths = feature_mask.sum(-1)
        encoded, encoded_lengths = self.audio_encoder(features.to(dtype), lengths)
        valid = (
            torch.arange(encoded.shape[1], device=encoded.device)[None, :]
            < encoded_lengths[:, None]
        )
        return self.audio_projector(encoded[valid])

    @torch.no_grad()
    def encode_vision(
        self, pixel_values: torch.Tensor, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        """Projected features of the images (or videos) in order, one row per vision token."""
        assert (
            self.vision_tower is not None
        ), "preprocessing admits images and videos for LFM3-Omni only"
        dtype = self.multi_modal_projector.linear_1.weight.dtype
        vision = self.vision_tower(
            pixel_values.to(dtype), grid_thw=grid_thw, return_dict=True
        )
        return self.multi_modal_projector(vision.last_hidden_state)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
    ) -> LogitsProcessorOutput:
        if forward_batch.forward_mode.is_decode():
            input_embeds = self.decode_input_embed_buffer[: input_ids.shape[0]]
        elif input_embeds is None:
            raise RuntimeError(
                "LFM3 prefill takes its input embeddings from the model runner"
            )
        else:
            pass
        hidden_states = self.language_model(
            input_ids, positions, forward_batch, input_embeds=input_embeds
        )
        if forward_batch.forward_mode.is_extend():
            last_index = (
                torch.cumsum(
                    forward_batch.extend_seq_lens.to(hidden_states.device), dim=0
                )
                - 1
            )
            hidden_states = hidden_states[last_index]
        else:
            pass
        logits = F.linear(
            hidden_states, self.language_model.embed_tokens.weight
        ).float()
        # Media placeholders never come out of the text head; text-only rows cannot open speech.
        logits.index_fill_(1, self.placeholder_token_ids, float("-inf"))
        rows = logits.shape[0]
        text_only = self.text_only_buffer[:rows, None]
        marker_logits = logits.index_select(1, self.speech_marker_ids)
        logits.index_copy_(
            1,
            self.speech_marker_ids,
            torch.where(
                text_only, torch.full_like(marker_logits, float("-inf")), marker_logits
            ),
        )
        return LogitsProcessorOutput(
            next_token_logits=logits, hidden_states=hidden_states
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        parameters = dict(self.named_parameters())
        backbone_parameters = {
            name[len("language_model.") :]: parameter
            for name, parameter in parameters.items()
            if name.startswith("language_model.")
        }
        loaded: set[str] = set()
        unexpected: list[str] = []
        for checkpoint_name, tensor in weights:
            name = canonical_weight_name(checkpoint_name)
            if checkpoint_name.startswith(OMNI_MODULE_PREFIXES):
                # note (ruilin): LFM3-Omni's checkpoint-code modules sit at the same paths, without the model prefix.
                target = checkpoint_name[len("model.") :]
            elif name.startswith(CHECKPOINT_PREFIX):
                target = self.language_model.load_weight(
                    name[len(CHECKPOINT_PREFIX) :], tensor, backbone_parameters
                )
                if target is None:
                    unexpected.append(checkpoint_name)
                else:
                    loaded.add("language_model." + target)
                continue
            elif name == "lfm.lm_head.weight":
                # note (ruilin): LFM3-Audio stores the tied text head; the token embedding is the same tensor.
                continue
            elif name.startswith(DEPTH_PREFIXES):
                if self.depth_decoder.load_weight(name, tensor):
                    loaded.add("depth_decoder." + name)
                else:
                    unexpected.append(checkpoint_name)
                continue
            elif name.startswith("encoder.downsample."):
                target = (
                    "audio_encoder.downsample." + name[len("encoder.downsample.") :]
                )
            elif name.startswith("encoder.lfm.lfm."):
                target = "audio_encoder.lfm." + name[len("encoder.lfm.lfm.") :]
            elif name.startswith(PROJECTOR_PREFIX):
                target = "audio_projector." + next(
                    new + name[len(old) :]
                    for old, new in PROJECTOR_RENAMES.items()
                    if name.startswith(old)
                )
            else:
                target = name
            if target in parameters:
                with torch.no_grad():
                    parameters[target].copy_(tensor.to(parameters[target].dtype))
                loaded.add(target)
            else:
                unexpected.append(checkpoint_name)
        if unexpected:
            raise ValueError(f"Unexpected LFM3 weights: {unexpected[:8]}")
        else:
            pass
        device = self.language_model.embed_tokens.weight.device
        for module in (
            self.depth_decoder,
            self.audio_encoder,
            self.audio_projector,
            self.vision_tower,
            self.multi_modal_projector,
        ):
            if module is not None:
                module.to(device)
            else:
                pass
        return loaded


EntryClass = Lfm3AudioSGLangModel
