# SPDX-License-Identifier: Apache-2.0
"""LFM3 AR stage model: LFM3 backbone, speech-frame embeddings, depth decoder and media encoders.

Serves LFM3-Audio (audio encoder) and LFM3-Omni (audio encoder plus vision
tower). The encoders are the checkpoint's own modules (custom code).

One step produces the backbone's last hidden state and text logits. The model
runner samples a speech frame from the hidden state when the request is
speaking and feeds the frame back as the next step's input embedding: a decode
row whose input id is ``<|audio_out|>`` reads its embedding from
``decode_input_embed_buffer`` instead of the token table.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.runtime_context import get_schedule
from torch import nn

from sglang_omni.models.lfm3_audio.backbone import CHECKPOINT_PREFIX, Lfm3TextModel
from sglang_omni.models.lfm3_audio.depth import DEPTH_PREFIXES, Lfm3DepthDecoder
from sglang_omni.models.lfm3_audio.hf_config import (
    Lfm3AudioTokens,
    Lfm3SGLangConfig,
    is_export_format,
    lfm3_tokens_from_checkpoint,
    load_checkpoint_config,
)

logger = logging.getLogger(__name__)

ENCODER_PREFIXES = ("encoder.", "mlp.model.")
# The export's projector is an nn.Sequential(LayerNorm, Linear, GELU, Linear).
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
# LFM3-Omni modules loaded as they are: checkpoint prefix -> attribute.
OMNI_MODULES = {
    "model.audio_projector.": "audio_projector",
    "model.vision_tower.": "vision_tower",
    "model.multi_modal_projector.": "multi_modal_projector",
}


def canonical_weight_name(name: str) -> str:
    for old, new in OMNI_RENAMES:
        if name.startswith(old):
            return new + name[len(old) :]
        else:
            pass
    return name


def build_media_modules(checkpoint_dir: str) -> dict[str, nn.Module]:
    """The checkpoint's own audio encoder and projector, plus LFM3-Omni's vision tower and projector."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    def load(name: str) -> type:
        return get_class_from_dynamic_module(name, checkpoint_dir)

    ref_config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    if is_export_format(load_checkpoint_config(checkpoint_dir)):
        return {
            "audio_encoder": load("modeling_lfm3_audio.Lfm3AudioEncoder")(
                ref_config.audio_encoder_config
            ),
            "audio_projector": load("modeling_lfm3_audio.Lfm3AudioProjector")(
                ref_config
            ),
        }
    else:
        return {
            "audio_encoder": load("modeling_lfm3_omni.Lfm3OmniEncoder")(
                ref_config.audio_encoder_config
            ),
            "audio_projector": load("modeling_lfm3_omni.Lfm3OmniProjector")(ref_config),
            "vision_tower": load("modeling_lfm3_vl.Lfm3VlVisionModel")(
                ref_config.vision_config
            ),
            "multi_modal_projector": load(
                "modeling_lfm3_omni.Lfm3OmniMultiModalProjector"
            )(ref_config),
        }


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
        checkpoint_dir = getattr(config, "checkpoint_dir", None)
        if checkpoint_dir:
            self.tokens = lfm3_tokens_from_checkpoint(checkpoint_dir)
        else:
            # Unit tests build the model without a checkpoint.
            self.tokens = Lfm3AudioTokens(
                num_codebooks=config.num_codebooks,
                codebook_size=config.codebook_size,
                audio_eos_code=config.codebook_size - 1,
                maskgit_steps=config.maskgit_steps,
                interleaved_n_text=config.interleaved_n_text,
                interleaved_n_audio=config.interleaved_n_audio,
            )
        self.hidden_size = config.hidden_size
        self.language_model = Lfm3TextModel(config, prefix="language_model")
        self.codebook_embedding = nn.Embedding(
            config.num_codebooks * config.codebook_size, config.hidden_size
        )
        # Built on the CPU as Transformers builds them, so the rotary frequency
        # buffers match the reference; load_weights moves them to the device.
        with torch.device("cpu"):
            self.depth_decoder = Lfm3DepthDecoder(
                config.depth_config,
                config.hidden_size,
                config.num_codebooks,
                config.codebook_size,
            )
            media = build_media_modules(checkpoint_dir) if checkpoint_dir else {}
        self.audio_encoder = media.get("audio_encoder")
        self.audio_projector = media.get("audio_projector")
        self.vision_tower = media.get("vision_tower")
        self.multi_modal_projector = media.get("multi_modal_projector")
        max_batch_size = get_schedule().max_running_requests
        self.register_buffer(
            "decode_input_embed_buffer",
            torch.zeros(max_batch_size, config.hidden_size),
            persistent=False,
        )
        # Rows whose request asked for text only also may not open speech.
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
            "placeholder_ids",
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
        # The runtime's window excludes the current token, so the attention
        # backend builds the window KV indices of the sliding layers.
        return self.config.sliding_window - 1

    def frame_embeddings(self, codes: torch.Tensor) -> torch.Tensor:
        """Sum of the codebook embeddings of each frame: (N, 12) -> (N, hidden)."""
        return self.codebook_embedding(codes + self.codebook_offsets).sum(-2)

    @torch.no_grad()
    def encode_audio(
        self, features: torch.Tensor, feature_mask: torch.Tensor
    ) -> torch.Tensor:
        """Projected speech features of the clips in order, one row per audio token."""
        if self.audio_encoder is None:
            raise RuntimeError("LFM3 was built without its audio encoder")
        else:
            pass
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
        if self.vision_tower is None:
            raise RuntimeError("This LFM3 checkpoint has no vision tower")
        else:
            pass
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
        batch_rows = None
        if input_embeds is None:
            input_embeds = self.language_model.embed_tokens(input_ids)
            if forward_batch.forward_mode.is_decode():
                batch_rows = input_ids.shape[0]
                is_frame = (input_ids == self.tokens.audio_out_token_id)[:, None]
                input_embeds = torch.where(
                    is_frame,
                    self.decode_input_embed_buffer[:batch_rows].to(input_embeds.dtype),
                    input_embeds,
                )
            else:
                pass
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
        logits.index_fill_(1, self.placeholder_ids, float("-inf"))
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
        params = dict(self.named_parameters())
        backbone_params = {
            key[len("language_model.") :]: value
            for key, value in params.items()
            if key.startswith("language_model.")
        }
        loaded: set[str] = set()
        skipped: list[str] = []
        for checkpoint_name, tensor in weights:
            module_prefix = next(
                (
                    prefix
                    for prefix in OMNI_MODULES
                    if checkpoint_name.startswith(prefix)
                ),
                None,
            )
            name = canonical_weight_name(checkpoint_name)
            if module_prefix is not None:
                attribute = OMNI_MODULES[module_prefix]
                module = getattr(self, attribute)
                if module is None:
                    continue
                else:
                    pass
                local = checkpoint_name[len(module_prefix) :]
                copy_into(module, local, tensor)
                loaded.add(f"{attribute}.{local}")
            elif name.startswith(CHECKPOINT_PREFIX):
                target = self.language_model.load_weight(
                    name[len(CHECKPOINT_PREFIX) :], tensor, backbone_params
                )
                if target is None:
                    skipped.append(checkpoint_name)
                else:
                    loaded.add("language_model." + target)
            elif name == "lfm.lm_head.weight":
                # Tied to the token embedding.
                continue
            elif name == "codebook_embedding.weight":
                self.codebook_embedding.weight.data.copy_(tensor)
                loaded.add("codebook_embedding.weight")
            elif name.startswith(DEPTH_PREFIXES):
                if not self.depth_decoder.load_weight(name, tensor):
                    skipped.append(checkpoint_name)
                else:
                    loaded.add("depth_decoder." + name)
            elif name.startswith(ENCODER_PREFIXES):
                if self.audio_encoder is None:
                    continue
                else:
                    pass
                self.load_encoder_weight(name, tensor)
                loaded.add(name)
            else:
                skipped.append(checkpoint_name)
        if skipped:
            raise ValueError(f"Unexpected LFM3 weights: {skipped[:8]}")
        else:
            pass
        device = self.language_model.embed_tokens.weight.device
        for module in self.cpu_built_modules():
            module.to(device)
        return loaded

    def cpu_built_modules(self) -> list[nn.Module]:
        modules = (
            self.depth_decoder,
            self.audio_encoder,
            self.audio_projector,
            self.vision_tower,
            self.multi_modal_projector,
        )
        return [module for module in modules if module is not None]

    def load_encoder_weight(self, name: str, tensor: torch.Tensor) -> None:
        if name.startswith("encoder.downsample."):
            module, local = (
                self.audio_encoder,
                "downsample." + name[len("encoder.downsample.") :],
            )
        elif name.startswith("encoder.lfm.lfm."):
            module, local = self.audio_encoder, "lfm." + name[len("encoder.lfm.lfm.") :]
        else:
            module = self.audio_projector
            local = next(
                new + name[len(old) :]
                for old, new in PROJECTOR_RENAMES.items()
                if name.startswith(old)
            )
        copy_into(module, local, tensor)


def copy_into(module: nn.Module, name: str, tensor: torch.Tensor) -> None:
    """Copy a checkpoint tensor into a parameter or buffer of one of the checkpoint's own modules."""
    targets = dict(module.named_parameters())
    targets.update(dict(module.named_buffers()))
    target = targets[name]
    with torch.no_grad():
        target.copy_(tensor.to(target.dtype))


EntryClass = Lfm3AudioSGLangModel
