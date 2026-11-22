# SPDX-License-Identifier: Apache-2.0
"""LFM3-Audio AR stage model: LFM3 backbone, speech-frame embeddings, depth decoder, audio encoder.

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
from sglang_omni.models.lfm3_audio.hf_config import Lfm3AudioTokens, Lfm3SGLangConfig

logger = logging.getLogger(__name__)

ENCODER_PREFIXES = ("encoder.", "mlp.model.")
# The export's projector is an nn.Sequential(LayerNorm, Linear, GELU, Linear).
PROJECTOR_RENAMES = {
    "mlp.model.0.": "layer_norm.",
    "mlp.model.1.": "linear_1.",
    "mlp.model.3.": "linear_2.",
}


def load_reference_classes(checkpoint_dir: str) -> tuple[type, type, object]:
    """The checkpoint's own audio encoder, projector and config classes (custom code)."""
    from transformers import AutoConfig
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    encoder_cls = get_class_from_dynamic_module(
        "modeling_lfm3_audio.Lfm3AudioEncoder", checkpoint_dir
    )
    projector_cls = get_class_from_dynamic_module(
        "modeling_lfm3_audio.Lfm3AudioProjector", checkpoint_dir
    )
    config = AutoConfig.from_pretrained(checkpoint_dir, trust_remote_code=True)
    return encoder_cls, projector_cls, config


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
        self.depth_decoder = Lfm3DepthDecoder(
            config.depth_config,
            config.hidden_size,
            config.num_codebooks,
            config.codebook_size,
        )
        checkpoint_dir = getattr(config, "checkpoint_dir", None)
        if checkpoint_dir:
            encoder_cls, projector_cls, ref_config = load_reference_classes(
                checkpoint_dir
            )
            self.audio_encoder = encoder_cls(ref_config.audio_encoder_config)
            self.audio_projector = projector_cls(ref_config)
        else:
            # Unit tests build the model without the checkpoint's custom code.
            self.audio_encoder = None
            self.audio_projector = None
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
            raise RuntimeError("LFM3-Audio was built without its audio encoder")
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
        loaded: set[str] = set()
        skipped: list[str] = []
        for name, tensor in weights:
            if name.startswith(CHECKPOINT_PREFIX):
                local = "language_model." + name[len(CHECKPOINT_PREFIX) :]
                target = self.language_model.load_weight(
                    local[len("language_model.") :],
                    tensor,
                    {
                        key[len("language_model.") :]: value
                        for key, value in params.items()
                        if key.startswith("language_model.")
                    },
                )
                if target is None:
                    skipped.append(name)
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
                    skipped.append(name)
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
                skipped.append(name)
        if skipped:
            raise ValueError(f"Unexpected LFM3-Audio weights: {skipped[:8]}")
        else:
            pass
        return loaded

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
        param = dict(module.named_parameters())[local]
        with torch.no_grad():
            param.copy_(tensor.to(param.dtype))


EntryClass = Lfm3AudioSGLangModel
