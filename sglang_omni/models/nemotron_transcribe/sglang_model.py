# SPDX-License-Identifier: Apache-2.0
"""Native perception embeddings and the SGLang NemotronH language decoder."""

from __future__ import annotations

import logging
from collections.abc import Iterable, Iterator

import torch
import torch.nn as nn
from sglang.srt.distributed.parallel_state import GroupCoordinator
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.managers.mm_utils import (
    MultiModalityDataPaddingPatternMultimodalTokens,
    general_mm_embed_routine,
)
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from sglang.srt.utils import add_prefix

from sglang_omni.models.nemotron_transcribe.checkpoint_contract import (
    classify_checkpoint_name,
)
from sglang_omni.models.nemotron_transcribe.decoder_weights import (
    checkpoint_to_language_model_weights,
)
from sglang_omni.models.nemotron_transcribe.hf_config import NemotronTranscribeConfig
from sglang_omni.models.nemotron_transcribe.perception import (
    AudioPerception,
    build_audio_perception,
    load_perception_weights,
)

logger = logging.getLogger(__name__)

PERCEPTION_DTYPE = torch.bfloat16


def build_perception(config: NemotronTranscribeConfig) -> AudioPerception:
    """Build the native perception module from the checkpoint config sections."""
    if not config.pe_encoder_config:
        raise ValueError("Nemotron Transcribe checkpoint has no pe_encoder_config")
    else:
        pass
    return build_audio_perception(config.perception, config.pe_encoder_config)


class NemotronTranscribeForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: NemotronTranscribeConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.audio_token_id = int(config.audio_token_id)
        self.perception = build_perception(config)
        self.language_model = NemotronHForCausalLM(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("language_model", prefix),
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        self.encoder_max_batch_size: int | None = None

    @property
    def pp_group(self) -> GroupCoordinator:
        return self.language_model.pp_group

    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    @torch.no_grad()
    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        """Encode audio items in bounded batches; output follows item order."""
        assert self.encoder_max_batch_size is not None
        outputs = []
        for start in range(0, len(items), self.encoder_max_batch_size):
            outputs.extend(
                self.encode_audio_batch(
                    items[start : start + self.encoder_max_batch_size]
                )
            )
        return torch.cat(outputs, dim=0)

    def encode_audio_batch(self, items: list[MultimodalDataItem]) -> list[torch.Tensor]:
        device = next(self.perception.parameters()).device
        waves = [
            torch.as_tensor(item.feature).reshape(-1).to(device, torch.float32)
            for item in items
        ]
        lengths = torch.tensor([w.numel() for w in waves], device=device)
        batch = torch.zeros(
            len(waves), max(wave.numel() for wave in waves), device=device
        )
        for i, wave in enumerate(waves):
            batch[i, : wave.numel()] = wave

        embs, emb_lens = self.perception(
            input_signal=batch, input_signal_length=lengths
        )

        outputs = []
        for i, item in enumerate(items):
            expected = int(item.model_specific_data["num_audio_tokens"])
            emb = embs[i, : int(emb_lens[i])].to(PERCEPTION_DTYPE)
            if emb.shape[0] != expected:
                raise RuntimeError(
                    f"Audio encoder produced {emb.shape[0]} embeddings for {expected} placeholders"
                )
            else:
                pass
            outputs.append(emb)
        return outputs

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: object,
    ) -> LogitsProcessorOutput:
        # note (yhayaran): get_audio_feature runs here only when the pre-LM encoder is disabled.
        return general_mm_embed_routine(
            input_ids=input_ids,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.AUDIO: self.get_audio_feature},
            positions=positions,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        perception_weights: dict[str, torch.Tensor] = {}

        def language_model_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in weights:
                kind, parameter_name = classify_checkpoint_name(name)
                if kind == "perception":
                    perception_weights[parameter_name] = tensor
                elif kind == "skip":
                    continue
                else:
                    yield name, tensor

        self.language_model.load_weights(
            checkpoint_to_language_model_weights(language_model_weights())
        )

        load_perception_weights(self.perception, perception_weights, PERCEPTION_DTYPE)
        logger.info(f"Loaded {len(perception_weights)} perception tensors")


EntryClass = NemotronTranscribeForConditionalGeneration
