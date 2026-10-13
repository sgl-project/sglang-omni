# SPDX-License-Identifier: Apache-2.0
"""FireRed Conformer/Adapter with SGLang's native Qwen2 execution."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
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
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.utils import add_prefix
from torch import nn
from transformers import PretrainedConfig

from sglang_omni.models.firered_asr2.audio import FireRedAdapter, num_audio_tokens
from sglang_omni.models.firered_asr2.conformer import ConformerEncoder


class FireRedASR2ForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        # note (cocoa00): Preserve FP32 acoustic weights before loading to avoid BF16 rounding.
        self.encoder = ConformerEncoder(**config.audio_encoder_conf).float()
        self.encoder_projector = FireRedAdapter(
            self.encoder.odim,
            config.hidden_size,
            config.encoder_downsample_rate,
        ).float()
        self.language_model = Qwen2ForCausalLM(
            config, quant_config, prefix=add_prefix("language_model", prefix)
        )
        self.pattern = MultiModalityDataPaddingPatternMultimodalTokens()
        self.encoder_max_batch_size = 4

    def pad_input_ids(
        self, input_ids: list[int], mm_inputs: MultimodalInputs
    ) -> list[int]:
        return self.pattern.pad_input_tokens(input_ids, mm_inputs)

    @torch.no_grad()
    def get_audio_feature(self, audio_items: list[MultimodalDataItem]) -> torch.Tensor:
        if not audio_items:
            raise ValueError("FireRedASR2 received an empty audio batch")
        else:
            pass
        encoder_parameter = next(self.encoder.parameters())
        outputs = []
        for start in range(0, len(audio_items), self.encoder_max_batch_size):
            audio_batch = audio_items[start : start + self.encoder_max_batch_size]
            audio_features = [item.feature.squeeze(0) for item in audio_batch]
            lengths = torch.tensor(
                [x.shape[0] for x in audio_features], device=encoder_parameter.device
            )
            if any(x.ndim != 2 or x.shape[1] != 80 for x in audio_features):
                raise ValueError("FireRedASR2 features must have shape [frames, 80]")
            else:
                pass
            padded = nn.utils.rnn.pad_sequence(audio_features, batch_first=True).to(
                device=encoder_parameter.device, dtype=encoder_parameter.dtype
            )
            encoded, encoder_lengths, _ = self.encoder(padded, lengths)
            projected, _ = self.encoder_projector(encoded, encoder_lengths)
            for index, feature in enumerate(audio_features):
                audio_token_count = num_audio_tokens(
                    feature.shape[0], self.encoder_projector.ds
                )
                declared_audio_token_count = audio_batch[index].num_audio_tokens
                if audio_token_count != declared_audio_token_count:
                    raise ValueError(
                        f"Audio token length mismatch: {declared_audio_token_count} != {audio_token_count}"
                    )
                else:
                    pass
                outputs.append(projected[index, :audio_token_count])
        return torch.cat(outputs, dim=0).to(
            dtype=self.language_model.get_input_embeddings().weight.dtype
        )

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        **kwargs: torch.Tensor | None,
    ) -> LogitsProcessorOutput:
        return general_mm_embed_routine(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            language_model=self.language_model,
            data_embedding_funcs={Modality.AUDIO: self.get_audio_feature},
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        """Load the merged HF checkpoint, keeping official audio parameter names."""
        audio_tensors = {
            name: tensor
            for name, tensor in self.state_dict(keep_vars=True).items()
            if name.startswith(("encoder.", "encoder_projector."))
        }
        loaded = set()

        def language_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if name.startswith(("encoder.", "encoder_projector.")):
                    if name not in audio_tensors:
                        raise ValueError(f"Unexpected FireRed audio weight: {name}")
                    else:
                        pass
                    target = audio_tensors[name]
                    if target.shape != weight.shape:
                        raise ValueError(
                            f"Weight shape mismatch for {name}: {weight.shape} != {target.shape}"
                        )
                    else:
                        pass
                    with torch.no_grad():
                        target.copy_(weight)
                    loaded.add(name)
                elif name.startswith("llm."):
                    yield name.removeprefix("llm."), weight
                elif name.startswith("language_model."):
                    yield name.removeprefix("language_model."), weight
                else:
                    raise ValueError(f"Unexpected FireRed checkpoint weight: {name}")

        self.language_model.load_weights(language_weights())
        # note (cocoa00): Regenerating positional encodings under BF16 changes their values.
        missing = audio_tensors.keys() - loaded
        if missing:
            raise ValueError(f"Missing FireRed audio weights: {sorted(missing)[:10]}")
        else:
            pass


EntryClass = FireRedASR2ForConditionalGeneration
