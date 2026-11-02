# SPDX-License-Identifier: Apache-2.0
"""Hy-SimulMT audio embeddings with the Hunyuan V3 decoder."""

from collections.abc import Iterable, Iterator

import torch
from sglang.srt.configs.qwen3_omni import Qwen3OmniMoeAudioEncoderConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.managers.mm_utils import prepare_mm_inputs
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.hunyuan_v3 import HYV3ForCausalLM
from sglang.srt.models.qwen3_omni_moe import Qwen3OmniMoeAudioEncoder
from torch import nn
from torch.nn import functional as F
from transformers import HYV3Config


class FrameStackAdapter(nn.Module):
    """Project adjacent encoder frames into one language-model embedding."""

    def __init__(self, encoder_dim: int, llm_dim: int, fsk_size: int) -> None:
        super().__init__()
        self.stack_size = fsk_size
        self.layers = nn.Sequential(
            nn.Linear(encoder_dim * fsk_size, llm_dim),
            nn.SiLU(),
            nn.Dropout(0.0),
            nn.Linear(llm_dim, llm_dim),
            nn.LayerNorm(llm_dim, eps=1e-5),
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # note (JingwenGu0829): VisionAttention can retain a singleton batch dimension for this clip.
        hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
        padding_frames = -hidden_states.shape[0] % self.stack_size
        hidden_states = F.pad(hidden_states, (0, 0, 0, padding_frames))
        stacked_frames = hidden_states.reshape(
            -1, hidden_states.shape[-1] * self.stack_size
        )
        return self.layers(stacked_frames)


class HySimulMTAudioTower(nn.Module):
    def __init__(self, config: HYV3Config) -> None:
        super().__init__()
        audio_config = config.audio_config
        self.speech_encoder = Qwen3OmniMoeAudioEncoder(
            Qwen3OmniMoeAudioEncoderConfig(
                **{**audio_config, "attn_implementation": "eager"}
            )
        )
        self.speech_adapter = FrameStackAdapter(
            encoder_dim=audio_config["encoder_dim"],
            llm_dim=audio_config["llm_dim"],
            fsk_size=audio_config["fsk_size"],
        )

    def forward(self, mel: torch.Tensor) -> torch.Tensor:
        weight = self.speech_encoder.conv2d1.weight
        mel = mel.to(device=weight.device, dtype=weight.dtype)
        feature_lengths = torch.tensor([mel.shape[-1]], device=weight.device)
        encoded = self.speech_encoder(
            mel, feature_lens=feature_lengths
        ).last_hidden_state
        return self.speech_adapter(encoded)


class HYV3SpeechForCausalLM(HYV3ForCausalLM):
    def __init__(
        self,
        config: HYV3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        if quant_config is not None:
            raise ValueError("Hy-SimulMT currently requires unquantized weights")
        else:
            pass
        super().__init__(config, quant_config, prefix)
        self.audio_tower = HySimulMTAudioTower(config)

    def get_input_embeddings(self) -> VocabParallelEmbedding:
        return self.model.embed_tokens

    def get_audio_feature(self, items: list[MultimodalDataItem]) -> torch.Tensor:
        # note (JingwenGu0829): Stack each clip separately so odd tails never pair with the next clip.
        return torch.cat([self.audio_tower(item.feature) for item in items])

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> LogitsProcessorOutput:
        input_embeddings = prepare_mm_inputs(
            input_ids,
            forward_batch,
            self,
            {},
            data_embedding_funcs={Modality.AUDIO: self.get_audio_feature},
        )
        return super().forward(input_ids, positions, forward_batch, input_embeddings)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        audio_parameters = dict(self.audio_tower.named_parameters())
        expected_names: set[str] = set()
        for name in audio_parameters:
            if ".qkv_proj." in name:
                expected_names.update(
                    name.replace(".qkv_proj.", f".{projection}_proj.")
                    for projection in ("q", "k", "v")
                )
            else:
                expected_names.add(
                    name.replace(".self_attn.proj.", ".self_attn.out_proj.")
                )
        loaded_names: set[str] = set()

        def language_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, weight in weights:
                if not name.startswith("audio_tower."):
                    yield name, weight
                else:
                    audio_name = name.removeprefix("audio_tower.")
                    if audio_name not in expected_names or audio_name in loaded_names:
                        raise ValueError(
                            f"Unexpected or duplicate audio weight: {name}"
                        )
                    else:
                        loaded_names.add(audio_name)
                    mapped_name = audio_name.replace(
                        ".self_attn.out_proj.", ".self_attn.proj."
                    )
                    for projection in ("q", "k", "v"):
                        if f".{projection}_proj." in mapped_name:
                            mapped_name = mapped_name.replace(
                                f".{projection}_proj.", ".qkv_proj."
                            )
                            parameter = audio_parameters[mapped_name]
                            parameter.weight_loader(parameter, weight, projection)
                            break
                        else:
                            pass
                    else:
                        parameter = audio_parameters[mapped_name]
                        loader = getattr(
                            parameter, "weight_loader", default_weight_loader
                        )
                        loader(parameter, weight)

        super().load_weights(language_weights())
        missing_names = expected_names - loaded_names
        if missing_names:
            raise ValueError(f"Missing audio weights: {sorted(missing_names)}")
        else:
            pass
