# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni D2 release loading: BF16 Qwen3-Omni modules with every released adapter merged."""

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors.torch import load_file
from torch import nn
from transformers import (
    AutoTokenizer,
    PreTrainedTokenizerBase,
    Qwen3OmniMoeForConditionalGeneration,
    WhisperFeatureExtractor,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeAudioEncoder,
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeTalkerForConditionalGeneration,
    Qwen3OmniMoeThinkerTextModel,
)

from sglang_omni.models.qwen3_omni_d2.config import NATIVE_FRAME_MS
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.hf import qwen_d2_interval_ms

logger = logging.getLogger(__name__)

RELEASE_FILES = ["d2.json", "d2.safetensors", "encoder.safetensors"]
DUPLEX_PAD_TOKEN = "<|duplex_pad|>"
# Every D2 adapter, including the rank-32 audio encoder ones, uses alpha = 2 * rank.
ADAPTER_SCALING = 2.0
ADAPTER_FACTOR_PAIRS = (("lora_a", "lora_b"), ("fresh_lora_a", "fresh_lora_b"))
ADAPTER_SUFFIXES = tuple(
    f".{factor}.weight" for pair in ADAPTER_FACTOR_PAIRS for factor in pair
)
DUPLEX_ROW_NAMES = {
    "stream_type_embedding.weight",
    "self_silence_embed",
    "text_control_rows.input_rows.weight",
    "text_control_rows.output_rows.weight",
    "audio_control_rows.input_rows.weight",
    "audio_control_rows.output_rows.weight",
}
CHELSIE_SPEAKER = "chelsie"


@dataclass(frozen=True, kw_only=True)
class Qwen3OmniD2Release:
    directory: Path
    base_directory: Path
    interval_frames: int


def resolve_release(model_path: str) -> Qwen3OmniD2Release:
    """Find the D2 release files and the Qwen3-Omni revision its manifest pins."""
    directory = Path(resolve_checkpoint(model_path, allow_patterns=RELEASE_FILES))
    interval_ms = qwen_d2_interval_ms(str(directory))
    if interval_ms is None:
        raise ValueError("Expected a Qwen3-Omni D2 release with a d2.json manifest")
    else:
        base_model = json.loads((directory / "d2.json").read_text())["base_model"]
    base_directory = resolve_checkpoint(
        f"{base_model['repo_id']}@{base_model['revision']}"
    )
    return Qwen3OmniD2Release(
        directory=directory,
        base_directory=Path(base_directory),
        interval_frames=interval_ms // NATIVE_FRAME_MS,
    )


def normalize_release_names(
    encoder_tensors: dict[str, torch.Tensor], d2_tensors: dict[str, torch.Tensor]
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    """Map both release files onto Qwen3-Omni parameter names; d2.safetensors wins on overlap.

    Returns the Qwen3-Omni tensors and the duplex rows that live outside the vocabulary.
    """
    qwen_tensors = {
        name.replace("audio_tower.", "thinker.audio_tower.", 1): tensor
        for name, tensor in encoder_tensors.items()
    }
    duplex_rows: dict[str, torch.Tensor] = {}
    for name, tensor in d2_tensors.items():
        if name in DUPLEX_ROW_NAMES:
            duplex_rows[name] = tensor
        elif name.startswith("qwen.thinker.audio_tower.audio_tower."):
            qwen_tensors[
                name.replace(
                    "qwen.thinker.audio_tower.audio_tower.", "thinker.audio_tower.", 1
                )
            ] = tensor
        elif name.startswith("qwen."):
            qwen_tensors[name.removeprefix("qwen.")] = tensor
        else:
            raise ValueError(f"Unexpected D2 release tensor {name}")
    if set(duplex_rows) != DUPLEX_ROW_NAMES:
        raise ValueError(
            f"D2 release is missing {sorted(DUPLEX_ROW_NAMES - set(duplex_rows))}"
        )
    else:
        return qwen_tensors, duplex_rows


@torch.no_grad()
def merge_release_tensors(
    model: nn.Module, qwen_tensors: dict[str, torch.Tensor]
) -> None:
    """Copy released parameters into BF16 storage, then add each adapter update.

    An update is computed in FP32 from BF16-rounded factors and rounded once
    before the BF16 add, the same arithmetic as the release's inference merge.
    """
    adapters: dict[str, dict[str, torch.Tensor]] = {}
    for name, tensor in qwen_tensors.items():
        if name.endswith(ADAPTER_SUFFIXES):
            module_path, factor_name, _ = name.rsplit(".", 2)
            adapters.setdefault(module_path, {})[factor_name] = tensor
        else:
            parameter = model.get_parameter(name.replace(".base.", "."))
            if parameter.shape != tensor.shape:
                raise ValueError(
                    f"D2 tensor {name} has shape {tuple(tensor.shape)}, expected {tuple(parameter.shape)}"
                )
            else:
                parameter.copy_(tensor)
    for module_path, factors in adapters.items():
        weight = model.get_parameter(f"{module_path}.weight")
        for down_name, up_name in ADAPTER_FACTOR_PAIRS:
            if down_name in factors and up_name in factors:
                down = factors[down_name].to(weight.device, torch.bfloat16).float()
                up = factors[up_name].to(weight.device, torch.bfloat16).float()
                weight.add_((up @ down * ADAPTER_SCALING).to(weight.dtype))
            elif down_name in factors or up_name in factors:
                raise ValueError(
                    f"Adapter {module_path} has only one {down_name} factor"
                )
            else:
                pass


class Qwen3OmniD2Model:
    """The released modules on one device and the rows the Thinker keeps beside its vocabulary."""

    def __init__(
        self,
        qwen: Qwen3OmniMoeForConditionalGeneration,
        duplex_rows: dict[str, torch.Tensor],
        tokenizer: PreTrainedTokenizerBase,
        feature_extractor: WhisperFeatureExtractor,
        interval_frames: int,
    ) -> None:
        self.thinker: Qwen3OmniMoeThinkerTextModel = qwen.thinker.model
        self.lm_head: nn.Linear = qwen.thinker.lm_head
        self.audio_tower: Qwen3OmniMoeAudioEncoder = qwen.thinker.audio_tower
        self.talker: Qwen3OmniMoeTalkerForConditionalGeneration = qwen.talker
        self.code2wav: Qwen3OmniMoeCode2Wav = qwen.code2wav
        self.tokenizer = tokenizer
        self.interval_frames = interval_frames
        self.device = self.lm_head.weight.device
        self.mel_filters = torch.from_numpy(feature_extractor.mel_filters).float()
        self.fft_window_samples = int(feature_extractor.n_fft)
        self.hop_samples = int(feature_extractor.hop_length)
        thinker_config = qwen.config.thinker_config
        talker_config = qwen.config.talker_config
        self.response_token_id = int(thinker_config.audio_start_token_id)
        self.interrupt_token_id = int(thinker_config.audio_end_token_id)
        tokenizer.add_special_tokens({"additional_special_tokens": [DUPLEX_PAD_TOKEN]})
        self.pad_token_id = int(tokenizer.convert_tokens_to_ids(DUPLEX_PAD_TOKEN))
        self.text_vocabulary_size = len(tokenizer)
        self.special_token_ids = frozenset(tokenizer.all_special_ids) | {
            self.pad_token_id,
            self.response_token_id,
            self.interrupt_token_id,
        }
        vocabulary_size = self.lm_head.out_features
        self.invalid_vocabulary_mask = torch.zeros(
            vocabulary_size, dtype=torch.bool, device=self.device
        )
        self.invalid_vocabulary_mask[
            sorted(
                self.special_token_ids
                - {self.pad_token_id, self.response_token_id, self.interrupt_token_id}
            )
        ] = True
        self.invalid_vocabulary_mask[self.text_vocabulary_size :] = True
        self.tts_pad_token_id = int(qwen.config.tts_pad_token_id)
        self.speaker_id = int(talker_config.speaker_id[CHELSIE_SPEAKER])
        self.codec_bos_id = int(talker_config.codec_bos_id)
        self.codec_eos_id = int(talker_config.codec_eos_token_id)
        self.codec_vocabulary_size = int(talker_config.text_config.vocab_size)
        self.accept_hidden_layer = int(talker_config.accept_hidden_layer)
        rows = {
            name: tensor.to(self.device, torch.bfloat16)
            for name, tensor in duplex_rows.items()
        }
        self.self_stream_row, self.environment_stream_row = rows[
            "stream_type_embedding.weight"
        ]
        self.self_silence_row = rows["self_silence_embed"]
        control_ids = [self.response_token_id, self.interrupt_token_id]
        embedding = self.thinker.embed_tokens.weight
        official_input_rows = embedding[control_ids].clone()
        # note (cuzimi): Folded rows keep every lookup equal to the released embed path: PAD is replaced, RESPONSE/INTERRUPT add a BF16 delta.
        embedding[self.pad_token_id] = rows["text_control_rows.input_rows.weight"][0]
        embedding[control_ids] = official_input_rows + (
            rows["audio_control_rows.input_rows.weight"] - official_input_rows
        )
        self.pad_output_row = rows["text_control_rows.output_rows.weight"]
        self.control_output_delta = (
            rows["audio_control_rows.output_rows.weight"]
            - self.lm_head.weight[control_ids]
        )
        self.control_token_index = torch.tensor([control_ids], device=self.device)
        self.pad_token_index = torch.tensor([[self.pad_token_id]], device=self.device)


def load_release_model(
    release: Qwen3OmniD2Release, device: torch.device
) -> Qwen3OmniD2Model:
    logger.info(f"Loading Qwen3-Omni D2 base model from {release.base_directory}")
    qwen = Qwen3OmniMoeForConditionalGeneration.from_pretrained(
        release.base_directory,
        dtype=torch.bfloat16,
        device_map={"": str(device)},
        attn_implementation="sdpa",
        experts_implementation="grouped_mm",
        local_files_only=True,
    )
    qwen.thinker.visual = None
    # note (cuzimi): The released runtime casts the whole model, rotary buffers included, to BF16.
    qwen.eval().requires_grad_(False).to(torch.bfloat16)
    qwen_tensors, duplex_rows = normalize_release_names(
        load_file(release.directory / "encoder.safetensors"),
        load_file(release.directory / "d2.safetensors"),
    )
    merge_release_tensors(qwen, qwen_tensors)
    logger.info(f"Merged {len(qwen_tensors)} D2 release tensors")
    return Qwen3OmniD2Model(
        qwen,
        duplex_rows,
        AutoTokenizer.from_pretrained(release.base_directory),
        WhisperFeatureExtractor.from_pretrained(release.base_directory),
        release.interval_frames,
    )
