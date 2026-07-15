# Copyright [2023-11-28] <sxc19@mails.tsinghua.edu.cn, Xingchen Song>
#            2024 Alibaba Inc (authors: Xiang Lyu)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch

from sglang_omni.models.cosyvoice3.cosyvoice.transformer.activation import Swish
from sglang_omni.models.cosyvoice3.cosyvoice.transformer.attention import (
    MultiHeadedAttention,
    RelPositionMultiHeadedAttention,
)
from sglang_omni.models.cosyvoice3.cosyvoice.transformer.embedding import (
    EspnetRelPositionalEncoding,
    LearnablePositionalEncoding,
    NoPositionalEncoding,
    PositionalEncoding,
    RelPositionalEncoding,
    WhisperPositionalEncoding,
)
from sglang_omni.models.cosyvoice3.cosyvoice.transformer.subsampling import (
    Conv1dSubsampling2,
    Conv2dSubsampling4,
    Conv2dSubsampling6,
    Conv2dSubsampling8,
    EmbedinigNoSubsampling,
    LegacyLinearNoSubsampling,
    LinearNoSubsampling,
)

# LOCAL MODIFICATION (sglang-omni): the upstream get_model_type() helper and the llm/flow/hift
# imports it relied on were removed. get_model_type() dispatched to cosyvoice.cli.model, which
# sglang-omni intentionally does not vendor — the TTS pipeline is driven by sglang_omni stages,
# not the CosyVoice cli. Only the COSYVOICE_*_CLASSES registries below are kept; they are
# consumed by the vendored flow/encoder modules via their config `!name:` tags.


COSYVOICE_ACTIVATION_CLASSES = {
    "hardtanh": torch.nn.Hardtanh,
    "tanh": torch.nn.Tanh,
    "relu": torch.nn.ReLU,
    "selu": torch.nn.SELU,
    "swish": getattr(torch.nn, "SiLU", Swish),
    "gelu": torch.nn.GELU,
}

COSYVOICE_SUBSAMPLE_CLASSES = {
    "linear": LinearNoSubsampling,
    "linear_legacy": LegacyLinearNoSubsampling,
    "embed": EmbedinigNoSubsampling,
    "conv1d2": Conv1dSubsampling2,
    "conv2d": Conv2dSubsampling4,
    "conv2d6": Conv2dSubsampling6,
    "conv2d8": Conv2dSubsampling8,
    "paraformer_dummy": torch.nn.Identity,
}

COSYVOICE_EMB_CLASSES = {
    "embed": PositionalEncoding,
    "abs_pos": PositionalEncoding,
    "rel_pos": RelPositionalEncoding,
    "rel_pos_espnet": EspnetRelPositionalEncoding,
    "no_pos": NoPositionalEncoding,
    "abs_pos_whisper": WhisperPositionalEncoding,
    "embed_learnable_pe": LearnablePositionalEncoding,
}

COSYVOICE_ATTENTION_CLASSES = {
    "selfattn": MultiHeadedAttention,
    "rel_selfattn": RelPositionMultiHeadedAttention,
}
