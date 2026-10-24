# Copyright (c)  (Mddct: Dinghao Zhou)
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
# Copyright (c) 2023 OpenAI. (authors: Whisper Team)
#               2024 Tsinghua Univ. (authors: Xingchen Song)
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
# Modifications: retain MiniCPM-o inference only; local imports and typing.
"""Speech tokenizer for MiniCPM-o."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.components.token2wav.speech_tokenizer_model import (
    AudioEncoderV2,
    FSQVectorQuantization,
    ModelConfig,
)
from sglang_omni.models.minicpm_o.components.token2wav.speech_tokenizer_weights import (
    load_tokenizer_weights,
)

WINDOW_FRAMES = 3000
OVERLAP_FRAMES = 400
HALF_OVERLAP_TOKENS = 50


class S3TokenizerV2(torch.nn.Module):
    """Encode one speaker reference, batching overlapping windows above 30 seconds."""

    def __init__(self, checkpoint: Path, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = ModelConfig() if config is None else config
        self.encoder = AudioEncoderV2(
            self.config.n_mels,
            self.config.n_audio_state,
            self.config.n_audio_head,
            self.config.n_audio_layer,
            2,
            self.config.use_sdpa,
        )
        self.quantizer = FSQVectorQuantization(
            self.config.n_audio_state, self.config.n_codebook_size
        )
        self.load_state_dict(load_tokenizer_weights(checkpoint), strict=True)

    @torch.inference_mode()
    def forward(
        self, mel: torch.Tensor, mel_len: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert mel.shape[0] == mel_len.numel() == 1
        frames = int(mel_len.item())
        if frames <= WINDOW_FRAMES:
            hidden, code_len = self.encoder(mel, mel_len)
            code = self.quantizer.encode(hidden)
            return (code, code_len)
        else:
            pass

        segments = []
        lengths = []
        for start in range(0, frames, WINDOW_FRAMES - OVERLAP_FRAMES):
            segment = mel[0, :, start : min(start + WINDOW_FRAMES, frames)]
            lengths.append(segment.shape[1])
            segments.append(F.pad(segment, (0, WINDOW_FRAMES - segment.shape[1])))
        hidden, code_len = self.encoder(
            torch.stack(segments),
            torch.tensor(lengths, dtype=torch.int64, device=mel.device),
        )
        codes = self.quantizer.encode(hidden)
        merged = []
        for index, length in enumerate(code_len.tolist()):
            segment = codes[index, :length]
            start = 0 if index == 0 else HALF_OVERLAP_TOKENS
            stop = -HALF_OVERLAP_TOKENS if index < len(segments) - 1 else length
            merged.append(segment[start:stop])
        tokens = torch.cat(merged).to(torch.int64).unsqueeze(0)
        return tokens, torch.tensor(
            [tokens.shape[1]], dtype=torch.int64, device=mel.device
        )
