# SPDX-License-Identifier: MIT
# Copyright (c) 2025, Prince Canuma and contributors.
# Adapted from MLX-Audio 0.4.6 Qwen3-TTS speech_tokenizer.py and config.py.
#
# MIT License
#
# Copyright (c) 2024 Prince Canuma
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Request-local KV and convolution history for incremental MLX decoding."""

import mlx.core as mx
import mlx.nn as nn
from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousAttentionKVCache

from sglang_omni.models.qwen3_tts.mlx.decoder import (
    DECODE_CHUNK_FRAMES,
    DECODE_LEFT_CONTEXT_FRAMES,
    CausalConv1d,
    CausalTransposeConv1d,
    DecoderBlock,
    Qwen3TTSMlxSpeechDecoder,
)


class Qwen3TTSMlxDecoderStream:
    """Decode new frames while preserving the offline decoder's segment boundaries."""

    def __init__(self, decoder: Qwen3TTSMlxSpeechDecoder) -> None:
        self.decoder: Qwen3TTSMlxSpeechDecoder = decoder
        self.attention_cache: list[ContiguousAttentionKVCache] = [
            ContiguousAttentionKVCache(
                max_seq_len=DECODE_CHUNK_FRAMES + DECODE_LEFT_CONTEXT_FRAMES
            )
            for _ in decoder.pre_transformer.layers
        ]
        self.convolution_history: dict[int, mx.array] = {}
        self.context_codes: mx.array | None = None
        self.decoded_frames: int = 0

    def convolve(
        self, convolution: CausalConv1d | CausalTransposeConv1d, hidden: mx.array
    ) -> mx.array:
        if isinstance(convolution, CausalConv1d):
            history_frames = convolution.padding
            stride = 1
        else:
            stride = convolution.conv.stride
            history_frames = (convolution.conv.weight.shape[1] - 1) // stride
        if self.decoded_frames == 0 or (
            isinstance(convolution, CausalConv1d) and history_frames == 0
        ):
            if history_frames > 0:
                self.convolution_history[id(convolution)] = hidden[:, -history_frames:]
            else:
                pass
            return convolution(hidden)
        else:
            if history_frames > 0:
                history = self.convolution_history.get(id(convolution))
                if history is None:
                    history = mx.zeros(
                        (hidden.shape[0], history_frames, hidden.shape[2]),
                        dtype=hidden.dtype,
                    )
                elif history.shape[1] < history_frames:
                    history = mx.pad(
                        history,
                        [(0, 0), (history_frames - history.shape[1], 0), (0, 0)],
                    )
                else:
                    pass
                hidden = mx.concatenate([history, hidden], axis=1)
                self.convolution_history[id(convolution)] = hidden[:, -history_frames:]
            else:
                pass
            if isinstance(convolution, CausalConv1d):
                return convolution(hidden, padding=0)
            else:
                output_channels = convolution.conv.weight.shape[0]
                hidden = mx.conv_general(
                    hidden,
                    convolution.phase_weight(
                        mx.result_type(hidden.dtype, convolution.conv.weight.dtype)
                    ),
                )
                return (
                    hidden.reshape(
                        hidden.shape[0], hidden.shape[1] * stride, output_channels
                    )
                    + convolution.conv.bias
                )

    def forward(self, codes: mx.array) -> mx.array:
        hidden = self.decoder.quantizer.decode(codes.transpose(0, 2, 1))
        hidden = self.convolve(self.decoder.pre_conv, hidden.transpose(0, 2, 1))
        hidden = self.decoder.pre_transformer(hidden, self.attention_cache)
        for transpose_convolution, block in self.decoder.upsample:
            hidden = self.convolve(transpose_convolution, hidden)
            residual = hidden
            hidden = block.norm(self.convolve(block.dwconv, hidden))
            hidden = block.pwconv2(nn.gelu(block.pwconv1(hidden)))
            hidden = residual + hidden * block.gamma
        for layer in self.decoder.decoder:
            if isinstance(layer, CausalConv1d):
                hidden = self.convolve(layer, hidden)
            elif isinstance(layer, DecoderBlock):
                activation, transpose_convolution, *residual_units = layer.block
                hidden = self.convolve(transpose_convolution, activation(hidden))
                for unit in residual_units:
                    residual = hidden
                    hidden = self.convolve(unit.conv1, unit.act1(hidden))
                    hidden = unit.conv2(unit.act2(hidden)) + residual
            else:
                hidden = layer(hidden)
        return mx.clip(hidden.squeeze(-1), -1.0, 1.0)

    def decode(self, codes: mx.array) -> tuple[mx.array, mx.array]:
        """Return this chunk's waveform and valid lengths without retaining its audio."""
        self.decoder.validate_codes(codes)
        waveforms: list[mx.array] = []
        start_frame = 0
        while start_frame < codes.shape[1]:
            segment_offset = self.decoded_frames % DECODE_CHUNK_FRAMES
            if segment_offset == 0 and self.context_codes is not None:
                # note (Codex): Match offline decoding's 300-frame windows and 25-frame overlap.
                for cache in self.attention_cache:
                    cache.reset()
                self.convolution_history.clear()
                self.forward(self.context_codes)
            else:
                pass
            end_frame = min(
                start_frame + DECODE_CHUNK_FRAMES - segment_offset, codes.shape[1]
            )
            chunk_codes = codes[:, start_frame:end_frame]
            waveforms.append(self.forward(chunk_codes))
            if self.context_codes is not None:
                context_codes = mx.concatenate(
                    [self.context_codes, chunk_codes], axis=1
                )
            else:
                context_codes = chunk_codes
            self.context_codes = context_codes[:, -DECODE_LEFT_CONTEXT_FRAMES:]
            self.decoded_frames += end_frame - start_frame
            start_frame = end_frame
        waveform = mx.concatenate(waveforms, axis=-1)
        lengths = (codes[..., 0] > 0).sum(axis=1) * self.decoder.decode_upsample_rate
        mx.eval(
            waveform,
            lengths,
            self.context_codes,
            *self.convolution_history.values(),
            *(array for cache in self.attention_cache for array in cache.state),
        )
        return waveform, lengths
