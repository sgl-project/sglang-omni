# SPDX-License-Identifier: Apache-2.0
"""Exact-length incremental Fish codec execution over a fixed state arena."""

from __future__ import annotations

import math
from typing import Protocol

import torch
from torch import nn
from torch.nn import functional as F

from sglang_omni.models.fishaudio_drama3.channels_last_decoder import (
    ChannelsLastDecoder,
)
from sglang_omni.models.fishaudio_drama3.stateful_state import (
    DecoderStateArena,
    StreamState,
    decode_attention,
)
from sglang_omni.models.fishaudio_drama3.stateful_waveform import (
    StatefulWaveformDecoder,
)
from sglang_omni.utils.channels_last_conv import is_channels_last_conv_device


class AttentionDecode(Protocol):
    def __call__(
        self,
        transformer: nn.Module,
        hidden_states: torch.Tensor,
        arena: DecoderStateArena,
        path: str,
        slot_index: torch.Tensor,
    ) -> torch.Tensor: ...


class StatefulDACDecoder:
    """Decode new batch-one code frames using generation-checked slot leases."""

    def __init__(
        self,
        codec: nn.Module,
        dtype: torch.dtype = torch.float32,
        *,
        capacity: int,
        gemm_upsample: bool = False,
        folded_codebooks: bool = False,
        fuse_snake_history: bool = False,
        compile_attention: bool = False,
    ) -> None:
        if capacity < 1:
            raise ValueError("State arena capacity must be positive")
        else:
            pass
        self.capacity = capacity
        self.max_frame_count = 64
        self.codec = codec
        self.dtype = dtype
        self.num_codebooks = len(codec.quantizer.quantizer.quantizers) + 1
        self.validate_module(codec.quantizer.post_module)
        self.validate_module(codec.quantizer.upsample)
        self.validate_module(codec.decoder)
        decode_modules = [
            codec.quantizer.post_module,
            codec.quantizer.upsample,
            codec.decoder,
        ]
        for vector_quantizer in (
            *codec.quantizer.semantic_quantizer.quantizers,
            *codec.quantizer.quantizer.quantizers,
        ):
            decode_modules.extend(
                (vector_quantizer.codebook, vector_quantizer.out_proj)
            )
        parameters = tuple(
            parameter for module in decode_modules for parameter in module.parameters()
        )
        if codec.training:
            raise ValueError("Stateful decoding requires an evaluation-mode codec")
        elif any(parameter.dtype != dtype for parameter in parameters):
            raise ValueError(
                "Codec parameters must already have the requested decode dtype"
            )
        else:
            pass
        self.device = parameters[0].device
        self.frame_length = math.prod(codec.quantizer.downsample_factor) * math.prod(
            codec.decoder_rates
        )
        if self.frame_length != codec.frame_length:
            raise ValueError(
                "Codec frame_length disagrees with actual upsampling rates"
            )
        else:
            pass

        self.codebook_tables: list[torch.Tensor] = []
        if folded_codebooks:
            with torch.no_grad():
                for vector_quantizer in (
                    *codec.quantizer.semantic_quantizer.quantizers,
                    *codec.quantizer.quantizer.quantizers,
                ):
                    self.codebook_tables.append(
                        vector_quantizer.out_proj(
                            vector_quantizer.codebook.weight.detach()
                            .transpose(0, 1)
                            .unsqueeze(0)
                        )[0]
                        .transpose(0, 1)
                        .contiguous()
                    )
        else:
            pass
        self.upsample_weights: dict[int, torch.Tensor] = {}
        if gemm_upsample:
            for module in codec.quantizer.upsample.modules():
                if type(module).__name__ == "CausalTransConvNet":
                    convolution = module.conv
                    if (
                        convolution.kernel_size == convolution.stride
                        and convolution.groups == 1
                    ):
                        self.upsample_weights[id(module)] = (
                            convolution.weight.detach()
                            .permute(0, 2, 1)
                            .reshape(convolution.in_channels, -1)
                            .contiguous()
                        )
                    else:
                        raise ValueError(
                            "GEMM upsampling requires nonoverlapping groups=1 convolutions"
                        )
                else:
                    pass
        else:
            pass
        self.arena = DecoderStateArena(codec, capacity, self.device, dtype)
        if (
            fuse_snake_history or compile_attention
        ) and not is_channels_last_conv_device(self.device):
            raise ValueError("Fused stateful codec execution requires a CUDA device")
        else:
            pass
        self.fuse_snake_history: bool = fuse_snake_history
        self.compile_attention: bool = compile_attention
        self.attention_decode: AttentionDecode = (
            torch.compile(
                decode_attention,
                fullgraph=True,
                dynamic=False,
                recompile_limit=128,
                isolate_recompiles=True,
                options={"emulate_precision_casts": True},
            )
            if compile_attention
            else decode_attention
        )
        self.waveform_decoder: StatefulWaveformDecoder | None = (
            StatefulWaveformDecoder(
                ChannelsLastDecoder(codec, dtype=dtype),
                self.arena,
                fuse_snake_history=fuse_snake_history,
            )
            if is_channels_last_conv_device(self.device)
            else None
        )
        self.scratch_slot_index = capacity
        self.rope_capacity = min(
            (
                module.freqs_cis.shape[0]
                for module in codec.quantizer.post_module.modules()
                if type(module).__name__ == "WindowLimitedTransformer"
            ),
            default=2**63 - 1,
        )

    def validate_module(self, module: nn.Module) -> None:
        module_name = type(module).__name__
        if module_name == "WindowLimitedTransformer":
            if not module.causal:
                raise ValueError(
                    "Stateful decoding does not support noncausal attention"
                )
            elif not isinstance(module.look_ahead_conv, nn.Identity):
                raise ValueError(
                    "Stateful decoding does not support lookahead convolutions"
                )
            elif module.config.pos_embed_type != "rope":
                raise ValueError("Stateful decoding requires RoPE attention")
            elif module.window_size is None or module.window_size < 1:
                raise ValueError("Stateful decoding requires a finite attention window")
            elif any(
                layer.attention.n_head != layer.attention.n_local_heads
                for layer in module.layers
            ):
                raise ValueError(
                    "Codec native attention requires equal query and key head counts"
                )
            else:
                pass
        elif module_name == "CausalConvNet":
            if module.conv.stride != (1,):
                raise ValueError("Decode convolutions must have stride one")
            else:
                pass
        elif module_name == "CausalTransConvNet":
            convolution = module.conv
            if (
                convolution.dilation != (1,)
                or convolution.padding != (0,)
                or convolution.output_padding != (0,)
            ):
                raise ValueError("Unsupported transposed convolution alignment")
            elif convolution.kernel_size[0] < convolution.stride[0]:
                raise ValueError("Transposed convolution kernel must cover its stride")
            else:
                pass
        elif module_name == "ResidualUnit":
            if not module.causal:
                raise ValueError(
                    "Stateful decoding does not support noncausal residual units"
                )
            else:
                self.validate_module(module.block)
        elif module_name == "ConvNeXtBlock":
            self.validate_module(module.dwconv)
        elif module_name in ("Decoder", "DecoderBlock"):
            self.validate_module(
                module.model if module_name == "Decoder" else module.block
            )
        elif isinstance(module, nn.Sequential):
            for child in module:
                self.validate_module(child)
        elif isinstance(module, (nn.Identity, nn.Tanh)) or module_name == "Snake1d":
            pass
        else:
            raise ValueError(f"Unsupported stateful codec module: {module_name}")

    def create_state(self, slot_index: int | None = None) -> StreamState:
        if slot_index is None:
            slot_index = next(
                (
                    index
                    for index in range(self.capacity)
                    if index not in self.arena.active_slots
                ),
                -1,
            )
        else:
            pass
        if not 0 <= slot_index < self.capacity:
            raise ValueError("No free production state slot within capacity")
        elif slot_index in self.arena.active_slots:
            raise ValueError("State slot already has an active lease")
        else:
            self.arena.active_slots.add(slot_index)
            return StreamState(
                decoder_identity=id(self),
                slot_index=slot_index,
                generation=self.arena.host_generations[slot_index],
            )

    def try_create_state(self) -> StreamState | None:
        if len(self.arena.active_slots) >= self.capacity:
            return None
        else:
            return self.create_state()

    def reset(self, slot_index: int) -> None:
        if not 0 <= slot_index <= self.capacity:
            raise ValueError("State slot index outside arena")
        else:
            self.arena.reset_slot(slot_index)

    def reset_slot(self, slot_index: int) -> None:
        self.reset(slot_index)

    def validate_state(self, state: StreamState) -> None:
        if state.decoder_identity != id(self):
            raise ValueError("Stream state belongs to another decoder")
        elif not 0 <= state.slot_index < self.capacity:
            raise ValueError("Stream state slot outside production capacity")
        elif (
            state.generation != self.arena.host_generations[state.slot_index]
            or state.slot_index not in self.arena.active_slots
        ):
            raise ValueError("Stream state has a stale slot lease")
        else:
            pass

    def release_state(self, state: StreamState) -> None:
        self.validate_state(state)
        self.reset(state.slot_index)

    def discard_state(self, state: StreamState) -> None:
        self.validate_state(state)
        state.is_failed = True
        for tensor in self.arena.persistent_tensors():
            tensor[state.slot_index].zero_()

    def decode_module(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        slot_index: torch.Tensor,
        path: str,
    ) -> torch.Tensor:
        module_name = type(module).__name__
        if module_name == "CausalConvNet":
            convolution = module.conv
            context_length = (convolution.kernel_size[0] - 1) * convolution.dilation[0]
            if context_length:
                history = self.arena.histories[path]
                hidden_states = torch.cat(
                    (history.index_select(0, slot_index), hidden_states), dim=-1
                )
                history.index_copy_(0, slot_index, hidden_states[..., -context_length:])
            else:
                pass
            return F.conv1d(
                hidden_states,
                convolution.weight,
                convolution.bias,
                dilation=convolution.dilation,
                groups=convolution.groups,
            ).contiguous()
        elif module_name == "CausalTransConvNet":
            convolution = module.conv
            output_length = hidden_states.shape[-1] * convolution.stride[0]
            overlap_length = convolution.kernel_size[0] - convolution.stride[0]
            if id(module) in self.upsample_weights:
                waveform = (
                    torch.matmul(
                        hidden_states.transpose(1, 2),
                        self.upsample_weights[id(module)],
                    )
                    .reshape(
                        hidden_states.shape[0], output_length, convolution.out_channels
                    )
                    .transpose(1, 2)
                )
            else:
                waveform = F.conv_transpose1d(
                    hidden_states,
                    convolution.weight,
                    None,
                    stride=convolution.stride,
                    groups=convolution.groups,
                )
            if overlap_length:
                carry = self.arena.histories[path]
                waveform[..., :overlap_length] += carry.index_select(0, slot_index)
                carry.index_copy_(0, slot_index, waveform[..., output_length:])
            else:
                pass
            waveform = waveform[..., :output_length]
            if convolution.bias is not None:
                waveform = waveform + convolution.bias[None, :, None]
            else:
                pass
            return waveform.contiguous()
        elif module_name == "WindowLimitedTransformer":
            return self.attention_decode(
                module, hidden_states, self.arena, path, slot_index
            )
        elif module_name == "ResidualUnit":
            return hidden_states + self.decode_module(
                module.block, hidden_states, slot_index, f"{path}.block"
            )
        elif module_name == "ConvNeXtBlock":
            residual = hidden_states
            hidden_states = self.decode_module(
                module.dwconv, hidden_states, slot_index, f"{path}.dwconv"
            )
            hidden_states = module.pwconv2(
                module.act(module.pwconv1(module.norm(hidden_states.transpose(1, 2))))
            )
            if module.gamma is not None:
                hidden_states = hidden_states * module.gamma
            else:
                pass
            return residual + hidden_states.transpose(1, 2)
        elif module_name in ("Decoder", "DecoderBlock"):
            return self.decode_module(
                module.model if module_name == "Decoder" else module.block,
                hidden_states,
                slot_index,
                f"{path}.layers",
            )
        elif isinstance(module, nn.Sequential):
            for child_index, child in enumerate(module):
                hidden_states = self.decode_module(
                    child, hidden_states, slot_index, f"{path}.{child_index}"
                )
            return hidden_states
        else:
            return module(hidden_states)

    def validate_delta(self, codes: torch.Tensor, state: StreamState) -> int:
        self.validate_state(state)
        if state.is_failed:
            raise ValueError("Cannot decode a failed stream")
        elif state.is_final:
            raise ValueError("Cannot decode a finalized stream")
        elif (
            state.processed_frames != self.arena.host_processed_frames[state.slot_index]
        ):
            raise ValueError("Stream processed frame mirror disagrees with arena")
        elif codes.ndim != 3 or codes.shape[1] != self.num_codebooks:
            raise ValueError(f"Expected codes [batch, {self.num_codebooks}, frames]")
        elif codes.shape[0] != 1:
            raise ValueError("Stateful arena requires batch size one")
        elif codes.dtype != torch.long or codes.device != self.device:
            raise ValueError("Codes must be int64 on the codec device")
        elif codes.shape[-1] > self.max_frame_count:
            raise ValueError("Delta exceeds maximum 64 frames")
        elif state.processed_frames + codes.shape[-1] > self.rope_capacity:
            raise ValueError(
                "Stream exceeds the codec's absolute RoPE position capacity"
            )
        else:
            return codes.shape[-1]

    def commit_delta(
        self, state: StreamState, frame_count: int, is_final: bool
    ) -> None:
        state.processed_frames += frame_count
        self.arena.host_processed_frames[state.slot_index] = state.processed_frames
        state.batch_size = 1
        state.is_final = is_final

    @torch.no_grad()
    def decode_delta(
        self, codes: torch.Tensor, state: StreamState, is_final: bool = False
    ) -> torch.Tensor:
        frame_count = self.validate_delta(codes, state)
        if frame_count == 0:
            waveform = torch.empty((1, 1, 0), device=self.device, dtype=self.dtype)
        else:
            slot_index = torch.tensor(
                [state.slot_index], device=self.device, dtype=torch.int64
            )
            try:
                waveform = self.decode_slot(codes, slot_index, frame_count)
            except Exception:
                if self.device.type == "cuda":
                    torch.cuda.current_stream(self.device).synchronize()
                else:
                    pass
                self.discard_state(state)
                raise
        self.commit_delta(state, frame_count, is_final)
        return waveform

    @torch.no_grad()
    def decode_slot(
        self, codes: torch.Tensor, slot_index: torch.Tensor, frame_count: int
    ) -> torch.Tensor:
        assert codes.shape == (1, self.num_codebooks, frame_count)
        assert 1 <= frame_count <= self.max_frame_count
        assert slot_index.shape == (1,) and slot_index.dtype == torch.int64
        quantizer = self.codec.quantizer
        semantic_codes = codes[:, :1].clamp(
            max=quantizer.semantic_quantizer.codebook_size - 1
        )
        residual_codes = codes[:, 1:].clamp(max=quantizer.quantizer.codebook_size - 1)
        if self.codebook_tables:
            semantic_states = F.embedding(semantic_codes[:, 0], self.codebook_tables[0])
            residual_states = F.embedding(residual_codes[:, 0], self.codebook_tables[1])
            for codebook_index in range(1, residual_codes.shape[1]):
                residual_states = residual_states + F.embedding(
                    residual_codes[:, codebook_index],
                    self.codebook_tables[codebook_index + 1],
                )
            hidden_states = (
                (semantic_states + residual_states).transpose(1, 2).contiguous()
            )
        else:
            hidden_states = quantizer.semantic_quantizer.from_codes(semantic_codes)[0]
            hidden_states = (
                hidden_states + quantizer.quantizer.from_codes(residual_codes)[0]
            )
        hidden_states = self.decode_module(
            quantizer.post_module, hidden_states, slot_index, "post"
        )
        hidden_states = self.decode_module(
            quantizer.upsample, hidden_states, slot_index, "upsample"
        )
        if self.waveform_decoder is None:
            waveform = self.decode_module(
                self.codec.decoder, hidden_states, slot_index, "decoder"
            )
        else:
            waveform = self.waveform_decoder.decode_module(
                self.codec.decoder,
                hidden_states.transpose(1, 2).contiguous(),
                slot_index,
                "decoder",
            ).transpose(1, 2)
        assert waveform.shape == (codes.shape[0], 1, frame_count * self.frame_length)
        position = self.arena.positions.index_select(0, slot_index)
        self.arena.positions.index_copy_(0, slot_index, position + frame_count)
        return waveform
