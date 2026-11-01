# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni code2wav whose convolutions run channels last and whose pre-transformer runs on
fused kernels, and its incremental decode from a codec state pool."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from sglang.kernels.ops.activation import silu_and_mul
from sglang.kernels.ops.attention.flash_attention import flash_attn_with_kvcache
from sglang.kernels.ops.attention.flash_attention_v3 import _is_fa3_supported
from sglang.kernels.ops.attention.rope import apply_rope_inplace_with_kvcache
from sglang.kernels.ops.layernorm import rmsnorm
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeCode2WavConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCausalConvNet,
    Qwen3OmniMoeCausalTransConvNet,
    Qwen3OmniMoeCode2Wav,
    Qwen3OmniMoeCode2WavDecoderBlock,
    Qwen3OmniMoeCode2WavDecoderResidualUnit,
    Qwen3OmniMoeCode2WavTransformerLayer,
    Qwen3OmniMoeCode2WavTransformerModel,
    Qwen3OmniMoeConvNeXtBlock,
    Qwen3OmniMoeSnakeBeta,
)

from sglang_omni.kernels.bias_snake_beta import SnakeBetaCoefficients
from sglang_omni.kernels.single_output_conv import clamped_single_output_conv
from sglang_omni.models.qwen3_omni.components.codec_state_pool import (
    CodecStatePool,
    CodecStateSpec,
    RingAddresses,
    ring_addresses,
)
from sglang_omni.models.qwen3_omni.components.incremental_codec import (
    SlotDecode,
    conv_stack_state_layout,
    slot_conv_stack,
)
from sglang_omni.platforms.interface import JointRopeInplaceKernel
from sglang_omni.utils.channels_last_conv import (
    channels_last_conv1d,
    channels_last_conv_transpose1d,
    channels_last_weight,
)
from sglang_omni.utils.merged_projections import merge_projection_weights
from sglang_omni.utils.snake_beta import FusedSnakeBeta


def causal_conv(
    module: Qwen3OmniMoeCausalConvNet, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Causal conv of a (B, L, C) activation, returned as (B, L, C_out)."""
    conv = module.conv
    assert conv.stride == (1,), "code2wav causal convs are stride 1"
    length = hidden_states.shape[1]
    padded = F.pad(hidden_states, (0, 0, module.padding, (-length) % conv.dilation[0]))
    return channels_last_conv1d(padded, conv, conv.weight, conv.bias, length)


def causal_transconv(
    module: Qwen3OmniMoeCausalTransConvNet, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Causal transposed conv of a (B, L, C) activation, returned as (B, L_out, C_out)."""
    conv = module.conv
    output = channels_last_conv_transpose1d(hidden_states, conv, conv.weight, conv.bias)
    return output[:, module.left_pad : output.shape[1] - module.right_pad].contiguous()


def channels_last_block(
    module: torch.nn.Module, hidden_states: torch.Tensor
) -> torch.Tensor:
    """Run one code2wav module on a (B, L, C) activation as its forward runs on (B, C, L)."""
    if isinstance(module, Qwen3OmniMoeCausalConvNet):
        return causal_conv(module, hidden_states)
    elif isinstance(module, Qwen3OmniMoeCausalTransConvNet):
        return causal_transconv(module, hidden_states)
    elif isinstance(module, (Qwen3OmniMoeSnakeBeta, FusedSnakeBeta)):
        return module(hidden_states.transpose(1, 2)).transpose(1, 2)
    elif isinstance(module, Qwen3OmniMoeConvNeXtBlock):
        # note (ratish): the depthwise conv stays on PyTorch's own (B, C, L) kernel,
        # faster than cuDNN's channels-last depthwise and free of its per-shape setup.
        normed = module.norm(
            module.dwconv(hidden_states.transpose(1, 2)).transpose(1, 2)
        )
        return hidden_states + module.gamma * module.pwconv2(
            module.act(module.pwconv1(normed))
        )
    elif isinstance(module, Qwen3OmniMoeCode2WavDecoderResidualUnit):
        output = hidden_states
        for block in (module.act1, module.conv1, module.act2, module.conv2):
            output = channels_last_block(block, output)
        return output + hidden_states
    elif isinstance(module, Qwen3OmniMoeCode2WavDecoderBlock):
        for block in module.block:
            hidden_states = channels_last_block(block, hidden_states)
        return hidden_states
    else:
        raise TypeError(
            f"no channels-last form for code2wav module {type(module).__name__}"
        )


class FusedCode2WavTransformer(torch.nn.Module):
    """Code2wav's pre-transformer on its own weights with one qkv and one gate-up GEMM per layer,
    and RMSNorm, rotary, SwiGLU and layer scale plus residual one kernel each."""

    def __init__(
        self,
        transformer: Qwen3OmniMoeCode2WavTransformerModel,
        rope: JointRopeInplaceKernel,
        max_positions: int,
    ) -> None:
        super().__init__()
        rotary = transformer.rotary_emb
        assert not transformer.config.attention_bias, "code2wav attention has no bias"
        assert transformer.config.hidden_act == "silu", "code2wav's MLP is SwiGLU"
        assert rotary.rope_type == "default", "the cos/sin table is built once"
        self.transformer = transformer
        self.rope = rope
        positions = torch.arange(
            max_positions,
            device=rotary.inv_freq.device,
            dtype=torch.float32,
        )
        angles = torch.outer(positions, rotary.inv_freq.float())
        self.register_buffer(
            "cos_sin_cache",
            torch.cat((angles.cos(), angles.sin()), dim=-1) * rotary.attention_scaling,
            persistent=False,
        )
        merge_projection_weights(transformer.layers)

    def forward(self, inputs_embeds: torch.Tensor) -> BaseModelOutputWithPast:
        batch_size, length, hidden_size = inputs_embeds.shape
        head_dim = self.transformer.layers[0].self_attn.head_dim
        sliding_window = self.transformer.config.sliding_window
        positions = torch.arange(length, device=inputs_embeds.device).repeat(batch_size)
        if length <= sliding_window:
            window_mask = None
        else:
            query = torch.arange(length, device=inputs_embeds.device)[:, None]
            key = torch.arange(length, device=inputs_embeds.device)[None, :]
            window_mask = (key <= query) & (query - key < sliding_window)
        hidden_states = inputs_embeds.reshape(batch_size * length, hidden_size)
        for layer in self.transformer.layers:
            query, key, value = self.projected_qkv(layer, hidden_states)
            self.rope(query, key, self.cos_sin_cache, positions, is_neox=True)
            attended = self.attend(
                layer,
                query.view(batch_size, length, -1, head_dim),
                key.view(batch_size, length, -1, head_dim),
                value.view(batch_size, length, -1, head_dim),
                window_mask,
            )
            hidden_states = self.layer_output(layer, hidden_states, attended)
        hidden_states = rmsnorm(
            hidden_states,
            self.transformer.norm.weight,
            self.transformer.norm.variance_epsilon,
        )
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states.view(batch_size, length, hidden_size)
        )

    def forward_slot_frames(
        self,
        inputs_embeds: torch.Tensor,
        pool: CodecStatePool,
        addresses: RingAddresses,
        *,
        first_decode: bool,
        is_paged_attention: bool,
    ) -> torch.Tensor:
        """The normed hidden states of the rows' new frames. Every layer's rope kernel stores
        its new K/V in the rows' rings; a first decode attends to its own frames and a later
        one to its ring window, through the page table when is_paged_attention."""
        batch_size, length, hidden_size = inputs_embeds.shape
        head_dim = self.transformer.layers[0].self_attn.head_dim
        sliding_window = self.transformer.config.sliding_window
        if first_decode and length <= sliding_window:
            window_mask = None
        elif first_decode:
            frame_offsets = torch.arange(length, device=inputs_embeds.device)
            distances = frame_offsets[:, None] - frame_offsets[None, :]
            window_mask = (distances >= 0) & (distances < sliding_window)
        elif is_paged_attention:
            window_mask = None
            page_table = addresses.window_rows.to(torch.int32)
            window_lengths = addresses.window_lengths.to(torch.int32)
        else:
            # note (ratish): window rows run in position order, a row's new frames last.
            frame_offsets = torch.arange(length, device=inputs_embeds.device)
            query_offsets = (addresses.window_lengths - length)[:, None] + frame_offsets
            distances = query_offsets[:, :, None] - torch.arange(
                pool.ring_frames, device=inputs_embeds.device
            )
            window_mask = ((distances >= 0) & (distances < sliding_window))[:, None]
        hidden_states = inputs_embeds.reshape(batch_size * length, hidden_size)
        for layer_index, layer in enumerate(self.transformer.layers):
            query, key, value = self.projected_qkv(layer, hidden_states)
            apply_rope_inplace_with_kvcache(
                query,
                key,
                value.view(batch_size * length, -1, head_dim),
                pool.keys[layer_index],
                pool.values[layer_index],
                self.cos_sin_cache,
                addresses.frame_positions,
                addresses.write_rows,
                is_neox=True,
            )
            if first_decode or not is_paged_attention:
                if first_decode:
                    keys = key.view(batch_size, length, -1, head_dim)
                    values = value.view(batch_size, length, -1, head_dim)
                else:
                    keys = pool.keys[layer_index][addresses.window_rows].view(
                        batch_size, pool.ring_frames, -1, head_dim
                    )
                    values = pool.values[layer_index][addresses.window_rows].view(
                        batch_size, pool.ring_frames, -1, head_dim
                    )
                attended = self.attend(
                    layer,
                    query.view(batch_size, length, -1, head_dim),
                    keys,
                    values,
                    window_mask,
                )
            else:
                attended = flash_attn_with_kvcache(
                    query.view(batch_size, length, -1, head_dim),
                    pool.keys[layer_index].view(-1, 1, key.shape[1], head_dim),
                    pool.values[layer_index].view(-1, 1, key.shape[1], head_dim),
                    page_table=page_table,
                    cache_seqlens=window_lengths,
                    softmax_scale=layer.self_attn.scaling,
                    causal=True,
                    window_size=(pool.retained_frames, 0),
                ).reshape(batch_size * length, -1)
            hidden_states = self.layer_output(layer, hidden_states, attended)
        hidden_states = rmsnorm(
            hidden_states,
            self.transformer.norm.weight,
            self.transformer.norm.variance_epsilon,
        )
        return hidden_states.view(batch_size, length, hidden_size)

    def projected_qkv(
        self,
        layer: Qwen3OmniMoeCode2WavTransformerLayer,
        hidden_states: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """The layer's (tokens, heads, head_dim) query and key before rotary, and its
        (tokens, kv_heads * head_dim) value, from one qkv GEMM."""
        attention = layer.self_attn
        normed = rmsnorm(
            hidden_states,
            layer.input_layernorm.weight,
            layer.input_layernorm.variance_epsilon,
        )
        query, key, value = F.linear(normed, attention.qkv_weight).split(
            [
                attention.q_proj.out_features,
                attention.k_proj.out_features,
                attention.v_proj.out_features,
            ],
            dim=-1,
        )
        query = query.view(hidden_states.shape[0], -1, attention.head_dim)
        key = key.view(hidden_states.shape[0], -1, attention.head_dim)
        return query, key, value

    def attend(
        self,
        layer: Qwen3OmniMoeCode2WavTransformerLayer,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        window_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """SDPA of (B, L, heads, head_dim) queries over (B, S, kv_heads, head_dim) keys and
        values, causal without a mask, as (B * L, heads * head_dim)."""
        attention = layer.self_attn
        attended = F.scaled_dot_product_attention(
            query.transpose(1, 2),
            keys.transpose(1, 2),
            values.transpose(1, 2),
            attn_mask=window_mask,
            is_causal=window_mask is None,
            scale=attention.scaling,
            enable_gqa=attention.num_key_value_groups > 1,
        )
        return attended.transpose(1, 2).reshape(query.shape[0] * query.shape[1], -1)

    def layer_output(
        self,
        layer: Qwen3OmniMoeCode2WavTransformerLayer,
        residual: torch.Tensor,
        attended: torch.Tensor,
    ) -> torch.Tensor:
        """The layer's output: its attention projection, then its SwiGLU MLP, each added to the
        residual by the layer scale in one kernel."""
        hidden_states = torch.addcmul(
            residual,
            layer.self_attn_layer_scale.scale,
            F.linear(attended, layer.self_attn.o_proj.weight),
        )
        normed = rmsnorm(
            hidden_states,
            layer.post_attention_layernorm.weight,
            layer.post_attention_layernorm.variance_epsilon,
        )
        activated = silu_and_mul(F.linear(normed, layer.mlp.gate_up_weight))
        return torch.addcmul(
            hidden_states,
            layer.mlp_layer_scale.scale,
            F.linear(activated, layer.mlp.down_proj.weight),
        )


class Qwen3OmniCode2Wav(Qwen3OmniMoeCode2Wav):
    """Code2Wav that carries (B, L, C) activations once its convs are channels last."""

    def __init__(self, config: Qwen3OmniMoeCode2WavConfig) -> None:
        super().__init__(config)
        self.is_channels_last = False

    def use_channels_last(self) -> None:
        """Store every conv weight channels last and run the channels-last forward."""
        for module in self.modules():
            if isinstance(module, (torch.nn.Conv1d, torch.nn.ConvTranspose1d)):
                module.weight.data = channels_last_weight(module)
            else:
                pass
        self.is_channels_last = True

    def use_fused_transformer(
        self, rope: JointRopeInplaceKernel, max_positions: int
    ) -> None:
        """Run the pre-transformer on fused kernels, with rope as its rotary kernel for
        frames at positions below max_positions."""
        self.pre_transformer = FusedCode2WavTransformer(
            self.pre_transformer, rope, max_positions
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        if not self.is_channels_last:
            return super().forward(codes)
        elif codes.shape[1] != self.config.num_quantizers:
            raise ValueError(
                f"Expected {self.config.num_quantizers} layer of codes, "
                f"got {codes.shape[1]}"
            )
        else:
            pass
        hidden_states = self.code_embedding(codes + self.code_offset).mean(1)
        hidden_states = self.pre_transformer(
            inputs_embeds=hidden_states
        ).last_hidden_state
        for blocks in self.upsample:
            for block in blocks:
                hidden_states = channels_last_block(block, hidden_states)
        for block in self.decoder[:-1]:
            hidden_states = channels_last_block(block, hidden_states)
        final_conv = self.decoder[-1]
        return clamped_single_output_conv(
            F.pad(hidden_states, (0, 0, final_conv.padding, 0)),
            final_conv.conv,
            hidden_states.shape[1],
        )


class Qwen3OmniIncrementalCodec:
    """Decodes streams' new codec frames from their slots in a codec state pool: a stream's
    frames decoded in any split give the waveform Qwen3OmniCode2Wav gives on all of them at
    once."""

    def __init__(self, model: Qwen3OmniCode2Wav) -> None:
        assert self.supports(
            model
        ), "slot decodes need the fused channels-last code2wav"
        self.model = model
        # note (ratish): FlashAttention 3 runs on SM 8.x and 9.x with CUDA 12.3 or newer; other
        # GPUs gather each row's window out of the ring for SDPA.
        self.is_paged_attention = _is_fa3_supported()
        transformer = model.pre_transformer.transformer
        conv_histories, transconv_overlaps = conv_stack_state_layout(model)
        self.spec = CodecStateSpec(
            conv_histories=tuple(conv_histories),
            transconv_overlaps=tuple(transconv_overlaps),
            num_layers=len(transformer.layers),
            num_key_value_heads=transformer.config.num_key_value_heads,
            head_dim=transformer.layers[0].self_attn.head_dim,
            retained_context=transformer.window_size - 1,
        )
        # note (ratish): a SnakeBeta's coefficients depend on its weights alone, so they are
        # computed once here and every slot decode reads them.
        self.snake_coefficients = {
            module: SnakeBetaCoefficients.from_activation(module)
            for module in model.decoder.modules()
            if isinstance(module, (Qwen3OmniMoeSnakeBeta, FusedSnakeBeta))
        }

    @staticmethod
    def supports(model: Qwen3OmniCode2Wav) -> bool:
        """Slot decodes run the fused transformer on channels-last activations."""
        return (
            isinstance(model.pre_transformer, FusedCode2WavTransformer)
            and model.is_channels_last
        )

    def decode_slots(
        self,
        codes: torch.Tensor,
        pool: CodecStatePool,
        slot_index: torch.Tensor,
        *,
        first_decode: bool,
    ) -> torch.Tensor:
        """The (B, 1, samples) waveform of the (B, Q, T) codes after the frames the rows' slots
        hold, their state advanced in place; a graph captures it with a static index. A first
        decode starts every row's stream and reads no slot."""
        new_frames = codes.shape[-1]
        assert new_frames + (0 if first_decode else pool.retained_frames) <= (
            pool.ring_frames
        ), "a decode's window and new frames fit the ring"
        if first_decode:
            start_positions = torch.zeros_like(slot_index)
        else:
            start_positions = pool.frame_positions[slot_index]
        addresses = ring_addresses(
            slot_index,
            start_positions,
            new_frames,
            pool.ring_frames,
            pool.retained_frames,
        )
        hidden_states = self.model.code_embedding(codes + self.model.code_offset).mean(
            1
        )
        hidden_states = self.model.pre_transformer.forward_slot_frames(
            hidden_states,
            pool,
            addresses,
            first_decode=first_decode,
            is_paged_attention=self.is_paged_attention,
        )
        waveform = slot_conv_stack(
            self.model,
            hidden_states,
            SlotDecode(pool=pool, slot_index=slot_index, is_cold=first_decode),
            self.snake_coefficients,
        )
        pool.frame_positions.index_put_((slot_index,), start_positions + new_frames)
        return waveform
