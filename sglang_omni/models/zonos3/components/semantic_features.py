# SPDX-License-Identifier: Apache-2.0
"""The w2v-BERT teacher that maps a reference lane's filterbank to semantic features at codec frames."""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio.transforms as torchaudio_transforms
import triton
from torch import nn
from transformers import SeamlessM4TFeatureExtractor, Wav2Vec2BertModel

from sglang_omni.models.zonos3.components.codec_layers import sinc_resample
from sglang_omni.models.zonos3.components.teacher_kernels import (
    gated_causal_depthwise_convolution_kernel,
    relative_key_attention_kernel,
)
from sglang_omni.models.zonos3.components.teacher_norm_kernels import (
    residual_layer_norm,
)

TEACHER_SAMPLE_RATE_HZ = 16000
# note (zhaochenyang20): The teacher's SeamlessM4TFeatureExtractor filterbank, computed in torch on the encoder's device; the extractor's per-frame numpy loop held the GIL for about 17% of the serving process.
INT16_FULL_SCALE = 32768
TEACHER_WINDOW_SAMPLES = 400
TEACHER_HOP_SAMPLES = 160
TEACHER_FFT_SAMPLES = 512
TEACHER_PREEMPHASIS_COEFFICIENT = 0.97
TEACHER_MEL_FLOOR = 1.192092955078125e-07
TEACHER_NORMALIZATION_EPSILON = 1e-7
# Consecutive 80-bin filterbank frames stacked into one 160-dim teacher frame.
TEACHER_FRAME_STACK = 2
# Each codec frame averages the teacher frames under a centered window this many codec frames wide.
SEMANTIC_WINDOW_RATIO = 2.0
TEACHER_POSITION_EMBEDDINGS_TYPE = "relative_key"
# The conformer's half-step feed-forward residuals.
FEED_FORWARD_RESIDUAL_SCALE = 0.5
# note (zhaochenyang20): launch shapes timed on one H100 for a 999-frame lane, 16 heads of 64; 64-query blocks ran a layer's attention in 34 us against 36 us for 128-query blocks.
ATTENTION_BLOCK_QUERIES = 64
# note (zhaochenyang20): cuDNN's SDPA forward walks 128 keys per step; the same online-softmax steps keep 99.9% of attention outputs bitwise equal to it, where 64-key steps keep 90%.
ATTENTION_BLOCK_KEYS = 128
# note (zhaochenyang20): with 64 x 128 blocks, 4 warps and 2 stages ran a layer's attention in 31 us against 43 us for 3 stages, 45 us for 8 warps and 106 us for 2 warps (999-frame lane, CUDA graph replay, one H100).
ATTENTION_WARPS = 4
ATTENTION_STAGES = 2
# note (zhaochenyang20): between GEMMs, whose shared-memory carveout leaves little L1, per-tap reloads ran the GLU and depthwise convolution in 34 us per layer; a gathered window runs them in 12 us.
CONVOLUTION_BLOCK_FRAMES = 64
# note (zhaochenyang20): with 64-frame blocks, 64 channels and 8 warps ran a layer's GLU and depthwise convolution in 9.0 us against 9.4 us for 4 warps, 9.8-11.2 us for 32 channels and 10.1-33.5 us for 128 (999-frame lane, CUDA graph replay, one H100).
CONVOLUTION_BLOCK_CHANNELS = 64
CONVOLUTION_WARPS = 8


class W2vBertSemanticFeatures(nn.Module):
    """Frozen bf16 Wav2Vec2BertModel whose last hidden state is box-averaged to codec frames."""

    def __init__(
        self, teacher_directory: str | Path, input_sample_rate_hz: int
    ) -> None:
        super().__init__()
        self.input_sample_rate_hz: int = input_sample_rate_hz
        feature_extractor = SeamlessM4TFeatureExtractor()
        sample_indices = torch.arange(TEACHER_WINDOW_SAMPLES)
        frequency_bin_indices = torch.arange(TEACHER_FFT_SAMPLES // 2 + 1)
        phase_indices = (
            sample_indices[:, None] * frequency_bin_indices % TEACHER_FFT_SAMPLES
        )
        fourier_phases_radians = (
            2 * math.pi * phase_indices.double() / TEACHER_FFT_SAMPLES
        )
        # note (zhaochenyang20): The windowed 512-point real DFT is one matmul; with torch.fft.rfft here, 3 of 8 SeedTTS EN servers hit a CUDA illegal memory access or CUFFT_INTERNAL_ERROR while the vocoder thread ran istft.
        self.register_buffer(
            "windowed_fourier_basis",
            torch.from_numpy(feature_extractor.window)[:, None]
            * torch.cat(
                [fourier_phases_radians.cos(), fourier_phases_radians.sin()], dim=1
            ),
            persistent=False,
        )
        self.register_buffer(
            "mel_filters",
            torch.from_numpy(feature_extractor.mel_filters),
            persistent=False,
        )
        self.teacher = Wav2Vec2BertModel.from_pretrained(
            str(teacher_directory), dtype=torch.bfloat16
        ).eval()
        assert (
            self.teacher.config.position_embeddings_type
            == TEACHER_POSITION_EMBEDDINGS_TYPE
        ), self.teacher.config.position_embeddings_type
        assert (
            self.teacher.config.hidden_size % CONVOLUTION_BLOCK_CHANNELS == 0
        ), self.teacher.config.hidden_size
        # The teacher's semantics are its bf16 autocast pass, which computes every layer norm in float32.
        for module in self.teacher.modules():
            if isinstance(module, nn.LayerNorm):
                module.float()
            else:
                pass
        # One projection GEMM per layer reads the query, key and value weights stacked.
        for layer in self.teacher.encoder.layers:
            attention = layer.self_attn
            projections = [attention.linear_q, attention.linear_k, attention.linear_v]
            attention.query_key_value_weight = nn.Parameter(
                torch.cat([projection.weight for projection in projections]),
                requires_grad=False,
            )
            attention.query_key_value_bias = nn.Parameter(
                torch.cat([projection.bias for projection in projections]),
                requires_grad=False,
            )
            del attention.linear_q, attention.linear_k, attention.linear_v
        # A plain attribute, not a buffer: the teacher audio is resampled on the CPU with the float64-derived kernel the codec was trained with.
        self.teacher_resample_kernel: torch.Tensor = torchaudio_transforms.Resample(
            input_sample_rate_hz, TEACHER_SAMPLE_RATE_HZ
        ).kernel

    def teacher_waveform(self, waveform: torch.Tensor) -> torch.Tensor:
        """The (1, samples) CPU waveform resampled to the teacher's 16 kHz on the CPU."""
        rate_divisor = math.gcd(self.input_sample_rate_hz, TEACHER_SAMPLE_RATE_HZ)
        return sinc_resample(
            waveform,
            self.teacher_resample_kernel,
            self.input_sample_rate_hz // rate_divisor,
            TEACHER_SAMPLE_RATE_HZ // rate_divisor,
        )

    def filterbank_features(self, teacher_waveform: torch.Tensor) -> torch.Tensor:
        """(1, teacher frames, 160) float32 features of a (1, samples) 16 kHz waveform, on the waveform's device.

        A lane always holds an even number of filterbank frames, so no stacked frame needs padding or masking.
        """
        assert teacher_waveform.shape[0] == 1, teacher_waveform.shape
        # note (zhaochenyang20): float64 throughout; TF32 on CUDA moved quiet mel bins of SeedTTS EN references by up to 4.5 after normalization.
        analysis_frames = (teacher_waveform[0].double() * INT16_FULL_SCALE).unfold(
            0, TEACHER_WINDOW_SAMPLES, TEACHER_HOP_SAMPLES
        )
        analysis_frames = analysis_frames - analysis_frames.mean(1, keepdim=True)
        analysis_frames = torch.cat(
            [
                analysis_frames[:, :1] * (1 - TEACHER_PREEMPHASIS_COEFFICIENT),
                analysis_frames[:, 1:]
                - TEACHER_PREEMPHASIS_COEFFICIENT * analysis_frames[:, :-1],
            ],
            dim=1,
        )
        # The extractor stores the spectrum as complex64 before it squares the magnitude in float64.
        real_spectrum, imaginary_spectrum = (
            (analysis_frames @ self.windowed_fourier_basis)
            .float()
            .double()
            .chunk(2, dim=1)
        )
        power_spectrum = real_spectrum**2 + imaginary_spectrum**2
        log_mel_energies = (
            (power_spectrum @ self.mel_filters)
            .clamp(min=TEACHER_MEL_FLOOR)
            .log()
            .float()
        )
        normalized_log_mel = (log_mel_energies - log_mel_energies.mean(0)) / torch.sqrt(
            log_mel_energies.var(0, correction=1) + TEACHER_NORMALIZATION_EPSILON
        )
        return normalized_log_mel.reshape(
            1, -1, normalized_log_mel.shape[1] * TEACHER_FRAME_STACK
        )

    def teacher_states(self, input_features: torch.Tensor) -> torch.Tensor:
        """(batch, teacher frames, hidden) float32 last hidden states of (batch, teacher frames, 160) bfloat16 features.

        The teacher's bf16 autocast pass with every lane frame attended: norms in float32, projections and convolutions in bfloat16, and a residual stream that is bfloat16 in the first layer and float32 after its final norm. Non-CUDA features run the attention and the gated convolution in torch ops.
        """
        # note (zhaochenyang20): fused norms, attention and depthwise convolution took one lane's semantic features from 9.6 ms to 3.1 ms of CUDA graph replay on one H100.
        projection = self.teacher.feature_projection
        _, normalized_features, _ = residual_layer_norm(
            input_features, projection.layer_norm, torch.bfloat16
        )
        hidden_states = projection.projection(normalized_features)
        batch_size, frame_count, hidden_size = hidden_states.shape
        layers = self.teacher.encoder.layers
        _, normalized_states, _ = residual_layer_norm(
            hidden_states, layers[0].ffn1_layer_norm, torch.bfloat16
        )
        for layer_index, layer in enumerate(layers):
            hidden_states, normalized_states, _ = residual_layer_norm(
                hidden_states,
                layer.self_attn_layer_norm,
                torch.bfloat16,
                layer.ffn1(normalized_states),
                FEED_FORWARD_RESIDUAL_SCALE,
            )
            attention = layer.self_attn
            query_key_value_states = F.linear(
                normalized_states,
                attention.query_key_value_weight,
                attention.query_key_value_bias,
            )
            if normalized_states.is_cuda:
                attention_output = torch.empty_like(normalized_states)
                relative_key_attention_kernel[
                    (
                        triton.cdiv(frame_count, ATTENTION_BLOCK_QUERIES),
                        batch_size * attention.num_heads,
                    )
                ](
                    query_key_value_states,
                    attention.distance_embedding.weight,
                    attention_output,
                    frame_count,
                    query_key_value_states.stride(0),
                    query_key_value_states.stride(1),
                    attention_output.stride(0),
                    attention_output.stride(1),
                    attention.scaling,
                    HEAD_COUNT=attention.num_heads,
                    HEAD_SIZE=attention.head_size,
                    LEFT_DISTANCES=attention.left_max_position_embeddings,
                    RIGHT_DISTANCES=attention.right_max_position_embeddings,
                    DISTANCE_BLOCK=triton.next_power_of_2(
                        attention.distance_embedding.num_embeddings
                    ),
                    BLOCK_QUERIES=ATTENTION_BLOCK_QUERIES,
                    BLOCK_KEYS=ATTENTION_BLOCK_KEYS,
                    num_warps=ATTENTION_WARPS,
                    num_stages=ATTENTION_STAGES,
                )
            else:
                query, key, value = query_key_value_states.view(
                    batch_size, frame_count, 3, attention.num_heads, attention.head_size
                ).permute(2, 0, 3, 1, 4)
                positions = torch.arange(frame_count, device=normalized_states.device)
                distance_indices = (positions[None, :] - positions[:, None]).clamp(
                    -attention.left_max_position_embeddings,
                    attention.right_max_position_embeddings,
                ) + attention.left_max_position_embeddings
                position_bias = (
                    (query @ attention.distance_embedding.weight.T)
                    .float()
                    .gather(-1, distance_indices.expand(*query.shape[:2], -1, -1))
                )
                probabilities = (
                    (query.float() @ key.float().transpose(-1, -2) + position_bias)
                    * attention.scaling
                ).softmax(dim=-1)
                attention_output = (
                    (probabilities.to(value.dtype) @ value)
                    .transpose(1, 2)
                    .reshape(batch_size, frame_count, hidden_size)
                )
            hidden_states, normalized_states, _ = residual_layer_norm(
                hidden_states,
                layer.conv_module.layer_norm,
                torch.bfloat16,
                attention.linear_out(attention_output),
            )
            convolution = layer.conv_module
            kernel_size = convolution.depthwise_conv.kernel_size[0]
            projected_states = F.linear(
                normalized_states, convolution.pointwise_conv1.weight[..., 0]
            )
            if normalized_states.is_cuda:
                convolved_states = torch.empty_like(normalized_states)
                gated_causal_depthwise_convolution_kernel[
                    (
                        triton.cdiv(frame_count, CONVOLUTION_BLOCK_FRAMES),
                        hidden_size // CONVOLUTION_BLOCK_CHANNELS,
                        batch_size,
                    )
                ](
                    projected_states,
                    convolution.depthwise_conv.weight,
                    convolved_states,
                    frame_count,
                    CHANNELS=hidden_size,
                    KERNEL_SIZE=kernel_size,
                    BLOCK_FRAMES=CONVOLUTION_BLOCK_FRAMES,
                    BLOCK_CHANNELS=CONVOLUTION_BLOCK_CHANNELS,
                    WINDOW_FRAMES=triton.next_power_of_2(
                        CONVOLUTION_BLOCK_FRAMES + kernel_size - 1
                    ),
                    num_warps=CONVOLUTION_WARPS,
                )
            else:
                projected_values, gates = projected_states.float().chunk(2, dim=-1)
                gated_states = (projected_values * torch.sigmoid(gates)).to(
                    projected_states.dtype
                )
                convolved_states = (
                    F.conv1d(
                        F.pad(
                            gated_states.float().transpose(1, 2), (kernel_size - 1, 0)
                        ),
                        convolution.depthwise_conv.weight.float(),
                        groups=hidden_size,
                    )
                    .transpose(1, 2)
                    .to(normalized_states.dtype)
                )
            _, activated_states, _ = residual_layer_norm(
                convolved_states,
                convolution.depthwise_layer_norm,
                torch.bfloat16,
                should_apply_silu=True,
            )
            hidden_states, normalized_states, _ = residual_layer_norm(
                hidden_states,
                layer.ffn2_layer_norm,
                torch.bfloat16,
                F.linear(activated_states, convolution.pointwise_conv2.weight[..., 0]),
            )
            # The final norm's float32 output is the next residual; the next layer's first norm reads it in the same pass.
            _, hidden_states, normalized_states = residual_layer_norm(
                hidden_states,
                layer.final_layer_norm,
                torch.float32,
                layer.ffn2(normalized_states),
                FEED_FORWARD_RESIDUAL_SCALE,
                (
                    layers[layer_index + 1].ffn1_layer_norm
                    if layer_index + 1 < len(layers)
                    else None
                ),
            )
        return hidden_states

    @torch.inference_mode()
    def forward(self, teacher_waveform: torch.Tensor, frame_count: int) -> torch.Tensor:
        """Map a (1, samples) 16 kHz device waveform to (1, hidden, frame_count) float32 features.

        Each codec frame box-averages the teacher states under a centered window; its edges fall on fractional teacher frames, so boundary frames count in proportion to their overlap, via prefix sums.
        """
        filterbank_features = self.filterbank_features(teacher_waveform)
        device = filterbank_features.device
        input_features = filterbank_features.to(torch.bfloat16)
        teacher_hidden_states = self.teacher_states(input_features).transpose(1, 2)
        teacher_frame_count = teacher_hidden_states.shape[-1]
        ratio = teacher_frame_count / frame_count
        centers = (
            torch.arange(frame_count, device=device, dtype=torch.float32) + 0.5
        ) * ratio
        half_width = 0.5 * SEMANTIC_WINDOW_RATIO * ratio
        window_bounds = torch.stack([centers - half_width, centers + half_width]).clamp(
            0, teacher_frame_count
        )
        bound_indices = window_bounds.floor().long()
        prefix_sums = F.pad(teacher_hidden_states.cumsum(dim=-1), (1, 0))
        bound_integrals = prefix_sums[..., bound_indices] + teacher_hidden_states[
            ..., bound_indices.clamp(max=teacher_frame_count - 1)
        ] * (window_bounds - bound_indices)
        return (bound_integrals[..., 1, :] - bound_integrals[..., 0, :]) / (
            window_bounds[1] - window_bounds[0]
        )
