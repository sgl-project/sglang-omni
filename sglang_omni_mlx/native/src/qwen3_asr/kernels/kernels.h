// SPDX-License-Identifier: Apache-2.0
// MLX primitives on Metal kernels tuned for Qwen3-ASR's single-request shapes,
// which ship in sglang_qwen3_asr.metallib and, for the neural accelerators, in
// sglang_qwen3_asr_nax.metallib: quantized products with fused QKV and gated
// MLP layers, the decoder's one-token attention and greedy argmax, and the
// encoder's convolutions. Inputs outside those shapes run MLX's own ops.
#pragma once

#include <optional>
#include <string>
#include <vector>

#include "mlx/mlx.h"

namespace qwen3_asr {

enum class KernelSet {
  // MLX's own ops.
  kMlx,
  // Kernels tuned for the Qwen3-ASR shapes.
  kTuned,
};

KernelSet ParseKernelSet(const std::string &name);
// Process-wide, set once at startup.
void SetKernelSet(KernelSet kernel_set);

// The packed weight [N, K * bits / 32] of an affine-quantized layer, with its
// scales and biases [N, K / group_size].
struct QuantizedWeights {
  mlx::core::array weight;
  mlx::core::array scales;
  mlx::core::array biases;
};

// An RMS normalization of a product's input, x * rsqrt(mean(x^2) + eps) *
// weight.
struct InputNorm {
  mlx::core::array weight;
  float eps;
};

// x [..., K] @ weight^T for each layer, [..., N] each, as one product when
// the kernels can fuse them. With a norm, x is normalized first; a decode
// product normalizes it as it loads it.
std::vector<mlx::core::array>
QuantizedProjections(const mlx::core::array &x,
                     const std::vector<QuantizedWeights> &layers,
                     int group_size, int bits,
                     const std::optional<InputNorm> &norm = std::nullopt);

// silu(x @ gate^T) * (x @ up^T), x normalized first as in QuantizedProjections.
mlx::core::array
QuantizedGatedLinear(const mlx::core::array &x, const QuantizedWeights &gate,
                     const QuantizedWeights &up, int group_size, int bits,
                     const std::optional<InputNorm> &norm = std::nullopt);

// residual + x @ weight^T; for one vector the add runs inside the product.
mlx::core::array QuantizedLinearAdd(const mlx::core::array &x,
                                    const QuantizedWeights &layer,
                                    const mlx::core::array &residual,
                                    int group_size, int bits);

// Whether DecodeAttention takes these query and key rows.
bool DecodeAttentionFusable(const mlx::core::array &queries,
                            const mlx::core::array &keys);

// One decode token's attention from its QKV product rows [1, 1, heads * 128]:
// queries RMS-normalized with q_norm and rotated at token_offset, keys likewise
// with k_norm, and values, written into the caches [1, kv_heads, capacity,
// 128] at token_offset, and two query heads per KV head attending over the
// cached tokens through it. Returns the output [1, query_heads, 1, 128] and
// the two caches, which share the given caches' buffers.
std::vector<mlx::core::array>
DecodeAttention(const mlx::core::array &queries, const mlx::core::array &keys,
                const mlx::core::array &values, const mlx::core::array &q_norm,
                const mlx::core::array &k_norm,
                const mlx::core::array &key_cache,
                const mlx::core::array &value_cache, int token_offset,
                float eps, float rope_base, float scale);

// gelu(conv2d(x, weight, stride 2, padding 1) + bias) for x [N, H, W, C] and
// weight [O, 3, 3, C]: one kernel for float16 x with one channel, or on GPUs
// with neural accelerators for float16 x whose channels its tile divides;
// MLX's ops otherwise.
mlx::core::array ConvolutionGelu(const mlx::core::array &x,
                                 const mlx::core::array &weight,
                                 const mlx::core::array &bias);

// The argmax over the last axis of logits [..., vocabulary] as uint32, ties to
// the lower index; one bfloat16 row reduces over many threadgroups.
mlx::core::array GreedyTokens(const mlx::core::array &logits);

} // namespace qwen3_asr
