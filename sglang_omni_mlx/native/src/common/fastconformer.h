// SPDX-License-Identifier: Apache-2.0
// NeMo FastConformer pieces the native ASR encoders share.
#pragma once

#include <string>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace omni_common {

// Transformer-XL relative positions length - 1 down to -(length - 1), as
// interleaved sin and cos rows [1, 2 * length - 1, width] in float32. Each row
// depends only on its position, so the rows equal a slice of NeMo's 5000-row
// table.
mlx::core::array RelativePositionTable(int length, int width);

// The macaron half of a block: linear2(silu(linear1(x))).
mlx::core::array ConformerFeedForward(const Checkpoint &checkpoint,
                                      const mlx::core::array &x,
                                      const std::string &prefix);

// NeMo relative-position self-attention of queries over keys_values [1,
// length, width], with pos_bias_u and pos_bias_v; relative_positions are
// RelativePositionTable rows for the key count. The position scores ride in as
// the additive mask, so the fused kernel adds them to the content scores.
mlx::core::array
RelativePositionAttention(const Checkpoint &checkpoint,
                          const mlx::core::array &queries,
                          const mlx::core::array &keys_values,
                          const mlx::core::array &relative_positions,
                          const std::string &prefix, int head_count);

// RelativePositionAttention over whole chunks at once: queries [1, chunks *
// chunk_frames, width] over keys_values [1, context + chunks * chunk_frames,
// width], each chunk seeing the context frames before it and itself, as one
// call per chunk would. The first padding_frames keys are not audio and are
// masked out; relative_positions are RelativePositionTable rows for context +
// chunk_frames keys.
mlx::core::array ChunkedRelativePositionAttention(
    const Checkpoint &checkpoint, const mlx::core::array &queries,
    const mlx::core::array &keys_values, int chunk_frames, int padding_frames,
    const mlx::core::array &relative_positions, const std::string &prefix,
    int head_count);

// The convolution module: pointwise_conv1's GLU, depthwise_conv with its batch
// norm folded in by FoldBatchNorms, SiLU and pointwise_conv2, over channels-
// last x [1, length, width].
mlx::core::array ConformerConvolution(const Checkpoint &checkpoint,
                                      const mlx::core::array &x,
                                      const std::string &prefix);

} // namespace omni_common
