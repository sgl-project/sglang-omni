// SPDX-License-Identifier: Apache-2.0
// Quantized linear for four to eight rows that reads each weight once, for the
// verify pass of speculative decoding. It applies to any quantized linear with
// that many rows, including batched decoding, not only the verify pass.
#pragma once

#include <optional>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace omni_common {

// x [..., in_features] times the transposed affine-quantized weight; empty when
// the shapes, bit width, dtype or the opt-in switch
// SGLANG_OMNI_MLX_MULTI_ROW_QMV leave the call to MLX.
std::optional<mlx::core::array>
MultiRowQuantizedLinear(const mlx::core::array &x,
                        const mlx::core::array &weight,
                        const mlx::core::array &scales,
                        const std::optional<mlx::core::array> &biases,
                        const QuantizationConfig &quantization);

} // namespace omni_common
