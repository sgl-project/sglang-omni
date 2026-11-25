// SPDX-License-Identifier: Apache-2.0
// The Whisper audio encoder MOSS-Transcribe-Diarize and ARK-ASR share: the
// checkpoint's conv weights in MLX's layout, and one encoder layer. Each
// model keeps its own positions (MOSS's learned table, ARK-ASR's rotary) and
// adapter.
#pragma once

#include <string>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace omni_common {

// The checkpoint keeps PyTorch's Conv1d layout [out, in, kernel]; MLX takes
// [out, kernel, in]. Transposes the 3-dim weights under prefix + ".conv".
WeightMap WhisperConvMlxLayout(WeightMap weights, const std::string &prefix);

// One pre-norm encoder layer: self-attention with a residual, then a GELU MLP
// with a residual, under prefix + ".layers.N". With rotary_query_key the
// queries and keys get rotary positions over the first half of each head in
// adjacent pairs (ARK-ASR); MOSS adds its learned positions before the layers
// and passes false.
mlx::core::array WhisperEncoderLayer(const Checkpoint &checkpoint,
                                     const mlx::core::array &x,
                                     const std::string &layer_prefix,
                                     int head_count, bool rotary_query_key);

} // namespace omni_common
