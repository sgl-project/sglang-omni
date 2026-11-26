// SPDX-License-Identifier: Apache-2.0
// Draft types for the checkpoint-free MOSS batch timing regression.
#pragma once

#include <atomic>
#include <filesystem>
#include <functional>

#include "common/qwen3_decoder.h"
#include "mlx/mlx.h"

namespace omni_common {

enum class DraftPolicy { kAdaptive, kFixed };

struct DraftOptions {
  std::filesystem::path directory;
  int quantization_bits = 0;
};

struct DraftStatistics {
  int verify_round_count = 0;
  int proposed_token_count = 0;
  int accepted_token_count = 0;
  double prefill_ms = 0.0;
  double decode_rounds_ms = 0.0;
};

class DFlashDraft {
public:
  DFlashDraft(const DraftOptions &, const Qwen3DecoderConfig &) {}

  DraftStatistics Decode(const Qwen3Decoder &, const mlx::core::array &, int,
                         int, const std::function<bool(int)> &,
                         const std::atomic<bool> &,
                         DraftPolicy = DraftPolicy::kFixed) const {
    return {};
  }
};

} // namespace omni_common
