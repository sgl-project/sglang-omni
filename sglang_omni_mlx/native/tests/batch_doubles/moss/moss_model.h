// SPDX-License-Identifier: Apache-2.0
// In-memory MOSS model for the batch timing regression.
#pragma once

#include <filesystem>
#include <numeric>
#include <vector>

#include "common/qwen3_decoder.h"
#include "mlx/mlx.h"

namespace moss {

class MossTranscribeDiarize {
public:
  explicit MossTranscribeDiarize(const std::filesystem::path &) {}

  mlx::core::array
  EncodeAudio(const mlx::core::array &,
              const std::vector<int> &window_token_counts) const {
    const int token_count = std::accumulate(window_token_counts.begin(),
                                            window_token_counts.end(), 0);
    return mlx::core::zeros({token_count, 2});
  }

  const omni_common::Qwen3Decoder &decoder() const { return decoder_; }
  int mel_bin_count() const { return 2; }
  int audio_token_id() const { return 2; }
  int audio_merge_size() const { return 1; }

private:
  omni_common::Qwen3Decoder decoder_;
};

} // namespace moss
