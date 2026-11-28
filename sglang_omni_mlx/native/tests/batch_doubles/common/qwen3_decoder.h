// SPDX-License-Identifier: Apache-2.0
// Checkpoint-free Qwen3 decoder for the MOSS batch timing regression.
#pragma once

#include <cstddef>
#include <functional>
#include <vector>

#include "common/kv_cache.h"
#include "mlx/mlx.h"

namespace omni_common {

struct Qwen3DecoderConfig {};

struct PrefilledSequence {
  std::vector<KVCache> caches;
  int prompt_length = 0;
  mlx::core::array first_token;
};

class Qwen3Decoder {
public:
  Qwen3Decoder() = default;

  mlx::core::array EmbedTokens(const mlx::core::array &ids) const {
    return mlx::core::zeros({1, static_cast<int>(ids.shape(1)), 2});
  }

  mlx::core::array Forward(const mlx::core::array &embeddings,
                           std::vector<KVCache> &) const {
    return embeddings;
  }

  mlx::core::array LastLogits(const mlx::core::array &) const {
    return mlx::core::array({0.0f, 0.0f, 0.0f, 1.0f});
  }

  std::vector<KVCache> NewCaches() const { return {}; }

  const Qwen3DecoderConfig &config() const { return config_; }

  void DecodeGreedyBatch(std::vector<PrefilledSequence> &sequences,
                         const std::function<bool(size_t, int)> &accept) const {
    std::vector<bool> finished(sequences.size(), false);
    size_t finished_count = 0;
    while (finished_count < sequences.size()) {
      for (size_t index = 0; index < sequences.size(); ++index) {
        if (finished[index]) {
          continue;
        } else {
        }
        if (accept(index, 3)) {
          finished[index] = true;
          ++finished_count;
        } else {
        }
      }
    }
  }

private:
  Qwen3DecoderConfig config_;
};

} // namespace omni_common
