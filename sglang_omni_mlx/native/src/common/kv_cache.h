// SPDX-License-Identifier: Apache-2.0
// Per-layer key/value cache shared by the native decoders.
#pragma once

#include <optional>
#include <utility>
#include <vector>

#include "mlx/mlx.h"

namespace omni_common {

// Positions a cache grows by when it runs out of room.
inline constexpr int kKvCacheStepTokens = 256;

// Per-layer key/value cache that grows in fixed steps. Keys and values may
// differ in head width, as in MLA.
class KVCache {
public:
  std::pair<mlx::core::array, mlx::core::array>
  UpdateAndFetch(const mlx::core::array &keys, const mlx::core::array &values);
  // Drops the newest token_count positions; the next update overwrites them.
  void Trim(int token_count);
  int offset() const { return offset_; }
  // The cache's buffers, to evaluate; empty before the first update.
  std::vector<mlx::core::array> Arrays() const;
  // The filled keys and values, [1, kv_heads, offset, head_dim].
  std::pair<mlx::core::array, mlx::core::array> Filled() const;

private:
  std::optional<mlx::core::array> keys_;
  std::optional<mlx::core::array> values_;
  int offset_ = 0;
};

} // namespace omni_common
