// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <optional>
#include <utility>

#include "mlx/mlx.h"

namespace omni_common {

// Per-layer key/value cache that grows in fixed steps.
class KVCache {
public:
  std::pair<mlx::core::array, mlx::core::array>
  UpdateAndFetch(const mlx::core::array &keys, const mlx::core::array &values);
  int offset() const { return offset_; }

private:
  std::optional<mlx::core::array> keys_;
  std::optional<mlx::core::array> values_;
  int offset_ = 0;
};

} // namespace omni_common
