// SPDX-License-Identifier: Apache-2.0
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_decoder.h"

#include <algorithm>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

int main() {
  namespace mx = mlx::core;
  mx::set_default_device(mx::Device(mx::Device::cpu));
  for (int invalid_step_tokens : {0, -1}) {
    bool rejected = false;
    try {
      qwen3_asr::KVCache cache(invalid_step_tokens);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    if (!rejected) {
      return 1;
    } else {
    }
  }

  bool overflow_rejected = false;
  try {
    qwen3_asr::KVCache cache(16);
    const mx::array huge = mx::broadcast_to(
        mx::array(0.0f), {1, 1, std::numeric_limits<int>::max(), 1});
    cache.UpdateAndFetch(huge, huge);
  } catch (const std::overflow_error &) {
    overflow_rejected = true;
  }
  if (!overflow_rejected) {
    return 2;
  } else {
  }

  qwen3_asr::KVCache compact(16);
  qwen3_asr::KVCache original;
  for (int step = 0; step < 22; ++step) {
    if (step == 19) {
      compact.Trim(3);
      original.Trim(3);
    } else {
    }
    const int new_token_count = step == 0 ? 2 : 1;
    std::vector<float> values;
    for (int index = 0; index < new_token_count * 2; ++index) {
      values.push_back(static_cast<float>(step * 5 + index));
    }
    const mx::array key(values.begin(), {1, 1, new_token_count, 2});
    const mx::array value = mx::multiply(key, mx::array(2.0f));
    const auto [compact_keys, compact_values] =
        compact.UpdateAndFetch(key, value);
    const auto [default_keys, default_values] =
        original.UpdateAndFetch(key, value);
    mx::eval(compact_keys, compact_values, default_keys, default_values);
    if (compact_keys.shape() != default_keys.shape() ||
        compact_values.shape() != default_values.shape()) {
      return 3;
    } else {
    }
    if (!std::equal(compact_keys.data<float>(),
                    compact_keys.data<float>() + compact_keys.size(),
                    default_keys.data<float>())) {
      return 4;
    } else {
    }
    if (!std::equal(compact_values.data<float>(),
                    compact_values.data<float>() + compact_values.size(),
                    default_values.data<float>())) {
      return 5;
    } else {
    }
  }
  std::cout << "invalid steps and overflow rejected; growth and trim match "
               "default cache\n";
}
