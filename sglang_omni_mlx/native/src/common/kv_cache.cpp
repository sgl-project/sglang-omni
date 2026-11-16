// SPDX-License-Identifier: Apache-2.0
#include "common/kv_cache.h"

#include <cassert>

namespace omni_common {

namespace mx = mlx::core;

std::vector<mx::array> KVCache::Arrays() const {
  if (keys_.has_value()) {
    return {*keys_, *values_};
  } else {
    return {};
  }
}

std::pair<mx::array, mx::array>
KVCache::UpdateAndFetch(const mx::array &keys, const mx::array &values) {
  const int new_token_count = keys.shape(2);
  const auto positions = [](const mx::array &heads, int token_count) {
    return mx::Shape{heads.shape(0), heads.shape(1), token_count,
                     heads.shape(3)};
  };
  if (!keys_.has_value() || offset_ + new_token_count > keys_->shape(2)) {
    const int grown_token_count = (new_token_count + kKvCacheStepTokens - 1) /
                                  kKvCacheStepTokens * kKvCacheStepTokens;
    const mx::array extra_keys =
        mx::zeros(positions(keys, grown_token_count), keys.dtype());
    const mx::array extra_values =
        mx::zeros(positions(values, grown_token_count), values.dtype());
    if (!keys_.has_value()) {
      keys_ = extra_keys;
      values_ = extra_values;
    } else {
      keys_ = mx::concatenate(
          {mx::slice(*keys_, {0, 0, 0, 0}, positions(*keys_, offset_)),
           extra_keys},
          2);
      values_ = mx::concatenate(
          {mx::slice(*values_, {0, 0, 0, 0}, positions(*values_, offset_)),
           extra_values},
          2);
    }
  } else {
  }
  const mx::Shape start = {0, 0, offset_, 0};
  keys_ = mx::slice_update(*keys_, keys, start,
                           positions(keys, offset_ + new_token_count));
  values_ = mx::slice_update(*values_, values, start,
                             positions(values, offset_ + new_token_count));
  offset_ += new_token_count;
  return {mx::slice(*keys_, {0, 0, 0, 0}, positions(*keys_, offset_)),
          mx::slice(*values_, {0, 0, 0, 0}, positions(*values_, offset_))};
}

void KVCache::Trim(int token_count) {
  assert(token_count <= offset_);
  offset_ -= token_count;
}

std::pair<mx::array, mx::array> KVCache::Filled() const {
  const mx::Shape stop = {keys_->shape(0), keys_->shape(1), offset_,
                          keys_->shape(3)};
  return {mx::slice(*keys_, {0, 0, 0, 0}, stop),
          mx::slice(*values_, {0, 0, 0, 0}, stop)};
}

} // namespace omni_common
