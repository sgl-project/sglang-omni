// SPDX-License-Identifier: Apache-2.0
#include "common/kv_cache.h"

namespace omni_common {

namespace mx = mlx::core;

namespace {

constexpr int kKvCacheStepTokens = 256;

} // namespace

std::pair<mx::array, mx::array>
KVCache::UpdateAndFetch(const mx::array &keys, const mx::array &values) {
  const int new_token_count = keys.shape(2);
  if (!keys_.has_value() || offset_ + new_token_count > keys_->shape(2)) {
    const int step_count =
        (new_token_count + kKvCacheStepTokens - 1) / kKvCacheStepTokens;
    const mx::Shape grown = {keys.shape(0), keys.shape(1),
                             step_count * kKvCacheStepTokens, keys.shape(3)};
    const mx::array extra_keys = mx::zeros(grown, keys.dtype());
    const mx::array extra_values = mx::zeros(grown, values.dtype());
    if (!keys_.has_value()) {
      keys_ = extra_keys;
      values_ = extra_values;
    } else {
      const mx::Shape kept = {keys_->shape(0), keys_->shape(1), offset_,
                              keys_->shape(3)};
      keys_ = mx::concatenate(
          {mx::slice(*keys_, {0, 0, 0, 0}, kept), extra_keys}, 2);
      values_ = mx::concatenate(
          {mx::slice(*values_, {0, 0, 0, 0}, kept), extra_values}, 2);
    }
  } else {
  }
  const mx::Shape start = {0, 0, offset_, 0};
  const mx::Shape stop = {keys.shape(0), keys.shape(1),
                          offset_ + new_token_count, keys.shape(3)};
  keys_ = mx::slice_update(*keys_, keys, start, stop);
  values_ = mx::slice_update(*values_, values, start, stop);
  offset_ += new_token_count;
  const mx::Shape fetched = {keys_->shape(0), keys_->shape(1), offset_,
                             keys_->shape(3)};
  return {mx::slice(*keys_, {0, 0, 0, 0}, fetched),
          mx::slice(*values_, {0, 0, 0, 0}, fetched)};
}

} // namespace omni_common
