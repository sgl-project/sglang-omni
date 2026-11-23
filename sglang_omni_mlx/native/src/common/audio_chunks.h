// SPDX-License-Identifier: Apache-2.0
// Audio chunk boundaries at low-energy windows.
#pragma once

#include <cstddef>
#include <utility>
#include <vector>

namespace omni_common {

// Center of the earliest quietest complete window in [start_sample,
// end_sample). Throws std::invalid_argument for invalid bounds or a window that
// does not fit.
size_t QuietestAudioPosition(const std::vector<float> &samples,
                             size_t start_sample, size_t end_sample,
                             size_t window_samples);

// Consecutive ranges no longer than max_chunk_samples, cut at a quiet window
// in the last search_samples before each limit; short audio stays whole.
// When split, reserve at least min_tail_samples for the final range by moving
// the preceding search earlier. Requires min_tail_samples <= max_chunk_samples
// / 2 and 0 < window_samples <= search_samples <= max_chunk_samples.
std::vector<std::pair<size_t, size_t>>
BoundedEnergyCutChunks(const std::vector<float> &samples,
                       size_t max_chunk_samples, size_t search_samples,
                       size_t window_samples, size_t min_tail_samples);

} // namespace omni_common
