// SPDX-License-Identifier: Apache-2.0
#include "common/audio.h"

#include <algorithm>
#include <stdexcept>

namespace omni_common {

size_t QuietestAudioPosition(const std::vector<float> &samples,
                             size_t start_sample, size_t end_sample,
                             size_t window_samples) {
  if (start_sample > end_sample || end_sample > samples.size() ||
      window_samples == 0 || window_samples > end_sample - start_sample) {
    throw std::invalid_argument("the energy window must fit the audio range");
  } else {
  }
  const size_t window_count = end_sample - start_sample - window_samples + 1;
  const float inverse_window = 1.0f / static_cast<float>(window_samples);
  float window_sum = 0.0f;
  for (size_t i = 0; i < window_samples; ++i) {
    const float square = samples[start_sample + i] * samples[start_sample + i];
    window_sum += square;
  }
  float minimum_energy = window_sum * inverse_window;
  size_t minimum_index = 0;
  for (size_t i = 1; i < window_count; ++i) {
    const float leaving = samples[start_sample + i - 1];
    const float entering = samples[start_sample + i + window_samples - 1];
    const float entering_square = entering * entering;
    const float leaving_square = leaving * leaving;
    window_sum += entering_square - leaving_square;
    const float energy = window_sum * inverse_window;
    if (energy < minimum_energy) {
      minimum_energy = energy;
      minimum_index = i;
    } else {
    }
  }
  return start_sample + minimum_index + window_samples / 2;
}

std::vector<std::pair<size_t, size_t>>
EnergyCutChunks(const std::vector<float> &samples, size_t max_chunk_samples,
                size_t search_samples, size_t window_samples,
                size_t min_tail_samples) {
  if (window_samples == 0 || window_samples > search_samples ||
      search_samples > max_chunk_samples ||
      min_tail_samples > max_chunk_samples / 2) {
    throw std::invalid_argument(
        "chunk counts must satisfy 0 < window <= search <= maximum and "
        "minimum tail <= maximum / 2");
  } else {
  }
  std::vector<std::pair<size_t, size_t>> chunks;
  for (size_t start_sample = 0; start_sample < samples.size();) {
    const size_t end_sample =
        start_sample +
        std::min(max_chunk_samples, samples.size() - start_sample);
    if (end_sample == samples.size()) {
      chunks.emplace_back(start_sample, end_sample);
      break;
    } else {
    }
    const size_t search_end_sample =
        std::min(end_sample, samples.size() - min_tail_samples);
    const size_t search_start_sample =
        search_end_sample -
        std::min(search_samples, search_end_sample - start_sample);
    const size_t cut_sample =
        search_end_sample - search_start_sample > window_samples
            ? std::max(start_sample + 1,
                       QuietestAudioPosition(samples, search_start_sample,
                                             search_end_sample, window_samples))
            : search_end_sample;
    chunks.emplace_back(start_sample, cut_sample);
    start_sample = cut_sample;
  }
  return chunks;
}

} // namespace omni_common
