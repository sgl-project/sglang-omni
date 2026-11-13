// SPDX-License-Identifier: Apache-2.0
#include "common/audio_chunks.h"

#include <algorithm>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

void Require(bool condition, std::string_view message) {
  if (!condition) {
    throw std::runtime_error(std::string(message));
  } else {
  }
}

void TestQuietestWindow() {
  std::vector<float> samples(40, 1.0f);
  std::fill(samples.begin() + 4, samples.begin() + 8, 0.0f);
  std::fill(samples.begin() + 25, samples.begin() + 29, 0.0f);
  Require(omni_common::QuietestAudioPosition(samples, 20, 35, 4) == 27,
          "the search must not use a quieter window outside its range");
  Require(omni_common::QuietestAudioPosition(samples, 0, 40, 4) == 6,
          "equal minima must select the earliest complete window");
  Require(omni_common::QuietestAudioPosition(samples, 25, 29, 4) == 27,
          "one complete window must return its center");
  std::fill(samples.begin() + 31, samples.begin() + 35, 0.0f);
  Require(omni_common::QuietestAudioPosition(samples, 29, 35, 4) == 33,
          "the final complete window must be included in the search");
  Require(omni_common::QuietestAudioPosition(samples, 0, 4, 3) == 1,
          "an odd window center must round down");
}

void TestBackwardQuietCut() {
  std::vector<float> samples(65, 1.0f);
  std::fill(samples.begin() + 26, samples.begin() + 28, 0.5f);
  std::fill(samples.begin() + 30, samples.begin() + 32, 0.0f);
  const auto chunks = omni_common::BoundedEnergyCutChunks(samples, 30, 5, 2, 0);
  Require(chunks.front().second == 27,
          "a quiet cut must precede the cap, even with quieter audio after it");
  Require(chunks.back().second == samples.size(),
          "quiet cuts must retain the tail of the recording");
}

void TestFlatAudio() {
  const std::vector<float> samples(61, 0.5f);
  const auto chunks = omni_common::BoundedEnergyCutChunks(samples, 30, 5, 2, 0);
  Require(chunks.front().second == 26,
          "an energy tie must choose the first complete window");
  const auto no_search =
      omni_common::BoundedEnergyCutChunks(samples, 30, 2, 2, 0);
  Require(no_search.front().second == 30,
          "without multiple candidate windows the cut must stay at the cap");
}

void TestCoverage() {
  for (const size_t max_samples :
       {size_t{1}, size_t{2}, size_t{17}, size_t{30}}) {
    std::vector<size_t> sample_counts(4 * max_samples + 2);
    std::iota(sample_counts.begin(), sample_counts.end(), size_t{0});
    sample_counts.push_back(41 * max_samples + 7);
    for (const size_t sample_count : sample_counts) {
      const std::vector<float> samples(sample_count, 0.5f);
      for (const size_t min_tail_samples :
           {size_t{0}, max_samples / 3, max_samples / 2}) {
        for (const size_t search_samples :
             {std::max(size_t{1}, max_samples / 3), max_samples}) {
          for (const size_t window_samples : {size_t{1}, search_samples}) {
            const auto chunks = omni_common::BoundedEnergyCutChunks(
                samples, max_samples, search_samples, window_samples,
                min_tail_samples);
            size_t next_sample = 0;
            for (const auto &[start_sample, end_sample] : chunks) {
              Require(start_sample == next_sample &&
                          start_sample < end_sample &&
                          end_sample <= sample_count,
                      "chunks must be contiguous without loss or duplication");
              Require(end_sample - start_sample <= max_samples,
                      "no chunk may exceed the encoder's sample limit");
              next_sample = end_sample;
            }
            Require(next_sample == sample_count,
                    "the final sample must be covered");
            if (sample_count > max_samples) {
              Require(chunks.back().second - chunks.back().first >=
                          min_tail_samples,
                      "the final chunk must retain the requested context");
            } else if (sample_count > 0) {
              Require(chunks.size() == 1 && chunks.front().first == 0 &&
                          chunks.front().second == sample_count,
                      "short audio must remain whole and unpadded");
            } else {
              Require(chunks.empty(), "empty audio must not create a chunk");
            }
          }
        }
      }
    }
  }
}

void TestMinimumTail() {
  std::vector<float> samples(31, 1.0f);
  std::fill(samples.begin() + 19, samples.begin() + 21, 0.5f);
  std::fill(samples.begin() + 26, samples.begin() + 28, 0.0f);
  const auto untailed_chunks =
      omni_common::BoundedEnergyCutChunks(samples, 30, 5, 2, 0);
  Require(untailed_chunks.front().second == 27,
          "without a minimum tail the cut must stay at the quietest window");
  const auto chunks =
      omni_common::BoundedEnergyCutChunks(samples, 30, 5, 2, 10);
  Require(chunks.size() == 2 && chunks.front().second == 20 &&
              chunks.back().first == 20 && chunks.back().second == 31,
          "a short final tail must move the preceding quiet search earlier");
  const auto no_search =
      omni_common::BoundedEnergyCutChunks(samples, 30, 2, 2, 10);
  Require(no_search.front().second == 21 &&
              no_search.back().second - no_search.back().first == 10,
          "a single-window search must still reserve the minimum tail");
}

void TestInvalidCounts() {
  const std::vector<float> samples(10, 0.5f);
  for (const auto &[max_samples, search_samples, window_samples] :
       {std::tuple<size_t, size_t, size_t>{0, 0, 0},
        {5, 0, 1},
        {5, 2, 0},
        {5, 2, 3},
        {5, 6, 1}}) {
    bool rejected = false;
    try {
      omni_common::BoundedEnergyCutChunks(samples, max_samples, search_samples,
                                          window_samples, 0);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    Require(rejected, "invalid chunk counts must fail before planning");
  }
  for (const auto &[max_samples, min_tail_samples] :
       {std::pair<size_t, size_t>{0, 1}, {1, 1}, {5, 3}, {30, 16}}) {
    bool rejected = false;
    try {
      omni_common::BoundedEnergyCutChunks(samples, max_samples, 1, 1,
                                          min_tail_samples);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    Require(rejected, "minimum tails greater than half the cap must fail");
  }
  for (const auto &[start_sample, end_sample, window_samples] :
       {std::tuple<size_t, size_t, size_t>{5, 4, 1},
        {0, 11, 1},
        {0, 10, 0},
        {0, 1, 2},
        {5, 5, 1}}) {
    bool rejected = false;
    try {
      omni_common::QuietestAudioPosition(samples, start_sample, end_sample,
                                         window_samples);
    } catch (const std::invalid_argument &) {
      rejected = true;
    }
    Require(rejected, "invalid energy windows must fail before reading audio");
  }
}

} // namespace

int main() {
  try {
    TestQuietestWindow();
    TestCoverage();
    TestBackwardQuietCut();
    TestFlatAudio();
    TestMinimumTail();
    TestInvalidCounts();
    std::cout << "audio chunk tests passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
