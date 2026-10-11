// SPDX-License-Identifier: Apache-2.0
#include "speech_segments.h"

#include <algorithm>
#include <cmath>

namespace silero_vad {

namespace {

// Speech decisions are made on blocks of eight 512-sample chunks, 256 ms each.
constexpr int kChunksPerBlock = 8;

} // namespace

std::vector<std::pair<size_t, size_t>>
SegmentSpeech(const SileroVAD &vad, const std::vector<float> &samples,
              const SpeechSegmentConfig &config) {
  const std::vector<float> chunk_probabilities =
      vad.PredictProbabilities(samples);
  const int block_samples = kChunkSamples * kChunksPerBlock;
  const float block_seconds =
      static_cast<float>(block_samples) / static_cast<float>(kSampleRate);
  const int block_count =
      static_cast<int>(chunk_probabilities.size() / kChunksPerBlock);
  // Note (khazic): a block is speech unless every one of its chunks is silent.
  std::vector<float> block_probabilities(block_count);
  for (int block = 0; block < block_count; ++block) {
    float silence = 1.0f;
    for (int chunk = 0; chunk < kChunksPerBlock; ++chunk) {
      silence *= 1.0f - chunk_probabilities[block * kChunksPerBlock + chunk];
    }
    block_probabilities[block] = 1.0f - silence;
  }
  const auto blocks_of = [&](int milliseconds) {
    return static_cast<float>(milliseconds) / 1000.0f / block_seconds;
  };
  const int speech_pad_blocks =
      std::max(0, static_cast<int>(blocks_of(config.speech_pad_milliseconds)));
  // Note (khazic): minimum durations round up: 500 ms takes two blocks, not
  // one.
  const int min_speech_blocks = std::max(
      1,
      static_cast<int>(std::ceil(blocks_of(config.min_speech_milliseconds))));
  const int min_silence_blocks = std::max(
      1,
      static_cast<int>(std::ceil(blocks_of(config.min_silence_milliseconds))));

  std::vector<std::pair<size_t, size_t>> runs;
  int segment_start = 0;
  // The open run's last speech block; -1 when no run is open.
  int last_speech = -1;
  const auto close_run = [&]() {
    const int segment_end =
        std::min(last_speech + 1 + speech_pad_blocks, block_count);
    const size_t start = static_cast<size_t>(segment_start) * block_samples;
    const size_t end = std::min(
        static_cast<size_t>(segment_end) * block_samples, samples.size());
    if (segment_end - segment_start >= min_speech_blocks && start < end) {
      runs.emplace_back(start, end);
    } else {
    }
    last_speech = -1;
  };
  for (int block = 0; block < block_count; ++block) {
    if (block_probabilities[block] >= config.threshold) {
      if (last_speech < 0) {
        segment_start = std::max(0, block - speech_pad_blocks);
      } else {
      }
      last_speech = block;
    } else if (last_speech >= 0 && block - last_speech >= min_silence_blocks) {
      close_run();
    } else {
    }
  }
  if (last_speech >= 0) {
    close_run();
  } else {
  }

  // Note (khazic): runs closer than the merge gap join while they fit one
  // chunk; longer runs split at the chunk length.
  const size_t max_chunk_samples = static_cast<size_t>(
      std::max(1, static_cast<int>(config.max_chunk_seconds *
                                   static_cast<float>(kSampleRate))));
  const long max_gap_samples = static_cast<long>(
      config.merge_gap_seconds * static_cast<float>(kSampleRate));
  std::vector<std::pair<size_t, size_t>> chunks;
  for (const auto &[start, end] : runs) {
    if (!chunks.empty() &&
        static_cast<long>(start) - static_cast<long>(chunks.back().second) <=
            max_gap_samples &&
        end - chunks.back().first <= max_chunk_samples) {
      chunks.back().second = end;
    } else {
      for (size_t cut = start; cut < end;
           cut = std::min(cut + max_chunk_samples, end)) {
        chunks.emplace_back(cut, std::min(cut + max_chunk_samples, end));
      }
    }
  }
  return chunks;
}

} // namespace silero_vad
