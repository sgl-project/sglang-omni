// SPDX-License-Identifier: Apache-2.0
// The speech segmentation Voxt runs before Cohere Transcribe on long audio, on
// Silero VAD's per-chunk probabilities, op for op as Voxt's Swift port.
#pragma once

#include <utility>
#include <vector>

#include "silero_vad.h"

namespace silero_vad {

// How speech runs become chunks: Voxt's long-form settings.
struct SpeechSegmentConfig {
  float threshold = 0.0f;
  int min_speech_milliseconds = 0;
  int min_silence_milliseconds = 0;
  int speech_pad_milliseconds = 0;
  float merge_gap_seconds = 0.0f;
  float max_chunk_seconds = 0.0f;
};

// [start, end) sample ranges of the speech in 16 kHz samples: runs of
// speech blocks, merged across short gaps and split at the longest chunk.
// Empty when there is no speech.
std::vector<std::pair<size_t, size_t>>
SegmentSpeech(const SileroVAD &vad, const std::vector<float> &samples,
              const SpeechSegmentConfig &config);

} // namespace silero_vad
