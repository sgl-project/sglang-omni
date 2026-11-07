// SPDX-License-Identifier: Apache-2.0
// What the speech models read a voice reference with: any WAV file as mono
// samples, and the resampler the CUDA servers use.
#pragma once

#include <string_view>
#include <vector>

namespace omni_common {

struct DecodedAudio {
  std::vector<float> samples;
  int sample_rate = 0;
  // The file's channels, before they were averaged.
  int channel_count = 0;
};

// A WAV file held in memory (8, 16, 24 or 32-bit PCM, 32 or 64-bit float,
// plain or WAVE_FORMAT_EXTENSIBLE) as mono float32 samples, channels averaged.
// Throws std::invalid_argument for anything else and for non-finite samples.
DecodedAudio DecodeAnyWav(std::string_view file_bytes);

// Throws std::invalid_argument for a rate pair whose filter would be
// impractically large.
void CheckResampleRates(int from_rate, int to_rate);

// torchaudio.functional.resample with its defaults: a Hann-windowed sinc with
// six zero crossings, rolloff 0.99. Throws as CheckResampleRates does.
std::vector<float> Resample(const std::vector<float> &samples, int from_rate,
                            int to_rate);

} // namespace omni_common
