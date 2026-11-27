// SPDX-License-Identifier: Apache-2.0
// What the speech models read a voice reference with: any WAV file as mono
// samples, and the resampler the CUDA servers use.
#pragma once

#include <string>
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

// The base64 payload of a data: URL, read as CUDA's speech server reads a
// reference's: a header with ;base64, in any case, and an audio/* media type,
// audio/wav when left out, then a nonempty payload. Throws
// std::invalid_argument naming field, in the CUDA server's words, otherwise.
std::string AudioDataUrlPayload(std::string_view url, const std::string &field);

// Throws std::invalid_argument for a rate pair whose filter would be
// impractically large.
void CheckResampleRates(int from_rate, int to_rate);

// torchaudio.functional.resample with its defaults: a Hann-windowed sinc with
// six zero crossings, rolloff 0.99. Throws as CheckResampleRates does.
std::vector<float> Resample(const std::vector<float> &samples, int from_rate,
                            int to_rate);

} // namespace omni_common
