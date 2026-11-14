// SPDX-License-Identifier: Apache-2.0
// Voice reference preparation: a mono WAV (any rate) resampled to 24 kHz,
// silence-trimmed and truncated to at most the last max_seconds.
#pragma once

#include <string>
#include <string_view>
#include <vector>

namespace haru {

// Reads a mono PCM16/float32 WAV file, resamples to 24 kHz, cuts leading and
// trailing audio quieter than -45 dBFS and keeps the last max_seconds.
std::vector<float> PrepareReference(const std::string &wav_path,
                                    int max_seconds = 180);
// WAV bytes -> 24 kHz reference samples (the same pipeline as PrepareReference).
std::vector<float> PrepareReferenceBytes(std::string_view wav_bytes,
                                         int max_seconds = 180);

} // namespace haru
