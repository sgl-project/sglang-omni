// SPDX-License-Identifier: Apache-2.0
// Voice reference preparation: any WAV file as mono samples at the model rate,
// silence-trimmed and truncated to at most the last max_seconds.
#pragma once

#include <string_view>
#include <vector>

namespace haru {

// Reads the WAV bytes, resamples to sample_rate, cuts leading and trailing
// audio quieter than -45 dBFS and keeps the last max_seconds.
// Throws std::invalid_argument for a file that is not such a WAV.
std::vector<float> PrepareReference(std::string_view wav_bytes, int sample_rate,
                                    int max_seconds = 180);

} // namespace haru
