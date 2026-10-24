// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

namespace dots_tts {

std::vector<float> Resample(const std::vector<float> &samples,
                            int original_sample_rate, int target_sample_rate);
std::vector<float> TrimSilence(const std::vector<float> &samples, float top_db);
std::vector<float> CleanOnset(const std::vector<float> &samples,
                              int sample_rate);

} // namespace dots_tts
