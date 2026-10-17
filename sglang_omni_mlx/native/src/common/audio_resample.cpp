// SPDX-License-Identifier: Apache-2.0
#include "common/audio_resample.h"

#include <cmath>
#include <stdexcept>

#include "soxr.h"

namespace qwen3_asr {

std::vector<float> ResampleAudio(const std::vector<float> &samples,
                                 int source_sample_rate,
                                 int target_sample_rate) {
  if (source_sample_rate <= 0 || target_sample_rate <= 0 || samples.empty()) {
    throw std::invalid_argument("resampling requires audio and positive rates");
  } else if (source_sample_rate == target_sample_rate) {
    return samples;
  } else {
    const size_t output_length =
        static_cast<size_t>(std::ceil(static_cast<double>(samples.size()) *
                                      target_sample_rate / source_sample_rate));
    std::vector<float> output(output_length, 0.0f);
    size_t consumed = 0, produced = 0;
    const auto quality = soxr_quality_spec(SOXR_HQ, 0);
    const auto runtime = soxr_runtime_spec(1);
    const auto error =
        soxr_oneshot(source_sample_rate, target_sample_rate, 1, samples.data(),
                     samples.size(), &consumed, output.data(), output.size(),
                     &produced, nullptr, &quality, &runtime);
    if (error || consumed != samples.size()) {
      throw std::runtime_error(
          error ? error : "audio resampling did not consume input");
    } else {
      return output;
    }
  }
}

} // namespace qwen3_asr
