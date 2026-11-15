// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <vector>

namespace omni_common {

std::vector<float> ResampleAudio(const std::vector<float> &samples,
                                 int source_sample_rate,
                                 int target_sample_rate);

} // namespace omni_common
