// SPDX-License-Identifier: Apache-2.0
// Fun-CosyVoice3's prompt mel.
#pragma once

#include <vector>

#include "mlx/mlx.h"

namespace cosyvoice3 {

// The flow's prompt mel of 24 kHz samples, as Matcha-TTS computes it: a
// reflect-padded 1920-point STFT every 480 samples under a periodic Hann
// window, magnitude, 80 Slaney mel bins to 12 kHz, natural log; float32
// [1, frames, 80].
mlx::core::array PromptMel(const std::vector<float> &samples_24k);

} // namespace cosyvoice3
