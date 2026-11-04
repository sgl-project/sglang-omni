// SPDX-License-Identifier: Apache-2.0
// Fun-CosyVoice3's reference-audio front end: WAV decoding, the resampler the
// CUDA server uses, and the flow's prompt mel.
#pragma once

#include <string_view>
#include <vector>

#include "mlx/mlx.h"

namespace cosyvoice3 {

// The longest reference the speech tokenizer takes, as on CUDA.
inline constexpr double kMaxReferenceSeconds = 30.0;
// The shortest reference the prompt mel's reflect padding takes, with margin.
inline constexpr double kMinReferenceSeconds = 0.5;

struct DecodedAudio {
  std::vector<float> samples;
  int sample_rate = 0;
};

// A WAV file held in memory (8, 16, 24 or 32-bit PCM, 32 or 64-bit float,
// plain or WAVE_FORMAT_EXTENSIBLE) as mono float32 samples, channels
// averaged. Throws std::invalid_argument for anything else, and for audio
// outside [kMinReferenceSeconds, kMaxReferenceSeconds], before decoding it.
DecodedAudio DecodeWav(std::string_view file_bytes);

// torchaudio.functional.resample with its defaults: a Hann-windowed sinc with
// six zero crossings, rolloff 0.99. Throws std::invalid_argument for a rate
// pair whose filter would be impractically large.
std::vector<float> Resample(const std::vector<float> &samples, int from_rate,
                            int to_rate);

// The flow's prompt mel of 24 kHz samples, as Matcha-TTS computes it: a
// reflect-padded 1920-point STFT every 480 samples under a periodic Hann
// window, magnitude, 80 Slaney mel bins to 12 kHz, natural log; float32
// [1, frames, 80].
mlx::core::array PromptMel(const std::vector<float> &samples_24k);

} // namespace cosyvoice3
