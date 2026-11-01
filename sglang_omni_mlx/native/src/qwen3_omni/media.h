// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen3_omni/encoders.h"
#include "qwen3_omni/request.h"

namespace qwen3_omni {
inline constexpr size_t MAXIMUM_MEDIA_BYTES = 32 * 1024 * 1024;
struct VisualInput {
  mx::array patches;
  VisualGrid grid;
  double sampled_fps;
};
std::vector<uint8_t> media_bytes(const std::string &source,
                                 const Cancellation &cancelled);
std::vector<float> decode_audio(const std::vector<uint8_t> &bytes,
                                const Cancellation &cancelled);
std::vector<float> audio_samples(const std::string &source,
                                 const Cancellation &cancelled);
VisualInput image_pixels(const std::string &source,
                         const Cancellation &cancelled);
VisualInput video_pixels(const std::string &source, const ChatRequest &request,
                         const Cancellation &cancelled);
mx::array patchify(const std::vector<float> &pixels, int frames, int height,
                   int width);
mx::array audio_mel(const std::vector<float> &samples);
std::vector<uint8_t> encode_wav(const std::vector<float> &samples);
} // namespace qwen3_omni
