// SPDX-License-Identifier: Apache-2.0
// Ming-Omni-TTS's /v1/audio/speech fields as CUDA's CreateSpeechRequest and
// Ming request builder read them, apart from the model: ming_tts_server and
// stub_server's stub_ming_tts both parse requests with it.
#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

namespace ming_tts {

// AudioVAE's sample rate, which the checkpoint's config must name, and the
// one CampPlus takes.
inline constexpr int kAudioSampleRate = 44100;
inline constexpr int kSpeakerSampleRate = 16000;
// Room for 30 s of reference as base64 mono float32 WAV at up to 192 kHz
// (29 MiB).
inline constexpr size_t kMaxRequestBodyBytes = 32 * 1024 * 1024;

// A reference recording at the rates its two encoders take.
struct ReferenceAudio {
  std::vector<float> samples_16k;
  std::vector<float> samples_44k;
};

struct SpeechRequest {
  // The input and the reference transcript, stripped as CUDA strips them.
  std::string text;
  std::optional<std::string> reference_text;
  std::optional<ReferenceAudio> reference_audio;
  int max_patches = 0;
  float cfg = 0, sigma = 0, temperature = 0;
};

// Throws std::invalid_argument for a request to answer with 400. Fields the
// CUDA server ignores are ignored, prompt among them.
SpeechRequest ParseSpeechRequest(const nlohmann::ordered_json &body);

} // namespace ming_tts
