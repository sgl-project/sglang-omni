// SPDX-License-Identifier: Apache-2.0
// Native C++ MLX Voxtral TTS server.

#include <atomic>
#include <filesystem>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/audio.h"
#include "common/tts_service.h"
#include "voxtral_tts/config.h"
#include "voxtral_tts/model.h"
#include "voxtral_tts/speech_tokenizer.h"

namespace {

constexpr const char *kModelKind = "voxtral_tts";
constexpr const char *kDefaultVoice = "cheerful_female";

struct LoadedVoxtral {
  explicit LoadedVoxtral(const std::filesystem::path &model_directory)
      : config(voxtral_tts::VoxtralConfig::Load(model_directory)),
        model(config), tokenizer(config.checkpoint_directory) {
    for (const auto &voice_entry : config.voices) {
      voice_embeddings.emplace(voice_entry.first,
                               model.LoadVoiceEmbedding(voice_entry.first));
    }
  }

  voxtral_tts::VoxtralConfig config;
  voxtral_tts::VoxtralModel model;
  voxtral_tts::SpeechTokenizer tokenizer;
  std::map<std::string, mlx::core::array> voice_embeddings;
};

omni_server::TTSModel
LoadVoxtral(const std::filesystem::path &model_directory) {
  auto loaded = std::make_shared<LoadedVoxtral>(model_directory);
  omni_server::TTSModel result;
  for (const auto &voice_entry : loaded->config.voices) {
    result.voices.insert(voice_entry.first);
  }
  if (result.voices.empty()) {
    throw std::runtime_error("Voxtral checkpoint has no configured voices");
  } else {
  }
  result.default_voice = result.voices.contains(kDefaultVoice)
                             ? kDefaultVoice
                             : *result.voices.begin();
  result.synthesize = [loaded](const omni_server::SynthesisRequest &request,
                               const std::atomic<bool> &cancelled) {
    const std::vector<int> prompt_token_ids =
        loaded->tokenizer.EncodeSpeechRequest(request.text, request.voice);
    const voxtral_tts::GeneratedCodes generated_codes = loaded->model.Generate(
        prompt_token_ids, loaded->voice_embeddings.at(request.voice),
        request.max_new_tokens, request.seed, cancelled);
    const std::vector<float> samples =
        loaded->model.DecodeAudio(generated_codes.frames, cancelled);
    return omni_server::SynthesisResult{
        qwen3_asr::EncodeWav(
            samples, loaded->model.config().audio_tokenizer.sampling_rate_hz),
        generated_codes.finish_reason};
  };
  return result;
}

} // namespace

int main(int argument_count, char **argument_values) {
  const omni_server::TTSKind kind{kModelKind, LoadVoxtral};
  return omni_server::ServeTTS(argument_count, argument_values, kind);
}
