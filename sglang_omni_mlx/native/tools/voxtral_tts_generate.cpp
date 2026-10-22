// SPDX-License-Identifier: Apache-2.0
// Generate Voxtral acoustic code frames for native/reference parity checks.
#include <atomic>
#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"
#include "voxtral_tts/config.h"
#include "voxtral_tts/model.h"
#include "voxtral_tts/speech_tokenizer.h"

int main(int argument_count, char **argument_values) {
  std::string model_path;
  std::string text;
  std::string voice;
  uint64_t seed = 0;
  int max_new_tokens = 4096;
  for (int argument_index = 1; argument_index < argument_count;
       ++argument_index) {
    const std::string flag = argument_values[argument_index];
    const auto read_value = [&]() {
      if (argument_index + 1 >= argument_count) {
        throw std::invalid_argument(flag + " needs a value");
      } else {
      }
      return std::string(argument_values[++argument_index]);
    };
    if (flag == "--model-path") {
      model_path = read_value();
    } else if (flag == "--text") {
      text = read_value();
    } else if (flag == "--voice") {
      voice = read_value();
    } else if (flag == "--seed") {
      seed = std::stoull(read_value());
    } else if (flag == "--max-new-tokens") {
      max_new_tokens = std::stoi(read_value());
    } else {
      throw std::invalid_argument("unknown argument " + flag);
    }
  }
  if (model_path.empty() || voice.empty() || text.empty() ||
      max_new_tokens < 1) {
    throw std::invalid_argument(
        "--model-path, --voice, nonempty --text and positive --max-new-tokens "
        "are required");
  } else {
  }
  const voxtral_tts::VoxtralConfig config =
      voxtral_tts::VoxtralConfig::Load(model_path);
  const voxtral_tts::VoxtralModel model(config);
  const voxtral_tts::SpeechTokenizer tokenizer(model_path);
  const std::vector<int> prompt_token_ids =
      tokenizer.EncodeSpeechRequest(text, voice);
  const mlx::core::array voice_embedding = model.LoadVoiceEmbedding(voice);
  const std::atomic<bool> cancelled(false);
  const voxtral_tts::GeneratedCodes result = model.Generate(
      prompt_token_ids, voice_embedding, max_new_tokens, seed, cancelled);
  std::cout << nlohmann::json({{"frames", result.frames},
                               {"finish_reason", result.finish_reason}})
                   .dump()
            << "\n";
  return 0;
}
