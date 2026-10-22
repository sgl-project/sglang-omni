// SPDX-License-Identifier: Apache-2.0
// Encodes Voxtral speech prompts for token parity checks.
#include <iostream>
#include <stdexcept>
#include <string>

#include "nlohmann/json.hpp"
#include "voxtral_tts/speech_tokenizer.h"

int main(int argc, char **argv) {
  std::string checkpoint_directory;
  std::string voice;
  std::string text;
  for (int argument_index = 1; argument_index < argc; ++argument_index) {
    const std::string argument = argv[argument_index];
    const auto read_value = [&]() -> std::string {
      if (argument_index + 1 >= argc) {
        throw std::invalid_argument(argument + " needs a value");
      } else {
      }
      return argv[++argument_index];
    };
    if (argument == "--model-path") {
      checkpoint_directory = read_value();
    } else if (argument == "--voice") {
      voice = read_value();
    } else if (argument == "--text") {
      text = read_value();
    } else {
      throw std::invalid_argument("unknown argument " + argument);
    }
  }
  if (checkpoint_directory.empty() || voice.empty()) {
    throw std::invalid_argument("--model-path and --voice are required");
  } else {
  }
  const voxtral_tts::SpeechTokenizer tokenizer(checkpoint_directory);
  std::cout << nlohmann::json({{"prompt_token_ids",
                                tokenizer.EncodeSpeechRequest(text, voice)}})
                   .dump()
            << "\n";
  return 0;
}
