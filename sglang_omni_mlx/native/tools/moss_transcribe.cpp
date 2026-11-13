// SPDX-License-Identifier: Apache-2.0
// Transcribes WAV files with the native MOSS-Transcribe-Diarize runtime and
// prints one JSON line each, for parity checks against Voxt's Swift backend.
//
//   moss_transcribe --model-path DIR [--prompt TEXT] [--max-new-tokens N]
//     [--stop-at-end-of-text] [--stop-on-token-loop] a.wav...
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/audio.h"
#include "common/transcription.h"
#include "moss/moss_transcriber.h"
#include "nlohmann/json.hpp"

int main(int argc, char **argv) {
  std::string model_path;
  std::string draft_model_path;
  int draft_quantization_bits = 0;
  moss::MossOptions options;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto value = [&]() {
      if (i + 1 >= argc) {
        std::cerr << argv[0] << ": " << argument << " needs a value\n";
        std::exit(2);
      } else {
      }
      return std::string(argv[++i]);
    };
    if (argument == "--model-path") {
      model_path = value();
    } else if (argument == "--draft-model-path") {
      draft_model_path = value();
    } else if (argument == "--draft-bits") {
      draft_quantization_bits = std::stoi(value());
    } else if (argument == "--prompt") {
      options.prompt = value();
    } else if (argument == "--max-new-tokens") {
      options.max_new_tokens = std::stoi(value());
    } else if (argument == "--stop-at-end-of-text") {
      options.stop_at_end_of_text = true;
    } else if (argument == "--stop-on-token-loop") {
      options.stop_on_token_loop = true;
    } else if (argument.rfind("--", 0) == 0) {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    } else {
      files.push_back(argument);
    }
  }
  if (draft_quantization_bits != 0 && draft_quantization_bits != 4 &&
      draft_quantization_bits != 6 && draft_quantization_bits != 8) {
    throw std::invalid_argument("--draft-bits must be 4, 6 or 8");
  } else if (draft_quantization_bits != 0 && draft_model_path.empty()) {
    throw std::invalid_argument("--draft-bits needs --draft-model-path");
  } else if (options.max_new_tokens.value_or(0) < 0) {
    throw std::invalid_argument("max_new_tokens must not be negative");
  } else {
  }
  const auto load_started = std::chrono::steady_clock::now();
  const moss::MossTranscriber transcriber(model_path, draft_model_path,
                                          draft_quantization_bits);
  std::cerr << "loaded in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             load_started)
                   .count()
            << " s\n";
  const std::atomic<bool> cancel(false);
  for (const auto &file : files) {
    std::ifstream stream(file, std::ios::binary);
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    const auto started = std::chrono::steady_clock::now();
    const auto result = transcriber.Transcribe(
        omni_common::DecodeWav(bytes.str()), options, cancel);
    const nlohmann::ordered_json line = {
        {"file", file},
        {"text", result.text},
        {"language", result.language.has_value()
                         ? nlohmann::ordered_json(*result.language)
                         : nlohmann::ordered_json()},
        {"generated_token_count", result.generated_token_count},
        {"finish_reason", omni_common::FinishReasonName(result.finish_reason)},
        {"verify_rounds", result.verify_round_count},
        {"segments", omni_common::SpeakerSegmentsJson(result.segments)},
        {"seconds", std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started)
                        .count()},
    };
    std::cout << line.dump() << std::endl;
  }
  return 0;
}
