// SPDX-License-Identifier: Apache-2.0
// Transcribes WAV files with the native Whisper runtime and prints one JSON
// line each, for parity checks against Voxt's Swift backend.
//
//   whisper_transcribe --model-path DIR [--language L] [--max-new-tokens N]
//     [--temperature T] a.wav...
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "audio.h"
#include "nlohmann/json.hpp"
#include "whisper_transcriber.h"

int main(int argc, char **argv) {
  std::string model_path;
  whisper::WhisperOptions options;
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
    } else if (argument == "--language") {
      options.language = value();
    } else if (argument == "--max-new-tokens") {
      options.max_new_tokens = std::stoi(value());
    } else if (argument == "--temperature") {
      options.temperature = std::stof(value());
    } else if (argument.rfind("--", 0) == 0) {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    } else {
      files.push_back(argument);
    }
  }
  const auto load_started = std::chrono::steady_clock::now();
  const whisper::WhisperTranscriber transcriber(model_path);
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
        qwen3_asr::DecodeWav(bytes.str()), options, cancel);
    const nlohmann::json line = {
        {"file", file},
        {"text", result.text},
        {"language", result.language.has_value()
                         ? nlohmann::json(*result.language)
                         : nlohmann::json()},
        {"generated_token_count", result.generated_token_count},
        {"finish_reason", qwen3_asr::FinishReasonName(result.finish_reason)},
        {"seconds", std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started)
                        .count()},
    };
    std::cout << line.dump() << std::endl;
  }
  return 0;
}
