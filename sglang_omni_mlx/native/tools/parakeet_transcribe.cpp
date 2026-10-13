// SPDX-License-Identifier: Apache-2.0
// Transcribes WAV files with the native Parakeet TDT runtime and prints one
// JSON line each, for parity checks against Voxt's Swift backend.
//
//   parakeet_transcribe --model-path DIR [--chunk-duration S] a.wav...
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "audio.h"
#include "nlohmann/json.hpp"
#include "parakeet_transcriber.h"

int main(int argc, char **argv) {
  std::string model_path;
  parakeet::ParakeetOptions options;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument.rfind("--", 0) == 0 &&
        (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)) {
      std::cerr << argv[0] << ": missing value for " << argument << "\n";
      return 2;
    } else {
    }
    const auto value = [&]() { return std::string(argv[++i]); };
    if (argument == "--model-path") {
      model_path = value();
    } else if (argument == "--chunk-duration") {
      options.chunk_duration_seconds = std::stod(value());
    } else if (argument.rfind("--", 0) == 0) {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    } else {
      files.push_back(argument);
    }
  }
  const auto load_started = std::chrono::steady_clock::now();
  const parakeet::ParakeetTranscriber transcriber(model_path);
  std::cerr << "loaded in "
            << std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                             load_started)
                   .count()
            << " s\n";
  const std::atomic<bool> cancel(false);
  for (const std::string &file : files) {
    std::ifstream stream(file, std::ios::binary);
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    const auto started = std::chrono::steady_clock::now();
    const qwen3_asr::TranscriptionResult result = transcriber.Transcribe(
        qwen3_asr::DecodeWav(bytes.str()), options, cancel);
    const nlohmann::json line = {
        {"file", file},
        {"text", result.text},
        {"language", nlohmann::json()},
        {"generated_token_count", result.generated_token_count},
        {"finish_reason", qwen3_asr::FinishReasonName(result.finish_reason)},
        {"segments", qwen3_asr::SpeakerSegmentsJson(result.segments)},
        {"seconds", std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started)
                        .count()},
    };
    std::cout << line.dump() << std::endl;
  }
  return 0;
}
