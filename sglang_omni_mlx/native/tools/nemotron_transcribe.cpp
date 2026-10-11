// SPDX-License-Identifier: Apache-2.0
// Transcribes WAV files with the native Nemotron 3.5 ASR streaming runtime and
// prints one JSON line each, for parity checks against Voxt's Swift backend.
//
//   nemotron_transcribe --model-path DIR [--language L] [--chunk-frames N]
//     a.wav...
//
// Without --chunk-frames, files take the Final pass (the native chunk); with
// it, they stream in chunks of N 80 ms frames, as a live session would.
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "audio.h"
#include "nemotron_transcriber.h"
#include "nlohmann/json.hpp"

int main(int argc, char **argv) {
  std::string model_path;
  nemotron::NemotronOptions options;
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
    } else if (argument == "--language") {
      options.language = value();
    } else if (argument == "--chunk-frames") {
      options.chunk_frames = std::stoi(value());
      if (*options.chunk_frames < 1) {
        std::cerr << argv[0] << ": --chunk-frames must be positive\n";
        return 2;
      } else {
      }
    } else if (argument.rfind("--", 0) == 0) {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    } else {
      files.push_back(argument);
    }
  }
  const auto load_started = std::chrono::steady_clock::now();
  const nemotron::NemotronTranscriber transcriber(model_path);
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
    const std::vector<float> samples = qwen3_asr::DecodeWav(bytes.str());
    const auto started = std::chrono::steady_clock::now();
    const qwen3_asr::TranscriptionResult result =
        transcriber.Transcribe(samples, options, cancel);
    nlohmann::json line = {
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
