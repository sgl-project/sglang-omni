// SPDX-License-Identifier: Apache-2.0
// Transcribes WAV files with the native ARK-ASR runtime and prints one JSON
// line each, for parity checks against the reference implementation.
//
//   arkasr_transcribe --model-path DIR [--language L] [--max-new-tokens N]
//     [--draft-model-path DIR [--draft-bits 4|6|8]] a.wav...
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "arkasr/arkasr_transcriber.h"
#include "common/audio.h"
#include "common/generation.h"
#include "common/transcription.h"
#include "nlohmann/json.hpp"

int main(int argc, char **argv) {
  std::string model_path;
  omni_common::DraftOptions draft;
  const auto draft_flags = draft.Flags();
  arkasr::ArkOptions options;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument.rfind("--", 0) == 0 &&
        (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)) {
      std::cerr << argv[0] << ": missing value for " << argument << "\n";
      return 2;
    } else if (argument == "--model-path") {
      model_path = argv[++i];
    } else if (draft_flags.count(argument) != 0) {
      draft_flags.at(argument)(argv[++i]);
    } else if (argument == "--language") {
      options.language = argv[++i];
    } else if (argument == "--max-new-tokens") {
      options.max_new_tokens = std::stoi(argv[++i]);
    } else if (argument.rfind("--", 0) == 0) {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    } else {
      files.push_back(argument);
    }
  }
  try {
    draft.Check();
  } catch (const std::invalid_argument &error) {
    std::cerr << argv[0] << ": " << error.what() << "\n";
    return 2;
  }
  const auto load_started = std::chrono::steady_clock::now();
  const arkasr::ArkTranscriber transcriber(model_path, draft);
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
    const omni_common::TranscriptionResult result = transcriber.Transcribe(
        omni_common::DecodeWav(bytes.str()), options, cancel);
    const nlohmann::json line = {
        {"file", file},
        {"text", result.text},
        {"language", result.language.has_value()
                         ? nlohmann::json(*result.language)
                         : nlohmann::json()},
        {"generated_token_count", result.generated_token_count},
        {"finish_reason", omni_common::FinishReasonName(result.finish_reason)},
        {"verify_rounds", result.verify_round_count},
        {"proposed_draft_token_count", result.proposed_draft_token_count},
        {"accepted_draft_token_count", result.accepted_draft_token_count},
        {"seconds", std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started)
                        .count()},
    };
    std::cout << line.dump() << std::endl;
  }
  return 0;
}
