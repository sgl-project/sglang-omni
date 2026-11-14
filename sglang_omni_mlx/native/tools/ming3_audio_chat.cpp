// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <utility>

#include "common/audio.h"
#include "ming3_audio/ming3_audio_model.h"

int main(int argc, char **argv) {
  try {
    std::filesystem::path directory, audio_path;
    ming3_audio::ChatRequest request;
    int context_tokens = ming3_audio::kDefaultContextTokens;
    auto speculative_algorithm = ming3_audio::SpeculativeAlgorithm::kNone;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help") {
        std::cout
            << "ming3_audio_chat --model-path DIR --audio WAV --prompt TEXT "
               "[--system TEXT] [--max-new-tokens N] [--context-length N] "
               "[--speculative-algorithm NEXTN]\n";
        return 0;
      } else if (i + 1 >= argc) {
        throw std::invalid_argument(option + " needs a value");
      } else {
      }
      const std::string value = argv[++i];
      if (option == "--model-path") {
        directory = value;
      } else if (option == "--audio") {
        audio_path = value;
      } else if (option == "--prompt") {
        request.after_audio = value;
      } else if (option == "--system") {
        request.system_prompt = value;
      } else if (option == "--speculative-algorithm") {
        speculative_algorithm = ming3_audio::ParseSpeculativeAlgorithm(value);
      } else if (option == "--context-length") {
        context_tokens = ming3_audio::ParseContextLength(value);
      } else if (option == "--max-new-tokens") {
        size_t consumed = 0;
        request.max_new_tokens = std::stoi(value, &consumed);
        if (consumed != value.size()) {
          throw std::invalid_argument("--max-new-tokens must be an integer");
        } else {
        }
      } else {
        throw std::invalid_argument("unknown option " + option);
      }
    }
    if (directory.empty() || audio_path.empty() ||
        request.after_audio.empty()) {
      throw std::invalid_argument(
          "--model-path, --audio and --prompt are required");
    } else {
    }
    std::ifstream input(audio_path, std::ios::binary);
    if (!input) {
      throw std::invalid_argument("cannot read audio file");
    } else {
    }
    std::ostringstream bytes;
    bytes << input.rdbuf();
    request.samples = omni_common::DecodeWav(bytes.str());
    const ming3_audio::Model model(directory, context_tokens,
                                   speculative_algorithm);
    const auto cancelled = [] { return false; };
    const auto started = std::chrono::steady_clock::now();
    const auto result =
        model.Generate(model.Prepare(std::move(request)), cancelled);
    auto output = nlohmann::json(
        {{"text", result.text},
         {"finish_reason", result.finish_reason},
         {"prompt_tokens", result.prompt_tokens},
         {"completion_tokens", result.completion_tokens},
         {"seconds", std::chrono::duration<double>(
                         std::chrono::steady_clock::now() - started)
                         .count()}});
    if (speculative_algorithm == ming3_audio::SpeculativeAlgorithm::kNextN) {
      output["draft_tokens"] = result.draft_tokens;
      output["accepted_draft_tokens"] = result.accepted_draft_tokens;
    } else {
    }
    std::cout << output.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
