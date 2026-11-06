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
    bool use_mtp = false;
    int context_tokens = ming3_audio::kDefaultContextTokens;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help") {
        std::cout
            << "ming3_audio_chat --model-path DIR --audio WAV --prompt TEXT "
               "[--system TEXT] [--max-new-tokens N] [--context-length N] "
               "[--mtp 0|1]\n";
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
      } else if (option == "--context-length") {
        context_tokens = ming3_audio::ParseContextLength(value);
      } else if (option == "--mtp") {
        if (value != "0" && value != "1") {
          throw std::invalid_argument("--mtp must be 0 or 1");
        } else {
          use_mtp = value == "1";
        }
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
    const ming3_audio::Model model(directory, context_tokens, use_mtp);
    const auto cancelled = [] { return false; };
    const auto started = std::chrono::steady_clock::now();
    const auto result =
        model.Generate(model.Prepare(std::move(request)), cancelled);
    std::cout << nlohmann::json(
                     {{"text", result.text},
                      {"finish_reason", result.finish_reason},
                      {"prompt_tokens", result.prompt_tokens},
                      {"completion_tokens", result.completion_tokens},
                      {"verify_round_count", result.verify_round_count},
                      {"proposed_draft_token_count",
                       result.proposed_draft_token_count},
                      {"accepted_draft_token_count",
                       result.accepted_draft_token_count},
                      {"seconds",
                       std::chrono::duration<double>(
                           std::chrono::steady_clock::now() - started)
                           .count()}})
                     .dump()
              << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
