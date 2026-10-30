// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#include "common/audio.h"
#include "ming3_audio/ming3_audio_model.h"

int main(int argc, char **argv) {
  try {
    std::filesystem::path directory, audio_path, trace_directory;
    ming3_audio::ChatRequest request;
    for (int i = 1; i < argc; ++i) {
      const std::string option = argv[i];
      if (option == "--help") {
        std::cout
            << "ming3_audio_chat --model-path DIR --audio WAV --prompt TEXT "
               "[--system TEXT] [--max-new-tokens N] [--trace-directory DIR]\n";
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
      } else if (option == "--max-new-tokens") {
        size_t consumed = 0;
        request.max_new_tokens = std::stoi(value, &consumed);
        if (consumed != value.size()) {
          throw std::invalid_argument("--max-new-tokens must be an integer");
        } else {
        }
      } else if (option == "--trace-directory") {
        trace_directory = value;
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
    const ming3_audio::Model model(directory);
    const auto cancelled = [] { return false; };
    omni_common::WeightMap trace;
    const auto started = std::chrono::steady_clock::now();
    const auto result = model.Generate(
        request, cancelled, trace_directory.empty() ? nullptr : &trace);
    if (!trace_directory.empty()) {
      std::filesystem::create_directories(trace_directory);
      mlx::core::save_safetensors(
          (trace_directory / "native.safetensors").string(), trace);
    } else {
    }
    std::cout << nlohmann::json(
                     {{"text", result.text},
                      {"finish_reason", result.finish_reason},
                      {"prompt_tokens", result.prompt_tokens},
                      {"completion_tokens", result.completion_tokens},
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
