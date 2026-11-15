// SPDX-License-Identifier: Apache-2.0
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <memory>
#include <regex>
#include <stdexcept>

#include "music3/music3_model.h"

int main(int argc, char **argv) {
  try {
    std::filesystem::path directory, requests, output;
    bool prompt_only = false;
    bool should_check_cancellation = false;
    bool is_serial_offload_enabled = false;
    for (int i = 1; i < argc; ++i) {
      const std::string flag = argv[i];
      if (flag == "--prompt-only") {
        prompt_only = true;
      } else if (flag == "--check-cancellation") {
        should_check_cancellation = true;
      } else if (i + 1 >= argc) {
        throw std::invalid_argument(flag + " needs a value");
      } else if (flag == "--model-path") {
        directory = argv[++i];
      } else if (flag == "--stage-offload-components") {
        music3::ValidateSerialOffload(argv[++i]);
        is_serial_offload_enabled = true;
      } else if (flag == "--requests") {
        requests = argv[++i];
      } else if (flag == "--output-directory") {
        output = argv[++i];
      } else {
        throw std::invalid_argument("unknown argument " + flag);
      }
    }
    if (directory.empty() || requests.empty() ||
        (!prompt_only && output.empty())) {
      throw std::invalid_argument(
          "usage: music3_synthesize --model-path DIR "
          "--requests JSONL [--prompt-only | --output-directory DIR] "
          "[--stage-offload-components ar,dit] [--check-cancellation]");
    } else {
    }
    mlx::core::set_cache_limit(1024ULL * 1024 * 1024);
    std::unique_ptr<music3::Model> model;
    std::unique_ptr<omni_common::Tokenizer> prompt_tokenizer;
    if (prompt_only) {
      prompt_tokenizer =
          std::make_unique<omni_common::Tokenizer>(directory / "tokenizer");
    } else {
      model =
          std::make_unique<music3::Model>(directory, is_serial_offload_enabled);
      std::filesystem::create_directories(output);
    }
    std::ifstream input(requests);
    if (!input) {
      throw std::runtime_error("cannot read " + requests.string());
    } else {
    }
    std::atomic<bool> cancelled{false};
    std::string line;
    while (std::getline(input, line)) {
      auto body = nlohmann::json::parse(line);
      const std::string identifier = body.at("id");
      if (!std::regex_match(identifier, std::regex("[A-Za-z0-9_-]{1,128}"))) {
        throw std::invalid_argument(
            "id must contain only letters, digits, hyphens and underscores");
      } else {
      }
      if (prompt_only) {
        const std::string prompt =
            music3::BuildPrompt(body.at("instructions").get<std::string>(),
                                body.at("input").get<std::string>());
        std::cout << nlohmann::json(
                         {{"id", identifier},
                          {"prompt", prompt},
                          {"token_ids", prompt_tokenizer->Encode(prompt)}})
                         .dump()
                  << std::endl;
        continue;
      } else {
      }
      const auto request = music3::ParseRequest(
          body, model->tokenizer(), "minimax_music3", model->context_tokens());
      const size_t idle_active_memory_bytes = mlx::core::get_active_memory();
      bool is_cancelled_after_ar = false;
      size_t cancelled_active_memory_bytes = 0;
      if (should_check_cancellation) {
        try {
          model->Synthesize(request, cancelled,
                            [&]() { cancelled.store(true); });
        } catch (const music3::Cancelled &) {
          is_cancelled_after_ar = true;
        }
        cancelled.store(false);
        cancelled_active_memory_bytes = mlx::core::get_active_memory();
      } else {
      }
      mlx::core::reset_peak_memory();
      const auto started = std::chrono::steady_clock::now();
      double ar_elapsed_seconds = 0;
      const auto result = model->Synthesize(request, cancelled, [&]() {
        ar_elapsed_seconds = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
      });
      const double elapsed = std::chrono::duration<double>(
                                 std::chrono::steady_clock::now() - started)
                                 .count();
      const auto path = output / (identifier + ".wav");
      const auto bytes = music3::EncodeAudio(result.samples, false);
      std::ofstream audio(path, std::ios::binary);
      audio.write(bytes.data(), bytes.size());
      audio.close();
      if (!audio) {
        throw std::runtime_error("cannot write " + path.string());
      } else {
      }
      nlohmann::json synthesis_result = {
          {"id", identifier},
          {"file", path.string()},
          {"generated_frames", result.frames},
          {"sample_count", result.samples.size() / 2},
          {"sample_rate", music3::kSampleRate},
          {"elapsed_s", elapsed},
          {"ar_elapsed_s", ar_elapsed_seconds},
          {"peak_memory_bytes", mlx::core::get_peak_memory()}};
      if (should_check_cancellation) {
        synthesis_result["cancelled_after_ar"] = is_cancelled_after_ar;
        synthesis_result["idle_active_memory_bytes"] = idle_active_memory_bytes;
        synthesis_result["cancelled_active_memory_bytes"] =
            cancelled_active_memory_bytes;
      } else {
      }
      std::cout << synthesis_result.dump() << std::endl;
      mlx::core::clear_cache();
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "music3_synthesize: " << error.what() << "\n";
    return 1;
  }
}
