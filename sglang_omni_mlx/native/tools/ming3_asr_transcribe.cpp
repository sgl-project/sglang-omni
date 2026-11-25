#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/audio.h"
#include "ming3_asr/ming3_asr_transcriber.h"

int main(int argc, char **argv) {
  try {
    std::string model_path;
    std::string draft_model_path;
    int draft_quantization_bits = 0;
    omni_common::DraftPolicy draft_policy = omni_common::DraftPolicy::kAdaptive;
    ming3_asr::MingOptions options;
    std::vector<std::string> files;
    for (int index = 1; index < argc; ++index) {
      const std::string argument = argv[index];
      if (argument == "--help") {
        std::cout << "Usage: ming3_asr_transcribe --model-path DIR [--language "
                     "en|zh] "
                     "[--max-new-tokens N] [--draft-model-path DIR] "
                     "[--draft-bits 4|6|8] [--draft-policy adaptive|fixed] "
                     "file.wav...\n";
        return 0;
      } else if (argument.starts_with("--")) {
        if (index + 1 >= argc) {
          throw std::invalid_argument("missing value for " + argument);
        } else {
        }
        const std::string value = argv[++index];
        if (argument == "--model-path") {
          model_path = value;
        } else if (argument == "--language") {
          options.language = value;
        } else if (argument == "--max-new-tokens") {
          size_t consumed = 0;
          options.max_new_tokens = std::stoi(value, &consumed);
          if (consumed != value.size()) {
            throw std::invalid_argument("max-new-tokens must be an integer");
          } else {
          }
        } else if (argument == "--draft-model-path") {
          draft_model_path = value;
        } else if (argument == "--draft-bits") {
          draft_quantization_bits = std::stoi(value);
        } else if (argument == "--draft-policy") {
          draft_policy = omni_common::ParseDraftPolicy(value);
        } else {
          throw std::invalid_argument("unknown option " + argument);
        }
      } else {
        files.push_back(argument);
      }
    }
    if (draft_quantization_bits != 0 && draft_quantization_bits != 4 &&
        draft_quantization_bits != 6 && draft_quantization_bits != 8) {
      throw std::invalid_argument("--draft-bits must be 4, 6 or 8");
    } else if (draft_quantization_bits != 0 && draft_model_path.empty()) {
      throw std::invalid_argument("--draft-bits needs --draft-model-path");
    } else if (model_path.empty() || files.empty()) {
      throw std::invalid_argument(
          "--model-path and at least one WAV file are required");
    } else {
    }
    const auto load_started = std::chrono::steady_clock::now();
    const ming3_asr::MingTranscriber transcriber(
        model_path, draft_model_path, draft_quantization_bits, draft_policy);
    const double load_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      load_started)
            .count();
    const std::atomic<bool> cancel(false);
    for (const auto &file : files) {
      std::ifstream input(file, std::ios::binary);
      if (!input) {
        throw std::runtime_error("cannot open " + file);
      } else {
      }
      std::ostringstream bytes;
      bytes << input.rdbuf();
      auto samples = omni_common::DecodeWav(bytes.str());
      const double audio_seconds =
          static_cast<double>(samples.size()) / omni_common::kSampleRate;
      const auto started = std::chrono::steady_clock::now();
      const auto request = transcriber.Prepare(std::move(samples), options);
      const auto result = transcriber.Transcribe(request, cancel);
      const nlohmann::json output = {
          {"file", file},
          {"text", result.text},
          {"language", result.language.has_value()
                           ? nlohmann::json(*result.language)
                           : nlohmann::json()},
          {"generated_token_count", result.generated_token_count},
          {"finish_reason",
           omni_common::FinishReasonName(result.finish_reason)},
          {"verify_round_count", result.verify_round_count},
          {"proposed_draft_token_count", result.proposed_draft_token_count},
          {"accepted_draft_token_count", result.accepted_draft_token_count},
          {"load_seconds", load_seconds},
          {"peak_memory_bytes", mlx::core::get_peak_memory()},
          {"audio_seconds", audio_seconds},
          {"seconds", std::chrono::duration<double>(
                          std::chrono::steady_clock::now() - started)
                          .count()}};
      std::cout << output.dump() << std::endl;
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << argv[0] << ": " << error.what() << '\n';
    return 1;
  }
}
