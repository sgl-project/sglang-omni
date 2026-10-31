// SPDX-License-Identifier: Apache-2.0
// Transcribes WAV files with the native runtime and prints one JSON line each,
// for parity checks against the Python reference server.
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/audio.h"
#include "common/tokenizer.h"
#include "nlohmann/json.hpp"
#include "qwen3_asr/kernels/kernels.h"
#include "qwen3_asr/qwen3_asr_model.h"
#include "qwen3_asr/qwen3_asr_transcriber.h"

int main(int argc, char **argv) {
  using qwen3_asr::AudioLayout;
  std::string model_path;
  std::string encode_text;
  std::string dump_mel_filters;
  std::string encode_lines;
  bool encode = false;
  bool profile = false;
  qwen3_asr::TranscriptionOptions options;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    const auto value = [&]() -> std::string {
      if (i + 1 >= argc)
        throw std::invalid_argument(argument + " needs a value");
      return argv[++i];
    };
    if (argument == "--model-path") {
      model_path = value();
    } else if (argument == "--layout") {
      options.layout = value() == "voxt_swift" ? AudioLayout::kVoxtSwift
                                               : AudioLayout::kReference;
    } else if (argument == "--language") {
      options.language = qwen3_asr::NormalizeLanguage(value());
    } else if (argument == "--kernels") {
      qwen3_asr::SetKernelSet(qwen3_asr::ParseKernelSet(value()));
    } else if (argument == "--stop-at-end-of-text") {
      options.stop_at_end_of_text = true;
    } else if (argument == "--stop-on-token-loop") {
      options.stop_on_token_loop = true;
    } else if (argument == "--max-new-tokens") {
      options.max_new_tokens = std::stoi(value());
    } else if (argument == "--encode") {
      encode = true;
      encode_text = value();
    } else if (argument == "--encode-lines") {
      encode_lines = value();
    } else if (argument == "--profile") {
      profile = true;
    } else if (argument == "--dump-mel-filters") {
      dump_mel_filters = value();
    } else {
      files.push_back(argument);
    }
  }
  if (!dump_mel_filters.empty()) {
    std::ofstream out(dump_mel_filters, std::ios::binary);
    const auto &filters = omni_common::MelFilterBank();
    const auto &window = omni_common::PeriodicHannWindow();
    out.write(reinterpret_cast<const char *>(filters.data()),
              filters.size() * sizeof(float));
    out.write(reinterpret_cast<const char *>(window.data()),
              window.size() * sizeof(float));
    return 0;
  } else if (!encode_lines.empty()) {
    const omni_common::Tokenizer tokenizer(model_path);
    std::ifstream lines(encode_lines);
    std::string line;
    while (std::getline(lines, line)) {
      const std::string text = nlohmann::json::parse(line).get<std::string>();
      const std::vector<int> ids = tokenizer.Encode(text);
      std::cout << nlohmann::json(
                       {{"ids", ids},
                        {"decoded", tokenizer.Decode(ids, false)},
                        {"decoded_skip", tokenizer.Decode(ids, true)}})
                       .dump()
                << "\n";
    }
    return 0;
  } else if (encode) {
    const omni_common::Tokenizer tokenizer(model_path);
    std::cout << nlohmann::json(tokenizer.Encode(encode_text)).dump() << "\n";
    return 0;
  } else {
  }
  const auto load_started = std::chrono::steady_clock::now();
  const qwen3_asr::Qwen3ASRTranscriber transcriber(model_path);
  const double load_seconds =
      std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                    load_started)
          .count();
  const size_t model_memory_bytes = mlx::core::get_active_memory();
  std::cerr << "loaded in " << load_seconds << " s\n";
  const std::atomic<bool> cancel(false);
  for (const auto &file : files) {
    std::ifstream stream(file, std::ios::binary);
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    qwen3_asr::TranscriptionProfile timings;
    mlx::core::synchronize();
    mlx::core::reset_peak_memory();
    const auto started = std::chrono::steady_clock::now();
    const auto result =
        transcriber.Transcribe(omni_common::DecodeWav(bytes.str()), options,
                               cancel, profile ? &timings : nullptr);
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
    if (profile) {
      line["profile"] = {
          {"load_seconds", load_seconds},
          {"model_memory_bytes", model_memory_bytes},
          {"mel_seconds", timings.mel_seconds},
          {"encoder_seconds", timings.encoder_seconds},
          {"prefill_seconds", timings.prefill_seconds},
          {"decode_seconds", timings.decode_seconds},
          {"prompt_tokens", timings.prompt_tokens},
          {"peak_memory_bytes", mlx::core::get_peak_memory()},
      };
    } else {
    }
    std::cout << line.dump() << std::endl;
    // Note (Jiaxin Deng): as the server does between requests; buffers of one
    // clip's sizes rarely fit the next, so the cache would grow without end.
    mlx::core::clear_cache();
  }
  return 0;
}
