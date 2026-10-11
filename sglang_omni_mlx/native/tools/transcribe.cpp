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

#include "audio.h"
#include "model.h"
#include "nlohmann/json.hpp"
#include "tokenizer.h"
#include "transcriber.h"

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
    const auto &filters = qwen3_asr::MelFilterBank();
    const auto &window = qwen3_asr::PeriodicHannWindow();
    out.write(reinterpret_cast<const char *>(filters.data()),
              filters.size() * sizeof(float));
    out.write(reinterpret_cast<const char *>(window.data()),
              window.size() * sizeof(float));
    return 0;
  } else if (!encode_lines.empty()) {
    const qwen3_asr::Tokenizer tokenizer(model_path);
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
  } else if (profile) {
    namespace mx = mlx::core;
    using Clock = std::chrono::steady_clock;
    const auto ms = [](Clock::time_point a, Clock::time_point b) {
      return std::chrono::duration<double, std::milli>(b - a).count();
    };
    const qwen3_asr::Qwen3ASR model(model_path);
    const qwen3_asr::Tokenizer tokenizer(model_path);
    std::ifstream stream(files.at(0), std::ios::binary);
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    const std::vector<float> samples = qwen3_asr::DecodeWav(bytes.str());
    for (int round = 0; round < 3; ++round) {
      const auto t0 = Clock::now();
      mx::array mel = qwen3_asr::LogMel(samples, AudioLayout::kVoxtSwift);
      mx::eval(mel);
      const auto t1 = Clock::now();
      mx::array encoded = model.EncodeAudio(mel, AudioLayout::kVoxtSwift);
      mx::eval(encoded);
      const auto t2 = Clock::now();
      std::string prompt =
          "<|im_start|>system\n<|im_end|>\n<|im_start|>user\n<|audio_start|>";
      for (int i = 0;
           i < qwen3_asr::TokenCount(mel.shape(-1), AudioLayout::kVoxtSwift);
           ++i)
        prompt += "<|audio_pad|>";
      prompt += "<|audio_end|><|im_end|>\n<|im_start|>assistant\nlanguage "
                "English<asr_text>";
      const std::vector<int> ids = tokenizer.Encode(prompt);
      const mx::array input(ids.data(), {1, static_cast<int>(ids.size())},
                            mx::int32);
      std::vector<qwen3_asr::KVCache> caches = model.NewCaches();
      mx::array token =
          mx::argmax(model.Decode(model.EmbedTokens(input), caches));
      mx::eval(token);
      const auto t3 = Clock::now();
      constexpr int steps = 100;
      for (int i = 0; i < steps; ++i) {
        token = mx::argmax(model.Decode(
            model.EmbedTokens(mx::reshape(token, {1, 1})), caches));
        mx::async_eval({token});
      }
      mx::eval(token);
      const auto t4 = Clock::now();
      std::cout << "samples " << samples.size() / 16000.0 << "s mel "
                << ms(t0, t1) << "ms enc " << ms(t1, t2) << "ms prefill("
                << ids.size() << ") " << ms(t2, t3) << "ms decode "
                << ms(t3, t4) / steps << "ms/token\n";
    }
    return 0;
  } else if (encode) {
    const qwen3_asr::Tokenizer tokenizer(model_path);
    std::cout << nlohmann::json(tokenizer.Encode(encode_text)).dump() << "\n";
    return 0;
  } else {
  }
  const auto load_started = std::chrono::steady_clock::now();
  const qwen3_asr::Qwen3ASRTranscriber transcriber(model_path);
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
