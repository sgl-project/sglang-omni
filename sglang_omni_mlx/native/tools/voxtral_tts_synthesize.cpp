// SPDX-License-Identifier: Apache-2.0
// Synthesizes speech with the native Voxtral TTS runtime and prints one JSON
// line of timings; with --codes-output, also the prompt ids and frame codes,
// for parity checks against the reference.
//
//   voxtral_tts_synthesize --model-path DIR --text TEXT [--voice V]
//       [--seed N] [--max-new-tokens N] [--output out.wav]
//       [--codes-output codes.json]
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"
#include "voxtral_tts/voxtral_tts_model.h"

namespace {

constexpr const char *kDefaultVoice = "cheerful_female";
constexpr int kDefaultMaxFrames = 4096;

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::string text;
  std::string voice = kDefaultVoice;
  std::optional<uint64_t> seed;
  int max_frames = kDefaultMaxFrames;
  std::string output_path;
  std::string codes_output_path;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (i + 1 >= argc) {
      std::cerr << argv[0] << ": missing value for " << argument << "\n";
      return 2;
    } else {
    }
    const std::string value = argv[++i];
    if (argument == "--model-path") {
      model_path = value;
    } else if (argument == "--text") {
      text = value;
    } else if (argument == "--voice") {
      voice = value;
    } else if (argument == "--seed") {
      seed = std::stoull(value);
    } else if (argument == "--max-new-tokens") {
      max_frames = std::stoi(value);
    } else if (argument == "--output") {
      output_path = value;
    } else if (argument == "--codes-output") {
      codes_output_path = value;
    } else {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    }
  }
  const auto load_started = std::chrono::steady_clock::now();
  const voxtral_tts::VoxtralTtsModel model(model_path);
  std::cerr << "loaded in " << SecondsSince(load_started) << " s\n";
  if (seed.has_value()) {
    // Note (Minyi Tang): MLX's default key sequence is per thread; this seeds
    // the thread that generates.
    mlx::core::random::seed(*seed);
  } else {
  }

  const auto started = std::chrono::steady_clock::now();
  const std::vector<int> prompt_ids =
      model.tokenizer().EncodeSpeechRequest(text, voice);
  voxtral_tts::CodecStream stream(model.codec());
  std::vector<float> samples;
  nlohmann::json frame_codes = nlohmann::json::array();
  std::optional<double> first_audio_seconds;
  const auto keep = [&](const std::vector<float> &delta) {
    if (!delta.empty() && !first_audio_seconds.has_value()) {
      first_audio_seconds = SecondsSince(started);
    } else {
    }
    samples.insert(samples.end(), delta.begin(), delta.end());
  };
  const std::atomic<bool> cancel(false);
  const qwen3_asr::FinishReason finish_reason =
      model.Generate(prompt_ids, voice, max_frames, cancel,
                     [&](const mlx::core::array &codes) {
                       const int32_t *values = codes.data<int32_t>();
                       frame_codes.push_back(
                           std::vector<int32_t>(values, values + codes.size()));
                       keep(stream.Push(codes));
                     });
  keep(stream.Finish());
  const double seconds = SecondsSince(started);
  const double audio_seconds =
      static_cast<double>(samples.size()) / model.codec().sample_rate();

  if (!output_path.empty()) {
    std::ofstream(output_path, std::ios::binary) << voxtral_tts::WavBytes(
        voxtral_tts::Pcm16Bytes(samples), model.codec().sample_rate());
  } else {
  }
  if (!codes_output_path.empty()) {
    std::ofstream(codes_output_path) << nlohmann::json{
        {"prompt_ids", prompt_ids},
        {"codes", frame_codes}}.dump();
  } else {
  }
  const nlohmann::json line = {
      {"prompt_token_count", prompt_ids.size()},
      {"frame_count", frame_codes.size()},
      {"finish_reason", qwen3_asr::FinishReasonName(finish_reason)},
      {"seconds", seconds},
      {"first_audio_seconds", first_audio_seconds.value_or(seconds)},
      {"audio_seconds", audio_seconds},
      {"real_time_factor", audio_seconds > 0 ? seconds / audio_seconds : 0.0},
  };
  std::cout << line.dump() << std::endl;
  return 0;
}
