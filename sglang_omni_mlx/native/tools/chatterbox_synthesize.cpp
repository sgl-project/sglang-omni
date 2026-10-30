// SPDX-License-Identifier: Apache-2.0
// Synthesizes speech with the native Chatterbox-Turbo runtime and prints one
// JSON line of timings and peak memory; with --tokens-output, also the text
// ids, the cloned voice's prompt speech tokens and the generated speech
// tokens, and with --samples-output the float32 samples, for parity checks
// against the reference. --speech-tokens skips T3 and voices the given
// space-separated speech tokens.
//
//   chatterbox_synthesize --model-path DIR --text TEXT
//       [--reference-audio WAV] [--seed N] [--max-new-tokens N]
//       [--speech-tokens "T1 T2 ..."]
//       [--temperature T] [--top-k K] [--top-p P] [--repetition-penalty R]
//       [--output out.wav] [--tokens-output tokens.json]
//       [--samples-output samples.f32]
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <sstream>
#include <string>
#include <vector>

#include "chatterbox/chatterbox_model.h"
#include "nlohmann/json.hpp"
#include "voxtral_tts/voxtral_tts_codec.h"

namespace {

constexpr int kDefaultMaxSpeechTokens = 1000;

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

std::vector<int> Ints(const mlx::core::array &values) {
  const mlx::core::array flat =
      mlx::core::astype(mlx::core::reshape(values, {-1}), mlx::core::int32);
  mlx::core::eval(flat);
  return std::vector<int>(flat.data<int32_t>(),
                          flat.data<int32_t>() + flat.size());
}

} // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::string text;
  std::string reference_path;
  std::optional<uint64_t> seed;
  int max_speech_tokens = kDefaultMaxSpeechTokens;
  chatterbox::SamplingOptions sampling;
  std::string output_path;
  std::string tokens_output_path;
  std::string samples_output_path;
  std::optional<std::vector<int>> given_speech_tokens;
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
    } else if (argument == "--reference-audio") {
      reference_path = value;
    } else if (argument == "--seed") {
      seed = std::stoull(value);
    } else if (argument == "--max-new-tokens") {
      max_speech_tokens = std::stoi(value);
    } else if (argument == "--temperature") {
      sampling.temperature = std::stof(value);
    } else if (argument == "--top-k") {
      sampling.top_k = std::stoi(value);
    } else if (argument == "--top-p") {
      sampling.top_p = std::stof(value);
    } else if (argument == "--repetition-penalty") {
      sampling.repetition_penalty = std::stof(value);
    } else if (argument == "--output") {
      output_path = value;
    } else if (argument == "--tokens-output") {
      tokens_output_path = value;
    } else if (argument == "--samples-output") {
      samples_output_path = value;
    } else if (argument == "--speech-tokens") {
      std::istringstream tokens(value);
      given_speech_tokens.emplace(std::istream_iterator<int>(tokens),
                                  std::istream_iterator<int>());
    } else {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    }
  }
  const auto load_started = std::chrono::steady_clock::now();
  const chatterbox::ChatterboxModel model(model_path);
  const double load_seconds = SecondsSince(load_started);
  mlx::core::reset_peak_memory();
  if (seed.has_value()) {
    // Note (Minyi Tang): MLX's default key sequence is per thread; this
    // seeds the thread that generates.
    mlx::core::random::seed(*seed);
  } else {
  }

  const auto started = std::chrono::steady_clock::now();
  chatterbox::Voice voice = model.builtin_voice();
  if (!reference_path.empty()) {
    std::ifstream stream(reference_path, std::ios::binary);
    const std::string wav_bytes((std::istreambuf_iterator<char>(stream)),
                                std::istreambuf_iterator<char>());
    chatterbox::ChatterboxModel::CheckReference(wav_bytes);
    voice = model.CloneVoice(wav_bytes);
  } else {
  }
  const double conditioning_seconds = SecondsSince(started);
  const std::vector<int> text_ids = model.EncodeText(text);
  const auto tokens_started = std::chrono::steady_clock::now();
  std::vector<int> speech_tokens;
  const std::atomic<bool> cancel(false);
  qwen3_asr::FinishReason finish_reason = qwen3_asr::FinishReason::kStop;
  if (given_speech_tokens.has_value()) {
    speech_tokens = *given_speech_tokens;
  } else {
    finish_reason = model.GenerateSpeechTokens(
        voice, text_ids, sampling, max_speech_tokens, cancel,
        [&](int token) { speech_tokens.push_back(token); });
  }
  const double t3_seconds = SecondsSince(tokens_started);
  const auto audio_started = std::chrono::steady_clock::now();
  const std::vector<float> samples = model.Synthesize(speech_tokens, voice);
  const double s3gen_seconds = SecondsSince(audio_started);
  const double seconds = SecondsSince(started);
  const double audio_seconds =
      static_cast<double>(samples.size()) / chatterbox::kOutputSampleRate;

  if (!output_path.empty()) {
    std::ofstream(output_path, std::ios::binary) << voxtral_tts::WavBytes(
        voxtral_tts::Pcm16Bytes(samples), chatterbox::kOutputSampleRate);
  } else {
  }
  if (!samples_output_path.empty()) {
    std::ofstream(samples_output_path, std::ios::binary)
        .write(reinterpret_cast<const char *>(samples.data()),
               static_cast<std::streamsize>(samples.size() * sizeof(float)));
  } else {
  }
  if (!tokens_output_path.empty()) {
    std::ofstream(tokens_output_path) << nlohmann::json{
        {"text_ids", text_ids},
        {"voice_tokens", reference_path.empty()
                             ? std::vector<int>()
                             : Ints(voice.t3.prompt_speech_tokens)},
        {"speech_tokens", speech_tokens}}.dump();
  } else {
  }
  const nlohmann::json line = {
      {"text_token_count", text_ids.size()},
      {"speech_token_count", speech_tokens.size()},
      {"finish_reason", qwen3_asr::FinishReasonName(finish_reason)},
      {"load_seconds", load_seconds},
      {"conditioning_seconds", conditioning_seconds},
      {"t3_seconds", t3_seconds},
      {"s3gen_seconds", s3gen_seconds},
      {"seconds", seconds},
      {"audio_seconds", audio_seconds},
      {"real_time_factor", audio_seconds > 0 ? seconds / audio_seconds : 0.0},
      {"peak_memory_bytes", mlx::core::get_peak_memory()},
  };
  std::cout << line.dump() << std::endl;
  return 0;
}
