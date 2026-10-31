// SPDX-License-Identifier: Apache-2.0
// Synthesizes speech with the native Higgs Audio v3 TTS runtime and prints
// one JSON line of timings and memory; with --codes-output, also the prompt
// ids and delayed rows, for parity checks against the reference. --stream
// decodes with the server's streaming schedule instead of once at the end.
//
//   higgs_tts_synthesize --model-path DIR --text TEXT [--seed N]
//       [--max-new-tokens N] [--temperature T] [--top-p P] [--top-k K]
//       [--quantize-bits 0|8] [--stream] [--output out.wav]
//       [--codes-output codes.json]
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "higgs_tts/higgs_tts_model.h"
#include "nlohmann/json.hpp"
#include "voxtral_tts/voxtral_tts_codec.h"

namespace {

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::string text;
  std::optional<uint64_t> seed;
  int max_rows = higgs_tts::kMaxRows;
  higgs_tts::SamplingOptions sampling = higgs_tts::kSpeechSampling;
  int quantize_bits = higgs_tts::kDefaultQuantizeBits;
  bool stream = false;
  std::string output_path;
  std::string codes_output_path;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--stream") {
      stream = true;
      continue;
    } else if (i + 1 >= argc) {
      std::cerr << argv[0] << ": missing value for " << argument << "\n";
      return 2;
    } else {
    }
    const std::string value = argv[++i];
    if (argument == "--model-path") {
      model_path = value;
    } else if (argument == "--text") {
      text = value;
    } else if (argument == "--seed") {
      seed = std::stoull(value);
    } else if (argument == "--max-new-tokens") {
      max_rows = std::stoi(value);
    } else if (argument == "--temperature") {
      sampling.temperature = std::stod(value);
    } else if (argument == "--top-p") {
      sampling.top_p = std::stod(value);
    } else if (argument == "--top-k") {
      sampling.top_k = std::stoi(value);
    } else if (argument == "--quantize-bits") {
      quantize_bits = std::stoi(value);
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
  const higgs_tts::HiggsTtsModel model(model_path, quantize_bits);
  const double load_seconds = SecondsSince(load_started);
  if (seed.has_value()) {
    // Note (Minyi Tang): MLX's default key sequence is per thread; this
    // seeds the thread that generates.
    mlx::core::random::seed(*seed);
  } else {
  }

  const auto started = std::chrono::steady_clock::now();
  const std::vector<int> prompt_ids = model.EncodePrompt(text);
  higgs_tts::CodecStream codec_stream(model.codec());
  std::vector<int32_t> delayed_codes;
  std::vector<float> samples;
  nlohmann::json rows = nlohmann::json::array();
  std::optional<double> first_audio_seconds;
  const auto keep = [&](const std::vector<float> &delta) {
    if (!delta.empty() && !first_audio_seconds.has_value()) {
      first_audio_seconds = SecondsSince(started);
    } else {
    }
    samples.insert(samples.end(), delta.begin(), delta.end());
  };
  const std::atomic<bool> cancel(false);
  const qwen3_asr::FinishReason finish_reason = model.Generate(
      prompt_ids, sampling, max_rows, cancel,
      [&](const mlx::core::array &codes) {
        const int32_t *values = codes.data<int32_t>();
        rows.push_back(std::vector<int32_t>(values, values + codes.size()));
        delayed_codes.insert(delayed_codes.end(), values,
                             values + codes.size());
        if (stream) {
          keep(codec_stream.Push(codes));
        } else {
        }
      });
  const int row_count = static_cast<int>(rows.size());
  if (stream) {
    keep(codec_stream.Finish());
  } else if (row_count >= higgs_tts::kCodebookCount) {
    const mlx::core::array audio =
        model.codec().Decode(higgs_tts::RawFrames(delayed_codes, 0, row_count));
    mlx::core::eval(audio);
    keep(std::vector<float>(audio.data<float>(),
                            audio.data<float>() + audio.size()));
  } else {
  }
  const double seconds = SecondsSince(started);
  const double audio_seconds =
      static_cast<double>(samples.size()) / model.codec().sample_rate();

  if (!output_path.empty()) {
    std::ofstream(output_path, std::ios::binary) << voxtral_tts::WavBytes(
        voxtral_tts::Pcm16Bytes(samples), model.codec().sample_rate());
  } else {
  }
  if (!codes_output_path.empty()) {
    std::ofstream(codes_output_path)
        << nlohmann::json{{"prompt_ids", prompt_ids}, {"rows", rows}}.dump();
  } else {
  }
  const nlohmann::json line = {
      {"prompt_token_count", prompt_ids.size()},
      {"row_count", row_count},
      {"finish_reason", qwen3_asr::FinishReasonName(finish_reason)},
      {"load_seconds", load_seconds},
      {"seconds", seconds},
      {"first_audio_seconds", first_audio_seconds.value_or(seconds)},
      {"audio_seconds", audio_seconds},
      {"real_time_factor", audio_seconds > 0 ? seconds / audio_seconds : 0.0},
      {"peak_memory_gib", static_cast<double>(mlx::core::get_peak_memory()) /
                              (1024.0 * 1024.0 * 1024.0)},
  };
  std::cout << line.dump() << std::endl;
  return 0;
}
