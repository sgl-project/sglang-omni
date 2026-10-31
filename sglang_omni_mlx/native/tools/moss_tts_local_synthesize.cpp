// SPDX-License-Identifier: Apache-2.0
// Synthesizes speech with the native MOSS-TTS-Local runtime and prints one
// JSON line of timings; with --codes-output, also the prompt ids and frame
// codes, for parity checks against the reference.
//
//   moss_tts_local_synthesize --model-path DIR --codec-path DIR --text TEXT
//       [--instructions T] [--language L] [--token-count N] [--seed N]
//       [--max-new-tokens N] [--temperature T] [--top-p P] [--top-k K]
//       [--repetition-penalty R] [--quantization-bits 4|8|16]
//       [--output out.wav] [--codes-output codes.json]
//   moss_tts_local_synthesize --model-path DIR --prompt-only < requests.jsonl
//
// --prompt-only reads one JSON request a line (text, and optionally
// instructions, language and token_count) and prints each one's prompt ids.
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "moss_tts_local/moss_tts_local_model.h"
#include "nlohmann/json.hpp"

namespace {

constexpr int kDefaultQuantizationBits = 8;

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::string codec_path;
  moss_tts_local::SpeechPrompt prompt;
  // Note (Jiaxin Deng): the CUDA server's defaults, the checkpoint's own.
  moss_tts_local::GenerationParameters parameters;
  parameters.text = {1.0, 1.0, 50, 1.0};
  parameters.audio = {1.7, 0.8, 25, 1.0};
  int quantization_bits = kDefaultQuantizationBits;
  std::string output_path;
  std::string codes_output_path;
  bool prompt_only = false;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--prompt-only") {
      prompt_only = true;
      continue;
    } else if (i + 1 >= argc) {
      std::cerr << argv[0] << ": missing value for " << argument << "\n";
      return 2;
    } else {
    }
    const std::string value = argv[++i];
    if (argument == "--model-path") {
      model_path = value;
    } else if (argument == "--codec-path") {
      codec_path = value;
    } else if (argument == "--text") {
      prompt.text = value;
    } else if (argument == "--instructions") {
      prompt.instruction = value;
    } else if (argument == "--language") {
      prompt.language = value;
    } else if (argument == "--token-count") {
      prompt.token_count = std::stoi(value);
    } else if (argument == "--seed") {
      parameters.seed = std::stoull(value);
    } else if (argument == "--max-new-tokens") {
      parameters.max_frames = std::stoi(value);
    } else if (argument == "--temperature") {
      // As the server's public fields, these set both heads.
      parameters.text.temperature = parameters.audio.temperature =
          std::stod(value);
    } else if (argument == "--top-p") {
      parameters.text.top_p = parameters.audio.top_p = std::stod(value);
    } else if (argument == "--top-k") {
      parameters.text.top_k = parameters.audio.top_k = std::stoi(value);
    } else if (argument == "--repetition-penalty") {
      parameters.audio.repetition_penalty = std::stod(value);
    } else if (argument == "--quantization-bits") {
      quantization_bits = std::stoi(value);
    } else if (argument == "--output") {
      output_path = value;
    } else if (argument == "--codes-output") {
      codes_output_path = value;
    } else {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    }
  }
  if (prompt_only) {
    const moss_tts_local::PromptEncoder encoder(model_path);
    for (std::string line; std::getline(std::cin, line);) {
      const nlohmann::json request = nlohmann::json::parse(line);
      moss_tts_local::SpeechPrompt fields;
      fields.text = request.at("text").get<std::string>();
      if (request.contains("instructions")) {
        fields.instruction = request.at("instructions").get<std::string>();
      } else {
      }
      if (request.contains("language")) {
        fields.language = request.at("language").get<std::string>();
      } else {
      }
      if (request.contains("token_count")) {
        fields.token_count = request.at("token_count").get<int>();
      } else {
      }
      std::vector<int> ids;
      for (const std::vector<int> &row : encoder.Rows(fields)) {
        ids.push_back(row[0]);
      }
      std::cout << nlohmann::json(ids).dump() << std::endl;
    }
    return 0;
  } else {
  }
  const auto load_started = std::chrono::steady_clock::now();
  const moss_tts_local::MossTtsLocalModel model(model_path, codec_path,
                                                quantization_bits);
  std::cerr << "loaded in " << SecondsSince(load_started) << " s\n";

  const auto started = std::chrono::steady_clock::now();
  const std::vector<std::vector<int>> prompt_rows =
      model.prompt_encoder().Rows(prompt);
  moss_tts_local::CodecStream stream(model.codec());
  std::vector<float> samples;
  nlohmann::json frame_codes = nlohmann::json::array();
  nlohmann::json frame_seconds = nlohmann::json::array();
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
      prompt_rows, parameters, cancel, [&](const mlx::core::array &codes) {
        const int32_t *values = codes.data<int32_t>();
        frame_codes.push_back(
            std::vector<int32_t>(values, values + codes.size()));
        frame_seconds.push_back(SecondsSince(started));
        keep(stream.Push(codes));
      });
  keep(stream.Finish());
  const double seconds = SecondsSince(started);
  const int channel_count = model.codec().channel_count();
  const double audio_seconds = static_cast<double>(samples.size()) /
                               channel_count / model.codec().sample_rate();

  if (!output_path.empty()) {
    std::ofstream(output_path, std::ios::binary)
        << moss_tts_local::WavBytes(moss_tts_local::Pcm16Bytes(samples),
                                    model.codec().sample_rate(), channel_count);
  } else {
  }
  if (!codes_output_path.empty()) {
    std::vector<int> prompt_ids;
    for (const std::vector<int> &row : prompt_rows) {
      prompt_ids.push_back(row[0]);
    }
    std::ofstream(codes_output_path) << nlohmann::json{
        {"prompt_ids", prompt_ids},
        {"codes", frame_codes},
        {"frame_seconds", frame_seconds}}.dump();
  } else {
  }
  const nlohmann::json line = {
      {"prompt_token_count", prompt_rows.size()},
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
