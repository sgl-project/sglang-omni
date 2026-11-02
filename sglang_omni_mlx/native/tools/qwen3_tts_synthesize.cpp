// SPDX-License-Identifier: Apache-2.0
// Synthesizes speech with the native Qwen3-TTS CustomVoice runtime and prints
// one JSON line of timings; with --codes-output, also the prompt ids and frame
// codes, for parity checks against the reference.
//
//   qwen3_tts_synthesize --model-path DIR --text TEXT [--speaker S]
//       [--language L] [--seed N] [--max-new-tokens N] [--output out.wav]
//       [--codes-output codes.json] [--instructions TEXT]
//       [--streaming-text] [--repetition-window N]
//
// --streaming-text feeds one text token per frame and --repetition-window
// limits the penalty to the last N frames, as mlx-audio does; by default the
// prompt and sampling are the CUDA server's.
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

#include "common/audio.h"
#include "nlohmann/json.hpp"
#include "qwen3_tts/qwen3_tts_model.h"

namespace {

constexpr const char *kDefaultSpeaker = "vivian";
constexpr const char *kDefaultLanguage = "auto";
constexpr int kDefaultMaxFrames = 2048;

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::string text;
  std::string speaker = kDefaultSpeaker;
  std::string language = kDefaultLanguage;
  std::string instructions;
  bool streaming_text = false;
  qwen3_tts::SamplingConfig sampling;
  int max_frames = kDefaultMaxFrames;
  std::string output_path;
  std::string codes_output_path;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--streaming-text") {
      streaming_text = true;
      continue;
    } else {
    }
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
    } else if (argument == "--speaker") {
      speaker = value;
    } else if (argument == "--language") {
      language = value;
    } else if (argument == "--instructions") {
      instructions = value;
    } else if (argument == "--repetition-window") {
      sampling.repetition_window_frames = std::stoi(value);
    } else if (argument == "--seed") {
      sampling.seed = std::stoull(value);
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
  const qwen3_tts::Qwen3TtsModel model(model_path);
  std::cerr << "loaded in " << SecondsSince(load_started) << " s\n";

  const auto started = std::chrono::steady_clock::now();
  qwen3_tts::SpeechPrompt prompt =
      model.BuildPrompt(text, speaker, language, instructions);
  prompt.text_up_front = !streaming_text;
  qwen3_tts::SpeechStream stream(model.speech_decoder(), false);
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
  const qwen3_asr::FinishReason finish_reason = model.Generate(
      prompt, sampling, max_frames, cancel, [&](const mlx::core::array &codes) {
        const int32_t *values = codes.data<int32_t>();
        frame_codes.push_back(
            std::vector<int32_t>(values, values + codes.size()));
        keep(stream.Push(codes));
      });
  keep(stream.Finish());
  const double seconds = SecondsSince(started);
  const double audio_seconds = static_cast<double>(samples.size()) /
                               model.speech_decoder().sample_rate();

  if (!output_path.empty()) {
    std::ofstream(output_path, std::ios::binary) << omni_common::WavBytes(
        omni_common::Pcm16Bytes(samples), model.speech_decoder().sample_rate());
  } else {
  }
  if (!codes_output_path.empty()) {
    std::ofstream(codes_output_path) << nlohmann::json{
        {"prompt_ids", model.PromptIds(prompt)},
        {"codes", frame_codes}}.dump();
  } else {
  }
  const nlohmann::json line = {
      {"prompt_token_count", model.PromptLength(prompt)},
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
