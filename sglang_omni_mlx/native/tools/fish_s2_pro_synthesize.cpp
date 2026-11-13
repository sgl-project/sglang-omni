// SPDX-License-Identifier: Apache-2.0
// Synthesizes speech with the native Fish S2-Pro runtime and prints one JSON
// line of timings; with --codes-output, also the text positions' prompt ids
// and the frame codes, for the golden check.
//
//   fish_s2_pro_synthesize --model-path DIR --text TEXT
//       [--ref-audio ref.wav [--ref-text TEXT]] [--seed N]
//       [--temperature T] [--top-p P] [--top-k K] [--repetition-penalty R]
//       [--max-new-tokens N] [--output out.wav] [--codes-output codes.json]
#include <algorithm>
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "common/audio.h"
#include "common/reference_audio.h"
#include "common/transcription.h"
#include "fish_s2_pro/fish_s2_pro_model.h"
#include "nlohmann/json.hpp"

namespace {

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

} // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::string text;
  std::string reference_audio_path;
  std::string reference_text;
  fish_s2_pro::SamplingOptions sampling;
  int max_frames = fish_s2_pro::kDefaultMaxFrames;
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
    } else if (argument == "--ref-audio") {
      reference_audio_path = value;
    } else if (argument == "--ref-text") {
      reference_text = value;
    } else if (argument == "--seed") {
      const long long seed = std::stoll(value);
      if (seed < 0) {
        std::cerr << argv[0] << ": --seed must be greater than or equal to 0\n";
        return 2;
      } else {
      }
      sampling.seed = static_cast<uint32_t>(seed & fish_s2_pro::kSeedMask);
    } else if (argument == "--temperature") {
      sampling.temperature = std::stof(value);
    } else if (argument == "--top-p") {
      sampling.top_p = std::stof(value);
    } else if (argument == "--top-k") {
      sampling.top_k = std::stoi(value);
    } else if (argument == "--repetition-penalty") {
      sampling.repetition_penalty = std::stof(value);
    } else if (argument == "--max-new-tokens") {
      // The server's cap, applied without an error as the CUDA engine does.
      max_frames = std::min(std::stoi(value), fish_s2_pro::kMaxFrames);
    } else if (argument == "--output") {
      output_path = value;
    } else if (argument == "--codes-output") {
      codes_output_path = value;
    } else {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    }
  }
  // The server's limit, so the tool's time and memory match its.
  mlx::core::set_cache_limit(fish_s2_pro::kCacheLimitBytes);
  const auto load_started = std::chrono::steady_clock::now();
  const fish_s2_pro::FishS2ProModel model(model_path);
  std::cerr << "loaded in " << SecondsSince(load_started) << " s\n";

  const auto started = std::chrono::steady_clock::now();
  std::vector<fish_s2_pro::VoiceReference> references;
  if (!reference_audio_path.empty()) {
    std::ifstream stream(reference_audio_path, std::ios::binary);
    std::ostringstream contents;
    contents << stream.rdbuf();
    const omni_common::DecodedAudio wav =
        omni_common::DecodeAnyWav(contents.str());
    references.push_back(
        {reference_text,
         model.codec().Encode(omni_common::Resample(
             wav.samples, wav.sample_rate, fish_dac::kSampleRate))});
  } else {
  }
  const fish_s2_pro::SpeechPrompt prompt = model.BuildPrompt(text, references);
  const double prompt_seconds = SecondsSince(started);

  std::vector<mlx::core::array> frames;
  nlohmann::json frame_codes = nlohmann::json::array();
  const std::atomic<bool> cancel(false);
  const omni_common::FinishReason finish_reason = model.Generate(
      prompt, max_frames, sampling, cancel, [&](const mlx::core::array &codes) {
        const int32_t *values = codes.data<int32_t>();
        frame_codes.push_back(
            std::vector<int32_t>(values, values + codes.size()));
        frames.push_back(codes);
      });
  const double generate_seconds = SecondsSince(started);
  std::vector<float> samples;
  if (!frames.empty()) {
    const mlx::core::array audio =
        model.codec().Decode(mlx::core::stack(frames, 1));
    mlx::core::eval(audio);
    samples.assign(audio.data<float>(), audio.data<float>() + audio.size());
  } else {
  }
  const double seconds = SecondsSince(started);
  const double audio_seconds =
      static_cast<double>(samples.size()) / fish_dac::kSampleRate;

  if (!output_path.empty()) {
    std::ofstream(output_path, std::ios::binary) << omni_common::WavBytes(
        omni_common::Pcm16Bytes(samples), fish_dac::kSampleRate);
  } else {
  }
  if (!codes_output_path.empty()) {
    // The text positions alone, which no chip's codec arithmetic changes.
    std::vector<int> prompt_text_ids;
    for (size_t position = 0; position < prompt.ids.size(); ++position) {
      if (!prompt.is_audio[position]) {
        prompt_text_ids.push_back(prompt.ids[position]);
      } else {
      }
    }
    std::ofstream(codes_output_path) << nlohmann::json{
        {"prompt_ids", prompt_text_ids},
        {"codes", frame_codes}}.dump();
  } else {
  }
  const nlohmann::json line = {
      {"prompt_token_count", prompt.ids.size()},
      {"frame_count", frame_codes.size()},
      {"finish_reason", omni_common::FinishReasonName(finish_reason)},
      {"prompt_seconds", prompt_seconds},
      {"generate_seconds", generate_seconds},
      {"seconds", seconds},
      {"audio_seconds", audio_seconds},
      {"real_time_factor", audio_seconds > 0 ? seconds / audio_seconds : 0.0},
      {"peak_memory_gb", mlx::core::get_peak_memory() / 1e9},
  };
  std::cout << line.dump() << std::endl;
  return 0;
}
