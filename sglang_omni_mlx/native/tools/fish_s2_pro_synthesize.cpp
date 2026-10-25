// SPDX-License-Identifier: Apache-2.0
// Synthesizes speech with the native Fish S2-Pro runtime and prints one JSON
// line of timings; with --codes-output, also the prompt ids (all, and the text
// positions alone), the reference's codes and the frame codes, for parity
// checks against the reference.
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

#include "fish_s2_pro/fish_s2_pro_model.h"
#include "nlohmann/json.hpp"

namespace {

constexpr int kDefaultMaxFrames = 1024;
constexpr int kMaxFrames = 2048;
constexpr uint32_t kSeedMask = 0x7FFFFFFF;

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
      sampling.seed = static_cast<uint32_t>(seed & kSeedMask);
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
      max_frames = std::min(std::stoi(value), kMaxFrames);
    } else if (argument == "--output") {
      output_path = value;
    } else if (argument == "--codes-output") {
      codes_output_path = value;
    } else {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    }
  }
  // Freed MLX buffers go back to the system, as the server's do.
  mlx::core::set_cache_limit(0);
  const auto load_started = std::chrono::steady_clock::now();
  const fish_s2_pro::FishS2ProModel model(model_path);
  std::cerr << "loaded in " << SecondsSince(load_started) << " s\n";

  const auto started = std::chrono::steady_clock::now();
  std::vector<fish_s2_pro::VoiceReference> references;
  if (!reference_audio_path.empty()) {
    std::ifstream stream(reference_audio_path, std::ios::binary);
    std::ostringstream contents;
    contents << stream.rdbuf();
    const fish_s2_pro::DecodedWav wav =
        fish_s2_pro::DecodeAnyWav(contents.str());
    references.push_back(
        {reference_text,
         model.codec().Encode(fish_s2_pro::Resample(
             wav.samples, wav.sample_rate, model.codec().sample_rate()))});
  } else {
  }
  const fish_s2_pro::SpeechPrompt prompt = model.BuildPrompt(text, references);
  const double prompt_seconds = SecondsSince(started);

  std::vector<mlx::core::array> frames;
  nlohmann::json frame_codes = nlohmann::json::array();
  const std::atomic<bool> cancel(false);
  // The server's limit: the frames stop where the prompt and its audio would
  // outrun the model's positions.
  const int clamped_frames =
      std::min(max_frames, std::max(model.max_positions() - 1 -
                                        static_cast<int>(prompt.ids.size()),
                                    1));
  const qwen3_asr::FinishReason finish_reason =
      model.Generate(prompt, clamped_frames, sampling, cancel,
                     [&](const mlx::core::array &codes) {
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
      static_cast<double>(samples.size()) / model.codec().sample_rate();

  if (!output_path.empty()) {
    std::ofstream(output_path, std::ios::binary) << fish_s2_pro::WavBytes(
        fish_s2_pro::Pcm16Bytes(samples), model.codec().sample_rate());
  } else {
  }
  if (!codes_output_path.empty()) {
    nlohmann::json reference_codes = nlohmann::json::array();
    for (const fish_s2_pro::VoiceReference &reference : references) {
      const mlx::core::array codes =
          mlx::core::astype(reference.codes, mlx::core::int32);
      mlx::core::eval(codes);
      const int32_t *values = codes.data<int32_t>();
      nlohmann::json rows = nlohmann::json::array();
      for (int codebook = 0; codebook < codes.shape(0); ++codebook) {
        rows.push_back(std::vector<int32_t>(
            values + static_cast<size_t>(codebook) * codes.shape(1),
            values + static_cast<size_t>(codebook + 1) * codes.shape(1)));
      }
      reference_codes.push_back(rows);
    }
    // The text positions alone, which no chip's codec arithmetic changes.
    std::vector<int> prompt_text_ids;
    for (size_t position = 0; position < prompt.ids.size(); ++position) {
      if (!prompt.is_audio[position]) {
        prompt_text_ids.push_back(prompt.ids[position]);
      } else {
      }
    }
    std::ofstream(codes_output_path) << nlohmann::json{
        {"prompt_ids", prompt.ids},
        {"prompt_text_ids", prompt_text_ids},
        {"reference_codes", reference_codes},
        {"codes", frame_codes}}.dump();
  } else {
  }
  const nlohmann::json line = {
      {"prompt_token_count", prompt.ids.size()},
      {"frame_count", frame_codes.size()},
      {"finish_reason", qwen3_asr::FinishReasonName(finish_reason)},
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
