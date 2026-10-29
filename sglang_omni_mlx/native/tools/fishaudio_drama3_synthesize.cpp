// SPDX-License-Identifier: Apache-2.0
// Synthesizes speech with the native Drama 3 runtime and prints one JSON line
// of timings; with --codes-output, also the first chunk's prompt ids and every
// frame's codes, for parity checks against the reference. Audio rows of the
// prompt count as -1 there, since another chip's encoder may round a
// reference's codes differently.
//
//   fishaudio_drama3_synthesize --model-path DIR --text TEXT
//       [--reference-audio REF.wav | --reference-codes CODES.json]
//       [--reference-text TRANSCRIPT]...
//       [--seed N] [--temperature T] [--top-p P] [--top-k K]
//       [--max-new-tokens N] [--stream] [--output out.wav]
//       [--codes-output codes.json]
//
// Reference audio is mono 44.1 kHz WAV; reference codes are a JSON
// [codebooks][frames] grid. References pair with transcripts in order.
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/audio.h"
#include "fishaudio_drama3/fishaudio_drama3_model.h"
#include "fishaudio_drama3/fishaudio_drama3_text.h"
#include "nlohmann/json.hpp"

namespace {

double SecondsSince(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

std::string ReadFile(const std::string &path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error("cannot read " + path);
  } else {
  }
  return std::string(std::istreambuf_iterator<char>(stream), {});
}

void WriteFile(const std::string &path, const std::string &bytes) {
  std::ofstream stream(path, std::ios::binary);
  stream << bytes;
  if (!stream) {
    throw std::runtime_error("cannot write " + path);
  } else {
  }
}

// Thrown for a command line this tool cannot run, answered with exit code 2.
class UsageError : public std::invalid_argument {
public:
  using std::invalid_argument::invalid_argument;
};

double NumberOption(const std::string &argument, const std::string &value) {
  size_t parsed = 0;
  double number = 0;
  try {
    number = std::stod(value, &parsed);
  } catch (const std::logic_error &) {
    parsed = 0;
  }
  if (value.empty() || parsed != value.size() || !std::isfinite(number)) {
    throw UsageError(argument + " must be a finite number");
  } else {
  }
  return number;
}

int Run(int argc, char **argv) {
  std::string model_path;
  std::string text;
  // Each reference's audio or codes file, and whether it is codes.
  std::vector<std::pair<std::string, bool>> reference_paths;
  std::vector<std::string> reference_texts;
  std::optional<uint64_t> seed;
  fishaudio_drama3::Sampling sampling;
  int max_frames = fishaudio_drama3::kDefaultMaxFrames;
  bool stream = false;
  std::string output_path;
  std::string codes_output_path;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--stream") {
      stream = true;
      continue;
    } else if (i + 1 >= argc) {
      throw UsageError("missing value for " + argument);
    } else {
    }
    const std::string value = argv[++i];
    if (argument == "--model-path") {
      model_path = value;
    } else if (argument == "--text") {
      text = value;
    } else if (argument == "--reference-audio") {
      reference_paths.emplace_back(value, false);
    } else if (argument == "--reference-codes") {
      reference_paths.emplace_back(value, true);
    } else if (argument == "--reference-text") {
      reference_texts.push_back(value);
    } else if (argument == "--seed") {
      const double number = NumberOption(argument, value);
      if (number < 0 || number != std::floor(number) ||
          value.find_first_not_of("0123456789") != std::string::npos) {
        throw UsageError("--seed must be a non-negative integer");
      } else {
      }
      try {
        seed = std::stoull(value);
      } catch (const std::out_of_range &) {
        throw UsageError("--seed must be below 2^64");
      }
    } else if (argument == "--temperature") {
      sampling.temperature = static_cast<float>(NumberOption(argument, value));
    } else if (argument == "--top-p") {
      sampling.top_p = static_cast<float>(NumberOption(argument, value));
    } else if (argument == "--top-k") {
      const double number = NumberOption(argument, value);
      if (number < 1 || number != std::floor(number)) {
        throw UsageError("--top-k must be an integer of at least 1");
      } else {
      }
      sampling.top_k = static_cast<int>(std::min(number, 1e9));
    } else if (argument == "--max-new-tokens") {
      const double number = NumberOption(argument, value);
      if (number < 1 || number > 1e9 || number != std::floor(number)) {
        throw UsageError("--max-new-tokens must be a positive integer");
      } else {
      }
      max_frames = static_cast<int>(number);
    } else if (argument == "--output") {
      output_path = value;
    } else if (argument == "--codes-output") {
      codes_output_path = value;
    } else {
      throw UsageError("unknown option " + argument);
    }
  }
  if (model_path.empty()) {
    throw UsageError("--model-path is required");
  } else if (reference_paths.size() != reference_texts.size()) {
    throw UsageError("every reference needs its --reference-text");
  } else if (!std::isfinite(sampling.temperature) || sampling.temperature < 0) {
    throw UsageError("--temperature must be a finite number of at least 0");
  } else if (!(sampling.top_p > 0 && sampling.top_p <= 1)) {
    throw UsageError("--top-p must be in (0, 1]");
  } else {
  }
  try {
    fishaudio_drama3::ValidateText(text, "--text");
    for (const std::string &reference_text : reference_texts) {
      fishaudio_drama3::ValidateText(reference_text, "--reference-text");
    }
  } catch (const std::invalid_argument &error) {
    throw UsageError(error.what());
  }
  const std::vector<std::string> chunks = fishaudio_drama3::SplitText(text);
  if (chunks.empty()) {
    throw UsageError("--text has nothing to speak");
  } else {
  }
  const auto load_started = std::chrono::steady_clock::now();
  fishaudio_drama3::Drama3Model model(model_path);
  std::cerr << "loaded in " << SecondsSince(load_started) << " s\n";

  const auto started = std::chrono::steady_clock::now();
  std::vector<fishaudio_drama3::ReferenceInput> references;
  for (size_t index = 0; index < reference_paths.size(); ++index) {
    const auto &[path, is_codes] = reference_paths[index];
    if (is_codes) {
      const nlohmann::json grid = nlohmann::json::parse(ReadFile(path));
      std::vector<std::vector<int32_t>> codes;
      for (const nlohmann::json &codebook : grid) {
        std::vector<int32_t> &row = codes.emplace_back();
        for (const nlohmann::json &code : codebook) {
          if (!code.is_number_unsigned() ||
              code.get<uint64_t>() > std::numeric_limits<int32_t>::max()) {
            throw UsageError(path + " must hold non-negative integer codes");
          } else {
          }
          row.push_back(code.get<int32_t>());
        }
      }
      references.push_back({reference_texts[index], {}, codes});
    } else {
      std::vector<float> samples = omni_common::DecodeWav(
          ReadFile(path), fishaudio_drama3::kCodecSampleRate);
      if (!std::all_of(samples.begin(), samples.end(),
                       [](float sample) { return std::isfinite(sample); })) {
        throw UsageError(path + " must hold finite samples");
      } else {
      }
      references.push_back(
          {reference_texts[index], std::move(samples), std::nullopt});
    }
  }
  std::shared_ptr<fishaudio_drama3::Voice> voice;
  try {
    voice = model.GetVoice(references);
  } catch (const std::invalid_argument &error) {
    throw UsageError(error.what());
  }
  if (seed.has_value()) {
    // Note (Yitong Li): MLX's default key sequence is per thread; this seeds
    // the thread that generates, after the references are encoded.
    mlx::core::random::seed(*seed);
  } else {
  }
  const fishaudio_drama3::PromptFrames first_prompt =
      fishaudio_drama3::BuildFrames(model.tokenizer(), voice->references, {},
                                    chunks.front(), model.codebook_count(),
                                    model.codebook_size(),
                                    model.semantic_begin());
  const mlx::core::array prompt_column = mlx::core::contiguous(
      mlx::core::reshape(mlx::core::slice(first_prompt.frames, {0, 0},
                                          {first_prompt.frames.shape(0), 1}),
                         {-1}));
  mlx::core::eval(prompt_column);
  std::vector<int32_t> prompt_ids(prompt_column.data<int32_t>(),
                                  prompt_column.data<int32_t>() +
                                      prompt_column.size());
  for (int32_t &prompt_id : prompt_ids) {
    if (prompt_id >= model.semantic_begin() &&
        prompt_id < model.semantic_begin() + model.codebook_size()) {
      prompt_id = -1;
    } else {
    }
  }
  std::vector<float> samples;
  nlohmann::json frame_codes = nlohmann::json::array();
  std::optional<double> first_audio_seconds;
  const std::atomic<bool> cancel(false);
  const fishaudio_drama3::SynthesisResult result = model.Synthesize(
      text, *voice, sampling, max_frames, stream, cancel,
      [&](const mlx::core::array &codes) {
        const int32_t *values = codes.data<int32_t>();
        frame_codes.push_back(
            std::vector<int32_t>(values, values + codes.size()));
      },
      [&](std::vector<float> audio) {
        if (!audio.empty() && !first_audio_seconds.has_value()) {
          first_audio_seconds = SecondsSince(started);
        } else {
        }
        samples.insert(samples.end(), audio.begin(), audio.end());
      });
  const double seconds = SecondsSince(started);
  const double audio_seconds =
      static_cast<double>(samples.size()) / fishaudio_drama3::kCodecSampleRate;

  if (!output_path.empty()) {
    WriteFile(output_path,
              omni_common::WavBytes(omni_common::Pcm16Bytes(samples),
                                    fishaudio_drama3::kCodecSampleRate));
  } else {
  }
  if (!codes_output_path.empty()) {
    WriteFile(codes_output_path,
              nlohmann::json{{"prompt_ids", prompt_ids}, {"codes", frame_codes}}
                  .dump());
  } else {
  }
  const nlohmann::json line = {
      {"frame_count", frame_codes.size()},
      {"prompt_row_count", result.prompt_row_count},
      {"finish_reason", qwen3_asr::FinishReasonName(result.finish_reason)},
      {"seconds", seconds},
      {"first_audio_seconds", first_audio_seconds.value_or(seconds)},
      {"audio_seconds", audio_seconds},
      {"real_time_factor", audio_seconds > 0 ? seconds / audio_seconds : 0.0},
      {"peak_memory_gb",
       static_cast<double>(mlx::core::get_peak_memory()) / 1e9},
  };
  std::cout << line.dump() << std::endl;
  return 0;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return Run(argc, argv);
  } catch (const UsageError &error) {
    std::cerr << argv[0] << ": " << error.what() << "\n";
    return 2;
  } catch (const std::exception &error) {
    std::cerr << argv[0] << ": " << error.what() << "\n";
    return 1;
  }
}
