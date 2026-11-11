// SPDX-License-Identifier: Apache-2.0
// Synthesizes speech with the native ZONOS3-Mini runtime and prints one JSON
// line of timings; with --codes-output, also the prompt's text column, every
// generated delayed row and the aligned frames, for parity checks against
// the reference.
//
//   zonos3_synthesize --model-path DIR --text TEXT
//       [--voice NAME | --reference-audio REF.wav] [--seed N]
//       [--temperature T] [--top-k K] [--top-p P] [--min-p P]
//       [--repetition-penalty R] [--max-new-tokens N] [--speed S]
//       [--language L] [--no-text-normalization] [--stream]
//       [--output out.wav] [--codes-output codes.json]
//       [--samples-output samples.f32]
//
// --voice none generates without a speaker; reference audio is mono 44.1 kHz
// WAV.
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <iterator>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/audio.h"
#include "common/reference_audio.h"
#include "nlohmann/json.hpp"
#include "zonos3/zonos3_model.h"

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

int IntegerOption(const std::string &argument, const std::string &value,
                  int minimum) {
  const double number = NumberOption(argument, value);
  if (number < minimum || number > 1e9 || number != std::floor(number)) {
    throw UsageError(argument + " must be an integer of at least " +
                     std::to_string(minimum));
  } else {
  }
  return static_cast<int>(number);
}

int Run(int argc, char **argv) {
  std::string model_path;
  zonos3::SynthesisRequest request;
  std::string voice = zonos3::kDefaultVoice;
  std::string reference_path;
  double speed = 1.0;
  std::string output_path;
  std::string codes_output_path;
  std::string samples_output_path;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--stream") {
      request.stream = true;
      continue;
    } else if (argument == "--no-text-normalization") {
      request.normalize_text = false;
      continue;
    } else if (i + 1 >= argc) {
      throw UsageError("missing value for " + argument);
    } else {
    }
    const std::string value = argv[++i];
    if (argument == "--model-path") {
      model_path = value;
    } else if (argument == "--text") {
      request.text = value;
    } else if (argument == "--voice") {
      voice = value;
    } else if (argument == "--reference-audio") {
      reference_path = value;
    } else if (argument == "--seed") {
      if (value.empty() ||
          value.find_first_not_of("0123456789") != std::string::npos) {
        throw UsageError("--seed must be a non-negative integer");
      } else {
      }
      try {
        request.sampling.seed = std::stoull(value);
      } catch (const std::out_of_range &) {
        throw UsageError("--seed must be below 2^64");
      }
    } else if (argument == "--temperature") {
      request.sampling.temperature = NumberOption(argument, value);
    } else if (argument == "--top-k") {
      request.sampling.top_k = IntegerOption(argument, value, 0);
    } else if (argument == "--top-p") {
      request.sampling.top_p = NumberOption(argument, value);
    } else if (argument == "--min-p") {
      request.sampling.min_p = NumberOption(argument, value);
    } else if (argument == "--repetition-penalty") {
      request.sampling.repetition_penalty = NumberOption(argument, value);
    } else if (argument == "--max-new-tokens") {
      request.max_rows = IntegerOption(argument, value, 1);
    } else if (argument == "--speed") {
      speed = NumberOption(argument, value);
    } else if (argument == "--language") {
      request.language = value;
    } else if (argument == "--output") {
      output_path = value;
    } else if (argument == "--codes-output") {
      codes_output_path = value;
    } else if (argument == "--samples-output") {
      samples_output_path = value;
    } else {
      throw UsageError("unknown option " + argument);
    }
  }
  if (model_path.empty()) {
    throw UsageError("--model-path is required");
  } else if (request.text.empty()) {
    throw UsageError("--text is required");
  } else if (!(speed > 0)) {
    throw UsageError("--speed must be positive");
  } else {
  }
  const auto load_started = std::chrono::steady_clock::now();
  zonos3::Zonos3Model model(model_path);
  std::cerr << "loaded in " << SecondsSince(load_started) << " s\n";

  const auto started = std::chrono::steady_clock::now();
  try {
    if (!reference_path.empty()) {
      const omni_common::DecodedAudio decoded =
          omni_common::DecodeAnyWav(ReadFile(reference_path));
      request.speaker_embedding = model.ReferenceEmbedding(
          decoded.sample_rate == zonos3::kSampleRate
              ? decoded.samples
              : omni_common::Resample(decoded.samples, decoded.sample_rate,
                                      zonos3::kSampleRate));
    } else if (voice != zonos3::kNoVoice) {
      request.speaker_embedding = model.VoiceEmbedding(voice);
    } else {
    }
    request.condition_buckets =
        speed == 1.0 ? model.ConditionIds(std::nullopt, std::nullopt)
                     : model.SpeedConditionIds(speed);
  } catch (const std::invalid_argument &error) {
    throw UsageError(error.what());
  }
  const zonos3::Prompt prompt = model.BuildPrompt(request);
  std::vector<int> prompt_ids;
  for (const auto &row : prompt.rows) {
    prompt_ids.push_back(row.back());
  }
  nlohmann::json rows = nlohmann::json::array();
  std::vector<float> samples;
  std::optional<double> first_audio_seconds;
  const std::atomic<bool> cancel(false);
  const zonos3::SynthesisResult result = model.Synthesize(
      request, cancel,
      [&](const mlx::core::array &codes) {
        const int32_t *values = codes.data<int32_t>();
        rows.push_back(std::vector<int32_t>(values, values + codes.size()));
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
      static_cast<double>(samples.size()) / zonos3::kSampleRate;
  if (!output_path.empty()) {
    WriteFile(output_path,
              omni_common::WavBytes(omni_common::Pcm16Bytes(samples),
                                    zonos3::kSampleRate));
  } else {
  }
  if (!samples_output_path.empty()) {
    WriteFile(samples_output_path,
              std::string(reinterpret_cast<const char *>(samples.data()),
                          samples.size() * sizeof(float)));
  } else {
  }
  if (!codes_output_path.empty()) {
    // The aligned frames, as the codec decodes them: codebook c of frame f
    // from generated row f + c.
    const int codebooks = model.config().codebook_count;
    nlohmann::json frames = nlohmann::json::array();
    for (int frame = 0; frame < result.frame_count; ++frame) {
      std::vector<int32_t> codes(codebooks);
      for (int codebook = 0; codebook < codebooks; ++codebook) {
        const int row = frame + codebook - prompt.silence_row_count;
        codes[codebook] = row >= 0 ? rows.at(row).at(codebook).get<int32_t>()
                                   : model.config().silence_frame[codebook];
      }
      frames.push_back(codes);
    }
    WriteFile(codes_output_path, nlohmann::json{{"prompt_ids", prompt_ids},
                                                {"rows", rows},
                                                {"frames", frames},
                                                {"samples", samples.size()}}
                                     .dump());
  } else {
  }
  const nlohmann::json line = {
      {"frame_count", result.frame_count},
      {"generated_row_count", result.generated_row_count},
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
