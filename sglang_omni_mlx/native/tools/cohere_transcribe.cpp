// SPDX-License-Identifier: Apache-2.0
// Transcribes WAV files with the native Cohere Transcribe runtime and prints
// one JSON line each, for parity checks against Voxt's Swift backend.
//
//   cohere_transcribe --model-path DIR [--language L] [--use-punctuation 0|1]
//     [--max-new-tokens N] [--temperature T] [--chunk-duration S]
//     [--min-chunk-duration S] [--vad-model-directory DIR --vad-threshold P
//     --vad-min-speech-ms MS --vad-min-silence-ms MS --vad-speech-pad-ms MS
//     --vad-merge-gap-seconds S --vad-max-chunk-seconds S] a.wav...
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "audio.h"
#include "cohere_transcriber.h"
#include "form.h"
#include "nlohmann/json.hpp"
#include "speech_segments.h"

int main(int argc, char **argv) {
  std::string model_path;
  std::string vad_model_directory;
  cohere_transcribe::CohereOptions options;
  std::set<std::string> vad_flags;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument.rfind("--", 0) == 0 &&
        (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)) {
      std::cerr << argv[0] << ": missing value for " << argument << "\n";
      return 2;
    }
    const auto value = [&]() { return std::string(argv[++i]); };
    if (argument == "--model-path") {
      model_path = value();
    } else if (argument == "--language") {
      options.language = value();
    } else if (argument == "--use-punctuation") {
      options.use_punctuation = qwen3_asr::FormFlag(value());
    } else if (argument == "--max-new-tokens") {
      options.max_new_tokens = std::stoi(value());
    } else if (argument == "--temperature") {
      options.temperature = std::stof(value());
    } else if (argument == "--chunk-duration") {
      options.chunk_duration_seconds = std::stof(value());
    } else if (argument == "--min-chunk-duration") {
      options.min_chunk_duration_seconds = std::stof(value());
    } else if (argument == "--vad-model-directory") {
      vad_model_directory = value();
    } else if (argument == "--vad-threshold") {
      vad_flags.insert(argument);
      options.speech_segments.threshold = std::stof(value());
    } else if (argument == "--vad-min-speech-ms") {
      vad_flags.insert(argument);
      options.speech_segments.min_speech_milliseconds = std::stoi(value());
    } else if (argument == "--vad-min-silence-ms") {
      vad_flags.insert(argument);
      options.speech_segments.min_silence_milliseconds = std::stoi(value());
    } else if (argument == "--vad-speech-pad-ms") {
      vad_flags.insert(argument);
      options.speech_segments.speech_pad_milliseconds = std::stoi(value());
    } else if (argument == "--vad-merge-gap-seconds") {
      vad_flags.insert(argument);
      options.speech_segments.merge_gap_seconds = std::stof(value());
    } else if (argument == "--vad-max-chunk-seconds") {
      vad_flags.insert(argument);
      options.speech_segments.max_chunk_seconds = std::stof(value());
    } else if (argument.rfind("--", 0) == 0) {
      std::cerr << argv[0] << ": unknown option " << argument << "\n";
      return 2;
    } else {
      files.push_back(argument);
    }
  }
  if (!vad_model_directory.empty() && vad_flags.size() != 6) {
    std::cerr << argv[0] << ": all six --vad-* settings are required\n";
    return 2;
  }
  const auto load_started = std::chrono::steady_clock::now();
  const cohere_transcribe::CohereTranscriber transcriber(model_path);
  std::optional<silero_vad::SileroVAD> voice_activity_detector;
  if (!vad_model_directory.empty()) {
    voice_activity_detector.emplace(vad_model_directory);
    options.voice_activity_detector = &*voice_activity_detector;
  } else {
  }
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
    const nlohmann::json line = {
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
