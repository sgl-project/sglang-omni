// SPDX-License-Identifier: Apache-2.0
// Transcribes WAV files with the native SenseVoice Small runtime and prints
// one JSON line each, for parity checks against Voxt's Swift backend.
//
//   sensevoice_transcribe --model-path DIR [--language L] [--use-itn 0|1]
//     [--vad-model-directory DIR --vad-threshold P --vad-min-speech-ms MS
//     --vad-min-silence-ms MS --vad-speech-pad-ms MS
//     --vad-max-chunk-seconds S --vad-chunk-overlap-seconds S] a.wav...
#include <atomic>
#include <chrono>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "audio.h"
#include "form.h"
#include "nlohmann/json.hpp"
#include "sensevoice_transcriber.h"

int main(int argc, char **argv) {
  std::string model_path;
  std::string vad_model_directory;
  sensevoice::SenseVoiceOptions options;
  std::set<std::string> vad_flags;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument.rfind("--", 0) == 0 &&
        (i + 1 >= argc || std::string(argv[i + 1]).rfind("--", 0) == 0)) {
      std::cerr << argv[0] << ": missing value for " << argument << "\n";
      return 2;
    } else {
    }
    const auto value = [&]() { return std::string(argv[++i]); };
    if (argument == "--model-path") {
      model_path = value();
    } else if (argument == "--language") {
      options.language = value();
    } else if (argument == "--use-itn") {
      options.use_itn = qwen3_asr::FormFlag(value());
    } else if (argument == "--vad-model-directory") {
      vad_model_directory = value();
    } else if (argument == "--vad-threshold") {
      vad_flags.insert(argument);
      options.speech.threshold = std::stof(value());
    } else if (argument == "--vad-min-speech-ms") {
      vad_flags.insert(argument);
      options.speech.min_speech_ms = std::stoi(value());
    } else if (argument == "--vad-min-silence-ms") {
      vad_flags.insert(argument);
      options.speech.min_silence_ms = std::stoi(value());
    } else if (argument == "--vad-speech-pad-ms") {
      vad_flags.insert(argument);
      options.speech.speech_pad_ms = std::stoi(value());
    } else if (argument == "--vad-max-chunk-seconds") {
      vad_flags.insert(argument);
      options.max_chunk_seconds = std::stod(value());
    } else if (argument == "--vad-chunk-overlap-seconds") {
      vad_flags.insert(argument);
      options.chunk_overlap_seconds = std::stod(value());
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
  } else if (!vad_model_directory.empty()) {
    try {
      sensevoice::ChunkSampleCounts(options.max_chunk_seconds,
                                    options.chunk_overlap_seconds);
    } catch (const std::invalid_argument &error) {
      std::cerr << argv[0] << ": " << error.what() << "\n";
      return 2;
    }
  } else {
  }
  const auto load_started = std::chrono::steady_clock::now();
  const sensevoice::SenseVoiceTranscriber transcriber(model_path);
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
  for (const std::string &file : files) {
    std::ifstream stream(file, std::ios::binary);
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    const auto started = std::chrono::steady_clock::now();
    const qwen3_asr::TranscriptionResult result = transcriber.Transcribe(
        qwen3_asr::DecodeWav(bytes.str()), options, cancel);
    const nlohmann::json line = {
        {"file", file},
        {"text", result.text},
        {"language", result.language.has_value()
                         ? nlohmann::json(*result.language)
                         : nlohmann::json()},
        {"generated_token_count", result.generated_token_count},
        {"finish_reason", qwen3_asr::FinishReasonName(result.finish_reason)},
        {"segments", qwen3_asr::SpeakerSegmentsJson(result.segments)},
        {"seconds", std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started)
                        .count()},
    };
    std::cout << line.dump() << std::endl;
  }
  return 0;
}
