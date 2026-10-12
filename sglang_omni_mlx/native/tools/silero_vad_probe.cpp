// SPDX-License-Identifier: Apache-2.0
// Writes Silero VAD probabilities (and speech ranges with --profiles) of WAV
// files, for parity checks against Voxt's Swift MLXAudioVAD.
//   silero_vad_probe --model-path DIR --out DIR [--profiles FILE] a.wav...
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "audio.h"
#include "nlohmann/json.hpp"
#include "silero_vad.h"

namespace {

void Write(const std::vector<float> &values,
           const std::filesystem::path &path) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
}

} // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::filesystem::path out;
  std::map<std::string, silero_vad::TimestampOptions> profiles;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--model-path") {
      model_path = argv[++i];
    } else if (argument == "--out") {
      out = argv[++i];
    } else if (argument == "--profiles") {
      std::ifstream profile_stream(argv[++i]);
      const nlohmann::json parsed = nlohmann::json::parse(profile_stream);
      for (const auto &profile : parsed.items()) {
        const nlohmann::json &values = profile.value();
        profiles[profile.key()] = {
            values.at("threshold").get<float>(),
            values.at("min_speech_duration_ms").get<int>(),
            values.at("min_silence_duration_ms").get<int>(),
            values.at("speech_pad_ms").get<int>()};
      }
    } else {
      files.push_back(argument);
    }
  }
  std::filesystem::create_directories(out);
  const silero_vad::SileroVAD vad(model_path);
  for (const auto &file : files) {
    std::ifstream stream(file, std::ios::binary);
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    const std::vector<float> samples = qwen3_asr::DecodeWav(bytes.str());
    silero_vad::StreamState state;
    std::vector<float> streamed;
    for (size_t offset = 0;
         offset + silero_vad::kChunkSamples <= samples.size();
         offset += silero_vad::kChunkSamples) {
      streamed.push_back(vad.Feed(samples.data() + offset, state));
    }
    const std::string stem = std::filesystem::path(file).stem().string();
    Write(streamed, out / (stem + ".stream.f32"));
    const std::vector<float> batch = vad.PredictProbabilities(samples);
    Write(batch, out / (stem + ".batch.f32"));
    if (!profiles.empty()) {
      nlohmann::json timestamps = nlohmann::json::object();
      for (const auto &[name, options] : profiles) {
        timestamps[name] = nlohmann::json::array();
        for (const silero_vad::Timestamp &range :
             silero_vad::ProbabilitiesToTimestamps(
                 batch, static_cast<long>(samples.size()), options)) {
          timestamps[name].push_back({range.start, range.end});
        }
      }
      std::ofstream(out / (stem + ".timestamps.json")) << timestamps.dump();
    } else {
    }
  }
  return 0;
}
