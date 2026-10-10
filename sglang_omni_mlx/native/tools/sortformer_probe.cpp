// SPDX-License-Identifier: Apache-2.0
// Runs Sortformer over 16 kHz WAV files with Voxt's meeting feed policy and
// writes each file's probabilities, segments and state, for parity checks
// against Voxt's Swift MLXAudioVAD.
#include <algorithm>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "audio.h"
#include "nlohmann/json.hpp"
#include "sortformer.h"

namespace {

void Write(const std::vector<float> &values,
           const std::filesystem::path &path) {
  std::ofstream out(path, std::ios::binary);
  out.write(reinterpret_cast<const char *>(values.data()),
            static_cast<std::streamsize>(values.size() * sizeof(float)));
}

// Note (Jiaxin Deng): mirrors Voxt MeetingSpeakerFeedPolicy so the probe feeds
// what the app feeds.
int SamplesPerFeed(const sortformer::Config &config) {
  const int frame_samples =
      config.processor.hop_length * config.fc_encoder.subsampling_factor;
  const int five_second_frames =
      config.processor.sampling_rate * 5 / frame_samples;
  const int frames = std::min({five_second_frames, config.modules.chunk_len,
                               config.modules.use_aosc
                                   ? config.modules.spkcache_update_period - 2
                                   : config.modules.chunk_len});
  return frames * frame_samples;
}

} // namespace

int main(int argc, char **argv) {
  std::string model_path;
  std::filesystem::path out;
  bool dump_features = false;
  std::vector<std::string> files;
  for (int i = 1; i < argc; ++i) {
    const std::string argument = argv[i];
    if (argument == "--model-path") {
      model_path = argv[++i];
    } else if (argument == "--out") {
      out = argv[++i];
    } else if (argument == "--dump-features") {
      dump_features = true;
    } else {
      files.push_back(argument);
    }
  }
  std::filesystem::create_directories(out);
  const sortformer::SortformerModel model(model_path);
  const sortformer::Config &config = model.config();
  Write(model.features().mel_filters(), out / "mel_filters.f32");
  Write(model.features().window(), out / "window.f32");
  const int samples_per_feed = SamplesPerFeed(config);
  const size_t frame_samples = static_cast<size_t>(
      config.processor.hop_length * config.fc_encoder.subsampling_factor);
  const sortformer::FeedOptions options;
  for (const auto &file : files) {
    std::ifstream stream(file, std::ios::binary);
    std::ostringstream bytes;
    bytes << stream.rdbuf();
    const std::vector<float> samples = qwen3_asr::DecodeWav(bytes.str());
    const std::string stem = std::filesystem::path(file).stem().string();
    sortformer::StreamingState state = model.InitStreamingState();
    std::vector<float> probabilities;
    nlohmann::json segments = nlohmann::json::array();
    nlohmann::json feed_milliseconds = nlohmann::json::array();
    for (size_t offset = 0; offset < samples.size();
         offset += samples_per_feed) {
      const size_t end = std::min(offset + samples_per_feed, samples.size());
      std::vector<float> chunk(samples.begin() + offset, samples.begin() + end);
      if (chunk.size() < frame_samples) {
        chunk.resize(frame_samples, 0.0f);
      } else {
      }
      if (dump_features && offset == 0) {
        // Note (Jiaxin Deng): the extractor returns a transposed view; copy it
        // row-major.
        const mlx::core::array features =
            mlx::core::contiguous(model.features()(chunk));
        mlx::core::eval(features);
        const float *data = features.data<float>();
        Write(std::vector<float>(data, data + features.size()),
              out / (stem + ".features0.f32"));
      } else {
      }
      const auto started = std::chrono::steady_clock::now();
      const sortformer::FeedResult result = model.Feed(chunk, state, options);
      feed_milliseconds.push_back(
          std::chrono::duration<double, std::milli>(
              std::chrono::steady_clock::now() - started)
              .count());
      probabilities.insert(probabilities.end(), result.probabilities.begin(),
                           result.probabilities.end());
      for (const sortformer::Segment &segment : result.segments) {
        segments.push_back({{"start", segment.start},
                            {"end", segment.end},
                            {"speaker", segment.speaker}});
      }
    }
    Write(probabilities, out / (stem + ".probs.f32"));
    std::ofstream(out / (stem + ".segments.json")) << segments.dump() << "\n";
    const nlohmann::json state_stats = {
        {"fifo_length", state.fifo_length()},
        {"spkcache_length", state.spkcache_length()},
        {"frames_processed", state.frames_processed},
        {"feed_count", feed_milliseconds.size()},
        {"feed_ms", feed_milliseconds}};
    std::ofstream(out / (stem + ".state.json")) << state_stats.dump() << "\n";
    std::cout << stem << " feeds=" << feed_milliseconds.size()
              << " frames=" << state.frames_processed
              << " segments=" << segments.size() << std::endl;
  }
  return 0;
}
