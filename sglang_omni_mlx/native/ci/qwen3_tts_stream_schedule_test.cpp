// SPDX-License-Identifier: Apache-2.0
#include "qwen3_tts/qwen3_tts_codec.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

namespace mx = mlx::core;

struct ScheduleCase {
  std::string name;
  int first_chunk_frames;
  int bridge_chunk_frames;
  int frame_count;
  std::vector<int> chunk_ends;
};

int main(int argc, char **argv) {
  try {
    if (argc != 4) {
      throw std::invalid_argument(
          "expected model directory, existing golden and output directory");
    } else {
    }
    const std::filesystem::path model_directory(argv[1]);
    const std::filesystem::path output_directory(argv[3]);
    std::filesystem::create_directories(output_directory);
    qwen3_tts::SpeechDecoder decoder(model_directory / "speech_tokenizer");
    const auto golden = nlohmann::json::parse(std::ifstream(argv[2]));
    const auto source_rows =
        golden.at("devices").at("Apple M5").at("clips").at("en").at("codes");
    std::vector<mx::array> codes;
    for (const auto &row : source_rows) {
      std::istringstream input(row.get<std::string>());
      std::vector<int32_t> values;
      int32_t value;
      while (input >> value) {
        values.push_back(value);
      }
      if (static_cast<int>(values.size()) != decoder.codebook_count()) {
        throw std::runtime_error(
            "golden frame has the wrong number of codebooks");
      } else {
      }
      codes.emplace_back(values.begin(),
                         mx::Shape{static_cast<int>(values.size())});
    }
    if (codes.empty()) {
      throw std::runtime_error("golden contains no frames");
    } else {
    }
    mx::eval(codes);
    const std::vector<ScheduleCase> cases = {
        {"tail-before-first", 5, 10, 4, {}},
        {"at-first", 5, 10, 5, {5}},
        {"tail-before-bridge", 5, 10, 14, {5}},
        {"at-bridge", 5, 10, 15, {5, 15}},
        {"tail-after-bridge", 5, 10, 16, {5, 15}},
        {"steady", 5, 10, 40, {5, 15, 40}},
        {"first-zero-bridge-one", 0, 1, 52, {25, 26, 51}},
        {"bridge-25", 5, 25, 55, {5, 30, 55}},
        {"default", 5, 0, 31, {5, 30}},
        {"long-default",
         5,
         0,
         461,
         {5, 30, 55, 80, 105, 130, 155, 180, 205, 230, 255, 280, 305, 330, 355,
          380, 405, 430, 455}},
        {"long-bridge",
         5,
         10,
         461,
         {5, 15, 40, 65, 90, 115, 140, 165, 190, 215, 240, 265, 290, 315, 340,
          365, 390, 415, 440}},
    };
    nlohmann::json report = nlohmann::json::array();
    for (const auto &test : cases) {
      qwen3_tts::SpeechStream stream(decoder, true, test.first_chunk_frames,
                                     test.bridge_chunk_frames);
      std::vector<int> ends;
      std::vector<float> audio;
      int previous_end = 0;
      for (int frame = 0; frame < test.frame_count; ++frame) {
        std::vector<float> chunk = stream.Push(codes[frame % codes.size()]);
        if (!chunk.empty()) {
          const int end = frame + 1;
          const size_t expected = static_cast<size_t>(end - previous_end) *
                                  decoder.samples_per_frame();
          if (chunk.size() != expected) {
            throw std::runtime_error(test.name +
                                     " returned the wrong chunk length");
          } else {
          }
          previous_end = end;
          ends.push_back(end);
          audio.insert(audio.end(), chunk.begin(), chunk.end());
        } else {
        }
      }
      std::vector<float> tail = stream.Finish();
      if (ends != test.chunk_ends ||
          tail.size() != static_cast<size_t>(test.frame_count - previous_end) *
                             decoder.samples_per_frame()) {
        throw std::runtime_error(test.name +
                                 " returned the wrong schedule or tail");
      } else {
      }
      audio.insert(audio.end(), tail.begin(), tail.end());
      for (float sample : audio) {
        if (!std::isfinite(sample)) {
          throw std::runtime_error(test.name + " returned non-finite audio");
        } else {
        }
      }
      if (test.name == "long-default" || test.name == "long-bridge") {
        std::ofstream output(output_directory / (test.name + ".f32"),
                             std::ios::binary);
        output.write(reinterpret_cast<const char *>(audio.data()),
                     audio.size() * sizeof(float));
        if (!output) {
          throw std::runtime_error("could not write long codec output");
        } else {
        }
      } else {
      }
      report.push_back({{"case", test.name},
                        {"frames", test.frame_count},
                        {"chunk_ends", ends},
                        {"tail_samples", tail.size()},
                        {"samples", audio.size()}});
    }
    std::cout << report.dump() << '\n';
    return 0;
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
