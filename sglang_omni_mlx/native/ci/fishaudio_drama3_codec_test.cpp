// SPDX-License-Identifier: Apache-2.0
#include "fish_dac/fish_dac.h"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "common/audio.h"
#include "nlohmann/json.hpp"

namespace mx = mlx::core;

int main(int argc, char **argv) {
  try {
    if (argc != 7 && argc != 8) {
      throw std::invalid_argument("expected model, codes, output prefix, block "
                                  "frames, budget, failure");
    }
    fish_dac::AudioCodec codec(argv[1]);
    std::ifstream input(argv[2]);
    nlohmann::json document;
    input >> document;
    const auto frames =
        document.at("codes").get<std::vector<std::vector<int32_t>>>();
    std::vector<mx::array> codes;
    for (const auto &frame : frames) {
      codes.emplace_back(frame.data(),
                         mx::Shape{static_cast<int>(frame.size())}, mx::int32);
    }
    const size_t block_frames = std::stoul(argv[4]);
    if (block_frames == 0) {
      throw std::invalid_argument("block frames must be positive");
    }
    const bool budget = std::string(argv[5]) == "budget";
    const bool failure = std::string(argv[6]) == "failure";
    const size_t caller_limit =
        argc == 8 ? std::stoull(argv[7]) : 192 * 1024 * 1024;
    const size_t original_limit = mx::set_cache_limit(caller_limit);
    fish_dac::CodecStream stream(codec);
    std::vector<float> audio;
    bool caught = false;
    size_t evaluations = 0;
    const auto profile = [&](const char *stage, size_t begin, size_t end) {
      const size_t observed_limit = mx::set_cache_limit(caller_limit);
      mx::set_cache_limit(observed_limit);
      if (observed_limit !=
          (budget ? std::min(caller_limit, size_t(256 * 1024 * 1024))
                  : caller_limit)) {
        throw std::runtime_error("unexpected codec allocator limit");
      }
      std::cerr << nlohmann::json{{"stage", stage},
                                  {"begin", begin},
                                  {"end", end},
                                  {"cache_limit", observed_limit},
                                  {"active", mx::get_active_memory()},
                                  {"cache", mx::get_cache_memory()},
                                  {"peak", mx::get_peak_memory()}}
                       .dump()
                << '\n';
      if (std::string(stage) == "codec_after_eval") {
        ++evaluations;
        if (failure) {
          throw std::runtime_error("codec callback cancellation fixture");
        }
      }
    };
    const auto started = std::chrono::steady_clock::now();
    try {
      const size_t request_frames =
          block_frames == 64 ? codes.size() : block_frames;
      for (size_t begin = 0; begin < codes.size(); begin += request_frames) {
        const auto samples = stream.Decode(
            codes, begin, std::min(codes.size(), begin + request_frames),
            budget, true, profile);
        audio.insert(audio.end(), samples.begin(), samples.end());
        const size_t restored_limit = mx::set_cache_limit(caller_limit);
        if (restored_limit != caller_limit) {
          throw std::runtime_error("codec success leaked allocator limit");
        }
      }
    } catch (const std::runtime_error &error) {
      if (!failure ||
          std::string(error.what()) != "codec callback cancellation fixture") {
        mx::set_cache_limit(original_limit);
        throw;
      }
      caught = true;
    }
    const size_t restored_limit = mx::set_cache_limit(original_limit);
    if (restored_limit != caller_limit || caught != failure ||
        evaluations == 0) {
      throw std::runtime_error("codec restoration or callback check failed");
    }
    if (!failure) {
      std::ofstream wav(std::string(argv[3]) + ".wav", std::ios::binary);
      wav << omni_common::WavBytes(omni_common::Pcm16Bytes(audio), 44100);
      std::ofstream raw(std::string(argv[3]) + ".f32", std::ios::binary);
      raw.write(reinterpret_cast<const char *>(audio.data()),
                audio.size() * sizeof(float));
    }
    std::cout << nlohmann::json{{"frames", frames.size()},
                                {"block_frames", block_frames},
                                {"budget", budget},
                                {"failure_caught", caught},
                                {"restored_limit", restored_limit},
                                {"caller_limit", caller_limit},
                                {"evaluations", evaluations},
                                {"samples", audio.size()},
                                {"seconds",
                                 std::chrono::duration<double>(
                                     std::chrono::steady_clock::now() - started)
                                     .count()}}
                     .dump()
              << '\n';
  } catch (const std::exception &error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
