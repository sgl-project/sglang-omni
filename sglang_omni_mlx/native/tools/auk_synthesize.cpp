// SPDX-License-Identifier: Apache-2.0
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <stdexcept>

#include "audio.h"
#include "auk_model.h"

namespace {

struct Arguments {
  std::filesystem::path model_directory;
  std::filesystem::path requests;
  std::filesystem::path output_directory;
  mlx::core::Dtype dtype = mlx::core::bfloat16;
};

Arguments ParseArguments(int argc, char **argv) {
  Arguments arguments;
  for (int i = 1; i < argc; ++i) {
    const std::string flag = argv[i];
    if (i + 1 >= argc) {
      throw std::invalid_argument(flag + " needs a value");
    } else {
    }
    const std::string value = argv[++i];
    if (flag == "--model-path") {
      arguments.model_directory = value;
    } else if (flag == "--requests") {
      arguments.requests = value;
    } else if (flag == "--output-directory") {
      arguments.output_directory = value;
    } else if (flag == "--dtype" && (value == "bf16" || value == "bfloat16")) {
      arguments.dtype = mlx::core::bfloat16;
    } else if (flag == "--dtype" && (value == "fp32" || value == "float32")) {
      arguments.dtype = mlx::core::float32;
    } else {
      throw std::invalid_argument("unsupported argument: " + flag + " " +
                                  value);
    }
  }
  if (arguments.model_directory.empty() || arguments.requests.empty() ||
      arguments.output_directory.empty()) {
    throw std::invalid_argument(
        "--model-path, --requests and --output-directory are required");
  } else {
    return arguments;
  }
}

} // namespace

int main(int argc, char **argv) {
  try {
    const Arguments arguments = ParseArguments(argc, argv);
    std::ifstream requests(arguments.requests);
    if (!requests) {
      throw std::runtime_error("cannot read " + arguments.requests.string());
    } else {
    }
    std::filesystem::create_directories(arguments.output_directory);
    auk::Model model(arguments.model_directory, arguments.dtype, 30.0f);
    std::set<std::string> request_ids;
    std::string line;
    while (std::getline(requests, line)) {
      const auto request = nlohmann::json::parse(line);
      const std::string request_id = request.at("id").get<std::string>();
      if (request_id.empty() || request_id.size() > 128 ||
          request_id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLM"
                                       "NOPQRSTUVWXYZ0123456789_-") !=
              std::string::npos ||
          !request_ids.insert(request_id).second) {
        throw std::invalid_argument("request id must be unique and contain "
                                    "only letters, digits, _ or -");
      } else {
      }
      auk::SynthesisRequest synthesis;
      synthesis.instruction = request.at("instruction").get<std::string>();
      synthesis.seconds = request.at("seconds").get<double>();
      if (!request.at("seed").is_number_integer() || request.at("seed") < 0) {
        throw std::invalid_argument("seed must be a nonnegative integer");
      } else {
        synthesis.seed = request.at("seed").get<uint64_t>();
      }
      if (request.contains("reference_wav")) {
        const auto path = request.at("reference_wav").get<std::string>();
        std::ifstream reference(path, std::ios::binary);
        if (!reference) {
          throw std::runtime_error("cannot read " + path);
        } else {
        }
        const std::string wav_bytes(std::istreambuf_iterator<char>{reference},
                                    {});
        auto audio = qwen3_asr::DecodeAudioWav(wav_bytes);
        synthesis.reference_audio = std::move(audio.samples);
        synthesis.reference_sample_rate = audio.sample_rate;
      } else {
      }
      const auto started = std::chrono::steady_clock::now();
      const auto result = model.Synthesize(synthesis, [] {});
      const double elapsed_seconds =
          std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                        started)
              .count();
      const auto output_path =
          arguments.output_directory / (request_id + ".wav");
      const std::string wav_bytes =
          qwen3_asr::EncodeWavPcm16(result.samples, result.sample_rate);
      std::ofstream output(output_path, std::ios::binary);
      output.write(wav_bytes.data(), wav_bytes.size());
      output.close();
      if (!output) {
        throw std::runtime_error("cannot write " + output_path.string());
      } else {
      }
      std::cout << nlohmann::json({{"id", request_id},
                                   {"file", output_path.string()},
                                   {"sample_rate", result.sample_rate},
                                   {"sample_count", result.samples.size()},
                                   {"elapsed_s", elapsed_seconds}})
                       .dump()
                << std::endl;
      mlx::core::synchronize();
      mlx::core::clear_cache();
    }
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "auk_synthesize: " << error.what() << "\n";
    return 1;
  }
}
