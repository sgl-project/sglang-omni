// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen3_omni/generation.h"
namespace qwen3_omni {
struct Arguments {
  std::filesystem::path model_path;
  std::string model_name = "qwen3-omni-mlx";
  std::string host = "127.0.0.1";
  int port = 8000;
  bool supervised = false;
  bool version = false;
  std::vector<std::filesystem::path> requests;
  std::vector<std::filesystem::path> allowed_paths;
};
Arguments parse_arguments(int count, char **values, bool server);
Json version_json();
Completion make_completion(const std::string &model);
Json streaming_payload(const GenerationChunk &chunk,
                       const Completion &completion, bool include_usage);
} // namespace qwen3_omni
