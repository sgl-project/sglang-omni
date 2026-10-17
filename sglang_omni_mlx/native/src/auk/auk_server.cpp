// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <memory>
#include <stdexcept>

#include "auk/auk_tts_service.h"

int main(int argc, char **argv) {
  mlx::core::Dtype dtype = mlx::core::bfloat16;
  float max_seconds = 30.0f;
  omni_server::ServedKind kind;
  kind.model_kind = "auk";
  kind.load = [&](const std::filesystem::path &model_directory) {
    return std::make_unique<auk::SpeechService>(model_directory, dtype,
                                                max_seconds);
  };
  kind.flags["--dtype"] = [&](const std::string &value) {
    if (value == "bfloat16") {
      dtype = mlx::core::bfloat16;
    } else if (value == "float32") {
      dtype = mlx::core::float32;
    } else {
      throw std::invalid_argument("--dtype must be bfloat16 or float32");
    }
  };
  kind.flags["--max-seconds"] = [&](const std::string &value) {
    size_t parsed = 0;
    max_seconds = std::stof(value, &parsed);
    if (parsed != value.size() || !std::isfinite(max_seconds) ||
        max_seconds < 0.02f || max_seconds > 30.0f) {
      throw std::invalid_argument("--max-seconds must be in [0.02, 30]");
    } else {
    }
  };
  return omni_server::Serve(argc, argv, {kind});
}
