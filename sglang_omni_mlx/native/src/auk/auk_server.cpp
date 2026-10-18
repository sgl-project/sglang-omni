// SPDX-License-Identifier: Apache-2.0
#include <cmath>
#include <limits>
#include <memory>
#include <stdexcept>

#include "auk/auk_tts_service.h"

int main(int argc, char **argv) {
  constexpr size_t GIBIBYTE = size_t{1} << 30;
  mlx::core::Dtype dtype = mlx::core::bfloat16;
  float max_seconds = 30.0f;
  size_t cache_limit_bytes = GIBIBYTE;
  omni_server::ServedKind kind;
  kind.model_kind = "auk";
  kind.load = [&](const std::filesystem::path &model_directory) {
    mlx::core::set_cache_limit(cache_limit_bytes);
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
  kind.flags["--cache-limit-gb"] = [&](const std::string &value) {
    size_t parsed = 0;
    const double cache_limit_gibibytes = std::stod(value, &parsed);
    const double requested_cache_bytes = cache_limit_gibibytes * GIBIBYTE;
    if (parsed != value.size() || !std::isfinite(cache_limit_gibibytes) ||
        cache_limit_gibibytes < 0 ||
        requested_cache_bytes >=
            static_cast<double>(std::numeric_limits<size_t>::max())) {
      throw std::invalid_argument("--cache-limit-gb must be finite, "
                                  "non-negative, and fit in size_t bytes");
    } else {
      cache_limit_bytes = static_cast<size_t>(requested_cache_bytes);
    }
  };
  return omni_server::Serve(argc, argv, {kind});
}
