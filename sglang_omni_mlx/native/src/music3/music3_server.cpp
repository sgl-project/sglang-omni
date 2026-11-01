// SPDX-License-Identifier: Apache-2.0
// Native MiniMax Music3 server: one song is generated at a time and a request
// made meanwhile gets 503. A request not done S seconds (--request-timeout-s,
// 1800 by default) after it is admitted gets 504, and its generation is
// cancelled, as is one whose client disconnects.
#include <stdexcept>
#include <string>

#include "music3/music3_service.h"

namespace {

constexpr int kMaxRequestTimeoutSeconds = 24 * 60 * 60;

} // namespace

int main(int argc, char **argv) {
  bool is_serial_offload_enabled = false;
  mlx::core::set_cache_limit(1024ULL * 1024 * 1024);
  double request_timeout_seconds = 1800;
  return omni_server::Serve(
      argc, argv,
      {{"minimax_music3",
        [&](const std::filesystem::path &directory) {
          return std::make_unique<music3::SpeechService>(
              directory, request_timeout_seconds, is_serial_offload_enabled);
        },
        {{"--request-timeout-s",
          [&](const std::string &value) {
            request_timeout_seconds = std::stod(value);
          }},
         {"--stage-offload-components",
          [&](const std::string &offload_components) {
            music3::ValidateSerialOffload(offload_components);
            is_serial_offload_enabled = true;
          }}},
        [&]() {
          if (!(request_timeout_seconds > 0 &&
                request_timeout_seconds <= kMaxRequestTimeoutSeconds)) {
            throw std::invalid_argument(
                "--request-timeout-s must be above 0 and at most " +
                std::to_string(kMaxRequestTimeoutSeconds));
          } else {
          }
        }}});
}
