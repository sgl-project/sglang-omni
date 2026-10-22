// SPDX-License-Identifier: Apache-2.0
#include "music3/music3_service.h"

int main(int argc, char **argv) {
  mlx::core::set_cache_limit(1024ULL * 1024 * 1024);
  return omni_server::Serve(argc, argv,
                            {{"minimax_music3",
                              [](const std::filesystem::path &directory) {
                                return std::make_unique<music3::SpeechService>(
                                    directory);
                              },
                              {},
                              {}}});
}
