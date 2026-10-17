// SPDX-License-Identifier: Apache-2.0
#include "service.h"
#include "tts_service.h"

int main(int argc, char **argv) {
  omni_server::ServedKind kind;
  kind.model_kind = "ming_tts";
  kind.load = [](const std::filesystem::path &model_directory) {
    return std::make_unique<ming_tts::MingSpeechService>(model_directory);
  };
  return omni_server::Serve(argc, argv, {kind});
}
