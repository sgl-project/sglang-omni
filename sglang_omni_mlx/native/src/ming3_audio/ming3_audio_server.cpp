// SPDX-License-Identifier: Apache-2.0
#include <filesystem>
#include <memory>

#include "common/service.h"
#include "ming3_audio/ming3_audio_model.h"
#include "ming3_audio/ming3_audio_service.h"

int main(int argc, char **argv) {
  return omni_server::Serve(
      argc, argv,
      {{"ming3_audio",
        [](const std::filesystem::path &directory) {
          return ming3_audio::CreateChatService(
              [model = std::make_shared<ming3_audio::Model>(directory)](
                  const ming3_audio::ChatRequest &request,
                  const ming3_audio::Cancellation &cancelled) {
                return model->Generate(request, cancelled);
              });
        },
        {},
        {}}});
}
