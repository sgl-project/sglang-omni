// SPDX-License-Identifier: Apache-2.0
#include <filesystem>
#include <memory>
#include <utility>

#include "common/service.h"
#include "ming3_audio/ming3_audio_model.h"
#include "ming3_audio/ming3_audio_service.h"

int main(int argc, char **argv) {
  int context_tokens = ming3_audio::kDefaultContextTokens;
  bool speculative = false;
  return omni_server::Serve(
      argc, argv,
      {{"ming3_audio",
        [&](const std::filesystem::path &directory) {
          auto model = std::make_shared<ming3_audio::Model>(
              directory, context_tokens, speculative);
          return ming3_audio::CreateChatService(
              [model](ming3_audio::ChatRequest request) {
                return model->Prepare(std::move(request));
              },
              [model](const ming3_audio::PreparedChatRequest &request,
                      const ming3_audio::Cancellation &cancelled) {
                return model->Generate(request, cancelled);
              });
        },
        {{"--context-length",
          [&](const std::string &value) {
            context_tokens = ming3_audio::ParseContextLength(value);
          }},
         {"--speculative-algorithm",
          [&](const std::string &value) {
            speculative = ming3_audio::ParseSpeculativeAlgorithm(value);
          }}},
        {}}});
}
