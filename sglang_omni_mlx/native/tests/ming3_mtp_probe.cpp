// SPDX-License-Identifier: Apache-2.0
#include <iostream>

#include "ming3_audio/ming3_audio_model.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    return 2;
  } else {
  }
  const ming3_audio::Model plain(argv[1], 1024);
  const ming3_audio::Model speculative(argv[1], 1024, true);
  nlohmann::json rows = nlohmann::json::array();
  for (int budget : {1, 2, 3, 17, 64}) {
    ming3_audio::ChatRequest request;
    request.samples = std::vector<float>(400, 0.01f);
    request.max_new_tokens = budget;
    const ming3_audio::PreparedChatRequest prepared{
        request, {1, 2, 3, 4}, 1, 1};
    const auto expected = plain.Generate(prepared, [] { return false; });
    for (int repeat = 0; repeat < 2; ++repeat) {
      const auto actual = speculative.Generate(prepared, [] { return false; });
      if (expected.text != actual.text ||
          expected.finish_reason != actual.finish_reason ||
          expected.completion_tokens != actual.completion_tokens) {
        std::cerr << "MTP differs from plain at budget " << budget << '\n';
        return 1;
      } else {
      }
      rows.push_back({{"budget", budget},
                      {"accepted", actual.accepted_draft_token_count},
                      {"proposed", actual.proposed_draft_token_count},
                      {"rounds", actual.verify_round_count},
                      {"tokens", actual.completion_tokens},
                      {"finish_reason", actual.finish_reason}});
    }
    int calls = 0;
    const int cancel_after = expected.completion_tokens > 3 ? 5 : 2;
    try {
      speculative.Generate(prepared, [&] { return ++calls >= cancel_after; });
      return 1;
    } catch (const ming3_audio::Cancelled &) {
    }
  }
  std::cout << rows.dump() << '\n';
}
