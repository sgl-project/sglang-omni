// SPDX-License-Identifier: Apache-2.0
// Decodes a small checkpoint with and without NEXTN and prints one JSON row
// per speculative request; exits 1 if any differs from plain decoding.
#include <iostream>

#include "ming3_audio/ming3_audio_model.h"

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: ming3_mtp_probe CHECKPOINT_DIRECTORY\n";
    return 2;
  } else {
  }
  const ming3_audio::Model plain(argv[1], 1024,
                                 ming3_audio::SpeculativeAlgorithm::kNone);
  const ming3_audio::Model speculative(
      argv[1], 1024, ming3_audio::SpeculativeAlgorithm::kNextN);
  nlohmann::json rows = nlohmann::json::array();
  for (int budget : {1, 2, 3, 4, 5, 17, 64}) {
    ming3_audio::ChatRequest request;
    request.samples = std::vector<float>(400, 0.01f);
    request.max_new_tokens = budget;
    const ming3_audio::PreparedChatRequest prepared{
        request, {1, 2, 3, 4}, 1, 1};
    const auto expected = plain.Generate(prepared, [] { return false; });
    // note (altale): the fourth check is the second decode round, so a long
    // request is cancelled with a speculative round in flight.
    int calls = 0;
    try {
      speculative.Generate(prepared, [&] { return ++calls >= 4; });
      if (expected.completion_tokens > 5) {
        std::cerr << "NEXTN ignored cancellation at budget " << budget << '\n';
        return 1;
      } else {
      }
    } catch (const ming3_audio::Cancelled &) {
    }
    for (int repeat = 0; repeat < 2; ++repeat) {
      const auto actual = speculative.Generate(prepared, [] { return false; });
      if (actual.text != expected.text ||
          actual.finish_reason != expected.finish_reason ||
          actual.completion_tokens != expected.completion_tokens) {
        std::cerr << "NEXTN differs from plain at budget " << budget << '\n';
        return 1;
      } else {
      }
      rows.push_back({{"budget", budget},
                      {"tokens", actual.completion_tokens},
                      {"finish_reason", actual.finish_reason},
                      {"draft_tokens", actual.draft_tokens},
                      {"accepted_draft_tokens", actual.accepted_draft_tokens}});
    }
  }
  std::cout << rows.dump() << '\n';
}
