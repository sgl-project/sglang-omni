// SPDX-License-Identifier: Apache-2.0
// Decodes a small Fun-ASR-Nano checkpoint with and without a DFlash draft and
// prints one JSON row per budget; exits 1 if a drafted request differs from
// plain decoding, or a penalized one verifies the draft.
#include <atomic>
#include <cmath>
#include <iostream>

#include "common/generation.h"
#include "funasr/funasr_transcriber.h"
#include "nlohmann/json.hpp"

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: funasr_dflash_probe CHECKPOINT_DIRECTORY "
                 "DRAFT_DIRECTORY\n";
    return 2;
  } else {
  }
  const funasr::FunAsrTranscriber plain(argv[1]);
  const funasr::FunAsrTranscriber drafted(argv[1], argv[2], 0,
                                          omni_common::DraftPolicy::kFixed);
  std::vector<float> samples(16000);
  for (size_t index = 0; index < samples.size(); ++index) {
    samples[index] = 0.1f * std::sin(0.05f * static_cast<float>(index));
  }
  const std::atomic<bool> cancel(false);
  nlohmann::json rows = nlohmann::json::array();
  for (int budget : {1, 2, 3, 7, 8, 9, 17}) {
    funasr::FunAsrOptions options;
    options.max_new_tokens = budget;
    const auto expected = plain.Transcribe(samples, options, cancel);
    const auto actual = drafted.Transcribe(samples, options, cancel);
    if (actual.text != expected.text ||
        actual.generated_token_count != expected.generated_token_count ||
        actual.finish_reason != expected.finish_reason) {
      std::cerr << "the draft differs from plain decoding at budget " << budget
                << '\n';
      return 1;
    } else {
    }
    options.repetition_penalty = 1.5f;
    if (drafted.Transcribe(samples, options, cancel).verify_round_count != 0) {
      std::cerr << "a penalized request verified the draft\n";
      return 1;
    } else {
    }
    rows.push_back(
        {{"budget", budget},
         {"text", actual.text},
         {"tokens", actual.generated_token_count},
         {"finish_reason", omni_common::FinishReasonName(actual.finish_reason)},
         {"verify_rounds", actual.verify_round_count},
         {"accepted_draft_tokens", actual.accepted_draft_token_count}});
  }
  std::cout << rows.dump() << '\n';
  return 0;
}
