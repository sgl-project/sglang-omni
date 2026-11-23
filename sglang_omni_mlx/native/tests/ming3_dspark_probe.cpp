// SPDX-License-Identifier: Apache-2.0
// Decodes a small Ming ASR checkpoint with and without a DSpark draft and
// prints one JSON row per drafted request; exits 1 if any differs from plain
// decoding.
#include <atomic>
#include <iostream>

#include "ming3_asr/ming3_asr_transcriber.h"

int main(int argc, char **argv) {
  if (argc != 3) {
    std::cerr << "usage: ming3_dspark_probe CHECKPOINT_DIRECTORY "
                 "DRAFT_DIRECTORY\n";
    return 2;
  } else {
  }
  const ming3_asr::MingModel model(argv[1]);
  const ming3_asr::MingConfig &config = model.config();
  const omni_common::DFlashDraft draft(
      argv[2], 0,
      {config.hidden_size, config.vocabulary_size, config.layer_count});
  const std::atomic<bool> cancel(false);
  const auto embeddings = model.EmbedTokens({1, 2, 3, 4, 5});
  nlohmann::json rows = nlohmann::json::array();
  for (int no_repeat_ngram_size : {0, 2, 3}) {
    for (int budget : {1, 2, 3, 5, 9, 17, 64}) {
      const auto plain = ming3_asr::DecodeGreedy(model, nullptr, embeddings,
                                                 config.eos_token_ids, budget,
                                                 no_repeat_ngram_size, cancel);
      const auto drafted = ming3_asr::DecodeGreedy(
          model, &draft, embeddings, config.eos_token_ids, budget,
          no_repeat_ngram_size, cancel);
      if (drafted.token_ids != plain.token_ids ||
          drafted.finish_reason != plain.finish_reason) {
        std::cerr << "DSpark differs from plain at budget " << budget
                  << " and no_repeat_ngram_size " << no_repeat_ngram_size
                  << '\n';
        return 1;
      } else {
      }
      const auto &statistics = drafted.draft_statistics;
      rows.push_back(
          {{"budget", budget},
           {"no_repeat_ngram_size", no_repeat_ngram_size},
           {"token_ids", drafted.token_ids},
           {"finish_reason",
            omni_common::FinishReasonName(drafted.finish_reason)},
           {"verify_round_count", statistics.verify_round_count},
           {"proposed_token_count", statistics.proposed_token_count},
           {"accepted_token_count", statistics.accepted_token_count}});
    }
  }
  std::cout << rows.dump() << '\n';
}
