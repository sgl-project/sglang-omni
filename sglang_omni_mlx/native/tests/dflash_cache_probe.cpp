// SPDX-License-Identifier: Apache-2.0
// Decodes requests on a small Qwen3 target plainly and with a DFlash draft
// over caches the probe keeps, and prints one JSON row per request: both
// outputs, how many tokens each committed (the stop token included), the
// caches' final lengths and the draft counts. A request's stop_id ends it
// without joining the output; -1 means none.
#include <atomic>
#include <filesystem>
#include <iostream>
#include <vector>

#include "common/dflash_draft.h"
#include "common/kv_cache.h"
#include "common/layers.h"
#include "common/qwen3_decoder.h"
#include "mlx/mlx.h"
#include "nlohmann/json.hpp"

namespace mx = mlx::core;

int main(int argc, char **argv) {
  if (argc != 4) {
    std::cerr << "usage: dflash_cache_probe TARGET_DIRECTORY DRAFT_DIRECTORY "
                 "REQUESTS_JSON\n";
    return 2;
  } else {
  }
  const std::filesystem::path target_directory = argv[1];
  const nlohmann::json config =
      omni_common::ReadJson(target_directory / "config.json");
  const omni_common::Checkpoint checkpoint(
      omni_common::LoadSafetensors(target_directory), config);
  const omni_common::Qwen3Decoder target(
      checkpoint, "", omni_common::Qwen3DecoderConfig::FromJson(config));
  const omni_common::DFlashDraft draft(argv[2], 0, target.config());
  const nlohmann::json requests = omni_common::ReadJson(argv[3]);
  const std::atomic<bool> cancel = false;
  nlohmann::json rows = nlohmann::json::array();
  for (const nlohmann::json &request : requests) {
    const std::vector<int> prompt_ids =
        request.at("prompt_ids").get<std::vector<int>>();
    const std::vector<int> guessed_ids =
        request.at("guessed_ids").get<std::vector<int>>();
    const int budget = request.at("budget").get<int>();
    const int stop_id = request.at("stop_id").get<int>();
    const int prefill_step_tokens =
        request.at("prefill_step_tokens").get<int>();
    const mx::array embeddings = target.EmbedTokens(
        mx::array(prompt_ids.data(), {1, static_cast<int>(prompt_ids.size())},
                  mx::int32));

    std::vector<int> plain_ids;
    int plain_commit_count = 0;
    const auto plain_commit = [&](int token_id) {
      ++plain_commit_count;
      if (token_id == stop_id) {
        return true;
      } else {
        plain_ids.push_back(token_id);
        return static_cast<int>(plain_ids.size()) == budget;
      }
    };
    std::vector<omni_common::KVCache> plain_caches = target.NewCaches();
    mx::array next_token =
        mx::argmax(target.LastLogits(target.Forward(embeddings, plain_caches)));
    target.DecodePlain(next_token, plain_caches, 1.0f, plain_commit, cancel);

    std::vector<int> drafted_ids;
    int drafted_commit_count = 0;
    const auto drafted_commit = [&](int token_id) {
      ++drafted_commit_count;
      if (token_id == stop_id) {
        return true;
      } else {
        drafted_ids.push_back(token_id);
        return static_cast<int>(drafted_ids.size()) == budget;
      }
    };
    std::vector<omni_common::KVCache> caches = target.NewCaches();
    std::vector<omni_common::KVCache> draft_caches = draft.NewCaches();
    const omni_common::DraftStatistics statistics =
        draft.Decode(target, embeddings, guessed_ids, budget,
                     prefill_step_tokens, drafted_commit, cancel,
                     omni_common::DraftPolicy::kFixed, caches, draft_caches);
    rows.push_back({{"plain_ids", plain_ids},
                    {"drafted_ids", drafted_ids},
                    {"plain_commit_count", plain_commit_count},
                    {"drafted_commit_count", drafted_commit_count},
                    {"target_cache_length", caches[0].offset()},
                    {"draft_cache_length", draft_caches[0].offset()},
                    {"verify_round_count", statistics.verify_round_count},
                    {"proposed_token_count", statistics.proposed_token_count},
                    {"accepted_token_count", statistics.accepted_token_count}});
  }
  std::cout << rows.dump() << '\n';
}
