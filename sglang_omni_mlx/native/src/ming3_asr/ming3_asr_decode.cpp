// SPDX-License-Identifier: Apache-2.0
// Ming ASR's greedy decoding, plain or verifying a DSpark draft's blocks.
#include <algorithm>
#include <limits>
#include <optional>
#include <vector>

#include "ming3_asr/ming3_asr_transcriber.h"

namespace ming3_asr {

namespace mx = mlx::core;

namespace {

constexpr int kPrefillChunkSize = 256;

} // namespace

GreedyDecoding DecodeGreedy(const MingModel &model,
                            const omni_common::DFlashDraft *draft,
                            const mx::array &embeddings,
                            const std::vector<int> &stop_token_ids,
                            int max_new_tokens, int no_repeat_ngram_size,
                            const std::atomic<bool> &cancel) {
  GreedyDecoding decoding;
  std::vector<int> &generated_token_ids = decoding.token_ids;
  // A stop token ends the output without joining it.
  const auto commit = [&](int token_id) {
    if (std::find(stop_token_ids.begin(), stop_token_ids.end(), token_id) !=
        stop_token_ids.end()) {
      decoding.finish_reason = omni_common::FinishReason::kStop;
      return true;
    } else {
      generated_token_ids.push_back(token_id);
      return static_cast<int>(generated_token_ids.size()) == max_new_tokens;
    }
  };
  // The greedy token after the output so far from one position's logits
  // [vocabulary_size] and their argmax, never one that would repeat an
  // ngram of no_repeat_ngram_size tokens.
  const auto next_token_id = [&](const mx::array &logits, int argmax_id) {
    const size_t ngram_token_count = static_cast<size_t>(no_repeat_ngram_size);
    if (ngram_token_count == 0 ||
        generated_token_ids.size() < ngram_token_count) {
      return argmax_id;
    } else {
    }
    std::vector<int> banned_token_ids;
    for (size_t start = 0;
         start + ngram_token_count <= generated_token_ids.size(); ++start) {
      if (std::equal(generated_token_ids.begin() + start,
                     generated_token_ids.begin() + start + ngram_token_count -
                         1,
                     generated_token_ids.end() - (ngram_token_count - 1))) {
        banned_token_ids.push_back(
            generated_token_ids[start + ngram_token_count - 1]);
      } else {
      }
    }
    if (std::find(banned_token_ids.begin(), banned_token_ids.end(),
                  argmax_id) == banned_token_ids.end()) {
      return argmax_id;
    } else {
      return mx::argmax(
                 mx::put_along_axis(
                     logits,
                     mx::array(banned_token_ids.data(),
                               {static_cast<int>(banned_token_ids.size())}),
                     mx::array(-std::numeric_limits<float>::infinity()), 0))
          .item<int>();
    }
  };

  const std::vector<int> aux_layers =
      draft != nullptr ? draft->aux_layers() : std::vector<int>();
  std::vector<omni_common::KVCache> draft_caches =
      draft != nullptr ? draft->NewCaches()
                       : std::vector<omni_common::KVCache>();
  std::vector<mx::array> aux_hidden_states;
  auto cache = model.NewCache();
  std::optional<mx::array> hidden_states;
  for (int start = 0; start < embeddings.shape(1); start += kPrefillChunkSize) {
    const int end = std::min(start + kPrefillChunkSize, embeddings.shape(1));
    aux_hidden_states.clear();
    hidden_states = model.Forward(
        mx::slice(embeddings, {0, start, 0}, {1, end, embeddings.shape(2)}),
        cache, cancel, aux_layers, &aux_hidden_states);
    if (draft != nullptr) {
      draft->AppendContext(aux_hidden_states, end - start, draft_caches);
    } else {
    }
  }
  const mx::array first_logits =
      mx::reshape(model.LastTokenLogits(hidden_states.value()), {-1});
  int anchor_id =
      next_token_id(first_logits, mx::argmax(first_logits).item<int>());
  bool finished = commit(anchor_id);
  if (draft == nullptr) {
    while (!finished) {
      if (cancel.load()) {
        throw omni_common::GenerationCancelled();
      } else {
      }
      const mx::array logits =
          mx::reshape(model.LastTokenLogits(model.Forward(
                          model.EmbedTokens({anchor_id}), cache, cancel)),
                      {-1});
      anchor_id = next_token_id(logits, mx::argmax(logits).item<int>());
      finished = commit(anchor_id);
    }
    return decoding;
  } else {
  }

  cache.KeepLastRound();
  omni_common::DraftStatistics &statistics = decoding.draft_statistics;
  while (!finished) {
    if (cancel.load()) {
      throw omni_common::GenerationCancelled();
    } else {
    }
    const mx::array proposed_ids = draft->ProposeBlock(
        anchor_id, model.EmbedTokens(draft->BlockIds(anchor_id)), draft_caches);
    // note (wilsonzheng0327): the block is cut to the tokens the budget still
    // allows, so the cache never holds a position plain decoding would not
    // have reached.
    const int verify_count =
        std::min(proposed_ids.shape(0) + 1,
                 max_new_tokens - static_cast<int>(generated_token_ids.size()));
    mx::eval(proposed_ids);
    std::vector<int> verify_ids{anchor_id};
    verify_ids.insert(verify_ids.end(), proposed_ids.data<int32_t>(),
                      proposed_ids.data<int32_t>() + verify_count - 1);
    aux_hidden_states.clear();
    const mx::array logits =
        model.Logits(model.Forward(model.EmbedTokens(verify_ids), cache, cancel,
                                   aux_layers, &aux_hidden_states));
    const mx::array argmax_ids = mx::astype(mx::argmax(logits, -1), mx::int32);
    mx::eval(argmax_ids);
    // note (wilsonzheng0327): row i predicts the token after verify_ids[0..i],
    // so its ngram ban is the plain one once proposals 0..i-1 are committed.
    const auto predicted_id = [&](int row) {
      return next_token_id(
          mx::reshape(mx::slice(logits, {row, 0}, {row + 1, logits.shape(1)}),
                      {-1}),
          argmax_ids.data<int32_t>()[row]);
    };
    int accepted_count = 0;
    int token_id = predicted_id(0);
    while (accepted_count < verify_count - 1 &&
           token_id == verify_ids[accepted_count + 1]) {
      ++accepted_count;
      finished = commit(token_id);
      if (finished) {
        break;
      } else {
      }
      token_id = predicted_id(accepted_count);
    }
    if (!finished) {
      anchor_id = token_id;
      finished = commit(anchor_id);
    } else {
    }
    ++statistics.verify_round_count;
    statistics.proposed_token_count += verify_count - 1;
    statistics.accepted_token_count += accepted_count;
    if (!finished) {
      // The cache keeps the anchor and the accepted proposals, whose hidden
      // states become the draft's context.
      model.Rewind(cache, verify_count - 1 - accepted_count);
      draft->AppendContext(aux_hidden_states, accepted_count + 1, draft_caches);
    } else {
    }
  }
  return decoding;
}

} // namespace ming3_asr
