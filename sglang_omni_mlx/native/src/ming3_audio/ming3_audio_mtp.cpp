// SPDX-License-Identifier: Apache-2.0
#include "ming3_audio/ming3_audio_model.h"

#include <stdexcept>

namespace ming3_audio {
namespace mx = mlx::core;

bool ParseSpeculativeAlgorithm(const std::string &value) {
  if (value != "NEXTN") {
    throw std::invalid_argument("--speculative-algorithm must be NEXTN");
  } else {
    return true;
  }
}

mx::array Model::Draft(const mx::array &next_embeddings,
                       const mx::array &hidden,
                       qwen3_asr::KVCache &cache) const {
  const auto prefix = "model.layers." + std::to_string(text_.num_hidden_layers);
  auto drafted = checkpoint_.Linear(
      mx::concatenate({RmsNorm(next_embeddings, prefix + ".enorm"),
                       RmsNorm(hidden, prefix + ".hnorm")},
                      -1),
      prefix + ".eh_proj");
  drafted = mx::add(
      drafted, LatentAttention(RmsNorm(drafted, prefix + ".input_layernorm"),
                               prefix + ".attention", cache));
  drafted = mx::add(drafted,
                    Moe(RmsNorm(drafted, prefix + ".post_attention_layernorm"),
                        prefix + ".mlp"));
  return RmsNorm(drafted, prefix + ".final_layernorm");
}

void Model::GenerateWithDraft(const mx::array &embeddings,
                              std::vector<LayerCache> &caches,
                              const PreparedChatRequest &prepared,
                              const Cancellation &cancel,
                              std::vector<int> &generated) const {
  const int max_tokens = prepared.request.max_new_tokens;
  const auto token_ids = [](const std::vector<int> &ids) {
    return mx::array(ids.data(), {1, int(ids.size())}, mx::int32);
  };
  const auto commit = [&](int token) {
    generated.push_back(token);
    return IsStopToken(token) || int(generated.size()) == max_tokens;
  };
  const auto next_draft = [&](const mx::array &next_ids,
                              const mx::array &hidden,
                              qwen3_asr::KVCache &draft_cache) {
    return mx::reshape(mx::astype(mx::argmax(Logits(LastPosition(Draft(
                                      Embed(next_ids), hidden, draft_cache)))),
                                  mx::int32),
                       {1, 1});
  };
  struct Round {
    mx::array hidden;
    mx::array predictions;
    mx::array draft;
    mx::array following_draft;
  };
  qwen3_asr::KVCache draft_cache;
  const auto submit = [&](const mx::array &anchor, const mx::array &draft) {
    const auto hidden =
        Forward(Embed(mx::concatenate({anchor, draft}, 1)), caches);
    const auto predictions =
        mx::astype(mx::argmax(Logits(hidden), -1), mx::int32);
    const auto following_draft = next_draft(
        mx::concatenate({draft, mx::slice(predictions, {0, 1}, {1, 2})}, 1),
        hidden, draft_cache);
    mx::async_eval({predictions, following_draft});
    return Round{hidden, predictions, draft, following_draft};
  };
  const auto finish_with_one_step = [&](const mx::array &anchor) {
    commit(mx::argmax(Logits(Forward(Embed(anchor), caches))).item<int>());
  };

  const auto prompt_hidden = Forward(embeddings, caches);
  const auto first = mx::reshape(
      mx::astype(mx::argmax(Logits(LastPosition(prompt_hidden))), mx::int32),
      {1, 1});
  const auto &prompt = prepared.prompt_ids;
  const auto first_draft = next_draft(
      mx::concatenate(
          {mx::array(prompt.data() + 1, {1, int(prompt.size()) - 1}, mx::int32),
           first},
          1),
      prompt_hidden, draft_cache);
  for (auto &cache : caches) {
    cache.keeps_verified_blocks = true;
  }
  mx::eval({first, first_draft});
  if (commit(first.item<int>())) {
    return;
  } else if (max_tokens - int(generated.size()) == 1) {
    finish_with_one_step(first);
    return;
  } else {
  }
  auto current = submit(first, first_draft);
  while (true) {
    if (cancel()) {
      throw Cancelled();
    } else {
    }
    // note (altale): the next round assumes this one's draft is accepted, so
    // the GPU runs it while this round is checked. Neither this round nor the
    // next is committed yet, and each commits two tokens at most. A stop draft
    // ends the request if accepted and rolls the next round back if not, so
    // the next round is wasted either way.
    std::optional<Round> next;
    if (int(generated.size()) + 4 <= max_tokens &&
        !IsStopToken(current.draft.item<int>())) {
      next = submit(mx::slice(current.predictions, {0, 1}, {1, 2}),
                    current.following_draft);
    } else {
    }
    mx::eval({current.predictions, current.draft});
    const int drafted = current.draft.item<int>();
    const int32_t *predicted = current.predictions.data<int32_t>();
    if (drafted == predicted[0]) {
      const int bonus = predicted[1];
      Rewind(caches, 2, 0);
      if (commit(drafted) || commit(bonus)) {
        return;
      } else if (next.has_value()) {
        current = *next;
      } else {
        finish_with_one_step(token_ids({bonus}));
        return;
      }
    } else {
      const int corrected = predicted[0];
      if (commit(corrected)) {
        return;
      } else {
      }
      // note (altale): undo the rejected draft and the whole next round; the
      // prediction layer appended two positions for each submitted round.
      Rewind(caches, 1, next.has_value() ? 3 : 1);
      draft_cache.Trim(next.has_value() ? 4 : 2);
      if (max_tokens - int(generated.size()) == 1) {
        finish_with_one_step(token_ids({corrected}));
        return;
      } else {
      }
      current = submit(token_ids({corrected}),
                       next_draft(token_ids({corrected}),
                                  mx::slice(current.hidden, {0, 0, 0},
                                            {1, 1, current.hidden.shape(2)}),
                                  draft_cache));
    }
  }
}

} // namespace ming3_audio
