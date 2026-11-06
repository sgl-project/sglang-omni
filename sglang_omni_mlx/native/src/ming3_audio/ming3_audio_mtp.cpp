// SPDX-License-Identifier: Apache-2.0
#include "ming3_audio/ming3_audio_model.h"

#include <algorithm>

namespace ming3_audio {
namespace mx = mlx::core;

mx::array Model::PredictNext(const mx::array &hidden_states,
                             const mx::array &next_token_ids,
                             qwen3_asr::KVCache &cache) const {
  const auto prefix = "model.layers." + std::to_string(text_.num_hidden_layers);
  auto hidden = checkpoint_.Linear(
      mx::concatenate({RmsNorm(Embed(next_token_ids), prefix + ".enorm"),
                       RmsNorm(hidden_states, prefix + ".hnorm")},
                      -1),
      prefix + ".eh_proj");
  hidden = mx::add(hidden,
                   LatentAttention(RmsNorm(hidden, prefix + ".input_layernorm"),
                                   prefix + ".attention", cache));
  hidden =
      mx::add(hidden, Moe(RmsNorm(hidden, prefix + ".post_attention_layernorm"),
                          prefix + ".mlp"));
  hidden = RmsNorm(hidden, prefix + ".final_layernorm");
  return mx::reshape(
      checkpoint_.Linear(
          mx::slice(hidden, {0, hidden.shape(1) - 1, 0}, hidden.shape()),
          "lm_head"),
      {-1});
}

ChatResult Model::GenerateWithMtp(const PreparedChatRequest &prepared,
                                  const mx::array &embeddings,
                                  const Cancellation &cancel) const {
  if (cancel()) {
    throw Cancelled();
  } else {
  }
  std::vector<LayerCache> caches(text_.num_hidden_layers);
  qwen3_asr::KVCache draft_cache;
  std::optional<mx::array> hidden_states;
  auto token = mx::argmax(Forward(embeddings, caches, &hidden_states));
  std::vector<int> shifted_prompt(prepared.prompt_ids.begin() + 1,
                                  prepared.prompt_ids.end());
  shifted_prompt.push_back(token.item<int>());
  auto proposal = mx::argmax(
      PredictNext(*hidden_states,
                  mx::array(shifted_prompt.data(),
                            {1, int(shifted_prompt.size())}, mx::int32),
                  draft_cache));
  ChatResult result;
  result.prompt_tokens = prepared.prompt_ids.size();
  std::vector<int> generated;
  const auto commit = [&](int token_id) {
    generated.push_back(token_id);
    if (std::find(stop_tokens_.begin(), stop_tokens_.end(), token_id) !=
        stop_tokens_.end()) {
      result.finish_reason = "stop";
      return true;
    } else {
      return int(generated.size()) == prepared.request.max_new_tokens;
    }
  };
  while (true) {
    if (cancel()) {
      throw Cancelled();
    } else if (commit(token.item<int>())) {
      break;
    } else {
    }
    const int anchor = token.item<int>();
    const int proposed = proposal.item<int>();
    const bool can_propose =
        int(generated.size()) + 1 < prepared.request.max_new_tokens;
    const std::vector<int> block = can_propose
                                       ? std::vector<int>{anchor, proposed}
                                       : std::vector<int>{anchor};
    // note (ruiling): KDA state includes convolution and recurrence, so
    // rejection restores all caches before replay.
    const auto snapshot = caches;
    const auto predictions =
        mx::argmax(Forward(Embed(mx::array(block.data(), {1, int(block.size())},
                                           mx::int32)),
                           caches, &hidden_states, true),
                   -1);
    mx::eval({predictions});
    ++result.verify_round_count;
    result.proposed_draft_token_count += int(can_propose);
    const int corrected = predictions.data<uint32_t>()[0];
    if (can_propose && proposed == corrected) {
      ++result.accepted_draft_token_count;
      if (commit(proposed)) {
        break;
      } else {
      }
      token = mx::array(predictions.data<uint32_t>()[1], mx::uint32);
      const int next_ids[] = {proposed, token.item<int>()};
      proposal = mx::argmax(PredictNext(
          *hidden_states, mx::array(next_ids, {1, 2}, mx::int32), draft_cache));
    } else {
      if (can_propose) {
        caches = snapshot;
        const auto replay =
            Forward(Embed(mx::array(&anchor, {1, 1}, mx::int32)), caches,
                    &hidden_states);
        mx::eval({replay});
      } else {
      }
      token = mx::array(corrected, mx::uint32);
      proposal = mx::argmax(
          PredictNext(*hidden_states, mx::array(&corrected, {1, 1}, mx::int32),
                      draft_cache));
    }
    mx::async_eval({proposal});
  }
  result.text = tokenizer_.Decode(generated, true);
  result.completion_tokens = generated.size();
  return result;
}

} // namespace ming3_audio
