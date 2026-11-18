// SPDX-License-Identifier: Apache-2.0
#include "common/dflash_draft.h"

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "common/draft_backoff.h"
#include "common/generation.h"
#include "common/kv_cache.h"

namespace omni_common {

namespace mx = mlx::core;

struct LoadedDraft {
  nlohmann::json config;
  omni_common::WeightMap weights;
};

namespace {

constexpr int kQuantizationGroupSize = 64;
// note (ZaneZhang-67): warm one useful plain token, then time four pipelined
// steps.
constexpr int kPlainProbeTokenCount = 5;

std::string ShapeText(const mx::Shape &shape) {
  std::string text = "[";
  for (size_t i = 0; i < shape.size(); ++i) {
    text += (i > 0 ? ", " : "") + std::to_string(shape[i]);
  }
  return text + "]";
}

// The draft's config and weights once they are known to fit the target, whose
// hidden states it reads and whose tokens it embeds and predicts.
LoadedDraft CheckedDraft(const std::filesystem::path &directory,
                         const Qwen3DecoderConfig &target) {
  const std::string draft = "the draft at " + directory.string();
  nlohmann::json config = omni_common::ReadJson(directory / "config.json");
  const int hidden_size =
      config.at("transformer_layer_config").at("hidden_size").get<int>();
  const int mask_token_id = config.at("mask_token_id").get<int>();
  const int block_size = config.at("block_size").get<int>();
  const int draft_vocab_size = config.at("draft_vocab_size").get<int>();
  const std::vector<int> aux_layers =
      config.at("aux_hidden_state_layer_ids").get<std::vector<int>>();
  if (hidden_size != target.hidden_size) {
    throw std::runtime_error(
        draft + " has hidden size " + std::to_string(hidden_size) +
        ", the target's is " + std::to_string(target.hidden_size));
  } else if (mask_token_id >= target.vocab_size) {
    throw std::runtime_error(draft + " masks with token " +
                             std::to_string(mask_token_id) +
                             ", past the target's vocabulary of " +
                             std::to_string(target.vocab_size));
  } else if (config.at("sample_from_anchor").get<bool>()) {
    throw std::runtime_error(draft + " samples from the anchor");
  } else if (config.at("sliding_window_non_causal").get<bool>()) {
    throw std::runtime_error(
        draft + " was trained with a bidirectional block; blocks are decoded "
                "causally");
  } else if (block_size < 2) {
    throw std::runtime_error(draft + " has block size " +
                             std::to_string(block_size) +
                             ", the anchor needs a proposal after it");
  } else if (aux_layers.empty()) {
    throw std::runtime_error(draft + " reads no target layer");
  } else if (std::adjacent_find(aux_layers.begin(), aux_layers.end(),
                                std::greater_equal<int>()) !=
             aux_layers.end()) {
    // note (wilsonzheng0327): the target captures hidden states in layer
    // order, which is the order fc was trained on.
    throw std::runtime_error(draft + " lists its target layers out of order");
  } else {
  }
  for (const int layer : aux_layers) {
    if (layer < 0 || layer >= target.num_hidden_layers) {
      throw std::runtime_error(draft + " reads target layer " +
                               std::to_string(layer) + " of " +
                               std::to_string(target.num_hidden_layers));
    } else {
    }
  }
  omni_common::WeightMap weights = omni_common::LoadSafetensors(directory);
  const auto expect_shape = [&](const std::string &name,
                                const mx::Shape &expected) {
    const auto found = weights.find(name);
    if (found == weights.end()) {
      throw std::runtime_error(draft + " lacks " + name);
    } else if (found->second.shape() != expected) {
      throw std::runtime_error(draft + " stores " + name + " as " +
                               ShapeText(found->second.shape()) +
                               ", the target needs " + ShapeText(expected));
    } else {
    }
  };
  expect_shape("fc.weight", {hidden_size, static_cast<int>(aux_layers.size()) *
                                              hidden_size});
  expect_shape("lm_head.weight", {draft_vocab_size, hidden_size});
  expect_shape("d2t", {draft_vocab_size});
  const mx::array target_ids = mx::astype(
      mx::add(weights.at("d2t"), mx::arange(draft_vocab_size)), mx::int32);
  mx::eval(target_ids);
  const auto [lowest, highest] =
      std::minmax_element(target_ids.data<int32_t>(),
                          target_ids.data<int32_t>() + target_ids.size());
  if (*lowest < 0 || *highest >= target.vocab_size) {
    throw std::runtime_error(draft + " maps its vocabulary to target tokens " +
                             std::to_string(*lowest) + " to " +
                             std::to_string(*highest) +
                             ", outside the target's vocabulary of " +
                             std::to_string(target.vocab_size));
  } else {
  }
  return {std::move(config), std::move(weights)};
}

} // namespace

DFlashDraft::DFlashDraft(const std::filesystem::path &directory,
                         int quantization_bits,
                         const Qwen3DecoderConfig &target)
    : DFlashDraft(CheckedDraft(directory, target), quantization_bits) {}

DFlashDraft::DFlashDraft(LoadedDraft draft, int quantization_bits)
    : checkpoint_(omni_common::QuantizedCheckpoint(
          std::move(draft.weights), nlohmann::json::object(),
          kQuantizationGroupSize,
          [quantization_bits](const std::string &) {
            return quantization_bits;
          })),
      decoder_(checkpoint_, "",
               Qwen3DecoderConfig::FromJson(
                   draft.config.at("transformer_layer_config"))),
      aux_layers_(draft.config.at("aux_hidden_state_layer_ids")
                      .get<std::vector<int>>()),
      block_size_(draft.config.at("block_size").get<int>()),
      mask_token_id_(draft.config.at("mask_token_id").get<int>()) {}

// note (wilsonzheng0327): the draft's trained sliding window is not enforced:
// on a 194 s recording a 1.7B draft trained with a 2,048 window accepts 3.8
// tokens a round with the whole context and 3.2 with the window, and the
// transcript is the target's either way.
void DFlashDraft::AppendContext(const std::vector<mx::array> &aux_hidden_states,
                                int token_count,
                                std::vector<KVCache> &caches) const {
  std::vector<mx::array> committed;
  for (const mx::array &hidden_states : aux_hidden_states) {
    committed.push_back(mx::slice(hidden_states, {0, 0, 0},
                                  {1, token_count, hidden_states.shape(2)}));
  }
  const mx::array context = checkpoint_.RmsNorm(
      checkpoint_.Linear(mx::concatenate(committed, -1), "fc"), "hidden_norm",
      decoder_.config().rms_norm_eps);
  decoder_.AppendKeysValues(context, caches);
}

mx::array DFlashDraft::ProposeBlock(const mx::array &block_embeddings,
                                    std::vector<KVCache> &caches) const {
  // note (wilsonzheng0327): the block attends causally, as the pinned drafts
  // were trained (sliding_window_non_causal is false, checked at load). A
  // bidirectional block, as in the DFlash reference, lowers the acceptance on
  // the 392-clip corpus: 4.86 to 4.42 a round on the 0.6B draft and 5.50 to
  // 4.95 on the 1.7B (measured by Jiaxin Deng), with the same transcripts.
  const mx::array hidden = decoder_.Forward(block_embeddings, caches);
  // note (wilsonzheng0327): only context keys and values stay in the draft
  // caches; the block's own are dropped once it has been proposed.
  for (KVCache &cache : caches) {
    cache.Trim(block_size_);
  }
  const mx::array proposals = checkpoint_.RmsNorm(
      mx::slice(hidden, {0, 1, 0}, {1, block_size_, hidden.shape(2)}), "norm",
      decoder_.config().rms_norm_eps);
  const mx::array draft_ids =
      mx::argmax(checkpoint_.Linear(proposals, "lm_head"), -1);
  // note (wilsonzheng0327): the draft predicts a reduced vocabulary; d2t
  // holds each draft id's offset to its target id.
  const mx::array target_ids =
      mx::add(draft_ids, mx::take(checkpoint_.Weight("d2t"), draft_ids, 0));
  return mx::reshape(mx::astype(target_ids, mx::int32), {block_size_ - 1});
}

DraftPolicy ParseDraftPolicy(const std::string &value) {
  if (value == "adaptive") {
    return DraftPolicy::kAdaptive;
  } else if (value == "fixed") {
    return DraftPolicy::kFixed;
  } else {
    throw std::invalid_argument("--draft-policy must be adaptive or fixed");
  }
}

DraftStatistics DFlashDraft::Decode(const Qwen3Decoder &target,
                                    const mx::array &embeddings,
                                    int max_new_tokens, int prefill_step_tokens,
                                    const std::function<bool(int)> &commit,
                                    const std::atomic<bool> &cancel,
                                    DraftPolicy policy) const {
  std::vector<KVCache> caches = target.NewCaches();
  std::vector<KVCache> draft_caches = decoder_.NewCaches();
  return Decode(target, embeddings, {}, max_new_tokens, prefill_step_tokens,
                commit, cancel, policy, caches, draft_caches);
}

DraftStatistics DFlashDraft::Decode(
    const Qwen3Decoder &target, const mx::array &embeddings,
    const std::vector<int> &guessed_ids, int max_new_tokens,
    int prefill_step_tokens, const std::function<bool(int)> &commit,
    const std::atomic<bool> &cancel, DraftPolicy policy,
    std::vector<KVCache> &caches, std::vector<KVCache> &draft_caches) const {
  std::vector<mx::array> aux_hidden_states;
  int processed = 0;
  while (prefill_step_tokens > 0 && embeddings.shape(1) - processed > 1) {
    if (cancel.load()) {
      throw GenerationCancelled();
    } else {
    }
    const int token_count =
        std::min(prefill_step_tokens, embeddings.shape(1) - processed - 1);
    const mx::array hidden = target.Forward(
        mx::slice(embeddings, {0, processed, 0},
                  {1, processed + token_count, embeddings.shape(2)}),
        caches, aux_layers_, &aux_hidden_states);
    AppendContext(aux_hidden_states, token_count, draft_caches);
    std::vector<mx::array> evaluated = {hidden};
    for (const KVCache &cache : draft_caches) {
      const auto arrays = cache.Arrays();
      evaluated.insert(evaluated.end(), arrays.begin(), arrays.end());
    }
    mx::eval(evaluated);
    aux_hidden_states.clear();
    processed += token_count;
  }
  if (cancel.load()) {
    throw GenerationCancelled();
  } else {
  }
  const int guessed_count = static_cast<int>(guessed_ids.size());
  const int prompt_count = embeddings.shape(1) - processed;
  bool finished = false;
  int committed_token_count = 0;
  int anchor_id;
  {
    mx::array prompt =
        mx::slice(embeddings, {0, processed, 0}, embeddings.shape());
    if (guessed_count > 0) {
      prompt = mx::concatenate(
          {prompt, target.EmbedTokens(mx::array(
                       guessed_ids.data(), {1, guessed_count}, mx::int32))},
          1);
    } else {
    }
    const mx::array prompt_hidden =
        target.Forward(prompt, caches, aux_layers_, &aux_hidden_states);
    // The prediction after the prompt, then after each guess.
    const mx::array predicted = mx::argmax(
        target.Logits(mx::slice(
            prompt_hidden, {0, prompt_count - 1, 0},
            {1, prompt_count + guessed_count, prompt_hidden.shape(2)})),
        -1);
    mx::eval(predicted);
    const uint32_t *predicted_ids = predicted.data<uint32_t>();
    int accepted_count = 0;
    while (!finished && accepted_count < guessed_count &&
           static_cast<int>(predicted_ids[accepted_count]) ==
               guessed_ids[accepted_count]) {
      finished = commit(guessed_ids[accepted_count]);
      ++accepted_count;
    }
    for (KVCache &cache : caches) {
      cache.Trim(guessed_count - accepted_count);
    }
    AppendContext(aux_hidden_states, prompt_count + accepted_count,
                  draft_caches);
    committed_token_count = accepted_count;
    anchor_id = static_cast<int>(predicted_ids[accepted_count]);
  }
  if (cancel.load()) {
    throw GenerationCancelled();
  } else {
  }
  if (!finished) {
    finished = commit(anchor_id);
    ++committed_token_count;
  } else {
  }
  DraftBackoff backoff;
  DraftStatistics statistics;
  while (!finished) {
    if (cancel.load()) {
      throw GenerationCancelled();
    } else {
    }
    const auto round_started = std::chrono::steady_clock::now();
    std::vector<int> block_ids(block_size_, mask_token_id_);
    block_ids[0] = anchor_id;
    const mx::array proposed_ids =
        ProposeBlock(target.EmbedTokens(mx::array(block_ids.data(),
                                                  {1, block_size_}, mx::int32)),
                     draft_caches);
    // note (wilsonzheng0327): the block is cut to the tokens the budget still
    // allows, so the caches never hold a position plain decoding would not
    // have reached.
    const int verify_count =
        std::min(block_size_, max_new_tokens - committed_token_count);
    const mx::array verify_ids =
        mx::slice(mx::concatenate(
                      {mx::array(&anchor_id, {1}, mx::int32), proposed_ids}, 0),
                  {0}, {verify_count});
    aux_hidden_states.clear();
    const mx::array hidden = target.Forward(
        target.EmbedTokens(mx::reshape(verify_ids, {1, verify_count})), caches,
        aux_layers_, &aux_hidden_states);
    const mx::array predictions = mx::argmax(target.Logits(hidden), -1);
    mx::eval({proposed_ids, predictions});
    // note (wilsonzheng0327): prediction i follows block token i, so proposal
    // i is accepted while every earlier one was and it matches prediction i;
    // the prediction after the last accepted token is committed as well.
    const int32_t *proposed = proposed_ids.data<int32_t>();
    const uint32_t *predicted = predictions.data<uint32_t>();
    int accepted_count = 0;
    while (!finished && accepted_count < verify_count - 1 &&
           proposed[accepted_count] ==
               static_cast<int>(predicted[accepted_count])) {
      finished = commit(proposed[accepted_count]);
      ++accepted_count;
    }
    if (!finished) {
      anchor_id = static_cast<int>(predicted[accepted_count]);
      finished = commit(anchor_id);
    } else {
    }
    committed_token_count += accepted_count + 1;
    ++statistics.verify_round_count;
    statistics.proposed_token_count += verify_count - 1;
    statistics.accepted_token_count += accepted_count;
    // note (wilsonzheng0327): the caches keep the anchor and the accepted
    // proposals; the committed positions' hidden states become the draft's
    // context.
    for (KVCache &cache : caches) {
      cache.Trim(verify_count - 1 - accepted_count);
    }
    AppendContext(aux_hidden_states, accepted_count + 1, draft_caches);
    if (policy == DraftPolicy::kFixed || finished) {
      continue;
    } else {
    }
    backoff.RecordRound(accepted_count + 1,
                        std::chrono::duration<double>(
                            std::chrono::steady_clock::now() - round_started)
                            .count());
    if (!backoff.ShouldMeasurePlain() ||
        max_new_tokens - committed_token_count <= kPlainProbeTokenCount) {
      continue;
    } else {
    }

    if (cancel.load()) {
      throw GenerationCancelled();
    } else {
    }
    std::vector<std::vector<mx::array>> probe_hidden_states;
    const uint32_t warm_anchor_id = static_cast<uint32_t>(anchor_id);
    const mx::array warm_hidden = target.Forward(
        target.EmbedTokens(mx::array(&warm_anchor_id, {1, 1}, mx::uint32)),
        caches, aux_layers_, &probe_hidden_states.emplace_back());
    anchor_id = static_cast<int>(
        mx::argmax(target.LastLogits(warm_hidden)).item<uint32_t>());
    finished = commit(anchor_id);
    ++committed_token_count;
    if (finished) {
      break;
    } else if (cancel.load()) {
      throw GenerationCancelled();
    } else {
    }
    const uint32_t probe_anchor_id = static_cast<uint32_t>(anchor_id);
    const mx::array probe_hidden = target.Forward(
        target.EmbedTokens(mx::array(&probe_anchor_id, {1, 1}, mx::uint32)),
        caches, aux_layers_, &probe_hidden_states.emplace_back());
    mx::array next_token = mx::argmax(target.LastLogits(probe_hidden));
    mx::eval({next_token});
    int measured_token_count = 0;
    const auto probe_commit = [&](int token_id) {
      anchor_id = token_id;
      finished = commit(token_id);
      ++measured_token_count;
      ++committed_token_count;
      return finished || measured_token_count == kPlainProbeTokenCount - 1;
    };
    const auto probe_started = std::chrono::steady_clock::now();
    target.DecodePlain(next_token, caches, 1.0f, probe_commit, cancel,
                       aux_layers_, &probe_hidden_states);
    // note (ZaneZhang-67): measure completed work with the same queued
    // lookahead as ordinary decoding, not the host's lazy graph construction.
    mx::eval({next_token});
    const double plain_seconds_per_token =
        std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                      probe_started)
            .count() /
        measured_token_count;
    if (finished) {
      break;
    } else {
    }
    if (backoff.ObservePlain(plain_seconds_per_token)) {
      std::cerr << "DFlash backoff after " << statistics.verify_round_count
                << " rounds: plain " << plain_seconds_per_token * 1000.0
                << " ms/token; continuing with target KV cache\n";
      probe_hidden_states.clear();
      aux_hidden_states.clear();
      draft_caches.clear();
      target.DecodePlain(next_token, caches, 1.0f, commit, cancel);
      break;
    } else {
      // note (ZaneZhang-67): plain queued the committed anchor's successor;
      // speculation needs that anchor uncached, and must not append its aux.
      for (KVCache &cache : caches) {
        cache.Trim(1);
      }
      probe_hidden_states.pop_back();
      for (const auto &captured : probe_hidden_states) {
        AppendContext(captured, 1, draft_caches);
      }
    }
  }
  return statistics;
}

} // namespace omni_common
