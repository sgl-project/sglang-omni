// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/dflash_draft.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "qwen3_asr/qwen3_asr_transcriber.h"

namespace qwen3_asr {

namespace mx = mlx::core;

struct LoadedDraft {
  nlohmann::json config;
  omni_common::WeightMap weights;
};

namespace {

constexpr int kQuantizationGroupSize = 64;

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

nlohmann::json QuantizationConfig(int quantization_bits) {
  nlohmann::json config;
  if (quantization_bits > 0) {
    config["quantization"] = {{"group_size", kQuantizationGroupSize},
                              {"bits", quantization_bits}};
  } else {
  }
  return config;
}

// Quantizes the two-dimensional weights (the linear layers and lm_head) as
// an MLX checkpoint stores them: .weight packed, with .scales and .biases.
omni_common::WeightMap QuantizeWeights(omni_common::WeightMap weights,
                                       int quantization_bits) {
  if (quantization_bits == 0) {
    return weights;
  } else {
  }
  omni_common::WeightMap quantized;
  while (!weights.empty()) {
    const auto stored = weights.extract(weights.begin());
    const std::string &name = stored.key();
    const mx::array &array = stored.mapped();
    if (array.ndim() == 2) {
      const std::vector<mx::array> parts =
          mx::quantize(array, kQuantizationGroupSize, quantization_bits);
      // note (wilsonzheng0327): evaluated one weight at a time, so each
      // stored weight is released before the next one is read.
      mx::eval(parts);
      const std::string prefix = name.substr(0, name.size() - 7);
      quantized.insert_or_assign(prefix + ".weight", parts[0]);
      quantized.insert_or_assign(prefix + ".scales", parts[1]);
      quantized.insert_or_assign(prefix + ".biases", parts[2]);
    } else {
      quantized.insert_or_assign(name, array);
    }
  }
  return quantized;
}

float RopeThetaScale(
    const std::map<int, float> &rope_theta_scale_by_prompt_token_count,
    int prompt_token_count) {
  float scale = 1.0f;
  for (const auto &[longer_than_token_count, longer_scale] :
       rope_theta_scale_by_prompt_token_count) {
    if (prompt_token_count > longer_than_token_count) {
      scale = longer_scale;
    } else {
    }
  }
  return scale;
}

} // namespace

DFlashDraft::DFlashDraft(
    const std::filesystem::path &directory, int quantization_bits,
    const Qwen3DecoderConfig &target,
    std::map<int, float> rope_theta_scale_by_prompt_token_count)
    : DFlashDraft(CheckedDraft(directory, target), quantization_bits,
                  std::move(rope_theta_scale_by_prompt_token_count)) {}

DFlashDraft::DFlashDraft(
    LoadedDraft draft, int quantization_bits,
    std::map<int, float> rope_theta_scale_by_prompt_token_count)
    : checkpoint_(QuantizeWeights(std::move(draft.weights), quantization_bits),
                  QuantizationConfig(quantization_bits)),
      decoder_(checkpoint_, "",
               Qwen3DecoderConfig::FromJson(
                   draft.config.at("transformer_layer_config"))),
      aux_layers_(draft.config.at("aux_hidden_state_layer_ids")
                      .get<std::vector<int>>()),
      block_size_(draft.config.at("block_size").get<int>()),
      mask_token_id_(draft.config.at("mask_token_id").get<int>()),
      rope_theta_scale_by_prompt_token_count_(
          std::move(rope_theta_scale_by_prompt_token_count)) {}

// note (wilsonzheng0327): the draft's trained sliding window is not enforced:
// on a 194 s recording a 1.7B draft trained with a 2,048 window accepts 3.8
// tokens a round with the whole context and 3.2 with the window, and the
// transcript is the target's either way.
void DFlashDraft::AppendContext(const Qwen3Decoder &draft_decoder,
                                const std::vector<mx::array> &aux_hidden_states,
                                int token_count,
                                std::vector<KVCache> &caches) const {
  std::vector<mx::array> committed;
  for (const mx::array &hidden_states : aux_hidden_states) {
    committed.push_back(mx::slice(hidden_states, {0, 0, 0},
                                  {1, token_count, hidden_states.shape(2)}));
  }
  const mx::array context = checkpoint_.RmsNorm(
      checkpoint_.Linear(mx::concatenate(committed, -1), "fc"), "hidden_norm",
      draft_decoder.config().rms_norm_eps);
  draft_decoder.AppendKeysValues(context, caches);
}

mx::array DFlashDraft::ProposeBlock(const Qwen3Decoder &draft_decoder,
                                    const mx::array &block_embeddings,
                                    std::vector<KVCache> &caches) const {
  // note (wilsonzheng0327): the block attends causally, as the pinned drafts
  // were trained (sliding_window_non_causal is false, checked at load). A
  // bidirectional block, as in the DFlash reference, lowers the acceptance on
  // the 392-clip corpus: 4.86 to 4.42 a round on the 0.6B draft and 5.50 to
  // 4.95 on the 1.7B (measured by Jiaxin Deng), with the same transcripts.
  const mx::array hidden = draft_decoder.Forward(block_embeddings, caches);
  // note (wilsonzheng0327): only context keys and values stay in the draft
  // caches; the block's own are dropped once it has been proposed.
  for (KVCache &cache : caches) {
    cache.Trim(block_size_);
  }
  const mx::array proposals = checkpoint_.RmsNorm(
      mx::slice(hidden, {0, 1, 0}, {1, block_size_, hidden.shape(2)}), "norm",
      draft_decoder.config().rms_norm_eps);
  const mx::array draft_ids =
      mx::argmax(checkpoint_.Linear(proposals, "lm_head"), -1);
  // note (wilsonzheng0327): the draft predicts a reduced vocabulary; d2t
  // holds each draft id's offset to its target id.
  const mx::array target_ids =
      mx::add(draft_ids, mx::take(checkpoint_.Weight("d2t"), draft_ids, 0));
  return mx::reshape(mx::astype(target_ids, mx::int32), {block_size_ - 1});
}

int DFlashDraft::Decode(const Qwen3Decoder &target, const mx::array &embeddings,
                        int max_new_tokens, int prefill_step_tokens,
                        const std::function<bool(int)> &commit,
                        const std::atomic<bool> &cancel) const {
  std::vector<KVCache> caches = target.NewCaches();
  std::vector<KVCache> draft_caches = decoder_.NewCaches();
  // The decoder only holds references and a config, so a per-request copy is
  // cheap and leaves the shared draft unchanged for concurrent requests.
  Qwen3DecoderConfig draft_config = decoder_.config();
  draft_config.rope_theta *=
      RopeThetaScale(rope_theta_scale_by_prompt_token_count_,
                     static_cast<int>(embeddings.shape(1)));
  const Qwen3Decoder draft_decoder(checkpoint_, "", draft_config);
  std::vector<mx::array> aux_hidden_states;
  int processed = 0;
  while (prefill_step_tokens > 0 && embeddings.shape(1) - processed > 1) {
    if (cancel.load()) {
      throw TranscriptionCancelled();
    } else {
    }
    const int token_count =
        std::min(prefill_step_tokens, embeddings.shape(1) - processed - 1);
    const mx::array hidden = target.Forward(
        mx::slice(embeddings, {0, processed, 0},
                  {1, processed + token_count, embeddings.shape(2)}),
        caches, aux_layers_, &aux_hidden_states);
    AppendContext(draft_decoder, aux_hidden_states, token_count, draft_caches);
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
    throw TranscriptionCancelled();
  } else {
  }
  int anchor_id;
  {
    const mx::array prompt_hidden = target.Forward(
        mx::slice(embeddings, {0, processed, 0}, embeddings.shape()), caches,
        aux_layers_, &aux_hidden_states);
    AppendContext(draft_decoder, aux_hidden_states,
                  embeddings.shape(1) - processed, draft_caches);
    anchor_id = static_cast<int>(
        mx::argmax(target.LastLogits(prompt_hidden)).item<uint32_t>());
  }
  if (cancel.load()) {
    throw TranscriptionCancelled();
  } else {
  }
  bool finished = commit(anchor_id);
  int committed_token_count = 1;
  int round_count = 0;
  while (!finished) {
    if (cancel.load()) {
      throw TranscriptionCancelled();
    } else {
    }
    std::vector<int> block_ids(block_size_, mask_token_id_);
    block_ids[0] = anchor_id;
    const mx::array proposed_ids =
        ProposeBlock(draft_decoder,
                     target.EmbedTokens(mx::array(block_ids.data(),
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
    ++round_count;
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
    // note (wilsonzheng0327): the caches keep the anchor and the accepted
    // proposals; the committed positions' hidden states become the draft's
    // context.
    for (KVCache &cache : caches) {
      cache.Trim(verify_count - 1 - accepted_count);
    }
    AppendContext(draft_decoder, aux_hidden_states, accepted_count + 1,
                  draft_caches);
  }
  return round_count;
}

} // namespace qwen3_asr
