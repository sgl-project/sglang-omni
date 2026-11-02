// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/dflash_draft.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

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

} // namespace

DFlashDraft::DFlashDraft(const std::filesystem::path &directory,
                         int quantization_bits,
                         const Qwen3DecoderConfig &target)
    : DFlashDraft(CheckedDraft(directory, target), quantization_bits) {}

DFlashDraft::DFlashDraft(LoadedDraft draft, int quantization_bits)
    : checkpoint_(QuantizeWeights(std::move(draft.weights), quantization_bits),
                  QuantizationConfig(quantization_bits)),
      decoder_(checkpoint_, "",
               Qwen3DecoderConfig::FromJson(
                   draft.config.at("transformer_layer_config"))),
      aux_layers_(draft.config.at("aux_hidden_state_layer_ids")
                      .get<std::vector<int>>()),
      block_size_(draft.config.at("block_size").get<int>()),
      mask_token_id_(draft.config.at("mask_token_id").get<int>()) {}

std::vector<KVCache> DFlashDraft::NewCaches() const {
  return decoder_.NewCaches();
}

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

} // namespace qwen3_asr
