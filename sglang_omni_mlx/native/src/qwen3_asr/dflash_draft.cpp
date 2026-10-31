// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/dflash_draft.h"

#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

constexpr int kQuantizationGroupSize = 64;

// The draft's config once it is known to fit the target, whose hidden states
// it reads and whose tokens it embeds and predicts.
nlohmann::json CheckedConfig(nlohmann::json config,
                             const std::filesystem::path &directory,
                             const Qwen3DecoderConfig &target) {
  const std::string draft = "the draft at " + directory.string();
  const int hidden_size =
      config.at("transformer_layer_config").at("hidden_size").get<int>();
  const int mask_token_id = config.at("mask_token_id").get<int>();
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
  } else {
  }
  for (const int layer :
       config.at("aux_hidden_state_layer_ids").get<std::vector<int>>()) {
    if (layer < 0 || layer >= target.num_hidden_layers) {
      throw std::runtime_error(draft + " reads target layer " +
                               std::to_string(layer) + " of " +
                               std::to_string(target.num_hidden_layers));
    } else {
    }
  }
  return config;
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
    : DFlashDraft(
          CheckedConfig(omni_common::ReadJson(directory / "config.json"),
                        directory, target),
          omni_common::LoadSafetensors(directory), quantization_bits) {}

DFlashDraft::DFlashDraft(const nlohmann::json &config,
                         omni_common::WeightMap weights, int quantization_bits)
    : checkpoint_(QuantizeWeights(std::move(weights), quantization_bits),
                  QuantizationConfig(quantization_bits)),
      decoder_(
          checkpoint_, "",
          Qwen3DecoderConfig::FromJson(config.at("transformer_layer_config"))),
      aux_layers_(
          config.at("aux_hidden_state_layer_ids").get<std::vector<int>>()),
      block_size_(config.at("block_size").get<int>()),
      mask_token_id_(config.at("mask_token_id").get<int>()) {}

std::vector<KVCache> DFlashDraft::NewCaches() const {
  return decoder_.NewCaches();
}

// note (wilsonzheng0327): the draft's trained sliding window is not enforced:
// on a 194 s recording the 1.7B draft accepts 3.8 tokens a round with the
// whole context and 3.2 with the window, and the transcript is the target's
// either way.
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
