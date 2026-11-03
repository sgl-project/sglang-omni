// SPDX-License-Identifier: Apache-2.0
// Drama 3's codebook tower: per frame, shared enrichment layers and one layer
// per residual codebook, each with its own KV cache over time.
#include <cmath>
#include <optional>
#include <string>
#include <vector>

#include "fishaudio_drama3/fishaudio_drama3_model.h"

namespace fishaudio_drama3 {

namespace mx = mlx::core;

namespace {

constexpr const char *kTower = "tower";

} // namespace

mx::array Drama3Model::IsSemantic(const mx::array &tokens) const {
  return mx::logical_and(
      mx::greater_equal(tokens, mx::array(semantic_begin_, tokens.dtype())),
      mx::less_equal(tokens, mx::array(semantic_end_, tokens.dtype())));
}

mx::array Drama3Model::EmbeddingSum(const mx::array &frames, int first,
                                    int end) const {
  const mx::array rows =
      mx::add(SliceLast(frames, 1 + first, 1 + end),
              mx::multiply(mx::arange(first, end, mx::int32),
                           mx::array(codebook_size_, mx::int32)));
  return mx::sum(
      checkpoint_.Embed(rows, std::string(kTower) + ".codebook_embeddings"),
      -2);
}

mx::array Drama3Model::TowerLayer(
    const mx::array &hidden, const mx::array &codes,
    const mx::array &is_semantic, int layer, qwen3_asr::KVCache &cache,
    const std::optional<mx::array> &unseen_embedding) const {
  const bool is_shared = layer < shared_tower_layer_count_;
  const std::string prefix =
      is_shared
          ? std::string(kTower) + ".shared_layers." + std::to_string(layer)
          : std::string(kTower) + ".residual_layers." +
                std::to_string(layer - shared_tower_layer_count_);
  mx::array embedded = checkpoint_.Embed(
      mx::clip(codes, mx::array(0, codes.dtype()),
               mx::array(codebook_size_ - 1, codes.dtype())),
      prefix + (is_shared ? ".semantic_embedding" : ".codebook_embedding"));
  embedded = mx::where(mx::expand_dims(is_semantic, -1), embedded,
                       mx::zeros_like(embedded));
  if (unseen_embedding.has_value()) {
    embedded = mx::add(embedded, *unseen_embedding);
  } else {
  }
  const mx::array mixed = checkpoint_.Linear(
      mx::concatenate({Fp32Norm(embedded, prefix + ".enorm", tower_norm_eps_),
                       Fp32Norm(hidden, prefix + ".hnorm", tower_norm_eps_)},
                      -1),
      prefix + ".eh_proj");

  const std::string block = prefix + ".transformer_layer";
  const int batch = mixed.shape(0);
  const int time = mixed.shape(1);
  const int query_width = tower_head_count_ * tower_head_dim_;
  const int key_value_width = tower_key_value_head_count_ * tower_head_dim_;
  const mx::array qkv = checkpoint_.Linear(
      Fp32Norm(mixed, block + ".attention_norm", tower_norm_eps_),
      block + ".attention.wqkv");
  const mx::array gate = SliceLast(qkv, query_width, 2 * query_width);
  const auto rotate = [&](const mx::array &input) {
    return mx::astype(mx::fast::rope(mx::astype(input, mx::float32),
                                     tower_rotary_dims_, true, tower_rope_base_,
                                     1.0f, cache.offset()),
                      input.dtype());
  };
  const mx::array queries = rotate(mx::transpose(
      Fp32Norm(mx::reshape(SliceLast(qkv, 0, query_width),
                           {batch, time, tower_head_count_, tower_head_dim_}),
               block + ".attention.q_norm", tower_norm_eps_),
      {0, 2, 1, 3}));
  const mx::array keys = rotate(mx::transpose(
      Fp32Norm(mx::reshape(
                   SliceLast(qkv, 2 * query_width,
                             2 * query_width + key_value_width),
                   {batch, time, tower_key_value_head_count_, tower_head_dim_}),
               block + ".attention.k_norm", tower_norm_eps_),
      {0, 2, 1, 3}));
  const mx::array values = mx::transpose(
      mx::reshape(SliceLast(qkv, 2 * query_width + key_value_width,
                            2 * query_width + 2 * key_value_width),
                  {batch, time, tower_key_value_head_count_, tower_head_dim_}),
      {0, 2, 1, 3});
  const auto [cached_keys, cached_values] = cache.UpdateAndFetch(keys, values);
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      queries, cached_keys, cached_values,
      static_cast<float>(std::pow(static_cast<double>(tower_head_dim_), -0.5)),
      time > 1 ? "causal" : "");
  const mx::array attention = checkpoint_.Linear(
      mx::multiply(mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                               {batch, time, query_width}),
                   mx::sigmoid(gate)),
      block + ".attention.wo");
  const mx::array after_attention = mx::add(mixed, attention);
  const mx::array feed_forward_input =
      Fp32Norm(after_attention, block + ".ffn_norm", tower_norm_eps_);
  const mx::array feed_forward = checkpoint_.Linear(
      mx::multiply(
          omni_common::Silu(checkpoint_.Linear(feed_forward_input,
                                               block + ".feed_forward.w1")),
          checkpoint_.Linear(feed_forward_input, block + ".feed_forward.w3")),
      block + ".feed_forward.w2");
  const mx::array layer_output =
      Fp32Norm(mx::add(after_attention, feed_forward),
               prefix + ".final_layernorm", tower_norm_eps_);
  return mx::astype(layer_output, mixed.dtype());
}

mx::array Drama3Model::Enrich(mx::array hidden, const mx::array &semantic_codes,
                              const mx::array &is_semantic,
                              const mx::array &frames,
                              std::vector<qwen3_asr::KVCache> &caches) const {
  const mx::array unseen =
      EmbeddingSum(frames, 1 + backbone_codebook_count_, codebook_count_);
  const mx::array frame_is_semantic =
      mx::expand_dims(IsSemantic(mx::squeeze(SliceLast(frames, 0, 1), -1)), -1);
  const mx::array unseen_embedding =
      mx::where(frame_is_semantic, unseen, mx::zeros_like(unseen));
  for (int layer = 0; layer < shared_tower_layer_count_; ++layer) {
    hidden = TowerLayer(
        hidden, semantic_codes, is_semantic, layer, caches[layer],
        layer == 0 ? std::optional<mx::array>(unseen_embedding) : std::nullopt);
  }
  return hidden;
}

mx::array Drama3Model::CodebookHead(const mx::array &hidden, int index) const {
  return checkpoint_.Linear(hidden, std::string(kTower) + ".codebook_heads." +
                                        std::to_string(index));
}

} // namespace fishaudio_drama3
