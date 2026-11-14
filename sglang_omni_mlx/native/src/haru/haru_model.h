// SPDX-License-Identifier: Apache-2.0
// Haru-4B's language model: a Qwen3.5 backbone (Gated DeltaNet layers with
// periodic full attention, dense SwiGLU MLPs) over one position per 133 ms
// frame, plus the audio projections and the tied text output.
#pragma once

#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "common/layers.h"
#include "haru/haru_config.h"
#include "mlx/mlx.h"
#include "qwen3_asr/qwen3_decoder.h"

namespace haru {

namespace mx = mlx::core;

// One layer's state: a gated delta layer's convolution tail and recurrent
// state, or an attention layer's keys and values.
struct LayerCache {
  std::optional<mx::array> convolution_state;  // [B, kernel - 1, conv_dim]
  std::optional<mx::array> recurrent_state;    // [B, Hv, Dv, Dk] float32
  qwen3_asr::KVCache attention;
};

class HaruModel {
public:
  // quantize_bits: 8 to quantize the backbone's Linear weights and the
  // embedding table at load; 0 keeps the checkpoint's precision.
  explicit HaruModel(const std::filesystem::path &model_directory,
                     int quantize_bits = 0);

  const HaruConfig &config() const { return config_; }
  const omni_common::Checkpoint &checkpoint() const { return checkpoint_; }

  // Empty per-layer caches for one conversation.
  std::vector<LayerCache> NewCaches() const;

  // tokens [1, T] and the two audio streams [1, T, D] (with masks [1, T] that
  // zero a stream at positions that carry none) -> embeddings [1, T, H].
  mx::array Embed(const mx::array &tokens,
                  const std::optional<mx::array> &model_latents,
                  const std::optional<mx::array> &user_latents,
                  const std::optional<mx::array> &model_mask,
                  const std::optional<mx::array> &user_mask) const;

  // embeddings -> the final-normed hidden states [1, T, H].
  mx::array Backbone(const mx::array &embeddings,
                     std::vector<LayerCache> &caches) const;

  // hidden -> text logits [..., vocab] in float32 (the tied output embedding).
  mx::array Logits(const mx::array &hidden) const;
  // Logits of a few tokens computed entirely in float32 (the PAD/EOP decision
  // is often closer than bfloat16 resolves).
  mx::array LogitsFp32(const mx::array &hidden, const mx::array &tokens) const;
  // Normalized head latents -> raw codec latents.
  mx::array UnscaleLatents(const mx::array &normalized) const;

private:
  // The zero-centred RMSNorm gain of prefix as the float32 1 + w the model
  // uses (bfloat16 only resolves 1/128 near 1).
  mx::array NormGain(const std::string &prefix) const;
  // RMS norm computed in float32 with the zero-centred float32 gain.
  mx::array Fp32Norm(const mx::array &x, const std::string &prefix,
                     float eps) const;
  mx::array GatedDeltaLayer(const mx::array &x, const std::string &prefix,
                            LayerCache &cache) const;
  mx::array AttentionLayer(const mx::array &x, const std::string &prefix,
                           qwen3_asr::KVCache &cache) const;
  // Dense SwiGLU feed-forward.
  mx::array MLP(const mx::array &x, const std::string &prefix) const;
  // Codec latents -> LM embedding space: fc1, RMS norm, fc2.
  mx::array AudioProject(const mx::array &latents,
                         const std::string &prefix) const;

  HaruConfig config_;
  omni_common::Checkpoint checkpoint_;
  int layer_count_;
  int rotary_dims_;
  mx::array latent_scale_;
  mx::array latent_bias_;
};

} // namespace haru
