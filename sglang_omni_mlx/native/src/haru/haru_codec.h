// SPDX-License-Identifier: Apache-2.0
// VibeVoice acoustic codec: 24 kHz audio <-> 64-dim latents at 7.5 Hz, as
// causal convolution stacks. Both directions take and return their left
// context, so the same code encodes a whole clip or one 3200-sample frame
// with identical results.
#pragma once

#include <string>
#include <utility>
#include <vector>

#include "common/layers.h"
#include "haru/haru_config.h"
#include "mlx/mlx.h"

namespace haru {

namespace mx = mlx::core;

class VibeVoiceCodec {
public:
  VibeVoiceCodec(const omni_common::Checkpoint &checkpoint,
                 const CodecConfig &config);

  // Per-layer left context: one array per encoder/decoder layer.
  using State = std::vector<mx::array>;
  State InitEncoderState() const;
  State InitDecoderState() const;

  // audio [T] -> latent means [T / 3200, D] in float32; advances state.
  mx::array Encode(const mx::array &audio, State &state) const;
  // latents [T, D] -> audio [T * 3200] in float32; advances state.
  mx::array Decode(const mx::array &latents, State &state) const;

  // mean [B, T, D] plus per-row noise -> sampled latents (VQ-style).
  mx::array Sample(const mx::array &mean, const mx::array &key) const;

private:
  // One causal dense conv as a matmul over its input windows. The checkpoint
  // stores [out, in, kernel]; the model evaluates [out, kernel * in].
  // Returns (output, next state).
  std::pair<mx::array, mx::array> Conv(const mx::array &x, const mx::array &state,
                                       const std::string &prefix, int stride,
                                       int kernel) const;
  // One transposed conv with overlap-add; state is the carried tail.
  std::pair<mx::array, mx::array> ConvTranspose(const mx::array &x,
                                                const mx::array &state,
                                                const std::string &prefix,
                                                int stride) const;
  // One depthwise-causal-conv + MLP block.
  std::pair<mx::array, mx::array> Block(const mx::array &x,
                                        const mx::array &state,
                                        const std::string &prefix) const;

  const omni_common::Checkpoint &checkpoint_;
  CodecConfig config_;
};

} // namespace haru
