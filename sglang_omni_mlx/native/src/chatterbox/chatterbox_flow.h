// SPDX-License-Identifier: Apache-2.0
// Chatterbox-Turbo's S3Gen flow: a conformer encoder lifts speech tokens to
// mel frames and a causal U-Net, conditioned on the voice, carries noise to the
// mel in two mean-flow steps.
#pragma once

#include <string>
#include <utility>

#include "common/layers.h"
#include "mlx/mlx.h"

namespace chatterbox {

// What the flow conditions on: the voice's prompt speech tokens int32 [1, n],
// its 24 kHz prompt mel [1, 2n, 80] and its speaker x-vector [1, 192].
struct FlowPrompt {
  mlx::core::array prompt_tokens;
  mlx::core::array prompt_mel;
  mlx::core::array speaker_embedding;
};

class FlowDecoder {
public:
  explicit FlowDecoder(const omni_common::Checkpoint &checkpoint);

  // The mel [1, 80, 2 * tokens] for speech_tokens int32 [1, tokens], prompt
  // part removed. Draws its noise from MLX's global random keys.
  mlx::core::array Mel(const mlx::core::array &speech_tokens,
                       const FlowPrompt &prompt) const;

private:
  // Encoder states [1, 2 * frames, 512] for token embeddings [1, frames, 512].
  mlx::core::array Encode(const mlx::core::array &token_embeddings) const;
  // Linear, layer norm and scaling of an encoder stage's input [1, frames,
  // 512]; returns the scaled input and its relative positions [1, 2 * frames
  // - 1, 512].
  std::pair<mlx::core::array, mlx::core::array>
  EmbedStage(const mlx::core::array &x, const std::string &prefix) const;
  mlx::core::array ConformerLayer(const mlx::core::array &x,
                                  const std::string &prefix,
                                  const mlx::core::array &positions) const;
  // The U-Net's mean velocity [1, 80, frames] at noisy mel x [1, 80, frames]
  // from time t to time r, both [1].
  mlx::core::array Velocity(const mlx::core::array &x,
                            const mlx::core::array &mu,
                            const mlx::core::array &speaker_conditioning,
                            const mlx::core::array &mel_conditioning,
                            const mlx::core::array &t,
                            const mlx::core::array &r) const;
  mlx::core::array TimeEmbedding(const mlx::core::array &time) const;
  // A resnet block and its transformer blocks on x [1, channels, frames].
  mlx::core::array UNetStage(const mlx::core::array &x,
                             const std::string &prefix,
                             const mlx::core::array &time_embedding) const;
  mlx::core::array CausalBlock(const mlx::core::array &x,
                               const std::string &prefix) const;
  mlx::core::array TransformerBlock(const mlx::core::array &x,
                                    const std::string &prefix) const;
  // A convolution with weights at prefix on channels-first x [1, channels,
  // frames], causal (left padded by the kernel less one) when asked.
  mlx::core::array ChannelsFirstConvolution(const mlx::core::array &x,
                                            const std::string &prefix,
                                            bool is_causal) const;
  mlx::core::array Convolution(const mlx::core::array &x,
                               const std::string &prefix) const;

  const omni_common::Checkpoint &checkpoint_;
  // Relative positions from +4999 down to -4999, [9999, 512], for inputs
  // up to 5000 frames.
  mlx::core::array relative_position_table_;
};

} // namespace chatterbox
