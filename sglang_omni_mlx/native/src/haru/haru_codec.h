// SPDX-License-Identifier: Apache-2.0
// VibeVoice acoustic codec: 24 kHz audio <-> 64-dim latents at 7.5 Hz, as
// causal convolution stacks. Both directions take and return their left
// context, so the same code encodes a whole clip or one 3200-sample frame
// with identical results.
#pragma once

#include <string>
#include <vector>

#include "common/layers.h"
#include "haru/haru_config.h"
#include "mlx/mlx.h"

namespace haru {

namespace mx = mlx::core;

// Rewrites the stored codec weights as the codec multiplies by them: dense
// convolutions as [kernel * in, out], and depthwise ones as [channels, kernel,
// 1] with the block's residual gain folded in.
void PrepareCodecWeights(omni_common::WeightMap &weights);

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
  enum class LayerKind { kConvolution, kTransposedConvolution, kBlock };
  struct Layer {
    LayerKind kind;
    std::string prefix;
    int kernel;
    int stride;
    // The shape of the layer's left context.
    int state_rows;
    int state_columns;
  };
  struct LayerOutput {
    mx::array output;
    mx::array next_state;
  };

  State InitState(const std::vector<Layer> &layers) const;
  mx::array Run(const std::vector<Layer> &layers, mx::array x,
                State &state) const;
  // One causal dense convolution as a matmul over its input windows.
  LayerOutput Convolve(const mx::array &x, const mx::array &state,
                       const Layer &layer) const;
  // One transposed convolution with overlap-add; state is the carried tail.
  LayerOutput ConvolveTransposed(const mx::array &x, const mx::array &state,
                                 const Layer &layer) const;
  // One causal depthwise convolution and feed-forward block.
  LayerOutput Block(const mx::array &x, const mx::array &state,
                    const Layer &layer) const;

  const omni_common::Checkpoint &checkpoint_;
  CodecConfig config_;
  std::vector<Layer> encoder_layers_;
  std::vector<Layer> decoder_layers_;
};

} // namespace haru
