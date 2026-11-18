// SPDX-License-Identifier: Apache-2.0
// Note (Jiaxin Deng): ported from the Chatterbox-Turbo S3 tokenizer v2
// (chatterbox_speech_tokenizer.cpp); v3 is the same network with twelve
// blocks. Both belong in src/common once they land.
#include "cosyvoice3/cosyvoice3_speech_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "common/audio.h"

namespace cosyvoice3 {

namespace mx = mlx::core;

namespace {

constexpr int kLayerCount = 12;
constexpr int kHiddenSize = 1280;
constexpr int kHeadCount = 20;
constexpr int kHeadDim = kHiddenSize / kHeadCount;
constexpr int kFsmnKernelSize = 31;
constexpr float kRotaryBase = 10000.0f;
constexpr float kAttentionNormEps = 1e-6f;
constexpr float kMlpNormEps = 1e-5f;
// Longer input is tokenized in overlapping 30 s windows, which is not
// supported here.
constexpr int kMaxMelFrames = 3000;
constexpr int kQuantizerDimensions = 8;
constexpr float kQuantizerLevelCount = 3.0f;
// tanh output is shrunk just inside (-1, 1) before rounding to {-1, 0, 1}.
constexpr float kQuantizerShrink = 0.999f;

std::string LayerPrefix(int layer) {
  return "encoder.blocks." + std::to_string(layer);
}

} // namespace

SpeechTokenizer::SpeechTokenizer(const omni_common::Checkpoint &checkpoint)
    : checkpoint_(checkpoint) {}

mx::array SpeechTokenizer::Block(const mx::array &x, int layer,
                                 const mx::array &attention_mask) const {
  const std::string prefix = LayerPrefix(layer);
  const int positions = x.shape(1);
  const mx::array normed =
      checkpoint_.LayerNorm(x, prefix + ".attn_ln", kAttentionNormEps);
  const mx::array query = checkpoint_.Linear(normed, prefix + ".attn.query");
  const mx::array key = checkpoint_.Linear(normed, prefix + ".attn.key");
  const mx::array value = checkpoint_.Linear(normed, prefix + ".attn.value");
  const mx::Shape head_shape = {1, positions, kHeadCount, kHeadDim};
  // Heads [1, heads, positions, head dim], rotated by halves.
  const auto rotated = [&](const mx::array &projection) {
    return mx::fast::rope(
        mx::transpose(mx::reshape(projection, head_shape), {0, 2, 1, 3}),
        kHeadDim, false, kRotaryBase, 1.0f, 0);
  };
  const mx::array q = rotated(query);
  const mx::array k = rotated(key);
  const mx::array v = mx::reshape(value, head_shape);

  // The FSMN memory: a centered depthwise convolution over the values.
  const mx::array flat_value = mx::reshape(v, {1, positions, kHiddenSize});
  const int left_padding = (kFsmnKernelSize - 1) / 2;
  const int right_padding = kFsmnKernelSize - 1 - left_padding;
  const mx::array padded_value = mx::concatenate(
      {mx::zeros({1, left_padding, kHiddenSize}, flat_value.dtype()),
       flat_value,
       mx::zeros({1, right_padding, kHiddenSize}, flat_value.dtype())},
      1);
  const mx::array memory =
      mx::add(mx::conv1d(padded_value,
                         checkpoint_.Weight(prefix + ".attn.fsmn_block.weight"),
                         1, 0, 1, kHiddenSize),
              flat_value);

  const mx::array scale(
      static_cast<float>(std::pow(static_cast<double>(kHeadDim), -0.25)));
  const mx::array attention = mx::fast::scaled_dot_product_attention(
      mx::multiply(q, scale), mx::multiply(k, scale),
      mx::transpose(v, {0, 2, 1, 3}), 1.0f, /*window_size=*/-1, "array",
      attention_mask);
  const mx::array attended = mx::reshape(mx::transpose(attention, {0, 2, 1, 3}),
                                         {1, positions, kHiddenSize});
  const mx::array after_attention = mx::add(
      x, mx::add(checkpoint_.Linear(attended, prefix + ".attn.out"), memory));

  const mx::array mlp_input =
      checkpoint_.LayerNorm(after_attention, prefix + ".mlp_ln", kMlpNormEps);
  const mx::array mlp_output =
      checkpoint_.Linear(omni_common::Gelu(checkpoint_.Linear(
                             mlp_input, prefix + ".mlp.layers.0")),
                         prefix + ".mlp.layers.2");
  return mx::add(after_attention, mlp_output);
}

mx::array SpeechTokenizer::Encode(const std::vector<float> &samples_16k) const {
  if (samples_16k.size() / omni_common::kHopLength > kMaxMelFrames) {
    throw std::invalid_argument(
        "the speech tokenizer takes at most 30 s of audio");
  } else {
  }
  // Note (Jiaxin Deng): Whisper's 128-band log-mel, as the reference
  // computes it; [1, mel frames, mel bands].
  mx::array x =
      mx::expand_dims(mx::transpose(omni_common::LogMel(
                          samples_16k, omni_common::FinalStftFrame::kDropped)),
                      0);
  x = omni_common::Gelu(
      mx::add(mx::conv1d(x, checkpoint_.Weight("encoder.conv1.weight"), 2, 1),
              checkpoint_.Weight("encoder.conv1.bias")));
  x = omni_common::Gelu(
      mx::add(mx::conv1d(x, checkpoint_.Weight("encoder.conv2.weight"), 2, 1),
              checkpoint_.Weight("encoder.conv2.bias")));
  const int token_count = x.shape(1);
  // Note (Jiaxin Deng): a single clip masks nothing, but the all-pass
  // additive mask (-0.0) keeps the masked attention kernel.
  const mx::array attention_mask = mx::full({1, 1, token_count}, -0.0f);
  for (int layer = 0; layer < kLayerCount; ++layer) {
    x = Block(x, layer, attention_mask);
  }

  mx::array levels = mx::tanh(
      mx::astype(checkpoint_.Linear(mx::reshape(x, {-1, kHiddenSize}),
                                    "quantizer.fsq_codebook.project_down"),
                 mx::float32));
  levels = mx::add(mx::round(mx::multiply(levels, mx::array(kQuantizerShrink))),
                   mx::array(1.0f));
  const mx::array place_values =
      mx::power(mx::array(kQuantizerLevelCount),
                mx::arange(0, kQuantizerDimensions, mx::float32));
  const mx::array tokens =
      mx::sum(mx::multiply(levels, mx::expand_dims(place_values, 0)), -1);
  return mx::astype(mx::reshape(tokens, {1, token_count}), mx::int32);
}

} // namespace cosyvoice3
