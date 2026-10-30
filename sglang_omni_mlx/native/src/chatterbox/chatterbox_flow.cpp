// SPDX-License-Identifier: Apache-2.0
#include "chatterbox/chatterbox_flow.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <tuple>
#include <vector>

namespace chatterbox {

namespace mx = mlx::core;

namespace {

constexpr const char *kEncoderPrefix = "s3gen.encoder";
constexpr const char *kEstimatorPrefix = "s3gen.decoder.estimator";
constexpr int kEncoderDimension = 512;
constexpr int kEncoderHeads = 8;
constexpr int kEncoderHeadDimension = 64;
constexpr int kEncoderLayers = 6;
constexpr int kUpsampledEncoderLayers = 4;
constexpr int kPreLookaheadTokens = 3;
constexpr int kMelFramesPerToken = 2;
constexpr int kPositionTableLength = 5000;
constexpr float kInputNormEps = 1e-5f;
constexpr float kConformerNormEps = 1e-12f;
constexpr float kDecoderNormEps = 1e-5f;
constexpr int kMelBins = 80;
constexpr int kTimeSinusoidDimension = 320;
constexpr float kTimeScale = 1000.0f;
constexpr int kDecoderHeads = 8;
constexpr int kDecoderHeadDimension = 64;
constexpr int kTransformerBlocksPerStage = 4;
constexpr int kMidStages = 12;
constexpr int kFlowSteps = 2;
constexpr float kSpeakerNormFloor = 1e-8f;
constexpr float kLeakyReluSlope = 0.01f;

std::vector<mx::array> MishGraph(const std::vector<mx::array> &inputs) {
  const mx::array &x = inputs[0];
  return {mx::multiply(x, mx::tanh(mx::logaddexp(x, mx::array(0.0f))))};
}

std::vector<mx::array> LeakyReluGraph(const std::vector<mx::array> &inputs) {
  const mx::array &x = inputs[0];
  return {mx::maximum(mx::multiply(mx::array(kLeakyReluSlope), x), x)};
}

// Note (Jiaxin Deng): compiled shapeless like the other activations, so each
// is one fused kernel with the same rounding.
mx::array Mish(const mx::array &x) {
  static const auto compiled = mx::compile(MishGraph, true);
  return compiled({x})[0];
}

mx::array LeakyRelu(const mx::array &x) {
  static const auto compiled = mx::compile(LeakyReluGraph, true);
  return compiled({x})[0];
}

// Rows of [rows, 512]: sin then cos of each position times 10000^(-i/512),
// interleaved.
mx::array PositionSinusoids(const mx::array &positions) {
  const float log_base = mx::log(mx::array(10000.0f)).item<float>();
  const mx::array frequencies = mx::exp(
      mx::multiply(mx::arange(0.0, kEncoderDimension, 2.0, mx::float32),
                   mx::array(static_cast<float>(-static_cast<double>(log_base) /
                                                kEncoderDimension))));
  const mx::array angles = mx::multiply(positions, frequencies);
  return mx::reshape(mx::concatenate({mx::expand_dims(mx::sin(angles), 2),
                                      mx::expand_dims(mx::cos(angles), 2)},
                                     -1),
                     {positions.shape(0), kEncoderDimension});
}

// Multi-head attention scores [1, heads, frames, keys] to outputs [1, frames,
// heads * head dimension] over values [1, heads, keys, head dimension].
mx::array AttendValues(const mx::array &scores, const mx::array &values) {
  const mx::array weights = mx::softmax(scores, -1);
  const mx::array output =
      mx::transpose(mx::matmul(weights, values), {0, 2, 1, 3});
  return mx::reshape(output, {output.shape(0), output.shape(1), -1});
}

mx::array SplitHeads(const mx::array &x, int heads, int head_dimension) {
  return mx::reshape(x, {x.shape(0), x.shape(1), heads, head_dimension});
}

// Sinusoids of relative positions from +(length - 1) down to -(length - 1),
// [2 * length - 1, 512]; each row depends only on its position.
mx::array RelativePositionTable(int length) {
  const mx::array positions =
      mx::expand_dims(mx::arange(0.0, length, 1.0, mx::float32), 1);
  const mx::array positive = PositionSinusoids(positions);
  const mx::array negative = PositionSinusoids(mx::negative(positions));
  const mx::array descending = mx::arange(length - 1, -1, -1, mx::int32);
  return mx::concatenate(
      {mx::take(positive, descending, 0),
       mx::slice(negative, {1, 0}, {length, kEncoderDimension})},
      0);
}

} // namespace

FlowDecoder::FlowDecoder(const omni_common::Checkpoint &checkpoint)
    : checkpoint_(checkpoint),
      relative_position_table_(RelativePositionTable(kPositionTableLength)) {
  mx::eval(relative_position_table_);
}

mx::array FlowDecoder::Convolution(const mx::array &x,
                                   const std::string &prefix) const {
  return mx::add(mx::conv1d(x, checkpoint_.Weight(prefix + ".weight")),
                 checkpoint_.Weight(prefix + ".bias"));
}

mx::array FlowDecoder::ChannelsFirstConvolution(const mx::array &x,
                                                const std::string &prefix,
                                                bool is_causal) const {
  mx::array input = x;
  if (is_causal) {
    const int padding = checkpoint_.Weight(prefix + ".weight").shape(1) - 1;
    input = mx::pad(x, {{0, 0}, {0, 0}, {padding, 0}});
  } else {
  }
  return mx::transpose(Convolution(mx::transpose(input, {0, 2, 1}), prefix),
                       {0, 2, 1});
}

std::pair<mx::array, mx::array>
FlowDecoder::EmbedStage(const mx::array &x, const std::string &prefix) const {
  const mx::array normalized =
      checkpoint_.LayerNorm(checkpoint_.Linear(x, prefix + ".linear"),
                            prefix + ".norm", kInputNormEps);
  const mx::array scaled = mx::multiply(
      normalized,
      mx::array(static_cast<float>(std::sqrt(double{kEncoderDimension}))));
  const int frames = x.shape(1);
  // Note (Jiaxin Deng): a longer input gets a table of its own length, which
  // holds the same rows around the center.
  const int length = std::max(frames, kPositionTableLength);
  const mx::array table = frames > kPositionTableLength
                              ? RelativePositionTable(length)
                              : relative_position_table_;
  const int center = length - 1;
  const mx::array positions =
      mx::expand_dims(mx::slice(table, {center - frames + 1, 0},
                                {center + frames, kEncoderDimension}),
                      0);
  return {scaled, positions};
}

mx::array FlowDecoder::ConformerLayer(const mx::array &x,
                                      const std::string &prefix,
                                      const mx::array &positions) const {
  const std::string attention = prefix + ".self_attn";
  const mx::array normalized =
      checkpoint_.LayerNorm(x, prefix + ".norm_mha", kConformerNormEps);
  const int frames = x.shape(1);
  const mx::array q =
      SplitHeads(checkpoint_.Linear(normalized, attention + ".linear_q"),
                 kEncoderHeads, kEncoderHeadDimension);
  const mx::array k = mx::transpose(
      SplitHeads(checkpoint_.Linear(normalized, attention + ".linear_k"),
                 kEncoderHeads, kEncoderHeadDimension),
      {0, 2, 1, 3});
  const mx::array v = mx::transpose(
      SplitHeads(checkpoint_.Linear(normalized, attention + ".linear_v"),
                 kEncoderHeads, kEncoderHeadDimension),
      {0, 2, 1, 3});
  const mx::array content_scores = mx::matmul(
      mx::transpose(mx::add(q, checkpoint_.Weight(attention + ".pos_bias_u")),
                    {0, 2, 1, 3}),
      mx::transpose(k, {0, 1, 3, 2}));
  const mx::array projected_positions = mx::transpose(
      SplitHeads(checkpoint_.Linear(positions, attention + ".linear_pos"),
                 kEncoderHeads, kEncoderHeadDimension),
      {0, 2, 1, 3});
  mx::array position_scores = mx::matmul(
      mx::transpose(mx::add(q, checkpoint_.Weight(attention + ".pos_bias_v")),
                    {0, 2, 1, 3}),
      mx::transpose(projected_positions, {0, 1, 3, 2}));
  const int position_count = position_scores.shape(3);
  if (position_count != frames) {
    position_scores =
        mx::slice(omni_common::RelativeShift(position_scores), {0, 0, 0, 0},
                  {1, kEncoderHeads, frames, position_count / 2 + 1});
  } else {
  }
  const mx::array scores = mx::multiply(
      mx::add(content_scores, position_scores),
      mx::array(static_cast<float>(std::pow(kEncoderHeadDimension, -0.5))));
  const mx::array attended =
      mx::add(x, checkpoint_.Linear(AttendValues(scores, v),
                                    attention + ".linear_out"));
  const mx::array feed_forward_input =
      checkpoint_.LayerNorm(attended, prefix + ".norm_ff", kConformerNormEps);
  return mx::add(
      attended,
      checkpoint_.Linear(omni_common::Silu(checkpoint_.Linear(
                             feed_forward_input, prefix + ".feed_forward.w_1")),
                         prefix + ".feed_forward.w_2"));
}

mx::array FlowDecoder::Encode(const mx::array &token_embeddings) const {
  const std::string prefix = kEncoderPrefix;
  auto [x, positions] = EmbedStage(token_embeddings, prefix + ".embed");
  // Note (Jiaxin Deng): every frame is valid (one unpadded sequence), so the
  // length masks are all ones and leave every value unchanged; they are
  // omitted here and in the U-Net.
  const std::string lookahead = prefix + ".pre_lookahead_layer";
  mx::array looked_ahead = LeakyRelu(
      Convolution(mx::pad(x, {{0, 0}, {0, kPreLookaheadTokens}, {0, 0}}),
                  lookahead + ".conv1"));
  looked_ahead = Convolution(mx::pad(looked_ahead, {{0, 0}, {2, 0}, {0, 0}}),
                             lookahead + ".conv2");
  x = mx::add(looked_ahead, x);
  for (int layer = 0; layer < kEncoderLayers; ++layer) {
    x = ConformerLayer(x, prefix + ".encoders." + std::to_string(layer),
                       positions);
  }
  x = mx::repeat(x, kMelFramesPerToken, 1);
  x = Convolution(mx::pad(x, {{0, 0}, {2 * kMelFramesPerToken, 0}, {0, 0}}),
                  prefix + ".up_layer.conv");
  std::tie(x, positions) = EmbedStage(x, prefix + ".up_embed");
  for (int layer = 0; layer < kUpsampledEncoderLayers; ++layer) {
    x = ConformerLayer(x, prefix + ".up_encoders." + std::to_string(layer),
                       positions);
  }
  return checkpoint_.LayerNorm(x, prefix + ".after_norm", kInputNormEps);
}

mx::array FlowDecoder::TimeEmbedding(const mx::array &time) const {
  const int half_dimension = kTimeSinusoidDimension / 2;
  const double frequency_step = std::log(10000.0) / (half_dimension - 1);
  const mx::array frequencies =
      mx::exp(mx::multiply(mx::arange(half_dimension, mx::float32),
                           mx::array(static_cast<float>(-frequency_step))));
  const mx::array angles = mx::multiply(
      mx::multiply(mx::array(kTimeScale), mx::expand_dims(time, 1)),
      mx::expand_dims(frequencies, 0));
  const mx::array sinusoids =
      mx::concatenate({mx::sin(angles), mx::cos(angles)}, -1);
  const std::string prefix = std::string(kEstimatorPrefix) + ".time_mlp";
  return checkpoint_.Linear(
      omni_common::Silu(checkpoint_.Linear(sinusoids, prefix + ".linear_1")),
      prefix + ".linear_2");
}

mx::array FlowDecoder::CausalBlock(const mx::array &x,
                                   const std::string &prefix) const {
  const mx::array convolved =
      ChannelsFirstConvolution(x, prefix + ".block.0.conv.conv", true);
  const mx::array normalized =
      mx::transpose(checkpoint_.LayerNorm(mx::transpose(convolved, {0, 2, 1}),
                                          prefix + ".block.1", kDecoderNormEps),
                    {0, 2, 1});
  return Mish(normalized);
}

mx::array FlowDecoder::TransformerBlock(const mx::array &x,
                                        const std::string &prefix) const {
  const std::string attention = prefix + ".attn1";
  const mx::array normalized =
      checkpoint_.LayerNorm(x, prefix + ".norm1", kDecoderNormEps);
  const auto heads = [&](const std::string &projection) {
    return mx::transpose(
        SplitHeads(checkpoint_.Linear(normalized, attention + projection),
                   kDecoderHeads, kDecoderHeadDimension),
        {0, 2, 1, 3});
  };
  const mx::array q = heads(".to_q");
  const mx::array k = heads(".to_k");
  const mx::array v = heads(".to_v");
  const mx::array scores = mx::multiply(
      mx::matmul(q, mx::transpose(k, {0, 1, 3, 2})),
      mx::array(static_cast<float>(std::pow(kDecoderHeadDimension, -0.5))));
  const mx::array attended = mx::add(
      x, checkpoint_.Linear(AttendValues(scores, v), attention + ".to_out.0"));
  const mx::array feed_forward_input =
      checkpoint_.LayerNorm(attended, prefix + ".norm3", kDecoderNormEps);
  return mx::add(
      attended,
      checkpoint_.Linear(omni_common::Gelu(checkpoint_.Linear(
                             feed_forward_input, prefix + ".ff.net.0.proj")),
                         prefix + ".ff.net.1"));
}

mx::array FlowDecoder::UNetStage(const mx::array &x, const std::string &prefix,
                                 const mx::array &time_embedding) const {
  const std::string resnet = prefix + ".resnet";
  mx::array hidden = CausalBlock(x, resnet + ".block1");
  hidden =
      mx::add(hidden, mx::expand_dims(checkpoint_.Linear(Mish(time_embedding),
                                                         resnet + ".mlp.0"),
                                      2));
  hidden = CausalBlock(hidden, resnet + ".block2");
  hidden = mx::add(
      hidden, ChannelsFirstConvolution(x, resnet + ".res_conv.conv", false));
  hidden = mx::transpose(hidden, {0, 2, 1});
  for (int block = 0; block < kTransformerBlocksPerStage; ++block) {
    hidden = TransformerBlock(hidden, prefix + ".transformer_blocks." +
                                          std::to_string(block));
  }
  return mx::transpose(hidden, {0, 2, 1});
}

mx::array FlowDecoder::Velocity(const mx::array &x, const mx::array &mu,
                                const mx::array &speaker_conditioning,
                                const mx::array &mel_conditioning,
                                const mx::array &t, const mx::array &r) const {
  const std::string prefix = kEstimatorPrefix;
  const mx::array time_embedding = checkpoint_.Linear(
      mx::concatenate({TimeEmbedding(t), TimeEmbedding(r)}, -1),
      prefix + ".time_embed_mixer");
  const int frames = x.shape(2);
  mx::array hidden = mx::concatenate(
      {x, mu,
       mx::broadcast_to(mx::expand_dims(speaker_conditioning, 2),
                        {1, kMelBins, frames}),
       mel_conditioning},
      1);
  hidden = UNetStage(hidden, prefix + ".down_blocks.0", time_embedding);
  const mx::array skip = hidden;
  hidden = ChannelsFirstConvolution(
      hidden, prefix + ".down_blocks.0.downsample.conv.conv", true);
  for (int stage = 0; stage < kMidStages; ++stage) {
    hidden = UNetStage(hidden, prefix + ".mid_blocks." + std::to_string(stage),
                       time_embedding);
  }
  hidden = UNetStage(mx::concatenate({hidden, skip}, 1),
                     prefix + ".up_blocks.0", time_embedding);
  hidden = ChannelsFirstConvolution(
      hidden, prefix + ".up_blocks.0.upsample.conv.conv", true);
  hidden = CausalBlock(hidden, prefix + ".final_block");
  return ChannelsFirstConvolution(hidden, prefix + ".final_proj.conv", false);
}

mx::array FlowDecoder::Mel(const mx::array &speech_tokens,
                           const FlowPrompt &prompt) const {
  const int token_count = speech_tokens.shape(1);
  const int prompt_mel_frames = prompt.prompt_mel.shape(1);
  const mx::array speaker = prompt.speaker_embedding;
  const mx::array speaker_conditioning = checkpoint_.Linear(
      mx::divide(speaker, mx::add(mx::linalg::norm(speaker, -1, true),
                                  mx::array(kSpeakerNormFloor))),
      "s3gen.spk_embed_affine_layer");
  const mx::array tokens = mx::astype(
      mx::concatenate({prompt.prompt_tokens, speech_tokens}, 1), mx::int32);
  const mx::array encoded =
      Encode(checkpoint_.Embed(tokens, "s3gen.input_embedding"));
  const mx::array projected = checkpoint_.Linear(encoded, "s3gen.encoder_proj");
  const int mel_frames = projected.shape(1);
  const mx::array mel_conditioning = mx::transpose(
      mx::concatenate(
          {prompt.prompt_mel,
           mx::zeros({1, mel_frames - prompt_mel_frames, kMelBins})},
          1),
      {0, 2, 1});
  const mx::array mu = mx::transpose(projected, {0, 2, 1});
  // Note (Jiaxin Deng): two draws in this order, the generated frames' noise
  // and then noise over every frame whose prompt part is kept, so a seeded run
  // consumes the same keys.
  const int generated_frames = kMelFramesPerToken * token_count;
  const mx::array generated_noise =
      mx::random::normal({1, kMelBins, generated_frames});
  const mx::array frame_noise = mx::random::normal(mu.shape());
  const int prompt_noise_frames = mel_frames - generated_frames;
  mx::array x = mx::concatenate(
      {mx::slice(frame_noise, {0, 0, 0}, {1, kMelBins, prompt_noise_frames}),
       generated_noise},
      2);
  const mx::array times = mx::linspace(0.0, 1.0, kFlowSteps + 1);
  for (int step = 0; step < kFlowSteps; ++step) {
    const mx::array t = mx::slice(times, {step}, {step + 1});
    const mx::array r = mx::slice(times, {step + 1}, {step + 2});
    const mx::array velocity =
        Velocity(x, mu, speaker_conditioning, mel_conditioning, t, r);
    x = mx::add(x, mx::multiply(mx::subtract(r, t), velocity));
  }
  return mx::slice(x, {0, 0, prompt_mel_frames}, {1, kMelBins, mel_frames});
}

} // namespace chatterbox
