// SPDX-License-Identifier: Apache-2.0
#include "haru/haru_codec.h"

namespace haru {

namespace mx = mlx::core;

namespace {

constexpr int kBlockKernel = 7;

mx::array SliceRows(const mx::array &x, int begin, int end) {
  return mx::slice(x, mx::Shape{begin, 0}, mx::Shape{end, x.shape(1)});
}

} // namespace

void PrepareCodecWeights(omni_common::WeightMap &weights) {
  std::vector<std::string> names;
  for (const auto &[name, weight] : weights) {
    if (name.starts_with("codec.") && name.ends_with(".weight") &&
        weight.ndim() == 3 && !name.starts_with("codec.decoder.up.")) {
      names.push_back(name);
    } else {
    }
  }
  for (const std::string &name : names) {
    const mx::array stored = weights.at(name); // [out, in, kernel]
    const int output_channels = stored.shape(0);
    if (name.ends_with(".conv.weight")) {
      // The block's residual gain is folded into its depthwise filter.
      const std::string block = name.substr(0, name.size() - 12);
      const mx::array gain =
          mx::astype(weights.at(block + ".gamma"), mx::float32);
      const mx::array filter =
          mx::astype(mx::transpose(stored, {0, 2, 1}), mx::float32);
      const mx::array bias = weights.at(block + ".conv.bias");
      weights.insert_or_assign(
          name,
          mx::astype(
              mx::multiply(filter, mx::reshape(gain, {output_channels, 1, 1})),
              stored.dtype()));
      weights.insert_or_assign(
          block + ".conv.bias",
          mx::astype(mx::multiply(bias, gain), bias.dtype()));
      weights.erase(block + ".gamma");
    } else {
      weights.insert_or_assign(
          name,
          mx::reshape(mx::transpose(stored, {2, 1, 0}),
                      {stored.shape(2) * stored.shape(1), output_channels}));
    }
  }
}

VibeVoiceCodec::VibeVoiceCodec(const omni_common::Checkpoint &checkpoint,
                               const CodecConfig &config)
    : checkpoint_(checkpoint), config_(config) {
  const int stage_count = static_cast<int>(config.ratios.size());
  const int context = kBlockKernel - 1;
  const int widest = config.base_channels << stage_count;
  encoder_layers_.push_back({LayerKind::kConvolution, "codec.encoder.stem",
                             kBlockKernel, 1, context, 1});
  decoder_layers_.push_back({LayerKind::kConvolution, "codec.decoder.stem",
                             kBlockKernel, 1, context, config.latent_dim});
  for (int stage = 0; stage <= stage_count; ++stage) {
    const std::string index = std::to_string(stage);
    const int encoder_channels = config.base_channels << stage;
    const int decoder_channels = widest >> stage;
    if (stage > 0) {
      const std::string previous = std::to_string(stage - 1);
      const int encoder_stride = config.ratios[stage_count - stage];
      encoder_layers_.push_back({LayerKind::kConvolution,
                                 "codec.encoder.down." + previous,
                                 2 * encoder_stride, encoder_stride,
                                 encoder_stride, encoder_channels / 2});
      const int decoder_stride = config.ratios[stage - 1];
      decoder_layers_.push_back({LayerKind::kTransposedConvolution,
                                 "codec.decoder.up." + previous,
                                 2 * decoder_stride, decoder_stride,
                                 decoder_stride, decoder_channels});
    } else {
    }
    for (int block = 0; block < config.encoder_depths[stage]; ++block) {
      encoder_layers_.push_back(
          {LayerKind::kBlock,
           "codec.encoder.stages." + index + "." + std::to_string(block),
           kBlockKernel, 1, context, encoder_channels});
    }
    for (int block = 0; block < config.decoder_depths[stage]; ++block) {
      decoder_layers_.push_back(
          {LayerKind::kBlock,
           "codec.decoder.stages." + index + "." + std::to_string(block),
           kBlockKernel, 1, context, decoder_channels});
    }
  }
  encoder_layers_.push_back({LayerKind::kConvolution, "codec.encoder.head",
                             kBlockKernel, 1, context, widest});
  decoder_layers_.push_back({LayerKind::kConvolution, "codec.decoder.head",
                             kBlockKernel, 1, context, config.base_channels});
}

VibeVoiceCodec::State
VibeVoiceCodec::InitState(const std::vector<Layer> &layers) const {
  State state;
  const mx::Dtype dtype =
      checkpoint_.Weight(layers.front().prefix + ".weight").dtype();
  for (const Layer &layer : layers) {
    state.push_back(mx::zeros({layer.state_rows, layer.state_columns}, dtype));
  }
  return state;
}

VibeVoiceCodec::State VibeVoiceCodec::InitEncoderState() const {
  return InitState(encoder_layers_);
}

VibeVoiceCodec::State VibeVoiceCodec::InitDecoderState() const {
  return InitState(decoder_layers_);
}

VibeVoiceCodec::LayerOutput VibeVoiceCodec::Convolve(const mx::array &x,
                                                     const mx::array &state,
                                                     const Layer &layer) const {
  const int stride = layer.stride;
  const int input_channels = x.shape(1);
  const int remainder = x.shape(0) % stride;
  const mx::array input =
      remainder == 0 ? x
                     : mx::pad(x, std::vector<std::pair<int, int>>{
                                      {0, stride - remainder}, {0, 0}});
  const mx::array context = mx::concatenate({state, input}, 0);
  std::vector<mx::array> window_parts;
  if (stride == 1) {
    for (int tap = 0; tap < layer.kernel; ++tap) {
      window_parts.push_back(SliceRows(context, tap, tap + input.shape(0)));
    }
  } else {
    const int hops = context.shape(0) / stride;
    const mx::array hop_rows =
        mx::reshape(context, {hops, stride * input_channels});
    window_parts = {SliceRows(hop_rows, 0, hops - 1),
                    SliceRows(hop_rows, 1, hops)};
  }
  return {mx::addmm(checkpoint_.Weight(layer.prefix + ".bias"),
                    mx::concatenate(window_parts, 1),
                    checkpoint_.Weight(layer.prefix + ".weight")),
          SliceRows(context, context.shape(0) - layer.state_rows,
                    context.shape(0))};
}

VibeVoiceCodec::LayerOutput
VibeVoiceCodec::ConvolveTransposed(const mx::array &x, const mx::array &state,
                                   const Layer &layer) const {
  const mx::array &stored =
      checkpoint_.Weight(layer.prefix + ".weight"); // [in, out, kernel]
  const int stride = layer.stride;
  const int output_channels = stored.shape(1);
  const int steps = x.shape(0);
  // [steps, out * kernel] -> [2, steps, stride, out]: each step's window as
  // the part it starts and the part that overlaps the next step.
  const mx::array windows = mx::transpose(
      mx::reshape(mx::matmul(x, mx::reshape(stored, {stored.shape(0), -1})),
                  {steps, output_channels, 2, stride}),
      {2, 0, 3, 1});
  const mx::array starts = mx::take(windows, 0, 0);
  const mx::array tails = mx::take(windows, 1, 0);
  const mx::array overlaps = mx::concatenate(
      {mx::expand_dims(state, 0),
       mx::slice(tails, mx::Shape{0, 0, 0},
                 mx::Shape{steps - 1, stride, output_channels})},
      0);
  return {mx::add(mx::reshape(mx::add(starts, overlaps), {-1, output_channels}),
                  checkpoint_.Weight(layer.prefix + ".bias")),
          mx::take(tails, steps - 1, 0)};
}

VibeVoiceCodec::LayerOutput VibeVoiceCodec::Block(const mx::array &x,
                                                  const mx::array &state,
                                                  const Layer &layer) const {
  const std::string &prefix = layer.prefix;
  const mx::array context = mx::concatenate(
      {state, checkpoint_.RmsNorm(x, prefix + ".norm", config_.norm_eps)}, 0);
  const mx::array mixed = mx::add(
      x, mx::squeeze(checkpoint_.Conv1d(mx::expand_dims(context, 0),
                                        prefix + ".conv", 1, 0, 1, x.shape(1)),
                     0));
  const mx::array feed_forward = checkpoint_.Linear(
      omni_common::Gelu(checkpoint_.Linear(
          checkpoint_.RmsNorm(mixed, prefix + ".ffn_norm", config_.norm_eps),
          prefix + ".fc1")),
      prefix + ".fc2");
  return {mx::add(mixed, mx::multiply(checkpoint_.Weight(prefix + ".ffn_gamma"),
                                      feed_forward)),
          SliceRows(context, x.shape(0), context.shape(0))};
}

mx::array VibeVoiceCodec::Run(const std::vector<Layer> &layers, mx::array x,
                              State &state) const {
  for (size_t index = 0; index < layers.size(); ++index) {
    const Layer &layer = layers[index];
    const LayerOutput layer_output =
        layer.kind == LayerKind::kConvolution ? Convolve(x, state[index], layer)
        : layer.kind == LayerKind::kTransposedConvolution
            ? ConvolveTransposed(x, state[index], layer)
            : Block(x, state[index], layer);
    x = layer_output.output;
    state[index] = layer_output.next_state;
  }
  return mx::astype(x, mx::float32);
}

mx::array VibeVoiceCodec::Encode(const mx::array &audio, State &state) const {
  return Run(encoder_layers_,
             mx::astype(mx::reshape(audio, {-1, 1}), state.front().dtype()),
             state);
}

mx::array VibeVoiceCodec::Decode(const mx::array &latents, State &state) const {
  return mx::reshape(
      Run(decoder_layers_,
          mx::astype(mx::reshape(latents, {-1, latents.shape(-1)}),
                     state.front().dtype()),
          state),
      {-1});
}

mx::array VibeVoiceCodec::Sample(const mx::array &mean,
                                 const mx::array &key) const {
  const auto keys = mx::random::split(key);
  const mx::array scale = mx::random::normal(mx::Shape{mean.shape(0), 1, 1},
                                             mx::float32, keys.first);
  const mx::array noise =
      mx::random::normal(mean.shape(), mx::float32, keys.second);
  // The VQ-style sample's noise scale (latent_std / 0.8).
  constexpr float kVqSampleScale = 0.8f;
  const float factor = config_.latent_std / kVqSampleScale;
  return mx::add(
      mx::astype(mean, mx::float32),
      mx::multiply(scale, mx::multiply(mx::array(factor, mx::float32), noise)));
}

} // namespace haru
