// SPDX-License-Identifier: Apache-2.0
#include "haru/haru_codec.h"

namespace haru {

namespace mx = mlx::core;

namespace {

// x sliced to [begin, end) along axis.
mx::array SliceAxis(const mx::array &x, int axis, int begin, int end) {
  mx::Shape start(x.ndim(), 0);
  mx::Shape stop = x.shape();
  start[axis] = begin;
  stop[axis] = end;
  return mx::slice(x, start, stop);
}

enum class Kind { Conv, ConvTranspose, Block };

struct Layer {
  Kind kind;
  std::string prefix;
  int in_ch = 0;
  int out_ch = 0;
  int kernel = 0;
  int stride = 1;
  int state_rows = 0;
  int state_cols = 0;
};

// The encoder's layer sequence.
std::vector<Layer> EncoderLayers(const CodecConfig &config) {
  const int c = config.base_channels;
  const int n = static_cast<int>(config.ratios.size());
  std::vector<Layer> layers;
  layers.push_back({Kind::Conv, "codec.encoder.stem", 1, c, 7, 1, 6, 1});
  for (int i = 0; i <= n; ++i) {
    const int ch = c << i;
    if (i > 0) {
      const int stride = config.ratios[n - i];  // ratios reversed
      layers.push_back({Kind::Conv, "codec.encoder.down." + std::to_string(i - 1),
                        ch >> 1, ch, 2 * stride, stride, stride, ch >> 1});
    }
    for (int b = 0; b < config.encoder_depths[i]; ++b) {
      layers.push_back({Kind::Block, "codec.encoder.stages." + std::to_string(i) +
                                         "." + std::to_string(b),
                        ch, ch, 0, 1, 6, ch});
    }
  }
  layers.push_back({Kind::Conv, "codec.encoder.head", c << n, config.latent_dim, 7,
                    1, 6, c << n});
  return layers;
}

std::vector<Layer> DecoderLayers(const CodecConfig &config) {
  const int c = config.base_channels;
  const int n = static_cast<int>(config.ratios.size());
  std::vector<Layer> layers;
  layers.push_back({Kind::Conv, "codec.decoder.stem", config.latent_dim, c << n, 7,
                    1, 6, config.latent_dim});
  for (int i = 0; i <= n; ++i) {
    const int ch = c << (n - i);
    if (i > 0) {
      const int stride = config.ratios[i - 1];
      // up[i-1]: in c<<(n-i+1), out c<<(n-i); state is [stride, out].
      layers.push_back({Kind::ConvTranspose, "codec.decoder.up." + std::to_string(i - 1),
                        ch << 1, ch, 0, stride, stride, ch});
    }
    for (int b = 0; b < config.decoder_depths[i]; ++b) {
      layers.push_back({Kind::Block, "codec.decoder.stages." + std::to_string(i) +
                                         "." + std::to_string(b),
                        ch, ch, 0, 1, 6, ch});
    }
  }
  layers.push_back({Kind::Conv, "codec.decoder.head", c, 1, 7, 1, 6, c});
  return layers;
}

} // namespace

VibeVoiceCodec::VibeVoiceCodec(const omni_common::Checkpoint &checkpoint,
                               const CodecConfig &config)
    : checkpoint_(checkpoint), config_(config) {}

VibeVoiceCodec::State VibeVoiceCodec::InitEncoderState() const {
  State state;
  const auto dtype = checkpoint_.Weight("codec.encoder.stem.weight").dtype();
  for (const auto &layer : EncoderLayers(config_)) {
    state.push_back(mx::zeros({layer.state_rows, layer.state_cols}, dtype));
  }
  return state;
}

VibeVoiceCodec::State VibeVoiceCodec::InitDecoderState() const {
  State state;
  const auto dtype = checkpoint_.Weight("codec.decoder.stem.weight").dtype();
  for (const auto &layer : DecoderLayers(config_)) {
    state.push_back(mx::zeros({layer.state_rows, layer.state_cols}, dtype));
  }
  return state;
}

std::pair<mx::array, mx::array>
VibeVoiceCodec::Conv(const mx::array &x, const mx::array &state,
                     const std::string &prefix, int stride, int kernel) const {
  const mx::array raw = checkpoint_.Weight(prefix + ".weight");  // [out, in, k]
  const int out_ch = raw.shape(0);
  const int in_ch = raw.shape(1);
  // The model evaluates the conv as a matmul over windows: [out, k * in].
  const mx::array weight = mx::reshape(mx::transpose(raw, {0, 2, 1}),
                                       {out_ch, kernel * in_ch});

  mx::array input = x;
  const int tail = x.shape(0) % stride;
  if (tail) {
    input = mx::pad(x, std::vector<std::pair<int, int>>{{0, stride - tail}, {0, 0}});
  }
  const mx::array context = mx::concatenate({state, input}, 0);  // [T+ctx, in]

  const auto build_windows = [&]() -> mx::array {
    if (stride == 1) {
      const int steps = input.shape(0);
      std::vector<mx::array> taps;
      taps.reserve(kernel);
      for (int k = 0; k < kernel; ++k) {
        taps.push_back(mx::slice(context, mx::Shape{k, 0},
                                 mx::Shape{k + steps, in_ch}));
      }
      return mx::concatenate(taps, 1);  // [steps, k*in]
    }
    const int hops = context.shape(0) / stride;
    const mx::array reshaped =
        mx::reshape(context, {hops, stride * in_ch});  // [hops, stride*in]
    return mx::concatenate(
        {mx::slice(reshaped, mx::Shape{0, 0}, mx::Shape{hops - 1, stride * in_ch}),
         mx::slice(reshaped, mx::Shape{1, 0}, mx::Shape{hops, stride * in_ch})},
        1);
  };
  const mx::array windows = build_windows();
  const mx::array out = mx::addmm(checkpoint_.Weight(prefix + ".bias"), windows,
                                  mx::transpose(weight));
  const int context_len = kernel - stride;
  const mx::array next = mx::slice(
      context, mx::Shape{context.shape(0) - context_len, 0},
      mx::Shape{context.shape(0), in_ch});
  return {out, next};
}

std::pair<mx::array, mx::array>
VibeVoiceCodec::ConvTranspose(const mx::array &x, const mx::array &state,
                              const std::string &prefix, int stride) const {
  const mx::array raw = checkpoint_.Weight(prefix + ".weight");  // [in, out, k]
  const int in_ch = raw.shape(0);
  const int out_ch = raw.shape(1);
  const int kernel = raw.shape(2);
  const mx::array weight = mx::reshape(mx::transpose(raw, {1, 2, 0}),
                                       {out_ch * kernel, in_ch});

  const int steps = x.shape(0);
  // [steps, out * kernel] -> [steps, out, 2, stride] -> [steps, 2, stride, out].
  const mx::array y = mx::transpose(
      mx::reshape(mx::matmul(x, mx::transpose(weight)),
                  {steps, out_ch, 2, stride}),
      {0, 2, 3, 1});
  const mx::array first = mx::squeeze(SliceAxis(y, 1, 0, 1), 1);   // [steps, stride, out]
  const mx::array second = mx::squeeze(SliceAxis(y, 1, 1, 2), 1);  // [steps, stride, out]
  const mx::array overlap = mx::concatenate(
      {mx::expand_dims(state, 0),
       mx::slice(second, mx::Shape{0, 0, 0}, mx::Shape{steps - 1, stride, out_ch})},
      0);
  const mx::array out = mx::add(
      mx::reshape(mx::add(first, overlap), {-1, out_ch}),
      checkpoint_.Weight(prefix + ".bias"));
  const mx::array next = mx::squeeze(
      mx::slice(second, mx::Shape{steps - 1, 0, 0}, mx::Shape{steps, stride, out_ch}),
      0);
  return {out, next};
}

std::pair<mx::array, mx::array>
VibeVoiceCodec::Block(const mx::array &x, const mx::array &state,
                      const std::string &prefix) const {
  const int dim = x.shape(1);
  const int steps = x.shape(0);
  constexpr int kKernel = 7;
  const mx::array n =
      mx::fast::rms_norm(x, checkpoint_.Weight(prefix + ".norm.weight"),
                         config_.norm_eps);
  // The checkpoint stores the depthwise filter as [dim, 1, 7]; the model
  // evaluates it as [dim, 7] (one row per channel).
  const mx::array raw_conv = checkpoint_.Weight(prefix + ".conv.weight");
  const mx::array weight = mx::reshape(mx::transpose(raw_conv, {0, 2, 1}),
                                       {raw_conv.shape(0), raw_conv.shape(2)});
  const mx::array gamma = checkpoint_.Weight(prefix + ".gamma");
  const mx::array context = mx::concatenate({state, n}, 0);  // [steps+6, dim]
  const mx::array taps = mx::astype(context, mx::float32);
  const mx::array w = mx::astype(weight, mx::float32);
  const mx::array g = mx::astype(gamma, mx::float32);
  mx::array acc = mx::astype(checkpoint_.Weight(prefix + ".conv.bias"), mx::float32);
  for (int k = 0; k < kKernel; ++k) {
    const mx::array tap =
        mx::slice(taps, mx::Shape{k, 0}, mx::Shape{k + steps, dim});
    const mx::array w_k = mx::reshape(
        mx::slice(w, mx::Shape{0, k}, mx::Shape{dim, k + 1}), {1, dim});
    acc = mx::add(acc, mx::multiply(tap, w_k));
  }
  const mx::array mixed = mx::astype(
      mx::add(mx::astype(x, mx::float32), mx::multiply(g, acc)), x.dtype());
  const mx::array next = mx::slice(context, mx::Shape{steps, 0},
                                   mx::Shape{steps + kKernel - 1, dim});

  const mx::array h =
      mx::fast::rms_norm(mixed, checkpoint_.Weight(prefix + ".ffn_norm.weight"),
                         config_.norm_eps);
  const mx::array ffn =
      checkpoint_.Linear(omni_common::Gelu(checkpoint_.Linear(h, prefix + ".fc1")),
                         prefix + ".fc2");
  return {mx::add(mixed, mx::multiply(checkpoint_.Weight(prefix + ".ffn_gamma"), ffn)),
          next};
}

mx::array VibeVoiceCodec::Encode(const mx::array &audio, State &state) const {
  const auto layers = EncoderLayers(config_);
  mx::array x = mx::astype(mx::reshape(audio, {-1, 1}),
                           checkpoint_.Weight("codec.encoder.stem.weight").dtype());
  for (size_t i = 0; i < layers.size(); ++i) {
    const auto &layer = layers[i];
    if (layer.kind == Kind::Conv) {
      auto [y, s] = Conv(x, state[i], layer.prefix, layer.stride, layer.kernel);
      x = y;
      state[i] = s;
    } else {
      auto [y, s] = Block(x, state[i], layer.prefix);
      x = y;
      state[i] = s;
    }
  }
  return mx::astype(x, mx::float32);
}

mx::array VibeVoiceCodec::Decode(const mx::array &latents, State &state) const {
  const auto layers = DecoderLayers(config_);
  mx::array x = mx::astype(mx::reshape(latents, {-1, latents.shape(-1)}),
                           checkpoint_.Weight("codec.decoder.stem.weight").dtype());
  for (size_t i = 0; i < layers.size(); ++i) {
    const auto &layer = layers[i];
    if (layer.kind == Kind::Conv) {
      auto [y, s] = Conv(x, state[i], layer.prefix, layer.stride, layer.kernel);
      x = y;
      state[i] = s;
    } else if (layer.kind == Kind::ConvTranspose) {
      auto [y, s] = ConvTranspose(x, state[i], layer.prefix, layer.stride);
      x = y;
      state[i] = s;
    } else {
      auto [y, s] = Block(x, state[i], layer.prefix);
      x = y;
      state[i] = s;
    }
  }
  return mx::astype(mx::reshape(x, {-1}), mx::float32);
}

mx::array VibeVoiceCodec::Sample(const mx::array &mean,
                                 const mx::array &key) const {
  const auto keys = mx::random::split(key);
  const mx::array scale =
      mx::random::normal(mx::Shape{mean.shape(0), 1, 1}, mx::float32, keys.first);
  const mx::array noise = mx::random::normal(mean.shape(), mx::float32, keys.second);
  // The VQ-style sample's noise scale (latent_std / 0.8).
  constexpr float kVqSampleScale = 0.8f;
  const float factor = config_.latent_std / kVqSampleScale;
  return mx::add(mx::astype(mean, mx::float32),
                 mx::multiply(scale, mx::multiply(mx::array(factor, mx::float32), noise)));
}

} // namespace haru
