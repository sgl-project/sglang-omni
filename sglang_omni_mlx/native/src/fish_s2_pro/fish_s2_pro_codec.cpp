// SPDX-License-Identifier: Apache-2.0
#include "fish_s2_pro/fish_s2_pro_codec.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <stdexcept>

namespace fish_s2_pro {

namespace mx = mlx::core;

namespace {

constexpr int kEncoderStrides[] = {2, 4, 8, 8};
constexpr int kDecoderStrides[] = {8, 8, 4, 2};
constexpr int kResidualDilations[] = {1, 3, 9};
constexpr int kQuantizerStageCount = 2;
constexpr int kResidualCodebookCount = 9;
constexpr int kSemanticCodebookSize = 4096;
constexpr int kLatentDimension = 1024;
constexpr int kTransformerHeadDim = 64;
// The encoder's output and the decoder's input run at 4 latent frames per
// code frame, 512 samples each.
constexpr int kSamplesPerLatentFrame = 512;
constexpr int kDecoderWindowFrames = 256;
constexpr int kDecoderContextFrames = 32;
constexpr int kEncoderConvolutionWindowFrames = 256;
constexpr int kEncoderConvolutionContextFrames = 32;
constexpr int kQuantizerTransformerLayers = 8;
constexpr int kQuantizerWindowFrames = 128;
constexpr int kEncoderTransformerLayers = 4;
constexpr int kEncoderWindowFrames = 512;
constexpr float kTransformerNormEps = 1e-5f;
constexpr float kConvNextNormEps = 1e-6f;
constexpr float kMaskedScore = -1e9f;
constexpr float kSnakeEps = 1e-9f;
constexpr float kNormFloor = 1e-12f;

constexpr int kStreamFirstWindowFrames = 40;
constexpr int kStreamFollowupFrames = 45;
constexpr int kStreamOverlapFrames = 20;
constexpr size_t kStreamCrossfadeSamples = 512;

constexpr float kPcm16FullScale = 32767.0f;
constexpr int kWaveFormatExtensible = 0xFFFE;
constexpr int kResampleZeroCrossings = 6;
constexpr double kResampleRolloff = 0.99;
// Bounds on the polyphase kernel table, which a rate with a small common
// divisor with the target (44101 Hz) would grow to gigabytes.
constexpr int kMaxResamplePhases = 4096;
constexpr int kMaxResampleTaps = 2048;

bool IsTransposedConvolution(const std::string &prefix) {
  for (int stage = 0; stage < kQuantizerStageCount; ++stage) {
    if (prefix == "quantizer.upsample." + std::to_string(stage) + ".0") {
      return true;
    } else {
    }
  }
  for (int block = 1; block <= 4; ++block) {
    if (prefix == "decoder.model." + std::to_string(block) + ".block.1") {
      return true;
    } else {
    }
  }
  return false;
}

// Weight-normed kernels folded as the codec computes them per call, and
// every convolution kernel in MLX's [out, kernel, in] layout.
omni_common::WeightMap PrepareWeights(omni_common::WeightMap weights) {
  omni_common::WeightMap prepared;
  const auto to_mlx_layout = [](const std::string &prefix,
                                const mx::array &torch_kernel) {
    return IsTransposedConvolution(prefix)
               ? mx::transpose(torch_kernel, {1, 2, 0})
               : mx::transpose(torch_kernel, {0, 2, 1});
  };
  for (const auto &[name, array] : weights) {
    const auto ends_with = [&](const std::string &suffix) {
      return name.size() >= suffix.size() &&
             name.compare(name.size() - suffix.size(), suffix.size(), suffix) ==
                 0;
    };
    if (ends_with(".weight_v")) {
      const std::string prefix = name.substr(0, name.size() - 9);
      const mx::array norm =
          mx::sqrt(mx::sum(mx::power(array, mx::array(2.0f)), {1, 2}, true));
      prepared.insert_or_assign(
          prefix + ".weight",
          to_mlx_layout(
              prefix,
              mx::divide(mx::multiply(weights.at(prefix + ".weight_g"), array),
                         norm)));
    } else if (ends_with(".weight_g") || ends_with(".causal_mask")) {
      // Note (Jiaxin Deng): each transformer's stored causal mask is never
      // read: the window-limited mask is built per call.
    } else if (ends_with(".conv.weight")) {
      const std::string prefix = name.substr(0, name.size() - 12);
      prepared.insert_or_assign(prefix + ".weight",
                                to_mlx_layout(prefix, array));
    } else if (ends_with(".conv.bias")) {
      prepared.insert_or_assign(name.substr(0, name.size() - 10) + ".bias",
                                array);
    } else {
      prepared.insert_or_assign(name, array);
    }
  }
  return prepared;
}

// x [batch, frames, heads, head_dim] rotated by phases [frames, half, 2],
// in float32.
mx::array Rotate(const mx::array &x, const mx::array &phases) {
  const int batch = x.shape(0);
  const int frames = x.shape(1);
  const int heads = x.shape(2);
  const int half = x.shape(3) / 2;
  const mx::array pairs =
      mx::reshape(mx::astype(x, mx::float32), {batch, frames, heads, half, 2});
  const mx::array frequencies =
      mx::reshape(mx::astype(phases, mx::float32), {1, frames, 1, half, 2});
  const auto part = [](const mx::array &array, int index) {
    return mx::squeeze(mx::slice(array, {0, 0, 0, 0, index},
                                 {array.shape(0), array.shape(1),
                                  array.shape(2), array.shape(3), index + 1}),
                       -1);
  };
  const mx::array even = part(pairs, 0);
  const mx::array odd = part(pairs, 1);
  const mx::array cosines = part(frequencies, 0);
  const mx::array sines = part(frequencies, 1);
  return mx::astype(
      mx::reshape(mx::stack({mx::subtract(mx::multiply(even, cosines),
                                          mx::multiply(odd, sines)),
                             mx::add(mx::multiply(odd, cosines),
                                     mx::multiply(even, sines))},
                            -1),
                  x.shape()),
      x.dtype());
}

} // namespace

AudioCodec::AudioCodec(omni_common::WeightMap weights)
    : checkpoint_(PrepareWeights(std::move(weights)),
                  nlohmann::json::object()) {}

mx::array AudioCodec::CausalConvolution(const mx::array &x,
                                        const std::string &prefix, int stride,
                                        int dilation, int groups) const {
  const mx::array &kernel = checkpoint_.Weight(prefix + ".weight");
  const int kernel_span = (kernel.shape(1) - 1) * dilation + 1;
  const int padding = kernel_span - stride;
  const int length = x.shape(2);
  // Right padding so the last window is whole, as the codec computes it.
  const double frame_count =
      static_cast<double>(length - kernel_span + padding) / stride + 1.0;
  const int ideal_length =
      (static_cast<int>(std::ceil(frame_count)) - 1) * stride +
      (kernel_span - padding);
  const int extra = std::max(0, ideal_length - length);
  const mx::array padded = mx::pad(x, {{0, 0}, {0, 0}, {padding, extra}});
  const mx::array y = mx::add(mx::conv1d(mx::swapaxes(padded, 1, 2), kernel,
                                         stride, 0, dilation, groups),
                              checkpoint_.Weight(prefix + ".bias"));
  return mx::swapaxes(y, 1, 2);
}

mx::array AudioCodec::CausalTransposedConvolution(const mx::array &x,
                                                  const std::string &prefix,
                                                  int stride) const {
  const mx::array &kernel = checkpoint_.Weight(prefix + ".weight");
  const mx::array y = mx::swapaxes(
      mx::add(mx::conv_transpose1d(mx::swapaxes(x, 1, 2), kernel, stride),
              checkpoint_.Weight(prefix + ".bias")),
      1, 2);
  // The kernel's overhang beyond the stride is cut from the right.
  const int trimmed = kernel.shape(1) - stride;
  return mx::slice(y, {0, 0, 0},
                   {y.shape(0), y.shape(1), y.shape(2) - trimmed});
}

mx::array AudioCodec::Snake(const mx::array &x,
                            const std::string &prefix) const {
  const mx::array &alpha = checkpoint_.Weight(prefix + ".alpha");
  return mx::add(
      x, mx::multiply(
             mx::reciprocal(mx::add(alpha, mx::array(kSnakeEps))),
             mx::power(mx::sin(mx::multiply(alpha, x)), mx::array(2.0f))));
}

mx::array AudioCodec::ResidualUnit(const mx::array &x,
                                   const std::string &prefix,
                                   int dilation) const {
  mx::array y = Snake(x, prefix + ".block.0");
  y = CausalConvolution(y, prefix + ".block.1", 1, dilation, 1);
  y = Snake(y, prefix + ".block.2");
  y = CausalConvolution(y, prefix + ".block.3", 1, 1, 1);
  return mx::add(x, y);
}

mx::array AudioCodec::ConvNextBlock(const mx::array &x,
                                    const std::string &prefix) const {
  mx::array y = CausalConvolution(x, prefix + ".dwconv", 1, 1, x.shape(1));
  y = checkpoint_.LayerNorm(mx::swapaxes(y, 1, 2), prefix + ".norm",
                            kConvNextNormEps);
  y = checkpoint_.Linear(
      omni_common::Gelu(checkpoint_.Linear(y, prefix + ".pwconv1")),
      prefix + ".pwconv2");
  y = mx::multiply(checkpoint_.Weight(prefix + ".gamma"), y);
  return mx::add(x, mx::swapaxes(y, 1, 2));
}

mx::array AudioCodec::TransformerRmsNorm(const mx::array &x,
                                         const std::string &prefix) const {
  const mx::array wide = mx::astype(x, mx::float32);
  const mx::array normed = mx::multiply(
      wide, mx::rsqrt(mx::add(mx::mean(mx::multiply(wide, wide), -1, true),
                              mx::array(kTransformerNormEps))));
  return mx::multiply(mx::astype(normed, x.dtype()),
                      checkpoint_.Weight(prefix + ".weight"));
}

mx::array AudioCodec::Transformer(const mx::array &x, const std::string &prefix,
                                  int layer_count, int head_count,
                                  int window_frames) const {
  mx::array hidden = mx::swapaxes(x, 1, 2);
  const int batch = hidden.shape(0);
  const int frames = hidden.shape(1);
  const int dimension = hidden.shape(2);
  // Note (Jiaxin Deng): the checkpoint stores each transformer's rotary
  // phases in bfloat16, and the codec rotates by those.
  const mx::array &table = checkpoint_.Weight(prefix + ".freqs_cis");
  if (frames > table.shape(0)) {
    throw std::invalid_argument("audio is too long for the codec");
  } else {
  }
  const mx::array phases =
      mx::slice(table, {0, 0, 0}, {frames, table.shape(1), 2});
  const mx::array rows = mx::reshape(mx::arange(frames), {frames, 1});
  const mx::array columns = mx::reshape(mx::arange(frames), {1, frames});
  const mx::array visible = mx::logical_and(
      mx::greater_equal(columns,
                        mx::maximum(mx::add(rows, mx::array(1 - window_frames)),
                                    mx::array(0))),
      mx::less_equal(columns, rows));
  const mx::array mask =
      mx::reshape(mx::where(visible, mx::array(0.0f), mx::array(kMaskedScore)),
                  {1, 1, frames, frames});
  const auto heads = [&](const mx::array &projection) {
    return mx::reshape(projection,
                       {batch, frames, head_count, kTransformerHeadDim});
  };
  for (int layer = 0; layer < layer_count; ++layer) {
    const std::string layer_prefix =
        prefix + ".layers." + std::to_string(layer);
    const mx::array qkv = checkpoint_.Linear(
        TransformerRmsNorm(hidden, layer_prefix + ".attention_norm"),
        layer_prefix + ".attention.wqkv");
    const mx::array queries = mx::transpose(
        Rotate(heads(mx::slice(qkv, {0, 0, 0}, {batch, frames, dimension})),
               phases),
        {0, 2, 1, 3});
    const mx::array keys =
        mx::transpose(Rotate(heads(mx::slice(qkv, {0, 0, dimension},
                                             {batch, frames, 2 * dimension})),
                             phases),
                      {0, 2, 1, 3});
    const mx::array values =
        mx::transpose(heads(mx::slice(qkv, {0, 0, 2 * dimension},
                                      {batch, frames, 3 * dimension})),
                      {0, 2, 1, 3});
    const mx::array attended = mx::fast::scaled_dot_product_attention(
        queries, keys, values,
        static_cast<float>(1.0 / std::sqrt(kTransformerHeadDim)), "", mask);
    hidden = mx::add(
        hidden,
        mx::multiply(
            checkpoint_.Linear(
                mx::reshape(mx::transpose(attended, {0, 2, 1, 3}),
                            {batch, frames, dimension}),
                layer_prefix + ".attention.wo"),
            checkpoint_.Weight(layer_prefix + ".attention_layer_scale.gamma")));
    const mx::array feed_forward_input =
        TransformerRmsNorm(hidden, layer_prefix + ".ffn_norm");
    hidden = mx::add(
        hidden,
        mx::multiply(
            checkpoint_.Linear(
                mx::multiply(
                    omni_common::Silu(checkpoint_.Linear(
                        feed_forward_input, layer_prefix + ".feed_forward.w1")),
                    checkpoint_.Linear(feed_forward_input,
                                       layer_prefix + ".feed_forward.w3")),
                layer_prefix + ".feed_forward.w2"),
            checkpoint_.Weight(layer_prefix + ".ffn_layer_scale.gamma")));
  }
  return mx::swapaxes(TransformerRmsNorm(hidden, prefix + ".norm"), 1, 2);
}

mx::array AudioCodec::NearestCodes(const mx::array &latents,
                                   const std::string &prefix) const {
  const int batch = latents.shape(0);
  const int frames = latents.shape(2);
  const mx::array encodings = mx::reshape(mx::transpose(latents, {0, 2, 1}),
                                          {batch * frames, latents.shape(1)});
  const mx::array &codebook = checkpoint_.Weight(prefix + ".codebook.weight");
  const auto unit = [](const mx::array &rows) {
    return mx::divide(
        rows, mx::maximum(
                  mx::sqrt(mx::sum(mx::power(rows, mx::array(2.0f)), 1, true)),
                  mx::array(kNormFloor)));
  };
  const mx::array encoding_units = unit(encodings);
  const mx::array codebook_units = unit(codebook);
  const mx::array distances = mx::add(
      mx::subtract(mx::sum(mx::power(encoding_units, mx::array(2.0f)), 1, true),
                   mx::multiply(mx::array(2.0f),
                                mx::matmul(encoding_units,
                                           mx::transpose(codebook_units)))),
      mx::transpose(
          mx::sum(mx::power(codebook_units, mx::array(2.0f)), 1, true)));
  return mx::reshape(
      mx::astype(mx::argmax(mx::negative(distances), 1), mx::int32),
      {batch, frames});
}

mx::array AudioCodec::QuantizedFeatures(const mx::array &codes,
                                        const std::string &prefix) const {
  const mx::array entries =
      mx::take(checkpoint_.Weight(prefix + ".codebook.weight"), codes, 0);
  return CausalConvolution(mx::transpose(entries, {0, 2, 1}),
                           prefix + ".out_proj", 1, 1, 1);
}

mx::array AudioCodec::Encode(const std::vector<float> &samples) const {
  const int length = static_cast<int>(samples.size());
  const int padded_length =
      (length + kSamplesPerFrame - 1) / kSamplesPerFrame * kSamplesPerFrame;
  std::vector<float> padded(samples);
  padded.resize(padded_length, 0.0f);
  const mx::array audio =
      mx::array(padded.data(), {1, 1, padded_length}, mx::float32);

  // Note (Jiaxin Deng): the encoder's convolutions reach about 14 latent
  // frames back, so they run in windows with that context, as the decoder's
  // do; the transformer after them sees the whole reference.
  const int latent_frames = padded_length / kSamplesPerLatentFrame;
  std::vector<mx::array> windows;
  for (int start = 0; start < latent_frames;
       start += kEncoderConvolutionWindowFrames) {
    const int context = std::min(start, kEncoderConvolutionContextFrames);
    const int stop =
        std::min(start + kEncoderConvolutionWindowFrames, latent_frames);
    mx::array y = CausalConvolution(
        mx::slice(audio, {0, 0, (start - context) * kSamplesPerLatentFrame},
                  {1, 1, stop * kSamplesPerLatentFrame}),
        "encoder.block.0", 1, 1, 1);
    for (int block = 1; block <= 4; ++block) {
      const std::string prefix =
          "encoder.block." + std::to_string(block) + ".block";
      for (int unit = 0; unit < 3; ++unit) {
        y = ResidualUnit(y, prefix + "." + std::to_string(unit),
                         kResidualDilations[unit]);
      }
      y = Snake(y, prefix + ".3");
      y = CausalConvolution(y, prefix + ".4", kEncoderStrides[block - 1], 1, 1);
    }
    windows.push_back(
        mx::slice(y, {0, 0, context}, {y.shape(0), y.shape(1), y.shape(2)}));
    mx::eval(windows.back());
  }
  mx::array x = mx::concatenate(windows, 2);
  x = Transformer(x, "encoder.block.4.block.5", kEncoderTransformerLayers,
                  x.shape(1) / kTransformerHeadDim, kEncoderWindowFrames);
  x = Snake(x, "encoder.block.5");
  x = CausalConvolution(x, "encoder.block.6", 1, 1, 1);

  for (int stage = 0; stage < kQuantizerStageCount; ++stage) {
    const std::string prefix = "quantizer.downsample." + std::to_string(stage);
    x = CausalConvolution(x, prefix + ".0", 2, 1, 1);
    x = ConvNextBlock(x, prefix + ".1");
  }
  x = Transformer(x, "quantizer.pre_module", kQuantizerTransformerLayers,
                  kLatentDimension / kTransformerHeadDim,
                  kQuantizerWindowFrames);

  // Each codebook quantizes what the ones before it left over.
  const auto quantize = [&](const mx::array &residual,
                            const std::string &prefix,
                            std::vector<mx::array> &codes) {
    const mx::array projected =
        CausalConvolution(residual, prefix + ".in_proj", 1, 1, 1);
    const mx::array indices = NearestCodes(projected, prefix);
    const mx::array entries = mx::transpose(
        mx::take(checkpoint_.Weight(prefix + ".codebook.weight"), indices, 0),
        {0, 2, 1});
    codes.push_back(indices);
    return CausalConvolution(
        mx::add(projected, mx::subtract(entries, projected)),
        prefix + ".out_proj", 1, 1, 1);
  };
  std::vector<mx::array> codes;
  mx::array residual = mx::subtract(
      x, quantize(x, "quantizer.semantic_quantizer.quantizers.0", codes));
  for (int codebook = 0; codebook < kResidualCodebookCount; ++codebook) {
    residual =
        mx::subtract(residual, quantize(residual,
                                        "quantizer.quantizer.quantizers." +
                                            std::to_string(codebook),
                                        codes));
  }
  const mx::array stacked = mx::concatenate(codes, 0);
  const int frame_count = (length + kSamplesPerFrame - 1) / kSamplesPerFrame;
  return mx::slice(stacked, {0, 0}, {stacked.shape(0), frame_count});
}

mx::array AudioCodec::Decode(const mx::array &codes) const {
  const int codebook_count = codes.shape(0);
  const int frames = codes.shape(1);
  const auto codebook_codes = [&](int codebook, int size) {
    return mx::clip(mx::slice(codes, {codebook, 0}, {codebook + 1, frames}),
                    mx::array(0), mx::array(size - 1));
  };
  const mx::array semantic =
      QuantizedFeatures(codebook_codes(0, kSemanticCodebookSize),
                        "quantizer.semantic_quantizer.quantizers.0");
  const auto residual_features = [&](int codebook) {
    return QuantizedFeatures(codebook_codes(codebook, kResidualCodebookSize),
                             "quantizer.quantizer.quantizers." +
                                 std::to_string(codebook - 1));
  };
  mx::array residual = residual_features(1);
  for (int codebook = 2; codebook < codebook_count; ++codebook) {
    residual = mx::add(residual, residual_features(codebook));
  }
  mx::array x = mx::add(semantic, residual);
  x = Transformer(x, "quantizer.post_module", kQuantizerTransformerLayers,
                  kLatentDimension / kTransformerHeadDim,
                  kQuantizerWindowFrames);
  for (int stage = 0; stage < kQuantizerStageCount; ++stage) {
    const std::string prefix = "quantizer.upsample." + std::to_string(stage);
    x = CausalTransposedConvolution(x, prefix + ".0", 2);
    x = ConvNextBlock(x, prefix + ".1");
  }

  mx::eval(x);

  // Note (Jiaxin Deng): the decoder after the upsampling is causal convolution
  // alone, which reaches about 19 frames back, so it decodes in windows with
  // that context: a whole long utterance's convolution unfoldings would need
  // tens of GB.
  const int latent_frames = x.shape(2);
  std::vector<mx::array> windows;
  for (int start = 0; start < latent_frames; start += kDecoderWindowFrames) {
    const int context = std::min(start, kDecoderContextFrames);
    const int stop = std::min(start + kDecoderWindowFrames, latent_frames);
    mx::array y = CausalConvolution(
        mx::slice(x, {0, 0, start - context}, {x.shape(0), x.shape(1), stop}),
        "decoder.model.0", 1, 1, 1);
    for (int block = 1; block <= 4; ++block) {
      const std::string prefix =
          "decoder.model." + std::to_string(block) + ".block";
      y = Snake(y, prefix + ".0");
      y = CausalTransposedConvolution(y, prefix + ".1",
                                      kDecoderStrides[block - 1]);
      for (int unit = 0; unit < 3; ++unit) {
        y = ResidualUnit(y, prefix + "." + std::to_string(unit + 2),
                         kResidualDilations[unit]);
      }
    }
    y = Snake(y, "decoder.model.5");
    y = mx::tanh(CausalConvolution(y, "decoder.model.6", 1, 1, 1));
    const int skipped = context * kSamplesPerLatentFrame;
    windows.push_back(mx::slice(mx::reshape(y, {-1}), {skipped},
                                {static_cast<int>(y.size())}));
    mx::eval(windows.back());
  }
  return mx::concatenate(windows, 0);
}

std::vector<float> CodecStream::Push(const mx::array &frame_codes) {
  frames_.push_back(frame_codes);
  const int total_frames = first_kept_frame_ + static_cast<int>(frames_.size());
  const int next_window_frames =
      next_window_frames_ > 0 ? next_window_frames_ : kStreamFirstWindowFrames;
  if (total_frames < next_window_frames) {
    next_window_frames_ = next_window_frames;
    return {};
  } else {
  }
  std::vector<float> samples = DecodeWindow(false);
  next_window_frames_ = total_frames + kStreamFollowupFrames;
  return samples;
}

std::vector<float> CodecStream::Finish() {
  const int total_frames = first_kept_frame_ + static_cast<int>(frames_.size());
  if (frames_.empty() ||
      (total_frames <= emitted_frames_ && pending_tail_.empty())) {
    return std::move(pending_tail_);
  } else {
    return DecodeWindow(true);
  }
}

std::vector<float> CodecStream::DecodeWindow(bool is_final) {
  const int total_frames = first_kept_frame_ + static_cast<int>(frames_.size());
  if (total_frames <= emitted_frames_) {
    return is_final ? std::move(pending_tail_) : std::vector<float>();
  } else {
  }
  const int window_start =
      std::max(first_kept_frame_, emitted_frames_ - kStreamOverlapFrames);
  const std::vector<mx::array> window(
      frames_.begin() + (window_start - first_kept_frame_), frames_.end());
  mx::array audio = codec_.Decode(mx::stack(window, 1));
  mx::eval(audio);
  const float *decoded = audio.data<float>();
  const size_t decoded_count = audio.size();
  const size_t overlap_samples =
      static_cast<size_t>(emitted_frames_ - window_start) *
      codec_.samples_per_frame();
  if (decoded_count <= overlap_samples) {
    return {};
  } else {
  }
  std::vector<float> delta;
  const size_t crossfade_count = std::min(
      {kStreamCrossfadeSamples, pending_tail_.size(), overlap_samples});
  if (crossfade_count > 0) {
    delta.assign(pending_tail_.begin(), pending_tail_.end() - crossfade_count);
    // Note (Jiaxin Deng): torch.linspace's float32 ramp, stepped from each end.
    const float step = 1.0f / static_cast<float>(crossfade_count - 1);
    for (size_t index = 0; index < crossfade_count; ++index) {
      const float fade_in =
          crossfade_count == 1 ? 0.0f
          : index < crossfade_count / 2
              ? static_cast<float>(index) * step
              : 1.0f - static_cast<float>(crossfade_count - 1 - index) * step;
      const float held =
          pending_tail_[pending_tail_.size() - crossfade_count + index];
      const float refreshed =
          decoded[overlap_samples - crossfade_count + index];
      delta.push_back(held * (1.0f - fade_in) + refreshed * fade_in);
    }
  } else {
    delta = pending_tail_;
  }
  delta.insert(delta.end(), decoded + overlap_samples, decoded + decoded_count);
  pending_tail_.clear();
  emitted_frames_ = total_frames;
  if (is_final) {
    return delta;
  } else {
  }
  const size_t held_count = std::min(kStreamCrossfadeSamples, delta.size());
  pending_tail_.assign(delta.end() - held_count, delta.end());
  delta.resize(delta.size() - held_count);
  const int keep_from = std::max(0, total_frames - kStreamOverlapFrames);
  if (keep_from > first_kept_frame_) {
    frames_.erase(frames_.begin(),
                  frames_.begin() + (keep_from - first_kept_frame_));
    first_kept_frame_ = keep_from;
  } else {
  }
  return delta;
}

DecodedWav DecodeAnyWav(std::string_view wav_bytes) {
  const auto read = [&](size_t offset, size_t size) {
    uint64_t value = 0;
    for (size_t index = 0; index < size; ++index) {
      value |=
          static_cast<uint64_t>(static_cast<uint8_t>(wav_bytes[offset + index]))
          << (8 * index);
    }
    return value;
  };
  if (wav_bytes.size() < 12 || wav_bytes.substr(0, 4) != "RIFF" ||
      wav_bytes.substr(8, 4) != "WAVE") {
    throw std::invalid_argument("reference audio must be a RIFF/WAVE file");
  } else {
  }
  int format = 0;
  int channels = 0;
  int sample_rate = 0;
  int bits_per_sample = 0;
  size_t data_offset = 0;
  size_t data_size = 0;
  bool has_format = false;
  size_t offset = 12;
  while (offset + 8 <= wav_bytes.size()) {
    const std::string_view chunk_id = wav_bytes.substr(offset, 4);
    const uint64_t chunk_size = read(offset + 4, 4);
    offset += 8;
    if (chunk_id == "data") {
      // Note (Jiaxin Deng): a streamed or truncated WAV declares more data
      // than it holds; the data is what is there.
      data_offset = offset;
      data_size = std::min<uint64_t>(chunk_size, wav_bytes.size() - offset);
    } else if (offset + chunk_size > wav_bytes.size()) {
      break;
    } else if (chunk_id == "fmt " && chunk_size >= 16) {
      format = static_cast<int>(read(offset, 2));
      channels = static_cast<int>(read(offset + 2, 2));
      sample_rate = static_cast<int>(read(offset + 4, 4));
      bits_per_sample = static_cast<int>(read(offset + 14, 2));
      // WAVE_FORMAT_EXTENSIBLE names the format in its sub-format GUID.
      if (format == kWaveFormatExtensible && chunk_size >= 26) {
        format = static_cast<int>(read(offset + 24, 2));
      } else {
      }
      has_format = true;
    } else {
    }
    offset += chunk_size + (chunk_size % 2);
  }
  if (!has_format || sample_rate <= 0) {
    throw std::invalid_argument("reference audio WAV has no fmt chunk");
  } else if (channels < 1) {
    throw std::invalid_argument("reference audio WAV has no channels");
  } else if (data_size == 0) {
    throw std::invalid_argument("reference audio WAV has no audio data");
  } else {
  }
  std::vector<float> samples;
  const auto decode_all = [&](int bytes_per_sample, auto convert) {
    const size_t count = data_size / bytes_per_sample;
    samples.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      samples.push_back(convert(data_offset + index * bytes_per_sample));
    }
  };
  if (format == 3 && bits_per_sample == 32) {
    decode_all(4, [&](size_t at) {
      const uint32_t bits = static_cast<uint32_t>(read(at, 4));
      float value = 0.0f;
      std::memcpy(&value, &bits, sizeof(value));
      return value;
    });
  } else if (format == 3 && bits_per_sample == 64) {
    decode_all(8, [&](size_t at) {
      const uint64_t bits = read(at, 8);
      double value = 0.0;
      std::memcpy(&value, &bits, sizeof(value));
      return static_cast<float>(value);
    });
  } else if (format == 1 && bits_per_sample == 16) {
    decode_all(2, [&](size_t at) {
      const auto value = static_cast<int16_t>(read(at, 2));
      return static_cast<float>(value) / 32768.0f;
    });
  } else if (format == 1 && bits_per_sample == 24) {
    decode_all(3, [&](size_t at) {
      // The sign bit moves to the top of an int32 before scaling back down.
      const auto value =
          static_cast<int32_t>(static_cast<uint32_t>(read(at, 3)) << 8);
      return static_cast<float>(value) / 2147483648.0f;
    });
  } else if (format == 1 && bits_per_sample == 32) {
    decode_all(4, [&](size_t at) {
      const auto value = static_cast<int32_t>(read(at, 4));
      return static_cast<float>(value) / 2147483648.0f;
    });
  } else if (format == 1 && bits_per_sample == 8) {
    decode_all(1, [&](size_t at) {
      const auto value = static_cast<float>(read(at, 1));
      return (value - 128.0f) / 128.0f;
    });
  } else {
    throw std::invalid_argument(
        "reference audio must be PCM (8, 16, 24 or 32-bit) or float WAV");
  }
  if (channels > 1) {
    std::vector<float> mono(samples.size() / channels);
    for (size_t frame = 0; frame < mono.size(); ++frame) {
      float sum = 0.0f;
      for (int channel = 0; channel < channels; ++channel) {
        sum += samples[frame * channels + channel];
      }
      mono[frame] = sum / static_cast<float>(channels);
    }
    samples = std::move(mono);
  } else {
  }
  return {std::move(samples), sample_rate};
}

std::vector<float> Resample(const std::vector<float> &samples, int source_rate,
                            int target_rate) {
  if (source_rate == target_rate || samples.empty()) {
    return samples;
  } else {
  }
  const int divisor = std::gcd(source_rate, target_rate);
  const int source_step = source_rate / divisor;
  const int target_step = target_rate / divisor;
  const double base_rate =
      std::min(source_step, target_step) * kResampleRolloff;
  const int width = static_cast<int>(
      std::ceil(kResampleZeroCrossings * source_step / base_rate));
  const int taps = 2 * width + source_step;
  if (target_step > kMaxResamplePhases || taps > kMaxResampleTaps) {
    throw std::invalid_argument(
        "reference audio at " + std::to_string(source_rate) +
        " Hz cannot be resampled to " + std::to_string(target_rate) + " Hz");
  } else {
  }
  // One kernel per output phase, as torchaudio's sinc_interp_hann builds.
  std::vector<double> kernels(static_cast<size_t>(target_step) * taps);
  for (int phase = 0; phase < target_step; ++phase) {
    for (int tap = 0; tap < taps; ++tap) {
      double t = (-static_cast<double>(phase) / target_step +
                  static_cast<double>(tap - width) / source_step) *
                 base_rate;
      t = std::clamp(t, -static_cast<double>(kResampleZeroCrossings),
                     static_cast<double>(kResampleZeroCrossings));
      const double window =
          std::pow(std::cos(t * M_PI / kResampleZeroCrossings / 2), 2);
      t *= M_PI;
      const double sinc = t == 0.0 ? 1.0 : std::sin(t) / t;
      kernels[static_cast<size_t>(phase) * taps + tap] =
          sinc * window * base_rate / source_step;
    }
  }
  const size_t length = samples.size();
  const size_t output_count = static_cast<size_t>(
      std::ceil(static_cast<double>(target_step) * length / source_step));
  std::vector<float> output;
  output.reserve(output_count);
  for (size_t frame = 0; output.size() < output_count; ++frame) {
    // The padded input starts width samples before the first sample.
    const long start = static_cast<long>(frame * source_step) - width;
    for (int phase = 0; phase < target_step && output.size() < output_count;
         ++phase) {
      double sum = 0.0;
      const double *kernel = &kernels[static_cast<size_t>(phase) * taps];
      for (int tap = 0; tap < taps; ++tap) {
        const long index = start + tap;
        if (index >= 0 && index < static_cast<long>(length)) {
          sum += kernel[tap] * samples[index];
        } else {
        }
      }
      output.push_back(static_cast<float>(sum));
    }
  }
  return output;
}

std::string Pcm16Bytes(const std::vector<float> &samples) {
  std::string bytes;
  bytes.reserve(samples.size() * 2);
  for (const float sample : samples) {
    // Note (Minyi Tang): NaN becomes silence, since std::clamp passes it
    // through and casting NaN to an integer is undefined.
    const float bounded =
        std::isnan(sample) ? 0.0f : std::clamp(sample, -1.0f, 1.0f);
    const auto value = static_cast<int16_t>(bounded * kPcm16FullScale);
    bytes.push_back(static_cast<char>(value & 0xff));
    bytes.push_back(static_cast<char>((value >> 8) & 0xff));
  }
  return bytes;
}

std::string WavBytes(const std::string &pcm, int sample_rate) {
  std::string bytes;
  const auto append = [&](uint32_t value, int size) {
    for (int index = 0; index < size; ++index) {
      bytes.push_back(static_cast<char>((value >> (8 * index)) & 0xff));
    }
  };
  const auto data_size = static_cast<uint32_t>(pcm.size());
  bytes += "RIFF";
  append(36 + data_size, 4);
  bytes += "WAVEfmt ";
  append(16, 4);
  append(1, 2);
  append(1, 2);
  append(sample_rate, 4);
  append(sample_rate * 2, 4);
  append(2, 2);
  append(16, 2);
  bytes += "data";
  append(data_size, 4);
  return bytes + pcm;
}

} // namespace fish_s2_pro
