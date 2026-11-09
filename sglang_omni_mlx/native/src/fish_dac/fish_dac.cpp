// SPDX-License-Identifier: Apache-2.0
#include "fish_dac/fish_dac.h"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <set>
#include <stdexcept>
#include <string_view>

#include "common/layers.h"

namespace fish_dac {

namespace mx = mlx::core;

namespace {

constexpr int kEncoderStrides[] = {2, 4, 8, 8};
constexpr int kTransformerHeadCount = 16;
constexpr int kTransformerHeadDim = 64;
constexpr float kTransformerNormEps = 1e-5f;
constexpr int kEncoderWindow = 512;
constexpr float kConvNeXtNormEps = 1e-6f;
constexpr float kMaskedScore = -1e9f;
// The encoder's output and the decoder's input run at 4 latent frames per code
// frame.
constexpr int kLatentFrameSamples = kFrameSamples / 4;
// Note (Jiaxin Deng): the encoder's convolutions reach about 14 latent frames
// back and the decoder's about 19, so each runs in windows with this context:
// a whole long utterance's convolution unfoldings would need tens of GB.
constexpr int kConvolutionWindowFrames = 256;
constexpr int kConvolutionContextFrames = 32;
constexpr const char *kEncoderTransformer = "encoder.block.4.block.5";
constexpr const char *kPreModule = "quantizer.pre_module";
// The weights only Encode reads, and those only decoding reads; the codebooks
// both read are in neither.
constexpr std::string_view kEncodingPrefixes[] = {
    "encoder.", "quantizer.downsample.", "quantizer.pre_module."};
constexpr std::string_view kDecodingPrefixes[] = {
    "decoder.", "quantizer.upsample.", "quantizer.post_module."};
// The (g, v) name suffixes of a weight-normed convolution: parametrized, or
// the older weight_g and weight_v.
constexpr std::pair<std::string_view, std::string_view> kWeightNormSuffixes[] =
    {{".conv.parametrizations.weight.original0",
      ".conv.parametrizations.weight.original1"},
     {".weight_g", ".weight_v"}};

// Rotates the pairs of x [batch, time, heads, dim] by freqs_cis [time, dim / 2,
// 2].
mx::array ApplyRotary(const mx::array &x, const mx::array &freqs_cis) {
  const int batch = x.shape(0);
  const int time = x.shape(1);
  const int heads = x.shape(2);
  const int pairs = x.shape(3) / 2;
  const mx::array pairs_of_x =
      mx::reshape(mx::astype(x, mx::float32), {batch, time, heads, pairs, 2});
  const mx::array freqs = mx::reshape(freqs_cis, {1, time, 1, pairs, 2});
  const auto part = [](const mx::array &array, int index) {
    return mx::squeeze(mx::slice(array, {0, 0, 0, 0, index},
                                 {array.shape(0), array.shape(1),
                                  array.shape(2), array.shape(3), index + 1}),
                       -1);
  };
  const mx::array x0 = part(pairs_of_x, 0);
  const mx::array x1 = part(pairs_of_x, 1);
  const mx::array cosine = part(freqs, 0);
  const mx::array sine = part(freqs, 1);
  const mx::array rotated =
      mx::stack({mx::subtract(mx::multiply(x0, cosine), mx::multiply(x1, sine)),
                 mx::add(mx::multiply(x1, cosine), mx::multiply(x0, sine))},
                -1);
  return mx::astype(mx::reshape(rotated, x.shape()), x.dtype());
}

} // namespace

std::string AudioCodec::Index(const std::string &prefix, int index) {
  return prefix + "." + std::to_string(index);
}

int AudioCodec::LayerCount(const std::string &prefix) const {
  int count = 0;
  while (
      checkpoint_.Has(Index(prefix + ".layers", count) + ".ffn_norm.weight")) {
    ++count;
  }
  return count;
}

// Note (Yitong Li): x + sin(alpha * x)^2 / alpha in the reference's operation
// order (a reciprocal, then a product), which other models' Snake does not
// share, for bit-identical audio.
mx::array AudioCodec::Snake(const mx::array &x, const mx::array &alpha) {
  return mx::add(
      x, mx::multiply(mx::reciprocal(mx::add(
                          alpha, omni_common::Scalar(1e-9, alpha.dtype()))),
                      mx::power(mx::sin(mx::multiply(alpha, x)),
                                omni_common::Scalar(2, x.dtype()))));
}

// The transformers' RMSNorm, computed in float32 and cast back before the
// gain.
mx::array AudioCodec::TransformerNorm(const mx::array &x,
                                      const mx::array &weight) {
  const mx::array x32 = mx::astype(x, mx::float32);
  const mx::array normed = mx::multiply(
      x32, mx::rsqrt(
               mx::add(mx::mean(mx::multiply(x32, x32), -1, true),
                       omni_common::Scalar(kTransformerNormEps, mx::float32))));
  return mx::multiply(mx::astype(normed, x.dtype()), weight);
}

mx::array AudioCodec::Convolve(const mx::array &x,
                               const ConvolutionWeights &weights, int stride,
                               int dilation, int groups) {
  mx::array y = mx::conv1d(mx::swapaxes(x, 1, 2), weights.kernel, stride, 0,
                           dilation, groups);
  if (weights.bias.has_value()) {
    y = mx::add(y, *weights.bias);
  } else {
  }
  return mx::swapaxes(y, 1, 2);
}

mx::array AudioCodec::ConvolveTransposed(const mx::array &x,
                                         const ConvolutionWeights &weights,
                                         int stride) {
  mx::array y =
      mx::conv_transpose1d(mx::swapaxes(x, 1, 2), weights.kernel, stride);
  if (weights.bias.has_value()) {
    y = mx::add(y, *weights.bias);
  } else {
  }
  return mx::swapaxes(y, 1, 2);
}

mx::array AudioCodec::SliceTime(const mx::array &x, int begin, int end) {
  return mx::slice(x, {0, 0, begin}, {x.shape(0), x.shape(1), end});
}

mx::array AudioCodec::WindowMask(int query_begin, int total, int window,
                                 mx::Dtype dtype) {
  const mx::array rows =
      mx::expand_dims(mx::arange(query_begin, total, mx::int32), 1);
  const mx::array columns = mx::expand_dims(mx::arange(total, mx::int32), 0);
  const mx::array allowed = mx::logical_and(
      mx::less_equal(columns, rows),
      mx::greater(columns, mx::subtract(rows, mx::array(window, mx::int32))));
  return mx::expand_dims(
      mx::astype(mx::where(allowed, mx::array(0.0f), mx::array(kMaskedScore)),
                 dtype),
      {0, 1});
}

// The codec's weights, each convolution's kernel stored once as
// prefix.kernel: weight norm folded as g * v / |v| over all but the first
// axis, moved from PyTorch to MLX layout.
// Note (Yitong Li): the fold keeps the reference's operation order, which
// Voxtral's differs from, for bit-identical audio.
omni_common::WeightMap
AudioCodec::LoadWeights(const std::filesystem::path &model_directory,
                        bool for_encoding) {
  auto [loaded, metadata] =
      mx::load_safetensors((model_directory / "codec.safetensors").string());
  const auto &skipped_prefixes =
      for_encoding ? kDecodingPrefixes : kEncodingPrefixes;
  const std::string_view plain_kernel = ".conv.weight";
  std::set<std::string> transposed_prefixes;
  for (int block = 0; block < kDownsampleCount; ++block) {
    transposed_prefixes.insert(Index("quantizer.upsample", block) + ".0");
  }
  for (int block = 0; block < 4; ++block) {
    transposed_prefixes.insert(Index("decoder.model", block + 1) + ".block.1");
  }
  // Transposed convolutions store [in, out, kernel]; the rest [out, in,
  // kernel].
  const auto to_mlx_layout = [&transposed_prefixes](const std::string &prefix,
                                                    const mx::array &kernel) {
    return transposed_prefixes.count(prefix) ? mx::transpose(kernel, {1, 2, 0})
                                             : mx::transpose(kernel, {0, 2, 1});
  };
  omni_common::WeightMap weights;
  for (auto &[name, array] : loaded) {
    const auto weight_norm = std::find_if(
        std::begin(kWeightNormSuffixes), std::end(kWeightNormSuffixes),
        [&name](const auto &suffixes) {
          return name.ends_with(suffixes.first) ||
                 name.ends_with(suffixes.second);
        });
    if (name.ends_with(".causal_mask") ||
        std::any_of(std::begin(skipped_prefixes), std::end(skipped_prefixes),
                    [&name](std::string_view prefix) {
                      return name.starts_with(prefix);
                    }) ||
        (weight_norm != std::end(kWeightNormSuffixes) &&
         name.ends_with(weight_norm->first))) {
      continue;
    } else if (weight_norm != std::end(kWeightNormSuffixes)) {
      const std::string prefix =
          name.substr(0, name.size() - weight_norm->second.size());
      const mx::array &g = loaded.at(prefix + std::string(weight_norm->first));
      const mx::array kernel = mx::divide(
          mx::multiply(g, array),
          mx::sqrt(
              mx::sum(mx::power(array, omni_common::Scalar(2, array.dtype())),
                      {1, 2}, true)));
      weights.insert_or_assign(prefix + ".kernel",
                               to_mlx_layout(prefix, kernel));
    } else if (name.ends_with(plain_kernel) && array.ndim() == 3) {
      const std::string prefix =
          name.substr(0, name.size() - plain_kernel.size());
      weights.insert_or_assign(prefix + ".kernel",
                               to_mlx_layout(prefix, array));
    } else {
      weights.insert_or_assign(name, array);
    }
  }
  return weights;
}

AudioCodec::AudioCodec(const std::filesystem::path &model_directory)
    : AudioCodec(model_directory, LoadWeights(model_directory, false)) {}

AudioCodec::AudioCodec(std::filesystem::path model_directory,
                       omni_common::WeightMap weights)
    : model_directory_(std::move(model_directory)),
      checkpoint_(weights, nlohmann::json::object()) {
  const std::string_view kernel_suffix = ".kernel";
  for (const auto &[name, kernel] : weights) {
    if (name.ends_with(kernel_suffix)) {
      const std::string prefix =
          name.substr(0, name.size() - kernel_suffix.size());
      const std::string bias_name = checkpoint_.Has(prefix + ".bias")
                                        ? prefix + ".bias"
                                        : prefix + ".conv.bias";
      std::optional<mx::array> bias = std::nullopt;
      if (checkpoint_.Has(bias_name)) {
        bias = checkpoint_.Weight(bias_name);
      } else {
      }
      convolutions_.insert_or_assign(prefix, ConvolutionWeights{kernel, bias});
    } else {
    }
  }
}

const ConvolutionWeights &
AudioCodec::Convolution(const std::string &prefix) const {
  return convolutions_.at(prefix);
}

mx::array AudioCodec::CausalConvolve(const mx::array &x,
                                     const std::string &prefix, int kernel_size,
                                     int stride, int dilation,
                                     int groups) const {
  const int effective_kernel = (kernel_size - 1) * dilation + 1;
  const int padding = effective_kernel - stride;
  const int length = x.shape(2);
  const double frame_count =
      static_cast<double>(length - effective_kernel + padding) / stride + 1.0;
  const int ideal_length =
      (static_cast<int>(std::ceil(frame_count)) - 1) * stride +
      (effective_kernel - padding);
  const int extra = std::max(0, ideal_length - length);
  return Convolve(mx::pad(x, {{0, 0}, {0, 0}, {padding, extra}}),
                  Convolution(prefix), stride, dilation, groups);
}

mx::array AudioCodec::ResidualUnit(const mx::array &x,
                                   const std::string &prefix,
                                   int dilation) const {
  mx::array y = Snake(x, checkpoint_.Weight(prefix + ".block.0.alpha"));
  y = CausalConvolve(y, prefix + ".block.1", 7, 1, dilation, 1);
  y = Snake(y, checkpoint_.Weight(prefix + ".block.2.alpha"));
  y = CausalConvolve(y, prefix + ".block.3", 1, 1, 1, 1);
  return mx::add(x, y);
}

mx::array AudioCodec::ConvNeXtPointwise(const mx::array &input,
                                        const mx::array &convolved,
                                        const std::string &prefix) const {
  mx::array x = mx::swapaxes(convolved, 1, 2);
  x = mx::fast::layer_norm(x, checkpoint_.Weight(prefix + ".norm.weight"),
                           checkpoint_.Weight(prefix + ".norm.bias"),
                           kConvNeXtNormEps);
  x = omni_common::Gelu(checkpoint_.Linear(x, prefix + ".pwconv1"));
  x = checkpoint_.Linear(x, prefix + ".pwconv2");
  x = mx::multiply(checkpoint_.Weight(prefix + ".gamma"), x);
  return mx::add(input, mx::swapaxes(x, 1, 2));
}

mx::array AudioCodec::TransformerLayer(
    const mx::array &x, const std::string &prefix, const mx::array &freqs_cis,
    const mx::array &mask,
    std::optional<std::pair<mx::array, mx::array>> &cache) const {
  const int batch = x.shape(0);
  const int time = x.shape(1);
  const int width = kTransformerHeadCount * kTransformerHeadDim;
  const mx::array qkv = mx::matmul(
      TransformerNorm(x, checkpoint_.Weight(prefix + ".attention_norm.weight")),
      mx::transpose(checkpoint_.Weight(prefix + ".attention.wqkv.weight")));
  const auto head_split = [&](int index) {
    return mx::reshape(
        mx::slice(qkv, {0, 0, index * width},
                  {batch, time, (index + 1) * width}),
        {batch, time, kTransformerHeadCount, kTransformerHeadDim});
  };
  mx::array queries = head_split(0);
  mx::array keys = head_split(1);
  mx::array values = head_split(2);
  if (cache.has_value()) {
    keys = mx::concatenate({cache->first, keys}, 1);
    values = mx::concatenate({cache->second, values}, 1);
  } else {
  }
  cache = std::make_pair(keys, values);
  const int total = keys.shape(1);
  queries = ApplyRotary(
      queries, mx::slice(freqs_cis, {total - time, 0, 0}, freqs_cis.shape()));
  keys = ApplyRotary(keys, freqs_cis);
  const mx::array attended = mx::fast::scaled_dot_product_attention(
      mx::transpose(queries, {0, 2, 1, 3}), mx::transpose(keys, {0, 2, 1, 3}),
      mx::transpose(values, {0, 2, 1, 3}),
      static_cast<float>(1.0 / std::sqrt(kTransformerHeadDim)), "", mask);
  const mx::array attention = mx::matmul(
      mx::reshape(mx::transpose(attended, {0, 2, 1, 3}), {batch, time, width}),
      mx::transpose(checkpoint_.Weight(prefix + ".attention.wo.weight")));
  const mx::array hidden = mx::add(
      x, mx::multiply(attention, checkpoint_.Weight(
                                     prefix + ".attention_layer_scale.gamma")));
  const mx::array normed =
      TransformerNorm(hidden, checkpoint_.Weight(prefix + ".ffn_norm.weight"));
  const mx::array gated = mx::multiply(
      omni_common::Silu(
          mx::matmul(normed, mx::transpose(checkpoint_.Weight(
                                 prefix + ".feed_forward.w1.weight")))),
      mx::matmul(normed, mx::transpose(checkpoint_.Weight(
                             prefix + ".feed_forward.w3.weight"))));
  const mx::array feed_forward = mx::matmul(
      gated,
      mx::transpose(checkpoint_.Weight(prefix + ".feed_forward.w2.weight")));
  return mx::add(hidden, mx::multiply(feed_forward,
                                      checkpoint_.Weight(
                                          prefix + ".ffn_layer_scale.gamma")));
}

mx::array AudioCodec::Transformer(const mx::array &x, const std::string &prefix,
                                  int window) const {
  mx::array hidden = mx::swapaxes(x, 1, 2);
  const int time = hidden.shape(1);
  const mx::array freqs_cis =
      mx::slice(checkpoint_.Weight(prefix + ".freqs_cis"), {0, 0, 0},
                {time, kTransformerHeadDim / 2, 2});
  const mx::array mask = WindowMask(0, time, window, hidden.dtype());
  for (int layer = 0; layer < LayerCount(prefix); ++layer) {
    std::optional<std::pair<mx::array, mx::array>> no_cache;
    hidden = TransformerLayer(hidden, Index(prefix + ".layers", layer),
                              freqs_cis, mask, no_cache);
  }
  hidden = TransformerNorm(hidden, checkpoint_.Weight(prefix + ".norm.weight"));
  return mx::swapaxes(hidden, 1, 2);
}

std::pair<mx::array, mx::array>
AudioCodec::Quantize(const mx::array &latents,
                     const std::string &prefix) const {
  const mx::array projected =
      Convolve(latents, Convolution(prefix + ".in_proj"), 1, 1, 1);
  const int batch = projected.shape(0);
  const int dim = projected.shape(1);
  const int time = projected.shape(2);
  const mx::array encodings =
      mx::reshape(mx::transpose(projected, {0, 2, 1}), {batch * time, dim});
  const auto unit = [](const mx::array &rows) {
    return mx::divide(
        rows,
        mx::maximum(mx::sqrt(mx::sum(
                        mx::power(rows, omni_common::Scalar(2, rows.dtype())),
                        1, true)),
                    omni_common::Scalar(1e-12, rows.dtype())));
  };
  const mx::array encodings_unit = unit(encodings);
  const mx::array codebook_unit =
      unit(checkpoint_.Weight(prefix + ".codebook.weight"));
  const auto squared_sum = [](const mx::array &rows) {
    return mx::sum(mx::power(rows, omni_common::Scalar(2, rows.dtype())), 1,
                   true);
  };
  const mx::array distances = mx::add(
      mx::subtract(squared_sum(encodings_unit),
                   mx::multiply(omni_common::Scalar(2, mx::float32),
                                mx::matmul(encodings_unit,
                                           mx::transpose(codebook_unit)))),
      mx::transpose(squared_sum(codebook_unit)));
  const mx::array codes =
      mx::reshape(mx::argmax(mx::negative(distances), 1), {batch, time});
  const mx::array decoded = DecodeCode(codes, prefix);
  const mx::array straight_through =
      mx::add(projected, mx::subtract(decoded, projected));
  return {codes, Convolve(straight_through, Convolution(prefix + ".out_proj"),
                          1, 1, 1)};
}

mx::array AudioCodec::DecodeCode(const mx::array &codes,
                                 const std::string &prefix) const {
  return mx::transpose(
      mx::take(checkpoint_.Weight(prefix + ".codebook.weight"), codes, 0),
      {0, 2, 1});
}

mx::array AudioCodec::Encode(const std::vector<float> &samples) const {
  const mx::array codes =
      AudioCodec(model_directory_, LoadWeights(model_directory_, true))
          .EncodeLoaded(samples);
  // Note (Minyi Tang): the freed encoding weights would otherwise stay in
  // MLX's buffer cache.
  mx::clear_cache();
  return codes;
}

mx::array AudioCodec::EncodeLoaded(const std::vector<float> &samples) const {
  const int length = static_cast<int>(samples.size());
  const int padded_length =
      (length + kFrameSamples - 1) / kFrameSamples * kFrameSamples;
  const int frame_count = padded_length / kFrameSamples;
  const int encoder_positions =
      checkpoint_.Weight(std::string(kEncoderTransformer) + ".freqs_cis")
          .shape(0);
  const int quantizer_positions =
      checkpoint_.Weight(std::string(kPreModule) + ".freqs_cis").shape(0);
  if (frame_count == 0) {
    throw std::invalid_argument("reference audio is empty");
  } else if (frame_count * 4 > encoder_positions ||
             frame_count > quantizer_positions) {
    throw std::invalid_argument("reference audio is too long");
  } else {
  }
  const mx::array audio =
      mx::pad(mx::array(samples.data(), {1, 1, length}, mx::float32),
              {{0, 0}, {0, 0}, {0, padded_length - length}});

  const int latent_frames = padded_length / kLatentFrameSamples;
  std::vector<mx::array> windows;
  for (int start = 0; start < latent_frames;
       start += kConvolutionWindowFrames) {
    const int context = std::min(start, kConvolutionContextFrames);
    const int stop = std::min(start + kConvolutionWindowFrames, latent_frames);
    mx::array y =
        CausalConvolve(SliceTime(audio, (start - context) * kLatentFrameSamples,
                                 stop * kLatentFrameSamples),
                       "encoder.block.0", 7, 1, 1, 1);
    for (int block = 0; block < 4; ++block) {
      const std::string prefix = Index("encoder.block", block + 1) + ".block";
      const int stride = kEncoderStrides[block];
      for (int unit = 0; unit < 3; ++unit) {
        y = ResidualUnit(y, Index(prefix, unit), kResidualDilations[unit]);
      }
      y = Snake(y, checkpoint_.Weight(Index(prefix, 3) + ".alpha"));
      y = CausalConvolve(y, Index(prefix, 4), 2 * stride, stride, 1, 1);
    }
    windows.push_back(SliceTime(y, context, y.shape(2)));
    mx::eval(windows.back());
  }
  mx::array x = Transformer(mx::concatenate(windows, 2), kEncoderTransformer,
                            kEncoderWindow);
  x = Snake(x, checkpoint_.Weight("encoder.block.5.alpha"));
  x = CausalConvolve(x, "encoder.block.6", 3, 1, 1, 1);

  for (int block = 0; block < kDownsampleCount; ++block) {
    const std::string prefix = Index("quantizer.downsample", block);
    x = CausalConvolve(x, prefix + ".0", 2, 2, 1, 1);
    const int dim = x.shape(1);
    x = ConvNeXtPointwise(x,
                          CausalConvolve(x, prefix + ".1.dwconv", 7, 1, 1, dim),
                          prefix + ".1");
  }
  x = Transformer(x, kPreModule, kQuantizerWindow);

  std::vector<mx::array> codes;
  auto [semantic_codes, semantic_quantized] =
      Quantize(x, "quantizer.semantic_quantizer.quantizers.0");
  codes.push_back(semantic_codes);
  mx::array residual =
      mx::subtract(x, mx::add(omni_common::Scalar(0, semantic_quantized),
                              semantic_quantized));
  for (int codebook = 0; codebook < kResidualCodebookCount; ++codebook) {
    auto [residual_codes, quantized] =
        Quantize(residual, Index("quantizer.quantizer.quantizers", codebook));
    codes.push_back(residual_codes);
    residual = mx::subtract(residual, quantized);
  }
  mx::array stacked =
      mx::astype(mx::squeeze(mx::stack(codes, 1), 0), mx::int32);
  stacked = mx::slice(stacked, {0, 0}, {stacked.shape(0), frame_count});
  mx::eval(stacked);
  return stacked;
}

std::vector<float> CodecStream::Decode(
    const std::vector<mx::array> &codes, size_t begin, size_t end,
    bool limit_idle_cache, bool synchronize,
    const std::function<void(const char *, size_t, size_t)> &profile) {
  constexpr size_t kDecodeFrames = 64;
  constexpr size_t kIdleCacheBytes = 256 * 1024 * 1024;
  std::optional<size_t> previous_cache_limit;
  if (limit_idle_cache) {
    previous_cache_limit = mx::set_cache_limit(kIdleCacheBytes);
    if (*previous_cache_limit < kIdleCacheBytes) {
      mx::set_cache_limit(*previous_cache_limit);
    }
  }
  std::vector<float> audio;
  try {
    if (limit_idle_cache) {
      mx::synchronize();
      mx::clear_cache();
    }
    for (size_t piece = begin; piece < end; piece += kDecodeFrames) {
      const size_t piece_end = std::min(end, piece + kDecodeFrames);
      const std::vector<mx::array> frames(codes.begin() + piece,
                                          codes.begin() + piece_end);
      profile("codec_before_push", piece, piece_end);
      const mx::array samples =
          mx::astype(Push(mx::stack(frames, 1)), mx::float32);
      mx::eval(samples);
      if (synchronize) {
        mx::synchronize();
      }
      profile("codec_after_eval", piece, piece_end);
      audio.insert(audio.end(), samples.data<float>(),
                   samples.data<float>() + samples.size());
      profile("codec_after_host_copy", piece, piece_end);
    }
  } catch (...) {
    if (previous_cache_limit.has_value()) {
      mx::set_cache_limit(*previous_cache_limit);
    }
    throw;
  }
  if (previous_cache_limit.has_value()) {
    mx::set_cache_limit(*previous_cache_limit);
  }
  return audio;
}

mx::array AudioCodec::QuantizedFeatures(const mx::array &codes) const {
  const int frames = codes.shape(1);
  const mx::array indices = mx::expand_dims(codes, 0);
  const int semantic_size =
      checkpoint_
          .Weight("quantizer.semantic_quantizer.quantizers.0.codebook.weight")
          .shape(0);
  const int residual_size =
      checkpoint_.Weight("quantizer.quantizer.quantizers.0.codebook.weight")
          .shape(0);
  const auto codebook_codes = [&](int codebook) {
    return mx::squeeze(
        mx::slice(indices, {0, codebook, 0}, {1, codebook + 1, frames}), 1);
  };
  const auto from_codes = [&](const mx::array &codebook_indices,
                              const std::string &prefix) {
    return Convolve(DecodeCode(codebook_indices, prefix),
                    Convolution(prefix + ".out_proj"), 1, 1, 1);
  };
  // A Python 0.0 that starts each sum takes the projection's dtype.
  const mx::array semantic_projection =
      from_codes(mx::clip(codebook_codes(0), mx::array(0, mx::int32),
                          mx::array(semantic_size - 1, mx::int32)),
                 "quantizer.semantic_quantizer.quantizers.0");
  const mx::array semantic =
      mx::add(omni_common::Scalar(0, semantic_projection), semantic_projection);
  std::optional<mx::array> residual;
  for (int codebook = 0; codebook < kResidualCodebookCount; ++codebook) {
    const mx::array projection = from_codes(
        mx::clip(codebook_codes(codebook + 1), mx::array(0, mx::int32),
                 mx::array(residual_size - 1, mx::int32)),
        Index("quantizer.quantizer.quantizers", codebook));
    residual = mx::add(
        residual.has_value() ? *residual : omni_common::Scalar(0, projection),
        projection);
  }
  return mx::add(semantic, *residual);
}

mx::array AudioCodec::Decode(const mx::array &codes) const {
  const auto upsample = [&](const mx::array &x, const std::string &prefix,
                            int stride) {
    return SliceTime(ConvolveTransposed(x, Convolution(prefix), stride), 0,
                     x.shape(2) * stride);
  };
  mx::array x =
      Transformer(QuantizedFeatures(codes), kPostModule, kQuantizerWindow);
  for (int block = 0; block < kDownsampleCount; ++block) {
    const std::string prefix = Index("quantizer.upsample", block);
    x = upsample(x, prefix + ".0", 2);
    const int dim = x.shape(1);
    x = ConvNeXtPointwise(x,
                          CausalConvolve(x, prefix + ".1.dwconv", 7, 1, 1, dim),
                          prefix + ".1");
  }
  mx::eval(x);

  const int latent_frames = x.shape(2);
  std::vector<mx::array> windows;
  for (int start = 0; start < latent_frames;
       start += kConvolutionWindowFrames) {
    const int context = std::min(start, kConvolutionContextFrames);
    const int stop = std::min(start + kConvolutionWindowFrames, latent_frames);
    mx::array y = CausalConvolve(SliceTime(x, start - context, stop),
                                 "decoder.model.0", 7, 1, 1, 1);
    for (int block = 0; block < 4; ++block) {
      const std::string prefix = Index("decoder.model", block + 1) + ".block";
      y = Snake(y, checkpoint_.Weight(Index(prefix, 0) + ".alpha"));
      y = upsample(y, Index(prefix, 1), kDecoderStrides[block]);
      for (int unit = 0; unit < 3; ++unit) {
        y = ResidualUnit(y, Index(prefix, unit + 2), kResidualDilations[unit]);
      }
    }
    y = Snake(y, checkpoint_.Weight("decoder.model.5.alpha"));
    y = mx::tanh(CausalConvolve(y, "decoder.model.6", 7, 1, 1, 1));
    windows.push_back(mx::slice(mx::reshape(y, {-1}),
                                {context * kLatentFrameSamples},
                                {static_cast<int>(y.size())}));
    mx::eval(windows.back());
  }
  return mx::concatenate(windows, 0);
}

} // namespace fish_dac
