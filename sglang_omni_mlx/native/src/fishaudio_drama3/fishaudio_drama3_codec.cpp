// SPDX-License-Identifier: Apache-2.0
#include "fishaudio_drama3/fishaudio_drama3_codec.h"

#include <algorithm>
#include <cmath>
#include <set>
#include <stdexcept>
#include <string_view>

#include "common/layers.h"

namespace fishaudio_drama3 {

namespace mx = mlx::core;

namespace {

constexpr int kEncoderStrides[] = {2, 4, 8, 8};
constexpr int kTransformerHeadCount = 16;
constexpr int kTransformerHeadDim = 64;
constexpr float kTransformerNormEps = 1e-5f;
constexpr int kEncoderWindow = 512;
constexpr float kConvNeXtNormEps = 1e-6f;
constexpr float kMaskedScore = -1e9f;
constexpr const char *kEncoderTransformer = "encoder.block.4.block.5";
constexpr const char *kPreModule = "quantizer.pre_module";
constexpr const char *kWeightNormG = ".conv.parametrizations.weight.original0";
constexpr const char *kWeightNormV = ".conv.parametrizations.weight.original1";

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

omni_common::WeightMap
LoadCodecWeights(const std::filesystem::path &model_directory) {
  auto [loaded, metadata] =
      mx::load_safetensors((model_directory / "codec.safetensors").string());
  omni_common::WeightMap weights;
  for (auto &[name, array] : loaded) {
    const std::string_view weight_norm_g = kWeightNormG;
    const std::string_view weight_norm_v = kWeightNormV;
    if (name.ends_with(".causal_mask")) {
      continue;
    } else if (name.ends_with(weight_norm_g)) {
      weights.insert_or_assign(
          name.substr(0, name.size() - weight_norm_g.size()) + ".weight_g",
          array);
    } else if (name.ends_with(weight_norm_v)) {
      weights.insert_or_assign(
          name.substr(0, name.size() - weight_norm_v.size()) + ".weight_v",
          array);
    } else {
      weights.insert_or_assign(name, array);
    }
  }
  return weights;
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

AudioCodec::AudioCodec(const std::filesystem::path &model_directory)
    : checkpoint_(LoadCodecWeights(model_directory), nlohmann::json::object()) {
  std::set<std::string> transposed_prefixes;
  for (int block = 0; block < kDownsampleCount; ++block) {
    transposed_prefixes.insert(Index("quantizer.upsample", block) + ".0");
  }
  for (int block = 0; block < 4; ++block) {
    transposed_prefixes.insert(Index("decoder.model", block + 1) + ".block.1");
  }
  // Every convolution's kernel, weight norm folded as g * v / |v| over all
  // but the first axis, moved from PyTorch to MLX layout.
  // Note (Yitong Li): the fold keeps the reference's operation order, which
  // Voxtral's differs from, for bit-identical audio.
  std::vector<mx::array> kernels;
  for (const auto &[name, array] : checkpoint_.weights()) {
    std::string prefix;
    bool is_weight_norm = false;
    if (name.ends_with(".weight_v")) {
      prefix = name.substr(0, name.size() - std::string(".weight_v").size());
      is_weight_norm = true;
    } else if (name.ends_with(".conv.weight") && array.ndim() == 3) {
      prefix = name.substr(0, name.size() - std::string(".conv.weight").size());
    } else {
      continue;
    }
    mx::array kernel = array;
    if (is_weight_norm) {
      const mx::array &g = checkpoint_.Weight(prefix + ".weight_g");
      kernel = mx::divide(
          mx::multiply(g, array),
          mx::sqrt(
              mx::sum(mx::power(array, omni_common::Scalar(2, array.dtype())),
                      {1, 2}, true)));
    } else {
    }
    // Transposed convolutions store [in, out, kernel]; the rest [out, in,
    // kernel].
    kernel = transposed_prefixes.count(prefix)
                 ? mx::transpose(kernel, {1, 2, 0})
                 : mx::transpose(kernel, {0, 2, 1});
    const std::string bias_name = checkpoint_.Has(prefix + ".bias")
                                      ? prefix + ".bias"
                                      : prefix + ".conv.bias";
    std::optional<mx::array> bias = std::nullopt;
    if (checkpoint_.Has(bias_name)) {
      bias = checkpoint_.Weight(bias_name);
    } else {
    }
    convolutions_.insert_or_assign(prefix, ConvolutionWeights{kernel, bias});
    kernels.push_back(kernel);
  }
  mx::eval(kernels);
}

const ConvolutionWeights &
AudioCodec::Convolution(const std::string &prefix) const {
  const auto found = convolutions_.find(prefix);
  if (found == convolutions_.end()) {
    throw std::runtime_error("codec checkpoint is missing " + prefix);
  } else {
  }
  return found->second;
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
  const int length = static_cast<int>(samples.size());
  const int padded_length = (length + kCodecFrameSamples - 1) /
                            kCodecFrameSamples * kCodecFrameSamples;
  const int frame_count = padded_length / kCodecFrameSamples;
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
  mx::array x = mx::pad(mx::array(samples.data(), {1, 1, length}, mx::float32),
                        {{0, 0}, {0, 0}, {0, padded_length - length}});

  x = CausalConvolve(x, "encoder.block.0", 7, 1, 1, 1);
  for (int block = 0; block < 4; ++block) {
    const std::string prefix = Index("encoder.block", block + 1) + ".block";
    const int stride = kEncoderStrides[block];
    for (int unit = 0; unit < 3; ++unit) {
      x = ResidualUnit(x, Index(prefix, unit), kResidualDilations[unit]);
    }
    x = Snake(x, checkpoint_.Weight(Index(prefix, 3) + ".alpha"));
    x = CausalConvolve(x, Index(prefix, 4), 2 * stride, stride, 1, 1);
    if (Index(prefix, 5) == kEncoderTransformer) {
      mx::array hidden = mx::swapaxes(x, 1, 2);
      const int time = hidden.shape(1);
      const mx::array freqs_cis =
          mx::slice(checkpoint_.Weight(Index(prefix, 5) + ".freqs_cis"),
                    {0, 0, 0}, {time, kTransformerHeadDim / 2, 2});
      const mx::array mask =
          WindowMask(0, time, kEncoderWindow, hidden.dtype());
      for (int layer = 0; layer < LayerCount(Index(prefix, 5)); ++layer) {
        std::optional<std::pair<mx::array, mx::array>> no_cache;
        hidden =
            TransformerLayer(hidden, Index(Index(prefix, 5) + ".layers", layer),
                             freqs_cis, mask, no_cache);
      }
      hidden = TransformerNorm(
          hidden, checkpoint_.Weight(Index(prefix, 5) + ".norm.weight"));
      x = mx::swapaxes(hidden, 1, 2);
    } else {
    }
  }
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
  {
    mx::array hidden = mx::swapaxes(x, 1, 2);
    const int time = hidden.shape(1);
    const mx::array freqs_cis =
        mx::slice(checkpoint_.Weight(std::string(kPreModule) + ".freqs_cis"),
                  {0, 0, 0}, {time, kTransformerHeadDim / 2, 2});
    const mx::array mask =
        WindowMask(0, time, kQuantizerWindow, hidden.dtype());
    for (int layer = 0; layer < LayerCount(kPreModule); ++layer) {
      std::optional<std::pair<mx::array, mx::array>> no_cache;
      hidden = TransformerLayer(
          hidden, Index(std::string(kPreModule) + ".layers", layer), freqs_cis,
          mask, no_cache);
    }
    hidden = TransformerNorm(
        hidden, checkpoint_.Weight(std::string(kPreModule) + ".norm.weight"));
    x = mx::swapaxes(hidden, 1, 2);
  }

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

} // namespace fishaudio_drama3
