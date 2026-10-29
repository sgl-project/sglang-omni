// SPDX-License-Identifier: Apache-2.0
#include <algorithm>
#include <stdexcept>

#include "fishaudio_drama3/fishaudio_drama3_codec.h"

namespace fishaudio_drama3 {

namespace mx = mlx::core;

mx::array CodecStream::CausalStep(const mx::array &x, const std::string &prefix,
                                  int kernel_size, int dilation, int groups) {
  const int padding = (kernel_size - 1) * dilation;
  mx::array input = x;
  if (padding > 0) {
    const auto found = left_contexts_.find(prefix);
    input = found == left_contexts_.end()
                ? mx::pad(x, {{0, 0}, {0, 0}, {padding, 0}})
                : mx::concatenate({found->second, x}, -1);
    left_contexts_.insert_or_assign(
        prefix,
        AudioCodec::SliceTime(input, input.shape(2) - padding, input.shape(2)));
  } else {
  }
  return AudioCodec::Convolve(input, codec_.Convolution(prefix), 1, dilation,
                              groups);
}

mx::array CodecStream::TransposedStep(const mx::array &x,
                                      const std::string &prefix, int stride) {
  const ConvolutionWeights &weights = codec_.Convolution(prefix);
  mx::array y = AudioCodec::ConvolveTransposed(x, weights, stride);
  const int valid = x.shape(2) * stride;
  const auto found = overlaps_.find(prefix);
  if (found != overlaps_.end()) {
    mx::array carry = found->second;
    if (weights.bias.has_value()) {
      // Both halves include the bias; it is kept once.
      carry = mx::subtract(carry, mx::expand_dims(*weights.bias, 1));
    } else {
    }
    const int carried = carry.shape(2);
    y = mx::concatenate({mx::add(AudioCodec::SliceTime(y, 0, carried), carry),
                         AudioCodec::SliceTime(y, carried, y.shape(2))},
                        -1);
    overlaps_.erase(found);
  } else {
  }
  if (y.shape(2) > valid) {
    overlaps_.insert_or_assign(prefix,
                               AudioCodec::SliceTime(y, valid, y.shape(2)));
  } else {
  }
  return AudioCodec::SliceTime(y, 0, valid);
}

mx::array CodecStream::ResidualUnitStep(const mx::array &x,
                                        const std::string &prefix,
                                        int dilation) {
  mx::array y = AudioCodec::Snake(
      x, codec_.checkpoint_.Weight(prefix + ".block.0.alpha"));
  y = CausalStep(y, prefix + ".block.1", 7, dilation, 1);
  y = AudioCodec::Snake(y,
                        codec_.checkpoint_.Weight(prefix + ".block.2.alpha"));
  y = CausalStep(y, prefix + ".block.3", 1, 1, 1);
  return mx::add(x, y);
}

mx::array CodecStream::TransformerStep(const mx::array &x,
                                       const std::string &prefix) {
  mx::array hidden = mx::swapaxes(x, 1, 2);
  const int layer_count = codec_.LayerCount(prefix);
  if (transformer_caches_.empty()) {
    transformer_caches_.resize(layer_count);
  } else {
  }
  const int cached = transformer_caches_[0].has_value()
                         ? transformer_caches_[0]->first.shape(1)
                         : 0;
  const int time = hidden.shape(1);
  const int total = cached + time;
  const mx::array &table = codec_.checkpoint_.Weight(prefix + ".freqs_cis");
  // Note (Yitong Li): the same absolute rotary rows as a whole decode while
  // the stream fits the table, then rebased onto the cached window.
  const int start =
      std::min(transformer_position_ - cached, table.shape(0) - total);
  if (start < 0) {
    throw std::invalid_argument("codec streaming chunk exceeds the table");
  } else {
  }
  transformer_position_ += time;
  const mx::array mask = AudioCodec::WindowMask(
      cached, total, AudioCodec::kQuantizerWindow, hidden.dtype());
  const mx::array freqs_cis = mx::slice(
      table, {start, 0, 0}, {start + total, table.shape(1), table.shape(2)});
  for (int layer = 0; layer < layer_count; ++layer) {
    hidden = codec_.TransformerLayer(
        hidden, AudioCodec::Index(prefix + ".layers", layer), freqs_cis, mask,
        transformer_caches_[layer]);
    auto &[keys, values] = *transformer_caches_[layer];
    const int keep = AudioCodec::kQuantizerWindow - 1;
    const int begin = std::max(0, keys.shape(1) - keep);
    keys = mx::slice(keys, {0, begin, 0, 0}, keys.shape());
    values = mx::slice(values, {0, begin, 0, 0}, values.shape());
  }
  hidden = AudioCodec::TransformerNorm(
      hidden, codec_.checkpoint_.Weight(prefix + ".norm.weight"));
  return mx::swapaxes(hidden, 1, 2);
}

mx::array CodecStream::Push(const mx::array &codes) {
  const int frames = codes.shape(1);
  const mx::array indices = mx::expand_dims(codes, 0);
  const int semantic_size =
      codec_.checkpoint_
          .Weight("quantizer.semantic_quantizer.quantizers.0.codebook.weight")
          .shape(0);
  const int residual_size =
      codec_.checkpoint_
          .Weight("quantizer.quantizer.quantizers.0.codebook.weight")
          .shape(0);
  const auto codebook_codes = [&](int codebook) {
    return mx::squeeze(
        mx::slice(indices, {0, codebook, 0}, {1, codebook + 1, frames}), 1);
  };
  const auto from_codes = [&](const mx::array &codebook_indices,
                              const std::string &prefix) {
    return AudioCodec::Convolve(codec_.DecodeCode(codebook_indices, prefix),
                                codec_.Convolution(prefix + ".out_proj"), 1, 1,
                                1);
  };
  // A Python 0.0 that starts each sum takes the projection's dtype.
  const mx::array semantic_projection =
      from_codes(mx::clip(codebook_codes(0), mx::array(0, mx::int32),
                          mx::array(semantic_size - 1, mx::int32)),
                 "quantizer.semantic_quantizer.quantizers.0");
  const mx::array semantic =
      mx::add(omni_common::Scalar(0, semantic_projection), semantic_projection);
  std::optional<mx::array> residual;
  for (int codebook = 0; codebook < AudioCodec::kResidualCodebookCount;
       ++codebook) {
    const mx::array projection = from_codes(
        mx::clip(codebook_codes(codebook + 1), mx::array(0, mx::int32),
                 mx::array(residual_size - 1, mx::int32)),
        AudioCodec::Index("quantizer.quantizer.quantizers", codebook));
    residual = mx::add(
        residual.has_value() ? *residual : omni_common::Scalar(0, projection),
        projection);
  }
  mx::array x =
      TransformerStep(mx::add(semantic, *residual), AudioCodec::kPostModule);
  for (int block = 0; block < AudioCodec::kDownsampleCount; ++block) {
    const std::string prefix = AudioCodec::Index("quantizer.upsample", block);
    x = TransposedStep(x, prefix + ".0", 2);
    const int dim = x.shape(1);
    x = codec_.ConvNeXtPointwise(
        x, CausalStep(x, prefix + ".1.dwconv", 7, 1, dim), prefix + ".1");
  }

  x = CausalStep(x, "decoder.model.0", 7, 1, 1);
  for (int block = 0; block < 4; ++block) {
    const std::string prefix =
        AudioCodec::Index("decoder.model", block + 1) + ".block";
    x = AudioCodec::Snake(
        x, codec_.checkpoint_.Weight(AudioCodec::Index(prefix, 0) + ".alpha"));
    x = TransposedStep(x, AudioCodec::Index(prefix, 1),
                       AudioCodec::kDecoderStrides[block]);
    for (int unit = 0; unit < 3; ++unit) {
      x = ResidualUnitStep(x, AudioCodec::Index(prefix, unit + 2),
                           AudioCodec::kResidualDilations[unit]);
    }
  }
  x = AudioCodec::Snake(x, codec_.checkpoint_.Weight("decoder.model.5.alpha"));
  x = mx::tanh(CausalStep(x, "decoder.model.6", 7, 1, 1));
  return mx::reshape(x, {frames * kCodecFrameSamples});
}

} // namespace fishaudio_drama3
