// SPDX-License-Identifier: Apache-2.0
#include "qwen3_asr/kernels/kernels.h"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>

#include "common/layers.h"
#include "mlx/primitives.h"
#include "qwen3_asr/kernels/kernel_library.h"

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

// The head width of the decode attention kernel, whose threadgroups take one
// query head each, of the two per KV head, with kAttentionSimdgroups
// simdgroups.
constexpr int kHeadDim = 128;
constexpr int kAttentionGqa = 2;
constexpr int kAttentionSimdgroups = 32;
// conv_gelu_nax's tile: output pixels, output channels and input channels per
// step, and its simdgroups.
constexpr int kConvolutionTilePixels = 64;
constexpr int kConvolutionTileChannels = 96;
constexpr int kConvolutionTileDepth = 32;
constexpr int kConvolutionSimdgroups = 4;
// conv_gelu_single_channel's output channels and pixels per thread, and
// threads per threadgroup.
constexpr int kSingleChannelOutputs = 8;
constexpr int kSingleChannelPixels = 4;
constexpr int kSingleChannelThreads = 256;
// The row argmax's first stage: threadgroups over slices of the row.
constexpr int kArgmaxSlices = 128;
constexpr int kArgmaxThreads = 256;

// One decode token's attention over its caches: its queries, and its key as
// it is written into the key cache with its value, RMS-normalized and rotated
// first. The last two outputs share the caches' buffers.
class DecodeAttentionPrimitive : public mx::Primitive {
public:
  DecodeAttentionPrimitive(mx::Stream stream, int token_offset, float eps,
                           float log2_base, float scale)
      : Primitive(stream), token_offset_(token_offset), eps_(eps),
        log2_base_(log2_base), scale_(scale) {}

  void eval_cpu(const std::vector<mx::array> &,
                std::vector<mx::array> &) override {
    throw std::runtime_error("Decode attention runs on the GPU only");
  }

  void eval_gpu(const std::vector<mx::array> &inputs,
                std::vector<mx::array> &outputs) override {
    const mx::array &key_cache = inputs[5];
    const mx::array &value_cache = inputs[6];
    for (const mx::array *cache : {&key_cache, &value_cache}) {
      if (!cache->flags().row_contiguous ||
          cache->data_size() != cache->size()) {
        throw std::runtime_error("Cache writes need a materialized cache");
      } else {
      }
    }
    outputs[0].set_data(mx::allocator::malloc(outputs[0].nbytes()));
    // Note (XXXXRT666): slice_update writes in place only when no command
    // buffer still in flight holds the cache. A decode step built while the
    // one before it runs often finds one that does and copies the whole cache.
    // The rows a step writes never overlap the rows earlier steps read, and
    // MLX orders writes to one buffer across command buffers.
    outputs[1].copy_shared_buffer(key_cache);
    outputs[2].copy_shared_buffer(value_cache);
    mx::metal::Device &device = mx::metal::device(stream().device);
    mx::metal::CommandEncoder &encoder =
        mx::metal::get_command_encoder(stream());
    const int query_heads = static_cast<int>(inputs[0].size()) / kHeadDim;
    encoder.set_compute_pipeline_state(Kernel(device, "decode_attention"));
    for (int i = 0; i < 5; ++i) {
      encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(outputs[1], 5);
    encoder.set_output_array(outputs[2], 6);
    encoder.set_output_array(outputs[0], 7);
    encoder.set_bytes(key_cache.shape(2), 8);
    encoder.set_bytes(token_offset_, 9);
    encoder.set_bytes(eps_, 10);
    encoder.set_bytes(log2_base_, 11);
    encoder.set_bytes(scale_, 12);
    encoder.dispatch_threadgroups(MTL::Size(query_heads, 1, 1),
                                  MTL::Size(32 * kAttentionSimdgroups, 1, 1));
  }

  const char *name() const override { return "DecodeAttention"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &attention =
        static_cast<const DecodeAttentionPrimitive &>(other);
    return token_offset_ == attention.token_offset_ && eps_ == attention.eps_ &&
           log2_base_ == attention.log2_base_ && scale_ == attention.scale_;
  }

private:
  int token_offset_;
  float eps_;
  float log2_base_;
  float scale_;
};

// The argmax of one row of logits, in two stages over many threadgroups.
class RowArgmax : public mx::UnaryPrimitive {
public:
  explicit RowArgmax(mx::Stream stream) : UnaryPrimitive(stream) {}

  void eval_cpu(const std::vector<mx::array> &, mx::array &) override {
    throw std::runtime_error("Row argmax runs on the GPU only");
  }

  void eval_gpu(const std::vector<mx::array> &inputs, mx::array &out) override {
    const mx::array &logits = inputs[0];
    const int size = static_cast<int>(logits.size());
    const int slice = (size + kArgmaxSlices - 1) / kArgmaxSlices;
    out.set_data(mx::allocator::malloc(out.nbytes()));
    mx::metal::Device &device = mx::metal::device(stream().device);
    mx::metal::CommandEncoder &encoder =
        mx::metal::get_command_encoder(stream());
    mx::array values({kArgmaxSlices}, mx::float32, nullptr, {});
    mx::array indices({kArgmaxSlices}, mx::uint32, nullptr, {});
    for (mx::array *partial : {&values, &indices}) {
      partial->set_data(mx::allocator::malloc(partial->nbytes()));
      encoder.add_temporary(*partial);
    }
    encoder.set_compute_pipeline_state(Kernel(device, "argmax_slices"));
    encoder.set_input_array(logits, 0);
    encoder.set_output_array(values, 1);
    encoder.set_output_array(indices, 2);
    encoder.set_bytes(size, 3);
    encoder.set_bytes(slice, 4);
    encoder.dispatch_threadgroups(MTL::Size(kArgmaxSlices, 1, 1),
                                  MTL::Size(kArgmaxThreads, 1, 1));
    encoder.set_compute_pipeline_state(Kernel(device, "argmax_merge"));
    encoder.set_input_array(values, 0);
    encoder.set_input_array(indices, 1);
    encoder.set_output_array(out, 2);
    encoder.set_bytes(kArgmaxSlices, 3);
    encoder.dispatch_threadgroups(MTL::Size(1, 1, 1),
                                  MTL::Size(kArgmaxThreads, 1, 1));
  }

  const char *name() const override { return "RowArgmax"; }

  bool is_equivalent(const mx::Primitive &) const override { return true; }
};

// gelu(conv2d(x, weight, stride 2, padding 1) + bias) for 3x3 weights: one
// thread per pixel's 8 output channels for single-channel x, the neural
// accelerators otherwise.
class ConvolutionGeluPrimitive : public mx::UnaryPrimitive {
public:
  ConvolutionGeluPrimitive(mx::Stream stream, bool single_channel)
      : UnaryPrimitive(stream), single_channel_(single_channel) {}

  void eval_cpu(const std::vector<mx::array> &, mx::array &) override {
    throw std::runtime_error("Convolution primitives run on the GPU only");
  }

  void eval_gpu(const std::vector<mx::array> &inputs, mx::array &out) override {
    const mx::array &x = inputs[0];
    const int output_channels = out.shape(3);
    const int pixels = out.shape(0) * out.shape(1) * out.shape(2);
    out.set_data(mx::allocator::malloc(out.nbytes()));
    mx::metal::Device &device = mx::metal::device(stream().device);
    mx::metal::CommandEncoder &encoder =
        mx::metal::get_command_encoder(stream());
    const std::array<int, 4> input_shape = {x.shape(0), x.shape(1), x.shape(2),
                                            x.shape(3)};
    const std::array<int, 2> output_size = {out.shape(1), out.shape(2)};
    encoder.set_compute_pipeline_state(
        single_channel_ ? Kernel(device, "conv_gelu_single_channel")
                        : Kernel(device, "conv_gelu_nax", kNaxLibrary));
    for (int i = 0; i < 3; ++i) {
      encoder.set_input_array(inputs[i], i);
    }
    encoder.set_output_array(out, 3);
    encoder.set_bytes(input_shape.data(), 4, 4);
    encoder.set_bytes(output_size.data(), 2, 5);
    encoder.set_bytes(output_channels, 6);
    if (single_channel_) {
      const int pixel_stride =
          (pixels + kSingleChannelPixels - 1) / kSingleChannelPixels;
      encoder.dispatch_threads(
          MTL::Size(pixel_stride * (output_channels / kSingleChannelOutputs), 1,
                    1),
          MTL::Size(kSingleChannelThreads, 1, 1));
    } else {
      encoder.dispatch_threadgroups(
          MTL::Size(output_channels / kConvolutionTileChannels,
                    (pixels + kConvolutionTilePixels - 1) /
                        kConvolutionTilePixels,
                    1),
          MTL::Size(32 * kConvolutionSimdgroups, 1, 1));
    }
  }

  const char *name() const override { return "ConvolutionGelu"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    return single_channel_ ==
           static_cast<const ConvolutionGeluPrimitive &>(other).single_channel_;
  }

private:
  bool single_channel_;
};

} // namespace

bool DecodeAttentionFusable(const mx::array &queries, const mx::array &keys) {
  return selected_kernel_set.load() == KernelSet::kTuned && TunedDevice(1) &&
         queries.dtype() == mx::bfloat16 &&
         queries.size() == kAttentionGqa * keys.size();
}

std::vector<mx::array>
DecodeAttention(const mx::array &queries, const mx::array &keys,
                const mx::array &values, const mx::array &q_norm,
                const mx::array &k_norm, const mx::array &key_cache,
                const mx::array &value_cache, int token_offset, float eps,
                float rope_base, float scale) {
  const int query_heads = static_cast<int>(queries.size()) / kHeadDim;
  return mx::array::make_arrays(
      {{1, query_heads, 1, kHeadDim}, key_cache.shape(), value_cache.shape()},
      {queries.dtype(), key_cache.dtype(), value_cache.dtype()},
      std::make_shared<DecodeAttentionPrimitive>(
          mx::default_stream(mx::default_device()), token_offset, eps,
          std::log2(rope_base), scale),
      {mx::contiguous(queries), mx::contiguous(keys), mx::contiguous(values),
       q_norm, k_norm, key_cache, value_cache});
}

mx::array ConvolutionGelu(const mx::array &x, const mx::array &weight,
                          const mx::array &bias) {
  const int channels = x.shape(3);
  const int output_channels = weight.shape(0);
  const mx::Shape output_shape = {x.shape(0), (x.shape(1) - 1) / 2 + 1,
                                  (x.shape(2) - 1) / 2 + 1, output_channels};
  const size_t limit = std::numeric_limits<int>::max();
  const size_t output_size = static_cast<size_t>(output_shape[0]) *
                             output_shape[1] * output_shape[2] *
                             output_channels;
  const bool fusable =
      selected_kernel_set.load() == KernelSet::kTuned &&
      mx::default_device() == mx::Device::gpu && x.dtype() == mx::float16 &&
      weight.dtype() == x.dtype() && bias.dtype() == x.dtype() &&
      weight.shape(1) == 3 && weight.shape(2) == 3 && x.size() <= limit &&
      output_size <= limit;
  const bool single_channel =
      channels == 1 && output_channels % kSingleChannelOutputs == 0;
  if (fusable &&
      (single_channel || (NeuralAcceleratorsAvailable() &&
                          channels % kConvolutionTileDepth == 0 &&
                          output_channels % kConvolutionTileChannels == 0))) {
    return mx::array(
        output_shape, x.dtype(),
        std::make_shared<ConvolutionGeluPrimitive>(
            mx::default_stream(mx::default_device()), single_channel),
        {mx::contiguous(x), mx::contiguous(weight), mx::contiguous(bias)});
  } else {
    return omni_common::Gelu(
        mx::add(mx::conv2d(x, weight, {2, 2}, {1, 1}), bias));
  }
}

mx::array GreedyTokens(const mx::array &logits) {
  const int vocabulary = logits.shape(-1);
  if (TunedDevice(1) && selected_kernel_set.load() == KernelSet::kTuned &&
      logits.dtype() == mx::bfloat16 &&
      static_cast<int>(logits.size()) == vocabulary) {
    mx::Shape shape = logits.shape();
    shape.pop_back();
    return mx::array(
        shape, mx::uint32,
        std::make_shared<RowArgmax>(mx::default_stream(mx::default_device())),
        {mx::contiguous(logits)});
  } else {
    return mx::argmax(logits, -1);
  }
}

} // namespace qwen3_asr
