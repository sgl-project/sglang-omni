// SPDX-License-Identifier: Apache-2.0
// QmvBatchLimit and BlockDims follow MLX's quantized matmul dispatch,
// Copyright © 2023-2026 Apple Inc.
#include "qwen3_asr/kernels/kernels.h"

#include <IOKit/IOKitLib.h>

#include <algorithm>
#include <array>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "common/layers.h"
#include "mlx/primitives.h"
#include "qwen3_asr/kernels/kernel_library.h"

namespace qwen3_asr {

namespace mx = mlx::core;

namespace {

constexpr int kGroupSize = 64;
// Every K the tuned kernels take is a whole number of 512-value QMV steps.
constexpr int kInputAlignment = 512;
// Fused layers split at multiples of every tuned tile width.
constexpr int kSegmentAlignment = 64;
// The NAX tile's rows and columns and its simdgroups, and MLX's NAX split-K
// rule (qmm_nax_k_partitions): about 6 threadgroups per GPU core, at most 256,
// and at least 256 columns a partition.
constexpr int kNaxTile = 64;
constexpr int kNaxSimdgroups = 4;
constexpr int kNaxThreadgroupsPerCore = 6;
constexpr int kNaxMaxThreadgroups = 256;
constexpr int kNaxMinPartition = 256;
// The NAX tile reads x four values at a time; other offsets take the tuned
// tiles.
constexpr int kNaxInputAlignment = 8;
constexpr int kTileRowStep = 16;
constexpr int kTileMaxRows = 160;
// The most prompt rows a 30 s realtime segment, Voxt's default cut, gives a
// product: about 410 audio and template rows, plus the retained transcript.
// Longer prompts keep the tiles chosen for shorter ones.
constexpr int kSegmentPromptRows = 512;

int QmvBatchLimit(int K, int N, mx::metal::Device &device) {
  const char size = device.get_architecture().back();
  const int generation = device.get_architecture_gen();
  if (generation >= 17 && size != 'd') {
    return K <= 2048 && N <= 2048 ? 33 : (K <= 4096 && N <= 4096 ? 25 : 13);
  } else if (generation >= 15 && size != 'd') {
    return K <= 2048 && N <= 2048 ? 13 : (K <= 4096 && N <= 4096 ? 15 : 13);
  } else if (size == 'd') {
    return K <= 2048 && N <= 2048 ? 32 : (K <= 4096 && N <= 4096 ? 18 : 12);
  } else if (generation >= 13) {
    return K <= 2048 && N <= 2048
               ? 14
               : (K <= 4096 && N <= 4096
                      ? 10
                      : (generation == 13 && size == 's' ? 7 : 6));
  } else {
    return K <= 2048 && N <= 2048 ? 18 : (K <= 4096 && N <= 4096 ? 12 : 10);
  }
}

MTL::Size BlockDims(int dim0, int dim1) {
  int pows[2] = {0, 0};
  int sum = 0;
  while (true) {
    const int presum = sum;
    if (dim0 >= (1 << (pows[0] + 1))) {
      ++pows[0];
      ++sum;
    } else {
    }
    if (sum == 10) {
      break;
    } else {
    }
    if (dim1 >= (1 << (pows[1] + 1))) {
      ++pows[1];
      ++sum;
    } else {
    }
    if (sum == presum || sum == 10) {
      break;
    } else {
    }
  }
  return MTL::Size(1ul << pows[0], 1ul << pows[1], 1);
}

// The fewest row tiles of at most max_rows that cover M, each as small as
// fills them.
int TileRows(int M, int max_rows = kTileMaxRows) {
  const int tiles = (M + max_rows - 1) / max_rows;
  const int rows = (M + tiles - 1) / tiles;
  return (rows + kTileRowStep - 1) / kTileRowStep * kTileRowStep;
}

// The row tiles past 160 prompt rows in size class 's': 64 or 80 rows,
// whichever pads M less.
int PaddedRows(int M) {
  return (M + 79) / 80 * 80 < (M + 63) / 64 * 64 ? 80 : 64;
}

// The GPU's core count from its registry entry; 0 when it is not found.
int GpuCoreCount() {
  static const int count = [] {
    int cores = 0;
    const io_service_t service = IOServiceGetMatchingService(
        kIOMainPortDefault, IOServiceMatching("AGXAccelerator"));
    if (service != IO_OBJECT_NULL) {
      const CFTypeRef property = IORegistryEntryCreateCFProperty(
          service, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
      if (property != nullptr && CFGetTypeID(property) == CFNumberGetTypeID()) {
        CFNumberGetValue(static_cast<CFNumberRef>(property), kCFNumberIntType,
                         &cores);
      } else {
      }
      if (property != nullptr) {
        CFRelease(property);
      } else {
      }
      IOObjectRelease(service);
    } else {
    }
    return cores;
  }();
  return count;
}

// MLX's GPU size class, the last letter of its architecture name: 'p' phone,
// 'g' base, 's' Pro and Max, 'd' Ultra.
char SizeClass() {
  static const char size_class =
      mx::metal::device(mx::Device::gpu).get_architecture().back();
  return size_class;
}

// Inputs are x, then each layer's arrays; the first layer's sit in buffers
// 0-2 and x in 3, as MLX's kernels take them.
void SetFirstLayerInputs(mx::metal::CommandEncoder &encoder,
                         const std::vector<mx::array> &inputs,
                         int arrays_per_layer) {
  for (int i = 0; i < arrays_per_layer; ++i) {
    encoder.set_input_array(inputs[1 + i], i);
  }
  encoder.set_input_array(inputs[0], 3);
}

// The second and third layers' arrays from first_buffer on, absent layers
// repeating the last one, which no segment bound lets a kernel read.
void SetLaterLayerInputs(mx::metal::CommandEncoder &encoder,
                         const std::vector<mx::array> &inputs,
                         int arrays_per_layer, int first_buffer) {
  const int layers = static_cast<int>(inputs.size() - 1) / arrays_per_layer;
  for (int layer = 1; layer < 3; ++layer) {
    const int source = std::min(layer, layers - 1);
    for (int i = 0; i < arrays_per_layer; ++i) {
      encoder.set_input_array(inputs[1 + arrays_per_layer * source + i],
                              first_buffer + 3 * (layer - 1) + i);
    }
  }
}

// Output columns where the second and third layer start.
std::array<int, 2> SegmentBounds(const std::vector<mx::array> &inputs,
                                 int arrays_per_layer) {
  const int layers = static_cast<int>(inputs.size() - 1) / arrays_per_layer;
  std::array<int, 2> bounds{0, 0};
  int start = 0;
  for (int layer = 0; layer < 2; ++layer) {
    start += layer < layers ? inputs[1 + arrays_per_layer * layer].shape(0) : 0;
    bounds[layer] =
        layer + 1 < layers ? start : std::numeric_limits<int>::max();
  }
  return bounds;
}

struct QmvShape {
  int rows;
  int k_splits;
  int row_groups;
};

// Measured with Metal System Trace on Apple M4 as chains of dependent products
// over the 0.6B and 1.7B checkpoints' fused QKV, gated MLP, o, down and head
// layers. A K of 1024 has two 512-value steps, so it splits at most in two.
//
// Size class 's' takes the shapes good on both the M4 Pro and the M5 Max,
// timed as such chains against MLX's own kernels and then as decode steps of
// both checkpoints: a normed QKV product more rows per K split, a normed
// 4-bit gated one two rows over two groups, a down product four K splits and
// the head fewer rows.
QmvShape TunedQmvShape(int K, int N, bool gated, bool normed) {
  if (SizeClass() != 's') {
    return gated       ? QmvShape{1, K > 1024 ? 4 : 2, 1}
           : K > 2048  ? QmvShape{1, 4, 1}
           : N > 65536 ? QmvShape{4, 4, 2}
           : N > 2048  ? (K > 1024 ? QmvShape{4, 1, 2} : QmvShape{2, 2, 2})
                       : QmvShape{2, 1, 4};
  } else if (gated) {
    return normed && K <= 1024 ? QmvShape{2, 1, 2}
                               : QmvShape{1, K > 1024 ? 4 : 2, 1};
  } else if (N > 65536) {
    return K > 1024 ? QmvShape{2, 4, 1} : QmvShape{4, 2, 4};
  } else if (N > 2048) {
    return normed ? (K > 1024 ? QmvShape{2, 4, 1} : QmvShape{4, 2, 2})
                  : (K > 1024 ? QmvShape{4, 1, 2} : QmvShape{2, 2, 2});
  } else if (N <= 1024) {
    return K > 2048 ? QmvShape{2, 4, 1} : QmvShape{2, 1, 4};
  } else {
    return K > 2048 ? QmvShape{2, 4, 1} : QmvShape{1, 4, 1};
  }
}

void TunedQmv(const std::vector<mx::array> &inputs, mx::array &out, int bits,
              int M, int N, int K, bool gated, const mx::array *residual,
              const mx::array *norm, float eps, mx::metal::Device &device,
              mx::metal::CommandEncoder &encoder) {
  const QmvShape shape = TunedQmvShape(K, N, gated, norm != nullptr);
  const std::string suffix =
      residual != nullptr ? "_res" : (norm != nullptr ? "_norm" : "");
  encoder.set_compute_pipeline_state(Kernel(
      device, std::string(gated ? "tuned_qmv_gated_b_" : "tuned_qmv_b_") +
                  std::to_string(bits) + "_r_" + std::to_string(shape.rows) +
                  "_k_" + std::to_string(shape.k_splits) + "_g_" +
                  std::to_string(shape.row_groups) + suffix));
  SetFirstLayerInputs(encoder, inputs, 3);
  encoder.set_output_array(out, 4);
  encoder.set_bytes(K, 5);
  encoder.set_bytes(N, 6);
  SetLaterLayerInputs(encoder, inputs, 3, 7);
  const std::array<int, 2> bounds = SegmentBounds(inputs, 3);
  encoder.set_bytes(bounds.data(), 2, 13);
  encoder.set_input_array(residual != nullptr ? *residual : inputs[0], 14);
  encoder.set_input_array(norm != nullptr ? *norm : inputs[0], 15);
  encoder.set_bytes(eps, 16);
  encoder.dispatch_threadgroups(
      MTL::Size(M, N / (shape.rows * shape.row_groups), 1),
      MTL::Size(32 * shape.k_splits * shape.row_groups, 1, 1));
}

struct TileShape {
  int bm;
  int bn;
  int bk;
  int split_k;
};

// Measured with Metal System Trace on Apple M4 as chains of dependent products
// over the 1.7B checkpoint's fused QKV, gated MLP, o and down layers at 42-146
// prompt rows, and the 0.6B checkpoint's o and down layers at 42-250; past 256
// rows, timed as chains on both checkpoints at 200-400 rows. One tile spans a
// short prompt, except a gated product past 112 rows; a segment's long prompt
// takes 64-row tiles.
//
// Size class 's': the M4 Pro (16 cores) and the M5 Max (40 cores, MLX's neural
// accelerator kernels off) were timed as chains of both checkpoints' products
// at 42-410 rows, and at the bucket clips' 58-399, against MLX's own kernels,
// and take the tiles good on both: 16- to 48-row tiles over short prompts,
// one tile with K split in two (QKV) or eight (6-bit down) over medium ones,
// and past 160 rows 64- or 80-row tiles, whichever pads the prompt less;
// TunedTiles leaves the rest to MLX. Over those lengths they beat MLX by 6% on
// the M4 Pro (the base rule 4.5%) and 5% on the M5 Max (the base rule loses
// 4.5%).
TileShape TunedQmmShape(int M, int N, int K, int bits, bool gated) {
  const int bm = TileRows(M);
  const bool long_prompt = M > 256 && M <= kSegmentPromptRows;
  if (SizeClass() == 's') {
    const int rows = TileRows(M, 112);
    const int padded = PaddedRows(M);
    if (gated && bits == 4) {
      return M <= 48    ? TileShape{16, 32, 64, 1}
             : M <= 80  ? TileShape{32, 32, 64, 1}
             : M <= 256 ? TileShape{48, 32, 64, 1}
                        : TileShape{padded, 32, 64, 1};
    } else if (gated) {
      return M > 256 ? TileShape{padded, 32, 32, 1}
                     : TileShape{rows, 32, rows <= 64 ? 64 : 32, 1};
    } else if (N >= 4096) {
      return M <= (bits == 4 ? 48 : 160) ? TileShape{rows, 32, 32, 2}
             : M <= 160                  ? TileShape{32, 32, 32, 1}
                                         : TileShape{padded, 32, 32, 1};
    } else if (K > 4096) {
      return M <= 160 ? TileShape{rows, 64, 32, 8}
                      : TileShape{padded, 64, 32, 4};
    } else if (M > 160) {
      return {padded, 32, 32, bits == 4 && M <= 256 ? 4 : 2};
    } else {
      return bits == 4 ? TileShape{16, 32, 32, 4}
                       : TileShape{TileRows(M, 80), 32, 32, 4};
    }
  } else if (gated) {
    return long_prompt ? TileShape{64, 32, 32, 1}
           : bm > 112  ? TileShape{32, 32, 64, 1}
                       : TileShape{bm, 32, bm <= 64 ? 64 : 32, 1};
  } else if (long_prompt && N <= 2048) {
    return {64, 64, 32, 1};
  } else if (K > 4096) {
    return {bm, bm <= 112 ? 64 : 32, 32, 4};
  } else if (N <= 1024 && M <= 144) {
    return {bm, 32, 32, 4};
  } else if (N <= 1024 && M <= 160) {
    return {32, 64, 32, 1};
  } else if (N <= 2048) {
    return {bm, 32, 32, 2};
  } else {
    return {bm, 32, 32, 1};
  }
}

// A prompt product's tile kernel of a family (tuned_qmm, nax_qmm) in library
// over grid, whose depth splits K into partitions summed in float32.
void DispatchQmm(const std::vector<mx::array> &inputs, mx::array &out, int M,
                 int N, int K, bool gated, const std::string &family,
                 const std::string &tile, const char *library,
                 const MTL::Size &grid, const MTL::Size &group,
                 mx::metal::Device &device,
                 mx::metal::CommandEncoder &encoder) {
  const int split_k = static_cast<int>(grid.depth);
  const int k_partition_size = K / split_k;
  SetFirstLayerInputs(encoder, inputs, 3);
  encoder.set_bytes(K, 5);
  encoder.set_bytes(N, 6);
  encoder.set_bytes(M, 7);
  encoder.set_bytes(k_partition_size, 8);
  SetLaterLayerInputs(encoder, inputs, 3, 9);
  const std::array<int, 2> bounds = SegmentBounds(inputs, 3);
  encoder.set_bytes(bounds.data(), 2, 15);
  if (gated) {
    encoder.set_compute_pipeline_state(
        Kernel(device, family + "_gated_b_" + tile, library));
    encoder.set_output_array(out, 4);
    encoder.dispatch_threadgroups(grid, group);
  } else if (split_k == 1) {
    encoder.set_compute_pipeline_state(
        Kernel(device, family + "_b_" + tile + "_s_0", library));
    encoder.set_output_array(out, 4);
    encoder.dispatch_threadgroups(grid, group);
  } else {
    mx::array partials({split_k, M, N}, mx::float32, nullptr, {});
    partials.set_data(mx::allocator::malloc(partials.nbytes()));
    encoder.add_temporary(partials);
    encoder.set_compute_pipeline_state(
        Kernel(device, family + "_b_" + tile + "_s_1", library));
    encoder.set_output_array(partials, 4);
    encoder.dispatch_threadgroups(grid, group);
    const int partition_stride = M * N;
    encoder.set_compute_pipeline_state(Kernel(device, "splitk_accum"));
    encoder.set_input_array(partials, 0);
    encoder.set_output_array(out, 1);
    encoder.set_bytes(split_k, 2);
    encoder.set_bytes(partition_stride, 3);
    encoder.set_bytes(N, 4);
    encoder.dispatch_threads(MTL::Size(N, M, 1), BlockDims(N, M));
  }
}

void TunedQmm(const std::vector<mx::array> &inputs, mx::array &out, int bits,
              int M, int N, int K, bool gated, mx::metal::Device &device,
              mx::metal::CommandEncoder &encoder) {
  const TileShape shape = TunedQmmShape(M, N, K, bits, gated);
  const std::string tile =
      std::to_string(bits) + "_" + std::to_string(shape.bm) + "_" +
      std::to_string(shape.bn) + "_" + std::to_string(shape.bk);
  DispatchQmm(
      inputs, out, M, N, K, gated, "tuned_qmm", tile, kLibrary,
      MTL::Size(N / shape.bn, (M + shape.bm - 1) / shape.bm, shape.split_k),
      MTL::Size(128, 1, 1), device, encoder);
}

// K partitions of a NAX product, doubled up to 4 while the grid stays within
// the threadgroups the GPU's cores take, more for long K. Gated tiles stay
// whole, as silu needs the whole sum.
int NaxSplitK(int M, int N, int K, bool gated) {
  const int tiles = N / kNaxTile * ((M + kNaxTile - 1) / kNaxTile);
  const int cores = GpuCoreCount();
  const int max_threadgroups =
      cores > 0
          ? std::min(kNaxMaxThreadgroups,
                     kNaxThreadgroupsPerCore * cores * std::max(K, 2048) / 2048)
          : kNaxMaxThreadgroups;
  int split_k = 1;
  while (!gated && split_k < 4 && tiles * split_k * 2 <= max_threadgroups &&
         K % (split_k * 2 * kGroupSize) == 0 &&
         K / (split_k * 2) >= kNaxMinPartition) {
    split_k *= 2;
  }
  return split_k;
}

void NaxQmm(const std::vector<mx::array> &inputs, mx::array &out, int bits,
            int M, int N, int K, bool gated, mx::metal::Device &device,
            mx::metal::CommandEncoder &encoder) {
  const int columns = gated ? kNaxTile / 2 : kNaxTile;
  const MTL::Size grid(N / columns, (M + kNaxTile - 1) / kNaxTile,
                       NaxSplitK(M, N, K, gated));
  const MTL::Size group(kNaxSimdgroups * 32, 1, 1);
  DispatchQmm(inputs, out, M, N, K, gated, "nax_qmm", std::to_string(bits),
              kNaxLibrary, grid, group, device, encoder);
}

// The tuned kernels over x and up to three layers, outputs side by side, or
// silu(x @ gate^T) * (x @ up^T) when gated. Inputs are x, then each layer's
// weight, scales and biases, then the residual or the input norm's weight.
class LinearPrimitive : public mx::UnaryPrimitive {
public:
  LinearPrimitive(mx::Stream stream, int bits, bool gated, bool add_residual,
                  std::optional<float> norm_eps)
      : UnaryPrimitive(stream), bits_(bits), gated_(gated),
        add_residual_(add_residual), norm_eps_(norm_eps) {}

  void eval_cpu(const std::vector<mx::array> &, mx::array &) override {
    throw std::runtime_error("Linear primitives run on the GPU only");
  }

  void eval_gpu(const std::vector<mx::array> &inputs, mx::array &out) override {
    mx::metal::Device &device = mx::metal::device(stream().device);
    mx::metal::CommandEncoder &encoder =
        mx::metal::get_command_encoder(stream());
    out.set_data(mx::allocator::malloc(out.nbytes()));
    const int K = inputs[0].shape(-1);
    const int M = static_cast<int>(inputs[0].size()) / K;
    const int N = out.shape(-1);
    if (add_residual_) {
      const std::vector<mx::array> layer(inputs.begin(), inputs.end() - 1);
      TunedQmv(layer, out, bits_, M, N, K, gated_, &inputs.back(), nullptr, 0,
               device, encoder);
    } else if (norm_eps_.has_value()) {
      const std::vector<mx::array> layers(inputs.begin(), inputs.end() - 1);
      TunedQmv(layers, out, bits_, M, N, K, gated_, nullptr, &inputs.back(),
               *norm_eps_, device, encoder);
    } else if (M == 1) {
      TunedQmv(inputs, out, bits_, M, N, K, gated_, nullptr, nullptr, 0, device,
               encoder);
    } else if (NeuralAcceleratorsAvailable() &&
               inputs[0].offset() % kNaxInputAlignment == 0) {
      NaxQmm(inputs, out, bits_, M, N, K, gated_, device, encoder);
    } else {
      TunedQmm(inputs, out, bits_, M, N, K, gated_, device, encoder);
    }
  }

  const char *name() const override { return "Linear"; }

  bool is_equivalent(const mx::Primitive &other) const override {
    const auto &linear = static_cast<const LinearPrimitive &>(other);
    return bits_ == linear.bits_ && gated_ == linear.gated_ &&
           add_residual_ == linear.add_residual_ &&
           norm_eps_ == linear.norm_eps_;
  }

private:
  int bits_;
  bool gated_;
  bool add_residual_;
  std::optional<float> norm_eps_;
};

// Whether a prompt product's tuned tiles beat MLX's kernels: in size class 's'
// not past 400 rows but for 6-bit gated products, nor for 4-bit gated ones of
// 81-160 rows, 4-bit narrow ones of 49-160 or 6-bit narrow ones of 129-160.
// The NAX tiles take every prompt product.
bool TunedTiles(int M, int N, int K, int bits, bool gated) {
  if (M == 1 || SizeClass() != 's' || NeuralAcceleratorsAvailable()) {
    return true;
  } else if (M > 400) {
    return gated && bits != 4;
  } else if (gated) {
    return bits != 4 || M <= 80 || M > 160;
  } else if (N >= 4096 || K > 4096 || M > 160) {
    return true;
  } else {
    return M <= (bits == 4 ? 48 : 128);
  }
}

// Whether the tuned kernels take x and these layers, as one product: 4 or 6
// bits, up to three layers each a whole number of tiles wide, and one vector
// or as many rows as MLX gives a matrix product. MLX's own ops take the rest.
bool Tuned(const mx::array &x, const std::vector<QuantizedWeights> &layers,
           int group_size, int bits, bool gated) {
  const int K = x.shape(-1);
  const int M = static_cast<int>(x.size()) / K;
  int N = 0;
  for (const QuantizedWeights &layer : layers) {
    N += layer.weight.shape(0);
  }
  bool tuned =
      selected_kernel_set.load() == KernelSet::kTuned && TunedDevice(M) &&
      TunedTiles(M, N, K, bits, gated) && group_size == kGroupSize &&
      x.dtype() == mx::bfloat16 && K % kInputAlignment == 0 &&
      (bits == 4 || bits == 6) && layers.size() <= 3 &&
      (M == 1 || M >= QmvBatchLimit(K, gated ? N / 2 : N,
                                    mx::metal::device(mx::default_device())));
  for (const QuantizedWeights &layer : layers) {
    tuned = tuned && layer.weight.ndim() == 2 &&
            layer.weight.shape(0) % kSegmentAlignment == 0 &&
            layer.scales.dtype() == mx::bfloat16 &&
            layer.biases.dtype() == mx::bfloat16;
  }
  return tuned;
}

std::vector<mx::array> Outputs(const mx::array &fused,
                               const std::vector<mx::array> &weights) {
  std::vector<mx::array> outputs;
  mx::Shape start(fused.ndim(), 0);
  mx::Shape stop = fused.shape();
  for (const mx::array &weight : weights) {
    stop.back() = start.back() + weight.shape(0);
    outputs.push_back(weights.size() == 1 ? fused
                                          : mx::slice(fused, start, stop));
    start.back() = stop.back();
  }
  return outputs;
}

mx::array Apply(const mx::array &x, std::vector<mx::array> arrays, int bits,
                bool gated, int N,
                const std::optional<mx::array> &residual = std::nullopt,
                const std::optional<InputNorm> &norm = std::nullopt) {
  mx::Shape out_shape = x.shape();
  out_shape.back() = N;
  arrays.insert(arrays.begin(), mx::contiguous(x));
  if (residual.has_value()) {
    arrays.push_back(mx::contiguous(*residual));
  } else if (norm.has_value()) {
    arrays.push_back(mx::contiguous(norm->weight));
  } else {
  }
  return mx::array(
      out_shape, x.dtype(),
      std::make_shared<LinearPrimitive>(
          mx::default_stream(mx::default_device()), bits, gated,
          residual.has_value(),
          norm.has_value() ? std::optional<float>(norm->eps) : std::nullopt),
      std::move(arrays));
}

// Whether the tuned vector product normalizes x as it loads it, saving the
// decode step a normalization kernel.
bool NormFusable(const std::optional<InputNorm> &norm, bool tuned,
                 const mx::array &x) {
  return norm.has_value() && tuned &&
         static_cast<int>(x.size()) == x.shape(-1) &&
         norm->weight.dtype() == x.dtype() &&
         static_cast<int>(norm->weight.size()) == x.shape(-1);
}

mx::array Normalized(const mx::array &x, const std::optional<InputNorm> &norm) {
  return norm.has_value() ? mx::fast::rms_norm(x, norm->weight, norm->eps) : x;
}

mx::array MlxLinear(const mx::array &x, const QuantizedWeights &layer,
                    int group_size, int bits) {
  return mx::quantized_matmul(x, layer.weight, layer.scales, layer.biases, true,
                              group_size, bits, "affine");
}

} // namespace

KernelSet ParseKernelSet(const std::string &name) {
  if (name == "mlx") {
    return KernelSet::kMlx;
  } else if (name == "tuned") {
    return KernelSet::kTuned;
  } else {
    throw std::invalid_argument("--kernels must be mlx or tuned");
  }
}

void SetKernelSet(KernelSet kernel_set) {
  selected_kernel_set.store(kernel_set);
}

std::vector<mx::array> QuantizedProjections(
    const mx::array &x, const std::vector<QuantizedWeights> &layers,
    int group_size, int bits, const std::optional<InputNorm> &norm) {
  const bool tuned = Tuned(x, layers, group_size, bits, false);
  std::vector<mx::array> weights;
  std::vector<mx::array> arrays;
  int N = 0;
  for (const QuantizedWeights &layer : layers) {
    weights.push_back(layer.weight);
    arrays.insert(arrays.end(), {layer.weight, layer.scales, layer.biases});
    N += layer.weight.shape(0);
  }
  if (NormFusable(norm, tuned, x)) {
    return Outputs(Apply(x, arrays, bits, false, N, std::nullopt, norm),
                   weights);
  } else if (tuned) {
    return Outputs(Apply(Normalized(x, norm), arrays, bits, false, N), weights);
  } else {
    const mx::array input = Normalized(x, norm);
    std::vector<mx::array> outputs;
    for (const QuantizedWeights &layer : layers) {
      outputs.push_back(MlxLinear(input, layer, group_size, bits));
    }
    return outputs;
  }
}

mx::array QuantizedGatedLinear(const mx::array &x, const QuantizedWeights &gate,
                               const QuantizedWeights &up, int group_size,
                               int bits, const std::optional<InputNorm> &norm) {
  const bool tuned = Tuned(x, {gate, up}, group_size, bits, true);
  const std::vector<mx::array> arrays = {gate.weight, gate.scales, gate.biases,
                                         up.weight,   up.scales,   up.biases};
  if (NormFusable(norm, tuned, x)) {
    return Apply(x, arrays, bits, true, gate.weight.shape(0), std::nullopt,
                 norm);
  } else if (tuned) {
    return Apply(Normalized(x, norm), arrays, bits, true, gate.weight.shape(0));
  } else {
    const mx::array input = Normalized(x, norm);
    return mx::multiply(
        omni_common::Silu(MlxLinear(input, gate, group_size, bits)),
        MlxLinear(input, up, group_size, bits));
  }
}

mx::array QuantizedLinearAdd(const mx::array &x, const QuantizedWeights &layer,
                             const mx::array &residual, int group_size,
                             int bits) {
  if (static_cast<int>(x.size()) == x.shape(-1) &&
      Tuned(x, {layer}, group_size, bits, false) &&
      residual.dtype() == x.dtype()) {
    return Apply(x, {layer.weight, layer.scales, layer.biases}, bits, false,
                 layer.weight.shape(0), residual);
  } else {
    return mx::add(residual,
                   QuantizedProjections(x, {layer}, group_size, bits)[0]);
  }
}

} // namespace qwen3_asr
