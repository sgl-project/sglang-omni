// SPDX-License-Identifier: Apache-2.0
#include "common/multi_row_matmul.h"

#include <cstdint>
#include <cstdlib>

namespace omni_common {

namespace mx = mlx::core;

namespace {

// Below four rows MLX is about as fast, so the kernel is not worth using.
constexpr int kMinRowCount = 4;
// The speculative verify block has eight rows; from ten rows on this kernel is
// several times slower than MLX.
constexpr int kMaxRowCount = 8;
// Scales and biases are indexed once per 64 inputs.
constexpr int kQuantizationGroupSize = 64;
constexpr int kLanesPerSimdgroup = 32;
constexpr int kValuesPerLane = 8;
// Inputs consumed per loop step by one simdgroup, so in_features must divide.
constexpr int kValuesPerSimdgroupStep = kLanesPerSimdgroup * kValuesPerLane;
// Two columns per simdgroup was fastest at eight rows (about 85 us against
// about 117 us for one column and 104 us for four on a 2048 to 6144 matrix).
constexpr int kOutputsPerSimdgroup = 2;
// One to eight simdgroups per threadgroup differed by about 1% on average.
constexpr int kSimdgroupsPerThreadgroup = 2;
// Past about 170 MB of weights this kernel falls behind MLX, which includes the
// vocabulary-sized output head; 128 MiB leaves a margin.
constexpr int64_t kMaxWeightByteCount = 128LL * 1024 * 1024;

// Each simdgroup computes kOutputsPerSimdgroup output columns for every row:
// a lane dequantizes eight consecutive weights once and applies them to all
// rows, so the weights are read once however many rows verify checks.
// x is read as ushort4 vectors; scalar half loads are much slower.
const char *const kMultiRowSource = R"METAL(
  const int lane = int(thread_index_in_simdgroup);
  const int first_output = (int(threadgroup_position_in_grid.x) * SIMDGROUPS +
                            int(simdgroup_index_in_threadgroup)) * OUTPUTS;
  const int group_count = IN_FEATURES / 64;
  const int weight_row_bytes = IN_FEATURES * BITS / 8;
  const device uchar *weight_bytes = (const device uchar *)weight;

  float partial_sums[OUTPUTS][ROWS];
  #pragma unroll
  for (int output = 0; output < OUTPUTS; ++output) {
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) partial_sums[output][row] = 0.0f;
  }

  for (int first_value = lane * 8; first_value < IN_FEATURES; first_value += 256) {
    float4 x_low[ROWS];
    float4 x_high[ROWS];
    float x_sum[ROWS];
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) {
      const device ushort4 *x_vectors =
          (const device ushort4 *)(x + row * IN_FEATURES + first_value);
      const ushort4 low_bits = x_vectors[0];
      const ushort4 high_bits = x_vectors[1];
      if (IS_BFLOAT16) {
        x_low[row] = float4(as_type<float>(uint(low_bits.x) << 16),
                            as_type<float>(uint(low_bits.y) << 16),
                            as_type<float>(uint(low_bits.z) << 16),
                            as_type<float>(uint(low_bits.w) << 16));
        x_high[row] = float4(as_type<float>(uint(high_bits.x) << 16),
                             as_type<float>(uint(high_bits.y) << 16),
                             as_type<float>(uint(high_bits.z) << 16),
                             as_type<float>(uint(high_bits.w) << 16));
      } else {
        x_low[row] = float4(float(as_type<half>(low_bits.x)),
                            float(as_type<half>(low_bits.y)),
                            float(as_type<half>(low_bits.z)),
                            float(as_type<half>(low_bits.w)));
        x_high[row] = float4(float(as_type<half>(high_bits.x)),
                             float(as_type<half>(high_bits.y)),
                             float(as_type<half>(high_bits.z)),
                             float(as_type<half>(high_bits.w)));
      }
      x_sum[row] = x_low[row].x + x_low[row].y + x_low[row].z + x_low[row].w +
                   x_high[row].x + x_high[row].y + x_high[row].z + x_high[row].w;
    }

    const int group = first_value / 64;
    #pragma unroll
    for (int output = 0; output < OUTPUTS; ++output) {
      const int column = first_output + output;
      const device uchar *packed =
          weight_bytes + long(column) * weight_row_bytes + first_value * BITS / 8;
      float4 quantized_low;
      float4 quantized_high;
      if (BITS == 8) {
        const uint2 words = *(const device uint2 *)packed;
        quantized_low = float4(float(words.x & 0xFF), float((words.x >> 8) & 0xFF),
                               float((words.x >> 16) & 0xFF), float(words.x >> 24));
        quantized_high = float4(float(words.y & 0xFF), float((words.y >> 8) & 0xFF),
                                float((words.y >> 16) & 0xFF), float(words.y >> 24));
      } else if (BITS == 4) {
        const uint word = *(const device uint *)packed;
        quantized_low = float4(float(word & 0xF), float((word >> 4) & 0xF),
                               float((word >> 8) & 0xF), float((word >> 12) & 0xF));
        quantized_high = float4(float((word >> 16) & 0xF), float((word >> 20) & 0xF),
                                float((word >> 24) & 0xF), float(word >> 28));
      } else {
        // Six-bit values are a little-endian bit stream: eight values, 48 bits.
        const device ushort *halves = (const device ushort *)packed;
        const ulong stream = ulong(halves[0]) | (ulong(halves[1]) << 16) |
                             (ulong(halves[2]) << 32);
        quantized_low = float4(float(stream & 63), float((stream >> 6) & 63),
                               float((stream >> 12) & 63), float((stream >> 18) & 63));
        quantized_high = float4(float((stream >> 24) & 63), float((stream >> 30) & 63),
                                float((stream >> 36) & 63), float((stream >> 42) & 63));
      }
      const float scale = float(scales[column * group_count + group]);
      const float bias = float(biases[column * group_count + group]);
      #pragma unroll
      for (int row = 0; row < ROWS; ++row) {
        const float weighted_sum = dot(x_low[row], quantized_low) +
                                   dot(x_high[row], quantized_high);
        partial_sums[output][row] += scale * weighted_sum + bias * x_sum[row];
      }
    }
  }

  #pragma unroll
  for (int output = 0; output < OUTPUTS; ++output) {
    #pragma unroll
    for (int row = 0; row < ROWS; ++row) {
      const float total = simd_sum(partial_sums[output][row]);
      if (lane == 0) out[row * OUT_FEATURES + first_output + output] = static_cast<T>(total);
    }
  }
)METAL";

} // namespace

std::optional<mx::array>
MultiRowQuantizedLinear(const mx::array &x, const mx::array &weight,
                        const mx::array &scales,
                        const std::optional<mx::array> &biases,
                        const QuantizationConfig &quantization) {
  static const bool is_enabled =
      std::getenv("SGLANG_OMNI_MLX_MULTI_ROW_QMV") != nullptr;
  const int in_features = x.shape(-1);
  const int out_features = weight.shape(0);
  const int row_count = static_cast<int>(x.size() / in_features);
  const int bits = quantization.bits;
  const bool is_bfloat16 = x.dtype() == mx::bfloat16;
  const int64_t weight_byte_count =
      static_cast<int64_t>(out_features) * in_features * bits / 8;

  const bool has_supported_quantization =
      biases.has_value() && quantization.mode == "affine" &&
      quantization.group_size == kQuantizationGroupSize &&
      (bits == 4 || bits == 6 || bits == 8) &&
      scales.dtype() == biases->dtype() &&
      weight.shape(1) == in_features * bits / 32;
  const bool has_supported_dtype = x.dtype() == mx::float16 || is_bfloat16;
  const bool has_supported_row_count =
      row_count >= kMinRowCount && row_count <= kMaxRowCount;
  const bool has_supported_shape =
      in_features % kValuesPerSimdgroupStep == 0 &&
      out_features % (kOutputsPerSimdgroup * kSimdgroupsPerThreadgroup) == 0;
  const bool is_faster_than_mlx = weight_byte_count <= kMaxWeightByteCount;

  if (!(is_enabled && has_supported_quantization && has_supported_dtype &&
        has_supported_row_count && has_supported_shape && is_faster_than_mlx)) {
    return std::nullopt;
  } else {
    static const auto kernel = mx::fast::metal_kernel(
        "multi_row_quantized_linear", {"x", "weight", "scales", "biases"},
        {"out"}, kMultiRowSource);
    mx::Shape output_shape = x.shape();
    output_shape.back() = out_features;
    const int simdgroup_count = out_features / kOutputsPerSimdgroup;
    const int thread_count = simdgroup_count * kLanesPerSimdgroup;
    const auto outputs = kernel(
        {mx::reshape(x, {row_count, in_features}), weight, scales, *biases},
        {{row_count, out_features}}, {x.dtype()}, {thread_count, 1, 1},
        {kSimdgroupsPerThreadgroup * kLanesPerSimdgroup, 1, 1},
        {{"T", x.dtype()},
         {"IS_BFLOAT16", is_bfloat16},
         {"ROWS", row_count},
         {"IN_FEATURES", in_features},
         {"OUT_FEATURES", out_features},
         {"BITS", bits},
         {"OUTPUTS", kOutputsPerSimdgroup},
         {"SIMDGROUPS", kSimdgroupsPerThreadgroup}},
        std::nullopt, false, {});
    return mx::reshape(outputs[0], output_shape);
  }
}

} // namespace omni_common
