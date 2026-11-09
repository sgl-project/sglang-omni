// SPDX-License-Identifier: Apache-2.0
// Checks MultiRowQuantizedLinear against mx::quantized_matmul, and that it
// leaves the calls it does not support to MLX. Built on request:
//   ninja multi_row_matmul_test && ./multi_row_matmul_test
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "common/layers.h"
#include "common/multi_row_matmul.h"

namespace mx = mlx::core;
using omni_common::MultiRowQuantizedLinear;
using omni_common::QuantizationConfig;

namespace {

constexpr int kGroupSize = 64;
constexpr float kMaxRelativeErrorFloat16 = 2e-3f;
constexpr float kMaxRelativeErrorBfloat16 = 2e-2f;

void Require(bool condition, const std::string &message) {
  if (!condition) {
    throw std::runtime_error(message);
  } else {
  }
}

float MaxRelativeError(const mx::array &actual, const mx::array &expected) {
  const mx::array difference = mx::abs(mx::subtract(
      mx::astype(actual, mx::float32), mx::astype(expected, mx::float32)));
  const mx::array scale = mx::max(mx::abs(mx::astype(expected, mx::float32)));
  return (mx::max(difference) / scale).item<float>();
}

// Returns the packed weight, scales and biases of a random matrix.
std::vector<mx::array> RandomQuantizedWeight(int in_features, int out_features,
                                             int bits, mx::Dtype dtype,
                                             int group_size = kGroupSize) {
  const mx::array weight =
      mx::astype(mx::random::normal({out_features, in_features}), dtype);
  return mx::quantize(weight, group_size, bits);
}

mx::array RandomInput(mx::Shape shape, mx::Dtype dtype) {
  return mx::astype(mx::random::normal(std::move(shape)), dtype);
}

std::optional<mx::array> Run(const mx::array &x,
                             const std::vector<mx::array> &parts, int bits,
                             int group_size = kGroupSize,
                             const std::string &mode = "affine") {
  return MultiRowQuantizedLinear(x, parts[0], parts[1], parts[2],
                                 QuantizationConfig{group_size, bits, mode});
}

void CheckMatchesMlx(int bits, mx::Dtype dtype, int rows, int in_features,
                     int out_features) {
  const std::vector<mx::array> parts =
      RandomQuantizedWeight(in_features, out_features, bits, dtype);
  const mx::array x = RandomInput({rows, in_features}, dtype);
  const std::optional<mx::array> actual = Run(x, parts, bits);
  const std::string name = std::to_string(bits) + "-bit " +
                           (dtype == mx::bfloat16 ? "bf16" : "fp16") + " " +
                           std::to_string(rows) + " rows " +
                           std::to_string(in_features) + "x" +
                           std::to_string(out_features);
  Require(actual.has_value(), name + ": the kernel declined");
  const mx::array expected = mx::quantized_matmul(
      x, parts[0], parts[1], parts[2], true, kGroupSize, bits);
  mx::eval(*actual, expected);
  Require(actual->shape() == expected.shape(), name + ": wrong shape");
  Require(actual->dtype() == dtype, name + ": wrong dtype");
  const float limit = dtype == mx::bfloat16 ? kMaxRelativeErrorBfloat16
                                            : kMaxRelativeErrorFloat16;
  const float error = MaxRelativeError(*actual, expected);
  Require(error <= limit, name + ": relative error " + std::to_string(error));
  std::cout << "ok " << name << " error " << error << "\n";
}

void CheckKeepsLeadingDimensions() {
  const std::vector<mx::array> parts =
      RandomQuantizedWeight(2048, 1024, 8, mx::float16);
  const mx::array x = RandomInput({2, 4, 2048}, mx::float16);
  const std::optional<mx::array> actual = Run(x, parts, 8);
  Require(actual.has_value(), "3-d input: the kernel declined");
  Require(actual->shape() == mx::Shape({2, 4, 1024}), "3-d input: wrong shape");
  const mx::array expected = mx::quantized_matmul(
      x, parts[0], parts[1], parts[2], true, kGroupSize, 8);
  mx::eval(*actual, expected);
  Require(MaxRelativeError(*actual, expected) <= kMaxRelativeErrorFloat16,
          "3-d input: wrong values");
  std::cout << "ok leading dimensions\n";
}

void CheckDeclines(const std::string &name, bool declined) {
  Require(declined, name + ": the kernel should have declined");
  std::cout << "ok declines " << name << "\n";
}

} // namespace

int main() {
  try {
    // The switch is read once, on the first call.
    setenv("SGLANG_OMNI_MLX_MULTI_ROW_QMV", "1", 1);
    for (const int bits : {4, 6, 8}) {
      for (const mx::Dtype dtype : {mx::float16, mx::bfloat16}) {
        for (const int rows : {4, 5, 8}) {
          CheckMatchesMlx(bits, dtype, rows, 2048, 6144);
          CheckMatchesMlx(bits, dtype, rows, 6144, 2048);
          CheckMatchesMlx(bits, dtype, rows, 2048, 1024);
        }
      }
    }
    CheckKeepsLeadingDimensions();

    const std::vector<mx::array> parts =
        RandomQuantizedWeight(2048, 6144, 8, mx::float16);
    CheckDeclines(
        "3 rows",
        !Run(RandomInput({3, 2048}, mx::float16), parts, 8).has_value());
    CheckDeclines(
        "9 rows",
        !Run(RandomInput({9, 2048}, mx::float16), parts, 8).has_value());
    const mx::array x = RandomInput({8, 2048}, mx::float16);
    CheckDeclines("another mode",
                  !Run(x, parts, 8, kGroupSize, "mxfp4").has_value());
    CheckDeclines("a wrong bit width", !Run(x, parts, 4).has_value());
    CheckDeclines("no biases",
                  !MultiRowQuantizedLinear(x, parts[0], parts[1], std::nullopt,
                                           QuantizationConfig{kGroupSize, 8})
                       .has_value());
    CheckDeclines("float32 input",
                  !Run(RandomInput({8, 2048}, mx::float32),
                       RandomQuantizedWeight(2048, 6144, 8, mx::float32), 8)
                       .has_value());
    CheckDeclines(
        "group size 32",
        !Run(x, RandomQuantizedWeight(2048, 6144, 8, mx::float16, 32), 8, 32)
             .has_value());
    CheckDeclines("an input width that is not a multiple of 256",
                  !Run(RandomInput({8, 1984}, mx::float16),
                       RandomQuantizedWeight(1984, 6144, 8, mx::float16), 8)
                       .has_value());
    // 67584 x 2048 at 8 bits is just over the 128 MiB weight limit.
    CheckDeclines("weights over the size limit",
                  !Run(x, RandomQuantizedWeight(2048, 67584, 8, mx::float16), 8)
                       .has_value());
    std::cout << "all passed\n";
    return 0;
  } catch (const std::exception &error) {
    std::cerr << "FAILED: " << error.what() << "\n";
    return 1;
  }
}
