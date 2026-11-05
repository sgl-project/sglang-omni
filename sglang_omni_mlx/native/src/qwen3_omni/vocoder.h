// SPDX-License-Identifier: Apache-2.0
#pragma once
#include "qwen3_omni/model.h"
namespace qwen3_omni {
class Vocoder {
public:
  Vocoder(Weights &weights, Json configuration);
  mx::array forward(const mx::array &codes,
                    const Cancellation &cancelled) const;
  int samples_per_frame() const;

private:
  mx::array convolution(const std::string &name, const mx::array &input,
                        int dilation = 1, int groups = 1) const;
  mx::array transpose_convolution(const std::string &name,
                                  const mx::array &input, int stride) const;
  mx::array snake(const std::string &name, const mx::array &input) const;
  Weights &weights;
  Json configuration;
};
} // namespace qwen3_omni
