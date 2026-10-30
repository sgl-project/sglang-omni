// SPDX-License-Identifier: Apache-2.0
// What the kernels share: their Metal libraries, the selected kernel set and
// the GPU's neural accelerators.
#pragma once

#include <dlfcn.h>

#include <atomic>
#include <filesystem>
#include <stdexcept>
#include <string>

#include "mlx/backend/metal/device.h"
#include "qwen3_asr/kernels/kernels.h"

namespace qwen3_asr {

inline constexpr int kFirstNeuralAcceleratorGeneration = 17;
inline constexpr int kFirstPhoneNeuralAcceleratorGeneration = 18;

// The kernels' libraries beside the binaries: the second, built for macOS 26.2
// on, holds those on the neural accelerators, and only systems and GPUs with
// them load it.
inline constexpr char kLibrary[] = "sglang_qwen3_asr";
inline constexpr char kNaxLibrary[] = "sglang_qwen3_asr_nax";

inline std::atomic<KernelSet> selected_kernel_set{KernelSet::kMlx};

inline const std::string &LibraryDirectory() {
  static const std::string directory = [] {
    Dl_info information;
    if (dladdr(reinterpret_cast<void *>(&LibraryDirectory), &information) ==
        0) {
      throw std::runtime_error("cannot locate the sglang_qwen3_asr metallibs");
    } else {
    }
    return std::filesystem::path(information.dli_fname).parent_path().string();
  }();
  return directory;
}

inline MTL::ComputePipelineState *Kernel(mlx::core::metal::Device &device,
                                         const std::string &name,
                                         const char *library = kLibrary) {
  return device.get_kernel(name,
                           device.get_library(library, LibraryDirectory()));
}

// MLX's is_nax_available, which its library does not export: macOS 26.2 on
// and a GPU of a generation with neural accelerators.
inline bool NeuralAcceleratorsAvailable() {
  static const bool available = [] {
    if (__builtin_available(macOS 26.2, iOS 26.2, *)) {
      mlx::core::metal::Device &device =
          mlx::core::metal::device(mlx::core::Device::gpu);
      return device.get_architecture_gen() >=
             (device.get_architecture().back() == 'p'
                  ? kFirstPhoneNeuralAcceleratorGeneration
                  : kFirstNeuralAcceleratorGeneration);
    } else {
      return false;
    }
  }();
  return available;
}

// Note (XXXXRT666): decode products are bound by memory bandwidth, so the
// tuned kernels run on every GPU. Prompt products are bound by compute: GPUs
// with neural accelerators run them on the NAX tiles, and phone GPUs without
// keep MLX's kernels, which the tuned tiles lose to.
inline bool TunedDevice(int rows) {
  static const bool tuned_prompts = [] {
    mlx::core::metal::Device &device =
        mlx::core::metal::device(mlx::core::Device::gpu);
    return device.get_architecture().back() != 'p' &&
           device.get_architecture_gen() < kFirstNeuralAcceleratorGeneration;
  }();
  return mlx::core::default_device() == mlx::core::Device::gpu &&
         (rows == 1 || tuned_prompts || NeuralAcceleratorsAvailable());
}

} // namespace qwen3_asr
