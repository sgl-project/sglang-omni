// SPDX-License-Identifier: Apache-2.0
// The neural accelerator kernels' instantiations, built ahead of time into
// sglang_qwen3_asr_nax.metallib for macOS 26.2 on.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.h"
#include "mlx/backend/metal/kernels/steel/gemm/nax.h"
#include "mlx/backend/metal/kernels/unary_ops.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
#include "qwen3_asr/kernels/quantized_matmul.h"
#include "qwen3_asr/kernels/nax.h"

#define instantiate_nax_qmm_bits(bits)                                        \
  instantiate_kernel("nax_qmm_b_" #bits "_s_0", nax_qmm, bfloat16_t, bits,     \
                     false)                                                   \
  instantiate_kernel("nax_qmm_b_" #bits "_s_1", nax_qmm, bfloat16_t, bits,     \
                     true)                                                    \
  instantiate_kernel("nax_qmm_gated_b_" #bits, nax_qmm_gated, bfloat16_t, bits)

instantiate_nax_qmm_bits(4)
instantiate_nax_qmm_bits(6)

// The tile conv_gelu_nax takes on every GPU with neural accelerators.
instantiate_kernel("conv_gelu_nax", conv_gelu_nax, half, 64, 96, 32, 4)
// clang-format on
