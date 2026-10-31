// SPDX-License-Identifier: Apache-2.0
// The kernels' instantiations, built ahead of time into
// sglang_qwen3_asr.metallib beside the binaries.

// clang-format off
#include "mlx/backend/metal/kernels/utils.h"
#include "mlx/backend/metal/kernels/steel/gemm/gemm.h"
#include "mlx/backend/metal/kernels/steel/gemm/kernels/steel_gemm_splitk.h"
#include "mlx/backend/metal/kernels/unary_ops.h"
#include "mlx/backend/metal/kernels/binary_ops.h"
#include "mlx/backend/metal/kernels/quantized_utils.h"
#include "mlx/backend/metal/kernels/quantized.h"
#include "qwen3_asr/kernels/quantized_matmul.h"
#include "qwen3_asr/kernels/fused_ops.h"

instantiate_kernel("splitk_accum", gemm_splitk_accum, float, bfloat16_t)

// The shapes TunedQmvShape picks, each also RMS-normalizing its input.
#define instantiate_tuned_qmv(bits, rows, k_splits, row_groups)               \
  instantiate_kernel("tuned_qmv_b_" #bits "_r_" #rows "_k_" #k_splits        \
                     "_g_" #row_groups,                                       \
                     tuned_qmv, bfloat16_t, bits, rows, k_splits, row_groups, \
                     false, false)                                            \
  instantiate_kernel("tuned_qmv_b_" #bits "_r_" #rows "_k_" #k_splits        \
                     "_g_" #row_groups "_norm",                               \
                     tuned_qmv, bfloat16_t, bits, rows, k_splits, row_groups, \
                     false, true)

#define instantiate_tuned_qmv_gated(bits, rows, k_splits, row_groups)         \
  instantiate_kernel("tuned_qmv_gated_b_" #bits "_r_" #rows "_k_" #k_splits  \
                     "_g_" #row_groups,                                       \
                     tuned_qmv_gated, bfloat16_t, bits, rows, k_splits,       \
                     row_groups, false)                                       \
  instantiate_kernel("tuned_qmv_gated_b_" #bits "_r_" #rows "_k_" #k_splits  \
                     "_g_" #row_groups "_norm",                               \
                     tuned_qmv_gated, bfloat16_t, bits, rows, k_splits,       \
                     row_groups, true)

#define instantiate_tuned_qmv_bits(bits)                                      \
  instantiate_tuned_qmv(bits, 1, 4, 1)                                        \
  instantiate_tuned_qmv(bits, 4, 4, 2)                                        \
  instantiate_tuned_qmv(bits, 4, 1, 2)                                        \
  instantiate_tuned_qmv(bits, 2, 2, 2)                                        \
  instantiate_tuned_qmv(bits, 2, 1, 4)                                        \
  instantiate_tuned_qmv(bits, 2, 4, 1)                                        \
  instantiate_tuned_qmv(bits, 4, 2, 2)                                        \
  instantiate_tuned_qmv(bits, 4, 2, 4)                                        \
  instantiate_tuned_qmv_gated(bits, 1, 4, 1)                                  \
  instantiate_tuned_qmv_gated(bits, 1, 2, 1)                                  \
  instantiate_tuned_qmv_gated(bits, 2, 1, 2)

instantiate_tuned_qmv_bits(4)
instantiate_tuned_qmv_bits(6)

// The decode step's o and down products with their residual add, in the shapes
// TunedQmvShape picks for 1024 or 2048 outputs.
#define instantiate_residual_shape(bits, rows, k_splits, row_groups)          \
  instantiate_kernel("tuned_qmv_b_" #bits "_r_" #rows "_k_" #k_splits        \
                     "_g_" #row_groups "_res",                                \
                     tuned_qmv, bfloat16_t, bits, rows, k_splits, row_groups, \
                     true, false)

#define instantiate_residual_qmv(bits)                                        \
  instantiate_residual_shape(bits, 2, 1, 4)                                   \
  instantiate_residual_shape(bits, 1, 4, 1)                                   \
  instantiate_residual_shape(bits, 2, 4, 1)

instantiate_residual_qmv(4)
instantiate_residual_qmv(6)

// The tiles TunedQmmShape picks, for every 16-row prompt tile.
#define instantiate_tuned_qmm(bits, bm, bn, bk)                               \
  instantiate_kernel("tuned_qmm_b_" #bits "_" #bm "_" #bn "_" #bk "_s_0",     \
                     tuned_qmm, bfloat16_t, bits, bm, bn, bk, 2, 2, false)    \
  instantiate_kernel("tuned_qmm_b_" #bits "_" #bm "_" #bn "_" #bk "_s_1",     \
                     tuned_qmm, bfloat16_t, bits, bm, bn, bk, 2, 2, true)

#define instantiate_tuned_qmm_gated(bits, bm, bn, bk)                         \
  instantiate_kernel("tuned_qmm_gated_b_" #bits "_" #bm "_" #bn "_" #bk,      \
                     tuned_qmm_gated, bfloat16_t, bits, bm, bn, bk, 2, 2)

#define instantiate_tuned_qmm_rows(bits, bm)                                  \
  instantiate_tuned_qmm(bits, bm, 32, 32)                                     \
  instantiate_tuned_qmm(bits, bm, 64, 32)                                     \
  instantiate_tuned_qmm_gated(bits, bm, 32, 32)                               \
  instantiate_tuned_qmm_gated(bits, bm, 32, 64)

#define instantiate_tuned_qmm_bits(bits)                                      \
  instantiate_tuned_qmm_rows(bits, 16)                                        \
  instantiate_tuned_qmm_rows(bits, 32)                                        \
  instantiate_tuned_qmm_rows(bits, 48)                                        \
  instantiate_tuned_qmm_rows(bits, 64)                                        \
  instantiate_tuned_qmm_rows(bits, 80)                                        \
  instantiate_tuned_qmm_rows(bits, 96)                                        \
  instantiate_tuned_qmm_rows(bits, 112)                                       \
  instantiate_tuned_qmm_rows(bits, 128)                                       \
  instantiate_tuned_qmm_rows(bits, 144)                                       \
  instantiate_tuned_qmm_rows(bits, 160)

instantiate_tuned_qmm_bits(4)
instantiate_tuned_qmm_bits(6)

instantiate_kernel("conv_gelu_single_channel", conv_gelu_single_channel, half,
                   4)

instantiate_kernel("decode_attention", decode_attention, bfloat16_t, 128, 2,
                   32)
instantiate_kernel("argmax_slices", argmax_slices, bfloat16_t, 256)
instantiate_kernel("argmax_merge", argmax_merge, 256)
// clang-format on
