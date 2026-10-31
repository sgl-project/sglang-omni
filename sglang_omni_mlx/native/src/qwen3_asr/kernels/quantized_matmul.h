// SPDX-License-Identifier: Apache-2.0
// Quantized products specialized for Qwen3-ASR's single-request shapes: QMV
// for decode steps and QMM tiles for prompts, built on MLX's quantized block
// loader and steel GEMM blocks.

// silu(gate) * up with the unfused graph's operations and per-op rounding.
template <typename T> METAL_FUNC T silu_multiply(T gate, T up) {
  return Multiply()(Multiply()(gate, Sigmoid()(gate)), up);
}

// Quantized values each lane reads per row per step.
constant constexpr int kLaneValues = 16;
constant constexpr int kSimdValues = kLaneValues * SIMD_SIZE;

// x[i] scaled so a code masked in place (not shifted down) times it is the
// code times x[i]; a code split across two words adds its high part times
// x_high[i].
template <int bits> struct PackedDot {
  MLX_MTL_CONST int words = kLaneValues * bits / 32;

  static METAL_FUNC void prepare(const thread float *x, thread float *x_low,
                                 thread float *x_high) {
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kLaneValues; ++i) {
      const int shift = (i * bits) % 32;
      x_low[i] = x[i] * (1.0f / float(1u << shift));
      x_high[i] = shift + bits > 32 ? x[i] * float(1u << (32 - shift)) : 0.0f;
    }
  }

  static METAL_FUNC float dot(const thread uint32_t *w,
                              const thread float *x_low,
                              const thread float *x_high) {
    float accum = 0;
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kLaneValues; ++i) {
      const int bit = i * bits;
      const int word = bit / 32;
      const int shift = bit % 32;
      if (shift + bits <= 32) {
        const uint32_t mask = ((1u << bits) - 1u) << shift;
        accum += float(w[word] & mask) * x_low[i];
      } else {
        const int low_bits = 32 - shift;
        const uint32_t low_mask = ((1u << low_bits) - 1u) << shift;
        const uint32_t high_mask = (1u << (bits - low_bits)) - 1u;
        accum += float(w[word] & low_mask) * x_low[i];
        accum += float(w[word + 1] & high_mask) * x_high[i];
      }
    }
    return accum;
  }
};

// One simdgroup's share of y = x @ w^T for `rows` consecutive rows of w (and,
// when gated, the same rows of w_up): K is split over k_splits simdgroups,
// whose float partial sums meet in `partials`. A gated product writes
// silu(x @ w^T) * (x @ w_up^T); with add_residual the product writes
// residual + x @ w^T, rounded as MLX's add of the two. A normed product takes
// x RMS-normalized with weight `norm`: it multiplies x by the weight as it
// loads it and scales the sums by the inverse RMS of x, summed alongside.
template <typename T, int bits, int rows, int k_splits, bool gated,
          bool add_residual, bool normed>
METAL_FUNC void
tuned_qmv_impl(const device uint32_t *w, const device T *scales,
               const device T *biases, const device uint32_t *w_up,
               const device T *scales_up, const device T *biases_up,
               const device T *x, device T *y, const int K,
               threadgroup float *partials, uint k_split, uint simd_lid,
               const device T *residual, const device T *norm, float eps) {
  using Dot = PackedDot<bits>;
  constexpr int group_size = 64;
  constexpr int products = gated ? 2 * rows : rows;
  // A normed product's last sum is x's squares.
  constexpr int sums = normed ? products + 1 : products;
  const int row_words = K * bits / 32;
  const int row_groups_k = K / group_size;
  const int lane_words = simd_lid * Dot::words;
  const int lane_group = simd_lid * kLaneValues / group_size;
  x += simd_lid * kLaneValues;
  norm += simd_lid * kLaneValues;

  float result[sums] = {0};
  for (int k = k_split * kSimdValues; k < K; k += k_splits * kSimdValues) {
    float x_values[kLaneValues];
    float x_sum = 0;
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < kLaneValues; i += 4) {
      const vec<T, 4> loaded = *(const device vec<T, 4> *)(x + k + i);
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        x_values[i + j] = float(loaded[j]);
      }
      if (normed) {
        const vec<T, 4> weight = *(const device vec<T, 4> *)(norm + k + i);
        STEEL_PRAGMA_UNROLL
        for (int j = 0; j < 4; ++j) {
          result[products] += x_values[i + j] * x_values[i + j];
          x_values[i + j] *= float(weight[j]);
        }
      }
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        x_sum += x_values[i + j];
      }
    }
    float x_low[kLaneValues];
    float x_high[kLaneValues];
    Dot::prepare(x_values, x_low, x_high);
    const int word_offset = k * bits / 32 + lane_words;
    const int group_offset = k / group_size + lane_group;
    STEEL_PRAGMA_UNROLL
    for (int product = 0; product < products; ++product) {
      const int row = product % rows;
      const bool up = gated && product >= rows;
      const device uint32_t *row_w = (up ? w_up : w) + row * row_words;
      const device T *row_scales =
          (up ? scales_up : scales) + row * row_groups_k;
      const device T *row_biases =
          (up ? biases_up : biases) + row * row_groups_k;
      uint32_t packed[Dot::words];
      STEEL_PRAGMA_UNROLL
      for (int i = 0; i < Dot::words; ++i) {
        packed[i] = row_w[word_offset + i];
      }
      const float scale = row_scales[group_offset];
      const float bias = row_biases[group_offset];
      result[product] += scale * Dot::dot(packed, x_low, x_high) + bias * x_sum;
    }
  }

  STEEL_PRAGMA_UNROLL
  for (int sum = 0; sum < sums; ++sum) {
    result[sum] = simd_sum(result[sum]);
  }
  if (k_splits > 1) {
    if (simd_lid == 0) {
      STEEL_PRAGMA_UNROLL
      for (int sum = 0; sum < sums; ++sum) {
        partials[k_split * sums + sum] = result[sum];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (k_split == 0 && simd_lid == 0) {
      STEEL_PRAGMA_UNROLL
      for (int sum = 0; sum < sums; ++sum) {
        result[sum] = 0;
        STEEL_PRAGMA_UNROLL
        for (int split = 0; split < k_splits; ++split) {
          result[sum] += partials[split * sums + sum];
        }
      }
    }
  }
  if (normed) {
    const float inv_rms = metal::precise::rsqrt(result[products] / K + eps);
    STEEL_PRAGMA_UNROLL
    for (int product = 0; product < products; ++product) {
      result[product] *= inv_rms;
    }
  }
  if (k_split == 0 && simd_lid == 0) {
    STEEL_PRAGMA_UNROLL
    for (int row = 0; row < rows; ++row) {
      if (gated) {
        y[row] = silu_multiply(static_cast<T>(result[row]),
                               static_cast<T>(result[rows + row]));
      } else if (add_residual) {
        y[row] =
            static_cast<T>(static_cast<float>(static_cast<T>(result[row])) +
                           static_cast<float>(residual[row]));
      } else {
        y[row] = static_cast<T>(result[row]);
      }
    }
  }
}

// y = x @ [w0; w1; w2]^T for one or a few vectors: up to three projections of
// x, rows segment_rows[0] and segment_rows[1] starting the second and third.
// A threadgroup holds row_groups simdgroup rows of `rows` outputs each.
template <typename T, int bits, int rows, int k_splits, int row_groups,
          bool add_residual, bool normed>
[[kernel]] void tuned_qmv(const device uint32_t *w0 [[buffer(0)]],
                          const device T *scales0 [[buffer(1)]],
                          const device T *biases0 [[buffer(2)]],
                          const device T *x [[buffer(3)]],
                          device T *y [[buffer(4)]],
                          const constant int &K [[buffer(5)]],
                          const constant int &N [[buffer(6)]],
                          const device uint32_t *w1 [[buffer(7)]],
                          const device T *scales1 [[buffer(8)]],
                          const device T *biases1 [[buffer(9)]],
                          const device uint32_t *w2 [[buffer(10)]],
                          const device T *scales2 [[buffer(11)]],
                          const device T *biases2 [[buffer(12)]],
                          const constant int *segment_rows [[buffer(13)]],
                          const device T *residual [[buffer(14)]],
                          const device T *norm [[buffer(15)]],
                          const constant float &eps [[buffer(16)]],
                          uint3 tid [[threadgroup_position_in_grid]],
                          uint simd_gid [[simdgroup_index_in_threadgroup]],
                          uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float partials[row_groups][k_splits * (rows + normed)];
  const uint k_split = simd_gid % k_splits;
  const uint row_group = simd_gid / k_splits;
  const int row0 = (tid.y * row_groups + row_group) * rows;
  const int segment = (row0 >= segment_rows[0]) + (row0 >= segment_rows[1]);
  const int local_row = row0 - (segment == 0 ? 0 : segment_rows[segment - 1]);
  const device uint32_t *w = segment == 0 ? w0 : (segment == 1 ? w1 : w2);
  const device T *scales =
      segment == 0 ? scales0 : (segment == 1 ? scales1 : scales2);
  const device T *biases =
      segment == 0 ? biases0 : (segment == 1 ? biases1 : biases2);
  const int row_words = K * bits / 32;
  const int row_groups_k = K / 64;
  tuned_qmv_impl<T, bits, rows, k_splits, false, add_residual, normed>(
      w + local_row * row_words, scales + local_row * row_groups_k,
      biases + local_row * row_groups_k, w, scales, biases, x + tid.x * K,
      y + tid.x * N + row0, K, partials[row_group], k_split, simd_lid,
      residual + tid.x * N + row0, norm, eps);
}

// silu(x @ gate^T) * (x @ up^T) for one or a few vectors.
template <typename T, int bits, int rows, int k_splits, int row_groups,
          bool normed>
[[kernel]] void
tuned_qmv_gated(const device uint32_t *w_gate [[buffer(0)]],
                const device T *scales_gate [[buffer(1)]],
                const device T *biases_gate [[buffer(2)]],
                const device T *x [[buffer(3)]], device T *y [[buffer(4)]],
                const constant int &K [[buffer(5)]],
                const constant int &N [[buffer(6)]],
                const device uint32_t *w_up [[buffer(7)]],
                const device T *scales_up [[buffer(8)]],
                const device T *biases_up [[buffer(9)]],
                const device T *norm [[buffer(15)]],
                const constant float &eps [[buffer(16)]],
                uint3 tid [[threadgroup_position_in_grid]],
                uint simd_gid [[simdgroup_index_in_threadgroup]],
                uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float partials[row_groups][k_splits * (2 * rows + normed)];
  const uint k_split = simd_gid % k_splits;
  const uint row_group = simd_gid / k_splits;
  const int row0 = (tid.y * row_groups + row_group) * rows;
  const int row_words = K * bits / 32;
  const int row_groups_k = K / 64;
  tuned_qmv_impl<T, bits, rows, k_splits, true, false, normed>(
      w_gate + row0 * row_words, scales_gate + row0 * row_groups_k,
      biases_gate + row0 * row_groups_k, w_up + row0 * row_words,
      scales_up + row0 * row_groups_k, biases_up + row0 * row_groups_k,
      x + tid.x * K, y + tid.x * N + row0, K, partials[row_group], k_split,
      simd_lid, x, norm, eps);
}

// Widest per-thread read of an x tile whose thread rows divide its rows.
template <int rows, int columns, int threads> constexpr int tile_reads() {
  for (int reads = 8; reads > 1; reads /= 2) {
    if (columns % reads == 0 && columns / reads <= threads &&
        rows % (threads / (columns / reads)) == 0) {
      return reads;
    }
  }
  return 1;
}

// MLX's qmm_t_impl with the threadgroup's M tile, N tile and simdgroup grid as
// parameters, so one threadgroup can take a whole prompt. Pointers start at
// the tile; U is float for split-K partial sums. A gated product also
// multiplies x by w_up and writes silu(x @ w^T) * (x @ w_up^T).
template <typename T, typename U, int bits, int BM, int BN, int BK, int WM,
          int WN, bool gated>
METAL_FUNC void
tuned_qmm_impl(const device uint8_t *w, const device T *scales,
               const device T *biases, const device uint8_t *w_up,
               const device T *scales_up, const device T *biases_up,
               const device T *x, device U *y, threadgroup T *Xs,
               threadgroup T *Ws, threadgroup T *Ws_up, const int K,
               const int N, const int rows, const int k_size, uint simd_gid,
               uint simd_lid) {
  constexpr int group_size = 64;
  constexpr int threads = WM * WN * SIMD_SIZE;
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  constexpr int x_reads = tile_reads<BM, BK, threads>();

  using mma_t = mlx::steel::BlockMMA<T, U, BM, BN, BK, WM, WN, false, true,
                                     BK_padded, BK_padded>;
  using loader_x_t =
      mlx::steel::BlockLoader<T, BM, BK, BK_padded, 1, threads, 1, x_reads>;
  using loader_w_t =
      QuantizedBlockLoader<T, BN, BK, BK_padded, 1, threads, group_size, bits>;

  loader_x_t loader_x(x, K, Xs, simd_gid, simd_lid);
  loader_w_t loader_w(w, scales, biases, K, Ws, simd_gid, simd_lid);
  loader_w_t loader_up(w_up, scales_up, biases_up, K, Ws_up, simd_gid,
                       simd_lid);
  mma_t mma_op(simd_gid, simd_lid);
  mma_t mma_up(simd_gid, simd_lid);

  for (int k = 0; k < k_size; k += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (rows < BM) {
      loader_x.load_safe(short2(BK, rows));
    } else {
      loader_x.load_unsafe();
    }
    loader_w.load_unsafe();
    if (gated) {
      loader_up.load_unsafe();
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(Xs, Ws);
    if (gated) {
      mma_up.mma(Xs, Ws_up);
    }
    loader_x.next();
    loader_w.next();
    if (gated) {
      loader_up.next();
    }
  }

  if (gated) {
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < decltype(mma_op.Ctile)::kElemsPerTile; i++) {
      mma_op.Ctile.elems()[i] =
          silu_multiply(static_cast<T>(mma_op.Ctile.elems()[i]),
                        static_cast<T>(mma_up.Ctile.elems()[i]));
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (rows < BM) {
    mma_op.store_result_safe(y, N, short2(BN, rows));
  } else {
    mma_op.store_result(y, N);
  }
}

// x @ [w0; w1; w2]^T, columns segment_columns[0] and segment_columns[1]
// starting the second and third projection. Unsplit products round straight
// to T; split ones write float partials, tid.z selecting the K partition.
template <typename T, int bits, int BM, int BN, int BK, int WM, int WN,
          bool split>
[[kernel]] void tuned_qmm(const device uint8_t *w0 [[buffer(0)]],
                          const device T *scales0 [[buffer(1)]],
                          const device T *biases0 [[buffer(2)]],
                          const device T *x [[buffer(3)]],
                          device void *y [[buffer(4)]],
                          const constant int &K [[buffer(5)]],
                          const constant int &N [[buffer(6)]],
                          const constant int &M [[buffer(7)]],
                          const constant int &k_partition_size [[buffer(8)]],
                          const device uint8_t *w1 [[buffer(9)]],
                          const device T *scales1 [[buffer(10)]],
                          const device T *biases1 [[buffer(11)]],
                          const device uint8_t *w2 [[buffer(12)]],
                          const device T *scales2 [[buffer(13)]],
                          const device T *biases2 [[buffer(14)]],
                          const constant int *segment_columns [[buffer(15)]],
                          uint3 tid [[threadgroup_position_in_grid]],
                          uint simd_gid [[simdgroup_index_in_threadgroup]],
                          uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  const int column = tid.x * BN;
  const int row = tid.y * BM;
  const int segment =
      (column >= segment_columns[0]) + (column >= segment_columns[1]);
  const int local_column =
      column - (segment == 0 ? 0 : segment_columns[segment - 1]);
  const device uint8_t *w = segment == 0 ? w0 : (segment == 1 ? w1 : w2);
  const device T *scales =
      segment == 0 ? scales0 : (segment == 1 ? scales1 : scales2);
  const device T *biases =
      segment == 0 ? biases0 : (segment == 1 ? biases1 : biases2);
  const int k_start = split ? tid.z * k_partition_size : 0;
  w += local_column * (K * bits / 8) + k_start * bits / 8;
  scales += local_column * (K / 64) + k_start / 64;
  biases += local_column * (K / 64) + k_start / 64;
  x += row * static_cast<int64_t>(K) + k_start;
  const int64_t output = row * static_cast<int64_t>(N) + column;
  if (split) {
    tuned_qmm_impl<T, float, bits, BM, BN, BK, WM, WN, false>(
        w, scales, biases, w, scales, biases, x,
        (device float *)y + tid.z * static_cast<int64_t>(M) * N + output, Xs,
        Ws, Ws, K, N, min(BM, M - row), k_partition_size, simd_gid, simd_lid);
  } else {
    tuned_qmm_impl<T, T, bits, BM, BN, BK, WM, WN, false>(
        w, scales, biases, w, scales, biases, x, (device T *)y + output, Xs, Ws,
        Ws, K, N, min(BM, M - row), K, simd_gid, simd_lid);
  }
}

// silu(x @ gate^T) * (x @ up^T).
template <typename T, int bits, int BM, int BN, int BK, int WM, int WN>
[[kernel]] void tuned_qmm_gated(
    const device uint8_t *w_gate [[buffer(0)]],
    const device T *scales_gate [[buffer(1)]],
    const device T *biases_gate [[buffer(2)]], const device T *x [[buffer(3)]],
    device T *y [[buffer(4)]], const constant int &K [[buffer(5)]],
    const constant int &N [[buffer(6)]], const constant int &M [[buffer(7)]],
    const device uint8_t *w_up [[buffer(9)]],
    const device T *scales_up [[buffer(10)]],
    const device T *biases_up [[buffer(11)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BK_padded = (BK + 16 / sizeof(T));
  threadgroup T Xs[BM * BK_padded];
  threadgroup T Ws[BN * BK_padded];
  threadgroup T Ws_up[BN * BK_padded];
  const int column = tid.x * BN;
  const int row = tid.y * BM;
  const int row_bytes = K * bits / 8;
  const int weight_offset = column * row_bytes;
  const int group_offset = column * (K / 64);
  tuned_qmm_impl<T, T, bits, BM, BN, BK, WM, WN, true>(
      w_gate + weight_offset, scales_gate + group_offset,
      biases_gate + group_offset, w_up + weight_offset,
      scales_up + group_offset, biases_up + group_offset,
      x + row * static_cast<int64_t>(K),
      y + row * static_cast<int64_t>(N) + column, Xs, Ws, Ws_up, K, N,
      min(BM, M - row), K, simd_gid, simd_lid);
}
