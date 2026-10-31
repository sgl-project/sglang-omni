// SPDX-License-Identifier: Apache-2.0
// Kernels on the neural accelerators, built on quantized_matmul.h. They need
// Metal 4, so they build into a library of their own for macOS 26.2 on.

// MLX's QuantizedBlockLoaderStaged: a thread fetches its words of a weight
// block into registers, then dequantizes them into threadgroup memory. Rows
// from up_row on are rows of the second weights.
template <typename T, short BROWS, short BCOLS, short dst_ld, short tgp_size,
          short group_size, short bits>
struct StagedBlockLoader {
  MLX_MTL_CONST short pack_factor = get_pack_factor<bits, 8>();
  MLX_MTL_CONST short bytes_per_pack = get_bytes_per_pack<bits>();
  MLX_MTL_CONST short BCOLS_PACKED = BCOLS / pack_factor;
  MLX_MTL_CONST short n_reads = (BCOLS_PACKED * BROWS) / tgp_size;
  MLX_MTL_CONST short n_words = n_reads * bytes_per_pack / 4;

  static_assert(BCOLS_PACKED * BROWS >= tgp_size &&
                    n_words * 4 == n_reads * bytes_per_pack,
                "Every thread loads whole words");
  static_assert(n_reads * pack_factor <= group_size &&
                    n_reads * pack_factor <= BCOLS,
                "A thread's values must lie in one row and quantization group");
  static_assert(bits == 4 || bits == 6, "4- or 6-bit codes");

  const short thread_idx;
  const short bi;
  const short bj;
  const device uint8_t *src;
  const device T *scales;
  const device T *biases;
  uint32_t raw[n_words];
  T scale;
  T bias;

  StagedBlockLoader(const device uint8_t *src_, const device T *scales_,
                    const device T *biases_, const device uint8_t *src_up,
                    const device T *scales_up, const device T *biases_up,
                    const short up_row, const int src_ld, ushort simd_group_id,
                    ushort simd_lane_id) thread
      : thread_idx(simd_group_id * 32 + simd_lane_id),
        bi((thread_idx * n_reads) / BCOLS_PACKED),
        bj((thread_idx * n_reads) % BCOLS_PACKED) {
    const bool up = bi >= up_row;
    const int row = up ? bi - up_row : bi;
    src = (up ? src_up : src_) + row * src_ld * bytes_per_pack / pack_factor +
          bj * bytes_per_pack;
    scales = (up ? scales_up : scales_) + row * src_ld / group_size;
    biases = (up ? biases_up : biases_) + row * src_ld / group_size;
  }

  // k: first column of the step.
  void fetch(const int k) thread {
    const device uint32_t *p =
        (const device uint32_t *)(src + k * bytes_per_pack / pack_factor);
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_words; i++) {
      raw[i] = p[i];
    }
    const int g = (k + bj * pack_factor) / group_size;
    scale = scales[g];
    bias = biases[g];
  }

  void store(threadgroup T *dst) const thread {
    constexpr short n_vals = n_reads * pack_factor;
    static_assert(n_vals % 8 == 0, "8-value stores");
    T v[n_vals];
    uint8_t b[n_words * 4];
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_words * 4; i++) {
      b[i] = uint8_t(raw[i / 4] >> (8 * (i % 4)));
    }
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_reads; i++) {
      dequantize_pack(b + i * bytes_per_pack, v + i * pack_factor);
    }
    dst += bi * dst_ld + bj * pack_factor;
    STEEL_PRAGMA_UNROLL
    for (short i = 0; i < n_vals; i += 8) {
      *(threadgroup vec<T, 8> *)(dst + i) = *(thread vec<T, 8> *)(v + i);
    }
  }

  // MLX's dequantize of one pack, whose codes sit in registers.
  void dequantize_pack(const thread uint8_t *w,
                       thread T *w_local) const thread {
    const float s = float(scale);
    const float b = float(bias);
    if (bits == 4) {
      w_local[0] = static_cast<T>(s * (w[0] & 0x0f) + b);
      w_local[1] = static_cast<T>(s / 16.0f * (w[0] & 0xf0) + b);
    } else {
      w_local[0] = static_cast<T>((w[0] & 0x3f) * s + b);
      w_local[1] =
          static_cast<T>((((w[0] >> 6) & 0x03) + ((w[1] & 0x0f) << 2)) * s + b);
      w_local[2] =
          static_cast<T>((((w[1] >> 4) & 0x0f) + ((w[2] & 0x03) << 4)) * s + b);
      w_local[3] = static_cast<T>(((w[2] >> 2) & 0x3f) * s + b);
    }
  }
};

// A simdgroup's 16 rows of a 64-column tile's sums into y, as T or U float
// split-K partial sums, or for a gated tile silu(gate) * up of its 32 gate and
// 32 up columns.
template <typename T, typename U, bool gated>
METAL_FUNC void nax_store_rows(thread mlx::steel::NAXTile<float, 1, 4> &d_tile,
                               device U *y, const int N, const int rows) {
  constexpr int SM = 16;
  constexpr int BN = 64;
  if (rows <= 0) {
    return;
  } else if (gated) {
    mlx::steel::NAXTile<float, 1, BN / 32> out_tile;
    STEEL_PRAGMA_UNROLL
    for (short j = 0; j < BN / 32; j++) {
      STEEL_PRAGMA_UNROLL
      for (short i = 0; i < mlx::steel::BaseNAXFrag::kElemsPerFrag; i++) {
        out_tile.frag_at(0, j)[i] =
            silu_multiply(static_cast<T>(d_tile.frag_at(0, j)[i]),
                          static_cast<T>(d_tile.frag_at(0, j + BN / 32)[i]));
      }
    }
    if (rows == SM) {
      out_tile.store(y, N);
    } else {
      out_tile.store_safe(y, N, short2(BN / 2, rows));
    }
  } else if (rows == SM) {
    d_tile.store(y, N);
  } else {
    d_tile.store_safe(y, N, short2(BN, rows));
  }
}

// MLX's qmm_t_nax_staged_impl for GPUs with neural accelerators: 4 simdgroups
// of 16 rows over a 64 x 64 tile, the weight blocks dequantized into two
// threadgroup buffers in turn and x prefetched into registers. U float writes
// split-K partial sums over k_size columns. A gated tile takes 32 gate rows
// and the same 32 up rows, and writes silu(gate) * up, 32 columns.
template <typename T, typename U, int bits, bool gated>
METAL_FUNC void nax_qmm_impl(const device uint8_t *w, const device T *scales,
                             const device T *biases, const device uint8_t *w_up,
                             const device T *scales_up,
                             const device T *biases_up, const device T *x,
                             device U *y, threadgroup T *Ws, const int K,
                             const int N, const int rows, const int k_size,
                             uint simd_gid, uint simd_lid) {
  constexpr int BM = 64;
  constexpr int BN = 64;
  constexpr int BK = 64;
  constexpr int SM = 16;
  constexpr int BK_padded = BK + 16 / sizeof(T);
  constexpr int tile_size = BN * BK_padded;
  using frag_t = typename mlx::steel::BaseNAXFrag::template dtype_frag_t<T>;

  StagedBlockLoader<T, BN, BK, BK_padded, (BM / SM) * SIMD_SIZE, 64, bits>
      loader(w, scales, biases, w_up, scales_up, biases_up, gated ? BN / 2 : BN,
             K, simd_gid, simd_lid);
  const int sm = SM * int(simd_gid);
  const int simd_rows = min(SM, rows - sm);
  // The two rows of x each thread holds in a fragment.
  const short2 sc = mlx::steel::BaseNAXFrag::get_coord();
  const device T *x0 =
      x + min(sm + int(sc.y), rows - 1) * static_cast<int64_t>(K) + sc.x;
  const device T *x1 =
      x + min(sm + int(sc.y) + 8, rows - 1) * static_cast<int64_t>(K) + sc.x;

  mlx::steel::NAXTile<float, 1, BN / 16> d_tile;
  d_tile.clear();
  vec<T, 4> a_cur[BK / 16][2];
  vec<T, 4> a_next[BK / 16][2];

  loader.fetch(0);
  loader.store(Ws);
  if (simd_rows > 0) {
    STEEL_PRAGMA_UNROLL
    for (short s = 0; s < BK / 16; s++) {
      a_cur[s][0] = *(const device vec<T, 4> *)(x0 + s * 16);
      a_cur[s][1] = *(const device vec<T, 4> *)(x1 + s * 16);
    }
  } else {
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  for (int k = 0, step = 0; k < k_size; k += BK, step++) {
    const bool more = k + BK < k_size;
    if (more) {
      loader.fetch(k + BK);
    } else {
    }
    if (simd_rows > 0) {
      const int next = more ? k + BK : k;
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        a_next[s][0] = *(const device vec<T, 4> *)(x0 + next + s * 16);
        a_next[s][1] = *(const device vec<T, 4> *)(x1 + next + s * 16);
      }
      const threadgroup T *ws =
          Ws + (step & 1) * tile_size + sc.y * BK_padded + sc.x;
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        frag_t a;
        STEEL_PRAGMA_UNROLL
        for (short c = 0; c < 4; c++) {
          a[c] = a_cur[s][0][c];
          a[4 + c] = a_cur[s][1][c];
        }
        STEEL_PRAGMA_UNROLL
        for (short j = 0; j < BN / 16; j += 2) {
          const threadgroup T *p0 = ws + (j * 16) * BK_padded + s * 16;
          const threadgroup T *p1 = p0 + 16 * BK_padded;
          frag_t b0;
          frag_t b1;
          STEEL_PRAGMA_UNROLL
          for (short h = 0; h < 2; h++) {
            const vec<T, 4> u0 =
                *(const threadgroup vec<T, 4> *)(p0 + h * 8 * BK_padded);
            const vec<T, 4> u1 =
                *(const threadgroup vec<T, 4> *)(p1 + h * 8 * BK_padded);
            STEEL_PRAGMA_UNROLL
            for (short c = 0; c < 4; c++) {
              b0[h * 4 + c] = u0[c];
              b1[h * 4 + c] = u1[c];
            }
          }
          mlx::steel::BaseNAXFrag::mma(d_tile.frag_at(0, j),
                                       d_tile.frag_at(0, j + 1), a,
                                       metal::bool_constant<false>{}, b0, b1,
                                       metal::bool_constant<true>{});
        }
      }
      STEEL_PRAGMA_UNROLL
      for (short s = 0; s < BK / 16; s++) {
        a_cur[s][0] = a_next[s][0];
        a_cur[s][1] = a_next[s][1];
      }
    } else {
    }
    if (more) {
      loader.store(Ws + ((step + 1) & 1) * tile_size);
    } else {
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  nax_store_rows<T, U, gated>(d_tile, y + sm * static_cast<int64_t>(N), N,
                              simd_rows);
}

// tuned_qmm on the neural accelerators, over 64 x 64 tiles.
template <typename T, int bits, bool split>
[[kernel]] void
nax_qmm(const device uint8_t *w0 [[buffer(0)]],
        const device T *scales0 [[buffer(1)]],
        const device T *biases0 [[buffer(2)]], const device T *x [[buffer(3)]],
        device void *y [[buffer(4)]], const constant int &K [[buffer(5)]],
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
  constexpr int BM = 64;
  constexpr int BN = 64;
  threadgroup T Ws[2 * BN * (64 + 16 / sizeof(T))];
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
    nax_qmm_impl<T, float, bits, false>(
        w, scales, biases, w, scales, biases, x,
        (device float *)y + tid.z * static_cast<int64_t>(M) * N + output, Ws, K,
        N, min(BM, M - row), k_partition_size, simd_gid, simd_lid);
  } else {
    nax_qmm_impl<T, T, bits, false>(w, scales, biases, w, scales, biases, x,
                                    (device T *)y + output, Ws, K, N,
                                    min(BM, M - row), K, simd_gid, simd_lid);
  }
}

// tuned_qmm_gated on the neural accelerators, 32 output columns a tile.
template <typename T, int bits>
[[kernel]] void nax_qmm_gated(const device uint8_t *w_gate [[buffer(0)]],
                              const device T *scales_gate [[buffer(1)]],
                              const device T *biases_gate [[buffer(2)]],
                              const device T *x [[buffer(3)]],
                              device T *y [[buffer(4)]],
                              const constant int &K [[buffer(5)]],
                              const constant int &N [[buffer(6)]],
                              const constant int &M [[buffer(7)]],
                              const device uint8_t *w_up [[buffer(9)]],
                              const device T *scales_up [[buffer(10)]],
                              const device T *biases_up [[buffer(11)]],
                              uint3 tid [[threadgroup_position_in_grid]],
                              uint simd_gid [[simdgroup_index_in_threadgroup]],
                              uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int BM = 64;
  constexpr int BN = 32;
  threadgroup T Ws[2 * 2 * BN * (64 + 16 / sizeof(T))];
  const int column = tid.x * BN;
  const int row = tid.y * BM;
  const int weight_offset = column * (K * bits / 8);
  const int group_offset = column * (K / 64);
  nax_qmm_impl<T, T, bits, true>(
      w_gate + weight_offset, scales_gate + group_offset,
      biases_gate + group_offset, w_up + weight_offset,
      scales_up + group_offset, biases_up + group_offset,
      x + row * static_cast<int64_t>(K),
      y + row * static_cast<int64_t>(N) + column, Ws, K, N, min(BM, M - row), K,
      simd_gid, simd_lid);
}

// gelu(conv2d(x, w, stride 2, padding 1) + bias) for the encoder's 3x3
// convolutions on the neural accelerators, as an implicit GEMM: rows are
// output pixels, columns output channels and K the 9 taps' input channels. A
// threadgroup gathers BM pixels' BK channels of a tap into threadgroup memory,
// zeros past the image edges, while matmul2d multiplies the chunk before by
// the OHWI weights read transposed.
template <typename T, int BM, int BN, int BK, int simdgroups>
[[kernel]] void
conv_gelu_nax(const device T *x [[buffer(0)]], const device T *w [[buffer(1)]],
              const device T *bias [[buffer(2)]], device T *y [[buffer(3)]],
              const constant int *input_shape [[buffer(4)]],
              const constant int *output_size [[buffer(5)]],
              const constant int &output_channels [[buffer(6)]],
              uint2 tid [[threadgroup_position_in_grid]],
              uint lid [[thread_index_in_threadgroup]]) {
  constexpr int threads = 32 * simdgroups;
  constexpr int row_vectors = BK / 8;
  constexpr int loads = (BM * row_vectors + threads - 1) / threads;
  threadgroup T tiles[2][BM * BK];
  const int height = input_shape[1];
  const int width = input_shape[2];
  const int channels = input_shape[3];
  const int output_height = output_size[0];
  const int output_width = output_size[1];
  const int pixels = input_shape[0] * output_height * output_width;
  const int first_pixel = tid.y * BM;
  const int first_channel = tid.x * BN;
  // Per gathered vector: its pixel's image offset, the input row and column
  // of its window's corner, and its slot in the tile (-1 past the tile).
  int image_offsets[loads];
  int window_rows[loads];
  int window_columns[loads];
  int slots[loads];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < loads; ++i) {
    const int vector = lid + i * threads;
    const int pixel = first_pixel + vector / row_vectors;
    const int column = pixel % output_width;
    const int row = (pixel / output_width) % output_height;
    const int image = pixel / (output_width * output_height);
    slots[i] = vector < BM * row_vectors ? vector * 8 : -1;
    window_rows[i] = pixel < pixels ? row * 2 - 1 : -height - 2;
    window_columns[i] = column * 2 - 1;
    image_offsets[i] =
        image * height * width * channels + vector % row_vectors * 8;
  }
  const int tap_chunks = channels / BK;
  const int chunks = 9 * tap_chunks;
  const auto gather = [&](int chunk, threadgroup T *tile) {
    const int tap = chunk / tap_chunks;
    const int first_input_channel = chunk % tap_chunks * BK;
    STEEL_PRAGMA_UNROLL
    for (int i = 0; i < loads; ++i) {
      const int input_row = window_rows[i] + tap / 3;
      const int input_column = window_columns[i] + tap % 3;
      vec<T, 4> low = vec<T, 4>(0);
      vec<T, 4> high = vec<T, 4>(0);
      if (slots[i] >= 0 && input_row >= 0 && input_row < height &&
          input_column >= 0 && input_column < width) {
        const device vec<T, 4> *source =
            (const device vec<T, 4> *)(x + image_offsets[i] +
                                       (input_row * width + input_column) *
                                           channels +
                                       first_input_channel);
        low = source[0];
        high = source[1];
      } else {
      }
      if (slots[i] >= 0) {
        *(threadgroup vec<T, 4> *)(tile + slots[i]) = low;
        *(threadgroup vec<T, 4> *)(tile + slots[i] + 4) = high;
      } else {
      }
    }
  };
  using Tile = metal::tensor<threadgroup T, metal::extents<int, BK, BM>,
                             metal::tensor_inline>;
  // Note (XXXXRT666): tensor ops take no const element types.
  const metal::tensor<device T, metal::dextents<int, 2>, metal::tensor_inline>
      weights((device T *)w,
              metal::dextents<int, 2>(9 * channels, output_channels));
  constexpr auto descriptor = mpp::tensor_ops::matmul2d_descriptor(
      BM, BN, BK, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<descriptor, metal::execution_simdgroups<simdgroups>>
      matmul;
  Tile first_tile(tiles[0], metal::extents<int, BK, BM>());
  auto first_weights = weights.template slice<BK, BN>(0, first_channel);
  auto sums = matmul.template get_destination_cooperative_tensor<
      decltype(first_tile), decltype(first_weights), float>();
  STEEL_PRAGMA_UNROLL
  for (uint16_t i = 0; i < sums.get_capacity(); ++i) {
    sums[i] = 0.0f;
  }
  gather(0, tiles[0]);
  for (int chunk = 0; chunk < chunks; ++chunk) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (chunk + 1 < chunks) {
      gather(chunk + 1, tiles[(chunk + 1) % 2]);
    } else {
    }
    Tile tile(tiles[chunk % 2], metal::extents<int, BK, BM>());
    auto chunk_weights =
        weights.template slice<BK, BN>(chunk * BK, first_channel);
    matmul.run(tile, chunk_weights, sums);
  }
  // The bias as a tile whose rows all hold it, loaded in the sums' layout.
  const metal::tensor<device T, metal::dextents<int, 2>, metal::tensor_inline>
      bias_rows((device T *)bias + first_channel,
                metal::dextents<int, 2>(BN, BM), metal::array<int, 2>{1, 0});
  auto biases = matmul.template get_destination_cooperative_tensor<
      decltype(first_tile), decltype(first_weights), T>();
  biases.load(bias_rows);
  auto outputs = matmul.template get_destination_cooperative_tensor<
      decltype(first_tile), decltype(first_weights), T>();
  STEEL_PRAGMA_UNROLL
  for (uint16_t i = 0; i < outputs.get_capacity(); ++i) {
    const float value = sums[i] + float(biases[i]);
    outputs[i] = T(value * (1.0f + erf(value * M_SQRT1_2_F)) * 0.5f);
  }
  metal::tensor<device T, metal::dextents<int, 2>, metal::tensor_inline>
      destination(y, metal::dextents<int, 2>(output_channels, pixels));
  auto tile_destination = destination.slice(first_channel, first_pixel);
  outputs.store(tile_destination);
}
