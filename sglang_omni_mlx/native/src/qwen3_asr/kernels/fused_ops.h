// SPDX-License-Identifier: Apache-2.0
// Kernels that each replace a chain of MLX ops: the decode step's attention
// with its norms, rotation and cache write, the greedy argmax over a row of
// logits, and the encoder's first convolution with its gelu.

// One head of a decode token's QKV product RMS-normalized with weight and
// rotated at position, rounded as MLX's rms_norm and rope: lane l of one
// simdgroup owns dims 2l and 2l + 1 and their rotation partners D/2 on.
template <typename T, int D>
METAL_FUNC void normalized_rotated(const device T *source,
                                   const device T *weight, float eps,
                                   float log2_base, int position,
                                   const thread int *dims, thread float *x) {
  constexpr int half_dim = D / 2;
  float squares = 0;
  STEEL_PRAGMA_UNROLL
  for (int j = 0; j < 4; ++j) {
    x[j] = static_cast<float>(source[dims[j]]);
    squares += x[j] * x[j];
  }
  const float inv_rms = metal::precise::rsqrt(simd_sum(squares) / D + eps);
  STEEL_PRAGMA_UNROLL
  for (int j = 0; j < 4; ++j) {
    x[j] = static_cast<float>(
        static_cast<T>(static_cast<float>(weight[dims[j]]) *
                       static_cast<float>(static_cast<T>(x[j] * inv_rms))));
  }
  STEEL_PRAGMA_UNROLL
  for (int pair = 0; pair < 2; ++pair) {
    const float d =
        static_cast<float>(dims[pair]) / static_cast<float>(half_dim);
    const float theta =
        static_cast<float>(position) * metal::exp2(-d * log2_base);
    const float cosine = metal::fast::cos(theta);
    const float sine = metal::fast::sin(theta);
    const float x1 = x[pair];
    const float x2 = x[pair + 2];
    x[pair] = static_cast<float>(static_cast<T>(x1 * cosine - x2 * sine));
    x[pair + 2] = static_cast<float>(static_cast<T>(x1 * sine + x2 * cosine));
  }
}

// One decode token's attention over the caches [kv_heads, cache_tokens, D],
// as MLX's sdpa_vector lays it out: a threadgroup per query head, its
// simdgroups taking every simdgroups-th key. Each threadgroup normalizes and
// rotates its query itself, and the simdgroup taking the new token at
// token_offset its key, which with the value the first query head of each KV
// head writes into the caches.
template <typename T, int D, int gqa, int simdgroups>
METAL_FUNC void decode_attention_head(
    const device T *queries, const device T *keys, const device T *values,
    const device T *q_norm, const device T *k_norm, device T *key_cache,
    device T *value_cache, device T *out, int cache_tokens, int token_offset,
    float eps, float log2_base, float scale, int head, uint lid, uint simd_gid,
    uint simd_lid, threadgroup float (*group_outputs)[D],
    threadgroup float *group_maxs, threadgroup float *group_sums) {
  constexpr int half_dim = D / 2;
  const int kv_head = head / gqa;
  const int dims[4] = {int(2 * simd_lid), int(2 * simd_lid + 1),
                       int(half_dim + 2 * simd_lid),
                       int(half_dim + 2 * simd_lid + 1)};
  const device T *head_keys = key_cache + int64_t(kv_head) * cache_tokens * D;
  const device T *head_values =
      value_cache + int64_t(kv_head) * cache_tokens * D;

  // The first key and value, read before the query is ready.
  const int first = min(int(simd_gid), token_offset);
  vec<T, 2> next[4] = {
      *(const device vec<T, 2> *)(head_keys + first * D + dims[0]),
      *(const device vec<T, 2> *)(head_keys + first * D + dims[2]),
      *(const device vec<T, 2> *)(head_values + first * D + dims[0]),
      *(const device vec<T, 2> *)(head_values + first * D + dims[2])};
  float q[4];
  normalized_rotated<T, D>(queries + head * D, q_norm, eps, log2_base,
                           token_offset, dims, q);
  STEEL_PRAGMA_UNROLL
  for (int j = 0; j < 4; ++j) {
    q[j] *= scale;
  }
  const bool takes_new = token_offset % simdgroups == int(simd_gid);
  float new_key[4] = {0, 0, 0, 0};
  float new_value[4] = {0, 0, 0, 0};
  if (takes_new) {
    normalized_rotated<T, D>(keys + kv_head * D, k_norm, eps, log2_base,
                             token_offset, dims, new_key);
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) {
      new_value[j] = static_cast<float>(values[kv_head * D + dims[j]]);
    }
    if (head % gqa == 0) {
      const int64_t row = int64_t(token_offset) * D;
      STEEL_PRAGMA_UNROLL
      for (int j = 0; j < 4; ++j) {
        device T *key_row =
            key_cache + int64_t(kv_head) * cache_tokens * D + row;
        device T *value_row =
            value_cache + int64_t(kv_head) * cache_tokens * D + row;
        key_row[dims[j]] = static_cast<T>(new_key[j]);
        value_row[dims[j]] = static_cast<T>(new_value[j]);
      }
    } else {
    }
  } else {
  }

  float max_score = Limits<float>::finite_min;
  float sum_exp = 0;
  float output[4] = {0, 0, 0, 0};
  for (int i = simd_gid; i <= token_offset; i += simdgroups) {
    const vec<T, 2> current[4] = {next[0], next[1], next[2], next[3]};
    const int following = min(i + simdgroups, token_offset);
    next[0] = *(const device vec<T, 2> *)(head_keys + following * D + dims[0]);
    next[1] = *(const device vec<T, 2> *)(head_keys + following * D + dims[2]);
    next[2] =
        *(const device vec<T, 2> *)(head_values + following * D + dims[0]);
    next[3] =
        *(const device vec<T, 2> *)(head_values + following * D + dims[2]);
    const bool fresh = i == token_offset;
    const float key[4] = {fresh ? new_key[0] : float(current[0][0]),
                          fresh ? new_key[1] : float(current[0][1]),
                          fresh ? new_key[2] : float(current[1][0]),
                          fresh ? new_key[3] : float(current[1][1])};
    const float value[4] = {fresh ? new_value[0] : float(current[2][0]),
                            fresh ? new_value[1] : float(current[2][1]),
                            fresh ? new_value[2] : float(current[3][0]),
                            fresh ? new_value[3] : float(current[3][1])};
    float score = 0;
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) {
      score += q[j] * key[j];
    }
    score = simd_sum(score);
    const float new_max = max(max_score, score);
    const float factor = fast::exp(max_score - new_max);
    const float exp_score = fast::exp(score - new_max);
    max_score = new_max;
    sum_exp = sum_exp * factor + exp_score;
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) {
      output[j] = output[j] * factor + exp_score * value[j];
    }
  }

  STEEL_PRAGMA_UNROLL
  for (int j = 0; j < 4; ++j) {
    group_outputs[simd_gid][dims[j]] = output[j];
  }
  if (simd_lid == 0) {
    group_maxs[simd_gid] = max_score;
    group_sums[simd_gid] = sum_exp;
  } else {
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (lid < D) {
    float head_max = Limits<float>::finite_min;
    STEEL_PRAGMA_UNROLL
    for (int s = 0; s < simdgroups; ++s) {
      head_max = max(head_max, group_maxs[s]);
    }
    float head_sum = 0;
    float head_output = 0;
    STEEL_PRAGMA_UNROLL
    for (int s = 0; s < simdgroups; ++s) {
      const float factor = fast::exp(group_maxs[s] - head_max);
      head_sum += group_sums[s] * factor;
      head_output += group_outputs[s][lid] * factor;
    }
    out[head * D + lid] = static_cast<T>(head_output / head_sum);
  } else {
  }
}

template <typename T, int D, int gqa, int simdgroups>
[[kernel]] void decode_attention(
    const device T *queries [[buffer(0)]], const device T *keys [[buffer(1)]],
    const device T *values [[buffer(2)]], const device T *q_norm [[buffer(3)]],
    const device T *k_norm [[buffer(4)]], device T *key_cache [[buffer(5)]],
    device T *value_cache [[buffer(6)]], device T *out [[buffer(7)]],
    const constant int &cache_tokens [[buffer(8)]],
    const constant int &token_offset [[buffer(9)]],
    const constant float &eps [[buffer(10)]],
    const constant float &log2_base [[buffer(11)]],
    const constant float &scale [[buffer(12)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint lid [[thread_index_in_threadgroup]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup float group_outputs[simdgroups][D];
  threadgroup float group_maxs[simdgroups];
  threadgroup float group_sums[simdgroups];
  decode_attention_head<T, D, gqa, simdgroups>(
      queries, keys, values, q_norm, k_norm, key_cache, value_cache, out,
      cache_tokens, token_offset, eps, log2_base, scale, tid.x, lid, simd_gid,
      simd_lid, group_outputs, group_maxs, group_sums);
}

// A logit and its index, ordered as MLX's argmax orders them: a NaN first,
// then the larger value, then the lower index.
struct ArgmaxCandidate {
  float value;
  uint32_t index;
};

METAL_FUNC ArgmaxCandidate argmax_better(ArgmaxCandidate best,
                                         ArgmaxCandidate other) {
  const bool other_wins =
      (isnan(other.value) &&
       (!isnan(best.value) || best.index > other.index)) ||
      (!isnan(best.value) &&
       (best.value < other.value ||
        (best.value == other.value && best.index > other.index)));
  return other_wins ? other : best;
}

// The best candidate of each thread's, reduced over the threadgroup into
// thread 0's through simd_best, one per simdgroup.
template <int threads>
METAL_FUNC ArgmaxCandidate
argmax_threadgroup(ArgmaxCandidate best, threadgroup ArgmaxCandidate *simd_best,
                   uint simd_gid, uint simd_lid) {
  for (ushort offset = 16; offset > 0; offset /= 2) {
    best = argmax_better(
        best, ArgmaxCandidate{simd_shuffle_down(best.value, offset),
                              simd_shuffle_down(best.index, offset)});
  }
  if (simd_lid == 0) {
    simd_best[simd_gid] = best;
  } else {
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_gid == 0) {
    best = simd_lid < threads / 32 ? simd_best[simd_lid]
                                   : ArgmaxCandidate{-INFINITY, 0xffffffffu};
    for (ushort offset = 16; offset > 0; offset /= 2) {
      best = argmax_better(
          best, ArgmaxCandidate{simd_shuffle_down(best.value, offset),
                                simd_shuffle_down(best.index, offset)});
    }
  } else {
  }
  return best;
}

// Greedy token of one row of logits in two stages: each threadgroup's best of
// a slice of the row, then the best of those.
template <typename T, int threads>
[[kernel]] void argmax_slices(const device T *logits [[buffer(0)]],
                              device float *slice_values [[buffer(1)]],
                              device uint32_t *slice_indices [[buffer(2)]],
                              const constant int &size [[buffer(3)]],
                              const constant int &slice [[buffer(4)]],
                              uint3 tid [[threadgroup_position_in_grid]],
                              uint lid [[thread_index_in_threadgroup]],
                              uint simd_gid [[simdgroup_index_in_threadgroup]],
                              uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup ArgmaxCandidate simd_best[threads / 32];
  const int start = tid.x * slice;
  const int end = min(start + slice, size);
  ArgmaxCandidate best{-INFINITY, 0xffffffffu};
  for (int i = start + lid; i < end; i += threads) {
    best = argmax_better(best, ArgmaxCandidate{float(logits[i]), uint32_t(i)});
  }
  best = argmax_threadgroup<threads>(best, simd_best, simd_gid, simd_lid);
  if (lid == 0) {
    slice_values[tid.x] = best.value;
    slice_indices[tid.x] = best.index;
  } else {
  }
}

template <int threads>
[[kernel]] void argmax_merge(const device float *slice_values [[buffer(0)]],
                             const device uint32_t *slice_indices [[buffer(1)]],
                             device uint32_t *out [[buffer(2)]],
                             const constant int &slices [[buffer(3)]],
                             uint lid [[thread_index_in_threadgroup]],
                             uint simd_gid [[simdgroup_index_in_threadgroup]],
                             uint simd_lid [[thread_index_in_simdgroup]]) {
  threadgroup ArgmaxCandidate simd_best[threads / 32];
  ArgmaxCandidate best{-INFINITY, 0xffffffffu};
  for (int i = lid; i < slices; i += threads) {
    best =
        argmax_better(best, ArgmaxCandidate{slice_values[i], slice_indices[i]});
  }
  best = argmax_threadgroup<threads>(best, simd_best, simd_gid, simd_lid);
  if (lid == 0) {
    out[0] = best.index;
  } else {
  }
}

// gelu(conv2d(x, w, stride 2, padding 1) + bias) for the encoder's first 3x3
// convolution, whose input has one channel: a thread sums the 9 taps of each
// of its pixels for 8 output channels in float and writes them once, where
// MLX's ops pass over the output three times. A thread's pixels lie
// pixel_stride apart, so a simdgroup's stores in each pass are consecutive.
template <typename T, int pixels_per_thread>
[[kernel]] void conv_gelu_single_channel(
    const device T *x [[buffer(0)]], const device T *w [[buffer(1)]],
    const device T *bias [[buffer(2)]], device T *y [[buffer(3)]],
    const constant int *input_shape [[buffer(4)]],
    const constant int *output_size [[buffer(5)]],
    const constant int &output_channels [[buffer(6)]],
    uint index [[thread_position_in_grid]]) {
  const int height = input_shape[1];
  const int width = input_shape[2];
  const int output_height = output_size[0];
  const int output_width = output_size[1];
  const int pixels = input_shape[0] * output_height * output_width;
  const int pixel_stride = (pixels + pixels_per_thread - 1) / pixels_per_thread;
  const int channel_groups = output_channels / 8;
  const int first_channel = index % channel_groups * 8;
  // The 8 channels' weights are 72 consecutive values from a 16-byte boundary.
  const device vec<T, 4> *weight_vectors =
      (const device vec<T, 4> *)(w + first_channel * 9);
  float weights[72];
  STEEL_PRAGMA_UNROLL
  for (int i = 0; i < 18; ++i) {
    const vec<T, 4> vector = weight_vectors[i];
    STEEL_PRAGMA_UNROLL
    for (int j = 0; j < 4; ++j) {
      weights[i * 4 + j] = float(vector[j]);
    }
  }
  float biases[8];
  STEEL_PRAGMA_UNROLL
  for (int channel = 0; channel < 8; ++channel) {
    biases[channel] = float(bias[first_channel + channel]);
  }
  for (int pass = 0; pass < pixels_per_thread; ++pass) {
    const int pixel = index / channel_groups + pass * pixel_stride;
    if (pixel < pixels) {
      const int column = pixel % output_width;
      const int row = pixel / output_width % output_height;
      const int image = pixel / (output_width * output_height);
      float taps[9];
      STEEL_PRAGMA_UNROLL
      for (int tap = 0; tap < 9; ++tap) {
        const int input_row = row * 2 - 1 + tap / 3;
        const int input_column = column * 2 - 1 + tap % 3;
        if (input_row >= 0 && input_row < height && input_column >= 0 &&
            input_column < width) {
          taps[tap] =
              float(x[(image * height + input_row) * width + input_column]);
        } else {
          taps[tap] = 0.0f;
        }
      }
      vec<T, 4> outputs[2];
      STEEL_PRAGMA_UNROLL
      for (int channel = 0; channel < 8; ++channel) {
        float value = biases[channel];
        STEEL_PRAGMA_UNROLL
        for (int tap = 0; tap < 9; ++tap) {
          value += taps[tap] * weights[channel * 9 + tap];
        }
        outputs[channel / 4][channel % 4] =
            T(value * (1.0f + erf(value * M_SQRT1_2_F)) * 0.5f);
      }
      device vec<T, 4> *destination =
          (device vec<T, 4> *)(y + pixel * output_channels + first_channel);
      destination[0] = outputs[0];
      destination[1] = outputs[1];
    } else {
    }
  }
}
