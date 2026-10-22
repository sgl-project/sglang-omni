// SPDX-License-Identifier: Apache-2.0
#include "voxtral_tts/audio_tokenizer.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

#include "mlx/fast.h"
#include "mlx/ops.h"

namespace voxtral_tts {

namespace mx = mlx::core;

namespace {

constexpr int kAttentionQueryChunkSize = 1024;
std::vector<float> AlibiSlopes(int head_count) {
  const auto power_of_two_slopes = [](int count) {
    const float base = std::pow(2.0f, -8.0f / static_cast<float>(count));
    std::vector<float> slopes(count);
    for (int head_index = 0; head_index < count; ++head_index) {
      slopes[head_index] = std::pow(base, static_cast<float>(head_index));
    }
    return slopes;
  };
  const int lower_power =
      1 << static_cast<int>(std::floor(std::log2(head_count)));
  if (lower_power == head_count) {
    return power_of_two_slopes(head_count);
  } else {
  }
  std::vector<float> slopes = power_of_two_slopes(lower_power);
  const std::vector<float> extended_slopes =
      power_of_two_slopes(2 * lower_power);
  for (int slope_index = 0; slope_index < head_count - lower_power;
       ++slope_index) {
    slopes.push_back(extended_slopes[2 * slope_index]);
  }
  return slopes;
}

} // namespace

AudioTokenizerDecoder::AudioTokenizerDecoder(
    const AudioTokenizerConfig &config,
    std::unique_ptr<qwen3_asr::Checkpoint> checkpoint)
    : config_(config), checkpoint_(std::move(checkpoint)),
      semantic_embedding_(checkpoint_->Weight("semantic_embedding")),
      downsample_factor_(config.patch_size) {
  int current_window_size = config.sliding_window_size;
  for (const int encoder_stride : config.encoder_convolution_strides) {
    if (config.half_attention_window_on_downsample && encoder_stride > 1) {
      current_window_size /= 2;
    } else {
    }
  }
  for (const int convolution_stride : config.convolution_strides) {
    downsample_factor_ *= convolution_stride;
  }
  if (config.half_attention_window_on_downsample &&
      config.convolution_strides.front() > 1) {
    current_window_size *= 2;
  } else {
  }
  int decoder_block_index = 1;
  for (size_t stack_index = 0; stack_index < config.transformer_lengths.size();
       ++stack_index) {
    const int layer_count = config.transformer_lengths[stack_index];
    stacks_.push_back({decoder_block_index, layer_count, current_window_size});
    ++decoder_block_index;
    const size_t next_stack_index = stack_index + 1;
    if (next_stack_index < config.transformer_lengths.size() &&
        (config.convolution_kernels[next_stack_index] != 1 ||
         config.convolution_strides[next_stack_index] != 1)) {
      transposed_convolution_block_indices_.push_back(decoder_block_index);
      ++decoder_block_index;
      if (config.half_attention_window_on_downsample &&
          config.convolution_strides[next_stack_index] > 1) {
        current_window_size *= 2;
      } else {
      }
    } else {
    }
  }
}

mx::array AudioTokenizerDecoder::CausalConv(
    const mx::array &hidden_states, const std::string &weight_prefix,
    int kernel_size, int stride, const std::string &padding_mode) const {
  const int input_length = hidden_states.shape(1);
  const int left_padding = kernel_size - stride;
  const int target_length = ((input_length + stride - 1) / stride) * stride;
  const int right_padding = target_length - input_length;
  const mx::array padded =
      mx::pad(hidden_states, {{0, 0}, {left_padding, right_padding}, {0, 0}},
              mx::array(0.0f), padding_mode);
  const mx::array output = mx::conv1d(
      padded, checkpoint_->Weight(weight_prefix + ".weight"), stride);
  if (checkpoint_->Has(weight_prefix + ".bias")) {
    return output + checkpoint_->Weight(weight_prefix + ".bias");
  } else {
    return output;
  }
}

mx::array
AudioTokenizerDecoder::CausalConvTranspose(const mx::array &hidden_states,
                                           const std::string &weight_prefix,
                                           int kernel_size, int stride) const {
  mx::array output = mx::conv_transpose1d(
      hidden_states, checkpoint_->Weight(weight_prefix + ".weight"), stride);
  if (checkpoint_->Has(weight_prefix + ".bias")) {
    output = output + checkpoint_->Weight(weight_prefix + ".bias");
  } else {
  }
  const int right_padding = kernel_size - stride;
  if (right_padding == 0) {
    return output;
  } else {
  }
  return mx::slice(
      output, {0, 0, 0},
      {output.shape(0), output.shape(1) - right_padding, output.shape(2)});
}

mx::array AudioTokenizerDecoder::RunTransformerStack(
    const mx::array &initial_hidden_states,
    const TransformerStack &stack) const {
  mx::array hidden_states = initial_hidden_states;
  const AudioTokenizerConfig &config = config_;
  const int head_count = config.head_count;
  const int kv_head_count = config.kv_head_count;
  const int head_dimension = config.head_dimension;
  const int kv_repeats = head_count / kv_head_count;
  const std::vector<float> slope_values = AlibiSlopes(head_count);
  const mx::array slopes(slope_values.data(), {head_count}, mx::float32);
  const int batch_size = hidden_states.shape(0);
  const int sequence_length = hidden_states.shape(1);

  for (int layer_offset = 0; layer_offset < stack.layer_count; ++layer_offset) {
    const std::string layer_prefix = "decoder_blocks." +
                                     std::to_string(stack.decoder_block_index) +
                                     ".layers." + std::to_string(layer_offset);
    mx::array normalized = checkpoint_->RmsNorm(
        hidden_states, layer_prefix + ".attention_norm", config.norm_epsilon);
    mx::array queries =
        checkpoint_->Linear(normalized, layer_prefix + ".attention.wq");
    mx::array keys =
        checkpoint_->Linear(normalized, layer_prefix + ".attention.wk");
    mx::array values =
        checkpoint_->Linear(normalized, layer_prefix + ".attention.wv");
    if (config.qk_norm) {
      queries = checkpoint_->RmsNorm(
          queries, layer_prefix + ".attention.q_norm", config.qk_norm_epsilon);
      keys = checkpoint_->RmsNorm(keys, layer_prefix + ".attention.k_norm",
                                  config.qk_norm_epsilon);
    } else {
    }
    queries = mx::transpose(mx::reshape(queries, {batch_size, sequence_length,
                                                  head_count, head_dimension}),
                            {0, 2, 1, 3});
    keys = mx::transpose(mx::reshape(keys, {batch_size, sequence_length,
                                            kv_head_count, head_dimension}),
                         {0, 2, 1, 3});
    values = mx::transpose(mx::reshape(values, {batch_size, sequence_length,
                                                kv_head_count, head_dimension}),
                           {0, 2, 1, 3});
    if (kv_repeats > 1) {
      keys = mx::repeat(keys, kv_repeats, 1);
      values = mx::repeat(values, kv_repeats, 1);
    } else {
    }
    const int right_window = config.causal ? 0 : stack.window_size;
    std::vector<mx::array> attention_outputs;
    for (int query_start = 0; query_start < sequence_length;
         query_start += kAttentionQueryChunkSize) {
      const int query_end =
          std::min(query_start + kAttentionQueryChunkSize, sequence_length);
      const int key_start = std::max(0, query_start - stack.window_size);
      const int key_end = std::min(sequence_length, query_end + right_window);
      const int query_length = query_end - query_start;
      const int key_length = key_end - key_start;
      const mx::array query_positions =
          mx::arange(query_start, query_end, mx::int32);
      const mx::array key_positions = mx::arange(key_start, key_end, mx::int32);
      const mx::array relative_positions = mx::expand_dims(key_positions, 0) -
                                           mx::expand_dims(query_positions, 1);
      mx::array attention_bias =
          mx::expand_dims(
              mx::expand_dims(mx::astype(slopes, hidden_states.dtype()), 1),
              2) *
          mx::expand_dims(mx::astype(relative_positions, hidden_states.dtype()),
                          0);
      const mx::array outside_window =
          mx::less(relative_positions,
                   mx::array(-stack.window_size, mx::int32)) |
          mx::greater(relative_positions, mx::array(right_window, mx::int32));
      attention_bias =
          mx::where(mx::expand_dims(outside_window, 0),
                    mx::array(-std::numeric_limits<float>::infinity(),
                              hidden_states.dtype()),
                    attention_bias);
      if (config.causal) {
        const mx::array future_position =
            mx::greater(relative_positions, mx::array(0, mx::int32));
        attention_bias =
            mx::where(mx::expand_dims(future_position, 0),
                      mx::array(-std::numeric_limits<float>::infinity(),
                                hidden_states.dtype()),
                      attention_bias);
      } else {
      }
      const mx::array query_chunk =
          mx::slice(queries, {0, 0, query_start, 0},
                    {batch_size, head_count, query_end, head_dimension});
      const mx::array key_chunk =
          mx::slice(keys, {0, 0, key_start, 0},
                    {batch_size, head_count, key_end, head_dimension});
      const mx::array value_chunk =
          mx::slice(values, {0, 0, key_start, 0},
                    {batch_size, head_count, key_end, head_dimension});
      attention_outputs.push_back(mx::fast::scaled_dot_product_attention(
          query_chunk, key_chunk, value_chunk,
          std::pow(static_cast<double>(head_dimension), -0.5), "",
          mx::expand_dims(attention_bias, 0)));
      if (query_length < 1 || key_length < 1) {
        throw std::logic_error("audio attention chunk has an empty range");
      } else {
      }
    }
    const mx::array attended = mx::reshape(
        mx::transpose(mx::concatenate(attention_outputs, 2), {0, 2, 1, 3}),
        {batch_size, sequence_length, head_count * head_dimension});
    mx::array attention_output =
        checkpoint_->Linear(attended, layer_prefix + ".attention.wo");
    if (config.layer_scale) {
      attention_output = attention_output *
                         checkpoint_->Weight(layer_prefix + ".attention_scale");
    } else {
    }
    hidden_states = hidden_states + attention_output;
    normalized = checkpoint_->RmsNorm(hidden_states, layer_prefix + ".ffn_norm",
                                      config.norm_epsilon);
    const mx::array gated =
        qwen3_asr::Silu(checkpoint_->Linear(
            normalized, layer_prefix + ".feed_forward.w1")) *
        checkpoint_->Linear(normalized, layer_prefix + ".feed_forward.w3");
    mx::array feed_forward_output =
        checkpoint_->Linear(gated, layer_prefix + ".feed_forward.w2");
    if (config.layer_scale) {
      feed_forward_output = feed_forward_output *
                            checkpoint_->Weight(layer_prefix + ".ffn_scale");
    } else {
    }
    hidden_states = hidden_states + feed_forward_output;
  }
  return hidden_states;
}

mx::array AudioTokenizerDecoder::Decode(const mx::array &audio_codes) const {
  if (audio_codes.ndim() != 3 || audio_codes.shape(1) < 2) {
    throw std::invalid_argument(
        "Voxtral audio codes must have shape [batch, codebooks, frames]");
  } else {
  }
  const mx::array semantic_codes = mx::take(audio_codes, 0, 1);
  const mx::Dtype decoder_dtype =
      checkpoint_->Weight("output_proj.conv.weight").dtype();
  const mx::array semantic_features =
      mx::astype(mx::transpose(mx::take(semantic_embedding_, semantic_codes, 0),
                               {0, 2, 1}),
                 decoder_dtype);
  const mx::array acoustic_codes =
      mx::astype(mx::slice(audio_codes, {0, 1, 0},
                           {audio_codes.shape(0), audio_codes.shape(1),
                            audio_codes.shape(2)}),
                 mx::float32);
  const mx::array acoustic_features = mx::astype(
      acoustic_codes *
              (2.0f / static_cast<float>(config_.acoustic_codebook_size - 1)) -
          1.0f,
      decoder_dtype);
  mx::array hidden_states = mx::transpose(
      mx::concatenate({semantic_features, acoustic_features}, 1), {0, 2, 1});
  hidden_states = CausalConv(hidden_states, "decoder_blocks.0.conv",
                             config_.convolution_kernels[0],
                             config_.convolution_strides[0], "edge");
  int transposed_convolution_index = 0;
  for (size_t stack_index = 0; stack_index < stacks_.size(); ++stack_index) {
    hidden_states = RunTransformerStack(hidden_states, stacks_[stack_index]);
    const size_t next_stack_index = stack_index + 1;
    if (next_stack_index < stacks_.size() &&
        (config_.convolution_kernels[next_stack_index] != 1 ||
         config_.convolution_strides[next_stack_index] != 1)) {
      const int decoder_block_index =
          transposed_convolution_block_indices_[transposed_convolution_index++];
      const std::string convolution_prefix =
          "decoder_blocks." + std::to_string(decoder_block_index) + ".conv";
      hidden_states =
          CausalConvTranspose(hidden_states, convolution_prefix,
                              config_.convolution_kernels[next_stack_index],
                              config_.convolution_strides[next_stack_index]);
    } else {
    }
  }
  hidden_states =
      CausalConv(hidden_states, "output_proj.conv",
                 config_.patch_projection_kernel_size, 1, "reflect");
  const int batch_size = hidden_states.shape(0);
  const int frame_count = hidden_states.shape(1);
  return mx::reshape(hidden_states,
                     {batch_size, 1, frame_count * config_.patch_size});
}

int AudioTokenizerDecoder::sampling_rate_hz() const {
  return config_.sampling_rate_hz;
}

int AudioTokenizerDecoder::downsample_factor() const {
  return downsample_factor_;
}

} // namespace voxtral_tts
